"""The hashed n-gram embedding: an 84.6M-row table, sharded across ranks, with its own sparse Adam.

What: every token position t reads two rows of one [NGRAM_VOCAB_SIZE, 768] table -- a bigram row
(hash of x[t-1], x[t]) from the first half and a trigram row (hash of x[t-2], x[t-1], x[t]) from the
second half. Each row is multiplied by a +-1 sign row drawn from a small shared pool by an independent
hash of the same tokens (the sign trick of #299: rows shared by several n-grams separate by sign), and
the two signed rows are summed into x0_bigram, which the model injects into the residual stream.

Why sharded: the table is 65e9 elements (130 GB in bf16), so no rank can hold a replica. Rank r owns
rows [r * V/world, (r + 1) * V/world) -- 16.2 GB -- and only a small, data-dependent set of rows is
live on a step, so every cost below scales with tokens, never with the table.

Update cycles: the table only changes at its Adam events (is_update_step: every odd step, then every
4th step from NGRAM_ADAM_PERIOD4_START). A *cycle* is the run of steps up to and including one event.
Every row a cycle reads is fetched ONCE, at the previous event, into the model's `ngram_cache` -- that
is exact, since the only writer of the table is an event and the fetch is ordered against it. The fetch
is a row pull (sharded_rows.py): request (the cycle's sorted-unique row ids, "want"; cache slot i holds
row want[i]), exchange_ids, then land -- every rank serves the rows asked of it from its shard into the
asker's cache, this rank's own block included (_TableOwners below: it travels as a local copy inside the
exchange, which keeps both of an event's buffers contiguous). A training serve goes on the wire as int8
rows with one fp32 scale each and is dequantised into the cache (start_fetch below); the serves that are
not a training read -- the cold start and the validation pulls -- stay bf16, so the final validation
evaluates exactly the shard.

The forward reads the cache at `slots` (searchsorted of each token's row ids in want) and adds a
zeros "sink" leaf, so each step's row gradient lands on the sink ([2T, 768]), never on a table-sized
tensor. accumulate_grad adds each step's sink gradient into the cycle's per-slot sum (in fp16) as soon
as its backward produced it; at the event that sum travels back along the pull's routes in reverse
(send_grads), and each owner sums what every rank sent per row -- in place, in that same fp16
(merge_entries_ below) -- and runs Adam on exactly those rows (adam_update, stashed there and run by
the update stream below).

The optimizer is Adam with beta1 = 0 and ONE fp32 second moment per row (the row's mean squared
gradient), so its state is 4 bytes per row instead of 2 * 768 * 4, and a row no gradient touched
does not move: with no momentum its update is exactly zero, and the cautious weight decay is gated on
the update. Its second moment still decays by beta2 every event; that decay is applied lazily, when a
row is next touched (perf/kernels/ngram_adam.py replays the missed events' beta2 from a history), so
an event costs only the rows it touches -- including the owner-side merge, which needs no sort.

When each stage of a pull runs relative to the forward, backward and optimizer (and the host prep
that builds the want list steps ahead) is perf/row_prefetch.py's job; this file is correct called
in order, synchronously.

The event's two streams: an event's work splits into the part the next forward needs -- serving the
next cycle's rows out of the shard, into the askers' caches -- and the part nothing needs until the
NEXT event: the gradient all_to_all, the owner merge, the row Adam and the second-moment replay. Each
part runs on a CUDA stream of its own, the table's exchanges run on a process group of their own, and
the main stream waits for two events before each forward (wait_land): the rows of the cycle it is
about to read, and the per-slot sum of the gradient its backward is about to overwrite. The serve
reads the shard as event k-1 left it, so a cycle trains on rows one event stale -- uniformly for the
rows this rank owns and for its peers' -- while every update is still applied exactly once, in order,
by its owner.

WHERE the second part executes, not just on which stream: a stream decouples work from the main one
but does not delay it, so work queued during the event step still runs there, against the optimizer's
bank reduce-scatters and all-gathers. The event therefore only STASHES that part (adam_update,
start_cycle's zero of the next cycle's per-slot gradient, bring_current's replay list); the next step
calls start_update() between its forward and its backward, which records an event on the main stream,
makes the update stream wait it and only then queues [gradient all_to_all -> the owed zero -> claim ->
merge -> Adam -> replay]. That window issues no collective of its own, so the gradient exchange has
the fabric to itself; its deadline is still event k+1, two to four steps later. The order inside the
queue is forced: the zero must sit after the all_to_all that reads the gradient buffer and before the
new cycle's first accumulate_grad, and the replay must sit after the Adam whose last_event stamps it
reads.

Provenance: record #360 (ANVIL2): table size, two-channel layout, hashes, beta1 = 0 row-RMS Adam,
the per-cycle row cache and pull, the period-4 cadence.
"""
import numpy as np
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from track_1_short.perf.kernels.ngram_adam import adam_rows_, bring_rows_current, claim_rows
from track_1_short.perf.kernels.row_scatter import scatter_add_rows
from track_1_short.sharded_rows import RowOwners, RowPull

# 224x the 377,280-row bigram table of earlier records. Rows [0, V/2) are the bigram channel,
# [V/2, V) the trigram channel. Divisible by 8 (the shard) and < 2**31 (row ids are int32).
NGRAM_VOCAB_SIZE = 84_602_880
NGRAM_DIM = 768
# The shared +-1 sign pool. A power of two, so `& (rows - 1)` is the non-negative remainder of the
# wrapped int32 hash (values ~500-15000 gave similar results in the bigram record).
NGRAM_SIGN_POOL_ROWS = 8192

# Row hashes (record #360): multiply each token by a large odd constant, xor, reduce mod the channel
# size. int32 products wrap; torch's `%` is Python-signed, so the result is non-negative.
BIGRAM_ROW_MULS = (36313, 27191)            # x[t], x[t-1]
TRIGRAM_ROW_MULS = (17351, 60961, 45259)    # x[t], x[t-1], x[t-2]
# Sign-pool hashes (record #360), independent of the row hashes; computed in the embedding kernel
# (perf/kernels/ngram_embed.py). Position t reads pool row
#   bigram:  (m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 1
#   trigram: (m[2] * x[t-2] ^ m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 2
BIGRAM_SIGN_MULS = (48271, 30011)           # x[t], x[t-1]
TRIGRAM_SIGN_MULS = (58699, 39779, 26801)   # x[t], x[t-1], x[t-2]

# Adam for the table (record #360). lr and weight decay are Adam's scheduled base values times these
# multipliers; beta1 is 0 for the whole run.
NGRAM_LR_MUL = 70.0
NGRAM_WD_MUL = 5.0
NGRAM_ADAM_BETA2 = 0.95
# From this step the table updates every 4th step instead of every 2nd (record #360). One event then
# stands for two: beta2 is squared, so the second moment decays at the same rate per step, and the
# weight-decay multiplier doubles.
NGRAM_ADAM_PERIOD4_START = 336
NGRAM_WD_MUL_PERIOD4 = 10.0
# The longest cycle, in steps: the cache holds one cycle's rows (two per token per step).
MAX_CYCLE_STEPS = 4
# A cycle's gradient is summed per cache slot in fp16, goes between ranks as it stands and is summed per
# row by the owner in place, still in fp16 (record #360 rounds it to bf16 for the wire and merges in
# bf16; keeping the sum itself needs no copy and rounds one time less).
GRAD_ACCUM_DTYPE = torch.float16

assert NGRAM_VOCAB_SIZE % 2 == 0 and NGRAM_VOCAB_SIZE < 2 ** 31
assert NGRAM_SIGN_POOL_ROWS & (NGRAM_SIGN_POOL_ROWS - 1) == 0
# The last period-2 event (an odd step) is then followed by a whole period-4 cycle.
assert NGRAM_ADAM_PERIOD4_START % 4 == 0


# -----------------------------------------------------------------------------
# Hashing

def ngram_row_ids(x: Tensor) -> Tensor:
    """Token ids [T] (host) -> table row ids [2T] int32: out[:T] bigram rows, out[T:] trigram rows.

    Positions without enough history get the channel's reserved last row. Note the bigram hash at
    t = 1 reads the reserved id, not x[0], as its previous token (it is computed in place after
    out[0] is set, exactly as in record #360).
    """
    x = x.to(torch.int32)
    half = NGRAM_VOCAB_SIZE // 2
    bigram_mod, trigram_mod = half - 1, NGRAM_VOCAB_SIZE - half - 1
    n = x.numel()
    out = torch.empty(2 * n, dtype=torch.int32)
    bigram, trigram = out[:n], out[n:]
    bigram.copy_(x)
    bigram[0] = bigram_mod
    bigram[1:] = torch.bitwise_xor(BIGRAM_ROW_MULS[0] * bigram[1:], BIGRAM_ROW_MULS[1] * bigram[:-1]) % bigram_mod
    trigram[0] = trigram[1] = half + trigram_mod
    trigram[2:] = torch.bitwise_xor(
        torch.bitwise_xor(TRIGRAM_ROW_MULS[0] * x[2:], TRIGRAM_ROW_MULS[1] * x[1:-1]),
        TRIGRAM_ROW_MULS[2] * x[:-2],
    ) % trigram_mod + half
    return out


def sorted_unique_rows(row_id_arrays: list[np.ndarray]) -> np.ndarray:
    """The sorted-unique row ids of several steps' [2T] arrays (a cycle's want list), int32.

    A sort and a neighbour compare rather than np.unique: numpy 2 routes integer unique through a
    hash set, which record #360 measured at 15-60 ms on a cycle's ids against 2-6 ms for the sort.
    """
    rows = np.sort(np.concatenate(row_id_arrays))
    if rows.size:
        rows = rows[np.r_[True, rows[1:] != rows[:-1]]]
    return rows.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Update cadence

def is_update_step(step: int) -> bool:
    """Every odd step (with the other Adam params) until NGRAM_ADAM_PERIOD4_START, then steps = 3 mod 4."""
    if step >= NGRAM_ADAM_PERIOD4_START:
        return step % 4 == 3
    return step % 2 == 1


def adam_beta2_and_wd_mul(step: int) -> tuple[float, float]:
    if step >= NGRAM_ADAM_PERIOD4_START:
        return NGRAM_ADAM_BETA2 ** 2, NGRAM_WD_MUL_PERIOD4
    return NGRAM_ADAM_BETA2, NGRAM_WD_MUL


# -----------------------------------------------------------------------------
# The owner merge, in place

# The row tile of perf/kernels/row_scatter.py for a 768-wide row: 3 chunks of 256.
MERGE_BLOCK = 256


@triton.jit
def _merge_entries_kernel(G, ROWS, ENTRY_OF_ROW, D: tl.constexpr, BLOCK: tl.constexpr, NCHUNK: tl.constexpr):
    """One program per entry: add the entries of a row that are not its claimed entry into the claimed
    one, in place, so the gradient the Adam event reads at a claimed entry is the row's sum.

    Race-free without a second buffer: the only rows written are claimed entries, and a claimed entry is
    never a source (its program is the one that exits), so no program reads a row another writes. The
    adds of one destination are 16-bit atomics, as in row_scatter.py's merge.
    """
    pid = tl.program_id(0)
    row = tl.load(ROWS + pid)
    claim = tl.load(ENTRY_OF_ROW + row)
    if claim != pid:
        offs = tl.arange(0, BLOCK).to(tl.int64)
        src = pid.to(tl.int64) * D
        dst = claim.to(tl.int64) * D
        for k in tl.static_range(NCHUNK):
            val = tl.load(G + (src + k * BLOCK + offs)).to(tl.float32)
            tl.atomic_add(G + (dst + k * BLOCK + offs), val, sem="relaxed")


def merge_entries_(grad: Tensor, entries: Tensor, entry_of_row: Tensor):
    """Sum each row's entries into its claimed entry's row of `grad`, in place (claim_rows first).

    What it replaces: zeroing a second [entries, 768] buffer and scatter-adding every entry into it
    (perf/kernels/ngram_adam.py's merge), 2.3 GB of traffic per cycle, where this moves only the entries
    that are not their row's claim -- worth 0.08 s of the run, so those are a sizeable fraction of them.
    """
    assert grad.ndim == 2 and grad.is_contiguous() and grad.shape[1] % MERGE_BLOCK == 0
    assert entries.dtype == torch.int32 and entries.is_contiguous() and entries.numel() == grad.shape[0]
    assert entry_of_row.dtype == torch.int32 and entry_of_row.is_contiguous()
    if entries.numel():
        d = grad.shape[1]
        _merge_entries_kernel[(entries.numel(),)](grad, entries, entry_of_row, D=d, BLOCK=MERGE_BLOCK,
                                                  NCHUNK=d // MERGE_BLOCK, num_warps=4, num_stages=1)


# -----------------------------------------------------------------------------
# The one-byte wire: int8 rows, one fp32 scale each

@triton.jit
def _quant_rows_kernel(SRC, IDX, Q, SCALE, D: tl.constexpr, BLOCK: tl.constexpr, HAS_IDX: tl.constexpr):
    """One program per destination row: read row IDX[pid] of SRC (row pid without HAS_IDX), write its
    int8 quantisation and the fp32 scale that undoes it.

    The scale is the row's amax / 127, so the quantisation error is at most half a step of amax / 127
    everywhere in the row -- ~0.8% of the row's RMS for a row whose amax is ~3.3 sigma, where bf16's
    is ~0.2%. Rounding is half-away-from-zero (no libdevice call) and |q| <= 127 exactly, so the int8
    never overflows. The amax cap only matters for a row that overflowed to inf upstream: it makes that
    row zero rather than NaN.
    """
    pid = tl.program_id(0)
    if HAS_IDX:
        srow = tl.load(IDX + pid).to(tl.int64)
    else:
        srow = pid.to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    x = tl.load(SRC + (srow * D + offs), mask=mask, other=0.0).to(tl.float32)
    amax = tl.minimum(tl.max(tl.abs(x), axis=0), 1.0e30)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    y = x * inv
    tl.store(Q + (pid.to(tl.int64) * D + offs), (y + tl.where(y >= 0.0, 0.5, -0.5)).to(tl.int8), mask=mask)
    tl.store(SCALE + pid, amax * (1.0 / 127.0))


@triton.jit
def _dequant_rows_kernel(Q, SCALE, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row: undo _quant_rows_kernel into OUT's dtype."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    base = pid.to(tl.int64) * D
    q = tl.load(Q + (base + offs), mask=mask, other=0).to(tl.float32)
    tl.store(OUT + (base + offs), (q * tl.load(SCALE + pid)).to(OUT.dtype.element_ty), mask=mask)


def quantize_rows(src: Tensor, idx: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """(int8 rows, fp32 scales) of `src` (of its rows `idx`, gathered, if given)."""
    assert src.ndim == 2 and src.is_contiguous()
    n = src.shape[0] if idx is None else idx.numel()
    d = src.shape[1]
    q = torch.empty(n, d, dtype=torch.int8, device=src.device)
    scale = torch.empty(n, dtype=torch.float32, device=src.device)
    if n:
        assert idx is None or (idx.dtype == torch.int32 and idx.is_contiguous())
        _quant_rows_kernel[(n,)](src, q if idx is None else idx, q, scale, D=d,
                                 BLOCK=triton.next_power_of_2(d), HAS_IDX=idx is not None, num_warps=4)
    return q, scale


def dequantize_rows(q: Tensor, scale: Tensor, dtype: torch.dtype, out: Tensor | None = None) -> Tensor:
    """Undo quantize_rows into `out` (allocated in `dtype` if not given)."""
    assert q.ndim == 2 and q.dtype == torch.int8 and q.is_contiguous()
    assert scale.dtype == torch.float32 and scale.is_contiguous() and scale.numel() == q.shape[0]
    n, d = q.shape
    if out is None:
        out = torch.empty(n, d, dtype=dtype, device=q.device)
    assert out.shape == q.shape and out.is_contiguous()
    if n:
        _dequant_rows_kernel[(n,)](q, scale, out, D=d, BLOCK=triton.next_power_of_2(d), num_warps=4)
    return out


# -----------------------------------------------------------------------------
# The table

class _TableOwners(RowOwners):
    """RowOwners that ask every rank for its rows -- including this one -- and exchange on a group of
    their own.

    Why ask ourselves: sharded_rows.py keeps a pull's own block (want[lo:hi]) off the wire, so the rows
    the exchange carries are the want list with a hole in it. Both buffers of an event then have to be
    packed and unpacked around that hole: the gradient of the cycle is copied out of the per-slot sum
    into a wire buffer (a 568 MB read and a 568 MB write per cycle) and the rows that come back are
    copied into the cache around the block served locally (another 1.1 GB). Sending our own block to
    ourselves -- a local device copy inside the all_to_all -- makes both buffers contiguous in want
    order: the per-slot gradient sum IS the send buffer and the row exchange writes straight into the
    cache. It costs two 71 MB self-copies per cycle and saves ~2.3 GB of it.

    Why their own group: the default group's NCCL stream also carries the optimizer's bank
    reduce-scatters and all-gathers, and one FIFO means the table's rows cannot start before the
    gradient they were queued behind while the gathers the next step waits for queue behind both.
    The counts (CPU tensors, gloo) stay on the default group, where they cost 1.5 ms of a run.
    """

    def __init__(self, num_rows: int, rank: int, world_size: int, group):
        super().__init__(num_rows, rank, world_size)
        self.group = group

    def request(self, want: Tensor, want_host: np.ndarray) -> RowPull:
        """Stage 1: cut the sorted want list by owner and start the count exchange (async, gloo). As
        sharded_rows.RowOwners.request, but our own block is asked for like any other."""
        cuts = np.searchsorted(want_host, self.bounds)
        send_counts = np.diff(cuts)
        send = torch.from_numpy(send_counts.astype(np.int64))
        recv = torch.empty_like(send)
        work = dist.all_to_all_single(recv, send, async_op=True)
        return RowPull(want=want, lo=int(cuts[self.rank]), hi=int(cuts[self.rank + 1]),
                       send_counts=send_counts.tolist(), counts_exchange=(work, recv, send))

    def exchange_rows(self, send: Tensor, send_counts: list[int], recv_counts: list[int], out=None):
        """sharded_rows.all_to_all_rows on this table's group, always async, into `out` if given."""
        if out is None:
            out = send.new_empty((sum(recv_counts), *send.shape[1:]))
        work = dist.all_to_all_single(out, send, output_split_sizes=recv_counts, input_split_sizes=send_counts,
                                      group=self.group, async_op=True)
        return out, work

    def exchange_ids(self, pull: RowPull):
        """Stage 2: wait for the counts, then send each owner the ids we want from it (async)."""
        work, recv, _ = pull.counts_exchange
        work.wait()
        pull.counts_exchange = None
        pull.recv_counts = recv.tolist()
        pull.peer_want = pull.want  # every rank's block, ours included: the order the rows come back in
        pull.asked, pull.ids_exchange = self.exchange_rows(pull.peer_want, pull.send_counts, pull.recv_counts)

    @torch.no_grad()
    def start_fetch(self, pull: RowPull, shard: Tensor, out: Tensor, exact: bool = False) -> tuple:
        """Stage 3, async: serve the rows asked of us from `shard` (as they are now) and start receiving
        the whole want list into `out`. Returns what complete_fetch needs, which also holds the send
        buffers until it has waited.

        One byte per element on the wire: the gather out of the shard writes int8 rows and one fp32
        scale each instead of bf16 rows -- the same traffic, two bytes read and one written where it
        read two and wrote two -- the exchange then carries half the bytes, and the dequantise writes
        `out` where the exchange used to write it. So the 40 GB this takes off the run's second-biggest
        wire costs no extra pass over memory, unlike the gradient direction (send_grads).

        Two exchanges, not one: packing the scale into each row's record (a 784-byte int8 record, typed
        fp32 so one all_to_all carries both) measured no better over two runs and one of them 0.11 s
        worse, so the second collective is not what this costs -- and 768-byte records keep the
        payload's loads and stores at their widest.

        `exact` serves bf16 end to end, for the serves that are not a training read: the cold start and
        the validation pulls, so that what the final validation evaluates is exactly the shard.
        """
        if pull.ids_exchange is not None:
            pull.ids_exchange.wait()
            pull.ids_exchange = None
        rows = pull.asked - self.first_row
        if exact:
            served = shard.index_select(0, rows)
            _, work = self.exchange_rows(served, pull.recv_counts, pull.send_counts, out=out)
            return out, (work,), (served,), None
        q, scale = quantize_rows(shard, rows)
        q_recv, q_work = self.exchange_rows(q, pull.recv_counts, pull.send_counts)
        s_recv, s_work = self.exchange_rows(scale, pull.recv_counts, pull.send_counts)
        return out, (q_work, s_work), (q, scale), (q_recv, s_recv)

    @staticmethod
    def complete_fetch(fetch: tuple):
        """Wait for start_fetch's exchange(s) and, if it quantised, undo that into `out`."""
        out, works, _hold, quant = fetch
        for work in works:
            work.wait()
        if quant is not None:
            dequantize_rows(quant[0], quant[1], out.dtype, out=out)

    def send_grads(self, pull: RowPull, wire: Tensor):
        """The gradient of the pull's rows, in want order, to their owners along its routes reversed.
        received[i] is the gradient of row pull.asked[i].

        Still two bytes per element, where the rows below are one: a one-byte gradient wire measured
        +0.09 s. The quantise and the dequantise are each a whole extra pass over the gradient (the
        exchange reads the per-slot sum in place and writes the merge buffer in place, so there is no
        gather to fuse them into, as there is on the row side), and this exchange runs under the next
        step's backward (start_update), where those passes compete with it for the memory system while
        the bytes they save come off a wire that is no longer the step's critical path.
        """
        return self.exchange_rows(wire, pull.send_counts, pull.recv_counts)


class NgramTable:
    """This rank's shard of the n-gram table, its Adam state, and the model's row cache.

    Created in main(); `cache` is the model's `ngram_cache` buffer. Every method that exchanges rows
    is a collective: all ranks call it at the same point.
    """

    def __init__(self, cache: Tensor, max_step_tokens: int, max_events: int, rank: int, world_size: int,
                 device: torch.device):
        assert NGRAM_VOCAB_SIZE % world_size == 0
        assert cache.shape[1] == NGRAM_DIM and cache.dtype == torch.bfloat16
        # A cycle reads at most two rows per token of MAX_CYCLE_STEPS steps.
        max_cycle_rows = 2 * MAX_CYCLE_STEPS * max_step_tokens
        assert cache.shape[0] >= max_cycle_rows, "ngram_cache cannot hold a cycle"
        self.cache = cache
        self.rank = rank
        self.world_size = world_size
        self.device = device
        # The table's own process group, for its row and gradient exchanges (see _TableOwners).
        self.group = dist.new_group()
        self.owners = _TableOwners(NGRAM_VOCAB_SIZE, rank, world_size, self.group)
        self.first_row = self.owners.first_row
        rows = self.owners.rows_per_rank
        # 10.6M x 768 bf16 = 16.2 GB. Zero init, as record #360.
        self.shard = torch.zeros(rows, NGRAM_DIM, dtype=torch.bfloat16, device=device)
        # Adam's second moment, one fp32 scalar per row (beta1 = 0: no first moment), current as of the
        # row's last_event; beta2_history[e] is event e's beta2, for replaying the events a row missed.
        self.exp_avg_sq = torch.zeros(rows, dtype=torch.float32, device=device)
        self.last_event = torch.zeros(rows, dtype=torch.int32, device=device)
        self.beta2_history = torch.zeros(max_events + 1, dtype=torch.float32, device=device)
        # The owner merge's row -> claimed entry map (perf/kernels/ngram_adam.py); only the entries of the
        # current event are ever read, so it is never cleared.
        self.entry_of_row = torch.zeros(rows, dtype=torch.int32, device=device)
        # The live cycle's gradient summed per cache slot, rebuilt at each event.
        self.grad_per_slot = torch.empty(max_cycle_rows, NGRAM_DIM, dtype=GRAD_ACCUM_DTYPE, device=device)
        self.sinks: dict[int, Tensor] = {}  # tokens per step -> the zeros leaf, allocated once
        # The event's two streams (see the module docstring) and the events that order them against the
        # main stream and each other.
        self.land_stream = torch.cuda.Stream(device)      # serve the next cycle's rows, into every cache
        self.update_stream = torch.cuda.Stream(device)    # per-step gradient sum, exchange, merge, Adam
        self.backward_done = torch.cuda.Event()  # main: the backward whose sink gradients the event reads
        self.grads_read = torch.cuda.Event()     # update: those gradients have been summed per slot
        self.served = torch.cuda.Event()         # land: the next cycle's rows have left the shard
        self.updated = torch.cuda.Event()        # update: the event's Adam is in the shard
        self.land_ready = torch.cuda.Event()     # land: the cache holds the next cycle's rows
        self.update_gate = torch.cuda.Event()    # main: the step the stashed update is to run under
        self.reset()

    def reset(self):
        """Back to the initial state, in place: the post-warmup reset must not copy 16 GB."""
        self.shard.zero_()
        self.exp_avg_sq.zero_()
        self.last_event.zero_()
        self.adam_events = 0
        self.live: RowPull | None = None    # the pull of the cycle being trained
        self.cached: RowPull | None = None  # the pull whose rows the cache holds now
        self.grads_in_flight = None
        self.landing = None  # a started land: (pull, the fetch complete_land waits), until complete_land
        self.grad_steps = 0  # steps of the live cycle whose gradient is in grad_per_slot
        # (pull, step, lr, weight_decay, eps) of an event whose merge + Adam the update stream still
        # owes; the step after the event queues it there (start_update), under its own compute.
        self.stashed = None
        self.zero_owed = 0          # rows of grad_per_slot the live cycle owes a zero, with the stash
        self.bring_owed = None      # the pull whose own rows owe a second-moment replay, with the stash
        self.grads_pending = False  # the update stream's gradient sum, unwaited by the main stream
        self.land_pending = False   # a land the main stream has not waited for yet

    # ---- the event's streams ----

    @staticmethod
    def _keep(stream: torch.cuda.Stream, *tensors: Tensor):
        """Tell the allocator that `stream` reads these tensors: they were allocated on the main stream
        (a pull's want and asked lists, a step's cache slots) and are dropped while the event runs."""
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)

    def wait_land(self):
        """Main stream, before a forward: wait for the rows of the cycle it reads (the exchange writes
        the cache itself) and for the per-slot sum of the gradient its backward is about to overwrite.
        Two event waits, never a synchronize; the merge and the Adam are not waited for (train_gpt.py)."""
        main = torch.cuda.current_stream()
        if self.grads_pending:
            self.grads_pending = False
            main.wait_event(self.grads_read)
        if self.land_pending:
            self.land_pending = False
            main.wait_event(self.land_ready)

    def finish_event(self):
        """Nothing of an event may still be in flight: queue a stashed update and join both streams into
        the calling one (the warmup teardown, each validation, the synchronous land path)."""
        self._flush_event()
        self.grads_pending = self.land_pending = False
        main = torch.cuda.current_stream()
        main.wait_stream(self.land_stream)
        main.wait_stream(self.update_stream)

    @torch.no_grad()
    def start_update(self):
        """Main stream, between a step's forward and its backward: run the stashed half of the previous
        event HERE, under this step's compute, rather than in the event step's optimizer phase.

        A stream only decouples work; it does not move it in time. Queued at the event the gradient
        all_to_all shares the fabric with the optimizer's bank reduce-scatters and all-gathers, and the
        merge and Adam share the memory system with them. Gated on an event recorded here it runs under
        a forward and a backward that issue no collective of their own. Its deadline is the next event,
        two to four steps away, and the step's own accumulate_grad is queued behind it on the same
        stream, so nothing is dropped and nothing is applied twice.
        """
        if self.stashed is None and not self.zero_owed and self.bring_owed is None:
            return
        self.update_gate.record(torch.cuda.current_stream())
        self.update_stream.wait_event(self.update_gate)
        self._flush_event()

    # ---- pulling rows into the cache ----

    def request(self, want: Tensor, want_host: np.ndarray) -> RowPull:
        """Stage 1 (sharded_rows.RowOwners.request), into a cache that must hold the rows."""
        assert want_host.shape[0] <= self.cache.shape[0], f"{want_host.shape[0]} rows overflow ngram_cache"
        return self.owners.request(want, want_host)

    def exchange_ids(self, pull: RowPull):
        self.owners.exchange_ids(pull)

    def land(self, pull: RowPull):
        """All of stage 3 on the calling stream (the cold start and the validation pulls): both event
        streams are drained first, so the shard this serves from is the one the last event left."""
        self.finish_event()
        inline = torch.cuda.current_stream()
        self.start_land(pull, stream=inline)
        self.complete_land(stream=inline)

    @torch.no_grad()
    def start_land(self, pull: RowPull, stream=None):
        """Stage 3, first half: serve the rows our peers (and we) asked for and start receiving the whole
        want list straight into the cache.

        The serve reads the shard as the previous event left it, so the cycle about to be trained sees
        one event of staleness, the same for our own rows and our peers'. The rows also go on the wire
        before this event's gradient, which the next step queues (start_update), so what the next
        forward needs does not queue behind it.
        """
        assert self.landing is None
        target = self.land_stream if stream is None else stream
        if stream is None:
            # The forward that read the cache rows we overwrite, and the previous event's Adam, whose
            # rows we are about to serve.
            target.wait_event(self.backward_done)
            target.wait_event(self.updated)
        with torch.cuda.stream(target):
            self._keep(target, pull.want, pull.asked)
            # Quantised on the async path (a training read), exact on the synchronous one.
            self.landing = (pull, self.owners.start_fetch(pull, self.shard, self.cache[:pull.num_rows],
                                                          exact=stream is not None))
            if stream is None:
                self.served.record(target)

    @torch.no_grad()
    def complete_land(self, stream=None):
        """Stage 3, second half: wait for the rows and undo their quantisation into the cache (the exact
        serves land in it directly). On the land stream the main stream waits the event this records
        (wait_land), not the rows."""
        pull, fetch = self.landing
        self.landing = None
        target = self.land_stream if stream is None else stream
        with torch.cuda.stream(target):
            self.owners.complete_fetch(fetch)
            if stream is None:
                self.land_ready.record(target)
                self.land_pending = True
        self.cached = pull

    def fill(self, want_host: np.ndarray) -> RowPull:
        """All three stages inline (cold start)."""
        pull = self.request(torch.from_numpy(want_host).to(self.device), want_host)
        self.exchange_ids(pull)
        self.land(pull)
        return pull

    def restore_cache(self):
        """Re-land the live cycle's rows after an eval pull overwrote the cache. The routes are reused,
        not re-exchanged; the rows come back as the last event left them (only an intermediate
        validation, which the record run does not do, lands mid-cycle)."""
        self.land(self.live)

    def slots(self, pull: RowPull, row_ids: Tensor) -> Tensor:
        """Cache slots of `row_ids` ([2T] int32, device), all of which `pull` fetched."""
        return torch.searchsorted(pull.want, row_ids, out_int32=True)

    def eval_pulls(self, row_id_batches: list[Tensor]) -> list[RowPull]:
        """Pulls for eval batches (device row ids): the want lists are built on the device and all
        batches' counts go in ONE exchange (record #360's batched validation fill). Not yet landed."""
        wants = [torch.unique(ids) for ids in row_id_batches]  # sorted
        bounds = torch.from_numpy(self.owners.bounds).to(self.device)
        cuts = torch.stack([torch.searchsorted(w.to(torch.int64), bounds) for w in wants]).cpu().numpy()
        send = cuts[:, 1:] - cuts[:, :-1]              # [batches, world], our own block included
        send_t = torch.from_numpy(np.ascontiguousarray(send.T))  # row r: what we ask rank r, per batch
        recv_t = torch.empty_like(send_t)
        dist.all_to_all_single(recv_t, send_t)
        recv = recv_t.numpy().T
        pulls = []
        for b, want in enumerate(wants):
            assert want.shape[0] <= self.cache.shape[0], f"{want.shape[0]} rows overflow ngram_cache"
            pull = RowPull(want=want.to(torch.int32), lo=int(cuts[b, self.rank]), hi=int(cuts[b, self.rank + 1]),
                           send_counts=send[b].tolist(), recv_counts=recv[b].tolist())
            pull.peer_want = pull.want
            pull.asked, work = self.owners.exchange_rows(pull.peer_want, pull.send_counts, pull.recv_counts)
            work.wait()
            pulls.append(pull)
        return pulls

    def load_eval_batch(self, row_ids: Tensor) -> Tensor:
        """Pull one eval batch's rows into the cache; returns its slots."""
        pull, = self.eval_pulls([row_ids])
        self.land(pull)
        return self.slots(pull, row_ids)

    # ---- the training cycle ----

    def start_cycle(self, pull: RowPull):
        """Make a landed pull the live cycle: the forward reads its rows, and its event's gradient rows
        are its rows. Its per-slot gradient buffer owes a zero, which start_update queues with the rest
        of the stashed event: behind the all_to_all that reads that buffer (the previous event's
        gradient is still in it, unsent) and ahead of this cycle's first accumulate_grad."""
        assert self.cached is pull
        self.live = pull
        self.grad_steps = 0
        self.zero_owed = pull.num_rows

    def grad_sink(self, num_tokens: int) -> Tensor:
        """The zeros leaf the forward adds to its looked-up rows; its gradient is the rows' gradient.
        One per step size, allocated once (the step graphs bake it)."""
        if num_tokens not in self.sinks:
            self.sinks[num_tokens] = torch.zeros(2 * num_tokens, NGRAM_DIM, dtype=torch.bfloat16,
                                                 device=self.device, requires_grad=True)
        return self.sinks[num_tokens]

    @torch.no_grad()
    def accumulate_grad(self, slots: Tensor, sink_grad: Tensor):
        """Add one step's sink gradient to the cycle's per-slot sum, on the update stream, right after
        the backward that produced it. `sink_grad` is free again once grads_read is reached (wait_land
        before the next forward), so no step of a cycle needs a gradient buffer of its own."""
        assert self.cached is self.live, "the forward read a cache that holds no training cycle"
        target = self.update_stream
        # Nothing may be added to a buffer that still owes its zero (start_update does this already;
        # this is the guard for any step that did not call it).
        self._flush_event()
        self.backward_done.record(torch.cuda.current_stream())
        target.wait_event(self.backward_done)
        with torch.cuda.stream(target):
            # The slots were allocated on the main stream and are dropped at the end of this step; the
            # gradient buffer is the backward graph's own and is never freed.
            self._keep(target, slots)
            scatter_add_rows(sink_grad, slots, self.grad_per_slot[:self.live.num_rows])
            self.grads_read.record(target)
            self.grads_pending = True
        self.grad_steps += 1

    @torch.no_grad()
    def send_grads(self):
        """Event, part 1: hand the cycle's per-slot gradient sum to the event. It is already the wire
        buffer -- want order, one row per cache slot -- so there is nothing to pack, and the exchange
        itself leaves in the next step (start_update), after the next cycle's rows: nothing waits for the
        gradient until the next event, two to four steps away."""
        pull = self.live
        assert self.grad_steps, "Adam event with no gradient since the last one"
        assert self.stashed is None, "the previous event's update never ran"
        assert pull.num_rows <= self.grad_per_slot.shape[0], f"{pull.num_rows} cycle rows > gradient buffer"
        self.grads_in_flight = self.grad_per_slot[:pull.num_rows]

    @torch.no_grad()
    def adam_update(self, step: int, lr: float, weight_decay: float, eps: float):
        """Event, part 2 (collective with send_grads): stash the row-sparse Adam event. The next step
        queues it on the update stream (start_update), after the serve of the next cycle's rows -- so no
        row is served from a half-updated shard, and the rows the next forward needs go on the wire
        first. `lr` and `weight_decay` are Adam's scheduled base values."""
        assert self.stashed is None, "the previous event's update never ran"
        assert self.grads_in_flight is not None, "an Adam event whose gradient was never summed"
        self.stashed = (self.live, step, lr, weight_decay, eps)

    @torch.no_grad()
    def _flush_event(self, stream=None):
        """Everything the update stream owes, in the one order that is correct: send each row's gradient
        sum to its owner, zero the new cycle's slots (the exchange above has read them), sum the received
        entries per row, one Adam event over every row the cycle touched, then the second-moment replay
        of the next cycle's own rows (it reads the last_event stamps the Adam just wrote). No-op if
        nothing is owed.

        Several ranks may send the same row: its entries are summed (merge_entries_ above), then averaged
        over ranks (the dense equivalent is an AVG reduce-scatter; the 1/world rides grad_mul / sq_mul).
        The update itself, and its plain-torch equivalent, are in perf/kernels/ngram_adam.py.
        """
        if self.stashed is None and not self.zero_owed and self.bring_owed is None:
            return
        target = self.update_stream if stream is None else stream
        if stream is None and self.stashed is not None:
            target.wait_event(self.served)  # the rows of the next cycle left the shard before Adam writes it
        with torch.cuda.stream(target):
            if self.stashed is not None:
                pull, step, lr, weight_decay, eps = self.stashed
                self.stashed = None
                wire = self.grads_in_flight
                self.grads_in_flight = None
                received, work = self.owners.send_grads(pull, wire)
                work.wait()
                self._zero_slots()  # the gradient has left the buffer; the new cycle may add to it
                self._keep(target, pull.asked)
                entries = pull.asked - self.first_row

                beta2, wd_mul = adam_beta2_and_wd_mul(step)
                lr = lr * NGRAM_LR_MUL
                self.adam_events += 1
                event = self.adam_events
                # Before the update reads it; a fill, not an H2D, so there is no pinned buffer to race.
                self.beta2_history[event].fill_(beta2)
                # One merge slot per entry: each row's gradient sums into its claimed entry's slot, in the
                # buffer the exchange filled (merge_entries_ above; no second buffer to allocate and zero).
                claim_rows(entries, self.entry_of_row)
                merge_entries_(received, entries, self.entry_of_row)
                adam_rows_(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history, received, entries,
                           self.entry_of_row, event, beta2=beta2, eps=eps,
                           step_size=lr * (1 - beta2 ** event) ** 0.5,  # beta1 = 0: no first-moment bias
                           decay=lr * lr * weight_decay * wd_mul,
                           grad_mul=1 / self.world_size, sq_mul=(1 - beta2) / self.world_size ** 2)
                if stream is None:
                    self.updated.record(target)  # the next event's serve reads what this wrote
            else:
                self._zero_slots()
            if self.bring_owed is not None:
                pull, self.bring_owed = self.bring_owed, None
                if self.adam_events:
                    self._keep(target, pull.want)
                    bring_rows_current(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history,
                                       pull.want[pull.lo:pull.hi] - self.first_row, self.adam_events)

    def _zero_slots(self):
        """The live cycle's per-slot gradient buffer, on the calling stream. Called once per cycle, with
        the previous event's gradient already on the wire and before this cycle's first accumulate_grad."""
        if self.zero_owed:
            self.grad_per_slot[:self.zero_owed].zero_()
            self.zero_owed = 0

    @torch.no_grad()
    def bring_current(self, pull: RowPull):
        """Stash the rows whose missed second-moment decays are to be replayed (the next cycle's rows this
        rank owns: they are the rows its next event touches, so the replay stays short). _flush_event
        queues it behind the event's Adam, which stamps the rows it touched; it writes no row a serve
        reads."""
        self.bring_owed = pull
