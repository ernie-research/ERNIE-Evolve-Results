"""
Single-file, single-GPU language model pretraining script.

Architecture: 9-layer, 768-dim transformer with ReLU^2 MLP, sliding-window
attention (short/long window pattern), Muon + AdamW optimizer with per-component
learning rates, QK-norm, RoPE, value embeddings, and softcap.

Feature stack: n-gram hash features (bigram + avalanche-hashed sparse trigram with
frequency-aware per-row learning rate), n-gram value-path injection, intra-document
position buckets, a causal induction head, and a learned multi-layer readout with
per-head attention temperature. Trained within a fixed wall-clock budget.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Full training configuration, baked in with os.environ.setdefault so that
# `python train.py` alone is the configuration that produced train.log; an explicit
# environment variable still overrides any entry. The os.environ.get(...) defaults
# further down are only the fallbacks for knobs this block does not set.
for _k, _v in {
    "SEED": "0",
    "USE_AVALANCHE_TRIGRAM": "1",
    # Table sizing is a zero-sum split of the ~135 GB the two n-gram tables own.
    # Measured on the full 490.7M-token training stream with train.py's own hash
    # functions: the corpus has 99.29M distinct causal trigram types but only
    # 10.44M bigram types, so with twice the bigram rows and a much smaller trigram
    # table the trigram table ran at 97.5% occupancy / 3.77 types per row (purity 0.698
    # -- 30% of what flows through a row belongs to a different trigram) while the
    # bigram table sat at 49.5% occupancy (purity 0.982). Halving the bigram table costs
    # almost nothing (purity 0.982 -> 0.960) and, together with dropping the trigram
    # momentum buffer below, funds the large trigram table (purity 0.832 at 72M rows).
    "TRIGRAM_TABLE_SIZE": "76800000",
    "BIGRAM_TABLE_SIZE": "8388608",
    "BIGRAM_FACTORED": "1",
    "TRIGRAM_LR": "0.45",
    "TRIGRAM_LR_FLOOR": "0.60",
    "BIGRAM_LR": "0.65",
    "USE_TRIGRAM_FREQ_LR": "1",
    "TRIGRAM_FREQ_LR_K": "16.0",
    "USE_BIGRAM_FREQ_LR": "1",
    "BIGRAM_FREQ_LR_K": "128.0",
    "BIGRAM_FREQ_LR_FLOOR": "0.0",
    "USE_AVALANCHE_BIGRAM": "0",
    "USE_INTRADOC_POS": "1",
    "POS_BUCKET_LR": "0.15",
    "USE_LM_HEAD_BIAS": "0",
    "LM_HEAD_BIAS_INIT": "1",
    "LM_HEAD_BIAS_LR": "0.03",
    "USE_ATTN_SCALE": "1",
    "ATTN_SCALE_LR": "0.005",
    "USE_LAYER_READOUT": "1",
    "LAYER_READOUT_LR": "0.01",
    "USE_READOUT_NORM": "0",
    "USE_DENSE_DWA": "0",
    "DENSE_DWA_LR": "0.01",
    "DENSE_DWA_K": "8",
    "USE_BIGRAM_INDUCTION": "1",
    "BIGRAM_INDUCTION_LR": "0.6",
    "INDUCTION_ORDER": "backoff",
    "INDUCTION_DOCMASK": "1",
    "USE_INDUCTION_DIST": "1",
    "INDUCTION_DIST_LR": "0.25",
    "NUM_INDUCTION_DIST_BUCKETS": "32",
    "USE_UNIGRAM_INDUCTION": "1",
    "UNIGRAM_INDUCTION_LR": "0.2",
    "UNIGRAM_INDUCTION_DOCMASK": "1",
    "COMPILE_LOOKUPS": "1",
    "USE_BIGRAM_RECALL": "0",
    "BIGRAM_RECALL_LR": "0.4",
    # How the trunk READS the sparse n-gram stack (see the coupling block below).
    # The trigram table's 76.8M rows are the most data-starved parameters here: 490M
    # training tokens spread over ~70M touched rows is single-digit gradient updates
    # per row, while any DENSE tensor on the same path sees all 490M. So the form of
    # the map from a row to what the trunk consumes is worth far more than the row's
    # own ability to shape itself, and without these knobs that map is the identity:
    # the same raw vector is added to the residual stream and handed verbatim to blocks
    # 6,7,8 as attention V, where only a per-head scalar gate acts on it. These knobs
    # give that map learned structure -- per-channel scales, identity at init -- for
    # 3072 dense params, one extra optimizer tensor and +0.4% of the step.
    "NGRAM_RES_GATE": "1",
    "NGRAM_V_SCALE": "1",
    "NGRAM_COUPLING_LR": "0.005",
    # Learnable per-head attention sink (see the fa4::fa4_causal_sink op below).
    # Every softmax row gets one extra logit s_h with an implicit zero value
    # vector, so a head can decline to attend:
    #     out_i = sum_j e^{l_ij} v_j / (e^{s_h} + sum_j e^{l_ij})
    # The motivation is specific to this dataloader, not generic: prepare.py packs
    # best-fit documents into every 2048-token row (2.87 documents per row
    # measured on the real loader), and the attention mask is plain causal +
    # sliding window, so it never separates them. 10.8% of the keys a 256-window
    # head attends to and 42.4% of the keys a 1024-window head attends to are
    # therefore foreign text, and a sink-free softmax is forced to spend all of
    # its mass on them. ATTN_SINK_LAYERS selects which blocks get one:
    # "all" | "short" | "long" | an explicit index list.
    "USE_ATTN_SINK": "1",
    "ATTN_SINK_LAYERS": "all",
    # Sink schedule. With init 0.0 and lr 0.01 this parameter never arrives: mean|s|
    # climbs monotonically through the whole run and is still rising at the final step,
    # so attention spends the first ~40% of training with less than half the sink logit
    # it ends with. Two zero-cost corrections (constants only -- no kernel, no tensor
    # shape, no extra step time):
    #   * start at the operating point instead of at zero, with the depth ramp the
    #     converged profile has (about +0.5 at blocks 0-1 rising to +4.0 at blocks 6-8):
    #     s_i = INIT + i*INIT_SLOPE.
    #   * let it move at a rate that converges inside the budget. A full-budget sweep
    #     over lr {0.01, 0.03, 0.06, 0.12} has an interior optimum at 0.06, with the same
    #     ordering in the training loss at every matched step window, so it is not an
    #     artifact of the final evaluation; 0.12 overshoots the level the sink wants. At
    #     0.06 mean|s| plateaus well before the end of the run.
    "ATTN_SINK_INIT": "1.0",
    "ATTN_SINK_INIT_SLOPE": "0.35",
    "ATTN_SINK_LR": "0.06",
    # X0_LR, NGRAM_COUPLING_LR and ATTN_SINK_LRM_FLOOR were each swept on top of this
    # sink config and are left at their defaults: no setting beat them beyond
    # run-to-run noise.
    # Document-local attention on the long-window layers (see fa4::fa4_varlen_sink).
    # Only the three window-1024 layers, because that is where contamination is
    # worst (42.4% of attended keys are foreign, vs 10.8% at window 256) and
    # because the varlen call is +25% of that layer's attention: buying the same
    # thing at the six short layers would cost four times as much for a quarter of
    # the contamination. Capacity 256 segments covers the ~184 documents a 64x2048
    # batch really holds; if a batch ever exceeds it the surplus boundaries are
    # dropped, which merges two documents back into plain causal attention rather than
    # failing.
    "USE_DOC_LOCAL": "1",
    "DOC_LOCAL_LAYERS": "long",
    "DOC_LOCAL_CAP": "256",
    # Inside a document the sliding window is redundant book-keeping: the mean
    # packed document is ~676 tokens, so dropping the window on the doc-local
    # layers lets the long tail of >1024-token documents attend to their whole
    # prefix at (roughly) no extra cost.
    "DOC_LOCAL_FULL_DOC": "1",
    # Eval-time doc expert: settings chosen offline on dumped per-token losses.
    "MIX_SHINGLE_K": "6",
    "MIX_SHINGLE_S": "1",
    "MIX_DOC_M": "16",
    "MIX_DOC_L": "32768",
    # Dense causal query offsets: re-retrieve more often early in a document (4, 12)
    # and fill the gaps of a geometric grid (40..1792).
    "MIX_QOFF": "4,8,12,16,24,32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,1792",
    # Doc-side reliability features for the mixture gate (see CorpusMixer._features and
    # _DOCFEAT_SIZES): doc max-continuation share x log2 rank of the doc argmax in the
    # net's top-16, and log2 c(doc ctx) x doc share. All read the context (and the net's
    # logits) only, never the target. Picked on the held-out slice A/B (gate fitted on
    # the even slice batches, scored on the odd ones) printed after the score.
    "MIX_DOCFEAT": "ds_rb,dc_ds",
    # Corpus memory beyond the trained-on stream (see _BgBuild): the n-gram tables and the
    # document index are built from every training-split document, including the ones
    # the network never reached (unread documents, documents still in the loader's
    # best-fit buffer, the cropped tails of long documents). "all" | "0" ("0": tables
    # from the trained-on stream only, built after the loop). Under the background
    # loader the worker reports its document state with every batch (see
    # BackgroundTrainLoader.HDR), so gate-slice documents stay excluded.
    "MIX_EXTRA": "all",
    # Seconds of the 300 s budget held back from the training loop for the post-loop
    # work (charged to training_seconds). The table build itself runs in a background
    # worker that is started at the first timed step (see _BgBuild.launch), so it is
    # wall-clock inside the timed loop and slows the steps it overlaps. What remains
    # after the loop is the slice forward with the final weights, the slice doc expert
    # on the GPU, the table upload and the gate fit: 7.6-8.9 s measured on B200 (slice
    # forward 3.9-5.8 s across runs, doc expert + fit ~3.1 s); 9.5 s leaves ~0.5 s over
    # the slowest of them.
    "MIX_RESERVE": "9.5",
    # Gate fit on the whole slice (sparse segment-sum backward, see _fit_sparse), 300
    # Adam iterations.
    "MIX_FIT_TOK": "0",
    "MIX_FIT_ITERS": "300",
    # Per-layer MLP hidden widths (one entry per layer, multiples of 128). The
    # n-gram tables and the value/bigram paths already supply shallow features, so
    # MLP capacity is worth more in the deep layers: at equal total width a deep-heavy
    # split beat uniform widths and shallow-heavy splits lost. The fused FP8 MLP kernels
    # below (fc GEMM + relu^2 + cast, dgrad GEMM + relu^2 backward) pay for the extra
    # width; widths were raised only while the added step time still paid for itself,
    # and deep layers wider than 6400 were worse.
    "MLP_HIDDEN_LIST": "4736,4736,4736,4736,6400,6400,6400,6400,6400",
}.items():
    os.environ.setdefault(_k, _v)

import gc
import contextlib
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

# ---------------------------------------------------------------------------
# Corpus memory for the eval-time mixture: tokenizer workers.
#
# The eval-time n-gram tables and document index (see _BgBuild) are built from the
# whole training split. _EXTRA_WORKER tokenizes training shards exactly as the loader
# does and streams (doc lengths, tokens) back over a pipe. The workers are spawned by
# _BgBuild.launch() at the first timed step of the training loop, never at import, so
# every second they spend is wall-clock inside training_seconds.
# ---------------------------------------------------------------------------
import sys
import subprocess as _subprocess
import threading as _threading
import numpy as _np

_EXTRA_WORKER = r'''
import sys, pickle, struct, numpy as np, pyarrow.parquet as pq
enc = pickle.load(open(sys.argv[1], "rb"))
nth = int(sys.argv[2])
out = sys.stdout.buffer
for it in sys.argv[3:]:
    si, path = it.split(":", 1)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    ids = enc.encode_ordinary_batch(texts, num_threads=nth)
    del texts
    lens = np.fromiter((len(i) for i in ids), dtype=np.int64, count=len(ids))
    flat = np.fromiter((v for i in ids for v in i), dtype=np.int16, count=int(lens.sum()))
    del ids
    out.write(struct.pack("<qqq", int(si), lens.size, flat.size))
    out.write(lens.tobytes())
    out.write(flat.tobytes())
    out.flush()
'''


# ---------------------------------------------------------------------------
# Background corpus-memory build.
#
# The n-gram tables and the document index are built from WHOLE training-split
# documents (every document of every training shard, minus the documents popped into
# the held-out gate slice). That set does not depend on how far training gets, so it
# can be built while the network trains instead of after the loop. A nice-19 worker
# process pinned to MIX_BG_THREADS cores (on the other NUMA node; the main process and
# the loader are pinned off them when the loop starts) reads the tokenizer pipes,
# assembles the stream, and per order hashes exactly as CorpusMixer.build does
# (key = (ctx << 13) | next), sorts 2*threads chunks in parallel and merges them by
# the top 8 key bits (256 buckets). The slice's n-gram chain (pre_ng) is evaluated
# there too. Results land in a memfd shared with this process; after the loop only
# the slice forward, the slice doc expert, the upload and the gate fit remain. The
# tokenizers and this worker are spawned at the first timed step of the loop
# (_BgBuild.launch, never earlier; checked after the loop), so the whole build is
# wall-clock inside training_seconds. If it has not finished when the loop ends, the
# post-loop waits for it inside the timer. The document index splits documents into
# <=MAX_SEQ_LEN-token pieces, so the doc expert's buffers keep their granularity.
# ---------------------------------------------------------------------------
_BG_WORKER = r'''
import sys, os, json, struct, time, mmap, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

cfg = json.loads(sys.argv[1])
NT = int(cfg["threads"])
pool = ThreadPoolExecutor(NT)
_out = sys.stdout
_olock = threading.Lock()


def emit(**kw):
    with _olock:
        _out.write(json.dumps(kw) + "\n")
        _out.flush()


M50 = (1 << 50) - 1
M42 = (1 << 42) - 1
M31 = (1 << 31) - 1


def mix64(h):
    h = h ^ ((h >> 33) & M31)
    h = h * -49064778989728563
    h = h ^ ((h >> 33) & M31)
    h = h * -4265267296055464877
    h = h ^ ((h >> 33) & M31)
    return h


def pmap(f, items):
    return list(pool.map(f, items))


def spans(n, parts):
    step = max(1, -(-n // parts))
    return [(a, min(n, a + step)) for a in range(0, n, step)]


# ---------------- arena (memfd shared with the parent) ----------------
mm = mmap.mmap(int(cfg["fd"]), int(cfg["cap"]), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
layout = {}
_aoff = [0]


def alloc(name, dtype, count):
    dt = np.dtype(dtype)
    off = _aoff[0]
    nb = dt.itemsize * int(count)
    _aoff[0] = off + ((nb + 65535) // 65536) * 65536
    assert _aoff[0] <= int(cfg["cap"]), "bg arena too small"
    layout[name] = [off, dt.str, int(count)]
    return np.frombuffer(mm, dtype=dt, count=int(count), offset=off)


# ---------------- tokenized shards from the tokenizer pipes ----------------
data = {}
t_tok = []


def rd(fd, cnt):
    f = os.fdopen(fd, "rb")
    for _ in range(cnt):
        hdr = f.read(24)
        if len(hdr) < 24:
            break
        si, nd, nt = struct.unpack("<qqq", hdr)
        lens = np.frombuffer(f.read(nd * 8), dtype=np.int64)
        flat = np.frombuffer(f.read(nt * 2), dtype=np.int16)
        data[si] = (lens, flat)
    f.close()
    t_tok.append(time.time())


ths = [threading.Thread(target=rd, args=(fd, cnt), daemon=True) for fd, cnt in cfg["pipes"]]
for th in ths:
    th.start()

# ---------------- slice state + slice tokens from the parent (stdin) ----------------
sin = sys.stdin.buffer
msg = json.loads(sin.readline())
t_msg = time.time()
orders = msg["orders"]
mults = msg["mults"]
K = int(msg["shingle_k"])
SS = int(msg["shingle_s"])
DFMAX = int(msg["df_max"])
PIECE = int(msg["piece"])
V = int(msg["V"])
bos = int(msg["bos"])
nsl, B, T = int(msg["nslice"]), int(msg["B"]), int(msg["T"])
nq = nsl * B * T


def readn(n):
    b = bytearray(n)
    v = memoryview(b)
    o = 0
    while o < n:
        r = sin.readinto(v[o:])
        if not r:
            raise RuntimeError("short stdin")
        o += r
    return b


qx = np.frombuffer(readn(nq * 2), dtype=np.int16).astype(np.int64)
qy = np.frombuffer(readn(nq * 2), dtype=np.int16).astype(np.int64)
n0 = int(msg["n0"])
buf0 = np.asarray(msg["buf0"], dtype=np.int64)
for th in ths:
    th.join()
t_b0 = time.time()
emit(ev="start", t_tok=max(t_tok) if t_tok else t_b0, t_msg=t_msg, t_build_start=t_b0)

# ---------------- whole-document stream (training split minus gate-slice docs) ----------------
nrows, off = cfg["nrows"], cfg["off"]
sis = sorted(data)
assert len(sis) == len(nrows), f"tokenizers delivered {len(sis)} of {len(nrows)} shards"
for si in sis:
    assert data[si][0].size == nrows[si]
lens = np.concatenate([data[si][0] for si in sis])
flat = np.concatenate([data[si][1] for si in sis])
gidx = np.concatenate([off[si] + np.arange(nrows[si], dtype=np.int64) for si in sis])
data.clear()
in_slice = (gidx < n0) & ~np.isin(gidx, buf0)
keep = ~in_slice
Lk = lens[keep] + 1                      # + BOS
G = int(Lk.size)
N = int(Lk.sum())
ds = np.cumsum(Lk) - Lk                  # doc starts in the stream
tok_keep = np.repeat(keep, lens)
S = alloc("S", np.int16, N)
isb = np.zeros(N, dtype=bool)
isb[ds] = True
S[~isb] = flat[tok_keep]
S[ds] = bos
del flat, tok_keep, isb
g_keep = alloc("gidx", np.int64, G)
g_keep[:] = gidx[keep]
dsa = alloc("ds", np.int64, G)
dsa[:] = ds
docidx = np.repeat(np.arange(G, dtype=np.int64), Lk)
lbd = np.arange(N, dtype=np.int64) - ds[docidx]          # distance to the doc's BOS
del docidx
po = lbd % PIECE                                          # offset inside its <=PIECE-token piece
pstart = np.flatnonzero(po == 0)
npieces = int(pstart.size)
assert npieces < (1 << 21), npieces
doc_start = alloc("doc_start", np.int64, npieces)
doc_start[:] = pstart
doc_len = alloc("doc_len", np.int64, npieces)
doc_len[:] = np.diff(pstart, append=N)
pid = np.cumsum(po == 0) - 1
cnt = np.bincount(S[1:].astype(np.int64), minlength=V).astype(np.float64)
p_uni = alloc("p_uni", np.float64, V)
p_uni[:] = (cnt + 1.0) / (cnt.sum() + V)
xp = S.astype(np.int64) + 1
h = np.zeros(N, dtype=np.int64)
CH = spans(N, 2 * NT)
t_stream = time.time()
emit(ev="stream", N=N, docs=G, slice_docs=int(in_slice.sum()), pieces=npieces, t=t_stream)

# ---------------- slice queries (pre_ng, the gate fit's n-gram chain) ----------------
pos = np.arange(T, dtype=np.int64)
qxr = qx.reshape(-1, T)
qlbd = (pos - np.maximum.accumulate(np.where(qxr == bos, pos, 0), axis=1)).reshape(-1)
qxp = qx + 1
qh = np.zeros(nq, dtype=np.int64)
qp = p_uni[qy]
qtop = np.zeros(nq, dtype=np.int64)
qc0 = np.zeros(nq, dtype=np.float64)
qc1 = np.zeros(nq, dtype=np.float64)


def f32d(a):
    return a.astype(np.float32).astype(np.float64)


def query(U, C, base, key):
    n = U.size
    if n == 0:
        z = np.zeros(base.shape, dtype=np.float64)
        return z, z, z
    lo = np.searchsorted(U, base)
    hi = np.searchsorted(U, base + 8192)
    i = np.searchsorted(U, key)
    ii = np.minimum(i, n - 1)
    match = (i < hi) & (U[ii] == key)
    cxy = np.where(match, C[ii + 1] - C[ii], 0)
    cc = C[hi] - C[lo]
    n1 = hi - lo
    return f32d(cxy), f32d(cc), f32d(n1)


def merge_runs(runs, name_u, name_c):
    """runs: sorted int64 arrays (values >= 0). Sorted unique values (+ int32 count cumsum)."""
    edges = np.array([b << 55 for b in range(1, 256)], dtype=np.int64)
    bnd = [np.concatenate([[0], np.searchsorted(r, edges), [r.size]]) for r in runs]

    def one(b):
        s = np.concatenate([r[bd[b]:bd[b + 1]] for r, bd in zip(runs, bnd)])
        if s.size == 0:
            return s, np.zeros(0, dtype=np.int64)
        s.sort()
        fl = np.empty(s.size, dtype=bool)
        fl[0] = True
        np.not_equal(s[1:], s[:-1], out=fl[1:])
        ix = np.flatnonzero(fl)
        return s[ix], np.diff(np.append(ix, s.size))
    res = pmap(one, range(256))
    nu = [r[0].size for r in res]
    uo = np.concatenate([[0], np.cumsum(nu)])
    U = alloc(name_u, np.int64, int(uo[-1]))
    C = None
    if name_c is not None:
        C = alloc(name_c, np.int32, int(uo[-1]) + 1)
        C[0] = 0
        tot = np.concatenate([[0], np.cumsum([int(r[1].sum()) for r in res])])

    def put(b):
        u, c = res[b]
        U[uo[b]:uo[b + 1]] = u
        if C is not None:
            C[uo[b] + 1:uo[b + 1] + 1] = (np.cumsum(c) + tot[b]).astype(np.int32)
    pmap(put, range(256))
    return U, C


tim = {}
done_i = 0
for oi, k in enumerate(orders):
    t0 = time.time()
    for i in range(done_i, k):
        m = int(mults[i])

        def upd(ab, i=i, m=m):
            a, b = ab
            a2 = max(a, i)
            if a2 < b:
                h[a2:b] += xp[a2 - i:b - i] * m
        pmap(upd, CH)
        if i == 0:
            qh += qxp * m
        else:
            qh[i:] += qxp[:-i] * m
    done_i = k
    want_sh = (k == K)

    def keys(ab, k=k, want_sh=want_sh):
        a, b = ab
        hm = mix64(h[a:b])
        ctx = (hm >> 14) & M50
        nb = min(N, b + 1)
        nx = xp[a + 1:nb] - 1
        if nx.size < b - a:
            nx = np.concatenate([nx, np.zeros(b - a - nx.size, dtype=np.int64)])
        v = lbd[a:b] >= k - 1
        if b == N:
            v[-1] = False
        key = ((ctx << 13) | nx)[v]
        key.sort()
        e = None
        if want_sh:
            sm = po[a:b] >= k - 1
            if SS > 1:
                sm &= ((hm >> 3) & (SS - 1)) == 0
            e = (((hm[sm] >> 22) & M42) << 21) | pid[a:b][sm]
            e.sort()
        return key, e
    res = pmap(keys, CH)
    t1 = time.time()
    U, C = merge_runs([r[0] for r in res], f"U{k}", f"C{k}")
    shr = [r[1] for r in res] if want_sh else None
    del res
    t2 = time.time()
    # slice chain at this order (queries sorted by key: monotone searches)
    qhm = mix64(qh)
    base = ((qhm >> 14) & M50) << 13
    key = base | qy
    qv = qlbd >= k - 1
    perm = np.argsort(key, kind="stable")
    qs = spans(nq, NT)

    def qpart(ab):
        a, b = ab
        pp = perm[a:b]
        return query(U, C, base[pp], key[pp])
    parts = pmap(qpart, qs)
    cxy = np.empty(nq); cc = np.empty(nq); n1 = np.empty(nq)
    for (a, b), (x1, x2, x3) in zip(qs, parts):
        pp = perm[a:b]
        cxy[pp] = x1; cc[pp] = x2; n1[pp] = x3
    has = qv & (cc > 0)
    qp = np.where(has, (cxy + n1 * qp) / np.maximum(cc + n1, 1.0), qp)
    qtop = np.where(has, oi + 1, qtop)
    qc0 = np.where(has, cc, qc0)
    qc1 = np.where(has, n1, qc1)
    del U, C
    if want_sh:
        Pp, _ = merge_runs(shr, "P", None)
        del shr
        g = Pp >> 21
        fl = np.empty(g.size, dtype=bool)
        fl[0] = True
        np.not_equal(g[1:], g[:-1], out=fl[1:])
        ix = np.flatnonzero(fl)
        nH = int(ix.size)
        H = alloc("H", np.int64, nH); H[:] = g[ix]
        Hs = alloc("Hstart", np.int64, nH); Hs[:] = ix
        dfa = alloc("df", np.int64, nH); dfa[:] = np.diff(np.append(ix, g.size))
        idf = np.round(16.0 * np.log(npieces / dfa.astype(np.float64))).astype(np.int64)
        idfa = alloc("idf", np.int64, nH); idfa[:] = np.where(dfa <= DFMAX, idf, 0)
        del g, fl, ix, idf
    tim[k] = [round(t1 - t0, 2), round(t2 - t1, 2), round(time.time() - t2, 2)]
    emit(ev="order", k=k, t=time.time(), tim=tim[k])

for nm, a in (("pre_p", qp), ("pre_top", qtop), ("pre_c0", qc0), ("pre_c1", qc1)):
    dst = alloc(nm, a.dtype, a.size)
    dst[:] = a
emit(ev="done", t=time.time(), layout=layout, size=_aoff[0], N=N, docs=G, pieces=npieces,
     slice_docs=int(in_slice.sum()), tim=tim, t_build_start=t_b0, t_stream=t_stream)
'''


def _parse_cpulist(s):
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _bg_cpu_plan(n):
    """(build cores, the rest, GPU NUMA node): by default (MIX_BG_NODE=other) the build
    gets the last n allowed cores of the NUMA node the GPU is NOT attached to, so its
    sorts/merges do not share memory bandwidth with the main process and the loader
    (MIX_BG_NODE=gpu: the GPU's own node). Everything else keeps the rest."""
    allowed = sorted(os.sched_getaffinity(0))
    node = None
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
        drv = "/sys/bus/pci/drivers/nvidia"
        devs = sorted(d for d in os.listdir(drv) if ":" in d and "." in d)
        node = int(open(os.path.join(drv, devs[int(vis)], "numa_node")).read())
    except Exception:
        node = None
    pool = allowed
    if node is not None and node >= 0:
        try:
            loc = set(_parse_cpulist(open(f"/sys/devices/system/node/node{node}/cpulist").read()))
            if os.environ.get("MIX_BG_NODE", "other") == "other":
                cand = [c for c in allowed if c not in loc]
            else:
                cand = [c for c in allowed if c in loc]
            if len(cand) >= n + 8:
                pool = cand
        except Exception:
            pass
    n = max(1, min(n, len(pool) - 4))
    build = pool[-n:]
    rest = [c for c in allowed if c not in set(build)]
    return build, rest, node


class _BgBuild:
    """Background whole-document table build, plus the loader document tracking the
    gate-slice exclusion needs. Nothing is spawned until launch()."""

    def __init__(self, mode):
        import prepare as _prep
        import ctypes as _ct
        import json as _json
        self._json = _json
        self.mode = mode
        val_path = os.path.join(_prep.DATA_DIR, _prep.VAL_FILENAME)
        self.paths = [p for p in _prep.list_parquet_files() if p != val_path]
        import pyarrow.parquet as _pq
        self.nrows = [_pq.ParquetFile(p).metadata.num_rows for p in self.paths]
        self.off = [0]
        for n in self.nrows:
            self.off.append(self.off[-1] + n)
        todo = list(range(len(self.paths)))
        nw = max(1, min(len(todo), int(os.environ.get("MIX_EXTRA_WORKERS", "5"))))
        nth = os.environ.get("MIX_EXTRA_THREADS", "8")
        tok_path = os.path.join(_prep.TOKENIZER_DIR, "tokenizer.pkl")
        nbt = int(os.environ.get("MIX_BG_THREADS", "32"))
        self.cpus_build, self.cpus_rest, self.numa = _bg_cpu_plan(nbt)
        libc = _ct.CDLL(None, use_errno=True)
        self.memfd = libc.memfd_create(b"mix_bg", 1)   # MFD_CLOEXEC (pass_fds re-enables it for the worker)
        if self.memfd < 0:
            raise OSError(_ct.get_errno(), "memfd_create")
        self.cap = int(float(os.environ.get("MIX_BG_ARENA_GB", "400")) * 1e9)
        os.ftruncate(self.memfd, self.cap)          # sparse: only written pages take memory
        self._launch_args = (todo, nw, nth, tok_path)
        self.proc, self.procs, self.reader = None, [], None
        self.t_launch = self.t_sent = None
        self._slice_msg = None
        self.events, self.result, self.err = [], None, None
        self.t_done_main = self.t_mapped = None
        self.mm = None
        # loader tracking
        self.ids = {}
        self.n = 0
        self.buf = None
        self.bg = None
        self.on = True
        self.snap_n, self.snap_buf = 0, []

    def launch(self):
        """Start the tokenizers and the build worker. Called once, at the first timed step
        of the training loop, so every second of the build is inside training_seconds;
        the slice message buffered by send_slice is delivered here."""
        assert self.proc is None, "background build launched twice"
        todo, nw, nth, tok_path = self._launch_args
        self.t_launch = time.time()
        pipes = []
        for w in range(nw):
            mine = [f"{si}:{self.paths[si]}" for si in todo[w::nw]]
            p = _subprocess.Popen([sys.executable, "-c", _EXTRA_WORKER, tok_path, nth] + mine,
                                  stdout=_subprocess.PIPE, preexec_fn=lambda: os.nice(10))
            self.procs.append(p)
            pipes.append([p.stdout.fileno(), len(mine)])
        cfg = {"threads": len(self.cpus_build), "fd": self.memfd, "cap": self.cap, "pipes": pipes,
               "nrows": self.nrows, "off": self.off}
        cpus = list(self.cpus_build)

        def _pre():
            os.nice(19)
            os.sched_setaffinity(0, cpus)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        self.proc = _subprocess.Popen([sys.executable, "-c", _BG_WORKER, self._json.dumps(cfg)],
                                      stdin=_subprocess.PIPE, stdout=_subprocess.PIPE,
                                      pass_fds=[self.memfd] + [fd for fd, _ in pipes],
                                      preexec_fn=_pre, env=env)
        for p in self.procs:
            p.stdout.close()          # the worker owns the read ends now
        self.reader = _threading.Thread(target=self._events, daemon=True)
        self.reader.start()
        if self._slice_msg is not None:
            self._write_slice()

    def _events(self):
        try:
            for line in self.proc.stdout:
                ev = self._json.loads(line)
                ev["t_main"] = time.time()
                self.events.append(ev)
                if ev.get("ev") == "done":
                    self.t_done_main = ev["t_main"]
                    self._map(ev["size"])
                    self.result = ev
                    break
        except Exception as e:  # reported after the loop
            self.err = e
        try:
            self.proc.wait(timeout=60)
            for p in self.procs:
                p.wait(timeout=5)
        except Exception:
            pass

    def _map(self, size):
        import mmap as _mmap
        import ctypes as _ct
        size = max(65536, int(size))
        self.mm = _mmap.mmap(self.memfd, size, _mmap.MAP_SHARED, _mmap.PROT_READ | _mmap.PROT_WRITE)
        # Pre-fault the mapping now (ctypes drops the GIL) so the post-loop upload does not
        # pay page faults: MADV_POPULATE_WRITE (23), else MADV_POPULATE_READ (22).
        try:
            libc = _ct.CDLL(None, use_errno=True)
            libc.madvise.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_int]
            addr = _ct.addressof(_ct.c_char.from_buffer(self.mm))
            if libc.madvise(addr, size, 23) != 0:
                libc.madvise(addr, size, 22)
        except Exception:
            pass
        self.t_mapped = time.time()

    def send_slice(self, mix_slice, state, msg):
        n0, buf0 = state
        B, T = mix_slice[0][0].shape
        msg = dict(msg, nslice=len(mix_slice), B=int(B), T=int(T), n0=int(n0), buf0=[int(v) for v in buf0])
        qx = torch.cat([a.reshape(-1) for a, _ in mix_slice]).to(torch.int16).cpu().numpy()
        qy = torch.cat([b.reshape(-1) for _, b in mix_slice]).to(torch.int16).cpu().numpy()
        self._slice_msg = ((self._json.dumps(msg) + "\n").encode(), qx.tobytes(), qy.tobytes())
        if self.proc is not None:
            self._write_slice()

    def _write_slice(self):
        self.t_sent = time.time()
        for b in self._slice_msg:
            self.proc.stdin.write(b)
        self.proc.stdin.flush()
        self.proc.stdin.close()
        self._slice_msg = None

    def pin_others(self, pids):
        """Keep every thread of this process and of `pids` off the build cores."""
        rest = set(self.cpus_rest)
        n = 0
        for pid in [os.getpid()] + [p for p in pids if p]:
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    try:
                        os.sched_setaffinity(int(tid), rest)
                        n += 1
                    except OSError:
                        pass
            except OSError:
                pass
        return n

    def wait(self, timeout):
        if self.reader is None:
            return False
        self.reader.join(timeout=timeout)
        return self.result is not None

    # ---- loader tracking ----
    def wrap(self, tokenizer):
        orig = tokenizer.encode

        def encode(text, prepend=None, num_threads=8):
            out = orig(text, prepend=prepend, num_threads=num_threads)
            if self.on and isinstance(text, list):
                base = self.n
                for i, l in enumerate(out):
                    self.ids[id(l)] = base + i
                self.n += len(out)
            return out
        tokenizer.encode = encode

    def attach(self, loader):
        if hasattr(loader, "doc_buf"):
            # background loader: the worker process reports (docs encoded, ids still in
            # its best-fit buffer) with every batch, as of the batch last handed out
            self.bg = loader
        else:
            self.buf = loader.gi_frame.f_locals["doc_buffer"]

    def state(self):
        if self.bg is not None:
            return self.bg.doc_n, self.bg.doc_buf.tolist()
        return self.n, [self.ids[id(l)] for l in self.buf]

    def snap(self):
        self.snap_n, self.snap_buf = self.state()

    def tensors(self):
        dmap = {"<i2": torch.int16, "<i4": torch.int32, "<i8": torch.int64, "<f8": torch.float64,
                "|b1": torch.bool}
        out = {}
        for nm, (off, dt, cnt) in self.result["layout"].items():
            if cnt == 0:
                out[nm] = torch.empty(0, dtype=dmap[dt])
            else:
                out[nm] = torch.frombuffer(self.mm, dtype=dmap[dt], count=cnt, offset=off)
        return out

    def kill(self):
        for p in [self.proc] + list(self.procs):
            if p is None:
                continue
            try:
                p.kill()
            except Exception:
                pass


_g_mx = torch.Generator().manual_seed(0x5eed + 1)
_MIX_MULTS_X = (torch.randint(-(2 ** 62), 2 ** 62, (64,), generator=_g_mx, dtype=torch.int64) * 2 + 1).tolist()
_MIX_EXTRA = os.environ.get("MIX_EXTRA", "all")
_extra = None
_MIX_BG = (os.environ.get("MIX_ENABLE", "1") == "1" and os.environ.get("MIX_BG_BUILD", "1") == "1"
           and _MIX_EXTRA == "all")
if _MIX_BG:
    _extra = _BgBuild(_MIX_EXTRA)   # nothing is spawned until _extra.launch() in the loop
# adamw_step_fused / muon_step_fused are compiled with fullgraph=True and specialise
# on (shape, dtype), so the number of DISTINCT parameter shapes the optimizer sees is
# a recompile count. torch's default limit is 8 and this configuration already needs 7
# for AdamW alone (two [vocab, d] dtypes, [32, d], [33, d], [V+1, d], [9], [6]); the
# packed n-gram coupling scale below adds another, and the variant that overflows the
# limit raises FailOnRecompileLimitHit *at runtime*, mid-training. Raising the ceiling
# only allows more specialised kernels to be compiled (each still static, each still
# fullgraph), it does not make any existing graph dynamic.
for _lim in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _lim):
        setattr(torch._dynamo.config, _lim, 64)

# ---- Measured dead ends in inductor's codegen knobs --------------------------------
# rms_norm BACKWARD is spread over ~20 distinct inductor kernels (one per fusion site,
# because every layer's ve/ng/bg/sink/doc-local context differs) that run at only
# 1.2-3.6 TB/s, so it looks like the obvious prize. Both codegen knobs that should
# collect it were measured and do not:
#   * coordinate_descent_tuning (+ check_all_directions): applies cleanly, benchmarks
#     tile/warp geometry per kernel, and buys nothing (same mean dt, same peak VRAM).
#   * triton.multi_kernel = 1: CRASHES this torch during its own warmup benchmark --
#     `TypeError: only integer tensors of a single element can be converted to an
#     index` from static_cuda_launcher.run, routed through torch/utils/_device.py
#     __torch_function__ (the model is built under a device context). Setting
#     use_static_cuda_launcher = False first does not avoid it.
# The remaining slack is not reachable from a config flag; it needs the fusion sites
# themselves to change shape.

def _make_sdpa_fallback():
    def _sdpa(q, k, v, causal=True, window_size=(-1, -1), learnable_sink=None):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if learnable_sink is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            # Exact sink: one extra column in the softmax whose logit is s_h and
            # whose value vector is zero, so it only enters the denominator.
            T = q.shape[-2]
            att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            if causal:
                att = att.masked_fill(
                    torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
            s = learnable_sink.to(torch.float32).view(1, -1, 1, 1).expand(att.shape[0], -1, T, 1)
            p = torch.softmax(torch.cat([att.float(), s], dim=-1), dim=-1)[..., :T]
            out = p.to(v.dtype) @ v
        return out.transpose(1, 2).contiguous()
    return _sdpa

try:
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 10:
        from flash_attn.cute import flash_attn_func as _fa4_raw
        from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

        @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
        def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
            ws = (window_left, 0) if window_left > 0 else (None, None)
            out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
            return out, lse

        @_fa4_causal_op.register_fake
        def _fa4_causal_fake(q, k, v, window_left):
            B, T, H, D = q.shape
            return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

        def _fa4_setup_context(ctx, inputs, output):
            q, k, v, window_left = inputs
            out, lse = output
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.window_left = window_left

        @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
        def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                        window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            wl = window_left if window_left > 0 else None
            dq, dk, dv = _fa4_bwd_raw(
                q, k, v, out, grad_output, lse,
                causal=True, window_size_left=wl, window_size_right=0,
            )
            return dq, dk, dv

        @_fa4_bwd_op.register_fake
        def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
            return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        def _fa4_backward(ctx, grad_output, grad_lse):
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
            return dq, dk, dv, None

        _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

        # ---- learnable per-head attention sink ------------------------------
        # Deliberately a SECOND pair of ops rather than an Optional[Tensor] on the
        # pair above: a block configured without a sink then still dispatches to
        # the byte-identical sink-free op, so "sink on layer subset X" is a
        # clean A/B and no schema change can perturb the sink-free path.
        #
        # flash-attn-4's fwd folds e^{s_h} into the running softmax denominator
        # (cost: one extra fp32 per row, measured 0.2232 -> 0.2239 ms at w256 on
        # GB200, i.e. free), and its bwd derives dsink from the row dpsum it
        # already computes, so no dlse needs to flow: grad_lse is None here and
        # dsink is still exact. That bwd is the whole price, +11.5%/+13.0% of the
        # attention backward, ~1.0 ms per step across all nine layers (GB200).
        @torch.library.custom_op("fa4::fa4_causal_sink", mutates_args=())
        def _fa4_causal_sink_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                window_left: int, sink: torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
            ws = (window_left, 0) if window_left > 0 else (None, None)
            out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws,
                                learnable_sink=sink, return_lse=True)
            return out, lse

        @_fa4_causal_sink_op.register_fake
        def _fa4_causal_sink_fake(q, k, v, window_left, sink):
            B, T, H, D = q.shape
            return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

        def _fa4_sink_setup_context(ctx, inputs, output):
            q, k, v, window_left, sink = inputs
            out, lse = output
            ctx.save_for_backward(q, k, v, out, lse, sink)
            ctx.window_left = window_left

        # dsink outside the kernel. In flash-attn-4 the sink enters the backward
        # only through the postprocess kernel, which reduces dsink with ONE CTA per
        # head looping over all 131072 rows -- a serial tail that makes the
        # postprocess ~2x its bandwidth time (measured on GB200: bwd 0.872 ->
        # 0.817 ms at w256, 1.852 -> 1.634 ms doc-local). dq/dk/dv do not depend on
        # the sink at all (it is already folded into lse), so they are bit-identical
        # (checked with torch.equal) when the sink is left out, and
        #     dsink_h = -sum_rows exp(s_h - lse) * rowsum(dO * O)
        # is the same per-row term the kernel sums (dsink_row_term), as a plain
        # torch reduction. Dense (short) layers take rowsum(dO*O) from the dpsum
        # buffer FA4's preprocess already wrote (dO there is a bare GEMM output, so
        # recomputing it would be a separate 402 MB read); the doc-local layers
        # recompute it in the graph, where inductor fuses it into the attn_gate
        # backward kernel that produces dO. (The kernel's lse == -inf special case
        # cannot occur: every causal row attends at least to itself.)
        #
        # flash-attn-4 allocates dpsum inside _flash_attn_bwd and hands it to its
        # module-level _bwd_preprocess; wrapping that function lets the dense sink
        # op keep a reference to it. Only active while the box is armed (holds one
        # None), so every other backward call is untouched.
        import flash_attn.cute.interface as _fa4_iface
        _fa4_dpsum_box = []
        _fa4_bwd_pre_orig = _fa4_iface._bwd_preprocess

        def _fa4_bwd_pre_capture(out, dout, dpsum, *args, **kwargs):
            if len(_fa4_dpsum_box) == 1 and _fa4_dpsum_box[0] is None:
                _fa4_dpsum_box[0] = dpsum
            return _fa4_bwd_pre_orig(out, dout, dpsum, *args, **kwargs)

        # the original looks its own compile cache up through the module global
        for _attr, _val in vars(_fa4_bwd_pre_orig).items():
            setattr(_fa4_bwd_pre_capture, _attr, _val)
        _fa4_iface._bwd_preprocess = _fa4_bwd_pre_capture
        FA4_DSINK_TORCH = os.environ.get("FA4_DSINK_TORCH", "1") == "1"

        # dQ postprocess (fp32 dq_accum -> bf16 dq, times softmax_scale) as a Triton
        # kernel. flash-attn-4's CuTe version stages every 128x128 tile through smem
        # and runs at ~half of bandwidth (nsys: 154 us/call, 9 calls/step); this
        # reads the accumulator in place in the SM100 2-CTA layout the bwd kernel
        # writes. Element (row r, col c) of each 128-row tile lives at flat offset
        #   (r//64)*8192 + ((c%64)//4)*512 + (c//64)*256 + (r%64)*4 + c%4
        # (derived by pushing index-coded accumulators through the original kernel).
        # It computes the same fp32 product and the same RN cast, so dq is
        # bit-identical (checked with torch.equal against the original, dense and
        # padded-varlen, at the training shapes): 87/97 us vs 154 us. Only the exact
        # configuration that was verified is routed here; anything else, or
        # any failure on the first call, goes to the original.
        FA4_DQ_POST_TRITON = os.environ.get("FA4_DQ_POST_TRITON", "1") == "1"
        import triton as _tr
        import triton.language as _tl

        @_tr.jit
        def _fa4_dq_post_kernel(acc_ptr, dq_ptr, cu_ptr, scale, T,
                                s_acc_b, s_acc_h, s_dq_b, s_dq_t, s_dq_h,
                                VARLEN: _tl.constexpr, NCU: _tl.constexpr,
                                NSEARCH: _tl.constexpr, BLOCK_T: _tl.constexpr):
            pid_t = _tl.program_id(0)
            h = _tl.program_id(1)
            b = _tl.program_id(2)
            t = pid_t * BLOCK_T + _tl.arange(0, BLOCK_T)
            tmask = t < T
            if VARLEN:
                # last sequence index with cu[i] <= t (padded tail entries are
                # zero-length duplicates of the final offset, so the last match is
                # the sequence that actually holds t)
                lo = _tl.zeros([BLOCK_T], dtype=_tl.int32)
                hi = _tl.full([BLOCK_T], NCU - 1, dtype=_tl.int32)
                for _ in _tl.static_range(NSEARCH):
                    mid = (lo + hi + 1) // 2
                    ok = _tl.load(cu_ptr + mid) <= t
                    lo = _tl.where(ok, mid, lo)
                    hi = _tl.where(ok, hi, mid - 1)
                cb = _tl.load(cu_ptr + lo)
                # flash-attn's padded_offset_q for sequence lo, plus the row in it
                P = (cb + lo * 128) // 128 * 128 + (t - cb)
            else:
                P = t
            tile = P // 128
            r = P % 128
            c = _tl.arange(0, 128)
            off = (tile * 16384 + (r // 64) * 8192 + (r % 64) * 4)[:, None] + \
                  (((c % 64) // 4) * 512 + (c // 64) * 256 + c % 4)[None, :]
            x = _tl.load(acc_ptr + b * s_acc_b + h * s_acc_h + off,
                         mask=tmask[:, None], other=0.0)
            _tl.store(dq_ptr + b * s_dq_b + h * s_dq_h + t[:, None] * s_dq_t + c[None, :],
                      (x * scale).to(_tl.bfloat16), mask=tmask[:, None])

        _fa4_post_orig = _fa4_iface._bwd_postprocess_convert
        _fa4_post_state = {"ok": FA4_DQ_POST_TRITON}

        def _fa4_post_fast(accum, output, scale, cu_seqlens, seqused, *args, **kwargs):
            if (_fa4_post_state["ok"] and seqused is None
                    and len(args) >= 7 and args[0] == 100 and args[2:7] == (128, 128, 128, 1, False)
                    and kwargs.get("sink_tensors") is None
                    and kwargs.get("use_2cta_instrs") is True
                    and kwargs.get("cluster_size", 1) == 1
                    and kwargs.get("cu_total_m_blocks") is None
                    and kwargs.get("hdim_multiple_of", 32) == 32
                    and not kwargs.get("fake_mode", False)
                    and accum is not None and accum.dtype == torch.float32
                    and output.dtype == torch.bfloat16 and output.shape[-1] == 128
                    and accum.stride(-1) == 1 and output.stride(-1) == 1
                    and ((cu_seqlens is None and accum.dim() == 3 and output.dim() == 4)
                         or (cu_seqlens is not None and accum.dim() == 2
                             and output.dim() == 3 and cu_seqlens.is_contiguous()))):
                try:
                    if cu_seqlens is None:
                        Bq, Tq, Hq, _ = output.shape
                        _fa4_dq_post_kernel[(_tr.cdiv(Tq, 64), Hq, Bq)](
                            accum, output, accum, float(scale), Tq,
                            accum.stride(0), accum.stride(1),
                            output.stride(0), output.stride(1), output.stride(2),
                            VARLEN=False, NCU=1, NSEARCH=1, BLOCK_T=64, num_warps=8)
                    else:
                        TOTq, Hq, _ = output.shape
                        ncu = cu_seqlens.shape[0]
                        _fa4_dq_post_kernel[(_tr.cdiv(TOTq, 64), Hq, 1)](
                            accum, output, cu_seqlens, float(scale), TOTq,
                            0, accum.stride(0), 0, output.stride(0), output.stride(1),
                            VARLEN=True, NCU=ncu, NSEARCH=max(1, (ncu - 1).bit_length()),
                            BLOCK_T=64, num_warps=8)
                    return
                except Exception as e:  # first-call compile failure: keep the original
                    print(f"[fa4] Triton dq postprocess disabled ({type(e).__name__}: {e})")
                    _fa4_post_state["ok"] = False
            return _fa4_post_orig(accum, output, scale, cu_seqlens, seqused, *args, **kwargs)

        for _attr, _val in vars(_fa4_post_orig).items():
            setattr(_fa4_post_fast, _attr, _val)
        _fa4_iface._bwd_postprocess_convert = _fa4_post_fast

        @torch.library.custom_op("fa4::fa4_bwd_sink", mutates_args=())
        def _fa4_bwd_sink_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                             window_left: int, sink: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            wl = window_left if window_left > 0 else None
            if FA4_DSINK_TORCH:
                # Same kernels minus the sink, then dsink from the dpsum = rowsum(dO*O)
                # the preprocess kernel already wrote (captured above): a 3 MB fp32
                # reduction instead of the postprocess' one-CTA-per-head serial loop.
                _fa4_dpsum_box.clear()
                _fa4_dpsum_box.append(None)
                dq, dk, dv = _fa4_bwd_raw(
                    q, k, v, out, grad_output, lse,
                    causal=True, window_size_left=wl, window_size_right=0,
                )
                dpsum = _fa4_dpsum_box.pop()
                _fa4_dpsum_box.clear()
                if dpsum is not None and dpsum.dim() == 3 and dpsum.shape[:2] == lse.shape[:2]:
                    T = lse.shape[-1]
                    sp = torch.exp(sink.float().view(1, -1, 1) - lse)
                    dsink = (-(sp * dpsum[..., :T]).sum((0, 2))).to(sink.dtype)
                    return dq, dk, dv, dsink
            dq, dk, dv, dsink = _fa4_bwd_raw(
                q, k, v, out, grad_output, lse,
                causal=True, window_size_left=wl, window_size_right=0,
                learnable_sink=sink,
            )
            return dq, dk, dv, dsink

        @_fa4_bwd_sink_op.register_fake
        def _fa4_bwd_sink_fake(q, k, v, out, grad_output, lse, window_left, sink):
            return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
                    torch.empty_like(sink))

        def _dsink_torch(sink, lse, out, grad_output):
            # doc-local layers: lse (H, TOT), out/grad_output (TOT, H, D). dO here is
            # the attn_gate backward's output, so inductor fuses this rowsum into it.
            dp = (grad_output.float() * out.float()).sum(-1).t()     # (H, TOT)
            sp = torch.exp(sink.float().unsqueeze(1) - lse)
            return (-(sp * dp).sum(1)).to(sink.dtype)

        def _fa4_sink_backward(ctx, grad_output, grad_lse):
            q, k, v, out, lse, sink = ctx.saved_tensors
            dq, dk, dv, dsink = torch.ops.fa4.fa4_bwd_sink(
                q, k, v, out, grad_output, lse, ctx.window_left, sink)
            return dq, dk, dv, None, dsink

        _fa4_causal_sink_op.register_autograd(
            _fa4_sink_backward, setup_context=_fa4_sink_setup_context)

        # ---- document-local (varlen) attention, with sink --------------------
        # The sink lets a head decline to attend; this deletes the keys it should
        # be declining. prepare.py's loader packs 2.87 documents into every 2048-token
        # row and the mask is causal+window only, so at window 1024 42.4% of the
        # keys a query attends to are a DIFFERENT document. Handing the kernel one
        # cu_seqlens that starts a new sequence at every BOS removes them exactly.
        #
        # Static-shape safety: cu_seqlens has a fixed capacity and is padded past the
        # real document count with duplicate copies of the final offset, so the
        # trailing sequences are zero-length and the SHAPE never moves -- only the
        # values do, and no host sync is needed to learn the real count. (Checked
        # on GB200: the padded form is accepted by both fwd and bwd and dq/dsink
        # come back finite.)
        from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_raw

        @torch.library.custom_op("fa4::fa4_varlen_sink", mutates_args=())
        def _fa4_varlen_sink_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                window_left: int, sink: torch.Tensor,
                                cu_seqlens: torch.Tensor, max_seqlen: int
                                ) -> tuple[torch.Tensor, torch.Tensor]:
            ws = (window_left, 0) if window_left > 0 else (None, None)
            out, lse = _fa4_varlen_raw(
                q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True, window_size=ws, learnable_sink=sink, return_lse=True)
            return out, lse

        @_fa4_varlen_sink_op.register_fake
        def _fa4_varlen_sink_fake(q, k, v, window_left, sink, cu_seqlens, max_seqlen):
            TOT, H, D = q.shape
            return torch.empty_like(q), torch.empty(H, TOT, device=q.device, dtype=torch.float32)

        def _fa4_varlen_setup_context(ctx, inputs, output):
            q, k, v, window_left, sink, cu_seqlens, max_seqlen = inputs
            out, lse = output
            ctx.save_for_backward(q, k, v, out, lse, sink, cu_seqlens)
            ctx.window_left = window_left
            ctx.max_seqlen = max_seqlen

        @torch.library.custom_op("fa4::fa4_varlen_bwd_sink", mutates_args=())
        def _fa4_varlen_bwd_sink_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    out: torch.Tensor, grad_output: torch.Tensor,
                                    lse: torch.Tensor, window_left: int, sink: torch.Tensor,
                                    cu_seqlens: torch.Tensor, max_seqlen: int
                                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            wl = window_left if window_left > 0 else None
            dq, dk, dv, dsink = _fa4_bwd_raw(
                q, k, v, out, grad_output, lse,
                causal=True, window_size_left=wl, window_size_right=0,
                learnable_sink=sink, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            )
            return dq, dk, dv, dsink

        @_fa4_varlen_bwd_sink_op.register_fake
        def _fa4_varlen_bwd_sink_fake(q, k, v, out, grad_output, lse, window_left,
                                      sink, cu_seqlens, max_seqlen):
            return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
                    torch.empty_like(sink))

        @torch.library.custom_op("fa4::fa4_varlen_bwd", mutates_args=())
        def _fa4_varlen_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               out: torch.Tensor, grad_output: torch.Tensor,
                               lse: torch.Tensor, window_left: int,
                               cu_seqlens: torch.Tensor, max_seqlen: int
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            wl = window_left if window_left > 0 else None
            dq, dk, dv = _fa4_bwd_raw(
                q, k, v, out, grad_output, lse,
                causal=True, window_size_left=wl, window_size_right=0,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            )
            return dq, dk, dv

        @_fa4_varlen_bwd_op.register_fake
        def _fa4_varlen_bwd_fake(q, k, v, out, grad_output, lse, window_left,
                                 cu_seqlens, max_seqlen):
            return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        def _fa4_varlen_backward(ctx, grad_output, grad_lse):
            q, k, v, out, lse, sink, cu_seqlens = ctx.saved_tensors
            if FA4_DSINK_TORCH:
                dq, dk, dv = torch.ops.fa4.fa4_varlen_bwd(
                    q, k, v, out, grad_output, lse, ctx.window_left,
                    cu_seqlens, ctx.max_seqlen)
                return (dq, dk, dv, None, _dsink_torch(sink, lse, out, grad_output),
                        None, None)
            dq, dk, dv, dsink = torch.ops.fa4.fa4_varlen_bwd_sink(
                q, k, v, out, grad_output, lse, ctx.window_left, sink,
                cu_seqlens, ctx.max_seqlen)
            return dq, dk, dv, None, dsink, None, None

        _fa4_varlen_sink_op.register_autograd(
            _fa4_varlen_backward, setup_context=_fa4_varlen_setup_context)

        def flash_attn_doc_local(q, k, v, window_size, learnable_sink, cu_seqlens):
            """q/k/v are (B, T, H, D) and contiguous; the varlen kernel wants the
            batch flattened, which is a view, not a copy."""
            B, T, H, D = q.shape
            wl = window_size[0] if isinstance(window_size, tuple) else window_size
            if wl is None or wl <= 0:
                wl = -1
            out, _lse = torch.ops.fa4.fa4_varlen_sink(
                q.reshape(-1, H, D), k.reshape(-1, H, D), v.reshape(-1, H, D),
                wl, learnable_sink, cu_seqlens, T)
            return out.view(B, T, H, D)

        def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), learnable_sink=None):
            wl = window_size[0] if isinstance(window_size, tuple) else window_size
            if wl is None or wl <= 0 or wl >= q.shape[1]:
                wl = -1
            if learnable_sink is None:
                out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
            else:
                out, _lse = torch.ops.fa4.fa4_causal_sink(q, k, v, wl, learnable_sink)
            return out

        print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
    else:
        from kernels import get_kernel
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
        flash_attn_doc_local = None
        print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")
except Exception as e:
    print(f"flash_attn unavailable ({e}), falling back to PyTorch SDPA")
    flash_attn_func = _make_sdpa_fallback()
    flash_attn_doc_local = None

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, get_token_bytes

# ---- fused softcap + cross-entropy, forward and backward in one pass -----------------
# The loss tail was two inductor kernels over the [131072, 8192] bf16 logits: a forward
# online-softmax (1.43 ms/step) and a backward that recomputes tanh/exp and writes the
# bf16 dlogits (1.33 ms/step). Both are SFU-bound, not bandwidth-bound: the online
# softmax pays two exps per element (rescale + term) on top of the libdevice tanh, and
# the backward pays tanh and exp again. This kernel reads each logit row once and
# writes dlogits once, and needs no running max: the softcap bounds every logit to
# (-15, 15), so sum_j e^{z_j} <= 8192 * e^15 = 2.7e10 is exactly representable in fp32
# and the max-subtraction is not needed for range. tanh(u) is 1 - 2/(e^{2u} + 1).
# Standalone on GB200: 1.02 ms against 2.76 ms for the two kernels it replaces,
# loss |diff| <= 6e-6, dlogits rel-Frobenius 2.1e-3 (= bf16 rounding of the grad).
#
# The gradient is written UNSCALED in the forward (as if dloss_tok = 1) and the actual
# per-token upstream gradient is applied after the lm_head GEMMs, where it is a row
# scale on the [N, d] side instead of an extra 2x2.1 GB pass over dlogits:
#     dX = diag(g) (G W),    dW = G^T (diag(g) X).
# The logits themselves are not saved: G replaces them, so memory is unchanged.
# Used only for training forwards (grad enabled, reduction='none', no lm_head bias);
# evaluation keeps the original F.cross_entropy path bit-for-bit.
USE_FUSED_CE = os.environ.get("USE_FUSED_CE", "1") == "1"
# Auxiliary next-next-token loss (training only, fused into the CE kernel below):
# per-token loss = CE(t+1) + w * CE(t+2) on the SAME softcapped logits, with
# w = MTP_W * max(0, 1 - progress / MTP_END). Training sees each token at most once,
# so extra supervision per token is the lever; it is annealed to zero before the
# warmdown tail so the head finishes as a pure next-token predictor.
MTP_W = float(os.environ.get("MTP_W", "0.3"))
MTP_END = float(os.environ.get("MTP_END", "0.65"))
# Uniform label smoothing on the t+1 CE, eps = LS_K * (current MTP weight): mild and
# early only, annealed to zero together with the t+2 aux.
LS_K = float(os.environ.get("LS_K", "0.3"))
# Never ask the t+2 aux to predict across a document boundary (t+1 target == BOS).
MTP_DOCMASK = int(os.environ.get("MTP_DOCMASK", "1"))
_mtp_w_t = torch.full((), MTP_W, device="cuda", dtype=torch.float32)
print(f"[mtp] MTP_W={MTP_W} MTP_END={MTP_END} MTP_DOCMASK={MTP_DOCMASK} LS_K={LS_K}")
_fused_ce_ok = False
if USE_FUSED_CE:
    try:
        import triton
        import triton.language as tl
        _fdiv = tl.extra.cuda.libdevice.fast_dividef

        @triton.jit
        def _add2(a0, a1, b0, b1):
            return a0 + b0, a1 + b1

        @triton.jit
        def _softcap_ce_kernel(logits_ptr, tgt_ptr, tgt2_ptr, w_ptr, loss_ptr, grad_ptr,
                               V: tl.constexpr, CAP: tl.constexpr, LS_K: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            cols = tl.arange(0, V)
            lp = logits_ptr + row * V
            gp = grad_ptr + row * V
            x = tl.load(lp + cols).to(tl.float32)
            th = 1.0 - _fdiv(2.0, tl.exp(x * (2.0 / CAP)) + 1.0)
            e = tl.exp(CAP * th)
            s, thsum = tl.reduce((e, th), 0, _add2)  # one pass: sum exp, sum tanh
            tgt = tl.load(tgt_ptr + row)
            valid = tgt != -1
            tgt_c = tl.where(valid, tgt, 0)
            # auxiliary next-next-token CE on the same softcapped logits, weight w
            # (read from a device scalar so the schedule never recompiles).
            tgt2 = tl.load(tgt2_ptr + row)
            valid2 = (tgt2 != -1) & valid
            w_raw = tl.load(w_ptr)
            w = tl.where(valid2, w_raw, 0.0)
            eps = tl.where(valid, LS_K * w_raw, 0.0)
            epsv = eps / V
            cw = 1.0 + w + eps
            tgt2_c = tl.where(valid2, tgt2, 0)
            inv = tl.where(valid, 1.0 / s, 0.0)
            g = (e * inv * cw - epsv) * (1.0 - th * th)
            tl.store(gp + cols, g.to(tl.bfloat16))
            # target columns: recompute their softcapped logits from scalar loads and
            # overwrite their gradients after CTA barriers (t2 first, then t1, so
            # t1 wins -- and carries both terms -- when t2 == t1).
            xt2 = tl.load(lp + tgt2_c).to(tl.float32)
            tht2 = 1.0 - _fdiv(2.0, tl.exp(xt2 * (2.0 / CAP)) + 1.0)
            zt2 = CAP * tht2
            gt2 = (tl.exp(zt2) * inv * cw - epsv - w) * (1.0 - tht2 * tht2)
            xt = tl.load(lp + tgt_c).to(tl.float32)
            tht = 1.0 - _fdiv(2.0, tl.exp(xt * (2.0 / CAP)) + 1.0)
            zt = CAP * tht
            same = (tgt2_c == tgt_c).to(tl.float32)
            gt = (tl.exp(zt) * inv * cw - epsv - valid.to(tl.float32) - w * same) * (1.0 - tht * tht)
            tl.debug_barrier()
            tl.store(gp + tgt2_c, gt2.to(tl.bfloat16))
            tl.debug_barrier()
            tl.store(gp + tgt_c, gt.to(tl.bfloat16))
            ls = tl.log(s)
            loss = (ls - zt) + w * (ls - zt2) + eps * (ls - thsum * (CAP / V))
            tl.store(loss_ptr + row, tl.where(valid, loss, 0.0))

        @torch.library.custom_op("sce::softcap_ce_fwd_bwd", mutates_args=())
        def _softcap_ce_op(logits: torch.Tensor, targets: torch.Tensor,
                           targets2: torch.Tensor, wt: torch.Tensor, cap: float
                           ) -> tuple[torch.Tensor, torch.Tensor]:
            N, V = logits.shape
            logits = logits.contiguous()
            targets = targets.contiguous()
            targets2 = targets2.contiguous()
            loss = torch.empty(N, device=logits.device, dtype=torch.float32)
            grad = torch.empty_like(logits)
            # 8 warps, not 16: measured 1.07 ms vs 1.49 ms per call at N=131072, V=8192
            # bf16 on GB200 (identical loss/grad), ~0.4 ms/step => ~20 more steps in 300 s.
            _softcap_ce_kernel[(N,)](logits, targets, targets2, wt, loss, grad,
                                     V=V, CAP=cap, LS_K=LS_K, num_warps=8)
            return loss, grad

        @_softcap_ce_op.register_fake
        def _softcap_ce_fake(logits, targets, targets2, wt, cap):
            N, V = logits.shape
            return (torch.empty(N, device=logits.device, dtype=torch.float32),
                    torch.empty_like(logits))

        class _LinearSoftcapCE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, targets, targets2, cap):
                logits = torch.mm(x, w.t())
                loss, g = torch.ops.sce.softcap_ce_fwd_bwd(logits, targets, targets2,
                                                           _mtp_w_t, cap)
                ctx.save_for_backward(x, w, g)
                return loss

            @staticmethod
            def backward(ctx, gl):
                x, w, g = ctx.saved_tensors
                gs = gl.unsqueeze(1)
                dx = (torch.mm(g, w) * gs).to(x.dtype)
                dw = torch.mm(g.t(), (x * gs).to(x.dtype))
                return dx, dw, None, None, None

        def fused_linear_softcap_ce(x, w, targets, targets2, cap):
            return _LinearSoftcapCE.apply(x, w, targets, targets2, cap)

        _fused_ce_ok = True
        print("[fused_ce] softcap+CE fwd/bwd Triton kernel ON")
    except Exception as _e:
        print(f"[fused_ce] unavailable ({type(_e).__name__}: {_e}); using F.cross_entropy")

TRIGRAM_TABLE_SIZE = int(os.environ.get("TRIGRAM_TABLE_SIZE", 27_000_000))
BIGRAM_TABLE_SIZE_ENV = int(os.environ.get("BIGRAM_TABLE_SIZE", 16777216))                            
BIGRAM_FACTORED = os.environ.get("BIGRAM_FACTORED", "1") == "1"                                             
BIGRAM_PRIME     = 31337
TRIGRAM_PRIME    = 1009

USE_AVALANCHE_TRIGRAM = os.environ.get("USE_AVALANCHE_TRIGRAM", "1") == "1"

USE_AVALANCHE_BIGRAM = os.environ.get("USE_AVALANCHE_BIGRAM", "0") == "1"

USE_INTRADOC_POS = os.environ.get("USE_INTRADOC_POS", "0") == "1"

USE_BIGRAM_INDUCTION = os.environ.get("USE_BIGRAM_INDUCTION", "0") == "1"
BIGRAM_INDUCTION_LR  = float(os.environ.get("BIGRAM_INDUCTION_LR", 0.3))                                  
INDUCTION_DOCMASK = os.environ.get("INDUCTION_DOCMASK", "0") == "1"                                                 
INDUCTION_ORDER = os.environ.get("INDUCTION_ORDER", "backoff").strip().lower()
USE_UNIGRAM_INDUCTION = os.environ.get("USE_UNIGRAM_INDUCTION", "0") == "1"
UNIGRAM_INDUCTION_LR = float(os.environ.get("UNIGRAM_INDUCTION_LR", 0.3))
UNIGRAM_INDUCTION_DOCMASK = os.environ.get("UNIGRAM_INDUCTION_DOCMASK", "1") == "1"
USE_INDUCTION_DIST = os.environ.get("USE_INDUCTION_DIST", "0") == "1"
NUM_INDUCTION_DIST_BUCKETS = int(os.environ.get("NUM_INDUCTION_DIST_BUCKETS", 32))
INDUCTION_DIST_LR = float(os.environ.get("INDUCTION_DIST_LR", 0.15))                                            
COMPILE_LOOKUPS = os.environ.get("COMPILE_LOOKUPS", "1") == "1"

USE_BIGRAM_RECALL = os.environ.get("USE_BIGRAM_RECALL", "0") == "1"
BIGRAM_RECALL_LR = float(os.environ.get("BIGRAM_RECALL_LR", 0.4))                                           

USE_LM_HEAD_BIAS   = os.environ.get("USE_LM_HEAD_BIAS", "0") == "1"
LM_HEAD_BIAS_INIT  = os.environ.get("LM_HEAD_BIAS_INIT", "1") == "1"
LM_HEAD_BIAS_LR    = float(os.environ.get("LM_HEAD_BIAS_LR", "0.03"))

USE_ATTN_SCALE = os.environ.get("USE_ATTN_SCALE", "0") == "1"
ATTN_SCALE_LR  = float(os.environ.get("ATTN_SCALE_LR", "0.02"))

USE_ATTN_SINK    = os.environ.get("USE_ATTN_SINK", "0") == "1"
ATTN_SINK_LAYERS = os.environ.get("ATTN_SINK_LAYERS", "all").strip().lower()
ATTN_SINK_INIT   = float(os.environ.get("ATTN_SINK_INIT", "0.0"))
# Depth ramp on the sink init: block i starts at ATTN_SINK_INIT + i*ATTN_SINK_INIT_SLOPE.
# The converged sink profile is strongly depth-ordered (mean|s|
# ~+0.5 at blocks 0-1 rising to ~+4.0 at blocks 6-8), so a single scalar init cannot
# put every block near its own operating point; the ramp can.
ATTN_SINK_INIT_SLOPE = float(os.environ.get("ATTN_SINK_INIT_SLOPE", "0.0"))
ATTN_SINK_LR     = float(os.environ.get("ATTN_SINK_LR", "0.01"))
# Floor under the sink group's warmdown multiplier, the way BIGRAM/TRIGRAM_LR_FLOOR
# already floors the sparse tables': lr = initial_lr * max(lrm, floor). 0.0 is exactly
# the plain warmdown schedule (max(lrm, 0) == lrm). A floor buys the sink LATE mobility
# without the early over-movement that a larger base LR brings.
ATTN_SINK_LRM_FLOOR = float(os.environ.get("ATTN_SINK_LRM_FLOOR", "0.0"))

# x0_lambdas (the token-embedding skip mix read by every block) live in their own
# AdamW group whose LR is the bare nanochat scalar default, 0.5. At that step size a
# scalar whose operating range is ~[1.5, 2.1] moves by up to 0.5 EVERY step while lrm
# is still 1.0, so the group is LR-saturated rather than converging: traced in
# training it leaves 0.05, is already +1.22 by step 3, overshoots to +3.1/+3.3 near
# step 900 and then decays back for the rest of the run. These two knobs allow a
# different step size and start; a sweep found no gain, so both stay at their defaults.
X0_LR   = float(os.environ.get("X0_LR", "0.5"))
X0_INIT = float(os.environ.get("X0_INIT", "0.05"))

USE_DOC_LOCAL    = os.environ.get("USE_DOC_LOCAL", "0") == "1"
DOC_LOCAL_LAYERS = os.environ.get("DOC_LOCAL_LAYERS", "long").strip().lower()
DOC_LOCAL_CAP    = int(os.environ.get("DOC_LOCAL_CAP", "256"))
DOC_LOCAL_FULL_DOC = os.environ.get("DOC_LOCAL_FULL_DOC", "0") == "1"

USE_LAYER_READOUT = os.environ.get("USE_LAYER_READOUT", "0") == "1"
LAYER_READOUT_LR  = float(os.environ.get("LAYER_READOUT_LR", "0.01"))

USE_READOUT_NORM  = os.environ.get("USE_READOUT_NORM", "0") == "1"

USE_DENSE_DWA = os.environ.get("USE_DENSE_DWA", "0") == "1"
DENSE_DWA_LR  = float(os.environ.get("DENSE_DWA_LR", "0.01"))
DENSE_DWA_K   = int(os.environ.get("DENSE_DWA_K", "8"))                                      

# Layers that receive the bigram offset as an additive, gated contribution to
# attention V (alongside the residual-stream injection every layer already sees).
# has_ve() gives learned value embeddings to even layers only, and the trigram
# offset goes to layers 6,7,8, so 1/3/5/7 are the layers with no value path at
# all; routing the already-computed bigram_offset there costs 4 * 32 * n_kv_head
# gate parameters and one extra read-modify-write of V per enabled layer.
BIGRAM_VE_LAYERS = tuple(
    int(x) for x in os.environ.get("BIGRAM_VE_LAYERS", "1 3 5 7").replace(",", " ").split()
)

LONG_LAYERS = tuple(
    int(x) for x in os.environ.get("LONG_LAYERS", "").replace(",", " ").split()
)


ATTN_OUT_GATE = os.environ.get("ATTN_OUT_GATE", "1") == "1"

# One-position key offset on the long (doc-local) layers: for the low-frequency RoPE
# pairs p >= KEY_OFFSET_LO (of head_dim/2), each key carries the PREVIOUS token's
# raw key channels, so one query can match "the token after X" in a single layer
# (one-layer induction). Applied to the raw projection output before RoPE/QK-norm so
# inductor folds the shifted load into the existing rope+rms_norm kernel. At base
# 200000 these pairs rotate < 1.1e-4 rad per position, so the one-position RoPE
# mismatch is negligible. KEY_OFFSET_LO >= head_dim/2 disables it.
# Applied before RoPE the shift costs ~nothing in step time (a post-norm version cost
# ~1.2% of steps).
KEY_OFFSET_LO = int(os.environ.get("KEY_OFFSET_LO", "48"))
KEY_OFFSET_LAYERS = os.environ.get("KEY_OFFSET_LAYERS", "long")

# Learnable per-channel causal key mix ("Canon-lite" depthwise causal conv on K only):
# k_new = a0*k + a1*k_{t-1} (+ a2*k_{t-2}, ...), applied to the raw keys before RoPE and
# QK-norm; a is fp32 [taps, head_dim/2] per layer, shared across heads and across the
# two members of each RoPE pair. Initialised to exactly the hard KEY_OFFSET switch
# above (a1 = 1, a0 = 0 on pairs >= KEY_OFFSET_LO of offset layers; a0 = 1 elsewhere),
# so step 0 is the KEY_MIX=0 model (identical step-0 loss). KEY_MIX=0 is the
# hard-switch graph, unchanged. Layers without a mix keep the hard switch.
# Enabled on the long layers only, 2 taps, lr 0.01 (the defaults). Its gain over the
# hard switch is within seed-to-seed noise; on all 9 layers, or with 3 taps, the
# grad(a) column reduction costs 1-4% of steps and eats the per-step gain. The learned
# end state on the long layers is a partial previous-token blend no hard mask
# expresses. KEY_MIX=fixed with KEY_MIX_TABLE_FILE (lines "layer v...", tap-major)
# freezes a learned table instead; that was no better.
KEY_MIX_MODE = os.environ.get("KEY_MIX", "1").strip().lower()
KEY_MIX = KEY_MIX_MODE in ("1", "fixed")
KEY_MIX_FIXED = KEY_MIX_MODE == "fixed"
KEY_MIX_TAPS = int(os.environ.get("KEY_MIX_TAPS", "2"))
KEY_MIX_LAYERS = os.environ.get("KEY_MIX_LAYERS", "long")
KEY_MIX_LR = float(os.environ.get("KEY_MIX_LR", "0.01"))
KEY_MIX_SUB = int(os.environ.get("KEY_MIX_SUB", "4"))
# Learnable 2-tap causal QUERY mix on the long layers (same mechanism as the key mix):
# q_new = a0*q_t + a1*q_{t-1}, applied to the raw queries before RoPE and QK-norm, a
# fp32 [2, head_dim/2] per layer shared across heads and RoPE pair members. Init a0=1,
# a1=0, so step 0 is the QMIX=0 model; QMIX=0 builds no parameter and leaves the
# graph unchanged. Its measured gain is small (seed-0 screens, ~0.25% step cost);
# value mix, short/all-layer query mix, 3 taps and a higher LR were all worse.
QMIX = os.environ.get("QMIX", "1").strip() == "1"
QMIX_LAYERS = os.environ.get("QMIX_LAYERS", "long")
QMIX_LR = float(os.environ.get("QMIX_LR", "0.01"))
# grad(a) row subsampling as KEY_MIX_SUB; 8 (16K tokens/coef) was no worse than 4
# and costs fewer steps.
QMIX_SUB = int(os.environ.get("QMIX_SUB", "8"))
_KEY_MIX_TABLE = {}
if KEY_MIX_FIXED:
    with open(os.environ["KEY_MIX_TABLE_FILE"]) as _f:
        for _ln in _f.read().strip().splitlines():
            _parts = _ln.split()
            _KEY_MIX_TABLE[int(_parts[0])] = [float(v) for v in _parts[1:]]
    KEY_MIX_TAPS = len(next(iter(_KEY_MIX_TABLE.values()))) // 64


class _KeyMixFn(torch.autograd.Function):
    """k_mix[t] = sum_j a[j] * k[t-j] (zero before t=0), a: [taps, H, Dh] fp32.

    Written as a Function so the backward needs only the raw k: the grad of a[j] is
    computed as sum_t g[t+j] * k[t] from the same back-shifted g that dk already
    needs. Plain autograd instead saved one padded (shifted) copy of k per tap per
    layer -- +1.8 GB at 2 taps, an OOM at 3 -- and ran ~4% slower."""
    @staticmethod
    def forward(ctx, k, a, sub=None):
        ctx.sub = KEY_MIX_SUB if sub is None else sub
        ctx.save_for_backward(k, a)
        ab = a.to(k.dtype)
        out = ab[0] * k
        for j in range(1, a.size(0)):
            out = out + ab[j] * F.pad(k[:, :-j], (0, 0, 0, 0, j, 0))
        return out

    @staticmethod
    def backward(ctx, g):
        k, a = ctx.saved_tensors
        ab = a.to(g.dtype)
        dk = ab[0] * g
        for j in range(1, a.size(0)):
            dk = dk + ab[j] * F.pad(g[:, j:], (0, 0, 0, 0, 0, j))
        # grad(a): a column reduction over the first B/KEY_MIX_SUB rows of the batch
        # only (scaled back up; an unbiased estimate from >= 32K tokens x 12 lanes per
        # coefficient), on the flattened [rows*T, H*Dh] view so each tap is a plain
        # contiguous slice pair. The few cross-row pairs the flat shift admits
        # (j per row boundary out of rows*T) are negligible.
        rows = max(1, g.size(0) // ctx.sub)
        scale = g.size(0) / rows
        gf = g[:rows].reshape(-1, a.size(1) * a.size(2)).float()
        kf = k[:rows].reshape(-1, a.size(1) * a.size(2)).float()
        da = [(gf * kf).sum(0)]
        for j in range(1, a.size(0)):
            da.append((gf[j:] * kf[:-j]).sum(0))
        da = torch.stack(da).view_as(a) * scale
        return dk, da.to(a.dtype), None


def _long_layer_indices(config):
    """The same selection GPT._compute_window_sizes makes, available at block
    construction time (it needs only n_layer / window_pattern / LONG_LAYERS)."""
    if LONG_LAYERS:
        return set(LONG_LAYERS)
    pattern = config.window_pattern.upper()
    idx = {i for i in range(config.n_layer) if pattern[i % len(pattern)] == "L"}
    idx.add(config.n_layer - 1)
    return idx


def _layer_spec_indices(spec, config):
    if spec in ("all", "", "*"):
        return set(range(config.n_layer))
    if spec == "long":
        return _long_layer_indices(config)
    if spec == "short":
        return set(range(config.n_layer)) - _long_layer_indices(config)
    if spec == "none":
        return set()
    return {int(x) for x in spec.replace(",", " ").split()}


def _attn_sink_layer_indices(config):
    """Which blocks get a learnable sink. 'long' is where cross-document
    contamination is worst (42.4% of attended keys at window 1024 vs 10.8% at
    window 256); 'short' is the cheap arm, since the sink's only real cost is in
    the attention backward and the six short layers own less of it."""
    if not USE_ATTN_SINK:
        return set()
    return _layer_spec_indices(ATTN_SINK_LAYERS, config)


def _doc_local_layer_indices(config):
    """Which blocks attend document-locally. The varlen kernel path requires a
    sink (it is the same op), so this is intersected with the sink layers."""
    if not USE_DOC_LOCAL or flash_attn_doc_local is None:
        return set()
    return _layer_spec_indices(DOC_LOCAL_LAYERS, config) & _attn_sink_layer_indices(config)

NUM_POS_BUCKETS  = 32                                             

NGRAM_VE_LAYERS  = (6, 7, 8)

# --- Form of the sparse n-gram -> trunk coupling -------------------------------
# Without these knobs the coupling is the identity map used twice: `x = x + trigram_out`
# on the embedding stream, and the *same* vector handed to every layer in NGRAM_VE_LAYERS
# as attention V (modulated only by that layer's per-head sigmoid gate). The trigram
# table is data-starved (490M training tokens over ~70M touched rows is single-digit
# updates per row) while a dense tensor on the same path sees every token, so the
# cheapest place to buy quality here is the *map*, not the rows.
#
# What is bought: ONE parameter, ng_scale, of shape [1 + n_sites, d], all ones at
# init, so the run starts from exactly the identity coupling's arithmetic.
#   row 0      -- per-channel gate on the residual injection, x = x + g0 * trigram.
#   row 1 + k  -- per-channel scale on site k's value injection, applied inside the
#                 attention epilogue as v = v + (head_gate * scale) * ng_v, i.e.
#                 folded into the multiply that is already there. Viewed as
#                 [n_kv_head, head_dim] it is also a per-head scale, so each of
#                 blocks 6,7,8 can pick a different *shape* of the same row instead
#                 of receiving three verbatim copies of it; the existing 2*sigmoid
#                 gate can only rescale all channels of a head together.
# Packing the four vectors into ONE tensor is not cosmetic: the AdamW group loop is
# per-parameter Python calling a fullgraph-compiled step, and with a per-step
# loss.item() sync that dispatch sits on the critical path. Four separate [768] and
# [6, 128] parameters measured 287.7 s over 3500 matched steps; the single packed
# tensor, with element-for-element identical arithmetic, measured 285.5 s against a
# 284.3 s without the knobs -- i.e. +0.4% of the step instead of +1.2%.
NGRAM_RES_GATE = os.environ.get("NGRAM_RES_GATE", "0") == "1"
NGRAM_V_SCALE = os.environ.get("NGRAM_V_SCALE", "0") == "1"
NGRAM_COUPLING_LR = float(os.environ.get("NGRAM_COUPLING_LR", "0.005"))

def _fmix32(h):
    """MurmurHash3 fmix32 avalanche finalizer on an int64 tensor holding a 32-bit
    value. Every input bit is diffused across all 32 output bits. int64 mul wraps
    mod 2^64; masking to 32 bits after each mul extracts the correct low bits of
    the (possibly overflowing signed-int64) 32x32 product - verified 0 mismatches
    vs a pure-Python reference. Deterministic, near-zero FLOP."""
    h = h & 0xFFFFFFFF
    h = h ^ (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h

def compute_trigram_keys(idx, table_size):
    """Causal trigram hash key (prev2, prev1, cur) -> [0, table_size).
    USE_AVALANCHE_TRIGRAM=True: pack (prev1,cur) into 26 bits, fmix32-diffuse,
    fold prev2 (scaled by the golden-ratio constant 0x9e3779b1) and fmix32 again,
    so distinct trigrams spread ~uniformly across the table (a free effective-
    capacity increase). Else the linear structured form.
    BOTH are CAUSAL - use only prev tokens {j-2,j-1,j} <= j, never a future token -
    and run in the no_grad lookup path. Deterministic; int64 mul wraps mod 2^64 but
    each pass masks to 32 bits so the index is valid in [0, table_size). Handles
    power-of-2 (bit-mask) and non-power-of-2 (modulo) table sizes identically."""
    B, T = idx.shape
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1).long()
    prev2 = torch.cat([idx.new_zeros(B, 2), idx[:, :-2]], dim=1).long()
    if USE_AVALANCHE_TRIGRAM:
        cur = idx.long()
        h = _fmix32(((prev1 << 13) | cur) & 0xFFFFFFFF)                                          
        h = _fmix32(h ^ ((prev2 * 0x9e3779b1) & 0xFFFFFFFF))                                
    else:
        h2 = prev1 * BIGRAM_PRIME + idx.long()
        h = prev2 * TRIGRAM_PRIME + h2

    if (table_size & (table_size - 1)) == 0:
        return h & (table_size - 1)
    return h % table_size

def compute_bigram_keys(idx, table_size):
    """Causal bigram hash key (prev1, cur) -> [0, table_size).
    USE_AVALANCHE_BIGRAM=True: pack (prev1,cur) into 26 bits then fmix32-diffuse,
    mirroring the trigram avalanche, so distinct bigrams spread ~uniformly
    across the table (fewer structured collisions -> more effective capacity).
    Else the linear form (prev1*BIGRAM_PRIME + cur) - byte-identical to
    the original when the flag is off. BOTH are CAUSAL (use only {prev1, cur} <= t)
    and run in the no_grad lookup path. h >= 0, so power-of-2 mask and modulo agree."""
    B, T = idx.shape
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1).long()
    cur = idx.long()
    if USE_AVALANCHE_BIGRAM:
        h = _fmix32(((prev1 << 13) | cur) & 0xFFFFFFFF)                                     
    else:
        h = prev1 * BIGRAM_PRIME + cur
    if (table_size & (table_size - 1)) == 0:
        return h & (table_size - 1)
    return h % table_size

def compute_intradoc_pos_buckets(idx, num_buckets, seq_len, bos_id):
    """Causal intra-document position bucket: for each t, the distance since the
    most recent BOS at or before t, coarsened into num_buckets over seq_len.
    Row position 0 is always BOS (BOS-aligned dataloader), so last_bos is always
    defined. Uses cummax over j<=t -> depends ONLY on tokens {<=t}, never a future
    token. Computed OUTSIDE the compiled model (no_grad), so no compile/graph risk."""
    B, T = idx.shape
    arange = torch.arange(T, device=idx.device, dtype=torch.long)               
    is_bos = (idx == bos_id)                                                      
    bos_pos = torch.where(is_bos, arange.unsqueeze(0).expand(B, T),
                          torch.zeros_like(idx))                                                  
    last_bos = torch.cummax(bos_pos, dim=1).values                                                    
    intra = arange.unsqueeze(0) - last_bos                                             
    buckets = (intra * num_buckets) // seq_len
    return buckets.clamp_(0, num_buckets - 1)

def compute_last_bos(idx, bos_id):
    """[B,T] long: for each t, the position of the most recent BOS at or before t
    (cummax over j<=t, strictly causal). Row pos 0 is always BOS."""
    B, T = idx.shape
    arange = torch.arange(T, device=idx.device, dtype=torch.long)
    is_bos = (idx == bos_id)
    bos_pos = torch.where(is_bos, arange.unsqueeze(0).expand(B, T), torch.zeros_like(idx))
    return torch.cummax(bos_pos, dim=1).values

def compute_bigram_induction_next(idx, vocab_size, last_bos=None):
    """Causal SECOND-ORDER induction. For each (b,t), find the most recent previous
    position j<t whose BIGRAM CONTEXT (idx[j-1], idx[j]) equals the current context
    (idx[t-1], idx[t]), and return idx[j+1] - "the token that followed the last time
    this exact 2-token context appeared". A strictly higher-precision copy signal than
    single-token induction. First occurrences of a bigram context return `vocab_size`
    (a dedicated 'novel' row). Strictly causal: j<t so j+1<=t (never a future token),
    and the match key at t reads only idx[t-1],idx[t] (both <= t). Vectorized per row
    via ONE stable argsort on the context id (positions sorted by (prev1,cur) id, ties
    ascending by position -> consecutive same-context sorted neighbours are consecutive
    occurrences), then scatter_ back. Runs in the no_grad lookup path, OUTSIDE the
    compiled graph. Returns long [B,T] in [0, vocab_size].
    If last_bos is given (doc-mask), a match is accepted only when j >= last_bos[t]
    (the prior occurrence lies within the current document) - copies never cross
    packed-doc boundaries; still strictly causal (last_bos is a cummax over <=t)."""
    B, T = idx.shape
    idx = idx.long()
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1)                            
    ctx = prev1 * vocab_size + idx                                                             
    order = torch.argsort(ctx, dim=1, stable=True)                                                 
    sorted_ctx = torch.gather(ctx, 1, order)
    same = torch.zeros_like(idx, dtype=torch.bool)
    same[:, 1:] = sorted_ctx[:, 1:] == sorted_ctx[:, :-1]                                                 
    prev_pos_sorted = torch.zeros_like(idx)
    prev_pos_sorted[:, 1:] = order[:, :-1]                                                                    
    neg = torch.full_like(idx, -1)
    prev_occ = torch.full_like(idx, -1)
    prev_occ.scatter_(1, order, torch.where(same, prev_pos_sorted, neg))                              
    has_prev = prev_occ >= 0
    if last_bos is not None:
        has_prev = has_prev & (prev_occ >= last_bos)                                          
    next_idx = (prev_occ + 1).clamp_(0, T - 1)                                         
    nxt = torch.gather(idx, 1, next_idx)
    return torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))

def _prev_occ_of_ctx(idx, ctx, last_bos=None):
    """[B,T] long: for each position t, the largest j<t whose context id ctx[j]==ctx[t]
    (the most-recent prior occurrence of the SAME context), else -1. One stable argsort:
    positions with equal ctx sort into ascending-position runs, so a run element's left
    neighbour in sorted order is its most-recent earlier occurrence. Strictly uses only
    ordering by position (j<t). If last_bos is given, occurrences before the current
    document (j<last_bos[t]) are rejected to -1 (so a later backoff can still fire)."""
    B, T = idx.shape
    order = torch.argsort(ctx, dim=1, stable=True)
    sorted_ctx = torch.gather(ctx, 1, order)
    same = torch.zeros_like(idx, dtype=torch.bool)
    same[:, 1:] = sorted_ctx[:, 1:] == sorted_ctx[:, :-1]
    prev_pos_sorted = torch.zeros_like(idx)
    prev_pos_sorted[:, 1:] = order[:, :-1]
    neg = torch.full_like(idx, -1)
    prev_occ = torch.full_like(idx, -1)
    prev_occ.scatter_(1, order, torch.where(same, prev_pos_sorted, neg))
    if last_bos is not None:
        prev_occ = torch.where(prev_occ >= last_bos, prev_occ, neg)                         
    return prev_occ

def compute_induction_next(idx, vocab_size, last_bos=None, order="backoff", return_prev_occ=False, return_bigram_next=False):
    """Causal induction with selectable match ORDER (see INDUCTION_ORDER note).
    order == 'trigram': match the 3-token context (idx[t-2],idx[t-1],idx[t]).
    order == 'backoff': prefer the 3-token match; where it is absent, fall back to the
                        2-token match (idx[t-1],idx[t]) - same recall as 'bigram', higher
                        precision. Returns idx[j+1] of the chosen match, else 'novel'
                        (index=vocab_size). Strictly causal: j<t => j+1<=t; the trigram
                        key reads only idx[t-2..t] (all <=t). Padding at t<2 mirrors the
                        prev1/prev2 zero-pad, so t=0,1 can only match same-pad
                        positions (a no-op in practice). Runs in the no_grad lookup path.
    return_prev_occ=True also returns the matched position j (or -1 where 'novel'),
    the byproduct the induction-match-distance feature buckets - costs NO extra argsort."""
    B, T = idx.shape
    idx = idx.long()
    if order == "unigram":

        prev_occ = _prev_occ_of_ctx(idx, idx, last_bos)
        has_prev = prev_occ >= 0
        next_idx = (prev_occ + 1).clamp_(0, T - 1)
        nxt = torch.gather(idx, 1, next_idx)
        result = torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))
        if return_prev_occ:
            return result, torch.where(has_prev, prev_occ, torch.full_like(idx, -1))
        return result
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1)             
    prev2 = torch.cat([idx.new_zeros(B, 2), idx[:, :-2]], dim=1)             
    ctx3 = (prev2 * vocab_size + prev1) * vocab_size + idx                                           
    prev_occ = _prev_occ_of_ctx(idx, ctx3, last_bos)
    po2 = None
    if order == "backoff":
        ctx2 = prev1 * vocab_size + idx                                              
        po2 = _prev_occ_of_ctx(idx, ctx2, last_bos)
        prev_occ = torch.where(prev_occ >= 0, prev_occ, po2)                                      
    has_prev = prev_occ >= 0
    next_idx = (prev_occ + 1).clamp_(0, T - 1)                                         
    nxt = torch.gather(idx, 1, next_idx)
    result = torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))
    outs = [result]
    if return_prev_occ:
        outs.append(torch.where(has_prev, prev_occ, torch.full_like(idx, -1)))
    if return_bigram_next:

        if po2 is None:
            ctx2 = prev1 * vocab_size + idx
            po2 = _prev_occ_of_ctx(idx, ctx2, last_bos)
        has_b = po2 >= 0
        b_next = torch.gather(idx, 1, (po2 + 1).clamp_(0, T - 1))
        outs.append(torch.where(has_b, b_next, torch.full_like(idx, vocab_size)))
    return tuple(outs) if len(outs) > 1 else outs[0]

def compute_induction_dist_buckets(prev_occ, num_buckets):
    """Log-bucket the induction MATCH DISTANCE d = t - prev_occ into [0, num_buckets-1];
    positions with no in-doc match (prev_occ < 0) map to the dedicated row `num_buckets`.
    prev_occ is the induction head's ALREADY-COMPUTED (docmasked, causal) match position,
    so this adds NO argsort. Strictly causal (prev_occ < t => d >= 1). Returns long [B,T]
    in [0, num_buckets]. Runs in the no_grad lookup path (inside the compiled graph)."""
    B, T = prev_occ.shape
    has_prev = prev_occ >= 0
    arange = torch.arange(T, device=prev_occ.device, dtype=torch.long)
    dist = (arange.unsqueeze(0) - prev_occ).clamp_(min=1)                                    
    logd = torch.log(dist.float()) / math.log(float(T))                         
    buckets = (logd * num_buckets).long().clamp_(0, num_buckets - 1)
    return torch.where(has_prev, buckets, torch.full_like(prev_occ, num_buckets))

def _lookups_impl(idx):
    """The exact per-step no_grad lookup block, in one function so it can
    be compiled as a single graph. Byte-identical to the eager per-function path
    (MAX_SEQ_LEN == config.sequence_len)."""
    bigram_keys = compute_bigram_keys(idx, BIGRAM_TABLE_SIZE)
    trigram_keys = compute_trigram_keys(idx, TRIGRAM_TABLE_SIZE)
    pos_bucket_ids = (compute_intradoc_pos_buckets(idx, NUM_POS_BUCKETS, MAX_SEQ_LEN, BOS_TOKEN_ID)
                      if USE_INTRADOC_POS else None)
    induction_dist_ids = None
    recall_next_tok = None
    if USE_BIGRAM_INDUCTION:
        lb = compute_last_bos(idx, BOS_TOKEN_ID) if INDUCTION_DOCMASK else None
        if INDUCTION_ORDER == "bigram":
            bind_next_tok = compute_bigram_induction_next(idx, vocab_size, lb)
        elif USE_INDUCTION_DIST:
            if USE_BIGRAM_RECALL:
                bind_next_tok, bind_prev_occ, recall_next_tok = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True, return_bigram_next=True)
            else:
                bind_next_tok, bind_prev_occ = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True)
            induction_dist_ids = compute_induction_dist_buckets(bind_prev_occ, NUM_INDUCTION_DIST_BUCKETS)
        else:
            if USE_BIGRAM_RECALL:
                bind_next_tok, recall_next_tok = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_bigram_next=True)
            else:
                bind_next_tok = compute_induction_next(idx, vocab_size, lb, INDUCTION_ORDER)
    else:
        bind_next_tok = None
    if USE_UNIGRAM_INDUCTION:
        lb_uni = compute_last_bos(idx, BOS_TOKEN_ID) if UNIGRAM_INDUCTION_DOCMASK else None
        uni_next_tok = compute_induction_next(idx, vocab_size, lb_uni, "unigram")
    else:
        uni_next_tok = None
    return bigram_keys, trigram_keys, pos_bucket_ids, bind_next_tok, uni_next_tok, induction_dist_ids, recall_next_tok

_lookups_compiled = torch.compile(_lookups_impl, dynamic=False)
_lookups_enabled = COMPILE_LOOKUPS

def get_lookups(idx):
    """Fused per-step lookups: the compiled graph when enabled, with a permanent
    eager fallback on any error so a compile failure can never crash the run."""
    global _lookups_enabled
    if _lookups_enabled:
        try:
            return _lookups_compiled(idx)
        except Exception as e:                    
            print(f"[fusion] compiled lookups failed ({type(e).__name__}: {e}); "
                  f"falling back to eager for the rest of the run", flush=True)
            _lookups_enabled = False
    return _lookups_impl(idx)

def compute_unigram_logfreq(targets, vocab_size, floor=1e-6):
    """Centered log-unigram-frequency prior [vocab] (fp32) from target tokens.
    Counts token occurrences in `targets` (ignoring ignore_index=-1), converts to
    a smoothed log-probability, and centers it (softmax is shift-invariant, so
    centering leaves the induced distribution unchanged but keeps the bias values
    modest & symmetric). Used ONLY to initialize the lm_head output bias from the
    first TRAINING batch - a better starting point than zeros, not a val leak."""
    flat = targets.reshape(-1)
    flat = flat[flat >= 0]                                                          
    counts = torch.bincount(flat, minlength=vocab_size).float()
    total = counts.sum().clamp(min=1.0)
    logf = (counts / total + floor).log()
    return logf - logf.mean()

class GPTWithBigram(nn.Module):
    """Evaluation wrapper: combines compiled GPT + sparse bigram + sparse trigram
    for evaluate_bpb."""
    def __init__(self, compiled_gpt, bigram_embed, bigram_prime, bigram_table_size,
                 trigram_embed, trigram_table_size):
        super().__init__()
        self._gpt = compiled_gpt
        self._bigram_embed = bigram_embed
        self._bigram_prime = bigram_prime
        self._bigram_table_size = bigram_table_size
        self._trigram_embed = trigram_embed
        self._trigram_table_size = trigram_table_size

    def forward(self, idx, targets=None, reduction='mean'):
        keys = compute_bigram_keys(idx, self._bigram_table_size)
        bigram_out = self._bigram_embed(keys)             
        trigram_keys = compute_trigram_keys(idx, self._trigram_table_size)
        trigram_out = self._trigram_embed(trigram_keys)             
        pos_bucket_ids = (compute_intradoc_pos_buckets(idx, NUM_POS_BUCKETS, MAX_SEQ_LEN, BOS_TOKEN_ID)
                          if USE_INTRADOC_POS else None)

        induction_dist_ids = None
        recall_next_tok = None
        if USE_BIGRAM_INDUCTION:
            lb = compute_last_bos(idx, BOS_TOKEN_ID) if INDUCTION_DOCMASK else None
            if INDUCTION_ORDER == "bigram":
                bind_next_tok = compute_bigram_induction_next(idx, vocab_size, lb)
            elif USE_INDUCTION_DIST:
                if USE_BIGRAM_RECALL:
                    bind_next_tok, bind_prev_occ, recall_next_tok = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True, return_bigram_next=True)
                else:
                    bind_next_tok, bind_prev_occ = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True)
                induction_dist_ids = compute_induction_dist_buckets(bind_prev_occ, NUM_INDUCTION_DIST_BUCKETS)
            else:
                if USE_BIGRAM_RECALL:
                    bind_next_tok, recall_next_tok = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_bigram_next=True)
                else:
                    bind_next_tok = compute_induction_next(idx, vocab_size, lb, INDUCTION_ORDER)
        else:
            bind_next_tok = None
        if USE_UNIGRAM_INDUCTION:
            lb_uni = compute_last_bos(idx, BOS_TOKEN_ID) if UNIGRAM_INDUCTION_DOCMASK else None
            uni_next_tok = compute_induction_next(idx, vocab_size, lb_uni, "unigram")
        else:
            uni_next_tok = None
        return self._gpt(idx, targets=targets, reduction=reduction,
                         bigram_offset=bigram_out, trigram_offset=trigram_out,
                         pos_bucket_ids=pos_bucket_ids, bind_next_tok=bind_next_tok,
                         uni_next_tok=uni_next_tok, induction_dist_ids=induction_dist_ids,
                         recall_next_tok=recall_next_tok)

# ---------------------------------------------------------------------------
# Eval-time corpus-memory mixture (probability space).
#
#   p_mix(y_t) = w_net p_net(y_t) + w_ng p_ng(y_t) + w_doc p_doc(y_t)
#
#   p_ng  : Witten-Bell interpolated n-gram over the training split (whole documents
#           minus the gate slice, see _BgBuild), orders MIX_ORDERS, contexts never
#           cross a BOS.
#   p_doc : retrieved-document cache. At query points inside each val document the
#           document's own past 8-token shingles vote (IDF-weighted) for training
#           documents; the top MIX_DOC_M of them are concatenated into a bounded
#           buffer and a Witten-Bell n-gram over that buffer (orders DOC_ORDERS,
#           order 0 = the buffer's unigram) backs off to p_ng.
#   w     : softmax gate over target-independent causal features (n-gram match
#           order, doc match order x vote strength, in-document position, running
#           in-document reliability of each expert vs the net), fitted on a held-out
#           slice of TRAINING batches that were never trained on and never entered a
#           table. Every expert is normalised, so the mixture is too.
# ---------------------------------------------------------------------------
MIX_ENABLE = os.environ.get("MIX_ENABLE", "1") == "1"
MIX_USE_DOC = os.environ.get("MIX_USE_DOC", "1") == "1"
MIX_ORDERS = [int(v) for v in os.environ.get("MIX_ORDERS", "1,2,3,4,5,6,7,8,10,12,14,16,20,24,32").split(",")]
DOC_ORDERS = [int(v) for v in os.environ.get("MIX_DOC_ORDERS", "0,1,2,3,4,5,6,8,10,12,16").split(",")]
MIX_SHINGLE_K = int(os.environ.get("MIX_SHINGLE_K", "8"))
MIX_SHINGLE_S = int(os.environ.get("MIX_SHINGLE_S", "2"))       # keep a shingle iff hash % S == 0 (power of 2)
MIX_DF_MAX = int(os.environ.get("MIX_DF_MAX", "64"))           # shingles in more docs than this do not vote
MIX_DOC_M = int(os.environ.get("MIX_DOC_M", "4"))
MIX_DOC_L = int(os.environ.get("MIX_DOC_L", "8192"))
MIX_QOFF = [int(v) for v in os.environ.get("MIX_QOFF", "16,32,64,128,256,512,1024,1536").split(",")]
MIX_GPU_GB = float(os.environ.get("MIX_GPU_GB", "16"))         # n-gram tables kept on GPU up to this; rest on pinned host
MIX_NSLICE = int(os.environ.get("MIX_NSLICE", "16"))            # held-out training batches for the gate fit
MIX_FIT_ITERS = int(os.environ.get("MIX_FIT_ITERS", "300"))
MIX_FIT_LR = float(os.environ.get("MIX_FIT_LR", "0.1"))
MIX_FIT_TOK = int(os.environ.get("MIX_FIT_TOK", "0"))          # random subset of slice tokens for the fit (0 = all)
MIX_FB_DECAY = float(os.environ.get("MIX_FB_DECAY", "0.8"))
MIX_FB_DECAY2 = float(os.environ.get("MIX_FB_DECAY2", "0.97"))     # optional second (slow) feedback EMA
# Network predictive entropy as a gate feature (needs the logits, so the eval wrapper
# calls the network with targets=None and forms the cross-entropy itself).
MIX_ENT = os.environ.get("MIX_ENT", "1") == "1"
MIX_CONF = os.environ.get("MIX_CONF", "1") == "1"
MIX_CONF_X = os.environ.get("MIX_CONF_X", "0") == "1"   # top n-gram context count / diversity as a gate feature
MIX_DIAG = os.environ.get("MIX_DIAG", "1") == "1"       # post-score A/B + validity checks (never feed val_bpb)
_ENT_EDGES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# Doc-side gate features (target-independent): names from _DOCFEAT_SIZES.
MIX_DOCFEAT = [v for v in os.environ.get("MIX_DOCFEAT", "").split(",") if v]
MIX_TOPK = 16
_T1_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]     # net top-1 prob -> 11
_DS_EDGES = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]                  # doc share -> 1..8 (0 = none)
_NT1, _NDS, _NRB = len(_T1_EDGES) + 1, len(_DS_EDGES) + 2, 7
_DOCFEAT_SIZES = {
    "ds_t1": _NDS * _NT1,            # doc max-continuation share x net top-1 prob
    "ds_rb": _NDS * _NRB,            # doc share x log2 rank of the doc argmax in net top-16
    "rb_t1": _NRB * _NT1,            # rank x net top-1 prob
    "vb_ds_ag": 9 * _NDS * 2,        # vote bucket x doc share x (doc argmax == net argmax)
    "dm_rb": (len(DOC_ORDERS) + 1) * _NRB,   # doc match order x rank
    "dc_ds": 8 * _NDS,               # log2 c(doc ctx) x doc share
}
assert all(k in _DOCFEAT_SIZES for k in MIX_DOCFEAT), MIX_DOCFEAT
assert all(k in MIX_ORDERS for k in DOC_ORDERS if k > 0)
assert MIX_SHINGLE_K in MIX_ORDERS

_M50 = (1 << 50) - 1
_M35 = (1 << 35) - 1
_M42 = (1 << 42) - 1
_M21 = (1 << 21) - 1


def _mix64(h):
    """murmur3 fmix64 on int64 (logical shifts emulated by masking)."""
    h = h ^ ((h >> 33) & ((1 << 31) - 1))
    h = h * -49064778989728563          # 0xff51afd7ed558ccd
    h = h ^ ((h >> 33) & ((1 << 31) - 1))
    h = h * -4265267296055464877        # 0xc4ceb9fe1a85ec53
    h = h ^ ((h >> 33) & ((1 << 31) - 1))
    return h


_g_mults = torch.Generator().manual_seed(0x5eed)
_MIX_MULTS = (torch.randint(-(2 ** 62), 2 ** 62, (64,), generator=_g_mults, dtype=torch.int64) * 2 + 1).tolist()


def _ctx_hashes(tok, lbd, orders):
    """tok [n] int64, lbd [n] (distance to the most recent BOS at or before t).
    Returns {k: (mixed hash of tok[t-k+1..t], valid)} with valid = context inside the
    document (k <= lbd + 1). Order 0 is the empty context."""
    out = {}
    if 0 in orders:
        out[0] = (torch.zeros_like(tok), torch.ones_like(tok, dtype=torch.bool))
    kmax = max(orders)
    h = torch.zeros_like(tok)
    xp = tok + 1
    for i in range(kmax):
        if i == 0:
            h = h + xp * _MIX_MULTS[0]
        else:
            h[i:] += xp[:-i] * _MIX_MULTS[i]
        if (i + 1) in orders:
            out[i + 1] = (_mix64(h), lbd >= i)
    return out


def _lbd_rows(x, bos):
    """x [B,T] -> distance to last BOS (row pos 0 is always BOS), flat [B*T]."""
    B, T = x.shape
    pos = torch.arange(T, device=x.device).expand(B, T)
    lb = torch.where(x == bos, pos, torch.zeros_like(pos)).cummax(1).values
    return (pos - lb).reshape(-1)


def _sorted_unique(parts, nb=8):
    """parts: list of int64 tensors (negative = invalid). Returns (sorted unique valid
    keys, counts). Sorts the disjoint top-bit ranges one at a time (valid keys are
    >= 0, so the top three bits split them into nb ordered buckets) to keep the sort's
    temporary memory at ~1/nb of the input."""
    shift = 63 - (nb.bit_length() - 1)
    Us, cs = [], []
    for b in range(nb):
        sub = torch.cat([p[(p >> shift) == b] for p in parts])
        ks = torch.sort(sub).values
        del sub
        u_, c_ = torch.unique_consecutive(ks, return_counts=True)
        del ks
        Us.append(u_)
        cs.append(c_)
    return torch.cat(Us), torch.cat(cs)


def _table_from_keys(keys):
    """keys int64 (-1 = invalid). -> (U sorted unique valid keys, C int32 cumsum of counts, len Nu+1)."""
    if keys.numel() > (1 << 27):
        U, cnt = _sorted_unique([keys])
    else:
        ks = torch.sort(keys).values
        U, cnt = torch.unique_consecutive(ks, return_counts=True)
        del ks
        if U.numel() > 0 and U[0].item() < 0:
            U, cnt = U[1:], cnt[1:]
    C = torch.zeros(U.numel() + 1, dtype=torch.int32, device=U.device)
    C[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    return U, C


def _table_query(U, C, base, key):
    """base = context key with next-bits zero, key = base | y. Returns (c(ctx,y), c(ctx), N1+(ctx)) float32."""
    dev = base.device
    if U.device != dev:
        base, key = base.to(U.device), key.to(U.device)
    n = U.numel()
    if n == 0:
        z = torch.zeros(base.shape, dtype=torch.float32, device=dev)
        return z, z, z
    lo = torch.searchsorted(U, base)
    hi = torch.searchsorted(U, base + 8192)
    i = torch.searchsorted(U, key)
    ii = i.clamp(max=n - 1)
    match = (i < hi) & (U[ii] == key)
    cxy = torch.where(match, (C[ii + 1] - C[ii]), torch.zeros_like(C[ii]))
    cctx = C[hi] - C[lo]
    n1 = hi - lo
    out = (cxy.float(), cctx.float(), n1.float())
    if U.device != dev:
        out = tuple(o.to(dev, non_blocking=True) for o in out)
    return out


def _wb_step(p, top, conf, U, C, base, v, y, level):
    """One Witten-Bell interpolation step at one order (see _wb_chain)."""
    cxy, cc, n1 = _table_query(U, C, base, base | y)
    has = v & (cc > 0)
    cxy, cc, n1 = cxy.double(), cc.double(), n1.double()
    p = torch.where(has, (cxy + n1 * p) / (cc + n1).clamp(min=1.0), p)
    top = torch.where(has, torch.full_like(top, level), top)
    if conf is not None:
        # target-independent confidence of the top seen context: c(ctx), N1+(ctx)
        conf[0] = torch.where(has, cc, conf[0])
        conf[1] = torch.where(has, n1, conf[1])
    return p, top


def _wb_chain(p, tables, bases, valids, y, conf=None):
    """Witten-Bell interpolation upward over `tables` starting from base distribution p
    (float64, already evaluated at y). Returns (p, index of the highest order with a
    seen context, 0 = none)."""
    top = torch.zeros(p.shape, dtype=torch.long, device=p.device)
    for oi, ((U, C), base, v) in enumerate(zip(tables, bases, valids)):
        p, top = _wb_step(p, top, conf, U, C, base, v, y, oi + 1)
    return p, top


def _table_ctxmax(U, C):
    """Per unique key of a (U, C) table: the id of its context (U >> 13) and, per
    context, the packed max continuation count*8192 + (8191 - token) (ties -> smaller
    token). Target-independent: it summarises the whole context range."""
    n = U.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=U.device), torch.zeros(1, dtype=torch.long, device=U.device)
    ck = U >> 13
    first = torch.ones(n, dtype=torch.long, device=U.device)
    first[1:] = (ck[1:] != ck[:-1]).long()
    cid = torch.cumsum(first, 0) - 1
    cnt = (C[1:] - C[:-1]).long()
    comb = cnt * 8192 + (8191 - (U & 8191))
    cmax = torch.zeros(n, dtype=torch.long, device=U.device)      # (n >= #contexts; no host sync)
    cmax.scatter_reduce_(0, cid, comb, "amax", include_self=False)
    return cid, cmax


def _wb_chain_top(p, tables, tops, bases, valids, y):
    """_wb_chain for GPU tables plus, at the highest order with a seen context, the
    max-continuation share c_max(ctx)/c(ctx), its argmax and c(ctx) -- all read from
    the context's whole range, never from y."""
    top = torch.zeros(p.shape, dtype=torch.long, device=p.device)
    share = torch.zeros(p.shape, dtype=torch.float64, device=p.device)
    arg = torch.full(p.shape, -1, dtype=torch.long, device=p.device)
    ccx = torch.zeros(p.shape, dtype=torch.float64, device=p.device)
    for oi, ((U, C), (cid, cmax), base, v) in enumerate(zip(tables, tops, bases, valids)):
        n = U.numel()
        if n == 0:
            continue
        key = base | y
        lo = torch.searchsorted(U, base)
        hi = torch.searchsorted(U, base + 8192)
        i = torch.searchsorted(U, key)
        ii = i.clamp(max=n - 1)
        match = (i < hi) & (U[ii] == key)
        cxy = torch.where(match, (C[ii + 1] - C[ii]), torch.zeros_like(C[ii])).double()
        cc = (C[hi] - C[lo]).double()
        n1 = (hi - lo).double()
        has = v & (cc > 0)
        p = torch.where(has, (cxy + n1 * p) / (cc + n1).clamp(min=1.0), p)
        top = torch.where(has, torch.full_like(top, oi + 1), top)
        cm = cmax[cid[lo.clamp(max=n - 1)]]
        share = torch.where(has, (cm >> 13).double() / cc.clamp(min=1.0), share)
        arg = torch.where(has, 8191 - (cm & 8191), arg)
        ccx = torch.where(has, cc, ccx)
    return p, top, share, arg, ccx


def _seg_ema(d, lbd, B, T, lam):
    """Causal in-document EMA: f_t = (1-lam) sum_{seg_start<=s<t} lam^(t-1-s) d_s."""
    d = d.view(B, T).double()
    pos = torch.arange(T, device=d.device)
    posf = pos.double()
    cs = torch.cumsum(d * lam ** (-posf), 1)
    csp = torch.cat([torch.zeros_like(cs[:, :1]), cs[:, :-1]], 1)
    st = (pos.unsqueeze(0) - lbd.view(B, T)).clamp(min=0)
    base = torch.gather(csp, 1, st)
    return ((csp - base) * lam ** (posf - 1) * (1 - lam)).reshape(-1)


_FB_EDGES = [-2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0]


def _vote_bucket(tval, tvt):
    # log2 vote bucket of the top doc (0 = no retrieval).
    return torch.where(tval, 1 + (torch.floor(torch.log2(1.0 + tvt.double() / 16.0)) - 2).clamp(0, 7).long(),
                       torch.zeros_like(tvt))


def _doc_expert(xf, yf, lbd, hs, p_ng, ngo, ix, bos, qoff, dense_pos=None, pd_ng=None, ii=None, yy=None, V=None):
    """The retrieved-document expert of CorpusMixer.experts, as a free function of the
    flat batch (xf, yf, lbd, context hashes hs), the n-gram chain p_ng it backs off to,
    and the document index `ix` (attributes S, doc_start, doc_len, P, H, Hstart, df,
    idf). Reads only the context and the training-split index; y enters only through
    the WB lookup of p_doc at y."""
    res = {"buf_tok": 0}
    n = xf.numel()
    NQ = len(MIX_QOFF)
    segid = torch.cumsum((xf == bos).long(), 0) - 1
    nseg = int(segid[-1].item()) + 1
    ngrp = nseg * NQ
    # ---- retrieval: shingle votes, cumulative over query points ----
    hm8, v8 = hs[MIX_SHINGLE_K]
    keep = v8 & (((hm8 >> 3) & (MIX_SHINGLE_S - 1)) == 0)
    pidx = keep.nonzero().squeeze(1)
    sh = (hm8[pidx] >> 22) & _M42
    nH = ix.H.numel()
    j = torch.searchsorted(ix.H, sh)
    jj = j.clamp(max=nH - 1)
    found = (j < nH) & (ix.H[jj] == sh)
    w = torch.where(found, ix.idf[jj], torch.zeros_like(jj))
    o = lbd[pidx]
    i0 = torch.searchsorted(qoff, o)
    sel = (w > 0) & (i0 < NQ)
    pidx, jj, i0 = pidx[sel], jj[sel], i0[sel]
    df = ix.df[jj]
    nqp = NQ - i0
    cnt = df * nqp
    E = int(cnt.sum().item())
    tv = torch.zeros(ngrp, dtype=torch.long, device=xf.device)
    if E > 0:
        qi = torch.repeat_interleave(torch.arange(cnt.numel(), device=xf.device), cnt)
        r = torch.arange(E, device=xf.device) - (torch.cumsum(cnt, 0) - cnt)[qi]
        dfq = df[qi]
        doc = ix.P[ix.Hstart[jj][qi] + r % dfq] & _M21
        g = segid[pidx][qi] * NQ + i0[qi] + r // dfq
        wv = ix.idf[jj][qi]
        uk, inv = torch.unique((g << 21) | doc, return_inverse=True)
        votes = torch.zeros(uk.numel(), dtype=torch.long, device=xf.device).scatter_add_(0, inv, wv)
        gk = uk >> 21
        dk = uk & _M21
        vq = votes.clamp(max=(1 << 19) - 1)
        order = torch.argsort((gk << 40) | ((((1 << 19) - 1) - vq) << 21) | dk)
        gk, dk, votes = gk[order], dk[order], votes[order]
        ar = torch.arange(gk.numel(), device=xf.device)
        first = torch.ones_like(gk, dtype=torch.bool)
        first[1:] = gk[1:] != gk[:-1]
        gst = torch.where(first, ar, torch.zeros_like(ar)).cummax(0).values
        rank = ar - gst
        topm = rank < MIX_DOC_M
        gk, dk, votes, rank = gk[topm], dk[topm], votes[topm], rank[topm]
        r0 = rank == 0
        tv[gk[r0]] = votes[r0]
        # ---- buffers: concatenation of the top-M docs, at most MIX_DOC_L tokens ----
        dl = ix.doc_len[dk]
        ar = torch.arange(gk.numel(), device=xf.device)
        first = torch.ones_like(gk, dtype=torch.bool)
        first[1:] = gk[1:] != gk[:-1]
        gst = torch.where(first, ar, torch.zeros_like(ar)).cummax(0).values
        excl = torch.cumsum(dl, 0) - dl
        sib = excl - excl[gst]
        clen = torch.minimum(dl, (MIX_DOC_L - sib).clamp(min=0))
        tot = int(clen.sum().item())
        bi = torch.repeat_interleave(torch.arange(clen.numel(), device=xf.device), clen)
        boff = torch.arange(tot, device=xf.device) - (torch.cumsum(clen, 0) - clen)[bi]
        btok = ix.S[ix.doc_start[dk][bi] + boff].long()
        bg = gk[bi]
        bnext = torch.cat([btok[1:], btok.new_zeros(1)])
        bnv = torch.cat([(bg[1:] == bg[:-1]) & (boff[1:] > 0), torch.zeros(1, dtype=torch.bool, device=xf.device)])
        hb = _ctx_hashes(btok, boff, DOC_ORDERS)
        doc_tables = []
        for k in DOC_ORDERS:
            hmb, vb = hb[k]
            keys = torch.where(vb & bnv, (bg << 48) | (((hmb >> 29) & _M35) << 13) | bnext,
                               torch.full_like(bg, -1))
            doc_tables.append(_table_from_keys(keys))
        doc_tops = [_table_ctxmax(U_, C_) for U_, C_ in doc_tables]
        res["buf_tok"] = tot
    # ---- query side ----
    qi_t = torch.searchsorted(qoff, lbd, right=True) - 1
    gt = segid * NQ + qi_t.clamp(min=0)
    tval = (qi_t >= 0) & (gt < (1 << 15))
    gt = gt.clamp(max=ngrp - 1)
    tvt = tv[gt]
    tval = tval & (tvt > 0)
    if E > 0:
        d_bases = [(gt << 48) | (((hs[k][0] >> 29) & _M35) << 13) for k in DOC_ORDERS]
        d_valid = [hs[k][1] & tval for k in DOC_ORDERS]
        p_doc, dmax, dshare, darg, dcc = _wb_chain_top(p_ng, doc_tables, doc_tops, d_bases, d_valid, yf)
        if dense_pos is not None:
            pd_doc, _ = _wb_chain(pd_ng, doc_tables, [b[ii] for b in d_bases],
                                  [v[ii] for v in d_valid], yy)
            res["dense_doc"] = pd_doc.view(-1, V).sum(1)
    else:
        p_doc, dmax = p_ng, torch.zeros_like(ngo)
        dshare = torch.zeros_like(p_ng)
        darg = torch.full_like(ngo, -1)
        dcc = torch.zeros_like(p_ng)
    vb_t = _vote_bucket(tval, tvt)
    res["tvt"] = tvt
    res["p_doc"] = p_doc
    res["dmax"], res["vb"], res["dshare"], res["darg"], res["dcc"] = dmax, vb_t, dshare, darg, dcc
    res["hit"] = tval
    return res


class CorpusMixer:
    def __init__(self, vocab_size, bos, T, arena=None):
        self.V = vocab_size
        self.bos = bos
        self.T = T
        self.arena = arena          # pinned int64 host tensor for the tables that do not fit on GPU
        self.ng_tables = []
        self.use_doc = MIX_USE_DOC
        self.n_exp = 3 if self.use_doc else 2
        self.feat_names = self.base_names() + (MIX_DOCFEAT if self.use_doc else [])
        self.gate = None
        self.stats = {}

    # ---------------- build (training tokens only) ----------------
    @torch.no_grad()
    def build(self, S, pre_xy=()):
        """S: int16 [N] GPU, the flat stream of trained-on rows (each row of T starts with BOS).
        pre_xy: batches (the gate-fit slice) whose global n-gram chain is evaluated while
        each order's table is still on the GPU, before it may be moved to host memory."""
        dev = S.device
        N = S.numel()
        T = self.T
        R = N // T
        CH = 1 << 25
        t0 = time.time()
        lbd = torch.empty(N, dtype=torch.int16, device=dev)
        rch = CH // T
        for a in range(0, R, rch):
            b = min(R, a + rch)
            lbd[a * T:b * T] = _lbd_rows(S[a * T:b * T].view(b - a, T).long(), self.bos).to(torch.int16)
        cnt = torch.bincount(S.view(R, T)[:, 1:].reshape(-1).long(), minlength=self.V).double()
        self.p_uni = (cnt + 1.0) / (cnt.sum() + self.V)
        is_bos = S == self.bos
        pre = []
        for px, py in pre_xy:
            pyf = py.reshape(-1)
            plbd = _lbd_rows(px, self.bos)
            phs = _ctx_hashes(px.reshape(-1), plbd, MIX_ORDERS)
            pre.append({"y": pyf, "p": self.p_uni[pyf],
                        "top": torch.zeros(pyf.shape, dtype=torch.long, device=dev),
                        "conf": [torch.zeros(pyf.shape, dtype=torch.float64, device=dev) for _ in range(2)],
                        "bases": [(((phs[k][0] >> 14) & _M50) << 13) for k in MIX_ORDERS],
                        "valid": [phs[k][1] for k in MIX_ORDERS]})
            del phs
        if self.use_doc:
            self.doc_start = is_bos.nonzero().squeeze(1)
            self.ndocs = self.doc_start.numel()
            assert self.ndocs < (1 << 21)
            self.doc_len = torch.diff(self.doc_start, append=torch.tensor([N], device=dev))
            self.S = S
            shingles = []
        h = torch.zeros(N, dtype=torch.int64, device=dev)
        keybuf = torch.empty(N, dtype=torch.int64, device=dev)
        gpu_bytes = 0
        arena_off = 0
        done_i = 0
        for k in MIX_ORDERS:
            for i in range(done_i, k):
                m = _MIX_MULTS[i]
                for a in range(0, N, CH):
                    b = min(N, a + CH)
                    a2 = max(a, i)
                    if a2 < b:
                        h[a2:b] += (S[a2 - i:b - i].long() + 1) * m
            done_i = k
            doc_off = 0
            for a in range(0, N, CH):
                b = min(N, a + CH)
                hm = _mix64(h[a:b])
                ctx = (hm >> 14) & _M50
                nb = min(N, b + 1)
                nx = S[a + 1:nb].long()
                if nx.numel() < b - a:
                    nx = torch.cat([nx, nx.new_zeros(b - a - nx.numel())])
                pos = torch.arange(a, b, device=dev)
                lb_c = lbd[a:b]
                valid = (lb_c >= k - 1) & (((pos + 1) % T) != 0)
                keybuf[a:b] = torch.where(valid, (ctx << 13) | nx, torch.full_like(ctx, -1))
                if self.use_doc and k == MIX_SHINGLE_K:
                    docid = torch.cumsum(is_bos[a:b].long(), 0) - 1 + doc_off
                    doc_off = docid[-1].item() + 1
                    keep = (lb_c >= k - 1) & (((hm >> 3) & (MIX_SHINGLE_S - 1)) == 0)
                    sh = (hm[keep] >> 22) & _M42
                    shingles.append((sh << 21) | docid[keep])
            U, C = _table_from_keys(keybuf)
            oi = MIX_ORDERS.index(k)
            for st in pre:
                st["p"], st["top"] = _wb_step(st["p"], st["top"], st["conf"], U, C, st["bases"][oi],
                                              st["valid"][oi], st["y"], oi + 1)
            nbytes = U.numel() * 12
            if gpu_bytes + nbytes <= MIX_GPU_GB * 1e9 or self.arena is None:
                gpu_bytes += nbytes
                self.ng_tables.append((U, C))
            else:
                n = U.numel()
                nC = (n + 2) // 2
                assert arena_off + n + nC <= self.arena.numel(), "host arena too small"
                Uh = self.arena[arena_off:arena_off + n]
                Ch = self.arena[arena_off + n:arena_off + n + nC].view(torch.int32)[:n + 1]
                arena_off += n + nC
                Uh.copy_(U, non_blocking=True)
                Ch.copy_(C, non_blocking=True)
                torch.cuda.synchronize()
                self.ng_tables.append((Uh, Ch))
                del U, C
        del h, keybuf, lbd
        self.pre_ng = [(st["p"], st["top"], st["conf"]) for st in pre]
        del pre
        torch.cuda.synchronize()
        self.stats["t_ng_build"] = time.time() - t0
        self.stats["ng_gpu_gb"] = gpu_bytes / 1e9
        self.stats["ng_host_gb"] = arena_off * 8 / 1e9
        if self.use_doc:
            t1 = time.time()
            P, _ = _sorted_unique(shingles)
            del shingles
            Hh, df = torch.unique_consecutive(P >> 21, return_counts=True)
            self.P = P
            self.H = Hh
            self.Hstart = torch.cumsum(df, 0) - df
            self.df = df
            idf = torch.round(16.0 * torch.log(self.ndocs / df.double())).long()
            self.idf = torch.where(df <= MIX_DF_MAX, idf, torch.zeros_like(idf))
            torch.cuda.synchronize()
            self.stats["t_doc_build"] = time.time() - t1
            self.stats["doc_index_gb"] = (P.numel() * 8 + Hh.numel() * 32) / 1e9
            self.stats["ndocs"] = self.ndocs
        self.qoff = torch.tensor(MIX_QOFF, device=dev)

    # ---------------- prebuilt (background worker) tables ----------------
    DOC_KEYS = ("S", "doc_start", "doc_len", "P", "H", "Hstart", "df", "idf", "p_uni",
                "pre_p", "pre_top", "pre_c0", "pre_c1")

    @staticmethod
    def upload_doc(tens, dev, stream=None):
        """H2D of everything the mixer needs on the GPU regardless of the table budget
        (stream + doc index + unigram + the slice's prebuilt n-gram chain)."""
        out = {}
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            for nm in CorpusMixer.DOC_KEYS:
                if nm in tens:
                    out[nm] = tens[nm].to(dev, non_blocking=True)
        if stream is not None:
            stream.synchronize()
        return out

    @torch.no_grad()
    def load_prebuilt(self, tens, gpu_doc, info, dev, nslice_tok):
        """Adopt the background worker's tables: the doc-side tensors already on the GPU
        (gpu_doc, from upload_doc), n-gram orders in MIX_ORDERS order onto the GPU while
        they fit MIX_GPU_GB, the rest stay in the shared host arena (queried on CPU, as
        not on the GPU)."""
        t0 = time.time()
        self.p_uni = gpu_doc["p_uni"]
        gpu_bytes = host_bytes = 0
        for k in MIX_ORDERS:
            U, C = tens[f"U{k}"], tens[f"C{k}"]
            nbytes = U.numel() * 12
            if gpu_bytes + nbytes <= MIX_GPU_GB * 1e9:
                gpu_bytes += nbytes
                U, C = U.to(dev, non_blocking=True), C.to(dev, non_blocking=True)
            else:
                host_bytes += nbytes
            self.ng_tables.append((U, C))
        n = nslice_tok
        pp, pt, c0, c1 = gpu_doc["pre_p"], gpu_doc["pre_top"], gpu_doc["pre_c0"], gpu_doc["pre_c1"]
        self.pre_ng = [(pp[a:a + n], pt[a:a + n], [c0[a:a + n], c1[a:a + n]])
                       for a in range(0, pp.numel(), n)]
        if self.use_doc:
            self.S = gpu_doc["S"]
            self.doc_start = gpu_doc["doc_start"]
            self.doc_len = gpu_doc["doc_len"]
            self.ndocs = int(info["pieces"])
            self.P, self.H, self.Hstart = gpu_doc["P"], gpu_doc["H"], gpu_doc["Hstart"]
            self.df, self.idf = gpu_doc["df"], gpu_doc["idf"]
            self.stats["doc_index_gb"] = (self.P.numel() * 8 + self.H.numel() * 32) / 1e9
            self.stats["ndocs"] = self.ndocs
        self.qoff = torch.tensor(MIX_QOFF, device=dev)
        torch.cuda.synchronize()
        self.stats["t_ng_build"] = time.time() - t0
        self.stats["ng_gpu_gb"] = gpu_bytes / 1e9
        self.stats["ng_host_gb"] = host_bytes / 1e9

    # ---------------- per-batch experts ----------------
    @torch.no_grad()
    def experts(self, x, y, dense_pos=None, pre=None):
        """x, y [B,T] int64. Returns dict with p_ng, p_doc (float64 at y), features.
        If dense_pos is given (flat indices), also returns the full-vocab sums of each
        expert at those positions (normalisation check)."""
        B, T = x.shape
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        lbd = _lbd_rows(x, self.bos)
        orders = sorted(set(MIX_ORDERS) | set(DOC_ORDERS))
        hs = _ctx_hashes(xf, lbd, orders)
        ng_bases = [(((hs[k][0] >> 14) & _M50) << 13) for k in MIX_ORDERS]
        ng_valid = [hs[k][1] for k in MIX_ORDERS]
        if pre is not None:
            p_ng, ngo, conf = pre
        else:
            conf = [torch.zeros(yf.shape, dtype=torch.float64, device=yf.device) for _ in range(2)]
            p_ng, ngo = _wb_chain(self.p_uni[yf], self.ng_tables, ng_bases, ng_valid, yf, conf)
        # confidence bucket of the top context: log2 c(ctx) (8) x share of distinct
        # continuations N1+/c (4)
        ccb = torch.floor(torch.log2(conf[0].clamp(min=1.0))).long().clamp(0, 7)
        rb = torch.bucketize(conf[1] / conf[0].clamp(min=1.0),
                             torch.tensor([0.1, 0.3, 0.6], device=yf.device, dtype=torch.float64))
        out = {"p_ng": p_ng, "ngo": ngo, "lbd": lbd, "ngc": ccb * 4 + rb}
        dense = {}
        if dense_pos is not None:
            V = self.V
            ii = dense_pos.repeat_interleave(V)
            yy = torch.arange(V, device=x.device).repeat(dense_pos.numel())
            pd_ng, _ = _wb_chain(self.p_uni[yy], self.ng_tables, [b[ii] for b in ng_bases],
                                 [v[ii] for v in ng_valid], yy)
            dense["ng"] = pd_ng.view(-1, V).sum(1)
        if not self.use_doc:
            out["p_doc"] = p_ng
            out["docf"] = torch.zeros_like(ngo)
            out["dmax"], out["vb"], out["darg"] = torch.zeros_like(ngo), torch.zeros_like(ngo), torch.full_like(ngo, -1)
            out["dshare"], out["dcc"] = torch.zeros_like(p_ng), torch.zeros_like(p_ng)
            out["dense"] = dense
            return out
        r_ = _doc_expert(
            xf, yf, lbd, hs, p_ng, ngo, self, self.bos, self.qoff, dense_pos,
            pd_ng if dense_pos is not None else None, ii if dense_pos is not None else None,
            yy if dense_pos is not None else None, self.V)
        if "buf_tok" in r_:
            self.stats["last_buf_tokens"] = r_["buf_tok"]
        if "dense_doc" in r_:
            dense["doc"] = r_["dense_doc"]
        dmax = r_["dmax"]
        vb_t = r_["vb"]
        out["p_doc"] = r_["p_doc"]
        out["docf"] = dmax * 9 + vb_t
        out["dmax"], out["vb"], out["dshare"], out["darg"], out["dcc"] = dmax, vb_t, r_["dshare"], r_["darg"], r_["dcc"]
        out["hit"] = r_["hit"]
        out["dense"] = dense
        return out

    # ---------------- gate ----------------
    def base_names(self):
        nm = ["ngo", "docf", "posb", "fb_ng", "fb_doc"]
        if MIX_FB_DECAY2 > 0:
            nm += ["fb_ng2", "fb_doc2"]
        if MIX_CONF:
            nm.append("ngc")
            if MIX_CONF_X:
                nm.append("ngcx")
        if MIX_ENT:
            nm += ["eb", "eb_ngo"]
        return nm

    def feat_sizes(self):
        no = len(MIX_ORDERS) + 1
        ne = len(_ENT_EDGES) + 1
        sz = {"ngo": no, "docf": (len(DOC_ORDERS) + 1) * 9, "posb": 12, "fb_ng": 9, "fb_doc": 9,
              "fb_ng2": 9, "fb_doc2": 9, "ngc": 32, "ngcx": 32 * no, "eb": ne, "eb_ngo": ne * no}
        sz.update(_DOCFEAT_SIZES)
        return sz

    def _features(self, ex, p_net, B, T, ent=None, top=None, names=None):
        """Target-independent gate features -> {name: int64 index tensor}. `top` is the
        network's (top-1 prob, top-16 ids) from the logits pass (context only)."""
        names = self.feat_names if names is None else names
        lbd = ex["lbd"]
        dev = lbd.device
        lp_net = torch.log(p_net.double().clamp(min=1e-30))
        edges = torch.tensor(_FB_EDGES, device=dev, dtype=torch.float64)
        d_ng = (torch.log(ex["p_ng"].clamp(min=1e-30)) - lp_net).clamp(-10, 10)
        d_doc = (torch.log(ex["p_doc"].clamp(min=1e-30)) - lp_net).clamp(-10, 10)
        f = {"ngo": ex["ngo"], "docf": ex["docf"],
             "posb": torch.floor(torch.log2(1.0 + lbd.double())).long().clamp(0, 11),
             "fb_ng": torch.bucketize(_seg_ema(d_ng, lbd, B, T, MIX_FB_DECAY), edges),
             "fb_doc": torch.bucketize(_seg_ema(d_doc, lbd, B, T, MIX_FB_DECAY), edges)}
        if MIX_FB_DECAY2 > 0:
            f["fb_ng2"] = torch.bucketize(_seg_ema(d_ng, lbd, B, T, MIX_FB_DECAY2), edges)
            f["fb_doc2"] = torch.bucketize(_seg_ema(d_doc, lbd, B, T, MIX_FB_DECAY2), edges)
        f["ngc"] = ex["ngc"]
        f["ngcx"] = ex["ngo"] * 32 + ex["ngc"]
        if ent is not None:
            eb = torch.bucketize(ent.float(), torch.tensor(_ENT_EDGES, device=dev))
            f["eb"] = eb
            f["eb_ngo"] = eb * (len(MIX_ORDERS) + 1) + ex["ngo"]
        if top is not None:
            p1, ids = top
            t1b = torch.bucketize(p1.float(), torch.tensor(_T1_EDGES, device=dev))
            dmax, darg = ex["dmax"], ex["darg"]
            nodoc = dmax == 0
            dsb = torch.where(nodoc, torch.zeros_like(dmax),
                              1 + torch.bucketize(ex["dshare"], torch.tensor(_DS_EDGES, device=dev, dtype=torch.float64)))
            eq = ids.long() == darg.unsqueeze(1)
            r = eq.int().argmax(1)
            rb = torch.where(r == 0, torch.zeros_like(r),
                             1 + torch.floor(torch.log2(r.clamp(min=1).double())).long())
            rb = torch.where(eq.any(1), rb, torch.full_like(rb, 5))
            rb = torch.where(nodoc, torch.full_like(rb, 6), rb)
            ag = (rb == 0).long()
            dccb = torch.floor(torch.log2(ex["dcc"].clamp(min=1.0))).long().clamp(0, 7)
            f["ds_t1"] = dsb * _NT1 + t1b
            f["ds_rb"] = dsb * _NRB + rb
            f["rb_t1"] = rb * _NT1 + t1b
            f["vb_ds_ag"] = (ex["vb"] * _NDS + dsb) * 2 + ag
            f["dm_rb"] = dmax * _NRB + rb
            f["dc_ds"] = dccb * _NDS + dsb
            f["_dsb"], f["_rb"] = dsb, rb       # diagnostics only (not gate inputs)
        return f

    @staticmethod
    def warmup_fit():
        """One tiny Adam fit on random data, so torch.optim's lazy imports and first-use
        initialisation (several seconds) are paid at startup, like compilation, rather than
        inside the timed gate fit. It touches no model parameter and no data."""
        W = torch.zeros(8, 2, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W], lr=0.01)
        X = torch.rand(64, 8, device="cuda")
        with torch.enable_grad():
            for _ in range(3):
                loss = torch.log(torch.softmax(X @ W, 1)[:, 0]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            W.grad = torch.ones_like(W)
            opt.step()
        torch.cuda.synchronize()

    def _gate_logits(self, feats, params, names=None):
        # F.embedding (sort-based dense backward) instead of W[f]: the indices are
        # massively duplicated and index backward's atomics made the fit 0.2 s/iter.
        names = self.feat_names if names is None else names
        z = params[-1].unsqueeze(0)
        for nm, W in zip(names, params[:-1]):
            z = z + F.embedding(feats[nm], W)
        return z

    def _mix(self, z, p_net, p_ng, p_doc):
        zfull = torch.cat([torch.zeros_like(z[:, :1]), z], 1)
        w = torch.softmax(zfull, 1)
        p = w[:, 0] * p_net + w[:, 1] * p_ng
        if self.n_exp == 3:
            p = p + w[:, 2] * p_doc
        return p, w

    def collect(self, batches, token_bytes, pre_ng=()):
        """batches: list of (x, y, loss_net, ent_net, top). Returns the concatenated,
        byte-masked gate inputs (all features) plus the batch index of every row."""
        feats, pn, pg, pd, msk, bidx = [], [], [], [], [], []
        for j, (x, y, loss_net, ent, top) in enumerate(batches):
            B, T = x.shape
            ex = self.experts(x, y, pre=pre_ng[j] if j < len(pre_ng) else None)
            p_net = torch.exp(-loss_net.double())
            feats.append(self._features(ex, p_net, B, T, ent, top))
            pn.append(p_net); pg.append(ex["p_ng"]); pd.append(ex["p_doc"])
            msk.append(token_bytes[y.reshape(-1)] > 0)
            bidx.append(torch.full_like(msk[-1], j, dtype=torch.long))
            del ex
        msk = torch.cat(msk)
        out = {"feats": {k: torch.cat([f_[k] for f_ in feats])[msk] for k in feats[0]}}
        out["pn"], out["pg"], out["pd"] = [torch.cat(v).float()[msk] for v in (pn, pg, pd)]
        out["bidx"] = torch.cat(bidx)[msk]
        return out

    def _fit_sparse(self, feats, names, pn, pg, pd, iters=None):
        """Adam on the gate (z = sum_f W_f[feat_f] + bias) with an exact analytic gradient:
        dL/dz_e = -(1/n) w_e (p_e/p - 1), scattered to W by a float64 weighted bincount
        over the feature indices -- no one-hot design matrix and no autograd, so the whole
        slice fits in the budget."""
        iters = MIX_FIT_ITERS if iters is None else iters
        sz = self.feat_sizes()
        sizes = [sz[nm] for nm in names]
        offs = [0]
        for s_ in sizes:
            offs.append(offs[-1] + s_)
        n = pn.numel()
        nf = len(names)
        ne = self.n_exp - 1
        dev = pn.device
        Fi = torch.stack([feats[nm] + o for nm, o in zip(names, offs)], 1)
        flat = Fi.reshape(-1)
        P = torch.stack([pn, pg, pd][:self.n_exp], 1)
        W = torch.zeros(offs[-1], ne, device=dev, requires_grad=True)
        bias = torch.full((ne,), -2.5, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W, bias], lr=MIX_FIT_LR)
        with torch.no_grad():
            for it in range(iters):
                z = F.embedding(Fi, W).sum(1) + bias
                w = torch.softmax(torch.cat([torch.zeros_like(z[:, :1]), z], 1), 1)
                p = (w * P).sum(1).clamp(min=1e-30)
                g = -(w[:, 1:] * (P[:, 1:] / p.unsqueeze(1) - 1.0)) / n
                gd = g.double()
                W.grad = torch.stack([torch.bincount(flat, weights=gd[:, e].unsqueeze(1).expand(n, nf).reshape(-1),
                                                     minlength=offs[-1]) for e in range(ne)], 1).float()
                bias.grad = g.sum(0)
                opt.step()
            params = [W[offs[i]:offs[i + 1]].detach() for i in range(nf)] + [bias.detach()]
            z = self._gate_logits(feats, params, names)
            p, w = self._mix(z, pn, pg, pd)
            lmix = -torch.log(p.clamp(min=1e-30)).mean().item()
        return params, lmix, w

    @torch.no_grad()
    def eval_gate(self, data, params, names, sel=None):
        f = data["feats"] if sel is None else {k: v[sel] for k, v in data["feats"].items()}
        pn, pg, pd = (data[k] if sel is None else data[k][sel] for k in ("pn", "pg", "pd"))
        z = self._gate_logits(f, params, names)
        p, w = self._mix(z, pn, pg, pd)
        return -torch.log(p.clamp(min=1e-30)).mean().item(), w

    def fit_gate(self, slice_data, token_bytes, keep=False):
        """slice_data: list of (x, y, loss_net, ent_net, top) held-out TRAINING batches."""
        t0 = time.time()
        data = self.collect(slice_data, token_bytes, getattr(self, "pre_ng", []))
        torch.cuda.synchronize()
        self.stats["t_fit_experts"] = time.time() - t0
        pn = data["pn"]
        if MIX_FIT_TOK > 0 and pn.numel() > MIX_FIT_TOK:
            _g = torch.Generator(device=pn.device).manual_seed(1234)
            sel = torch.rand(pn.numel(), device=pn.device, generator=_g) < MIX_FIT_TOK / pn.numel()
            fd = {k: v[sel] for k, v in data["feats"].items()}
            pn_, pg_, pd_ = data["pn"][sel], data["pg"][sel], data["pd"][sel]
        else:
            fd, pn_, pg_, pd_ = data["feats"], data["pn"], data["pg"], data["pd"]
        params, lmix, w = self._fit_sparse(fd, self.feat_names, pn_, pg_, pd_)
        self.stats["slice_nats_net"] = -torch.log(pn_.clamp(min=1e-30)).mean().item()
        self.stats["slice_nats_mix"] = lmix
        self.stats["slice_w_mean"] = w.mean(0).tolist()
        self.gate = params
        self.slice_cache = data if keep else None
        torch.cuda.synchronize()
        self.stats["t_fit"] = time.time() - t0

    @torch.no_grad()
    def mix_nll(self, x, y, loss_net, ent=None, top=None, return_parts=False):
        B, T = x.shape
        ex = self.experts(x, y)
        p_net = torch.exp(-loss_net.double())
        feats = self._features(ex, p_net, B, T, ent, top)
        z = self._gate_logits(feats, self.gate).double()
        p, w = self._mix(z, p_net, ex["p_ng"], ex["p_doc"])
        nll = -torch.log(p.clamp(min=1e-300))
        if return_parts:
            return nll, ex, w, feats
        return nll.float()


class MixtureEval(nn.Module):
    def __init__(self, base, mixer):
        super().__init__()
        self._base = base
        self._mixer = mixer

        self.dump = [] if os.environ.get("MIX_DUMP") else None

    def forward(self, idx, targets=None, reduction='mean'):
        loss_net, ent, top = net_loss_ent(self._base, idx, targets)
        if self.dump is not None:
            self.dump.append((loss_net.cpu(), None if ent is None else ent.cpu()))
        nll = self._mixer.mix_nll(idx, targets, loss_net, ent, top)
        return nll if reduction == 'none' else nll.mean()


@torch.no_grad()
def net_loss_ent(base, idx, targets):
    """Per-token network NLL (flat fp32) and, if MIX_ENT, the network's predictive
    entropy and (top-1 prob, top-MIX_TOPK ids). With MIX_ENT the network is called for
    its (softcapped, fp32) logits and the cross-entropy is formed here, exactly as
    F.cross_entropy does inside it. Entropy and top-k read the logits only."""
    if not MIX_ENT:
        return base(idx, targets, reduction='none').float().view(-1), None, None
    logits = base(idx, None)
    V = logits.size(-1)
    lg = logits.view(-1, V)
    yf = targets.reshape(-1)
    loss = torch.empty(lg.size(0), dtype=torch.float32, device=lg.device)
    ent = torch.empty_like(loss)
    p1 = torch.empty_like(loss)
    tid = torch.empty(lg.size(0), MIX_TOPK, dtype=torch.int32, device=lg.device)
    CHK = 16384
    for a in range(0, lg.size(0), CHK):
        z = lg[a:a + CHK].float()
        lse = torch.logsumexp(z, 1)
        pz = torch.softmax(z, 1)
        ent[a:a + CHK] = lse - (pz * z).sum(1)
        pv, pi = torch.topk(pz, MIX_TOPK, 1)
        p1[a:a + CHK] = pv[:, 0]
        tid[a:a + CHK] = pi.int()
        yy = yf[a:a + CHK]
        zt = z.gather(1, yy.clamp(min=0).unsqueeze(1)).squeeze(1)
        loss[a:a + CHK] = torch.where(yy >= 0, lse - zt, torch.zeros_like(lse))
    return loss, ent, (p1, tid)


SPARSE_OPT_COMPILE = os.environ.get("SPARSE_OPT_COMPILE", "1") == "1"
# Sync-free sparse step (see SparseBigramAdamW._dedup_fixed): each n-gram table gets
# SPARSE_PAD_ROWS extra rows beyond its hashed range to absorb the unused dedup slots.
# OFF by default because it MEASURED SLOWER: solo, same GPU, steps 50-400, precise dt
# mean 79.86 ms sync-free vs 79.03 ms with coalesce(). The coalesce() host syncs are
# not on the critical path -- they block only the host, which has nothing else to issue
# while Muon runs on the main stream -- whereas the fixed [B*T]-row layout makes the
# side-stream update gather/scatter every slot (pad duplicates included) plus fp32
# atomics on hot bigram rows, and that extra bandwidth is taken from Muon. Numerically
# it matches coalesce exactly (checked on real embedding grads); kept for reference.
SPARSE_SYNC_FREE = os.environ.get("SPARSE_SYNC_FREE", "0") == "1"
SPARSE_PAD_ROWS = 1024

# Row scatter for the sparse n-gram tables: table[indices] = rows, indices unique (they come
# out of coalesce()). The eager index_put_ (index_elementwise_kernel) moves one element per
# thread with a full index decode each, ~0.4-0.6 TB/s here: 267 us for the bigram weight
# rows, 269 us for its fp32 momentum rows and 438 us for the trigram weight rows, every step,
# on the side stream that shares the SMs with Muon. This copies whole contiguous rows (a
# pure data movement, bit-identical result): 387 -> 59 us (bf16) and 407 -> 108 us (fp32)
# standalone at 110K rows x 768. In-run, toggled every 10 steps within one process on each
# of 4 GPUs (steps 30-500): -0.40..-0.60 ms/step, ~0.5 ms mean, same loss trajectory.
SPARSE_TRITON_SCATTER = os.environ.get("SPARSE_TRITON_SCATTER", "1") == "1"
import triton as _str
import triton.language as _stl

@_str.jit
def _row_scatter_kernel(dst_ptr, src_ptr, idx_ptr, k, D: _stl.constexpr, BR: _stl.constexpr,
                        BD: _stl.constexpr):
    rows = _stl.program_id(0) * BR + _stl.arange(0, BR)
    rm = rows < k
    di = _stl.load(idx_ptr + rows, mask=rm, other=0).to(_stl.int64)
    r64 = rows.to(_stl.int64)
    for c0 in _stl.static_range(0, D, BD):
        cols = c0 + _stl.arange(0, BD)
        v = _stl.load(src_ptr + r64[:, None] * D + cols[None, :], mask=rm[:, None])
        _stl.store(dst_ptr + di[:, None] * D + cols[None, :], v, mask=rm[:, None])

def _scatter_rows_(dst, indices, src):
    """dst[indices] = src for a 2-D contiguous dst with unique indices."""
    D = dst.shape[1] if dst.dim() == 2 else 0
    if (not SPARSE_TRITON_SCATTER or dst.dim() != 2 or D % 256 != 0 or not dst.is_contiguous()
            or src.dtype != dst.dtype or src.shape != (indices.numel(), D)):
        dst[indices] = src
        return
    src = src.contiguous()
    k = indices.numel()
    if k == 0:
        return
    _row_scatter_kernel[(_str.cdiv(k, 4),)](dst, src, indices, k, D=D, BR=4, BD=256, num_warps=4)

@torch.compile(fullgraph=True)
def _sparse_adamw_freq_factored_core(weight, exp_avg, exp_avg_sq, cnt, indices, grad_raw,
                                     base_lr_t, lrm_t, K_t, floor_t, wd_t,
                                     b1_t, b2_t, eps_t, bias1_t, bias2_t):
    """The arithmetic of SparseBigramAdamW.step() for the (freq_lr, factored_v) case,
    in ONE compiled region so inductor fuses the ~20 eager kernels it replaces into a
    couple of passes over the [n, d] slices.

    Same ops in the same order and the same dtypes as the eager path (b1_t*m stays in
    exp_avg's dtype exactly as `beta1 * m` did, everything else in fp32), so the update
    is arithmetically the eager one -- this is a kernel-count/bandwidth change, not a
    rule change. The row GATHERS live inside (their outputs are [n, d], small); the four
    SCATTERS deliberately stay outside, because an index_put_ on a 25-42 GB table inside
    a compiled region relies on inductor's re-inplacing pass and would clone the whole
    table if that pass ever declined. Scalars arrive as 0-dim CPU tensors (the pattern
    adamw_step_fused already uses) so a changing LR/step never triggers a recompile.
    Returns the new (exp_avg, exp_avg_sq, weight, cnt) rows for the caller to scatter."""
    cnt_old = cnt[indices]
    c = cnt_old.float()
    conf = c / (c + K_t)
    row_mult = torch.clamp(1.0 - conf * (1.0 - floor_t), min=lrm_t)
    lr_vec = (base_lr_t * row_mult).unsqueeze(1)
    grad_vals = grad_raw.float()
    m = exp_avg[indices]
    v = exp_avg_sq[indices]
    p = weight[indices].float()
    p = p * (1.0 - lr_vec * wd_t)
    m_new = b1_t * m + (1.0 - b1_t) * grad_vals
    gsq = grad_vals.square().mean(dim=-1, keepdim=True)
    v_new = b2_t * v + (1.0 - b2_t) * gsq
    denom = (v_new / bias2_t).sqrt() + eps_t
    coef = lr_vec / denom
    p = p - (m_new * coef) / bias1_t
    return m_new.to(exp_avg.dtype), v_new, p.to(weight.dtype), cnt_old + 1

@torch.compile(fullgraph=True)
def _sparse_adamw_freq_factored_core_nomom(weight, exp_avg_sq, cnt, indices, grad_raw,
                                           base_lr_t, lrm_t, K_t, floor_t, wd_t,
                                           b2_t, eps_t, bias2_t):
    """The same (freq_lr, factored_v) update with beta1 == 0, i.e. NO first-moment
    buffer at all: m_new is the raw gradient and bias1 is exactly 1.0.

    Why a separate compiled core instead of `if nomom` inside the one above: the
    whole point of nomom is to never allocate the [n, d] exp_avg tensor (118 GB at
    76.8M rows), so there is no tensor to pass in and no gather/scatter
    to perform. And this arm MUST stay compiled -- running the eager update for the
    trigram table instead costs ~4.6% of step time (88.3 vs 84.4 ms/step), which
    alone is larger than the bpb this memory reallocation buys.

    Justification for dropping momentum on the trigram table specifically: a random
    trigram row is revisited about once every 9 optimizer steps (visits_ew 415 over
    3744 steps), so its exp_avg holds a stale gradient from several hundred steps of
    other data rather than a running average of this row's gradient. The bigram
    table, whose rows are hit far more often, keeps its momentum unchanged.

    Ops, order and dtypes otherwise match the momentum core exactly. Scalars arrive
    as 0-dim CPU tensors so a changing LR never triggers a recompile; the scatters
    stay outside for the same re-inplacing reason as above."""
    cnt_old = cnt[indices]
    c = cnt_old.float()
    conf = c / (c + K_t)
    row_mult = torch.clamp(1.0 - conf * (1.0 - floor_t), min=lrm_t)
    lr_vec = (base_lr_t * row_mult).unsqueeze(1)
    grad_vals = grad_raw.float()
    v = exp_avg_sq[indices]
    p = weight[indices].float()
    p = p * (1.0 - lr_vec * wd_t)
    m_new = grad_vals
    gsq = grad_vals.square().mean(dim=-1, keepdim=True)
    v_new = b2_t * v + (1.0 - b2_t) * gsq
    denom = (v_new / bias2_t).sqrt() + eps_t
    coef = lr_vec / denom
    p = p - m_new * coef
    return v_new, p.to(weight.dtype), cnt_old + 1

@torch.compile(fullgraph=True, dynamic=False)
def _sparse_dedup_fixed(idx, vals, pad_slots):
    """Fixed-shape coalesce (see SparseBigramAdamW._dedup_fixed). Duplicate rows are
    summed in fp32 and rounded to the grad dtype once, as coalesce() does."""
    sidx, perm = torch.sort(idx)
    new_run = torch.cat([torch.ones(1, dtype=torch.bool, device=idx.device),
                         sidx[1:] != sidx[:-1]])
    run_id = torch.cumsum(new_run.to(torch.int32), 0) - 1
    uniq = pad_slots.scatter(0, run_id.long(), sidx)
    g = torch.zeros(vals.shape, dtype=torch.float32, device=vals.device)
    g = g.index_add(0, run_id, vals[perm].float())
    return uniq, g.to(vals.dtype)

class SparseBigramAdamW:
    """
    Sparse AdamW for a large hash embedding table.
    Only updates the ~131K accessed rows per step, not all TABLE_SIZE rows.

    factored_v: if True, the second moment (exp_avg_sq) is stored as a per-ROW
    scalar [n,1] fp32 (Adafactor-style row-wise RMS) instead of per-coordinate
    [n,d]. This roughly HALVES the optimizer-state VRAM (only the [n,d] momentum
    stays large), funding a bigger table WITHOUT the zero-quantization / denom-
    explosion hazard of int8 states. Adam math is still done in fp32; only the
    between-step STORAGE of the second moment is factored. Momentum stays
    per-coordinate (state_dtype). Used for the grown trigram; the bigram keeps
    the full per-coordinate optimizer (byte-identical to the original).
    """
    def __init__(self, weight, initial_lr, betas=(0.8, 0.98), eps=1e-10, weight_decay=0.003,
                 state_dtype=torch.float32, name="bigram", factored_v=False,
                 freq_lr=False, freq_lr_K=8.0, freq_lr_floor=0.0, nomom=False):
        self.weight = weight
        self.lr = initial_lr
        self.initial_lr = initial_lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state_dtype = state_dtype
        self.factored_v = factored_v
        self.nomom = nomom

        self.freq_lr = freq_lr
        self.freq_lr_K = float(freq_lr_K)
        self.freq_lr_floor = float(freq_lr_floor)
        self.base_lr = initial_lr
        self.lrm = 1.0
        n, d = weight.shape
        device = weight.device
        if freq_lr:
            self.cnt = torch.zeros(n, dtype=torch.int32, device=device)
            print(f"[SparseBigramAdamW:{name}] freq_lr ON: per-row LR, K={self.freq_lr_K}, "
                  f"rare_row_floor={self.freq_lr_floor}, cnt tensor {n*4/1e9:.2f} GB int32")
        bytes_per = torch.finfo(state_dtype).bits // 8
        if nomom:
            assert betas[0] == 0.0, \
                f"nomom requires beta1 == 0 so the update is exactly momentum-free, got {betas}"
            # The single biggest tensor in the whole run is NOT allocated. Keep a 1x1
            # placeholder so anything that touches .exp_avg sees a tensor, not None.
            self.exp_avg = torch.zeros(1, 1, dtype=state_dtype, device=device)
            saved_gb = n * d * bytes_per / 1e9
            print(f"[SparseBigramAdamW:{name}] nomom ON (beta1=0): exp_avg NOT allocated, "
                  f"saving {n*d*bytes_per:,} bytes ({saved_gb:.1f} GB {state_dtype}); "
                  f"m_new = grad, bias1 = 1.0")
        else:
            self.exp_avg = torch.zeros(n, d, dtype=state_dtype, device=device)
        if factored_v:
            self.exp_avg_sq = torch.zeros(n, 1, dtype=torch.float32, device=device)
            m_gb = 0.0 if nomom else n * d * bytes_per / 1e9
            v_gb = n * 4 / 1e9
            print(f"[SparseBigramAdamW:{name}] Allocating {n:,}×{d} exp_avg ({m_gb:.1f} GB {state_dtype}) "
                  f"+ {n:,}×1 factored exp_avg_sq ({v_gb:.2f} GB fp32) on {device}")
        else:
            self.exp_avg_sq = torch.zeros(n, d, dtype=state_dtype, device=device)
            gb = 2 * n * d * bytes_per / 1e9
            print(f"[SparseBigramAdamW:{name}] Allocating {n:,}×{d} optimizer states ({gb:.1f} GB {state_dtype}) on {device}")
        self.n_real = n - (SPARSE_PAD_ROWS if SPARSE_SYNC_FREE else 0)
        self._pad_slots = None
        self.step_count = 0
        self._use_core = SPARSE_OPT_COMPILE and freq_lr and factored_v
        self._sc = {k: torch.tensor(0.0, dtype=torch.float32, device="cpu")
                    for k in ("base_lr", "lrm", "K", "floor", "wd", "b1", "b2",
                              "eps", "bias1", "bias2")} if self._use_core else None
        if self._use_core:
            print(f"[SparseBigramAdamW:{name}] compiled update core ON "
                  f"({'nomom' if nomom else 'momentum'} variant; fuses the per-step "
                  f"elementwise/convert kernels; scatters stay eager)")
        print(f"[SparseBigramAdamW:{name}] Done.")

    @torch.no_grad()
    def _dedup_fixed(self, grad):
        """Host-sync-free replacement for grad.coalesce().

        coalesce() has a data-dependent output size, so every call ends in a blocking
        cudaStreamSynchronize; with two tables that is two host stalls per step at the
        step tail. The uncoalesced grad of nn.Embedding(sparse=True) always has exactly
        N = B*T index rows, so dedup into a FIXED [N] slot layout instead: sort the
        indices, flag the start of each run, run_id = cumsum(flags) - 1, and index_add
        the rows into an [N, d] fp32 buffer (summing duplicates in fp32 and rounding once
        to bf16, as coalesce does; checked equal to coalesce on real embedding grads). The slots past the last run point at
        SPARSE_PAD_ROWS dedicated pad rows appended to the table beyond n_real, which no
        hash ever produces (keys are taken mod the real table size). Their grad is zero,
        so their update is a no-op on zero state, and duplicate scatters into a pad row
        all write the same value, so they cannot corrupt anything."""
        idx = grad._indices()[0]
        vals = grad._values()
        N = idx.numel()
        if self._pad_slots is None or self._pad_slots.numel() != N:
            self._pad_slots = (self.n_real + torch.arange(N, device=idx.device, dtype=torch.long)
                               % SPARSE_PAD_ROWS)
        return _sparse_dedup_fixed(idx, vals, self._pad_slots)

    @torch.no_grad()
    def step(self):
        weight = self.weight
        grad = weight.grad
        if grad is None:
            return

        self.step_count += 1
        step = self.step_count
        lr = self.lr
        beta1, beta2 = self.betas
        eps = self.eps
        wd = self.weight_decay

        if grad.is_sparse and SPARSE_SYNC_FREE:
            indices, grad_raw = self._dedup_fixed(grad)
        elif grad.is_sparse:
            grad_c = grad.coalesce()
            # .clone() the two payload tensors out of the sparse COO wrapper. They are
            # already contiguous, so this is one straight copy each, and it is what makes
            # the compiled fast path below usable at all: a tensor whose storage still
            # belongs to a sparse tensor cannot be converted by Dynamo's meta /
            # fake-tensor machinery ("meta converter nyi"), which made the compiled core
            # bail out to eager on every step. The bf16 clone is still cheaper than the
            # eager .float() it replaces, because the fused kernel then reads n*d bf16
            # instead of n*d fp32.
            indices = grad_c.indices()[0].clone()
            grad_raw = grad_c.values().clone()
        else:
            nz = grad.abs().any(dim=-1)
            indices = nz.nonzero(as_tuple=False).view(-1)
            if indices.numel() == 0:
                weight.grad = None
                return
            grad_raw = grad[indices]

        if not (grad.is_sparse and SPARSE_SYNC_FREE) and indices.numel() == 0:
            weight.grad = None
            return

        if self._use_core:
            sc = self._sc
            sc["base_lr"].fill_(self.base_lr)
            sc["lrm"].fill_(self.lrm)
            sc["K"].fill_(self.freq_lr_K)
            sc["floor"].fill_(self.freq_lr_floor)
            sc["wd"].fill_(wd)
            sc["b1"].fill_(beta1)
            sc["b2"].fill_(beta2)
            sc["eps"].fill_(eps)
            sc["bias1"].fill_(1.0 - beta1 ** step)
            sc["bias2"].fill_(1.0 - beta2 ** step)
            try:
                if self.nomom:
                    m_new = None
                    v_new, p_new, cnt_new = _sparse_adamw_freq_factored_core_nomom(
                        weight.data, self.exp_avg_sq, self.cnt,
                        indices, grad_raw,
                        sc["base_lr"], sc["lrm"], sc["K"], sc["floor"], sc["wd"],
                        sc["b2"], sc["eps"], sc["bias2"])
                else:
                    m_new, v_new, p_new, cnt_new = _sparse_adamw_freq_factored_core(
                        weight.data, self.exp_avg, self.exp_avg_sq, self.cnt,
                        indices, grad_raw,
                        sc["base_lr"], sc["lrm"], sc["K"], sc["floor"], sc["wd"],
                        sc["b1"], sc["b2"], sc["eps"], sc["bias1"], sc["bias2"])
            except Exception as e:
                print(f"\n[SparseBigramAdamW] compiled core failed ({type(e).__name__}: {e}); "
                      f"falling back to the eager update for the rest of the run", flush=True)
                self._use_core = False
            else:
                if m_new is not None:
                    _scatter_rows_(self.exp_avg, indices, m_new)
                self.exp_avg_sq[indices] = v_new
                _scatter_rows_(weight.data, indices, p_new)
                self.cnt[indices] = cnt_new
                weight.grad = None
                return

        grad_vals = grad_raw.float()
        m = None if self.nomom else self.exp_avg[indices]
        v = self.exp_avg_sq[indices]
        p = weight.data[indices].float()

        if self.freq_lr:
            c = self.cnt[indices].float()                                                      
            conf = c / (c + self.freq_lr_K)                                                
            row_floor = 1.0 - conf * (1.0 - self.freq_lr_floor)                                  
            row_mult = row_floor.clamp_(min=self.lrm)                                     
            lr_vec = (self.base_lr * row_mult).unsqueeze(1)                 
            self.cnt[indices] += 1
        else:
            lr_vec = None

        if lr_vec is None:
            p.mul_(1.0 - lr * wd)
        else:
            p.mul_(1.0 - lr_vec * wd)                                                         
        m_new = grad_vals if self.nomom else beta1 * m + (1.0 - beta1) * grad_vals
        if self.factored_v:

            gsq = grad_vals.square().mean(dim=-1, keepdim=True)
            v_new = beta2 * v + (1.0 - beta2) * gsq
        else:
            v_new = beta2 * v + (1.0 - beta2) * grad_vals.square()
        bias1 = 1.0 if self.nomom else 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denom = (v_new / bias2).sqrt_().add_(eps)                                       
        if lr_vec is None:
            p.addcdiv_(m_new, denom, value=-(lr / bias1))
        elif self.factored_v:

            coef = lr_vec / denom                                           
            p.addcmul_(m_new, coef, value=-(1.0 / bias1))
        else:
            p.addcdiv_(lr_vec * m_new, denom, value=-(1.0 / bias1))                             

        if not self.nomom:
            self.exp_avg[indices]    = m_new.to(self.exp_avg.dtype)
        self.exp_avg_sq[indices] = v_new.to(self.exp_avg_sq.dtype)
        weight.data[indices] = p.to(weight.dtype)
        weight.grad = None

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

SOFTCAP = 15

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos_full, sin_signed):
    """Half-turn RoPE written without torch.cat.

    `cos_full` is [cos, cos] and `sin_signed` is [sin, -sin], both precomputed to
    the full head_dim, so every lane evaluates the SAME expression
    `x * cos_full + swap_halves(x) * sin_signed`, and the half-swap is a size-2
    `flip`, i.e. pure index arithmetic on the load. Per lane this sums exactly the
    same two products as the `cat` form (the upper half sums them in the other
    order, which IEEE addition permits; the only difference is which of the two
    products an FMA contraction keeps at full width, i.e. <= 1 bf16 ulp).

    The point is the backward. With the `cat`, autograd emits two
    `slice_backward`s, and inductor expands them into four mask-predicated
    branches that each re-load q and dq and re-derive the whole rms_norm-backward
    expression -- five loads of each 201 MB tensor in one kernel (measured:
    `triton_per_fused__fused_rms_norm_backward..._slice_slice_backward_15`,
    3.04 ms/step in a GB200 profile). The flip form has no masks, so the
    backward is one unpredicated pass.
    """
    assert x.ndim == 4
    B, T, H, D = x.shape
    x_swap = x.view(B, T, H, 2, D // 2).flip(-2).reshape(B, T, H, D)
    return x * cos_full + x_swap * sin_signed

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

        self.inject_ngram_v = layer_idx in NGRAM_VE_LAYERS
        self.ng_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if self.inject_ngram_v else None

        self.inject_bigram_v = layer_idx in BIGRAM_VE_LAYERS
        self.bg_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if self.inject_bigram_v else None

        self.attn_scale = nn.Parameter(torch.ones(self.n_head)) if USE_ATTN_SCALE else None

        # One fp32 logit per head. Must be float32 (the kernel accepts f16/bf16/f32
        # but this value is added to a running fp32 max/denominator, and it is also
        # the parameter an optimizer owns). The model is built on `meta` and
        # to_empty()'d, so the initial value is set in init_weights, not here.
        self.attn_sink = (
            nn.Parameter(torch.empty(self.n_head, dtype=torch.float32))
            if layer_idx in _attn_sink_layer_indices(config) else None
        )
        self.doc_local = layer_idx in _doc_local_layer_indices(config)
        # Per-head sigmoid output gate on the long-window layers only: each head can
        # scale its contribution per query token (the sink only lets a head decline
        # *within* the softmax). Same [n_head, 32] shape as ve_gate/ng_gate, so it
        # joins their Muon bucket; zero-init makes 2*sigmoid(0) = 1 exactly at init.
        # On the 6 short layers it cost ~2% of steps for almost no gain; reading the gate
        # from the full-width x, or an elementwise Linear(32, n_embd) gate, was worse.
        self.attn_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if ATTN_OUT_GATE and layer_idx in _long_layer_indices(config) else None
        )
        self.key_offset = (0 <= KEY_OFFSET_LO < self.head_dim // 2
                           and layer_idx in _layer_spec_indices(KEY_OFFSET_LAYERS, config))
        self.key_mix = (nn.Parameter(torch.empty(KEY_MIX_TAPS, self.head_dim // 2, dtype=torch.float32),
                                     requires_grad=not KEY_MIX_FIXED)
                        if KEY_MIX and layer_idx in _layer_spec_indices(KEY_MIX_LAYERS, config) else None)
        self.q_mix = (nn.Parameter(torch.empty(2, self.head_dim // 2, dtype=torch.float32))
                      if QMIX and layer_idx in _layer_spec_indices(QMIX_LAYERS, config) else None)

    def forward(self, x, ve, ng_v, bg_v, cos_sin, window_size, ng_scale=None, cu_seqlens=None):
        B, T, C = x.size()
        # Q and K are kept as two separate GEMMs on purpose. Fusing them -- one
        # F.linear against torch.cat((c_q.weight, c_k.weight), 0), one
        # apply_rotary_emb and one norm on the [B, T, 2H, Dh] view, then slice --
        # is algebraically exact (row-concatenated GEMMs with a shared LHS are one
        # GEMM; RoPE rotates each 2D subspace of a head independently; rms_norm over
        # the last dim of the [.., 2H, Dh] view is the concatenation of the two
        # halves' per-head norms) and it does halve the launches: 2 fwd GEMMs -> 1,
        # 2 dX GEMMs -> 1, 2 dW GEMMs -> 1, 18 RoPE/QK-norm kernels -> 9 per
        # direction. MEASURED in the training loop (GB200): 2.1% SLOWER per step,
        # not faster.
        # The reason is the boundary, and it is structural rather than a tuning
        # miss. flash-attn needs contiguous [B, T, H, Dh] q and k (its custom op's
        # register_fake describes the output with empty_like(q), so a strided q
        # makes the fake and the real strides disagree), so the two slices must be
        # cloned: ~0.5 ms forward. Worse, the backward of that slice pair is the
        # interleave -- two [B, T, H, Dh] grads written back into one [B, T, 2H, Dh]
        # buffer, 0.8 GB per layer, ~1 ms -- which is the same slice_backward
        # pathology apply_rotary_emb's comment below was written to avoid. The GEMM
        # merge saves less than the strided boundary costs, because the projections
        # are only ~11.3 ms of a 77.3 ms device step and already run at 1400+
        # TFLOP/s. Do not retry this without a way to hand flash-attn strided k.
        if USE_FP8_QKV and self.c_q.weight.shape[0] == self.c_k.weight.shape[0] == self.c_v.weight.shape[0]:
            q, k, v = _FP8QKVFn.apply(x, self.c_q.weight, self.c_k.weight, self.c_v.weight)
            q = q.view(B, T, self.n_head, self.head_dim)
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            v = v.view(B, T, self.n_kv_head, self.head_dim)
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        if ng_v is not None and self.ng_gate is not None:
                                                                                   
            ng_v = ng_v.view(B, T, self.n_kv_head, self.head_dim)
            ng_gate = 2 * torch.sigmoid(self.ng_gate(x[..., :self.ve_gate_channels]))
            if ng_scale is not None:
                # Per-channel scale for THIS site, folded into the multiply that the
                # dynamic per-head gate already performs: one fused elementwise kernel,
                # no extra pass over the [B, T, d] n-gram vector.
                v = v + (ng_gate.unsqueeze(-1) * ng_scale.view(self.n_kv_head, self.head_dim)) * ng_v
            else:
                v = v + ng_gate.unsqueeze(-1) * ng_v
        if bg_v is not None and self.bg_gate is not None:

            bg_v = bg_v.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bg_gate(x[..., :self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bg_v
        if self.q_mix is not None:
            # Same layout/broadcast as the key mix below, over the n_head query heads.
            aq = torch.cat((self.q_mix, self.q_mix), dim=-1)
            aq = aq.unsqueeze(1).expand(-1, self.n_head, self.head_dim)
            q = _KeyMixFn.apply(q, aq, QMIX_SUB)
        if self.key_mix is not None:
            # [taps, half] -> [taps, n_kv_head, head_dim]: RoPE pair members share a
            # value; broadcasting as (H, Dh) keeps grad(key_mix) a contiguous column
            # reduction (see attn_scale below).
            a = torch.cat((self.key_mix, self.key_mix), dim=-1)
            a = a.unsqueeze(1).expand(-1, self.n_kv_head, self.head_dim)
            if KEY_MIX_FIXED:
                # Constant coefficients: no grad(a) reduction, just the shifted-load
                # FMA, which inductor fuses into the rope+norm kernels.
                ab = a.to(k.dtype)
                k_mix = ab[0] * k
                for j in range(1, ab.size(0)):
                    k_mix = k_mix + ab[j] * F.pad(k[:, :-j], (0, 0, 0, 0, j, 0))
                k = k_mix
            else:
                k = _KeyMixFn.apply(k, a)
        elif self.key_offset:
            # Mask from arange (no buffer: the model is built on meta, to_empty()'d).
            half = self.head_dim // 2
            sel = (torch.arange(self.head_dim, device=k.device) % half) >= KEY_OFFSET_LO
            k_prev = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
            k = torch.where(sel, k_prev, k)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if self.attn_scale is not None:
            # Broadcast the per-head scale as (n_head, head_dim) instead of
            # (1, 1, n_head, 1): identical forward numbers, but grad(attn_scale)
            # becomes a column reduction over 768 contiguous channels followed by a
            # 768 -> 6 sum. With the (1,1,H,1) broadcast the reduction is over dims
            # [0, 1, 3] and inductor lowers it to a 102-way split whose split stride
            # (164483) is not a multiple of the 128-element row, so every element
            # pays a `// 128`, two `%` and a predicate: 1.74 ms/step in a GB200
            # profile (`triton_red_fused__to_copy_mul_sum_12`) for 403 MB of traffic
            # per layer that should cost ~50 us.
            q = q * self.attn_scale.to(q.dtype).unsqueeze(1).expand(self.n_head, self.head_dim)
        if self.doc_local and cu_seqlens is not None:
            # Within a document the window is redundant unless the document is
            # longer than it; -1 means "whole document prefix".
            ws = -1 if DOC_LOCAL_FULL_DOC else window_size
            y = flash_attn_doc_local(q, k, v, ws, self.attn_sink, cu_seqlens)
        else:
            y = flash_attn_func(q, k, v, causal=True, window_size=window_size,
                                learnable_sink=self.attn_sink)
        if self.attn_gate is not None:
            g = 2 * torch.sigmoid(self.attn_gate(x[..., :self.ve_gate_channels]))
            if USE_FP8_CPROJ:
                # Gate multiply, e4m3 casts and the FP8 c_proj GEMMs in one autograd fn
                # (see _FP8GatedProjFn): the casts ride the gate kernel's own pass.
                return _FP8GatedProjFn.apply(y, g, self.c_proj.weight).view(B, T, -1)
            y = y * g.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

# ---- MLP: FP8 e4m3/e5m2 tensorwise GEMMs for forward and dgrad -------------------
# MLP_HIDDEN is the fallback hidden width of every layer. MLP_HIDDEN_LIST (set in the
# configuration block at the top) overrides it per layer and is what this run uses.
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "4864"))
# Optional per-layer override of MLP_HIDDEN (comma-separated, one per layer).
MLP_HIDDEN_LIST = [int(v) for v in os.environ.get("MLP_HIDDEN_LIST", "").replace(" ", "").split(",") if v]

def mlp_hidden_for(layer_idx, n_layer):
    if not MLP_HIDDEN_LIST:
        return MLP_HIDDEN
    assert len(MLP_HIDDEN_LIST) == n_layer, f"MLP_HIDDEN_LIST has {len(MLP_HIDDEN_LIST)} entries for {n_layer} layers"
    return MLP_HIDDEN_LIST[layer_idx]
USE_FP8_MLP = int(os.environ.get("USE_FP8_MLP", "1"))
_F8_E4 = torch.float8_e4m3fn
_F8_E5 = torch.float8_e5m2
_F8_E4_MAX = 448.0
_F8_E5_MAX = 57344.0
# Scales for the big activation/gradient tensors come from the amax of a strided
# row sample (every FP8_AMAX_STRIDE-th token) of the CURRENT tensor, times a
# head-room margin. The sample is a ~1/16 read, so the cast itself stays a pure
# pointwise op that inductor fuses into the kernel that already produces the tensor
# (relu^2 forward, relu^2 backward) instead of a full amax pass over [M, hidden].
# Anything above the scaled range clamps to the format max rather than overflowing.
FP8_AMAX_STRIDE = int(os.environ.get("FP8_AMAX_STRIDE", "16"))
FP8_H_MARGIN = float(os.environ.get("FP8_H_MARGIN", "4.0"))    # e4m3: 2 of ~15 binades
FP8_G_MARGIN = float(os.environ.get("FP8_G_MARGIN", "16.0"))   # e5m2 has ~30 binades

def _f8(t, s, fmt, mx):
    return (t.float() * s).clamp(-mx, mx).to(fmt)

def _w8(w):
    s = _F8_E4_MAX / w.detach().abs().amax().float().clamp(min=1e-12)
    return _f8(w, s, _F8_E4, _F8_E4_MAX), s

# Two hand-written pointwise kernels for the [M, hidden] tensors. Inductor's own
# versions of these ran at 4.4 TB/s (forward cast) and chose TRANSPOSED stores for the
# bf16 wgrad operands (backward), 272 us and 583 us per layer at hidden 3072; as opaque
# ops with fixed row-major outputs they run at ~7 TB/s (170 us / 512 us), and cuBLAS
# consumes the row-major wgrad operands through its transpose flags.
import triton as _tr
import triton.language as _tl

@_tr.jit
def _relu2_f8_kernel(fc_ptr, s_ptr, h8_ptr, n, BLOCK: _tl.constexpr):
    offs = _tl.program_id(0).to(_tl.int64) * BLOCK + _tl.arange(0, BLOCK)
    m = offs < n
    s = _tl.load(s_ptr)
    r = _tl.maximum(_tl.load(fc_ptr + offs, mask=m, other=0.0).to(_tl.float32), 0.0)
    _tl.store(h8_ptr + offs, _tl.minimum(r * r * s, 448.0).to(_tl.float8e4nv), mask=m)

@_tr.jit
def _relu2_bwd_kernel(fc_ptr, dh_ptr, s_ptr, g_ptr, g8_ptr, h_ptr, n, BLOCK: _tl.constexpr):
    offs = _tl.program_id(0).to(_tl.int64) * BLOCK + _tl.arange(0, BLOCK)
    m = offs < n
    s = _tl.load(s_ptr)
    r = _tl.maximum(_tl.load(fc_ptr + offs, mask=m, other=0.0).to(_tl.float32), 0.0)
    g = _tl.load(dh_ptr + offs, mask=m, other=0.0).to(_tl.float32) * (2.0 * r)
    _tl.store(g_ptr + offs, g.to(_tl.bfloat16), mask=m)
    _tl.store(g8_ptr + offs, _tl.clamp(g * s, -57344.0, 57344.0).to(_tl.float8e5), mask=m)
    _tl.store(h_ptr + offs, (r * r).to(_tl.bfloat16), mask=m)

@torch.library.custom_op("fp8mlp::relu2_f8", mutates_args=())
def _relu2_f8_op(fc: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    fc = fc.contiguous()
    h8 = torch.empty(fc.shape, device=fc.device, dtype=torch.float8_e4m3fn)
    n = fc.numel()
    _relu2_f8_kernel[(_tr.cdiv(n, 4096),)](fc, s, h8, n, BLOCK=4096, num_warps=8)
    return h8

@_relu2_f8_op.register_fake
def _(fc, s):
    return torch.empty(fc.shape, device=fc.device, dtype=torch.float8_e4m3fn)

# ---- fused c_fc GEMM + relu^2 + e4m3 cast (forward) --------------------------------
# The forward used to be cuBLAS fc = bf16(a8 @ Wfc8^T * sa*sw) -> [M, H] bf16 in HBM,
# then _relu2_f8_kernel reading it back to write h8. This persistent, warp-specialised
# Triton TMA GEMM applies the same epilogue to the fp32 accumulator in registers --
# fc = bf16(acc * sa*sw), r = relu(fc), h8 = e4m3(min(r*r*s_h, 448)) -- so the bf16 fc
# never touches memory. Standalone on GB200 (M=131072, K=768): 0.334 vs 0.653 ms at
# H=3968 and 0.467 vs 0.938 ms at H=5888 for GEMM + amax + cast, i.e. the fused kernel
# costs what the cuBLAS GEMM alone did; h8 is bit-identical to the two-kernel path on
# the benchmark tensors. s_h needs the sampled amax of fc BEFORE the cast, so the
# sampled rows (every FP8_AMAX_STRIDE-th token, the same rows as before) come from a
# 1/16-size cuBLAS GEMM on a8[::S] -- the same per-row arithmetic as the full GEMM.
USE_FUSED_FC = int(os.environ.get("USE_FUSED_FC", "1"))
from triton.tools.tensor_descriptor import TensorDescriptor as _TDesc
_NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_FF_BM, _FF_BN, _FF_BK, _FF_STAGES, _FF_GROUP_M = 128, 256, 128, 3, 8

@_tr.jit
def _fc_relu2_f8_kernel(a_desc, b_desc, c_desc, scale_ptr, sh_ptr, M, N, K,
                        BM: _tl.constexpr, BN: _tl.constexpr, BK: _tl.constexpr,
                        GROUP_M: _tl.constexpr, NUM_SMS: _tl.constexpr, STAGES: _tl.constexpr):
    pid0 = _tl.program_id(0)
    num_m = _tl.cdiv(M, BM)
    num_n = _tl.cdiv(N, BN)
    num_tiles = num_m * num_n
    k_tiles = _tl.cdiv(K, BK)
    sc = _tl.load(scale_ptr)
    sh = _tl.load(sh_ptr)
    for tile in _tl.range(pid0, num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        group_id = tile // (GROUP_M * num_n)
        first_m = group_id * GROUP_M
        gs = min(num_m - first_m, GROUP_M)
        pm = first_m + (tile % gs)
        pn = (tile % (GROUP_M * num_n)) // gs
        om = pm * BM
        on = pn * BN
        acc = _tl.zeros((BM, BN), dtype=_tl.float32)
        for ki in _tl.range(k_tiles, num_stages=STAGES):
            a = a_desc.load([om, ki * BK])
            b = b_desc.load([on, ki * BK])
            acc = _tl.dot(a, b.T, acc)
        fc = (acc * sc).to(_tl.bfloat16).to(_tl.float32)
        r = _tl.maximum(fc, 0.0)
        c_desc.store([om, on], _tl.minimum(r * r * sh, 448.0).to(_tl.float8e4nv))

@torch.library.custom_op("fp8mlp::fc_relu2_f8", mutates_args=())
def _fc_relu2_f8_op(a8: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor,
                    sh: torch.Tensor) -> torch.Tensor:
    a8 = a8.contiguous()
    w8 = w8.contiguous()
    M, K = a8.shape
    N = w8.shape[0]
    out = torch.empty((M, N), device=a8.device, dtype=torch.float8_e4m3fn)
    a_d = _TDesc.from_tensor(a8, [_FF_BM, _FF_BK])
    b_d = _TDesc.from_tensor(w8, [_FF_BN, _FF_BK])
    c_d = _TDesc.from_tensor(out, [_FF_BM, _FF_BN])
    tiles = _tr.cdiv(M, _FF_BM) * _tr.cdiv(N, _FF_BN)
    _fc_relu2_f8_kernel[(min(_NUM_SMS, tiles),)](
        a_d, b_d, c_d, scale, sh, M, N, K, BM=_FF_BM, BN=_FF_BN, BK=_FF_BK,
        GROUP_M=_FF_GROUP_M, NUM_SMS=_NUM_SMS, STAGES=_FF_STAGES, num_warps=8)
    return out

@_fc_relu2_f8_op.register_fake
def _(a8, w8, scale, sh):
    return torch.empty((a8.shape[0], w8.shape[0]), device=a8.device, dtype=torch.float8_e4m3fn)

@torch.library.custom_op("fp8mlp::relu2_bwd", mutates_args=())
def _relu2_bwd_op(fc: torch.Tensor, dh: torch.Tensor, s: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fc = fc.contiguous()
    dh = dh.contiguous()
    g = torch.empty(fc.shape, device=fc.device, dtype=torch.bfloat16)
    g8 = torch.empty(fc.shape, device=fc.device, dtype=torch.float8_e5m2)
    h = torch.empty(fc.shape, device=fc.device, dtype=torch.bfloat16)
    n = fc.numel()
    _relu2_bwd_kernel[(_tr.cdiv(n, 2048),)](fc, dh, s, g, g8, h, n, BLOCK=2048, num_warps=4)
    return g, g8, h

@_relu2_bwd_op.register_fake
def _(fc, dh, s):
    return (torch.empty(fc.shape, device=fc.device, dtype=torch.bfloat16),
            torch.empty(fc.shape, device=fc.device, dtype=torch.float8_e5m2),
            torch.empty(fc.shape, device=fc.device, dtype=torch.bfloat16))

# ---- FP8 e4m3 weight-gradient GEMMs ------------------------------------------------
# cuBLASLt FP8 is TN-only, so wgrad dW = G^T @ X needs both [M, *] operands as
# transposed (K-major along tokens) e4m3 copies. These kernels write those copies
# directly: a 2-D tile is loaded row-major, scaled/cast, tl.trans'd and stored into
# the [N, M] output, so no bf16 transpose ever materialises.
USE_FP8_WGRAD = int(os.environ.get("USE_FP8_WGRAD", "1"))
FP8_WG_MARGIN = float(os.environ.get("FP8_WG_MARGIN", "8.0"))  # e4m3 grads: 3 binades headroom

@_tr.jit
def _tcast_e4_kernel(x_ptr, s_ptr, out_ptr, M, N, out_row0,
                     BM: _tl.constexpr, BN: _tl.constexpr):
    pid_m = _tl.program_id(0)
    pid_n = _tl.program_id(1)
    rm = (pid_m * BM + _tl.arange(0, BM)).to(_tl.int64)
    rn = (pid_n * BN + _tl.arange(0, BN)).to(_tl.int64)
    s = _tl.load(s_ptr)
    x = _tl.load(x_ptr + rm[:, None] * N + rn[None, :]).to(_tl.float32)
    y = _tl.clamp(x * s, -448.0, 448.0).to(_tl.float8e4nv)
    yt = _tl.trans(y)
    _tl.store(out_ptr + (out_row0 + rn[:, None]) * M + rm[None, :], yt)

@_tr.jit
def _relu2_bwd_t_kernel(fc_ptr, dh_ptr, s5_ptr, s4_ptr, sh_ptr, g8_ptr, gt_ptr, ht_ptr, M, N,
                        BM: _tl.constexpr, BN: _tl.constexpr):
    pid_m = _tl.program_id(0)
    pid_n = _tl.program_id(1)
    rm = (pid_m * BM + _tl.arange(0, BM)).to(_tl.int64)
    rn = (pid_n * BN + _tl.arange(0, BN)).to(_tl.int64)
    s5 = _tl.load(s5_ptr)
    s4 = _tl.load(s4_ptr)
    sh = _tl.load(sh_ptr)
    off = rm[:, None] * N + rn[None, :]
    r = _tl.maximum(_tl.load(fc_ptr + off).to(_tl.float32), 0.0)
    g = _tl.load(dh_ptr + off).to(_tl.float32) * (2.0 * r)
    _tl.store(g8_ptr + off, _tl.clamp(g * s5, -57344.0, 57344.0).to(_tl.float8e5))
    toff = rn[:, None] * M + rm[None, :]
    gt = _tl.clamp(g * s4, -448.0, 448.0).to(_tl.float8e4nv)
    _tl.store(gt_ptr + toff, _tl.trans(gt))
    ht = _tl.minimum(r * r * sh, 448.0).to(_tl.float8e4nv)
    _tl.store(ht_ptr + toff, _tl.trans(ht))

_TB_M, _TB_N = 64, 64
# Retiled (standalone on GB200, M=131072): _tcast_e4_kernel 64x128 is 47.7 vs 51.5 us
# (bf16 in) and 41.8 vs 46.2 us (e4m3 in); _relu2_bwd_t_kernel 32x128 is 715 vs 766 us
# at H=4608. Outputs are bit-identical (pure re-tiling of an elementwise cast/transpose).
_TC_M, _TC_N = 64, 128
_RB_M, _RB_N = 32, 128

@torch.library.custom_op("fp8mlp::tcast_e4", mutates_args=())
def _tcast_e4_op(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    assert M % _TC_M == 0 and N % _TC_N == 0
    out = torch.empty((N, M), device=x.device, dtype=torch.float8_e4m3fn)
    _tcast_e4_kernel[(M // _TC_M, N // _TC_N)](x, s, out, M, N, 0, BM=_TC_M, BN=_TC_N, num_warps=4)
    return out

@_tcast_e4_op.register_fake
def _(x, s):
    M, N = x.shape
    return torch.empty((N, M), device=x.device, dtype=torch.float8_e4m3fn)

@torch.library.custom_op("fp8mlp::tcast3_e4", mutates_args=())
def _tcast3_e4_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    M, D = a.shape
    assert M % _TC_M == 0 and D % _TC_N == 0
    out = torch.empty((3 * D, M), device=a.device, dtype=torch.float8_e4m3fn)
    for i, t in enumerate((a, b, c)):
        _tcast_e4_kernel[(M // _TC_M, D // _TC_N)](t, s, out, M, D, i * D, BM=_TC_M, BN=_TC_N, num_warps=4)
    return out

@_tcast3_e4_op.register_fake
def _(a, b, c, s):
    M, D = a.shape
    return torch.empty((3 * D, M), device=a.device, dtype=torch.float8_e4m3fn)

# One read, two casts: the e5m2 row-major copy that feeds a dgrad GEMM and the e4m3
# transposed copy that feeds the matching wgrad GEMM come from the same [M, D] bf16
# gradient. They used to be two kernels (an inductor _f8 / _cat3_e5 pass and a
# _tcast_e4 pass), each reading the whole gradient; this writes both outputs from a
# single read. Same arithmetic per element (fp32 x*s, clamp, round-to-nearest cast),
# so both outputs are bit-identical to the two-kernel versions.
USE_DUAL_CAST = int(os.environ.get("USE_DUAL_CAST", "1"))

@_tr.jit
def _dual_cast_kernel(x_ptr, s5_ptr, s4_ptr, o5_ptr, ot_ptr, M, D, ld5, col0,
                      BM: _tl.constexpr, BN: _tl.constexpr):
    pid_m = _tl.program_id(0)
    pid_n = _tl.program_id(1)
    rm = (pid_m * BM + _tl.arange(0, BM)).to(_tl.int64)
    rn = (pid_n * BN + _tl.arange(0, BN)).to(_tl.int64)
    s5 = _tl.load(s5_ptr)
    s4 = _tl.load(s4_ptr)
    x = _tl.load(x_ptr + rm[:, None] * D + rn[None, :]).to(_tl.float32)
    _tl.store(o5_ptr + rm[:, None] * ld5 + (col0 + rn[None, :]),
              _tl.clamp(x * s5, -57344.0, 57344.0).to(_tl.float8e5))
    yt = _tl.trans(_tl.clamp(x * s4, -448.0, 448.0).to(_tl.float8e4nv))
    _tl.store(ot_ptr + (col0 + rn[:, None]) * M + rm[None, :], yt)

@torch.library.custom_op("fp8mlp::dual_cast", mutates_args=())
def _dual_cast_op(x: torch.Tensor, s5: torch.Tensor, s4: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    M, D = x.shape
    assert M % _TC_M == 0 and D % _TC_N == 0
    o5 = torch.empty((M, D), device=x.device, dtype=torch.float8_e5m2)
    ot = torch.empty((D, M), device=x.device, dtype=torch.float8_e4m3fn)
    _dual_cast_kernel[(M // _TC_M, D // _TC_N)](x, s5, s4, o5, ot, M, D, D, 0,
                                                BM=_TC_M, BN=_TC_N, num_warps=4)
    return o5, ot

@_dual_cast_op.register_fake
def _(x, s5, s4):
    M, D = x.shape
    return (torch.empty((M, D), device=x.device, dtype=torch.float8_e5m2),
            torch.empty((D, M), device=x.device, dtype=torch.float8_e4m3fn))

@torch.library.custom_op("fp8mlp::dual_cast3", mutates_args=())
def _dual_cast3_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s5: torch.Tensor,
                   s4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    M, D = a.shape
    assert M % _TC_M == 0 and D % _TC_N == 0
    o5 = torch.empty((M, 3 * D), device=a.device, dtype=torch.float8_e5m2)
    ot = torch.empty((3 * D, M), device=a.device, dtype=torch.float8_e4m3fn)
    for i, t in enumerate((a, b, c)):
        _dual_cast_kernel[(M // _TC_M, D // _TC_N)](t, s5, s4, o5, ot, M, D, 3 * D, i * D,
                                                    BM=_TC_M, BN=_TC_N, num_warps=4)
    return o5, ot

@_dual_cast3_op.register_fake
def _(a, b, c, s5, s4):
    M, D = a.shape
    return (torch.empty((M, 3 * D), device=a.device, dtype=torch.float8_e5m2),
            torch.empty((3 * D, M), device=a.device, dtype=torch.float8_e4m3fn))

@torch.library.custom_op("fp8mlp::relu2_bwd_t", mutates_args=())
def _relu2_bwd_t_op(fc: torch.Tensor, dh: torch.Tensor, s5: torch.Tensor, s4: torch.Tensor,
                    sh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fc = fc.contiguous()
    dh = dh.contiguous()
    M, N = fc.shape
    assert M % _RB_M == 0 and N % _RB_N == 0
    g8 = torch.empty((M, N), device=fc.device, dtype=torch.float8_e5m2)
    gt = torch.empty((N, M), device=fc.device, dtype=torch.float8_e4m3fn)
    ht = torch.empty((N, M), device=fc.device, dtype=torch.float8_e4m3fn)
    _relu2_bwd_t_kernel[(M // _RB_M, N // _RB_N)](fc, dh, s5, s4, sh, g8, gt, ht, M, N,
                                                  BM=_RB_M, BN=_RB_N, num_warps=4)
    return g8, gt, ht

@_relu2_bwd_t_op.register_fake
def _(fc, dh, s5, s4, sh):
    M, N = fc.shape
    return (torch.empty((M, N), device=fc.device, dtype=torch.float8_e5m2),
            torch.empty((N, M), device=fc.device, dtype=torch.float8_e4m3fn),
            torch.empty((N, M), device=fc.device, dtype=torch.float8_e4m3fn))

# relu^2 backward from the e4m3 h8 the forward already made, instead of the bf16 fc:
# r = sqrt(h8 / s_h) recovers relu(fc) to e4m3 precision (and is exactly the h the
# forward GEMM consumed), and hT8 is h8 itself transposed -- bit-identical to the
# old path's hT8. Reads 3 bytes/element instead of 4 and lets the forward drop fc.
@_tr.jit
def _relu2_bwd_h8_kernel(h8_ptr, dh_ptr, s5_ptr, s4_ptr, sh_ptr, g8_ptr, gt_ptr, ht_ptr, M, N,
                         BM: _tl.constexpr, BN: _tl.constexpr):
    pid_m = _tl.program_id(0)
    pid_n = _tl.program_id(1)
    rm = (pid_m * BM + _tl.arange(0, BM)).to(_tl.int64)
    rn = (pid_n * BN + _tl.arange(0, BN)).to(_tl.int64)
    s5 = _tl.load(s5_ptr)
    s4 = _tl.load(s4_ptr)
    sh = _tl.load(sh_ptr)
    off = rm[:, None] * N + rn[None, :]
    h8 = _tl.load(h8_ptr + off)
    r = _tl.sqrt(h8.to(_tl.float32) / sh)
    g = _tl.load(dh_ptr + off).to(_tl.float32) * (2.0 * r)
    _tl.store(g8_ptr + off, _tl.clamp(g * s5, -57344.0, 57344.0).to(_tl.float8e5))
    toff = rn[:, None] * M + rm[None, :]
    gt = _tl.clamp(g * s4, -448.0, 448.0).to(_tl.float8e4nv)
    _tl.store(gt_ptr + toff, _tl.trans(gt))
    _tl.store(ht_ptr + toff, _tl.trans(h8))

_RH_M, _RH_N = 64, 128

@torch.library.custom_op("fp8mlp::relu2_bwd_h8", mutates_args=())
def _relu2_bwd_h8_op(h8: torch.Tensor, dh: torch.Tensor, s5: torch.Tensor, s4: torch.Tensor,
                     sh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h8 = h8.contiguous()
    dh = dh.contiguous()
    M, N = h8.shape
    assert M % _RH_M == 0 and N % _RH_N == 0
    g8 = torch.empty((M, N), device=h8.device, dtype=torch.float8_e5m2)
    gt = torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn)
    ht = torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn)
    _relu2_bwd_h8_kernel[(M // _RH_M, N // _RH_N)](h8, dh, s5, s4, sh, g8, gt, ht, M, N,
                                                   BM=_RH_M, BN=_RH_N, num_warps=8)
    return g8, gt, ht

@_relu2_bwd_h8_op.register_fake
def _(h8, dh, s5, s4, sh):
    M, N = h8.shape
    return (torch.empty((M, N), device=h8.device, dtype=torch.float8_e5m2),
            torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn),
            torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn))

# ---- fused dgrad-proj GEMM + relu^2 backward (backward) ----------------------------
# Mirror of the forward fusion: dh = bf16(dy8 @ Wp8 * s_dy^-1 * s_wp^-1) stays in the
# fp32 TMEM accumulator, and the _relu2_bwd_h8 epilogue (r = sqrt(h8/s_h), g = dh*2r,
# g8 e5m2, gT8 e4m3 transposed, hT8 = h8 transposed) is applied to it, so the bf16 dh
# [M, H] never touches HBM. Plain-Triton versions of this kernel (persistent + auto
# warp-specialisation, or one tile per CTA) ran 1.4-2.3 ms at H=4096 because the
# h8 load and the three stores serialise with the MMA; this is a Gluon kernel with
# explicit partitions -- a 1-warp TMA loader (dy8/Wp^T ring + double-buffered h8
# tile), a 1-warp tcgen05 issuer into a double-buffered TMEM accumulator, and the
# 8-warp epilogue -- so tile i's epilogue overlaps tile i+1's GEMM. Standalone on
# GB200 (M=131072, K=768), against cuBLAS dh + _relu2_bwd_h8: 0.670 vs 0.831 ms at
# H=4096, 1.016 vs 1.293 ms at H=6272; g8 / gT8 / hT8 byte-identical. The g scales
# need the sampled amax of g before the cast; like the forward, those rows come from
# a 1/16-size cuBLAS GEMM on dy8[::S] (the same per-row arithmetic as the full GEMM).
USE_FUSED_DH = int(os.environ.get("USE_FUSED_DH", "1"))
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as _GDesc
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, tcgen05_commit,
    mbarrier, tma, fence_async_shared)

@gluon.jit
def _tile_coords(tile, num_m, num_n, BM: gl.constexpr, BN: gl.constexpr, GROUP_M: gl.constexpr):
    group_id = tile // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    gs = gl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (tile % gs)
    pn = (tile % (GROUP_M * num_n)) // gs
    return pm * BM, pn * BN


@gluon.jit
def _ld_part(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, h_ready, h_empty, M, N, K,
             BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr, STAGES: gl.constexpr):
    num_m = gl.cdiv(M, BM)
    num_n = gl.cdiv(N, BN)
    k_tiles = gl.cdiv(K, BK)
    c = 0
    t = 0
    for tile in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        om, on = _tile_coords(tile, num_m, num_n, BM, BN, GROUP_M)
        for ki in range(k_tiles):
            s = c % STAGES
            ph = (c // STAGES) & 1
            mbarrier.wait(empty.index(s), ph ^ 1)
            mbarrier.expect(ready.index(s), a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [om, ki * BK], ready.index(s), a_s.index(s))
            tma.async_copy_global_to_shared(b_desc, [on, ki * BK], ready.index(s), b_s.index(s))
            c += 1
        hb = t % 2
        hph = (t // 2) & 1
        mbarrier.wait(h_empty.index(hb), hph ^ 1)
        mbarrier.expect(h_ready.index(hb), h_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(h_desc, [om, on], h_ready.index(hb), h_s.index(hb))
        t += 1


@gluon.jit
def _mma_part(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, acc_t, acc_ready, acc_empty, M, N, K,
              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    num_m = gl.cdiv(M, BM)
    num_n = gl.cdiv(N, BN)
    k_tiles = gl.cdiv(K, BK)
    c = 0
    t = 0
    for tile in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        ab = t % 2
        aph = (t // 2) & 1
        mbarrier.wait(acc_empty.index(ab), aph ^ 1)
        acc = acc_t.index(ab)
        for ki in range(k_tiles):
            s = c % STAGES
            ph = (c // STAGES) & 1
            mbarrier.wait(ready.index(s), ph)
            tcgen05_mma(a_s.index(s), b_s.index(s).permute((1, 0)), acc, use_acc=ki > 0,
                        mbarriers=[empty.index(s)])
            c += 1
        tcgen05_commit(acc_ready.index(ab))
        t += 1


# Packed sm_100 epilogue math (2 elements per instruction via f32x2), bit-identical to
#   dh = bf16(acc*sc); r = sqrt.approx(h * rcp(sh)); g = dh*(r+r)
#   o5 = max(g*s5, -57344) -> e5m2 satfinite; o4 = max(g*s4, -448) -> e4m3 satfinite
# h * rcp(sh) == div.full(h, sh) for normal sh (s_h <= 4.48e32); the upper clamps are
# dropped because cvt.satfinite saturates identically (NaN -> -lim via the max first).
_DH_EPI_ASM = gl.constexpr("""{
.reg .b64 A, S, D, H, R, G, T;
.reg .b32 b, d0, d1, r0, r1, u0, u1;
mov.b64 A, {$4, $5};
mov.b64 S, {$8, $9};
mul.rn.f32x2 D, A, S;
mov.b64 {d0, d1}, D;
cvt.rn.bf16x2.f32 b, d1, d0;
shl.b32 d0, b, 16;
and.b32 d1, b, 0xffff0000;
mov.b64 H, {$6, $7};
mov.b64 S, {$10, $11};
mul.rn.f32x2 H, H, S;
mov.b64 {r0, r1}, H;
sqrt.approx.ftz.f32 r0, r0;
sqrt.approx.ftz.f32 r1, r1;
mov.b64 R, {r0, r1};
add.rn.f32x2 R, R, R;
mov.b64 D, {d0, d1};
mul.rn.f32x2 G, D, R;
mov.b64 S, {$12, $13};
mul.rn.f32x2 T, G, S;
mov.b64 {u0, u1}, T;
max.f32 $0, u0, 0fC7600000;
max.f32 $1, u1, 0fC7600000;
mov.b64 S, {$14, $15};
mul.rn.f32x2 T, G, S;
mov.b64 {u0, u1}, T;
max.f32 $2, u0, 0fC3E00000;
max.f32 $3, u1, 0fC3E00000;
}""")


@gluon.jit
def _epi_part(g8_desc, gt_desc, ht_desc, h_s, h_ready, h_empty, acc_t, acc_ready, acc_empty,
              o5_s, ot_s, oh_s, sc, s5, s4, sh, M, N,
              BM: gl.constexpr, BN: gl.constexpr, GROUP_M: gl.constexpr, REG: gl.constexpr):
    num_m = gl.cdiv(M, BM)
    num_n = gl.cdiv(N, BN)
    rsh = 1.0 / sh
    sc_t = gl.full([BM, BN], sc, gl.float32, REG)
    rsh_t = gl.full([BM, BN], rsh, gl.float32, REG)
    s5_t = gl.full([BM, BN], s5, gl.float32, REG)
    s4_t = gl.full([BM, BN], s4, gl.float32, REG)
    t = 0
    for tile in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        om, on = _tile_coords(tile, num_m, num_n, BM, BN, GROUP_M)
        ab = t % 2
        aph = (t // 2) & 1
        mbarrier.wait(acc_ready.index(ab), aph)
        acc = acc_t.index(ab).load(REG)
        mbarrier.wait(h_ready.index(ab), aph)
        h8 = h_s.index(ab).load(REG)
        mbarrier.arrive(acc_empty.index(ab), count=1)  # acc is in registers: let the next MMA start
        f5, f4 = gl.inline_asm_elementwise(
            _DH_EPI_ASM, "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r",
            [acc, h8.to(gl.float32), sc_t, rsh_t, s5_t, s4_t],
            dtype=(gl.float32, gl.float32), is_pure=True, pack=2)
        o5 = f5.to(gl.float8e5)
        o4 = f4.to(gl.float8e4nv)
        tma.store_wait(0)
        o5_s.store(o5)
        ot_s.permute((1, 0)).store(o4)
        oh_s.permute((1, 0)).store(h8)
        fence_async_shared()
        tma.async_copy_shared_to_global(g8_desc, [om, on], o5_s)
        tma.async_copy_shared_to_global(gt_desc, [on, om], ot_s)
        tma.async_copy_shared_to_global(ht_desc, [on, om], oh_s)
        mbarrier.arrive(h_empty.index(ab), count=1)
        t += 1
    tma.store_wait(0)


@gluon.jit
def _dh_relu2_bwd_gl(a_desc, b_desc, h_desc, g8_desc, gt_desc, ht_desc, scale_ptr, s5_ptr, s4_ptr, sh_ptr,
                     M, N, K, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
                     STAGES: gl.constexpr, REG: gl.constexpr):
    a_s = gl.allocate_shared_memory(a_desc.dtype, [STAGES, BM, BK], a_desc.layout)
    b_s = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BN, BK], b_desc.layout)
    h_s = gl.allocate_shared_memory(h_desc.dtype, [2, BM, BN], h_desc.layout)
    o5_s = gl.allocate_shared_memory(g8_desc.dtype, [BM, BN], g8_desc.layout)
    ot_s = gl.allocate_shared_memory(gt_desc.dtype, [BN, BM], gt_desc.layout)
    oh_s = gl.allocate_shared_memory(ht_desc.dtype, [BN, BM], ht_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    h_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    h_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(h_ready.index(i), count=1)
        mbarrier.init(h_empty.index(i), count=1)
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_empty.index(i), count=1)
    fence_async_shared()
    acc_t = allocate_tensor_memory(gl.float32, [2, BM, BN], TensorMemoryLayout([BM, BN], unpacked=True))
    sc = gl.load(scale_ptr)
    s5 = gl.load(s5_ptr)
    s4 = gl.load(s4_ptr)
    sh = gl.load(sh_ptr)
    gl.warp_specialize(
        (g8_desc, gt_desc, ht_desc, h_s, h_ready, h_empty, acc_t, acc_ready, acc_empty, o5_s, ot_s, oh_s,
         sc, s5, s4, sh, M, N, BM, BN, GROUP_M, REG),
        _epi_part,
        (a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, h_ready, h_empty, acc_t, acc_ready, acc_empty,
         M, N, K, BM, BN, BK, GROUP_M, STAGES),
        [_ld_wrap, _mma_wrap], [1, 1], [24, 24])


@gluon.jit
def _ld_wrap(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, h_ready, h_empty, acc_t, acc_ready, acc_empty,
             M, N, K, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
             STAGES: gl.constexpr):
    _ld_part(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, h_ready, h_empty, M, N, K,
             BM, BN, BK, GROUP_M, STAGES)


@gluon.jit
def _mma_wrap(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, h_ready, h_empty, acc_t, acc_ready, acc_empty,
              M, N, K, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
              STAGES: gl.constexpr):
    _mma_part(a_desc, b_desc, h_desc, a_s, b_s, h_s, ready, empty, acc_t, acc_ready, acc_empty, M, N, K,
              BM, BN, BK, STAGES)


_DH_GL_LAYOUTS = {}

def dh_relu2_bwd_gl(dy8, wpT8, h8, scale, s5, s4, sh, BM=128, BN=128, BK=128, STAGES=4, GROUP_M=12, NW=8):
    M, K = dy8.shape
    N = wpT8.shape[0]
    g8 = torch.empty((M, N), device=dy8.device, dtype=torch.float8_e5m2)
    gt = torch.empty((N, M), device=dy8.device, dtype=torch.float8_e4m3fn)
    ht = torch.empty((N, M), device=dy8.device, dtype=torch.float8_e4m3fn)
    key = (BM, BN, BK, NW)
    lay = _DH_GL_LAYOUTS.get(key)
    if lay is None:  # layout objects are pure functions of the tile shape: build them once
        L = gl.NVMMASharedLayout.get_default_for
        lay = _DH_GL_LAYOUTS[key] = (L([BM, BK], gl.float8e5), L([BN, BK], gl.float8e4nv),
                                     L([BM, BN], gl.float8e4nv), L([BM, BN], gl.float8e5),
                                     L([BN, BM], gl.float8e4nv),
                                     get_tmem_32x32b_reg_layout(BM, BN, [BM, BN], NW))
    la, lb, lh, l5, lt, reg = lay
    a_d = _GDesc(dy8, [M, K], [K, 1], [BM, BK], la)
    b_d = _GDesc(wpT8, [N, K], [K, 1], [BN, BK], lb)
    h_d = _GDesc(h8, [M, N], [N, 1], [BM, BN], lh)
    g8_d = _GDesc(g8, [M, N], [N, 1], [BM, BN], l5)
    gt_d = _GDesc(gt, [N, M], [M, 1], [BN, BM], lt)
    ht_d = _GDesc(ht, [N, M], [M, 1], [BN, BM], lt)
    tiles = _tr.cdiv(M, BM) * _tr.cdiv(N, BN)
    _dh_relu2_bwd_gl[(min(_NUM_SMS, tiles),)](a_d, b_d, h_d, g8_d, gt_d, ht_d, scale, s5, s4, sh, M, N, K,
                                         BM, BN, BK, GROUP_M, STAGES, reg, num_warps=NW)
    return g8, gt, ht


@torch.library.custom_op("fp8mlp::dh_relu2_bwd", mutates_args=())
def _dh_relu2_bwd_op(dy8: torch.Tensor, wpT8: torch.Tensor, h8: torch.Tensor,
                     scale: torch.Tensor, s5: torch.Tensor, s4: torch.Tensor,
                     sh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # dy8 [M, K] e5m2, wpT8 [N, K] e4m3 (Wproj^T, K-major), h8 [M, N] e4m3
    dy8, wpT8, h8 = dy8.contiguous(), wpT8.contiguous(), h8.contiguous()
    assert dy8.shape[0] % 128 == 0 and wpT8.shape[0] % 128 == 0 and dy8.shape[1] % 128 == 0
    return dh_relu2_bwd_gl(dy8, wpT8, h8, scale, s5, s4, sh)

@_dh_relu2_bwd_op.register_fake
def _(dy8, wpT8, h8, scale, s5, s4, sh):
    M, N = h8.shape
    return (torch.empty((M, N), device=h8.device, dtype=torch.float8_e5m2),
            torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn),
            torch.empty((N, M), device=h8.device, dtype=torch.float8_e4m3fn))

class _FP8MLPFn(torch.autograd.Function):
    """y = relu(a @ Wfc^T)^2 @ Wproj^T.

    FP8 for the two forward GEMMs and the two data-gradient GEMMs (e4m3 activations
    and weights, e5m2 gradients, fp32 accumulate, bf16 out). The two weight-gradient
    GEMMs stay bf16: FP8 wgrads measurably raised the training loss while FP8
    fwd+dgrad did not, and FP8 wgrads would also need transposed copies of
    both [M, *] operands (cuBLASLt FP8 is TN-only).
    a is the rms-normed (weightless) residual, so |a| <= sqrt(d): the static scale
    floor(448/sqrt(d)) is exact and needs no amax at all.
    """
    @staticmethod
    def forward(ctx, a, wfc, wproj):
        d = a.shape[-1]
        lead = a.shape[:-1]
        a2 = a.reshape(-1, d)
        s_a = float(_F8_E4_MAX / math.sqrt(d) // 1)
        inv_a = torch.full((), 1.0 / s_a, device=a.device, dtype=torch.float32)
        a8 = _f8(a2, s_a, _F8_E4, _F8_E4_MAX)
        wfc8, s_wfc = _w8(wfc)
        if USE_FUSED_FC:
            inv_wfc = 1.0 / s_wfc
            fc_s = torch._scaled_mm(a8[::FP8_AMAX_STRIDE].contiguous(), wfc8.t(), scale_a=inv_a,
                                    scale_b=inv_wfc, out_dtype=torch.bfloat16)
            rmax = F.relu(fc_s.amax().float())
            s_h = _F8_E4_MAX / (FP8_H_MARGIN * rmax * rmax).clamp(min=1e-30)
            h8 = torch.ops.fp8mlp.fc_relu2_f8(a8, wfc8, (inv_a * inv_wfc).reshape(1),
                                              s_h.reshape(1))
        else:
            fc = torch._scaled_mm(a8, wfc8.t(), scale_a=inv_a, scale_b=1.0 / s_wfc,
                                  out_dtype=torch.bfloat16)
            rmax = F.relu(fc[::FP8_AMAX_STRIDE].amax().float())
            s_h = _F8_E4_MAX / (FP8_H_MARGIN * rmax * rmax).clamp(min=1e-30)
            h8 = torch.ops.fp8mlp.relu2_f8(fc, s_h.reshape(1))
        wp8, s_wp = _w8(wproj)
        y = torch._scaled_mm(h8, wp8.t(), scale_a=1.0 / s_h, scale_b=1.0 / s_wp,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(a8, inv_a, h8, wfc8, s_wfc, wp8, s_wp, s_h)
        ctx.a_dtype = a.dtype
        ctx.lead = lead
        ctx.w_dtypes = (wfc.dtype, wproj.dtype)
        return y.view(*lead, -1)

    @staticmethod
    def backward(ctx, dy):
        a8, inv_a, h8, wfc8, s_wfc, wp8, s_wp, s_h = ctx.saved_tensors
        S = FP8_AMAX_STRIDE
        dy2 = dy.reshape(-1, dy.shape[-1])
        dy_smax = dy2[::S].abs().amax().float()
        s_dy = _F8_E5_MAX / (FP8_G_MARGIN * dy_smax).clamp(min=1e-30)
        if USE_FP8_WGRAD and USE_DUAL_CAST:
            s_dy4 = _F8_E4_MAX / (FP8_WG_MARGIN * dy_smax).clamp(min=1e-30)
            dy8, dyT8 = torch.ops.fp8mlp.dual_cast(dy2, s_dy.reshape(1), s_dy4.reshape(1))
            return _FP8MLPFn._backward_fp8wg(ctx, a8, inv_a, h8, wfc8, s_wfc, wp8, s_wp, s_h,
                                             dy2, dy_smax, s_dy, dy8, s_dy4, dyT8)
        dy8 = _f8(dy2, s_dy, _F8_E5, _F8_E5_MAX)
        if USE_FP8_WGRAD:
            return _FP8MLPFn._backward_fp8wg(ctx, a8, inv_a, h8, wfc8, s_wfc, wp8, s_wp, s_h,
                                             dy2, dy_smax, s_dy, dy8)
        # dgrad proj: dh = dy @ Wproj
        dh = torch._scaled_mm(dy8, wp8.t().contiguous().t(), scale_a=1.0 / s_dy,
                              scale_b=1.0 / s_wp, out_dtype=torch.bfloat16)
        fc = (h8.float() / s_h).sqrt().to(torch.bfloat16)  # relu(fc) rebuilt from h8
        g_smax = (dh[::S].float() * (2.0 * F.relu(fc[::S]).float())).abs().amax()
        s_g = _F8_E5_MAX / (FP8_G_MARGIN * g_smax).clamp(min=1e-30)
        g, g8, h = torch.ops.fp8mlp.relu2_bwd(fc, dh, s_g.reshape(1))
        # dgrad fc: da = g @ Wfc
        da = torch._scaled_mm(g8, wfc8.t().contiguous().t(), scale_a=1.0 / s_g,
                              scale_b=1.0 / s_wfc, out_dtype=torch.bfloat16)
        # weight gradients in bf16, against the same (e4m3-rounded) a the forward used
        dwp = torch.mm(dy2.to(torch.bfloat16).t(), h)
        dwfc = torch.mm(g.t(), (a8.float() * inv_a).to(torch.bfloat16))
        return (da.to(ctx.a_dtype).view(*ctx.lead, -1), dwfc.to(ctx.w_dtypes[0]),
                dwp.to(ctx.w_dtypes[1]))

    @staticmethod
    def _backward_fp8wg(ctx, a8, inv_a, h8, wfc8, s_wfc, wp8, s_wp, s_h, dy2, dy_smax, s_dy, dy8,
                        s_dy4=None, dyT8=None):
        S = FP8_AMAX_STRIDE
        r_s = (h8[::S].float() / s_h).sqrt()
        if USE_FUSED_DH:
            wpT8 = wp8.t().contiguous()
            dh_s = torch._scaled_mm(dy8[::S].contiguous(), wpT8.t(), scale_a=1.0 / s_dy,
                                    scale_b=1.0 / s_wp, out_dtype=torch.bfloat16)
            g_smax = (dh_s.float() * (2.0 * r_s)).abs().amax()
        else:
            dh = torch._scaled_mm(dy8, wp8.t().contiguous().t(), scale_a=1.0 / s_dy,
                                  scale_b=1.0 / s_wp, out_dtype=torch.bfloat16)
            g_smax = (dh[::S].float() * (2.0 * r_s)).abs().amax()
        s_g = _F8_E5_MAX / (FP8_G_MARGIN * g_smax).clamp(min=1e-30)
        s_g4 = _F8_E4_MAX / (FP8_WG_MARGIN * g_smax).clamp(min=1e-30)
        if USE_FUSED_DH:
            g8, gT8, hT8 = torch.ops.fp8mlp.dh_relu2_bwd(
                dy8, wpT8, h8, ((1.0 / s_dy) * (1.0 / s_wp)).reshape(1), s_g.reshape(1),
                s_g4.reshape(1), s_h.reshape(1))
        else:
            g8, gT8, hT8 = torch.ops.fp8mlp.relu2_bwd_h8(h8, dh, s_g.reshape(1), s_g4.reshape(1),
                                                         s_h.reshape(1))
        da = torch._scaled_mm(g8, wfc8.t().contiguous().t(), scale_a=1.0 / s_g,
                              scale_b=1.0 / s_wfc, out_dtype=torch.bfloat16)
        # wgrad proj: dWp[d, H] = dy^T @ h  (e4m3 x e4m3, both token-major copies)
        if dyT8 is None:
            s_dy4 = _F8_E4_MAX / (FP8_WG_MARGIN * dy_smax).clamp(min=1e-30)
            dyT8 = torch.ops.fp8mlp.tcast_e4(dy2, s_dy4.reshape(1))
        dwp = torch._scaled_mm(dyT8, hT8.t(), scale_a=1.0 / s_dy4, scale_b=1.0 / s_h,
                               out_dtype=torch.bfloat16)
        # wgrad fc: dWfc[H, d] = g^T @ a  (a8 is already e4m3 at scale 1/inv_a)
        aT8 = torch.ops.fp8mlp.tcast_e4(a8, torch.ones(1, device=a8.device, dtype=torch.float32))
        dwfc = torch._scaled_mm(gT8, aT8.t(), scale_a=1.0 / s_g4, scale_b=inv_a,
                                out_dtype=torch.bfloat16)
        return (da.to(ctx.a_dtype).view(*ctx.lead, -1), dwfc.to(ctx.w_dtypes[0]),
                dwp.to(ctx.w_dtypes[1]))

@_tr.jit
def _cat3_e5_kernel(a_ptr, b_ptr, c_ptr, s_ptr, out_ptr, n, D: _tl.constexpr, BLOCK: _tl.constexpr):
    # out[:, 0:D] | out[:, D:2D] | out[:, 2D:3D] = e5m2(s * [a | b | c]), all row-major.
    offs = _tl.program_id(0).to(_tl.int64) * BLOCK + _tl.arange(0, BLOCK)
    m = offs < n
    s = _tl.load(s_ptr)
    row = offs // D
    col = offs % D
    o = out_ptr + row * (3 * D) + col
    xa = _tl.load(a_ptr + offs, mask=m, other=0.0).to(_tl.float32)
    _tl.store(o, _tl.clamp(xa * s, -57344.0, 57344.0).to(_tl.float8e5), mask=m)
    xb = _tl.load(b_ptr + offs, mask=m, other=0.0).to(_tl.float32)
    _tl.store(o + D, _tl.clamp(xb * s, -57344.0, 57344.0).to(_tl.float8e5), mask=m)
    xc = _tl.load(c_ptr + offs, mask=m, other=0.0).to(_tl.float32)
    _tl.store(o + 2 * D, _tl.clamp(xc * s, -57344.0, 57344.0).to(_tl.float8e5), mask=m)

@torch.library.custom_op("fp8mlp::cat3_e5", mutates_args=())
def _cat3_e5_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    M, D = a.shape
    out = torch.empty((M, 3 * D), device=a.device, dtype=torch.float8_e5m2)
    n = M * D
    _cat3_e5_kernel[(_tr.cdiv(n, 2048),)](a, b, c, s, out, n, D=D, BLOCK=2048, num_warps=4)
    return out

@_cat3_e5_op.register_fake
def _(a, b, c, s):
    M, D = a.shape
    return torch.empty((M, 3 * D), device=a.device, dtype=torch.float8_e5m2)

class _FP8QKVFn(torch.autograd.Function):
    """q, k, v = a @ Wq^T, a @ Wk^T, a @ Wv^T as ONE FP8 GEMM against cat(Wq, Wk, Wv)
    (forward), and one FP8 GEMM for the input gradient cat(dq, dk, dv) @ cat(W)
    (dgrad); weight gradients stay bf16 exactly as nn.Linear computes them. The three
    weights stay separate parameters (Muon orthogonalises each on its own). a is the
    weightless rms_norm output, so the static scale floor(448/sqrt(d)) is exact."""
    @staticmethod
    def forward(ctx, a, wq, wk, wv):
        d = a.shape[-1]
        lead = a.shape[:-1]
        a2 = a.reshape(-1, d)
        s_a = float(_F8_E4_MAX / math.sqrt(d) // 1)
        inv_a = torch.full((), 1.0 / s_a, device=a.device, dtype=torch.float32)
        a8 = _f8(a2, s_a, _F8_E4, _F8_E4_MAX)
        w8, s_w = _w8(torch.cat([wq, wk, wv], 0))
        out = torch._scaled_mm(a8, w8.t(), scale_a=inv_a, scale_b=1.0 / s_w,
                               out_dtype=torch.bfloat16)
        nq, nk = wq.shape[0], wk.shape[0]
        ctx.save_for_backward(a, w8, s_w)
        ctx.w_dtypes = (wq.dtype, wk.dtype, wv.dtype)
        q = out[:, :nq].reshape(*lead, nq)
        k = out[:, nq:nq + nk].reshape(*lead, nk)
        v = out[:, nq + nk:].reshape(*lead, -1)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        a, w8, s_w = ctx.saved_tensors
        d = a.shape[-1]
        a2 = a.reshape(-1, d).to(torch.bfloat16)
        M = a2.shape[0]
        dq2 = dq.reshape(M, -1).to(torch.bfloat16)
        dk2 = dk.reshape(M, -1).to(torch.bfloat16)
        dv2 = dv.reshape(M, -1).to(torch.bfloat16)
        S = FP8_AMAX_STRIDE
        smax = torch.maximum(torch.maximum(dq2[::S].abs().amax(), dk2[::S].abs().amax()),
                             dv2[::S].abs().amax()).float()
        s_g = _F8_E5_MAX / (FP8_G_MARGIN * smax).clamp(min=1e-30)
        gT8 = None
        if USE_FP8_WGRAD and USE_DUAL_CAST:
            s_g4 = _F8_E4_MAX / (FP8_WG_MARGIN * smax).clamp(min=1e-30)
            g8, gT8 = torch.ops.fp8mlp.dual_cast3(dq2, dk2, dv2, s_g.reshape(1), s_g4.reshape(1))
        else:
            g8 = torch.ops.fp8mlp.cat3_e5(dq2, dk2, dv2, s_g.reshape(1))
        da = torch._scaled_mm(g8, w8.t().contiguous().t(), scale_a=1.0 / s_g,
                              scale_b=1.0 / s_w, out_dtype=torch.bfloat16)
        if USE_FP8_WGRAD:
            s_a = float(_F8_E4_MAX / math.sqrt(d) // 1)
            s_a_t = torch.full((1,), s_a, device=a.device, dtype=torch.float32)
            aT8 = torch.ops.fp8mlp.tcast_e4(a2, s_a_t)
            if gT8 is None:
                s_g4 = _F8_E4_MAX / (FP8_WG_MARGIN * smax).clamp(min=1e-30)
                gT8 = torch.ops.fp8mlp.tcast3_e4(dq2, dk2, dv2, s_g4.reshape(1))
            dw = torch._scaled_mm(gT8, aT8.t(), scale_a=1.0 / s_g4,
                                  scale_b=torch.full((), 1.0 / s_a, device=a.device, dtype=torch.float32),
                                  out_dtype=torch.bfloat16)
            nq, nk = dq2.shape[1], dk2.shape[1]
            dwq, dwk, dwv = dw[:nq], dw[nq:nq + nk], dw[nq + nk:]
        else:
            dwq = torch.mm(dq2.t(), a2)
            dwk = torch.mm(dk2.t(), a2)
            dwv = torch.mm(dv2.t(), a2)
        return (da.to(a.dtype).reshape(a.shape), dwq.to(ctx.w_dtypes[0]),
                dwk.to(ctx.w_dtypes[1]), dwv.to(ctx.w_dtypes[2]))

USE_FP8_QKV = int(os.environ.get("USE_FP8_QKV", "1"))

# ---- FP8 attention output projection on the gated (long-window) layers ------------
# out = (y * g) @ Wproj^T with y the flash-attn output [M, H*Dh] and g the per-head
# output gate [M, H]. c_proj is K = N = 768, so its three GEMMs are close to
# memory-bound and FP8 only pays if the casts add no pass of their own: on GB200
# bf16 fwd/dgrad/wgrad are 115/115/121 us per layer and FP8 74/73/86 us, while a
# standalone dual (row + transposed) cast of a [131072, 768] bf16 tensor is 69 us.
# On the gated layers the gate multiply already reads y and writes [M, 768], so one
# Triton kernel does the gate multiply and writes BOTH e4m3 copies the GEMMs need (the
# row-major forward operand and the token-major wgrad operand) in that same pass.
# The ungated short layers have no such producer pass (y comes straight out of the
# attention kernel), so they stay bf16. Scales and margins are the MLP's: sampled-amax
# FP8_H_MARGIN for the activation, FP8_G_MARGIN e5m2 / FP8_WG_MARGIN e4m3 for the
# output gradient through the shared dual_cast kernel. Eval stays bf16
# (USE_FP8_CPROJ is cleared with the other FP8 switches before evaluation).
USE_FP8_CPROJ = int(os.environ.get("USE_FP8_CPROJ", "1"))

@_tr.jit
def _gate_dual_e4_kernel(y_ptr, g_ptr, s_ptr, o_ptr, ot_ptr, M, D, H, DH,
                         BM: _tl.constexpr, BN: _tl.constexpr):
    pid_m = _tl.program_id(0)
    pid_n = _tl.program_id(1)
    rm = (pid_m * BM + _tl.arange(0, BM)).to(_tl.int64)
    rn = (pid_n * BN + _tl.arange(0, BN)).to(_tl.int64)
    s = _tl.load(s_ptr)
    head = (pid_n * BN) // DH  # BN divides DH: one head per column tile
    g = _tl.load(g_ptr + rm * H + head).to(_tl.float32)
    x = _tl.load(y_ptr + rm[:, None] * D + rn[None, :]).to(_tl.float32) * (g * s)[:, None]
    x8 = _tl.clamp(x, -448.0, 448.0).to(_tl.float8e4nv)
    _tl.store(o_ptr + rm[:, None] * D + rn[None, :], x8)
    _tl.store(ot_ptr + rn[:, None] * M + rm[None, :], _tl.trans(x8))

@torch.library.custom_op("fp8mlp::gate_dual_e4", mutates_args=())
def _gate_dual_e4_op(y: torch.Tensor, g: torch.Tensor, s: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    y = y.contiguous()
    g = g.contiguous()
    M, D = y.shape
    H = g.shape[1]
    DH = D // H
    assert M % _TC_M == 0 and DH % _TC_N == 0
    o = torch.empty((M, D), device=y.device, dtype=torch.float8_e4m3fn)
    ot = torch.empty((D, M), device=y.device, dtype=torch.float8_e4m3fn)
    _gate_dual_e4_kernel[(M // _TC_M, D // _TC_N)](y, g, s, o, ot, M, D, H, DH,
                                                   BM=_TC_M, BN=_TC_N, num_warps=4)
    return o, ot

@_gate_dual_e4_op.register_fake
def _(y, g, s):
    M, D = y.shape
    return (torch.empty((M, D), device=y.device, dtype=torch.float8_e4m3fn),
            torch.empty((D, M), device=y.device, dtype=torch.float8_e4m3fn))

class _FP8GatedProjFn(torch.autograd.Function):
    """out = (y * g[..., None]) @ W^T, FP8 forward, dgrad and wgrad.

    y: [B, T, H, Dh] attention output, g: [B, T, H] gate, W: [n_embd, H*Dh].
    The gated product is never materialised in bf16: only its two e4m3 copies are
    (row-major for the forward GEMM, token-major for wgrad). Backward returns the
    exact gate-multiply gradients against the dgrad output: dy = d(yg) * g,
    dg = sum_Dh d(yg) * y (the same expressions autograd uses for the bf16 path)."""
    @staticmethod
    def forward(ctx, y, g, w):
        B, T, H, Dh = y.shape
        M, D = B * T, H * Dh
        y2 = y.reshape(M, D)
        g2 = g.reshape(M, H)
        S = FP8_AMAX_STRIDE
        amax = (y2[::S].float().view(-1, H, Dh).abs()
                * g2[::S].float().unsqueeze(-1)).amax()
        s_y = _F8_E4_MAX / (FP8_H_MARGIN * amax).clamp(min=1e-30)
        y8, yT8 = torch.ops.fp8mlp.gate_dual_e4(y2, g2, s_y.reshape(1))
        w8, s_w = _w8(w)
        out = torch._scaled_mm(y8, w8.t(), scale_a=1.0 / s_y, scale_b=1.0 / s_w,
                               out_dtype=torch.bfloat16)
        ctx.save_for_backward(y, g, yT8, s_y, w8, s_w)
        ctx.w_dtype = w.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        y, g, yT8, s_y, w8, s_w = ctx.saved_tensors
        B, T, H, Dh = y.shape
        M = B * T
        d2 = dout.reshape(M, -1).to(torch.bfloat16)
        smax = d2[::FP8_AMAX_STRIDE].abs().amax().float()
        s_d5 = _F8_E5_MAX / (FP8_G_MARGIN * smax).clamp(min=1e-30)
        s_d4 = _F8_E4_MAX / (FP8_WG_MARGIN * smax).clamp(min=1e-30)
        d8, dT8 = torch.ops.fp8mlp.dual_cast(d2, s_d5.reshape(1), s_d4.reshape(1))
        dyg = torch._scaled_mm(d8, w8.t().contiguous().t(), scale_a=1.0 / s_d5,
                               scale_b=1.0 / s_w, out_dtype=torch.bfloat16)
        dw = torch._scaled_mm(dT8, yT8.t(), scale_a=1.0 / s_d4, scale_b=1.0 / s_y,
                              out_dtype=torch.bfloat16)
        dyg = dyg.view(B, T, H, Dh)
        dy = dyg * g.unsqueeze(-1)
        dg = (dyg.float() * y.float()).sum(-1).to(g.dtype)
        return dy.to(y.dtype), dg, dw.to(ctx.w_dtype)

class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden = mlp_hidden_for(layer_idx, config.n_layer)
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.use_fp8 = bool(USE_FP8_MLP)

    def forward(self, x):
        if not self.use_fp8:
            return self.c_proj(F.relu(self.c_fc(x)).square())
        return _FP8MLPFn.apply(x, self.c_fc.weight, self.c_proj.weight)

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x, ve, ng_v, bg_v, cos_sin, window_size, ng_scale=None, cu_seqlens=None):
        x = x + self.attn(norm(x), ve, ng_v, bg_v, cos_sin, window_size, ng_scale, cu_seqlens)
        x = x + self.mlp(norm(x))
        return x

# ---- small-table embedding backward as a one-hot GEMM ------------------------------
# The intra-doc position table (32 rows) and the induction-distance table (33 rows)
# are read by every one of the 131072 tokens of a step. Their dense backward, as
# inductor lowers it, is an fp32 atomic_add scatter fused into the end-of-backward
# rms_norm kernel: 131072 x 768 atomics landing on only 32 x 768 addresses, i.e. ~4000
# serialised updates per address. Written as dW = onehot(ids)^T @ dY instead it is one
# [n, M] x [M, d] tensor-core GEMM (M = 131072, fp32 accumulate) that reads dY once.
# Forward is the unchanged F.embedding; only the order of the fp32 summation in the
# weight gradient differs.
class _OneHotEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight):
        ctx.save_for_backward(ids)
        ctx.n = weight.shape[0]
        ctx.w_dtype = weight.dtype
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, g):
        (ids,) = ctx.saved_tensors
        d = g.shape[-1]
        rows = torch.arange(ctx.n, device=ids.device, dtype=ids.dtype)
        oh_t = (rows.unsqueeze(1) == ids.reshape(1, -1)).to(torch.bfloat16)   # [n, M]
        dw = torch.mm(oh_t, g.reshape(-1, d).to(torch.bfloat16))
        return None, dw.to(ctx.w_dtype)

USE_ONEHOT_SMALL_EMBED = os.environ.get("USE_ONEHOT_SMALL_EMBED", "1") == "1"

def small_embed(module, ids):
    if USE_ONEHOT_SMALL_EMBED and torch.is_grad_enabled():
        return _OneHotEmbedFn.apply(ids, module.weight)
    return module(ids)

class GPT(nn.Module):
    """GPT with:
    - Dense trigram table inside (receives bigram_offset as kwarg)
    - Position bucket embedding (32 buckets, zero init, LR=0.15)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.doc_local_layers = _doc_local_layer_indices(config)
        if self.doc_local_layers:
            print(f"[doc_local] varlen document-local attention on layers "
                  f"{sorted(self.doc_local_layers)} (cap {DOC_LOCAL_CAP} segments)")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size)) if USE_LM_HEAD_BIAS else None
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
                                                                                    
        self.layer_readout_w = nn.Parameter(torch.zeros(config.n_layer)) if USE_LAYER_READOUT else None

        self.dense_A = nn.Parameter(torch.zeros(config.n_layer, config.n_layer)) if USE_DENSE_DWA else None
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        self.pos_bucket_embed = nn.Embedding(NUM_POS_BUCKETS, config.n_embd)
        self.num_pos_buckets = NUM_POS_BUCKETS

        # Learned form for the n-gram -> trunk coupling (see NGRAM_RES_GATE above).
        # One packed [1 + n_sites, d] tensor: row 0 gates the residual injection, row
        # 1+k scales site k's value injection. Rows that a knob turns off are simply
        # not read (their gradient stays zero), which keeps this a single optimizer
        # tensor in every configuration.
        self.ng_sites = tuple(i for i in NGRAM_VE_LAYERS if i < config.n_layer)
        self.ng_use_res = NGRAM_RES_GATE
        self.ng_use_v = NGRAM_V_SCALE and bool(self.ng_sites)
        self.ng_scale = (nn.Parameter(torch.ones(1 + len(self.ng_sites), config.n_embd))
                         if (self.ng_use_res or self.ng_use_v) else None)

        self.use_bigram_induction = USE_BIGRAM_INDUCTION
        if self.use_bigram_induction:
            self.bigram_induction_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.use_unigram_induction = USE_UNIGRAM_INDUCTION
        if self.use_unigram_induction:
            self.unigram_induction_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)

        self.use_induction_dist = USE_INDUCTION_DIST
        if self.use_induction_dist:
            self.induction_dist_embed = nn.Embedding(NUM_INDUCTION_DIST_BUCKETS + 1, config.n_embd)
            self.num_induction_dist_buckets = NUM_INDUCTION_DIST_BUCKETS

        self.use_bigram_recall = USE_BIGRAM_RECALL
        if self.use_bigram_recall:
            self.bigram_recall_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 0.68 * 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(X0_INIT)

        if self.layer_readout_w is not None:
            self.layer_readout_w.zero_()
            self.layer_readout_w[-1] = 1.0

        if self.dense_A is not None:
            self.dense_A.zero_()
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.ng_gate is not None:
                torch.nn.init.zeros_(block.attn.ng_gate.weight)
            if block.attn.attn_gate is not None:
                torch.nn.init.zeros_(block.attn.attn_gate.weight)
                                                    
        torch.nn.init.zeros_(self.pos_bucket_embed.weight)
                                                                                            
        if self.use_bigram_induction:
            torch.nn.init.zeros_(self.bigram_induction_embed.weight)
        if self.use_bigram_recall:
            torch.nn.init.zeros_(self.bigram_recall_embed.weight)
        if self.use_unigram_induction:
            torch.nn.init.zeros_(self.unigram_induction_embed.weight)
        if self.use_induction_dist:
            torch.nn.init.zeros_(self.induction_dist_embed.weight)

        if self.lm_head_bias is not None:
            torch.nn.init.zeros_(self.lm_head_bias)

        # n-gram coupling scales: all ones, so step 0 is arithmetically the identity
        # coupling and the scales only move insofar as they earn gradient. (model is built on
        # `meta` and to_empty()'d -- anything not touched here is uninitialised
        # memory, so this line is load-bearing.)
        if self.ng_scale is not None:
            self.ng_scale.fill_(1.0)

        for _li, block in enumerate(self.transformer.h):
            if block.attn.key_mix is not None:
                km = block.attn.key_mix
                km.zero_()
                if KEY_MIX_FIXED:
                    km.copy_(torch.tensor(_KEY_MIX_TABLE[_li], dtype=torch.float32).view_as(km))
                else:
                    km[0].fill_(1.0)
                    if block.attn.key_offset:
                        km[0, KEY_OFFSET_LO:] = 0.0
                        km[1, KEY_OFFSET_LO:] = 1.0
            if block.attn.q_mix is not None:
                block.attn.q_mix.zero_()
                block.attn.q_mix[0].fill_(1.0)
            if block.attn.attn_scale is not None:
                block.attn.attn_scale.fill_(1.0)
            if block.attn.attn_sink is not None:
                # s = 0 is the literature default and is nearly a no-op in the
                # bulk of a row: with QK-norm the logits are ~N(0, 1), so a
                # 256-key window has log Z ~ 6 and the sink takes e^0/(e^6+e^0)
                # < 0.3% of the mass. Where it bites at init is exactly where the
                # window is not yet full -- the first tokens after each packed
                # document boundary -- which is the behaviour being bought.
                # That same argument is why a POSITIVE init is cheap: while
                # attention is still diffuse a sink of +2 is worth only
                # e^2/(e^6+e^2) ~ 2% of the mass, so starting the parameter at
                # its measured operating point costs almost nothing early and
                # saves it from spending the first ~40% of the run below half
                # its final magnitude (as it does with init 0 and lr 0.01: mean|s|
                # is still rising at the last step).
                block.attn.attn_sink.fill_(ATTN_SINK_INIT + _li * ATTN_SINK_INIT_SLOPE)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        self.pos_bucket_embed.to(dtype=torch.bfloat16)
        if self.use_bigram_induction:
            self.bigram_induction_embed.to(dtype=torch.bfloat16)
        if self.use_bigram_recall:
            self.bigram_recall_embed.to(dtype=torch.bfloat16)
        if self.use_unigram_induction:
            self.unigram_induction_embed.to(dtype=torch.bfloat16)
        if self.use_induction_dist:
            self.induction_dist_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # Store the tables already widened to the full head_dim: [cos, cos] and
        # [sin, -sin]. Negation is exact in bf16, and this is what lets
        # apply_rotary_emb be one cat-free expression (see there). Costs 10 MB.
        cos = torch.cat([cos, cos], 1).contiguous()
        sin = torch.cat([sin, -sin], 1).contiguous()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        long_window = config.sequence_len // 2
        short_window = config.sequence_len // 8
        if LONG_LAYERS:

            window_sizes = [
                (long_window, 0) if i in LONG_LAYERS else (short_window, 0)
                for i in range(config.n_layer)
            ]
            print(f"[window_sizes] LONG_LAYERS override -> {[w[0] for w in window_sizes]}")
            return window_sizes
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        print(f"[window_sizes] pattern '{pattern}' -> {[w[0] for w in window_sizes]}")
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        bigram_induction_numel = (self.bigram_induction_embed.weight.numel()
                                  if self.use_bigram_induction else 0)
        unigram_induction_numel = (self.unigram_induction_embed.weight.numel()
                                   if self.use_unigram_induction else 0)
        induction_dist_numel = (self.induction_dist_embed.weight.numel()
                                if self.use_induction_dist else 0)
        bigram_recall_numel = (self.bigram_recall_embed.weight.numel()
                               if self.use_bigram_recall else 0)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.pos_bucket_embed.weight.numel() + bigram_induction_numel +
                          unigram_induction_numel + induction_dist_numel + bigram_recall_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, value_embeds_lr=None,
                        matrix_lr=0.02, weight_decay=0.0,
                        adam_betas=(0.8, 0.95), scalar_lr=0.5, trigram_lr=0.15,
                        pos_bucket_lr=None, mlp_fc_lr=None, bigram_induction_lr=None,
                        unigram_induction_lr=None, induction_dist_lr=None, bigram_recall_lr=None):
        if value_embeds_lr is None:
            value_embeds_lr = embedding_lr
        if pos_bucket_lr is None:
            pos_bucket_lr = embedding_lr
        if mlp_fc_lr is None:
            mlp_fc_lr = matrix_lr
        if bigram_induction_lr is None:
            bigram_induction_lr = embedding_lr
        if unigram_induction_lr is None:
            unigram_induction_lr = embedding_lr
        if induction_dist_lr is None:
            induction_dist_lr = embedding_lr
        if bigram_recall_lr is None:
            bigram_recall_lr = embedding_lr
        model_dim = self.config.n_embd

        attn_scale_params = [b.attn.attn_scale for b in self.transformer.h if b.attn.attn_scale is not None]
        attn_sink_params = [b.attn.attn_sink for b in self.transformer.h if b.attn.attn_sink is not None]
        # Both are 1-D; neither may reach the Muon groups (Muon orthogonalises a
        # 2-D update and these are per-head vectors), so exclude them from the
        # shape-bucketed matrix sweep below.
        key_mix_all = [b.attn.key_mix for b in self.transformer.h if b.attn.key_mix is not None]
        key_mix_params = [p for p in key_mix_all if p.requires_grad]
        q_mix_params = [b.attn.q_mix for b in self.transformer.h if b.attn.q_mix is not None]
        attn_scale_ids = ({id(p) for p in attn_scale_params} | {id(p) for p in attn_sink_params}
                          | {id(p) for p in key_mix_all} | {id(p) for p in q_mix_params})
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in attn_scale_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        pos_bucket_params = list(self.pos_bucket_embed.parameters())
        dmodel_lr_scale = (model_dim / 768) ** -0.5
                                                                          
        mlp_widths = {mlp_hidden_for(i, self.config.n_layer) for i in range(self.config.n_layer)}
        mlp_fc_shapes = {torch.Size([h, model_dim]) for h in mlp_widths}
        mlp_proj_shapes = {torch.Size([model_dim, h]) for h in mlp_widths}
        print(f"Scaling LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.4f}")
        print(f"pos_bucket_lr={pos_bucket_lr:.4f} (trigram optimized externally by sparse AdamW)")
        print(f"matrix_lr={matrix_lr:.4f}, mlp_fc_lr={mlp_fc_lr:.4f} (effective c_fc: {mlp_fc_lr*2.0:.4f} with 2x aspect ratio)")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=value_embeds_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.003),
                                                                    
            dict(kind='adamw', params=pos_bucket_params, lr=pos_bucket_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=X0_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        if self.lm_head_bias is not None:
            print(f"lm_head_bias group: lr={LM_HEAD_BIAS_LR * dmodel_lr_scale:.4f} (wd=0, {self.lm_head_bias.numel()} params)")
            param_groups.append(dict(kind='adamw', params=[self.lm_head_bias],
                lr=LM_HEAD_BIAS_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if attn_scale_params:
            print(f"attn_scale group: lr={ATTN_SCALE_LR * dmodel_lr_scale:.4f} (wd=0, {len(attn_scale_params)} tensors x {attn_scale_params[0].numel()} heads)")
            param_groups.append(dict(kind='adamw', params=attn_scale_params,
                lr=ATTN_SCALE_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if attn_sink_params:
            print(f"attn_sink group: lr={ATTN_SINK_LR * dmodel_lr_scale:.4f} (wd=0, "
                  f"{len(attn_sink_params)} tensors x {attn_sink_params[0].numel()} heads, "
                  f"init {ATTN_SINK_INIT}, slope {ATTN_SINK_INIT_SLOPE}, layers={sorted(_attn_sink_layer_indices(self.config))})")
            param_groups.append(dict(kind='adamw', params=attn_sink_params,
                lr=ATTN_SINK_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.0, lrm_floor=ATTN_SINK_LRM_FLOOR))

        if key_mix_params:
            print(f"key_mix group: lr={KEY_MIX_LR * dmodel_lr_scale:.4f} (wd=0, "
                  f"{len(key_mix_params)} tensors x {tuple(key_mix_params[0].shape)})")
            param_groups.append(dict(kind='adamw', params=key_mix_params,
                lr=KEY_MIX_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if q_mix_params:
            print(f"q_mix group: lr={QMIX_LR * dmodel_lr_scale:.4f} (wd=0, "
                  f"{len(q_mix_params)} tensors x {tuple(q_mix_params[0].shape)})")
            param_groups.append(dict(kind='adamw', params=q_mix_params,
                lr=QMIX_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.layer_readout_w is not None:
            print(f"layer_readout group: lr={LAYER_READOUT_LR * dmodel_lr_scale:.4f} (wd=0, {self.layer_readout_w.numel()} weights, init last-only)")
            param_groups.append(dict(kind='adamw', params=[self.layer_readout_w],
                lr=LAYER_READOUT_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.dense_A is not None:
            print(f"dense_dwa group: lr={DENSE_DWA_LR * dmodel_lr_scale:.4f} (wd=0, {self.dense_A.numel()} matrix entries, K={DENSE_DWA_K}, init 0)")
            param_groups.append(dict(kind='adamw', params=[self.dense_A],
                lr=DENSE_DWA_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.ng_scale is not None:
            print(f"ngram_coupling_scale group: lr={NGRAM_COUPLING_LR * dmodel_lr_scale:.4f} "
                  f"(wd=0, one {tuple(self.ng_scale.shape)} tensor = {self.ng_scale.numel()} "
                  f"per-channel scales, init 1; res={self.ng_use_res} v={self.ng_use_v})")
            param_groups.append(dict(kind='adamw', params=[self.ng_scale],
                lr=NGRAM_COUPLING_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.use_bigram_induction:
            param_groups.append(dict(
                kind='adamw', params=list(self.bigram_induction_embed.parameters()),
                lr=bigram_induction_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"bigram_induction_lr={bigram_induction_lr:.4f}, "
                  f"bigram_induction_vocab={self.config.vocab_size + 1}")

        if self.use_bigram_recall:
            param_groups.append(dict(
                kind='adamw', params=list(self.bigram_recall_embed.parameters()),
                lr=bigram_recall_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"bigram_recall_lr={bigram_recall_lr:.4f}, "
                  f"bigram_recall_vocab={self.config.vocab_size + 1}")
        if self.use_unigram_induction:
            param_groups.append(dict(
                kind='adamw', params=list(self.unigram_induction_embed.parameters()),
                lr=unigram_induction_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"unigram_induction_lr={unigram_induction_lr:.4f}, "
                  f"unigram_induction_vocab={self.config.vocab_size + 1}")

        if self.use_induction_dist:
            param_groups.append(dict(
                kind='adamw', params=list(self.induction_dist_embed.parameters()),
                lr=induction_dist_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.001))
            print(f"induction_dist_lr={induction_dist_lr:.4f}, "
                  f"induction_dist_buckets={NUM_INDUCTION_DIST_BUCKETS + 1}")
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]

            group_lr = mlp_fc_lr if shape in mlp_fc_shapes else (MLP_PROJ_LR if shape in mlp_proj_shapes else matrix_lr)
            print(f"muon group: shape={tuple(shape)} n={len(group_params)} lr={group_lr:.4f}")
            param_groups.append(dict(
                kind='muon', params=group_params, lr=group_lr,
                momentum=0.95, ns_steps=5, beta2=0.85, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group.setdefault("lrm_floor", 0.0)
        return optimizer

    def _doc_cu_seqlens(self, idx):
        """cu_seqlens for document-local attention, built on-GPU with a static
        shape and no host sync.

        prepare.py's loader prepends BOS to every document and packs them back to back
        into rows of exactly sequence_len+1, so in `idx` the document starts are
        precisely the BOS positions, and because every row also starts with a BOS
        no segment ever straddles two rows. That makes the offsets a prefix sum:
        position i is the start of segment (number of BOS at or before i) - 1.

        The buffer is pre-filled with the total length and only the real offsets
        are scattered in, so the unused tail is already a run of duplicate final
        offsets, i.e. zero-length trailing sequences -- which is how the shape
        stays constant while the document count varies from step to step. Slot
        cap+1 is a scratch sink for the non-BOS positions (and for any boundary
        past the capacity, which merely merges two documents) and is sliced off.
        """
        B, T = idx.shape
        tot = B * T
        cap = DOC_LOCAL_CAP
        # Two-level prefix sum (per-row cumsum + exclusive cumsum of the B row
        # totals) instead of one flat cumsum over B*T: bit-identical, and it keeps
        # inductor from emitting a split-scan look-back kernel whose workspace is
        # sized for R0_BLOCK>=256 but can be tuned to R0_BLOCK=128, which reads past
        # the workspace and trips the scatter's index-out-of-bounds assert.
        is_bos2 = idx == BOS_TOKEN_ID
        row_cs = torch.cumsum(is_bos2.to(torch.int32), 1)
        row_tot = row_cs[:, -1]
        row_off = torch.cumsum(row_tot, 0) - row_tot
        seg = (row_cs + row_off[:, None]).reshape(-1) - 1
        is_bos = is_bos2.reshape(-1)
        pos = torch.arange(tot, device=idx.device, dtype=torch.int32)
        dump = torch.full_like(seg, cap + 1)
        tgt = torch.where(is_bos & (seg < cap), seg, dump).to(torch.int64)
        base = torch.full((cap + 2,), tot, device=idx.device, dtype=torch.int32)
        return base.scatter(0, tgt, pos)[:cap + 1]

    def forward(self, idx, targets=None, reduction='mean', bigram_offset=None, trigram_offset=None,
                pos_bucket_ids=None, bind_next_tok=None, uni_next_tok=None, induction_dist_ids=None,
                recall_next_tok=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        cu_seqlens = self._doc_cu_seqlens(idx) if self.doc_local_layers else None

        if pos_bucket_ids is not None:
            pos_emb = small_embed(self.pos_bucket_embed, pos_bucket_ids)
        else:
            pos = torch.arange(T, device=idx.device, dtype=torch.long)
            pos_buckets = (pos * self.num_pos_buckets) // self.config.sequence_len
            pos_buckets = pos_buckets.clamp(0, self.num_pos_buckets - 1)
            pos_emb = self.pos_bucket_embed(pos_buckets).unsqueeze(0)                  

        x = self.transformer.wte(idx)
        if bigram_offset is not None:
            x = x + bigram_offset
        # One cast of the packed scale tensor, reused by the residual gate and by every
        # value-injection site (the cast is 12 KB, the slices are free views).
        ng_scale = (self.ng_scale.to(x.dtype) if self.ng_scale is not None else None)
        if trigram_offset is not None:
            if ng_scale is not None and self.ng_use_res:
                x = x + trigram_offset * ng_scale[0]
            else:
                x = x + trigram_offset
        x = x + pos_emb

        if self.use_bigram_induction and bind_next_tok is not None:
            x = x + self.bigram_induction_embed(bind_next_tok)                     
                                                                                          
        if self.use_bigram_recall and recall_next_tok is not None:
            x = x + self.bigram_recall_embed(recall_next_tok)                      
        if self.use_unigram_induction and uni_next_tok is not None:
            x = x + self.unigram_induction_embed(uni_next_tok)                     
                                                                                       
        if self.use_induction_dist and induction_dist_ids is not None:
            x = x + small_embed(self.induction_dist_embed, induction_dist_ids)
        x = norm(x)
        x0 = x
        readout = self.layer_readout_w is not None
        dense = self.dense_A is not None
        layer_outs = []

        dense_states = [x0] if dense else None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            if dense and i >= 2:
                j_lo = max(1, i - DENSE_DWA_K)
                for j in range(j_lo, i):
                    x = x + self.dense_A[i, j] * dense_states[j]
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

            ng_v = None
            ng_v_scale = None
            if i in self.ng_sites and trigram_offset is not None:
                ng_v = trigram_offset
                if ng_scale is not None and self.ng_use_v:
                    ng_v_scale = ng_scale[1 + self.ng_sites.index(i)]

            bg_v = bigram_offset if (i in BIGRAM_VE_LAYERS and bigram_offset is not None) else None
            x = block(x, ve, ng_v, bg_v, cos_sin, self.window_sizes[i], ng_v_scale,
                      cu_seqlens if i in self.doc_local_layers else None)
            if readout:
                layer_outs.append(x)
            if dense:
                dense_states.append(x)
        if readout:

            if USE_READOUT_NORM:
                x = self.layer_readout_w[0] * norm(layer_outs[0])
                for i in range(1, self.config.n_layer):
                    x = x + self.layer_readout_w[i] * norm(layer_outs[i])
            else:
                x = self.layer_readout_w[0] * layer_outs[0]
                for i in range(1, self.config.n_layer):
                    x = x + self.layer_readout_w[i] * layer_outs[i]
        x = norm(x)

        softcap = SOFTCAP
        if (_fused_ce_ok and targets is not None and reduction == 'none'
                and self.lm_head_bias is None and torch.is_grad_enabled()):
            # Training path: same math as below (bf16 lm_head GEMM, fp32 softcap, CE with
            # ignore_index=-1, per-token losses), one fused kernel -- see _softcap_ce_kernel.
            B_, T_, C_ = x.shape
            # next-next targets: shifted per row, never across rows; last column -1.
            targets2 = torch.cat([targets[:, 1:], targets.new_full((B_, 1), -1)],
                                 dim=1).reshape(-1)
            if MTP_DOCMASK:
                # Where the t+1 target is BOS, the t+2 target is the first token of an
                # unrelated document: unpredictable from context, so ignore it.
                targets2 = torch.where(targets.reshape(-1) == BOS_TOKEN_ID,
                                       targets2.new_full((), -1), targets2)
            return fused_linear_softcap_ce(
                x.reshape(B_ * T_, C_).to(torch.bfloat16),
                self.lm_head.weight.to(torch.bfloat16),
                targets.reshape(-1), targets2, float(softcap))
        logits = self.lm_head(x)
        logits = logits.float()
        # Measured, not assumed: replacing this with the algebraically identical
        # softcap - 2*softcap/(1 + exp2(2*log2(e)/softcap * z)) -- one ex2.approx.f32
        # instead of libdevice tanh -- is worth 0.69 ms on the isolated loss tail but
        # exactly 0.00 ms here (1421 -> 1436 us forward, 1403 -> 1389 us backward,
        # i.e. noise). The real kernel is not tanh-bound, so the cheaper form buys
        # nothing and is not worth its 4.8e-6 numerical drift. Left as tanh.
        logits = softcap * torch.tanh(logits / softcap)

        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias

        if targets is not None:
            # F.cross_entropy here decomposes to log_softmax + nll_loss, and
            # log_softmax materialises its [131072, 8192] fp32 output: 4.295 GB
            # written in the forward and read again in the backward, which is most of
            # the 1.42 ms / 1.40 ms above. The obvious removal is
            # cross_entropy = logsumexp(z) - z[target] with the target term gathered
            # from the bf16 logits BEFORE the softcap (tanh is elementwise, so
            # gather-then-softcap is bit-identical), leaving `z` with a single
            # consumer so the cast, the softcap and the reduction fuse into one pass
            # that writes 0.5 MB of row sums instead of 4.295 GB. Verified exact
            # offline: max |loss| diff 0.0 for mean/sum, 2.9e-6 for reduction='none',
            # max grad diff 8.7e-11 on a 0.00049 gradient scale, ignore_index=-1
            # semantics preserved. MEASURED IN THE LOOP (GB200): 0.7% SLOWER
            # per step.
            # logsumexp's own two-pass max-then-sum re-reads the 2.15 GB of logits,
            # and the backward has to recompute z from them a second time, so the
            # fp32 row that was removed comes back as extra bf16 reads. The fp32
            # spill is not the binding cost here; leave the decomposition alone.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# Muon gradients: fresh per step, stacked once per bucket (MUON_GRAD_STACK=1), instead
# of autograd accumulating into a persistent zeroed buffer. With one micro-step per
# optimizer step the accumulate is 0 + g, i.e. the same numbers, but it cost one small
# CUDAFunctor_add launch per Muon parameter (~69 serial launches, ~0.26 ms/step at the
# head of the optimizer phase) plus a zero_() pass per bucket; the stack is one copy
# kernel per bucket and frees the 0.34 GB persistent gradient buffer.
MUON_GRAD_STACK = int(os.environ.get("MUON_GRAD_STACK", "1"))

class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    @torch.no_grad()
    def prestack_muon(self):
        """Give every Muon shape bucket ONE persistent [n, *shape] parameter buffer and
        one persistent gradient buffer, and rebind each parameter (and its .grad) to a
        view into them.

        The fused Muon kernel already operated on a stacked copy of the bucket and copied
        the result back, so this is bit-identical arithmetic; what disappears is per-step
        bookkeeping that was pure memory traffic: torch.stack(grads), torch.stack(params)
        and _foreach_copy_(params, stacked.unbind(0)) -- for the three big buckets that is
        ~1.5 GB of copies every step. Because .grad now aliases a persistent buffer,
        autograd accumulates into it in place and the buffer is zeroed after the update
        instead of being freed, so the caller must NOT set these grads to None.

        Must run before torch.compile(model) so the compiled graph sees the final
        parameter storage."""
        for group in self.param_groups:
            if group['kind'] != 'muon':
                continue
            params = group['params']
            if not params:
                continue
            buf = torch.stack([q.detach() for q in params]).contiguous()
            gbuf = None if MUON_GRAD_STACK else torch.zeros_like(buf)
            for i, q in enumerate(params):
                q.data = buf[i]
                q.grad = None if gbuf is None else gbuf[i]
            st = self.state[params[0]]
            st["stacked_params"] = buf
            st["stacked_grads"] = gbuf
            self._muon_prestacked = True

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        prestacked = "stacked_params" in state
        if prestacked:
            stacked_grads = state["stacked_grads"]
            stacked_params = state["stacked_params"]
            if stacked_grads is None:
                stacked_grads = torch.stack([q.grad if q.grad is not None else torch.zeros_like(q)
                                             for q in params])
        else:
            stacked_grads = torch.stack([q.grad for q in params])
            stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        if prestacked:
            if state["stacked_grads"] is not None:
                stacked_grads.zero_()
        else:
            torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

ASPECT_RATIO = 85
HEAD_DIM = 128
WINDOW_PATTERN = "SSL"

BIGRAM_TABLE_SIZE = BIGRAM_TABLE_SIZE_ENV                                   
BIGRAM_LR    = float(os.environ.get("BIGRAM_LR", 0.65))                                           

TRIGRAM_LR   = float(os.environ.get("TRIGRAM_LR", 0.45))
TRIGRAM_LR_FLOOR = float(os.environ.get("TRIGRAM_LR_FLOOR", "0.0"))
BIGRAM_LR_FLOOR = float(os.environ.get("BIGRAM_LR_FLOOR", "0.0"))

USE_TRIGRAM_FREQ_LR = os.environ.get("USE_TRIGRAM_FREQ_LR", "0") == "1"
TRIGRAM_FREQ_LR_K   = float(os.environ.get("TRIGRAM_FREQ_LR_K", "8.0"))

USE_BIGRAM_FREQ_LR = os.environ.get("USE_BIGRAM_FREQ_LR", "0") == "1"
BIGRAM_FREQ_LR_K   = float(os.environ.get("BIGRAM_FREQ_LR_K", "16.0"))
BIGRAM_FREQ_LR_FLOOR = float(os.environ.get("BIGRAM_FREQ_LR_FLOOR", "0.0"))

POS_BUCKET_LR = float(os.environ.get("POS_BUCKET_LR", 0.15))

TOTAL_BATCH_SIZE = 2**17
CORPUS_TOKENS = float(os.environ.get("CORPUS_TOKENS", "503e6"))
EMBEDDING_LR    = 0.7                                                                                
VALUE_EMBEDS_LR = 0.6                                                      
UNEMBEDDING_LR  = 0.008                                             
MATRIX_LR       = 0.020                                               
MLP_FC_LR       = float(os.environ.get("MLP_FC_LR", "0.013"))
MLP_PROJ_LR     = float(os.environ.get("MLP_PROJ_LR", "0.026"))  # zero-init c_proj: own Muon LR, above MATRIX_LR

SCALAR_LR       = 0.5
WEIGHT_DECAY    = 0.2
ADAM_BETAS      = (0.8, 0.98)
WARMUP_RATIO    = 0.0
WARMDOWN_RATIO  = 0.90
FINAL_LR_FRAC   = 0.02
MOMENTUM_WARMDOWN_DELAY = 0.45
MOMENTUM_END = 0.90
DEPTH = 9
DEVICE_BATCH_SIZE = 64

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
B200_BF16_PEAK_FLOPS = 2.25e15

# ---- background training-data loader ---------------------------------------------------
# prepare.make_dataloader's best-fit packing is pure Python on the main thread: ~50 ms of CPU
# per batch with refill spikes past 140 ms, run while the GPU executes the backward, so
# whenever a spike outlasts the backward the optimizer launches are late and the GPU
# idles. The SAME packing (same prepare._document_batches stream, same tokenizer, same
# best-fit rule, same tie-breaking, same epoch bookkeeping) now runs in a separate
# process, a few batches ahead, and streams int32 rows through a pipe. The main thread
# only reads ~0.5 MB, does the pinned H2D copy, and splits inputs/targets on the GPU
# with stream-ordered copies into persistent buffers -- the same values in the same
# places as prepare.make_dataloader (verified bit-identical over the first 20 batches).
# Evaluation is untouched: evaluate_bpb still builds prepare.py's own val loader.
_LOADER_WORKER_SRC = r"""
import os, sys, threading, queue
data_fd = os.dup(1)
os.dup2(2, 1)
sys.path.insert(0, sys.argv[3])
import numpy as np
import prepare
B, T, depth = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[4])
buffer_size = 1000
tokenizer = prepare.Tokenizer.from_directory()
bos_token = tokenizer.get_bos_token_id()
batches = prepare._document_batches("train")
row_capacity = T + 1
doc_buffer = []
# Parallel to doc_buffer: each document's index in the epoch-1 document stream (the
# order prepare._document_batches yields them). Reported in every batch's header so the
# eval-time mixer knows which documents the trainer consumed (see _BgBuild).
id_buffer = []
n_enc = 0
HDR = int(sys.argv[5])
epoch = 1
q = queue.Queue(maxsize=depth)
def writer():
    try:
        while True:
            mv = memoryview(q.get())
            while len(mv):
                n = os.write(data_fd, mv)
                mv = mv[n:]
    except (BrokenPipeError, OSError):
        os._exit(0)
threading.Thread(target=writer, daemon=True).start()
def refill_buffer():
    global epoch, n_enc
    doc_batch, epoch = next(batches)
    enc = tokenizer.encode(doc_batch, prepend=bos_token)
    doc_buffer.extend(enc)
    id_buffer.extend(range(n_enc, n_enc + len(enc)))
    n_enc += len(enc)
while True:
    out = np.empty(1 + HDR + B * row_capacity, dtype=np.int32)
    row_buffer = out[1 + HDR:].reshape(B, row_capacity)
    for row_idx in range(B):
        pos = 0
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill_buffer()
            remaining = row_capacity - pos
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len
            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                id_buffer.pop(best_idx)
                row_buffer[row_idx, pos:pos + len(doc)] = doc
                pos += len(doc)
            else:
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                id_buffer.pop(shortest_idx)
                row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                pos += remaining
    out[0] = epoch
    if HDR:
        # doc state AFTER this batch was packed: documents encoded so far, and which of
        # them are still waiting in the best-fit buffer
        nb = len(id_buffer)
        assert nb <= HDR - 2, nb
        out[1] = n_enc
        out[2] = nb
        out[3:3 + nb] = id_buffer
    q.put(out)
"""

class BackgroundTrainLoader:
    # int32 header slots per batch for the document state (2 + up to 1127 buffer ids)
    HDR = 2048

    def __init__(self, B, T, depth=6):
        import subprocess, sys as _sys
        self.B, self.T = B, T
        self.doc_n, self.doc_buf = 0, None
        lib_dir = os.path.dirname(os.path.abspath(__import__("prepare").__file__))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
        self.proc = subprocess.Popen(
            [_sys.executable, "-c", _LOADER_WORKER_SRC, str(B), str(T), lib_dir, str(depth), str(self.HDR)],
            stdout=subprocess.PIPE, env=env, bufsize=0)
        self.fd = self.proc.stdout.fileno()
        try:
            import fcntl
            fcntl.fcntl(self.fd, 1031, 1 << 20)  # F_SETPIPE_SZ
        except Exception:
            pass
        self.n = 1 + self.HDR + B * (T + 1)
        self.nbytes = self.n * 4
        self._host = None
        self._k = 0

    def _alloc(self):
        self._host = [torch.empty(self.n, dtype=torch.int32, pin_memory=True) for _ in range(2)]
        self._mv = [memoryview(h.numpy()).cast("B") for h in self._host]
        self._gpu_rows = torch.empty(self.n, dtype=torch.int32, device="cuda")
        gpu_buffer = torch.empty(2 * self.B * self.T, dtype=torch.long, device="cuda")
        self.inputs = gpu_buffer[:self.B * self.T].view(self.B, self.T)
        self.targets = gpu_buffer[self.B * self.T:].view(self.B, self.T)

    def __iter__(self):
        return self

    def __next__(self):
        if self._host is None:
            self._alloc()
        k = self._k = self._k ^ 1
        mv = self._mv[k]
        off = 0
        while off < self.nbytes:
            r = os.readv(self.fd, [mv[off:]])
            if r == 0:
                raise RuntimeError(f"train loader worker exited (rc={self.proc.poll()})")
            off += r
        host = self._host[k]
        hn = host.numpy()
        epoch = int(hn[0])
        self.doc_n = int(hn[1])
        self.doc_buf = hn[3:3 + int(hn[2])].astype(_np.int64)   # copy: host is reused
        self._gpu_rows.copy_(host, non_blocking=True)
        rows = self._gpu_rows[1 + self.HDR:].view(self.B, self.T + 1)
        self.inputs.copy_(rows[:, :-1])
        self.targets.copy_(rows[:, 1:])
        return self.inputs, self.targets, epoch

    def close(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass

USE_BG_LOADER = os.environ.get("USE_BG_LOADER", "1") == "1"
_bg_loader = BackgroundTrainLoader(DEVICE_BATCH_SIZE, MAX_SEQ_LEN) if USE_BG_LOADER else None

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
BOS_TOKEN_ID = tokenizer.get_bos_token_id()
print(f"Vocab size: {vocab_size:,}")
print(f"BOS_TOKEN_ID: {BOS_TOKEN_ID}  USE_INTRADOC_POS: {USE_INTRADOC_POS}")
print(f"USE_BIGRAM_INDUCTION: {USE_BIGRAM_INDUCTION}  BIGRAM_INDUCTION_LR: {BIGRAM_INDUCTION_LR}  "
      f"INDUCTION_ORDER: {INDUCTION_ORDER}  INDUCTION_DOCMASK: {INDUCTION_DOCMASK}")
print(f"USE_INDUCTION_DIST: {USE_INDUCTION_DIST}  INDUCTION_DIST_LR: {INDUCTION_DIST_LR}  "
      f"NUM_INDUCTION_DIST_BUCKETS: {NUM_INDUCTION_DIST_BUCKETS}  COMPILE_LOOKUPS: {COMPILE_LOOKUPS}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
print(f"BIGRAM_TABLE_SIZE: {BIGRAM_TABLE_SIZE:,} ({BIGRAM_TABLE_SIZE/1e6:.0f}M entries)")
print(f"TRIGRAM_TABLE_SIZE: {TRIGRAM_TABLE_SIZE:,} ({TRIGRAM_TABLE_SIZE/1e6:.1f}M entries, sparse)")
print(f"NUM_POS_BUCKETS: {NUM_POS_BUCKETS}")
print(f"NGRAM_VE_LAYERS (trigram->value path): {NGRAM_VE_LAYERS}")
print(f"NGRAM coupling: RES_GATE={NGRAM_RES_GATE}, V_SCALE={NGRAM_V_SCALE}, "
      f"COUPLING_LR={NGRAM_COUPLING_LR}")
print(f"BIGRAM_VE_LAYERS (bigram->value path): {BIGRAM_VE_LAYERS}")

if BIGRAM_FACTORED:
    bigram_opt_gb = (BIGRAM_TABLE_SIZE * 768 * 2 + BIGRAM_TABLE_SIZE * 4) / 1e9
else:
    bigram_opt_gb = 2 * BIGRAM_TABLE_SIZE * 768 * 2 / 1e9                               
                                                                                                
# The trigram optimizer runs momentum-free (nomom=True below), so its only per-row
# state is the factored fp32 v and the int32 visit counter -- no [n, d] exp_avg. That
# is what pays for the 76.8M-row table: its momentum buffer alone would be 118 GB.
trigram_opt_gb = (TRIGRAM_TABLE_SIZE * 4 + TRIGRAM_TABLE_SIZE * 4) / 1e9
bigram_weight_gb_est = BIGRAM_TABLE_SIZE * 768 * 2 / 1e9
trigram_weight_gb_est = TRIGRAM_TABLE_SIZE * 768 * 2 / 1e9
print(f"Bigram optimizer memory: {bigram_opt_gb:.1f} GB (factored={BIGRAM_FACTORED})")
print(f"Trigram optimizer memory: {trigram_opt_gb:.2f} GB (no momentum + fp32 factored v + int32 cnt)")
print(f"[VRAM estimate] bigram(weight+opt)={bigram_weight_gb_est+bigram_opt_gb:.1f}GB  "
      f"trigram(weight+opt)={trigram_weight_gb_est+trigram_opt_gb:.1f}GB  "
      f"ngram-total={bigram_weight_gb_est+bigram_opt_gb+trigram_weight_gb_est+trigram_opt_gb:.1f}GB "
      f"(+~34GB rest => predicted peak ~{bigram_weight_gb_est+bigram_opt_gb+trigram_weight_gb_est+trigram_opt_gb+34:.0f}GB)")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

num_params_gpt = sum(p.numel() for p in model.parameters())
print(f"GPT params (transformer + pos_bucket, excl. sparse n-gram tables): {num_params_gpt/1e6:.1f}M")
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs/token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
print(f"Gradient accumulation steps: {grad_accum_steps}")

print(f"Building bigram embedding table ({BIGRAM_TABLE_SIZE:,} × {config.n_embd})...")

bigram_embed = nn.Embedding(BIGRAM_TABLE_SIZE + (SPARSE_PAD_ROWS if SPARSE_SYNC_FREE else 0), config.n_embd, sparse=True,
                            dtype=torch.bfloat16, device=device)
bigram_embed.weight.data.zero_()
bigram_weight_gb = bigram_embed.weight.data.element_size() * bigram_embed.weight.data.numel() / 1e9
print(f"Bigram weight: {bigram_weight_gb:.1f} GB bf16")

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
    value_embeds_lr=VALUE_EMBEDS_LR, scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS, matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
    trigram_lr=TRIGRAM_LR, pos_bucket_lr=POS_BUCKET_LR,
    mlp_fc_lr=MLP_FC_LR, bigram_induction_lr=BIGRAM_INDUCTION_LR,
    unigram_induction_lr=UNIGRAM_INDUCTION_LR, induction_dist_lr=INDUCTION_DIST_LR,
    bigram_recall_lr=BIGRAM_RECALL_LR,
)

# Persistent pre-stacked Muon buckets: removes the two torch.stack()s and the
# _foreach_copy_() that ran for every shape bucket on every step (pure memory traffic,
# no arithmetic). Runs BEFORE torch.compile(model) so the graph sees the final storage.
# The Muon parameters' .grad now alias persistent buffers that autograd accumulates into
# and _step_muon zeroes, so only the AdamW-managed parameters get set_to_none each step.
optimizer.prestack_muon()
muon_param_ids = {id(p) for g in optimizer.param_groups if g['kind'] == 'muon'
                  for p in g['params']}
adamw_grad_params = [p for p in model.parameters() if MUON_GRAD_STACK or id(p) not in muon_param_ids]
print(f"[prestack_muon] {len(muon_param_ids)} Muon params in "
      f"{sum(1 for g in optimizer.param_groups if g['kind'] == 'muon')} persistent buckets; "
      f"{len(adamw_grad_params)} AdamW params keep set_to_none grads")

initial_bigram_lr = BIGRAM_LR
bigram_sparse_opt = SparseBigramAdamW(
    weight=bigram_embed.weight,
    initial_lr=initial_bigram_lr,
    betas=ADAM_BETAS,
    eps=1e-10,
    weight_decay=0.003,
    state_dtype=torch.bfloat16,
    factored_v=BIGRAM_FACTORED,                                                                       
    name="bigram",
    freq_lr=USE_BIGRAM_FREQ_LR,                                                                       
    freq_lr_K=BIGRAM_FREQ_LR_K,
    freq_lr_floor=BIGRAM_FREQ_LR_FLOOR,                                                       
)

print(f"Building trigram embedding table ({TRIGRAM_TABLE_SIZE:,} × {config.n_embd})...")

trigram_embed = nn.Embedding(TRIGRAM_TABLE_SIZE + (SPARSE_PAD_ROWS if SPARSE_SYNC_FREE else 0), config.n_embd, sparse=True,
                             dtype=torch.bfloat16, device=device)
trigram_embed.weight.data.zero_()
trigram_weight_gb = trigram_embed.weight.data.element_size() * trigram_embed.weight.data.numel() / 1e9
print(f"Trigram weight: {trigram_weight_gb:.1f} GB bf16")
initial_trigram_lr = TRIGRAM_LR
trigram_sparse_opt = SparseBigramAdamW(
    weight=trigram_embed.weight,
    initial_lr=initial_trigram_lr,
    betas=(0.0, ADAM_BETAS[1]),
    eps=1e-10,
    weight_decay=0.003,
    state_dtype=torch.bfloat16,
    factored_v=True,
    name="trigram",
    nomom=True,
    freq_lr=USE_TRIGRAM_FREQ_LR,
    freq_lr_K=TRIGRAM_FREQ_LR_K,
    freq_lr_floor=TRIGRAM_LR_FLOOR,
)

lm_head_bias_param = model.lm_head_bias if USE_LM_HEAD_BIAS else None

# No CUDA graphs on the training graph: on a B200 (178 GiB usable) the n-gram tables
# leave too little memory for the CUDA-graph private pool and "max-autotune" OOMs.
model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")

if _extra is not None and _bg_loader is None:
    _extra.wrap(tokenizer)   # in-process loader: track doc ids through the tokenizer
train_loader = (_bg_loader if _bg_loader is not None
                else make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train"))
# Gate-fit slice for the eval-time mixture: the first MIX_NSLICE training batches are
# set aside and NEVER trained on and never enter a mixer table (the capture below only
# records batches that go through a forward/backward). The network's loss on them is
# taken with the final weights after the loop. The background loader hands out
# persistent buffers, so each slice batch is cloned.
mix_slice = []
if MIX_ENABLE:
    for _ in range(MIX_NSLICE):
        _sx, _sy, _ = next(train_loader)
        mix_slice.append((_sx.clone(), _sy.clone()))
if _extra is not None:
    # documents popped so far went into the gate slice (they never enter a table); the
    # message is buffered and handed to the worker when it is launched in the loop
    _extra.attach(train_loader)
    _extra.send_slice(mix_slice, _extra.state(), {
        "orders": MIX_ORDERS, "mults": [int(v) for v in _MIX_MULTS], "shingle_k": MIX_SHINGLE_K,
        "shingle_s": MIX_SHINGLE_S, "df_max": MIX_DF_MAX, "piece": MAX_SEQ_LEN,
        "V": vocab_size, "bos": BOS_TOKEN_ID})
x, y, epoch = next(train_loader)

if lm_head_bias_param is not None and LM_HEAD_BIAS_INIT:
    with torch.no_grad():
        logfreq0 = compute_unigram_logfreq(y, vocab_size).to(lm_head_bias_param.dtype)
        lm_head_bias_param.data.copy_(logfreq0)
    print(f"lm_head_bias smart-init from first batch: min={logfreq0.min().item():.3f} "
          f"max={logfreq0.max().item():.3f} mean={logfreq0.mean().item():.3f}")

print(f"Time budget: {TIME_BUDGET}s")
print(f"BIGRAM_LR: {BIGRAM_LR}, BIGRAM_TABLE_SIZE: {BIGRAM_TABLE_SIZE:,}")
print(f"BIGRAM_FREQ_LR: {USE_BIGRAM_FREQ_LR}, K: {BIGRAM_FREQ_LR_K}, rare_row_floor: {BIGRAM_FREQ_LR_FLOOR}")
print(f"TRIGRAM_LR: {TRIGRAM_LR}, TRIGRAM_TABLE_SIZE: {TRIGRAM_TABLE_SIZE:,}")
print(f"POS_BUCKET_LR: {POS_BUCKET_LR}, NUM_POS_BUCKETS: {NUM_POS_BUCKETS}")
print(f"MLP_FC_LR: {MLP_FC_LR} (effective {MLP_FC_LR*2:.4f} with 2x aspect ratio), MATRIX_LR: {MATRIX_LR}")
print(f"USE_LM_HEAD_BIAS: {USE_LM_HEAD_BIAS} (off), LM_HEAD_BIAS_LR: {LM_HEAD_BIAS_LR}")
print(f"USE_ATTN_SCALE: {USE_ATTN_SCALE}, ATTN_SCALE_LR: {ATTN_SCALE_LR}")
print(f"USE_ATTN_SINK: {USE_ATTN_SINK}, ATTN_SINK_LAYERS: {ATTN_SINK_LAYERS}, "
      f"ATTN_SINK_INIT: {ATTN_SINK_INIT}, ATTN_SINK_LR: {ATTN_SINK_LR}")
print(f"USE_DOC_LOCAL: {USE_DOC_LOCAL}, DOC_LOCAL_LAYERS: {DOC_LOCAL_LAYERS}, "
      f"DOC_LOCAL_CAP: {DOC_LOCAL_CAP}, DOC_LOCAL_FULL_DOC: {DOC_LOCAL_FULL_DOC}")
print(f"USE_LAYER_READOUT: {USE_LAYER_READOUT}, LAYER_READOUT_LR: {LAYER_READOUT_LR}, USE_READOUT_NORM: {USE_READOUT_NORM}")
print(f"USE_DENSE_DWA: {USE_DENSE_DWA}, DENSE_DWA_LR: {DENSE_DWA_LR}, DENSE_DWA_K: {DENSE_DWA_K}")

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step, progress):
    ramp_frac = min(step / 200, 1)
    base_momentum = (1 - ramp_frac) * 0.85 + ramp_frac * 0.95
    WARMDOWN_START = 1.0 - WARMDOWN_RATIO
    if progress >= WARMDOWN_START:
        warmdown_frac = (progress - WARMDOWN_START) / WARMDOWN_RATIO
        delayed_frac = max(0.0, (warmdown_frac - MOMENTUM_WARMDOWN_DELAY) / (1.0 - MOMENTUM_WARMDOWN_DELAY))
        base_momentum = 0.95 - delayed_frac * (0.95 - MOMENTUM_END)
    return base_momentum

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

EVAL_NO_FP8 = int(os.environ.get("EVAL_NO_FP8", "1"))
EVAL_FP32_LM_HEAD = int(os.environ.get("EVAL_FP32_LM_HEAD", "1"))

class _FP32Head(nn.Module):
    """lm_head replacement for eval only: the same weight, computed in fp32."""
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return F.linear(x.float(), self.weight.float())

# The mixer's network expert runs through the same no-FP8 / fp32-lm_head precision as
# the plain evaluation below (MIX_EVAL_NOFP8). The switches are module attributes and
# module globals that the compiled graphs guard on, so the eval graph is compiled with
# them OFF before training, training runs with them ON (its own graph), and they are
# switched OFF again after the loop -- the slice forward and the scored evaluation then
# hit the eval graph compiled here, so the gate is fitted on exactly the network that
# is scored.
MIX_EVAL_NOFP8 = int(os.environ.get("MIX_EVAL_NOFP8", "1")) and EVAL_NO_FP8
_eval_prec_saved = {}

def _set_eval_precision(on):
    global USE_FP8_QKV, USE_FP8_CPROJ
    bm = model._orig_mod if hasattr(model, "_orig_mod") else model
    if on:
        _eval_prec_saved.setdefault("qkv", USE_FP8_QKV)
        _eval_prec_saved.setdefault("cproj", USE_FP8_CPROJ)
        _eval_prec_saved.setdefault("mlp", [blk.mlp.use_fp8 for blk in bm.transformer.h])
        for blk in bm.transformer.h:
            blk.mlp.use_fp8 = False
        USE_FP8_QKV = 0
        USE_FP8_CPROJ = 0
        if EVAL_FP32_LM_HEAD:
            if "head" not in _eval_prec_saved:
                _eval_prec_saved["head"] = bm.lm_head
                _eval_prec_saved["fp32head"] = _FP32Head(bm.lm_head.weight)
            bm.lm_head = _eval_prec_saved["fp32head"]
    else:
        for blk, v in zip(bm.transformer.h, _eval_prec_saved["mlp"]):
            blk.mlp.use_fp8 = v
        USE_FP8_QKV = _eval_prec_saved["qkv"]
        USE_FP8_CPROJ = _eval_prec_saved["cproj"]
        if "head" in _eval_prec_saved:
            bm.lm_head = _eval_prec_saved["head"]

if MIX_ENABLE and MIX_ENT:
    # Compile (and autotune) the eval-mode, targets=None inference graph now, like every
    # other compilation outside the budget, so the in-budget slice forward after training
    # does not compile. The warm-up runs the inference graph on the first batch under
    # no_grad; nothing is updated.
    _tw = time.time()
    eval_net = torch.compile(model._orig_mod, dynamic=False, mode="max-autotune-no-cudagraphs")
    model.eval()
    if MIX_EVAL_NOFP8:
        _set_eval_precision(True)
    _ew = GPTWithBigram(eval_net, bigram_embed, BIGRAM_PRIME, BIGRAM_TABLE_SIZE,
                        trigram_embed, TRIGRAM_TABLE_SIZE)
    with autocast_ctx:
        for _ in range(2):
            _l, _e, _tp = net_loss_ent(_ew, x, y)
    del _ew, _l, _e, _tp
    if MIX_EVAL_NOFP8:
        _set_eval_precision(False)
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[mix] eval graph warm-up {time.time() - _tw:.1f}s (outside the budget)", flush=True)

if MIX_ENABLE:
    CorpusMixer.warmup_fit()

# Trained-on token capture (int16, GPU; 1 GB for 4000 steps) and the host arena for the
# n-gram tables that do not fit on the card once training is over.
MIX_RESERVE = float(os.environ.get("MIX_RESERVE", "9.5")) if MIX_ENABLE else 0.0
TRAIN_BUDGET = TIME_BUDGET - MIX_RESERVE
mix_stream = mix_arena = None
if MIX_ENABLE and _MIX_BG:
    print(f"[mix] background build: starts at the first timed step (step 11) on {len(_extra.cpus_build)} cores "
          f"{_extra.cpus_build[0]}-{_extra.cpus_build[-1]} (GPU NUMA {_extra.numa}, MIX_BG_NODE={os.environ.get('MIX_BG_NODE', 'other')}), nice 19; "
          f"reserve {MIX_RESERVE:.1f}s of the budget for slice forward + doc expert + upload + gate fit", flush=True)
elif MIX_ENABLE:
    _cap_steps = int(os.environ.get("MIX_CAP_STEPS", "3900"))
    mix_stream = torch.empty(_cap_steps * TOTAL_BATCH_SIZE, dtype=torch.int16, device=device)
    _t_arena = time.time()
    mix_arena = torch.empty(int(float(os.environ.get("MIX_ARENA_GB", "140")) * 1e9) // 8,
                            dtype=torch.int64, pin_memory=True)
    print(f"[mix] arena {mix_arena.numel() * 8 / 1e9:.1f} GB pinned in {time.time() - _t_arena:.1f}s; "
          f"reserve {MIX_RESERVE:.1f}s of the budget for build + gate fit")
mix_ntok = 0
_dt_log = []
if MIX_ENABLE and _MIX_BG:
    # keep this process (all its threads) and the loader off the build worker's cores
    _npin = _extra.pin_others([_bg_loader.proc.pid if _bg_loader is not None else None]
                              + [p.pid for p in _extra.procs if p.poll() is None])
    print(f"[mix] pinned {_npin} main/loader/tokenizer threads to the {len(_extra.cpus_rest)} other cores",
          flush=True)

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Side stream for the sparse n-gram optimizer (see the launch site in the loop).
sparse_stream = torch.cuda.Stream() if os.environ.get("SPARSE_OPT_STREAM", "1") == "1" else None
sparse_ready = torch.cuda.Event() if sparse_stream is not None else None
print(f"[sparse_stream] {'ON' if sparse_stream is not None else 'OFF'}: the two n-gram table "
      f"updates are issued on a side stream, Muon/AdamW on the main stream")


_SCALAR_TRACE = os.environ.get("SCALAR_TRACE", "0") == "1"
_SCALAR_TRACE_STEPS = {3, 100, 400, 900, 1600, 2400, 3200, 3700}

# Step-input prefetch. Every step used to open with ~1.0 ms of pure host work -- the
# lookup graph launch plus the two sparse-embedding forwards -- while the GPU sat idle
# after the previous step's barrier (measured with CUDA events: GPU-time from step start
# to the model call == CPU-time, 1.0-1.15 ms). The same kernels are now ENQUEUED at the
# tail of the previous step, right after the n-gram tables' update has been joined into
# the main stream, where the host is otherwise idle and the GPU is still busy with Muon.
# Same inputs (the loader already handed out the next batch), same table weights (the
# embedding forward is stream-ordered after the sparse update), same kernels: the numbers
# are unchanged. The work stays inside the timed window of the step that issues it (it
# completes before that step's synchronize), so dt still measures everything.
PREFETCH_STEP_INPUTS = os.environ.get("PREFETCH_STEP_INPUTS", "1") == "1"

def _prep_step_inputs(idx):
    with torch.no_grad():
        (bk, tk, pb, bnt, unt, idi, rnt) = get_lookups(idx)
    return (bigram_embed(bk), trigram_embed(tk), pb, bnt, unt, idi, rnt)

def _train_forward(x, y, pre):
    (bigram_out, trigram_out, pos_bucket_ids, bind_next_tok, uni_next_tok,
     induction_dist_ids, recall_next_tok) = pre
    with autocast_ctx:
        loss_tok = model(x, y, reduction='none',
                         bigram_offset=bigram_out, trigram_offset=trigram_out,
                         pos_bucket_ids=pos_bucket_ids, bind_next_tok=bind_next_tok,
                         uni_next_tok=uni_next_tok, induction_dist_ids=induction_dist_ids,
                         recall_next_tok=recall_next_tok)
        n_valid = (y.view(-1) != -1).sum().clamp(min=1)
        loss = loss_tok.sum() / n_valid
    return loss / grad_accum_steps, loss.detach()

# Early forward. The step boundary is a full device sync, after which the GPU idles
# until the host has run the compiled forward's guards and launch bookkeeping
# and launched its input copies and graph. With EARLY_FWD the NEXT step's forward is
# enqueued at the tail of this step, right after its inputs are prefetched, while the
# GPU is still busy with Muon, so that launch latency hides behind device work. Same
# kernels on the same stream in the same order (the forward still reads the parameters
# the optimizer just wrote, and the same _mtp_w_t), so the numbers are unchanged; the
# forward's device time now lands inside the timed window of the step that issues it,
# and the step that consumes it starts directly with its backward. Skipped for the
# last ~half second of the budget and at the corpus wall, so the final step never
# carries a forward that no backward will consume.
EARLY_FWD = int(os.environ.get("EARLY_FWD", "1")) and grad_accum_steps == 1
# Early backward. Even with EARLY_FWD the GPU idled ~1 ms per step between the end of
# the early forward and the start of the backward graph: the step-boundary device sync
# drained the queue, then the host did the step bookkeeping and ran autograd plus the
# compiled backward's launch path. With EARLY_BWD the next step's
# backward is enqueued right behind its early forward (same kernels, same stream, same
# order -- nothing on the GPU sits between that forward and backward in the original
# loop either: the next step's _mtp_w_t fill, optimizer and lr writes all come after
# the backward). The step boundary then waits on an event recorded between that
# forward and backward instead of on the whole device, and the next step's clock
# starts at that same barrier (t0 = previous t1), so time is measured CONTINUOUSLY,
# barrier to barrier, and every kernel lands in exactly one step's dt -- the host
# bookkeeping between steps, which the old sync/t0 pair left out, now counts too. The
# train loss is read from a pinned copy enqueued before the barrier, so .item() does
# not wait on the in-flight backward.
EARLY_BWD = int(os.environ.get("EARLY_BWD", "1")) and EARLY_FWD
_barrier_ev = torch.cuda.Event()
_loss_host = None
_bwd_done = False
_t_barrier = None
_pending_fwd = None
_prefetched = None
while True:
    if _t_barrier is not None:
        t0 = _t_barrier  # this step's backward is already in flight; clock runs on
        _t_barrier = None
    else:
        torch.cuda.synchronize()
        t0 = time.time()

    if step == 11 and _extra is not None:
        # Step 11 is the first step whose dt enters total_training_time, and its clock
        # (t0) is already running: the build starts inside the timer, after compilation.
        _t_timed0 = t0
        _extra.launch()

    for micro_step in range(grad_accum_steps):
        if MIX_ENABLE and _MIX_BG:
            mix_ntok += x.numel()
            _extra.snap()       # loader doc state as of the batch being trained (leak check)
        elif MIX_ENABLE and mix_ntok + x.numel() <= mix_stream.numel():
            # x is the batch this micro-step trains on (its forward, and with EARLY_BWD
            # its backward, may already be in flight; x is only overwritten by the
            # stream-ordered next() below), so the capture is exactly the trained data.
            mix_stream[mix_ntok:mix_ntok + x.numel()].copy_(x.reshape(-1))
            mix_ntok += x.numel()
        if _pending_fwd is not None:
            loss, train_loss = _pending_fwd
            _pending_fwd = None
        else:
            if _prefetched is None:
                _prefetched = _prep_step_inputs(x)
            loss, train_loss = _train_forward(x, y, _prefetched)
        _prefetched = None
        if _bwd_done:
            _bwd_done = False  # enqueued at the tail of the previous step
        else:
            loss.backward()
        x, y, epoch = next(train_loader)

    # Schedule progress is the larger of time spent and corpus spent. The loader holds
    # 505.4M training tokens (3856 steps of 2**17) and the corpus-wall break below
    # fires a few steps before that; a run fast enough to reach the wall inside the
    # 300 s budget would otherwise stop mid-warmdown. CORPUS_TOKENS puts the end of the
    # LR/momentum/WD schedule at step 3838, safely before the wall, and is a no-op for
    # any run that is time-limited (time progress is then always the larger term).
    # Time progress runs against TRAIN_BUDGET: the mixer's post-loop reserve is
    # charged after the loop, so the schedule must finish before it.
    progress = min(max(total_training_time / TRAIN_BUDGET,
                       step * TOTAL_BATCH_SIZE / CORPUS_TOKENS), 1.0)
    lrm = get_lr_multiplier(progress)
    _mtp_w_t.fill_(MTP_W * max(0.0, 1.0 - progress / MTP_END) if MTP_END > 0 else 0.0)
    muon_momentum = get_muon_momentum(step, progress)
    muon_weight_decay = get_weight_decay(progress)

    for group in optimizer.param_groups:
        # lrm_floor is absent on every group except the attention-sink group (and there
        # it defaults to 0.0, i.e. max(lrm, 0.0) == lrm), so this is the plain warmdown
        # schedule unless ATTN_SINK_LRM_FLOOR is set. The sparse n-gram tables already
        # bypass the warmdown the same way, via BIGRAM/TRIGRAM_LR_FLOOR below.
        group["lr"] = group["initial_lr"] * max(lrm, group["lrm_floor"])
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    if bigram_sparse_opt.freq_lr:

        bigram_sparse_opt.base_lr = initial_bigram_lr
        bigram_sparse_opt.lrm = lrm
    else:
        bigram_sparse_opt.lr = initial_bigram_lr * max(lrm, BIGRAM_LR_FLOOR)

    if trigram_sparse_opt.freq_lr:

        trigram_sparse_opt.base_lr = initial_trigram_lr
        trigram_sparse_opt.lrm = lrm
    else:
        trigram_sparse_opt.lr = initial_trigram_lr * max(lrm, TRIGRAM_LR_FLOOR)

    # The two sparse n-gram updates and the Muon/AdamW update touch disjoint tensors, so
    # they may run CONCURRENTLY without changing a single number. They are also the two
    # complementary kinds of work on this GPU: each sparse update is random-row
    # gather/scatter into a 26 GB / 41 GB table, i.e. TLB- and latency-bound with the
    # memory pipes mostly idle, while Muon is dense bandwidth plus small bf16 bmms. Put
    # the sparse pair on a side stream that waits only on the backward (the event is
    # recorded before Muon is launched, so it does NOT serialise behind Muon) and rejoin
    # before the step's timing barrier, which keeps dt -- and therefore the 300 s budget
    # -- an honest measurement of everything this step did.
    #
    # Allocator safety: Muon is enqueued FIRST in Python, so every main-stream temporary
    # it needs is allocated before the side stream frees the sparse .grad tensors that
    # were allocated on the main stream; nothing on the main stream allocates between
    # that free and the join + torch.cuda.synchronize() below, so no freed block can be
    # handed out while the side stream is still reading it.
    if sparse_stream is not None:
        sparse_ready.record()
        optimizer.step()
        with torch.cuda.stream(sparse_stream):
            sparse_stream.wait_event(sparse_ready)
            bigram_sparse_opt.step()
            trigram_sparse_opt.step()
        torch.cuda.current_stream().wait_stream(sparse_stream)
    else:
        optimizer.step()
        bigram_sparse_opt.step()
        trigram_sparse_opt.step()

    if PREFETCH_STEP_INPUTS and epoch <= 1:
        _prefetched = _prep_step_inputs(x)

    for _p in adamw_grad_params:
        _p.grad = None

    if (EARLY_FWD and _prefetched is not None and epoch <= 1
            and total_training_time + 0.5 < TRAIN_BUDGET):
        _pending_fwd = _train_forward(x, y, _prefetched)

    if EARLY_BWD and _pending_fwd is not None:
        if _loss_host is None:
            _loss_host = torch.empty(train_loss.shape, dtype=train_loss.dtype, pin_memory=True)
        _loss_host.copy_(train_loss, non_blocking=True)
        _barrier_ev.record()
        _pending_fwd[0].backward()
        _bwd_done = True
        _barrier_ev.synchronize()
        t1 = time.time()
        _t_barrier = t1
        train_loss_f = _loss_host.item()
    else:
        train_loss_f = train_loss.item()
        torch.cuda.synchronize()
        t1 = time.time()
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    dt = t1 - t0
    _dt_log.append((t1, dt))

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TRAIN_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | mom: {muon_momentum:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Diagnostic only (off unless SCALAR_TRACE is set): dump the small learned
    # scalar groups at a few checkpoints. Every iteration of this loop already
    # ends on a torch.cuda.synchronize(), so the .item()/.tolist() reads below
    # add no sync that was not there, and the whole block is skipped when the
    # env var is absent -- which is how the scored configuration runs.
    if _SCALAR_TRACE and step in _SCALAR_TRACE_STEPS:
        with torch.no_grad():
            _sinks = [b.attn.attn_sink for b in model.transformer.h
                      if b.attn.attn_sink is not None]
            if _sinks:
                _s = torch.stack(_sinks)
                print(f"\n[trace {step:05d}] sink mean|s|={_s.abs().mean().item():.3f} "
                      f"max|s|={_s.abs().max().item():.3f} per-layer-mean="
                      f"{[round(v, 2) for v in _s.mean(dim=1).tolist()]}", flush=True)
            _sc = [b.attn.attn_scale for b in model.transformer.h
                   if b.attn.attn_scale is not None]
            if _sc:
                _a = torch.stack(_sc)
                print(f"[trace {step:05d}] attn_scale mean={_a.mean().item():.3f} "
                      f"max={_a.max().item():.3f}", flush=True)
            print(f"[trace {step:05d}] x0={[round(v, 2) for v in model.x0_lambdas.tolist()]} "
                  f"resid={[round(v, 2) for v in model.resid_lambdas.tolist()]}", flush=True)

    # Corpus wall. Ten training shards hold 505.4 M tokens = 3856 steps of 2**17; a
    # second pass collapses the training loss (the n-gram tables memorise rows they have
    # already seen) and ruins val_bpb. The loader reports `epoch` from its read-ahead
    # refill, which runs up to 1000 documents in front of what it has yielded, so when it
    # first says 2 the batch just handed out is still pure epoch-1 data and stopping here
    # never trains on a repeat. On a B200 the run is time-limited and this never fires.
    if epoch > 1:
        print(f"\n[corpus wall] loader reached epoch {epoch} after {step} steps "
              f"({step * TOTAL_BATCH_SIZE / 1e6:.1f}M tokens); stopping before any "
              f"repeated data with {TIME_BUDGET - total_training_time:.0f}s unspent.",
              flush=True)
        break

    if step > 10 and total_training_time >= TRAIN_BUDGET:
        break

print()
_pending_fwd = None  # never set on the stopping step; belt and braces before eval
torch.cuda.synchronize()

if _bg_loader is not None:
    _bg_loader.close()

# ---------------- eval-time mixture: post-loop work, charged to the budget ----------------
mixer = None
if MIX_ENABLE:
    torch.cuda.synchronize()
    _tm0 = time.time()
    # MIX_ENT=0 only: network loss on the held-out slice with the final weights, through
    # the training step's compiled forward+backward; the gradients are never applied -- no
    # optimizer step follows and they are dropped below -- so nothing is learned from the
    # slice. (With MIX_ENT=1, as configured, the eval graph below is used instead.)
    slice_data = []
    for _sx, _sy in (mix_slice if not MIX_ENT else []):
        x.copy_(_sx)
        y.copy_(_sy)
        with torch.no_grad():
            _lk = get_lookups(x)
        _bo = bigram_embed(_lk[0])
        _to = trigram_embed(_lk[1])
        with autocast_ctx:
            _lt = model(x, y, reduction='none', bigram_offset=_bo, trigram_offset=_to,
                        pos_bucket_ids=_lk[2], bind_next_tok=_lk[3], uni_next_tok=_lk[4],
                        induction_dist_ids=_lk[5], recall_next_tok=_lk[6])
            _l = _lt.sum() / (y.view(-1) != -1).sum().clamp(min=1)
        _l.backward()
        slice_data.append((_sx, _sy, _lt.detach().float().clone(), None, None))
        del _lk, _bo, _to, _lt, _l
    # Optimizer state is dead from here on; give its memory to the tables.
    for _p in list(adamw_grad_params) + list(model.parameters()) + [bigram_embed.weight, trigram_embed.weight]:
        _p.grad = None
    optimizer.state.clear()
    for _so in (bigram_sparse_opt, trigram_sparse_opt):
        for _a in ("exp_avg", "exp_avg_sq", "cnt"):
            if hasattr(_so, _a):
                setattr(_so, _a, None)
    # Release any CUDA-graph pools of the training graph (a no-op with
    # max-autotune-no-cudagraphs); evaluation uses its own graph.
    from torch._inductor import cudagraph_trees as _cgt
    _cgt.reset_cudagraph_trees()
    gc.collect()
    torch.cuda.empty_cache()
    _free = torch.cuda.mem_get_info()[0]
    _bg_up = None
    if _MIX_BG:
        # The worker finished during training (normally); join it, then upload the stream,
        # the doc index and the slice chain on a side stream while the slice forward runs.
        _tj0 = time.time()
        _bg_ok = _extra.wait(float(os.environ.get("MIX_BG_WAIT", "120")))
        _t_join = time.time()
        if not _bg_ok:
            print(f"[mix] background build FAILED or not done (err={_extra.err!r}, events="
                  f"{[e.get('ev') for e in _extra.events]}); no mixture", flush=True)
            _extra.kill()
        else:
            _bg_tens = _extra.tensors()
            _bg_doc = {}
            _bg_stream = torch.cuda.Stream()

            def _bg_upload():
                _bg_doc.update(CorpusMixer.upload_doc(_bg_tens, device, _bg_stream))
            _bg_up = _threading.Thread(target=_bg_upload)
            _bg_up.start()
    if MIX_ENT:
        # Slice loss + entropy with the final weights through the eval-mode inference
        # graph (compiled before the timer started, see the warm-up), no_grad.
        model.eval()
        if MIX_EVAL_NOFP8:
            _set_eval_precision(True)   # stays on for the scored evaluation
        _ew = GPTWithBigram(eval_net, bigram_embed, BIGRAM_PRIME, BIGRAM_TABLE_SIZE,
                            trigram_embed, TRIGRAM_TABLE_SIZE)
        with autocast_ctx:
            for _sx, _sy in mix_slice:
                _l, _e, _tp = net_loss_ent(_ew, _sx, _sy)
                slice_data.append((_sx, _sy, _l, _e, _tp))
        del _ew, _l, _e, _tp
        torch.cuda.synchronize()
        _tm05 = time.time()
        _free = torch.cuda.mem_get_info()[0]
        print(f"[mix] slice forward (eval graph) {_tm05 - _tm0:.2f}s, free {_free / 1e9:.1f} GB", flush=True)
    _bg_doc_bytes = 0
    if _bg_up is not None:
        _bg_up.join()
        torch.cuda.synchronize()
        _bg_doc_bytes = sum(v.numel() * v.element_size() for v in _bg_doc.values())
        _free = torch.cuda.mem_get_info()[0]
    _tm_up = time.time()
    # (the doc-side tensors are already resident, so they are added back to the free
    # memory before the GPU table budget is taken)
    MIX_GPU_GB = max(0.0, min(float(os.environ.get("MIX_GPU_GB_MAX", "40")),
                              (_free + _bg_doc_bytes - float(os.environ.get("MIX_HEADROOM_GB", "48")) * 1e9) / 1e9))
    torch.cuda.synchronize()
    _tm1 = time.time()
    if _MIX_BG:
        if _bg_up is not None:
            mixer = CorpusMixer(vocab_size, BOS_TOKEN_ID, MAX_SEQ_LEN, None)
            mixer.load_prebuilt(_bg_tens, _bg_doc, _extra.result, device, mix_slice[0][0].numel())
            mix_ntok_all = mixer.S.numel()
            del _bg_doc
    else:
        _mix_S = mix_stream[:mix_ntok]
    if not _MIX_BG:
        mixer = CorpusMixer(vocab_size, BOS_TOKEN_ID, MAX_SEQ_LEN, mix_arena)
        mixer.build(_mix_S, pre_xy=[(_a, _b) for _a, _b, _, _, _ in slice_data])
        mix_ntok_all = _mix_S.numel()
        del mix_stream, _mix_S
    torch.cuda.synchronize()
    _tm2 = time.time()
    if mixer is not None:
        _token_bytes = get_token_bytes(device="cuda")
        mixer.fit_gate(slice_data, _token_bytes, keep=MIX_DIAG)
    torch.cuda.synchronize()
    _tm3 = time.time()
    total_training_time += _tm3 - _tm0
    if _MIX_BG:
        # The whole background build must sit inside the timed window: it may only be
        # spawned at (or after) the start of the first step that counts.
        if _extra.t_launch is None or _extra.t_launch < _t_timed0:
            raise RuntimeError("background build was not launched inside the timed loop")
        _ts = t_start_training
        print(f"[mix] bg build launched {_extra.t_launch - _t_timed0:+.3f}s after the timed window "
              f"opened (step 11), {_extra.t_launch - _ts:+.1f}s after t_start_training", flush=True)
        _evs = {e["ev"]: e for e in _extra.events}
        _st, _dn = _evs.get("start", {}), _evs.get("done", {})
        _tb0 = _st.get("t_build_start", float("nan"))
        _tb1 = _dn.get("t", float("nan"))
        print(f"[mix] bg worker: launched {_extra.t_launch - _ts:+.1f}s | slice sent {_extra.t_sent - _ts:+.1f}s | "
              f"tokenizers done {_st.get('t_tok', float('nan')) - _ts:+.1f}s | build start {_tb0 - _ts:+.1f}s | "
              f"stream {_evs.get('stream', {}).get('t', float('nan')) - _ts:+.1f}s | build done {_tb1 - _ts:+.1f}s "
              f"(build {_tb1 - _tb0:.1f}s) | mapped+populated "
              f"{(_extra.t_mapped or float('nan')) - _ts:+.1f}s | loop end {_tm0 - _ts:+.1f}s | "
              f"joined {_t_join - _ts:+.1f}s (wait {_t_join - _tj0:.3f}s)", flush=True)
        if _dn:
            print(f"[mix] bg tables: {_dn['N'] / 1e6:.1f}M tok in {_dn['docs']} docs ({_dn['pieces']} pieces, "
                  f"{_dn['slice_docs']} slice docs excluded), arena {_dn['size'] / 1e9:.1f} GB | per order "
                  f"[keys, merge, slice] s: {_dn['tim']}", flush=True)
        # step dt before / during / after the build (steps > 10)
        _dl = _dt_log[11:]

        def _dstat(a, b):
            v = sorted(d for t, d in _dl if a <= t < b)
            return (f"{len(v)} steps median {v[len(v) // 2] * 1000:.2f}ms mean {sum(v) / len(v) * 1000:.2f}ms"
                    if v else "0 steps")
        print(f"[mix] step dt: before build {_dstat(0, _tb0)} | during {_dstat(_tb0, _tb1)} | "
              f"after {_dstat(_tb1, float('inf'))}", flush=True)
        print(f"[mix] post-loop: cleanup+join+slice fwd {_tm05 - _tm0 if MIX_ENT else 0:.3f}s | upload join "
              f"{_tm_up - (_tm05 if MIX_ENT else _tm0):.3f}s ({_bg_doc_bytes / 1e9:.1f} GB doc side) | table "
              f"upload {_tm2 - _tm1:.3f}s | fit {_tm3 - _tm2:.3f}s (experts "
              f"{mixer.stats.get('t_fit_experts', 0) if mixer is not None else 0:.3f}s)", flush=True)
    if mixer is None:
        pass
    else:
        print(f"[mix] free before build {_free / 1e9:.1f} GB -> GPU tables {mixer.stats['ng_gpu_gb']:.1f} GB, "
              f"host {mixer.stats['ng_host_gb']:.1f} GB; doc index {mixer.stats.get('doc_index_gb', 0):.1f} GB, "
              f"ndocs {mixer.stats.get('ndocs', 0)}")
        print(f"[mix] tokens {mix_ntok / 1e6:.1f}M captured, {mix_ntok_all / 1e6:.1f}M in tables | slice fwd {_tm1 - _tm0:.2f}s | build {_tm2 - _tm1:.2f}s | "
              f"fit {_tm3 - _tm2:.2f}s | total {_tm3 - _tm0:.2f}s (reserve {MIX_RESERVE:.1f}s) | "
              f"training_seconds now {total_training_time:.1f}")
        _lb2 = math.log(2)
        print(f"[mix] slice nats net {mixer.stats['slice_nats_net']:.5f} -> mix {mixer.stats['slice_nats_mix']:.5f} "
              f"| mean gate w {[round(v, 4) for v in mixer.stats['slice_w_mean']]}", flush=True)

total_tokens = step * TOTAL_BATCH_SIZE

if KEY_MIX:
    # Diagnostic: the learned causal key mix per layer (a0 on k_t, a1 on k_{t-1}, ...).
    with torch.no_grad():
        for _li, _b in enumerate((model._orig_mod if hasattr(model, "_orig_mod") else model).transformer.h):
            _km = _b.attn.key_mix
            if _km is None:
                continue
            _m = [round(v, 3) for v in _km.mean(dim=1).tolist()]
            _lo = [round(v, 3) for v in _km[:, :KEY_OFFSET_LO].mean(dim=1).tolist()]
            _hi = [round(v, 3) for v in _km[:, KEY_OFFSET_LO:].mean(dim=1).tolist()]
            if os.environ.get("KEY_MIX_DUMP"):
                with open(os.environ["KEY_MIX_DUMP"], "a") as _f:
                    _f.write(f"{_li} " + " ".join(f"{v:.4f}" for v in _km.flatten().tolist()) + "\n")
            print(f"key_mix L{_li} off={int(_b.attn.key_offset)} mean={_m} lo={_lo} hi={_hi} "
                  f"frac(a1>0.5)={(_km[1] > 0.5).float().mean().item():.3f} "
                  f"max|a2|={(_km[2].abs().max().item() if _km.size(0) > 2 else 0.0):.3f}", flush=True)

if QMIX:
    # Diagnostic: the learned causal query mix per layer (a0 on q_t, a1 on q_{t-1}).
    with torch.no_grad():
        for _li, _b in enumerate((model._orig_mod if hasattr(model, "_orig_mod") else model).transformer.h):
            _qm = _b.attn.q_mix
            if _qm is None:
                continue
            print(f"q_mix L{_li} a0 mean={_qm[0].mean().item():.4f} a1 mean={_qm[1].mean().item():.4f} "
                  f"a1 absmax={_qm[1].abs().max().item():.4f} "
                  f"a0 range=[{_qm[0].min().item():.3f},{_qm[0].max().item():.3f}]", flush=True)

# ---- evaluation without FP8 quantization -------------------------------------------
# Training uses FP8 e4m3 GEMMs for the MLP and the fused QKV projection (a speed
# trade). At inference that quantization noise is pure loss the trained weights do not
# need to pay, so the same weights are evaluated with bf16 MLP/QKV and an fp32
# lm_head. Training above is untouched; only the evaluation path changes. The
# optimizer states are dropped first because the eager (uncompiled) eval needs the
# headroom. (EVAL_NO_FP8, EVAL_FP32_LM_HEAD and _FP32Head are defined before the
# mixer's eval-graph warm-up, which uses them too.)

if os.environ.get("EVAL_ALSO_FP8"):
    # Diagnostic only (off in the scored config): the compiled FP8 eval number,
    # on the same weights, for a paired delta.
    model.eval(); bigram_embed.eval(); trigram_embed.eval()
    with autocast_ctx:
        _fp8_bpb = evaluate_bpb(GPTWithBigram(model, bigram_embed, BIGRAM_PRIME, BIGRAM_TABLE_SIZE,
                                              trigram_embed, TRIGRAM_TABLE_SIZE),
                                tokenizer, DEVICE_BATCH_SIZE)
    print(f"[diag] fp8 compiled eval val_bpb: {_fp8_bpb:.6f}", flush=True)

if EVAL_NO_FP8 and not (mixer is not None and MIX_ENT):
    # (With the mixer, evaluation runs through the eval graph compiled before training,
    # the same graph the gate was fitted on; see the MixtureEval wrap below.)
    del optimizer, bigram_sparse_opt, trigram_sparse_opt
    gc.collect()
    torch.cuda.empty_cache()
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    for blk in base_model.transformer.h:
        blk.mlp.use_fp8 = False
    USE_FP8_QKV = 0  # read at call time by CausalSelfAttention.forward
    USE_FP8_CPROJ = 0  # likewise: gated c_proj back to the bf16 y * g -> nn.Linear path
    if EVAL_FP32_LM_HEAD:
        base_model.lm_head = _FP32Head(base_model.lm_head.weight)
    eval_gpt = base_model
else:
    eval_gpt = model

model.eval()
bigram_embed.eval()
trigram_embed.eval()
eval_model = GPTWithBigram(eval_net if (mixer is not None and MIX_ENT) else eval_gpt,
                           bigram_embed, BIGRAM_PRIME, BIGRAM_TABLE_SIZE,
                           trigram_embed, TRIGRAM_TABLE_SIZE)
if mixer is not None:
    eval_model = MixtureEval(eval_model, mixer)
_t_eval = time.time()
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", str(DEVICE_BATCH_SIZE)))
val_bpb = None
try:
    with autocast_ctx:
        val_bpb = evaluate_bpb(eval_model, tokenizer, EVAL_BATCH_SIZE)
except torch.OutOfMemoryError:
    pass
if val_bpb is None:
    # Eager eval at the training batch did not fit: same tokens, smaller batches.
    print(f"eval OOM at batch {EVAL_BATCH_SIZE}; retrying at batch 16", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    with autocast_ctx:
        val_bpb = evaluate_bpb(eval_model, tokenizer, 16)

t_end = time.time()
print(f"eval_seconds: {t_end - _t_eval:.1f}")
if mixer is not None and MIX_DIAG and MIX_ENT and getattr(mixer, "slice_cache", None) is not None:
    # Diagnostics only, after the score: nothing here feeds val_bpb. The eval's cached
    # blocks are released first; the validity checks below otherwise OOM on a B200.
    gc.collect()
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            _td0 = time.time()
            _lb2 = math.log(2)
            _sc = mixer.slice_cache
            _nb = mixer.base_names()
            _nc = mixer.feat_names
            _extra_f = [k for k in _DOCFEAT_SIZES if k not in _nc]
            _vars = {"base(i005)": _nb, "cur": _nc}
            for _k in _DOCFEAT_SIZES:
                _vars[f"base+{_k}"] = _nb + [_k]
            for _k in _extra_f:
                _vars[f"cur+{_k}"] = _nc + [_k]
            _ev = (_sc["bidx"] % 2) == 0
            _od = ~_ev
            _fe = {k: v[_ev] for k, v in _sc["feats"].items()}
            _tokb = get_token_bytes(device="cuda")
            # val batches (first MIX_DIAG_NVAL of the eval stream), network outputs once
            _vl = make_dataloader(tokenizer, EVAL_BATCH_SIZE, MAX_SEQ_LEN, "val")
            _vb = []
            with autocast_ctx:
                for _i in range(int(os.environ.get("MIX_DIAG_NVAL", "24"))):
                    _x, _y, _ = next(_vl)
                    _x, _y = _x.clone(), _y.clone()
                    _l, _e, _tp = net_loss_ent(eval_model._base, _x, _y)
                    _vb.append((_x, _y, _l, _e, _tp))
            _vd = mixer.collect(_vb, _tokb)
            _vbytes = torch.cat([_tokb[b[1].reshape(-1)] for b in _vb])
            _vbytes = _vbytes[_vbytes > 0].double()
            _bpt = _vbytes.mean().item()            # bytes per scored token (val diag batches)
            _vnet = -torch.log(_vd["pn"].clamp(min=1e-30)).mean().item()
            print(f"[diag] A/B (gate fit on even slice batches -> odd slice nats | fit on full slice -> "
                  f"{len(_vb)} val batches nats, bpb-equiv); val net {_vnet:.5f}", flush=True)
            _res = {}
            for _nm, _names in _vars.items():
                _pe, _, _ = mixer._fit_sparse(_fe, _names, _sc["pn"][_ev], _sc["pg"][_ev], _sc["pd"][_ev])
                _lo, _ = mixer.eval_gate(_sc, _pe, _names, _od)
                _pf, _, _ = mixer._fit_sparse(_sc["feats"], _names, _sc["pn"], _sc["pg"], _sc["pd"])
                _lv, _wv = mixer.eval_gate(_vd, _pf, _names)
                _res[_nm] = (_lo, _lv)
                print(f"[diag]   {_nm:18s} slice-odd {_lo:.6f} | val {_lv:.6f} (bpb {_lv / _lb2 / _bpt:.6f}) "
                      f"| val w {[round(v, 4) for v in _wv.mean(0).tolist()]}", flush=True)
            _b0 = _res["base(i005)"]
            for _nm, (_lo, _lv) in _res.items():
                print(f"[diag]   delta vs base: {_nm:18s} slice-odd {_lo - _b0[0]:+.6f} nats | val "
                      f"{_lv - _b0[1]:+.6f} nats ({(_lv - _b0[1]) / _lb2 / _bpt:+.6f} bpb)", flush=True)
            # scored gate: mean doc weight per doc-share / rank bucket on the val batches
            _ls, _ws = mixer.eval_gate(_vd, mixer.gate, mixer.feat_names)
            print(f"[diag] scored gate on val batches: nats {_ls:.6f}, mean w {[round(v, 4) for v in _ws.mean(0).tolist()]}")
            for _fn, _nbk in (("_dsb", _NDS), ("_rb", _NRB)):
                _f = _vd["feats"][_fn]
                _row = []
                for _b in range(_nbk):
                    _m = _f == _b
                    _c = int(_m.sum().item())
                    if _c:
                        _row.append(f"{_b}:{_ws[_m, -1].mean().item():.4f}(n={_c / _f.numel():.3f})")
                print(f"[diag]   doc w by {_fn[1:]} bucket: " + " ".join(_row), flush=True)
            # ---- validity checks ----
            _x, _y, _l, _e, _tp = _vb[1]
            _B, _T = _x.shape
            _n = _x.numel()
            # (1) normalisation: full-vocab sums of each expert at random positions
            _g = torch.Generator(device="cuda").manual_seed(7)
            _dp = torch.randperm(_n, device="cuda", generator=_g)[:256]
            _exd = mixer.experts(_x, _y, dense_pos=_dp)
            _err_ng = (_exd["dense"]["ng"] - 1).abs().max().item()
            _err_doc = (_exd["dense"]["doc"] - 1).abs().max().item() if "doc" in _exd["dense"] else float("nan")
            print(f"[check] normalisation at {_dp.numel()} positions: max|sum p_ng - 1| {_err_ng:.2e}, "
                  f"max|sum p_doc - 1| {_err_doc:.2e} (gate is a softmax -> mixture sums to 1)", flush=True)
            del _exd
            # (2) target independence: features other than the causal feedback EMAs, and the
            # doc/n-gram summaries, are unchanged under random targets
            _yr = torch.randint(0, vocab_size, _y.shape, device="cuda", generator=_g)
            _ex1 = mixer.experts(_x, _y)
            _ex2 = mixer.experts(_x, _yr)
            _p1n = torch.exp(-_l.double())
            _f1 = mixer._features(_ex1, _p1n, _B, _T, _e, _tp)
            _f2 = mixer._features(_ex2, _p1n, _B, _T, _e, _tp)
            _bad = [k for k in _f1 if not k.startswith("fb_") and not torch.equal(_f1[k], _f2[k])]
            _bad += [k for k in ("dshare", "darg", "dmax", "dcc", "ngo", "ngc") if not torch.equal(_ex1[k], _ex2[k])]
            print(f"[check] target independence (random y): mismatched {_bad if _bad else 'none'}", flush=True)
            del _ex1, _ex2, _f1, _f2
            # (3) batch independence of the mixer: first 8 rows alone vs inside the batch
            _nf, _, _wf, _ = mixer.mix_nll(_x, _y, _l, _e, _tp, return_parts=True)
            _k8 = 8 * _T
            _n8, _, _w8, _ = mixer.mix_nll(_x[:8], _y[:8], _l[:_k8], _e[:_k8], (_tp[0][:_k8], _tp[1][:_k8]),
                                           return_parts=True)
            print(f"[check] batch independence (8 rows vs 64): max|dnll| {(_n8 - _nf[:_k8]).abs().max().item():.2e}, "
                  f"max|dw| {(_w8 - _wf[:_k8]).abs().max().item():.2e}", flush=True)
            # (4) causality: perturb x[r, t+1:] (non-BOS random tokens) in 4 rows and re-run
            # network + mixer; nll at s < t and gate weights at s <= t must not change
            _xp, _yp = _x.clone(), _y.clone()
            _ts = [100, 517, 1024, 1900]
            for _r, _t in zip(range(4), _ts):
                _rt = torch.randint(0, vocab_size, (_T - _t - 1,), device="cuda", generator=_g)
                _rt = torch.where(_rt == BOS_TOKEN_ID, (_rt + 1) % vocab_size, _rt)
                _xp[_r, _t + 1:] = _rt
                _yp[_r, _t:_T - 1] = _rt
            with autocast_ctx:
                _lp, _ep, _tpp = net_loss_ent(eval_model._base, _xp, _yp)
            _np_, _, _wp, _ = mixer.mix_nll(_xp, _yp, _lp, _ep, _tpp, return_parts=True)
            _dn, _dw, _dnet = 0.0, 0.0, 0.0
            for _r, _t in zip(range(4), _ts):
                a0 = _r * _T
                _dn = max(_dn, (_np_[a0:a0 + _t] - _nf[a0:a0 + _t]).abs().max().item())
                _dw = max(_dw, (_wp[a0:a0 + _t + 1] - _wf[a0:a0 + _t + 1]).abs().max().item())
                _dnet = max(_dnet, (_lp[a0:a0 + _t] - _l[a0:a0 + _t]).abs().max().item())
            _dother = (_np_[4 * _T:] - _nf[4 * _T:]).abs().max().item()
            print(f"[check] causality (suffix perturbed in 4 rows): prefix max|dnll| {_dn:.2e} "
                  f"(net alone {_dnet:.2e}), prefix max|dw| {_dw:.2e}; untouched rows max|dnll| {_dother:.2e}",
                  flush=True)
            print(f"[diag] diagnostics took {time.time() - _td0:.1f}s", flush=True)
            del _vb, _vd
    except Exception as _e:  # diagnostic only; never fail the run
        import traceback
        traceback.print_exc()
        print(f"[diag] skipped: {_e!r}", flush=True)
if (mixer is not None and _MIX_BG and getattr(mixer, "S", None) is not None
        and os.environ.get("MIX_LEAKCHECK", "1") == "1"):
    # Diagnostic only (after the score): share of the gate slice's 32-token windows that
    # occur in the table stream, split into documents the loader handed to a trained
    # batch ("captured") and the rest ("extra": unread, left in the buffer). A cropped
    # long document counts as captured here. Shared boilerplate gives similar rates; a
    # leaked slice document would make the rates jump.
    try:
        with torch.no_grad():
            _K = 32
            _mm = torch.tensor(_MIX_MULTS_X[:_K], dtype=torch.long, device=device)

            def _wh(v):
                v = v.long() + 1
                n = v.numel() - _K + 1
                h = torch.zeros(n, dtype=torch.long, device=v.device)
                for i in range(_K):
                    h += v[i:i + n] * _mm[i]
                return h
            _hs = torch.unique(torch.cat([_wh(_a.reshape(-1)) for _a, _b in mix_slice]))
            _Sa = mixer.S
            _bt = _extra.tensors()
            _dsl = _bt["ds"].to(device)
            _gk = _bt["gidx"].to(device)
            _sn, _sb = _extra.snap_n, _extra.snap_buf
            _trn = (_gk < _sn) & ~torch.isin(_gk, torch.tensor(_sb, dtype=torch.long, device=device))
            _dlen = torch.diff(_dsl, append=torch.tensor([_Sa.numel()], device=device))
            _hit_tr = _hit_ot = 0
            _ch = 1 << 25
            for s0 in range(0, _Sa.numel() - _K + 1, _ch):
                h = _wh(_Sa[s0:min(_Sa.numel(), s0 + _ch + _K - 1)])
                j = torch.searchsorted(_hs, h).clamp(max=_hs.numel() - 1)
                hp = torch.nonzero(_hs[j] == h).squeeze(1) + s0
                d = torch.searchsorted(_dsl, hp, right=True) - 1
                t_ = _trn[d]
                _hit_tr += int(t_.sum().item())
                _hit_ot += int((~t_).sum().item())
            _ntr = int(_dlen[_trn].sum().item())
            _not = int(_dlen[~_trn].sum().item())
            print(f"[mix] leak check: hits per slice 32-gram: captured {_hit_tr / _hs.numel():.5f} "
                  f"({_hit_tr / max(1, _ntr) * 1e6:.3f}/Mtok, {_ntr / 1e6:.1f}M tok in {int(_trn.sum())} docs), "
                  f"in extra {_hit_ot / _hs.numel():.5f} ({_hit_ot / max(1, _not) * 1e6:.3f}/Mtok, "
                  f"{_not / 1e6:.1f}M tok in {int((~_trn).sum())} docs)", flush=True)
    except Exception as _e:  # diagnostic only; never fail the run
        print(f"[mix] leak check skipped: {_e!r}")
if mixer is not None and eval_model.dump is not None:
    _dd = os.environ["MIX_DUMP"]
    os.makedirs(_dd, exist_ok=True)
    torch.save({"val_loss_net": [d[0] for d in eval_model.dump], "val_ent": [d[1] for d in eval_model.dump],
                "slice": [(a.cpu(), b.cpu(), c.cpu(), None if e is None else e.cpu()) for a, b, c, e, _ in slice_data],
                "mix_ntok": mix_ntok, "nslice": MIX_NSLICE, "steps": step}, os.path.join(_dd, "dump.pt"))
    print(f"[mix] dumped to {_dd}")
steady_state_mfu = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) /
                    total_training_time / B200_BF16_PEAK_FLOPS) if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params_gpt / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"bigram_table_size: {BIGRAM_TABLE_SIZE:,}")
print(f"bigram_lr:        {BIGRAM_LR}")
print(f"trigram_table_size: {TRIGRAM_TABLE_SIZE:,}")
print(f"trigram_lr:       {TRIGRAM_LR}")
print(f"pos_bucket_lr:    {POS_BUCKET_LR}")
print(f"num_pos_buckets:  {NUM_POS_BUCKETS}")
print(f"momentum_delay:   {MOMENTUM_WARMDOWN_DELAY}")
print(f"momentum_end:     {MOMENTUM_END}")
print(f"mlp_fc_lr:        {MLP_FC_LR}")
print(f"matrix_lr:        {MATRIX_LR}")
