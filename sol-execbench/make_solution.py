"""把一个 kernel 目录拼成官方评测吃的 solution.json。

    python make_solution.py l2-029 > solution.json

目录里的 spec.json 是提交时用的 spec（入口、语言、依赖、编译选项），其余源文件按相对路径原样放进
sources。拼出来的 solution.json 可以直接交给官方评测器，或者上传到官方提交接口。
"""
import json
import sys
from pathlib import Path

EXT = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"}


def main() -> int:
    d = Path(sys.argv[1])
    spec = json.loads((d / "spec.json").read_text())
    sources = [{"path": str(p.relative_to(d)), "content": p.read_text()}
               for p in sorted(d.rglob("*")) if p.is_file() and p.suffix in EXT]
    entry = spec["entry_point"].split("::")[0]
    if entry not in {s["path"] for s in sources}:
        raise SystemExit(f"入口 {entry} 不在源文件里")
    json.dump({"name": d.name, "author": "ERNIE Evolve", "spec": spec, "sources": sources},
              sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
