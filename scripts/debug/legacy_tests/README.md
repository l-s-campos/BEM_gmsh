# Former `test/test_*.jl` suites

These are **not** part of CI (`julia --project=. test/runtests.jl`).
They keep paper/sweep coverage. Paths that used `@__DIR__` assumed the
files lived in `test/` — from here, repo root is `../../..`.

```bash
julia --project=. scripts/debug/legacy_tests/test_sbm.jl
```
