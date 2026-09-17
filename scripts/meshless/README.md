# Meshless comparisons

Extra environment for scripts that compare BEM against Macchiato / RBF-FD /
WhatsThePoint. Those packages are **not** in the main `Project.toml`.

```bash
julia --project=scripts/meshless -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts/meshless scripts/meshless/macchiato_heat_compare.jl
```

