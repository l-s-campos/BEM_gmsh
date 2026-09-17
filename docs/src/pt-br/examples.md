# Exemplos

Scripts longos em `scripts/<família>/` (índice: `scripts/README.md`).

```julia
using BEM
dad = format2d(quadrado(ndiv=30, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
assemble!(dad, 20)
solve(dad)
@show rel_error(dad)
```

Campo exato das BCs padrão: ``T=x``. Mais trechos: [Examples](../examples.md).
