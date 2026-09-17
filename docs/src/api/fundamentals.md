# Fundamental solutions

Kernels are `fundamental(props, r, n) → KernelPair` with `.U` (single layer)
and `.T` (double layer). Laplace flux is ``q = -k ∂T/∂n``; Helmholtz uses
the acoustic ``q = ∂u/∂n``. 3-D Helmholtz is ``e^{iκR}/(4πR)``.
Hypersingular kernels: [`fundamental_hyper`](@ref) (2-D/3-D Helmholtz,
2-D/3-D Laplace).

H-matrix / NNCA assembly that only needs one layer should call
[`fundamental_U`](@ref) / [`fundamental_T`](@ref) (2-D Laplace uses
``\log R^2``, no extra ``\sqrt``).

```@docs
fundamental
fundamental_U
fundamental_T
fundamental_hyper
fundamental_stress
fundamental_grad
lekhnitskii_params
```
