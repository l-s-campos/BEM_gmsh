# FMM

Vendored fast multipole methods. Load with `using BEM.FMM`.
Trees come from `BEM.HMatrices`.

```@docs
BEM.FMM
BEM.FMM.KelvinFMMMatrix
BEM.FMM.fmm_kelvin2d_matrix
BEM.FMM.KelvinFMMMatrix3D
BEM.FMM.fmm_kelvin3d_matrix
```

```julia
using BEM.FMM
KF = FMM.fmm_laplace2d_double_layer_matrix(P, N; n_boundary=n)
K3 = FMM.fmm_kelvin3d_matrix(P3; μ=1.0, ν=0.3)
```
