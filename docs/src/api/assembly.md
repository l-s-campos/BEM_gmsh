# Assembly

Student entry: [`assemble!`](@ref). Dense path is [`H_G_full_direct`](@ref);
Laplace hierarchical path is [`H_G_Hmat`](@ref); 2-D GPU path is
[`H_G_gpu`](@ref) (Laplace or isotropic elasticity, `method=:gpu`).

```@docs
assemble!
H_G_full_direct
H_G_Hmat
H_G_gpu
gpu_float_support
ColWeightedOp
```
