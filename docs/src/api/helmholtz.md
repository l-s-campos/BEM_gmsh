# Helmholtz / Burton–Miller

Time-harmonic kernels (2-D Hankel ``H_0^{(1)}``, 3-D ``e^{iκR}/(4πR)``)
and the correlato Burton–Miller combination of CBIE + HBIE.
[`H_G_hyper`](@ref) assembles the hypersingular pair for Helmholtz in
2-D and 3-D (Laplace HBIE remains 2-D).

Far nodal lumping follows [`auto_near_factor`](@ref): keep `near_factor=1.5`
when ``κ L_{\max}\le 0.6`` (≳10 points per wavelength), otherwise
`near_factor=Inf`. Pass a `Real` to [`assemble!`](@ref) to override.

```@docs
Helmholtz
auto_near_factor
H_G_hyper
combine_burton_miller
```
