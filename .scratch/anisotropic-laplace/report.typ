= Three 2D strategies for anisotropic Laplace BEM

Status: ready-for-human

Orthotropic heat conduction on the unit square with a circular hole,
compared to a FEniCS reference on the right edge $x=1$.
This note is *two-dimensional only*.

Package flux throughout: $q = - n dot K nabla u$ (outward $n$).
FEniCS stores $K nabla T dot n$, so a right-edge influx of $+5$ is
$q = -5$ in BEM.jl.

== The problem

$
  nabla dot (K nabla u) = 0 quad "in" Omega,
  quad K = mat(5, 0; 0, 0.5).
$

$Omega$ is the unit square minus a disk of radius $0.25$ at $(0.5,0.5)$. Mesh builder:
`placa_furo_orto`. Quadratic discontinuous collocation
(`ordem=2`, `tipo=2`).

#figure(
  image("figures/geometry.png", width: 72%),
  caption: [
    Domain and BCs. Red: Dirichlet $T=0$ on $x=0$.
    Blue: Neumann $q=-5$ on $x=1$ (FEniCS $K nabla T dot n = 5$).
    All other edges, including the hole, are insulated ($q=0$).
    Grey points are interior collocation used by DIBEM (S1 and S3).
    S2 uses the boundary only.
  ],
)

The hole constricts the flux, so $T(x=1,y)$ peaks near $y=0.5$.
FEniCS ($h=0.02$) gives $T_"mid" = T(1,0.5) = 2.769$.
Without the hole one would have $T=x$ and $T(1,y)=1$.

Reference file: `data/Laplace/fenics_ortho_plate_Tright.csv`.

== Isotropic collocation BEM (shared by S1 and S3)

For isotropic conductivity $k$ the collocation BIE is

$
  c(ξ) u(ξ) + integral_Gamma u (partial Phi)/(partial n) dif Gamma
  = integral_Gamma q_"iso" Phi dif Gamma + "volume terms",
$

with $Phi = - ln r / (2 pi k)$ and package $q_"iso" = - k (partial u)/(partial n)$.
In matrix form, after moving the free term into the diagonal of $H$,

$ H u - G q_"iso" = "rhs". $

S1 and S3 assemble this system with `Laplace(1)` (so $k=1$ in $Φ$),
then *correct* $q_"iso"$ and the volume residual for anisotropy.
S2 never uses this isotropic pair: it replaces $Φ$ by the anisotropic
Green's function.

DIBEM supplies a mass matrix $M$ such that, for a domain source $f$,

$ H u - G q_"iso" = M f, $

with $f$ interpolated by PHS radial basis functions and the volume
integrals taken analytically in the radial coordinate (`int(rbf,x,xj)`
and the RIM primitive of $Φ$).


== Strategy 2 — anisotropic fundamental solution

Direct collocation with the anisotropic Green's function
(`AnisotropicLaplace(K)`). No DIBEM, no interiors, no Neumann map.

$
  Phi(r) = - (ln sqrt(r dot K^(-1) r)) / (2 pi sqrt(det K)),
  quad rho^2 = r dot K^(-1) r.
$

The dual kernel is the *physical* flux of $Phi$:

$ q^* = - n dot K nabla Phi. $

Because $K K^(-1) r = r$,

$
  nabla Phi = - (K^(-1) r) / (2 pi sqrt(det K) rho^2),
  quad q^* = (r dot n) / (2 pi sqrt(det K) rho^2).
$

The numerator is $r dot n$, not $n dot K^(-1) r$. The isotropic limit
$K = k I$ recovers `Laplace(k)` ($H$ has no extra $k$ in the
denominator, as in `Fundamental.jl`).

The linear system is the usual mixed BEM: $H u = G q$ with this
$(Phi, q^*)$, Dirichlet columns swapped for $-G$.

== Strategy 1 — isotropic Poisson + DIBEM residual (Hessian)

Keep the *isotropic* kernels and push anisotropy into a domain source.
With $K = k I + Delta K$,

$ nabla dot (K nabla u) = k nabla^2 u + nabla dot (Delta K nabla u) = 0, $

so

$
  nabla^2 u = - f^*,
  quad f^* = (1/k) nabla dot (Delta K nabla u) approx A_f u.
$

Package DIBEM identity $H u - G q_"iso" = M ∇^2 u$ then becomes

$
  H u - G q_"iso" = - M A_f u
  quad (b = 0),
$

or $(H + M A_f) u - G q_"iso" = 0$. The Neumann map replaces
$q_"iso"$ as above.

$A_f$ is built from a Hessian of nodal $u$. A *global* PHS interpolant
through the hole gives $|f^*| ~ O(100)$ even when the true residual is
modest. S1 therefore uses *local* RBF-FD (`nlocal = 21`, PHS3 with
`poly_deg=2`) on all collocation points (boundary + cell centroids).
The Hessian of the polynomial tail is filled ($x^2 -> 2$, $x y -> 1$,
…). On the coarsest mesh ($n=168$) that stencil is too thin around the
hole and S1 fails; medium and fine meshes are fine.

S1 *does* form $"Hess"(u)$. That is the difference from S3.

== Strategy 3 — one integration by parts, then DIBEM (no Hessian)

Do *not* differentiate $u$ twice. Integrate $nabla dot(Delta K nabla u)$ by parts
once against the isotropic fundamental solution $Phi$:

$
  integral_Omega Phi f^* dif Omega
  = (1/k) integral_Gamma Phi (n dot Delta K nabla u) dif Gamma
  - (1/k) integral_Omega nabla Phi dot (Delta K nabla u) dif Omega.
$

Three discrete pieces, all existing DIBEM machinery:

+ $integral_Omega Phi b$: the mass $M$ (here $b=0$).
+ $integral_Gamma Phi (n dot Delta K nabla u)$: $G$ times the same split
  $n dot Delta K nabla u = S_Gamma u - n Delta n q_"iso"$ as in the Neumann map.
+ $integral_Omega nabla Phi dot w$ with $w = Delta K nabla u$: Loeffler operator $N_alpha$ whose
  kernel is $partial_alpha Phi$. The radial integrals use the same `int(rbf)`
  coefficients $c$ as $M$, and the RIM primitive of $nabla Phi$ is
  $- R e / (2 pi k)$ in 2D ($e = r / R$).

$nabla u$ is an RBF interpolant of *values* (first derivatives only).
There is no Hessian of $u$.

After collecting terms,

$
  H u - G q_"iso"
  + (1/k) G (S_Gamma u - n Delta n q_"iso")
  - (1/k) A_V u = 0,
$

with $A_V = sum_(alpha,beta) Delta K_(alpha beta) N_alpha D_beta$ and $D_beta u approx partial_beta u$.
The Neumann map is then applied exactly as in S1.

== Right-edge profiles vs FEniCS

#figure(
  image("figures/profiles_medium.png", width: 100%),
  caption: [
    Medium mesh ($"lc"=0.06$, $n=285$ quadratic). All three strategies
    follow the FEniCS bump. S1 is slightly high on the lower ligament
    ($y < 0.4$); S2 and S3 lie on top of FEniCS.
  ],
)

#figure(
  image("figures/errors_medium.png", width: 100%),
  caption: [
    Pointwise error $T_"BEM" - T_"FEniCS"$ on $x=1$, same mesh.
    S2 error is a small positive bias ($~0.005$–$0.01$). S3 stays
    within $plus.minus 0.01$ except at the corners. S1 reaches $+0.04$ on the
    lower half.
  ],
)

#figure(
  image("figures/profiles_meshes.png", width: 100%),
  caption: [
    Three meshes. S2 and S3 are stable from coarse to fine.
    Coarse S1 is *not* plotted on scale: $T_"mid" = -1.57$ (RMS $1.23$)
    because the local Hessian stencil is too small around the hole.
    Medium and fine S1 recover the bump (RMS $≈ 0.011$).
  ],
)

== Table (FEniCS $T_"mid" = 2.769$)

#figure(
  table(
    columns: 6,
    [mesh], [$"lc"$], [$n$], [S2 RMS / $T_"mid"$], [S1 RMS / $T_"mid"$], [S3 RMS / $T_"mid"$],
    [coarse], [$0.10$], [$168$], [$0.0040$ / $2.780$], [$1.23$ / $-1.57$], [$0.0070$ / $2.745$],
    [medium], [$0.06$], [$285$], [$0.0048$ / $2.782$], [$0.0108$ / $2.767$], [$0.0038$ / $2.763$],
    [fine], [$0.04$], [$420$], [$0.0044$ / $2.782$], [$0.0104$ / $2.736$], [$0.0025$ / $2.773$],
  ),
  caption: [Relative $L^2$ error of $T(x=1,y)$ vs FEniCS, and the value at $y=0.5$.],
)

S2 is *not* larger than FEniCS on these 2D models: $T_"mid"$ is $2.78$
against $2.77$ (relative $0.5%$). S3 is the closest of the three on
medium and fine meshes. S1 needs a mesh fine enough for the local
Hessian; once that holds, it is within about $1%$ RMS.

== Manufactured anisotropic Poisson (known solution)

The holed-plate comparison has no closed-form $u$. To measure
*convergence* of the two DIBEM strategies, use the manufactured field
on the unit square (no hole)

$
  u = sin(pi x) sin(pi y),
  quad K = mat(5, 0; 0, 0.5).
$

Then $nabla u = pi (cos(pi x) sin(pi y), sin(pi x) cos(pi y))$ and

$ nabla dot (K nabla u) = - pi^2 (k_x + k_y) u, $

so the package source in $nabla dot (K nabla u) = -b$ is
$b = pi^2 (k_x + k_y) u$. Physical flux $q = - n dot K nabla u$.
All-Dirichlet data from $u$; the error is the relative $L^2$ norm on
*interior* collocation (boundary $u$ is prescribed). Quadratic
elements (`ordem=2`, `tipo=2`). Helper: `ana_aniso_poisson_sin`.

S2 is omitted: the anisotropic Green's function treats the homogeneous
BIE, so a volume source has to go through DIBEM (S1 or S3).

#figure(
  table(
    columns: 6,
    [ndiv], [$h$], [$n$], [$n_i$], [S1 error], [S3 error],
    [4], [$0.333$], [36], [9], [$1.57 times 10^(-1)$], [$1.28 times 10^(-1)$],
    [6], [$0.200$], [60], [25], [$4.82 times 10^(-2)$], [$3.29 times 10^(-2)$],
    [8], [$0.143$], [84], [49], [$1.11 times 10^(-2)$], [$1.25 times 10^(-2)$],
    [12], [$0.091$], [132], [121], [$5.67 times 10^(-3)$], [$4.04 times 10^(-3)$],
    [16], [$0.067$], [180], [225], [$3.15 times 10^(-3)$], [$2.03 times 10^(-3)$],
  ),
  caption: [Interior relative $L^2$ error. $h = 1\/("ndiv" - 1)$.],
)

Observed order from the last two meshes: S1 $approx 1.9$, S3
$approx 2.2$ (both track $O(h^2)$). S3 is slightly more accurate on
the finer meshes.

#figure(
  image("figures/aniso_poisson_convergence.png", width: 85%),
  caption: [
    Log–log interior error versus $h$. Green: S1 (Hessian residual).
    Purple: S3 (IBP, no Hessian). Dashed $O(h^2)$, dotted $O(h^3)$.
  ],
)
