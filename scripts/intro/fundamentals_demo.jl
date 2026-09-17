# Demo: fundamental solutions catalogue
using DrWatson
@quickactivate :BEM

println("="^60)
println(" Fundamental solutions demo")
println("="^60)

r = Point2D(0.3, 0.4)
n = Point2D(1.0, 0.0) / 1.0
nf = Point2D(0.0, 1.0)
R = norm(r)

# --- Laplace ---------------------------------------------------------------
lap = Laplace(1.0)
kp = fundamental(lap, r, n)
println("\n[Laplace 2D]  R=$R")
println("  G (single)  = ", kp.U)
println("  H (double)  = ", kp.T)
kh = fundamental_hyper(lap, r, n, nf)
println("  hyper G,H   = ", kh.U, ", ", kh.T)

# 3D
r3 = Point3D(0.3, 0.4, 0.5)
n3 = Point3D(0.0, 0.0, 1.0)
kp3 = fundamental(lap, r3, n3)
println("\n[Laplace 3D]")
println("  G, H = ", kp3.U, ", ", kp3.T)

# --- Helmholtz -------------------------------------------------------------
helm = Helmholtz(; ω=2π, c=1.0)
println("\n[Helmholtz] κ = ", wavenumber(helm))
kph = fundamental(helm, r, n)
println("  G = ", kph.U)
println("  H = ", kph.T)

# --- Isotropic elasticity (Tensorial Mat) ----------------------------------
elast = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
kpe = fundamental(elast, r, n)
println("\n[Kelvin 2D]  μ = ", shear_modulus(elast))
println("  U =\n", kpe.U)
println("  T =\n", kpe.T)

Ds = fundamental_stress(elast, r, n)
println("  D[:,:,1] stress kernel slice:\n", Ds.D[:, :, 1])
println("  S[:,:,1] stress kernel slice:\n", Ds.S[:, :, 1])

Ux, Tx, Uy, Ty = fundamental_grad(elast, r, n)
println("  ∂U/∂x =\n", Ux)

# 3D Kelvin
kpe3 = fundamental(elast, r3, n3)
println("\n[Kelvin 3D]")
println("  U =\n", kpe3.U)
println("  T =\n", kpe3.T)

# --- Anisotropic (Lekhnitskii) ---------------------------------------------
# graphite-epoxy-like orthotropy
pars = lekhnitskii_params(25.0, 1.0, 0.5, 0.25; θ_deg=30)
aniso = AnisotropicElasticity(pars)
kpa = fundamental(aniso, r, zero(r), n)
println("\n[Lekhnitskii 2D] μ = ", pars.mi)
println("  U =\n", kpa.U)
println("  T =\n", kpa.T)

println("\nDone.")
