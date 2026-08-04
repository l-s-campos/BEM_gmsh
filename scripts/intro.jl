using DrWatson
@quickactivate :BEM

include(datadir("Laplace", "Laplace_dad.jl"))

# Create a Laplace problem with k=1 (conductivity)
properties = Laplace(1.0)
msh = quadrado(ndiv=20, show=false)

# Load the geometry and create BEMdata
dad = format2d(msh, properties)
# dad = format3d(datadir("Laplace", "cubo.geo"), properties, pontointerno=false)

# Analytical field for the default square BCs: T = x  (q = -k ∂T/∂n)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))

# Dense assembly (use H_G_Hmat(dad) for large meshes)
H_G_full_direct(dad, 20)
DIBEM(dad)

# plot_geo(dad)

solve(dad)
@show rel_error(dad)

# lines(dad.T)
# Δt, tf = 0.01, 5.0
# solve_Houbolt(dad, Δt, tf)
# sol = solve_transient(dad, Δt, tf)
# sol = solve_transient_o2(dad, Δt, tf)

# --- H-matrix path (optional) ---
# dadH = format2d(msh, properties; pontointerno=false)
# attach_analytical!(dadH, ana_laplace_linear(; direction=SA[1.0, 0.0]))
# H_G_Hmat(dadH; atol=1e-6)
# solve(dadH)
# @show rel_error(dadH), compression_ratio(dadH.H)
