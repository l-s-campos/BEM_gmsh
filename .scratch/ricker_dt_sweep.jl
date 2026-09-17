# Sweep Δt on the anisotropic Ricker (isotropic FS + IBP).  Assemble once.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(joinpath(projectdir(), "scripts", "transient", "ricker_anisotropic_dibem.jl"))

ndiv = parse(Int, get(ENV, "RICKER_MESHES", "24"))
strategy = Symbol(get(ENV, "RICKER_ANISO", "ibp"))
nloc_raw = get(ENV, "RICKER_NLOCAL", "21")
nloc = isempty(strip(nloc_raw)) ? nothing : parse(Int, nloc_raw)
dts = [parse(Float64, s) for s in split(get(ENV, "RICKER_DT_SWEEP", "0.02,0.01,0.005,0.002,0.001,0.0005"), ',')]
t_end = parse(Float64, get(ENV, "RICKER_TF", "1.45"))

prob0 = AnisoRickerProblem(; ndiv, Δt=dts[1], t_end)
dad0 = build_dad(prob0)
K = aniso_K(prob0)
println("assemble ndiv=$ndiv  strategy=$strategy  nlocal=$(nloc === nothing ? "global" : nloc)")
t_asm = @elapsed begin
    H_G_full_direct(dad0; npg=10, threaded=false)
    DIBEM(dad0; rbf=PHS(3; poly_deg=1), npg=12)
    anisotropic_wave_shift!(dad0, K; strategy=strategy,
        rbf=PHS(3; poly_deg=2), nlocal=nloc)
end
Ms = Symmetric(Matrix(dad0.M))
Hs = Matrix(dad0.H)
evM = eigvals(Ms)
nneg = count(<( -1e-8), evM)
nzero = count(e -> abs(e) < 1e-10, evM)
@printf("  assembly %.2f s   nt=%d  nneg(M)=%d  n~0(M)=%d  λmin(M)=%.3e  λmax(M)=%.3e\n",
    t_asm, dad0.nt, nneg, nzero, minimum(evM), maximum(evM))

H0, G0, M0 = copy(dad0.H), copy(dad0.G), copy(dad0.M)
g = nodal_envelope(dad0, prob0)

println()
@printf("%-10s %8s %12s %10s %12s %10s %s\n", "Δt", "steps", "1/(βΔt²)", "max|p|", "max|p| last", "finite", "")
for Δt in dts
    dad = deepcopy(dad0)
    set_cache!(dad; H=copy(H0), G=copy(G0), M=copy(M0))
    has_cache(dad, :T) && (dad.cache.T = nothing)
    has_cache(dad, :q) && (dad.cache.q = nothing)
    has_cache(dad, :time) && (dad.cache.time = nothing)
    f_body = t -> (prob0.amp * ricker(t, prob0)) .* g
    t_sol = @elapsed solve_Newmark(dad, Δt, t_end; force=f_body)
    ok = all(isfinite, dad.T)
    mx = ok ? maximum(abs, dad.T) : Inf
    nT = size(dad.T, 2)
    nlast = max(1, nT ÷ 10)
    mxlast = ok ? maximum(abs, view(dad.T, :, (nT - nlast + 1):nT)) : Inf
    a0 = 1 / (0.25 * Δt^2)
    flag = !ok ? "BLOW" : (mx > 1.0 ? "GROW" : "ok")
    @printf(" %-10.4g %8d %12.3e %10.3e %12.3e %10s  %s  (%.1fs)\n",
        Δt, nT, a0, mx, mxlast, string(ok), flag, t_sol)
end
