# Sudden-load 2×static and T1 for the two TOE beams.
#   julia --project=. scripts/check_toe_beams_transient.jl
#   TOE_MESH=200  TOE_NTS=160
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))

const NPG = 12
const NTS = parse(Int, get(ENV, "TOE_NTS", "160"))
const MESH = parse(Int, get(ENV, "TOE_MESH", "200"))

function _boundary_probe(dad, p)
    n = dad.n
    ip = argmin(norm(dad.Nodes[i] - p) for i in 1:n)
    return ip, dad.Nodes[ip]
end

function _assemble(pname)
    dad, meta = elastodynamics_problem(pname; mesh_tag=MESH)
    H_G_full_direct(dad; npg=NPG, threaded=false)
    build_cell_mass(dad; npg=NPG)
    ip, p = _boundary_probe(dad, meta.probe)
    dadS = deepcopy(dad)
    BEM.solve(dadS)
    vS = dadS.u[2 * (ip - 1) + meta.comp]
    ua, va = meta.ana(p)
    vA = meta.comp == 1 ? ua : va
    return dad, meta, ip, p, vS, vA
end

function _step(dad, meta, ip, p, vS, vA, stepper::Symbol)
    s = vS == 0 ? (vA >= 0 ? 1.0 : -1.0) : sign(vS)
    dt = meta.tf / NTS
    dadT = deepcopy(dad)
    U = stepper === :houbolt ? solve_Houbolt(dadT, dt, meta.tf) :
        solve_Newmark(dadT, dt, meta.tf)
    t = collect(dadT.time)
    un = s .* collect(_elasto_hist(U, dadT, ip; comp=meta.comp))
    δB = abs(vS)
    ap = beam_amp_period(t, un; minfrac=0.25)
    n2 = max(length(un) ÷ 2, 1)
    mean_late = mean(un[n2:end])
    return (; name=meta.name, stepper, n=dad.n, nt=dad.nt, ip, p,
        δ_ana=abs(meta.δ), δ_bem=δB, v_ana=vA, v_bem=vS,
        amp_ana=2 * abs(meta.δ), amp_bem=2δB,
        u_peak=ap.u_peak, t_peak=ap.t_peak, T=meta.T, T_num=ap.T_num,
        npeak=ap.npeak, mean_late, tf=meta.tf, nts=NTS, notes=meta.notes)
end

function _print(r)
    @printf("%-18s %-8s  n=%3d  probe=(%.3f,%.3f)\n",
        r.name, r.stepper, r.n, r.p[1], r.p[2])
    @printf("  δ_ana=%8.4f  δ_BEM=%8.4f  2δ_BEM=%8.4f  u_peak=%8.4f  peak/2δ_BEM=%6.3f  peak/2δ_ana=%6.3f\n",
        r.δ_ana, r.δ_bem, r.amp_bem, r.u_peak,
        r.u_peak / (r.amp_bem + eps()), r.u_peak / (r.amp_ana + eps()))
    @printf("  T1=%8.4f  T_num=%8.4f  T_num/T1=%6.3f  t_peak=%8.4f  t_peak/(T1/2)=%6.3f  npeak=%d\n",
        r.T, r.T_num, r.T_num / (r.T + eps()),
        r.t_peak, r.t_peak / (r.T / 2 + eps()), r.npeak)
    @printf("  late mean=%8.4f  late/δ_BEM=%6.3f  %s\n",
        r.mean_late, r.mean_late / (r.δ_bem + eps()), r.notes)
    return nothing
end

function main()
    @printf("TOE sudden-load check  mesh=%d  nts=%d  tf=4 T1 (Timoshenko)\n", MESH, NTS)
    for pname in TOE_BEAM_PROBLEMS
        println()
        dad, meta, ip, p, vS, vA = _assemble(pname)
        @printf("assembled %s  n=%d nt=%d  T1=%.4f tf=%.4f  static BEM v=%.4f  δ_1D=%.4f\n",
            pname, dad.n, dad.nt, meta.T, meta.tf, vS, meta.δ)
        for st in (:houbolt, :newmark)
            _print(_step(dad, meta, ip, p, vS, vA, st))
        end
    end
    return nothing
end

main()
