# DIBEM-SIMP on Coelho cantilever, then iso-cut of low density.
#
#   julia --project=. scripts/topology/dibem_simp_cantilever.jl

using DrWatson
@quickactivate :BEM
using BEM.Topology

d = coelho_cantilever(; ne=8, nint=10, degree=1)
opt = DibemSimpOptions(;
    volfrac=0.35, n_simp=30, rmin=0.22, ρ_cut=0.4, β_end=8.0,
    ngrid=61, min_hole_dist=0.002, min_hole_area=2e-4,
    cut=true, match_area=true, pacheco=true, verbose=true, npg=10,
    pacheco_opt=PachecoOptions(maxiter=30, verbose=true,
        nucleate_first=false, nucleate_every=typemax(Int),
        min_hole_dist=0.002, min_hole_area=2e-4),
)
println("DIBEM-SIMP cantilever  volfrac=$(opt.volfrac)  n_simp=$(opt.n_simp)")
d, dad, ρ, hist = solve_dibem_simp!(d, opt)
println("holes=$(n_holes(d))  area=$(round(design_area(d); digits=4))  J=$(round(elastic_compliance(dad); sigdigits=4))")
println("gray last=$(isempty(hist) ? NaN : round(hist[end].gray; digits=3))")
println("done")
