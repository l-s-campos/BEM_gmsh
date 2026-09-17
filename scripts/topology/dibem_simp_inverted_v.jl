# DIBEM-SIMP on Pacheco inverted-V, then iso-cut of low density.
#
#   julia --project=. scripts/topology/dibem_simp_inverted_v.jl

using DrWatson
@quickactivate :BEM
using BEM.Topology

d = pacheco_inverted_v(; ne=12, nint=12, degree=1)
opt = DibemSimpOptions(;
    volfrac=0.5, n_simp=40, rmin=0.1, ρ_cut=0.45, β_end=8.0,
    ngrid=71, min_hole_dist=0.008, min_hole_area=4e-4,
    cut=true, pacheco=true, verbose=true, npg=10,
    pacheco_opt=PachecoOptions(maxiter=40, verbose=true,
        nucleate_first=false, nucleate_every=typemax(Int)),
)
println("DIBEM-SIMP inverted-V  volfrac=$(opt.volfrac)  n_simp=$(opt.n_simp)")
d, dad, ρ, hist = solve_dibem_simp!(d, opt)
println("holes=$(n_holes(d))  area=$(round(design_area(d); digits=4))  C=$(round(thermal_conductance(dad); sigdigits=4))")
println("gray last=$(isempty(hist) ? NaN : round(hist[end].gray; digits=3))")
println("done")
