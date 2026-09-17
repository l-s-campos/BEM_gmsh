# Cells vs remainder-DIBEM vs DRM for Laplace + elasticity body forces.
# julia --project=. scripts/dibem_cells_bodyforce.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))

rel(a, b) = norm(a - b) / (norm(b) + 1e-30)

function sample_nodes(pts, bfun)
    return Float64[float(bfun(p)) for p in pts]
end

function sample_vec(pts, bfun)
    nt = length(pts)
    b = zeros(2nt)
    @inbounds for i in 1:nt
        v = bfun(pts[i])
        b[2i-1] = v[1]
        b[2i] = v[2]
    end
    return b
end

function sample_cells_scalar(cells, bfun)
    return Float64[float(bfun(c.centroid)) for c in cells]
end

function sample_cells_vec(cells, bfun)
    nc = length(cells)
    b = zeros(2nc)
    @inbounds for k in 1:nc
        v = bfun(cells[k].centroid)
        b[2k-1] = v[1]
        b[2k] = v[2]
    end
    return b
end

function report_row(io, physics, force, method, err)
    @printf(io, "  %-9s  %-16s  %-14s  %.3e\n", physics, force, method, err)
end

println("="^72)
println(" Remainder DIBEM / DRM / constant cells  (body-force Mb vs cells)")
println("="^72)

# ----- Laplace -----
msh = Base.invokelatest(quadrado; ndiv=8, show=false, nome="bf_lap")
dad0 = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
H_G_full_direct(dad0; npg=10, threaded=false)
cm = build_cell_mass(deepcopy(dad0); npg=12)
pts = all_points(dad0)
forces_L = (
    ("1", p -> 1.0),
    ("x", p -> p[1]),
    ("x²+y²", p -> p[1]^2 + p[2]^2),
    ("sinπx sinπy", p -> sin(π * p[1]) * sin(π * p[2])),
)
Ms = (
    ("DIBEM -1", DIBEM(deepcopy(dad0); method=:dense, rbf=PHS(3; poly_deg=-1))),
    ("DIBEM p1", DIBEM(deepcopy(dad0); method=:dense, rbf=PHS(3; poly_deg=1))),
    ("DRM -1", build_drm_matrices(deepcopy(dad0), PHS(3; poly_deg=-1); npg=10).M),
    ("DRM p1", build_drm_matrices(deepcopy(dad0), PHS(3; poly_deg=1); npg=10).M),
    ("cells P", cm.M),
)

println("\nLaplace  n=$(dad0.n)  ni=$(dad0.ni)  cells=$(length(cm.cells))")
println("  physics    force             method          rel vs M_cell b_cell")
for (fname, bfun) in forces_L
    dref = cm.M_cell * sample_cells_scalar(cm.cells, bfun)
    for (mname, M) in Ms
        bn = sample_nodes(pts, bfun)
        report_row(stdout, "Laplace", fname, mname, rel(M * bn, dref))
    end
end

# ----- Elasticity -----
mshE = Base.invokelatest(quadrado_elasticity; ndiv=8, show=false, nome="bf_el")
de0 = format2d(mshE, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
    tipo=1, pontointerno=true)
H_G_full_direct(de0; npg=10, threaded=false)
cme = build_cell_mass(deepcopy(de0); npg=12)
pe = all_points(de0)
forces_E = (
    ("(1,0)", p -> SVector(1.0, 0.0)),
    ("(x,0)", p -> SVector(p[1], 0.0)),
    ("(x²,xy)", p -> SVector(p[1]^2, p[1] * p[2])),
    ("sin e_x", p -> SVector(sin(π * p[1]) * sin(π * p[2]), 0.0)),
    ("ω² r", p -> SVector(p[1], p[2])),
)
MsE = (
    ("DIBEM -1", DIBEM(deepcopy(de0); method=:dense, rbf=PHS(3; poly_deg=-1))),
    ("DIBEM p1", DIBEM(deepcopy(de0); method=:dense, rbf=PHS(3; poly_deg=1))),
    ("DRM -1", build_drm_matrices(deepcopy(de0); npg=10, kernel=:r, poly_deg=-1).M),
    ("DRM p1", build_drm_matrices(deepcopy(de0); npg=10, kernel=:r, poly_deg=1).M),
    ("cells P", cme.M),
)

println("\nElasticity  n=$(de0.n)  ni=$(de0.ni)  cells=$(length(cme.cells))")
println("  physics    force             method          rel vs M_cell b_cell")
for (fname, bfun) in forces_E
    dref = cme.M_cell * sample_cells_vec(cme.cells, bfun)
    for (mname, M) in MsE
        bn = sample_vec(pe, bfun)
        report_row(stdout, "Elast", fname, mname, rel(M * bn, dref))
    end
end
println("\nDone.")
