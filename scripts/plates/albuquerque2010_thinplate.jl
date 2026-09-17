# Albuquerque & Aliabadi, CMAME 199 (2010) — Kirchhoff plate + anisotropic
# membrane + Donnell DIBEM (paper is thin-plate + RIM).
# 15 quadratic BE / side, formatdata cell-centroid internals (15×15).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate
const Point2D = SVector{2,Float64}

const N_EL = 15
const N_INT = 225
const NPG, NSUB = 8, 6
const RBF = PHS(2; poly_deg=1)
const RBFG = PHS(3; poly_deg=1)

function coupling_ops(shellm, A, geom)
    mesh = shellm.plate
    nt = BEM.Plate._n(mesh) + BEM.Plate._ni(mesh)
    pts = Point2D[BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
    κ1, κ2, κ12 = curvature_fields(geom, pts)
    ops = rbf_gradient_ops(pts; rbf=RBFG)
    Dx, Dy = ops.Fx, ops.Fy
    c_ux = A[1, 1] .* κ1 .+ A[1, 2] .* κ2 .+ A[1, 3] .* κ12
    c_vy = A[1, 2] .* κ1 .+ A[2, 2] .* κ2 .+ A[2, 3] .* κ12
    c_g = A[1, 3] .* κ1 .+ A[2, 3] .* κ2 .+ A[3, 3] .* κ12
    kmem = c_ux .* κ1 .+ c_vy .* κ2 .+ 2 .* c_g .* κ12
    Qop = zeros(nt, 2nt)
    Qop[:, 1:2:end] .= Diagonal(c_ux) * Dx .+ Diagonal(c_g) * Dy
    Qop[:, 2:2:end] .= Diagonal(c_vy) * Dy .+ Diagonal(c_g) * Dx
    Fop = zeros(2nt, nt)
    Fop[1:2:end, :] .= Diagonal(c_ux) * Dx .+ Diagonal(c_g) * Dy
    Fop[2:2:end, :] .= Diagonal(c_g) * Dx .+ Diagonal(c_vy) * Dy
    Hw_w = -shellm.Mm * Fop
    return kmem, Qop, Hw_w, nt
end

function solve_thin_laminated!(plate, shellm, A, geom; qpts=nothing)
    dad = plate
    has_cache(dad, :H) || assemble_plate!(dad; npg=NPG)
    has_cache(dad, :M) || dibem_plate!(dad; npg=NPG, rbf=RBF)
    isempty(shellm.Hm) && BEM.Plate.assemble_membrane!(shellm; npg=NPG, nsub=NSUB)
    isempty(shellm.Mm) && BEM.Plate.dibem_membrane!(shellm; npg=NPG, rbf=RBF)
    kmem, Qop, Hw_w, nt = coupling_ops(shellm, A, geom)
    nt == dad.nt || error("plate nt=$(dad.nt) ≠ membrane nt=$nt")
    ndp = size(dad.H, 1)
    ndm = 2nt
    nbp = size(dad.G, 2)
    nbm = 2 * dad.n
    qc = dad.properties.q_c
    Mw = plate_Mw(dad)
    size(Mw, 1) == ndp || error("plate_Mw rows $(size(Mw, 1)) ≠ ndp=$ndp")
    Hp = copy(dad.H)
    @inbounds for j in 1:nt
        Hp[:, 2j - 1] .+= kmem[j] .* Mw[:, j]
    end
    Ha = Mw * Qop
    Hw = zeros(ndm, ndp)
    @inbounds for k in 1:nt
        Hw[:, 2k - 1] .= Hw_w[:, k]
    end
    ndof = ndp + ndm
    nb = nbp + nbm
    H = zeros(ndof, ndof)
    H[1:ndp, 1:ndp] .= Hp
    H[1:ndp, ndp + 1:end] .= Ha
    H[ndp + 1:end, 1:ndp] .= Hw
    H[ndp + 1:end, ndp + 1:end] .= shellm.Hm
    G = zeros(ndof, nb)
    G[1:ndp, 1:nbp] .= dad.G
    G[ndp + 1:end, nbp + 1:end] .= shellm.Gm
    q = zeros(ndof)
    qnodal = qpts === nothing ? fill(qc, nt) : qpts
    q[1:ndp] .= Mw * qnodal
    Dsc = bending_stiffness(dad.properties)
    is_kin = falses(ndof)
    known = zeros(ndof)
    n2 = 2 * dad.n
    @inbounds for dof in 1:n2
        known[dof] = dad.BV[dof]
        is_kin[dof] = dad.BC[dof] == 0
    end
    corners = has_cache(dad, :plate_corners) ? dad.plate_corners : []
    nH0 = 2 * dad.nt
    for (c, corner) in enumerate(corners)
        dof = nH0 + c
        known[dof] = corner.bc_val
        is_kin[dof] = corner.bc_type == 0
    end
    for k in 1:nbm
        if shellm.BCm[k] == 0
            is_kin[ndp + k] = true
            known[ndp + k] = shellm.BVm[k]
        end
    end
    A = copy(H)
    b = copy(q)
    @inbounds for dof in 1:n2
        val = known[dof]
        if is_kin[dof]
            b .-= A[:, dof] .* val
            A[:, dof] .= -G[:, dof] .* Dsc
        else
            b .+= G[:, dof] .* val
        end
    end
    nGc = 2 * dad.n
    for (c, corner) in enumerate(corners)
        dofH = nH0 + c
        dofG = nGc + c
        val = known[dofH]
        if is_kin[dofH]
            b .-= A[:, dofH] .* val
            A[:, dofH] .= -G[:, dofG] .* Dsc
        else
            b .+= G[:, dofG] .* val
        end
    end
    for k in 1:nbm
        dof = ndp + k
        gcol = nbp + k
        if is_kin[dof]
            b .-= A[:, dof] .* known[dof]
            A[:, dof] .= -G[:, gcol]
        end
    end
    x = A \ b
    u = zeros(ndp)
    @inbounds for dof in 1:ndp
        u[dof] = is_kin[dof] ? known[dof] : x[dof]
    end
    set_cache!(dad; u=u, T=u)
    return plate_w_int(dad, 1)
end

function make_thin(; a, h, R, q, plies, bc="SSSS", mem_bc=:navier_ss,
        G13=nothing, G23=nothing, corner_bc='F')
    G12 = plies[1][4]
    g13 = G13 === nothing ? G12 : G13
    g23 = G23 === nothing ? 0.2 * plies[1][2] : G23
    dummy = laminate_fsdt_props(plies; Ks=5 / 6, G13=g13, G23=g23, q_c=0.0, ρ=1.0)
    A, _, D, _, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=g13, G23=g23)
    mesh_f = build_square_fsdt(; a=a, n_el=N_EL, bc=bc, props=dummy, n_internal=N_INT)
    geom = !isfinite(R) || R == 0 ? FlatShell() : SphericalShell(R)
    shellm = LaminatedShell(mesh_f, A, geom; mem_bc=mem_bc)
    BEM.Plate.assemble_membrane!(shellm; npg=NPG, nsub=NSUB)
    BEM.Plate.dibem_membrane!(shellm; npg=NPG, rbf=RBF)
    props_pl = aniso_thin_plate_props(D; q_c=q, h=h)
    internal = BEM.Plate._fsdt_cell_centroids(a, a, N_INT)
    plate = build_square_plate(; a=a, n_el=N_EL, bc=bc, props=props_pl,
        n_internal=N_INT, internal=internal, corner_bc=corner_bc, p=2)
    assemble_plate!(plate; npg=NPG)
    dibem_plate!(plate; npg=NPG, rbf=RBF)
    return plate, shellm, A, geom, D
end

function print_cl_plate(plate; nd=x -> x, label="w")
    n, ni = plate.n, plate.ni
    pts = plate.internalNodes
    a = maximum(p[1] for p in plate.Nodes) - minimum(p[1] for p in plate.Nodes)
    ymid = a / 2
    rows = Tuple{Float64,Float64}[]
    for k in 1:ni
        p = pts[k]
        abs(p[2] - ymid) < 1e-8 || continue
        w = plate.u[2 * (n + k) - 1]
        push!(rows, (p[1] / a, nd(w)))
    end
    sort!(rows; by=first)
    @printf("  %8s %12s\n", "x/a", label)
    for (x, w) in rows
        @printf("  %8.3f %12.5e\n", x, w)
    end
end

println("="^72)
println(" Albuquerque 2010  Kirchhoff DIBEM + membrane DIBEM")
println("  n_el/edge=$N_EL  n_internal=$N_INT")
println("="^72)

# 3.1
println("\n## 3.1  SS spherical [0/90]s  a/h=100  uniform q  (Kirchhoff)")
E1, E2, ν12 = 25.0, 1.0, 0.25
G12 = 0.5 * E2
a, h, q0 = 1.0, 0.01, 1.0
ndw(w) = 1e3 * w * E2 * h^3 / (q0 * a^4)
TABLE2 = Dict(Inf => 6.8331, 100.0 => 6.7772, 50.0 => 6.6148, 20.0 => 5.6618,
    10.0 => 3.7208, 5.0 => 1.5358, 2.0 => 0.2844, 1.0 => 0.0715)
TABLE3 = Dict(Inf => 4.3368, 10.0 => 2.4030, 5.0 => 1.0279, 1.0 => 0.0532)
plies01 = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
@printf("  %8s %10s %10s %8s\n", "R/a", "ŵ BEM", "ŵ Reddy", "e%")
for Ra in (Inf, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0)
    R = isfinite(Ra) ? Ra * a : Inf
    plate, shellm, A, geom, _ = make_thin(; a=a, h=h, R=R, q=q0, plies=plies01)
    wc = solve_thin_laminated!(plate, shellm, A, geom)
    ŵ = ndw(wc)
    gold = TABLE2[Ra]
    @printf("  %8s %10.4f %10.4f %7.2f\n",
        isinf(Ra) ? "Inf" : string(Int(Ra)), ŵ, gold, 100 * abs(ŵ - gold) / gold)
end

println("\n## 3.1  sinusoidal q  (Kirchhoff)")
@printf("  %8s %10s %10s %8s\n", "R/a", "ŵ BEM", "ŵ Reddy", "e%")
for Ra in (Inf, 10.0, 5.0, 1.0)
    R = isfinite(Ra) ? Ra * a : Inf
    plate, shellm, A, geom, _ = make_thin(; a=a, h=h, R=R, q=q0, plies=plies01)
    pts = Point2D[BEM.Plate._plate_nodes(shellm.plate); BEM.Plate._plate_internal(shellm.plate)]
    qpts = [q0 * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
    wc = solve_thin_laminated!(plate, shellm, A, geom; qpts=qpts)
    ŵ = ndw(wc)
    gold = TABLE3[Ra]
    @printf("  %8s %10.4f %10.4f %7.2f\n",
        isinf(Ra) ? "Inf" : string(Int(Ra)), ŵ, gold, 100 * abs(ŵ - gold) / gold)
end

# 3.2
println("\n## 3.2  Orthotropic spherical  Kirchhoff")
a2, h2, q2 = 0.254, 0.0127, 2.07e6
E2o = 6.895e9
E1o = 2 * E2o
ν12o = 0.3
G12o = E2o / (2 * (1 - ν12o))
plies2 = [(E1o, E2o, ν12o, G12o, 0.0, h2)]
w0C, w0S = 8.423e-3, 27.01e-3
for (bc, mem, w0, tag, cbc) in (("CCCC", :clamped, w0C, "clamped", 'C'),
        ("SSSS", :navier_ss, w0S, "SS", 'F'))
    println("  --- ", tag, "  w0=", w0, " m")
    @printf("  %8s %12s %10s\n", "R/a", "w_c (m)", "w/w0")
    for Ra in (10.0, 5.0)
        plate, shellm, A, geom, _ = make_thin(; a=a2, h=h2, R=Ra * a2, q=q2,
            plies=plies2, bc=bc, mem_bc=mem, G13=G12o, G23=G12o, corner_bc=cbc)
        wc = solve_thin_laminated!(plate, shellm, A, geom)
        @printf("  %8.0f %12.5e %10.4f\n", Ra, wc, wc / w0)
        print_cl_plate(plate; nd=w -> w / w0, label="w/w0")
    end
end

# 3.3
println("\n## 3.3  SS spherical [45/-45]s  R/a=5  Kirchhoff")
plies45 = [(E1, E2, ν12, G12, θ, h / 4) for θ in (45.0, -45.0, -45.0, 45.0)]
plate3, shellm3, A3, geom3, _ = make_thin(; a=a, h=h, R=5a, q=q0, plies=plies45)
wc3 = solve_thin_laminated!(plate3, shellm3, A3, geom3)
@printf("  w_c=%.6e  ŵ=%.4f\n", wc3, ndw(wc3))
print_cl_plate(plate3; nd=ndw, label="ŵ")
println("\ndone")
