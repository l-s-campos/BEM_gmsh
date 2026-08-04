"""BC type in Gmsh/`format2d` physical names that marks a crack face."""
const CRACK_BC = 5

# =============================================================================
# Data structures
# =============================================================================

"""
Physical collocation node.
`eq_type`: 1 = disp BIE + RBM, 2 = disp BIE (crack), 3 = traction BIE (crack).
`twin`: index of the opposite-face node at the same geometric location (0 = none).
"""
mutable struct DualNode
    id::Int
    pos::Point2D
    eq_type::Int
    normal::Point2D
    twin::Int
end

"""
Quadratic **discontinuous** element.
Geometry uses continuous Lagrange shape functions on `geo`.
Collocation / DOF nodes `fis` sit at ``ξ = ±2/3, 0`` (never shared across elements).
"""
mutable struct DualElement
    id::Int
    geo::NTuple{3,Int}          # geometric node indices into mesh.geo_nodes
    fis::NTuple{3,Int}          # physical node indices into mesh.nodes
    eq_type::Int
    # BC per local node (1..3) and direction (1=x,2=y):
    # bc_type[loc, dir] = 0 Dirichlet (u known), 1 Neumann (t known)
    bc_type::Matrix{Int}        # 3×2
    bc_val::Matrix{Float64}     # 3×2
end

mutable struct DualMesh
    geo_nodes::Vector{Point2D}
    nodes::Vector{DualNode}
    elements::Vector{DualElement}
    E::Float64
    ν::Float64
    plane_strain::Bool
    # assembled
    H::Matrix{Float64}
    G::Matrix{Float64}          # 2n × 6·nel  (element-wise tractions)
    u::Vector{Float64}          # 2n displacements
    t_el::Vector{Float64}       # 6·nel tractions
    # crack bookkeeping
    crack_face_a::Vector{Int}   # element ids face A (eq=2)
    crack_face_b::Vector{Int}   # element ids face B (eq=3)
    tip_nodes::Vector{Int}      # physical nodes nearest tips (on face A)
end

function DualMesh(geo, nodes, elems; E=1.0, ν=0.3, plane_strain=true)
    n = length(nodes)
    nel = length(elems)
    return DualMesh(geo, nodes, elems, float(E), float(ν), plane_strain,
        zeros(2n, 2n), zeros(2n, 6nel), zeros(2n), zeros(6nel),
        Int[], Int[], Int[])
end

# material helpers
function _νeff(m::DualMesh)
    m.plane_strain && return m.ν
    return m.ν / (1 + m.ν)          # plane-stress → plane-strain map used in Kelvin
end
_μ(m::DualMesh) = m.E / (2(1 + m.ν))
function _κ(m::DualMesh)
    ν = m.ν
    m.plane_strain && return 3 - 4ν
    return (3 - ν) / (1 + ν)
end

# =============================================================================
# Shape functions
# =============================================================================

"""Continuous quadratic shape (geometry + continuous collocation)."""
function N_cont(ξ)
    return SVector(0.5ξ * (ξ - 1), 1 - ξ^2, 0.5ξ * (ξ + 1))
end
function dN_cont(ξ)
    return SVector(ξ - 0.5, -2ξ, ξ + 0.5)
end

"""Discontinuous quadratic collocation shape (nodes at ξ = ±2/3, 0)."""
function N_disc(ξ)
    return SVector(ξ * (9 / 8 * ξ - 3 / 4), 1 - 9 / 4 * ξ^2, ξ * (9 / 8 * ξ + 3 / 4))
end

const XI_DISC = (-2 / 3, 0.0, 2 / 3)

function elem_geometry(mesh::DualMesh, el::DualElement, ξ)
    g = el.geo
    X = (mesh.geo_nodes[g[1]], mesh.geo_nodes[g[2]], mesh.geo_nodes[g[3]])
    N = N_cont(ξ)
    dN = dN_cont(ξ)
    x = N[1] * X[1] + N[2] * X[2] + N[3] * X[3]
    dx = dN[1] * X[1] + dN[2] * X[2] + dN[3] * X[3]
    J = norm(dx)
    # outward-ish normal: rotate tangent 90° CW so domain is to the left when
    # walking p1→p3 (standard CCW outer boundary). For crack faces the walk
    # direction sets the face normal.
    n = J > 0 ? Point2D(dx[2] / J, -dx[1] / J) : Point2D(0.0, 0.0)
    return x, J, n
end

shape_fis(::DualElement, ξ) = N_disc(ξ)

# =============================================================================
# Fundamental solutions (Kelvin + D,S) — match Dual MATLAB / package
# =============================================================================

function kelvin_UT(rvec::Point2D, n::Point2D, ν, μ)
    R = max(norm(rvec), 1e-30)
    dr = rvec / R
    drdn = dot(dr, n)
    prod1 = 4π * (1 - ν)
    prod2 = (3 - 4ν) * log(1 / R)
    base = 2 * prod1 * μ
    u11 = (prod2 + dr[1]^2) / base
    u22 = (prod2 + dr[2]^2) / base
    u12 = (dr[1] * dr[2]) / base
    fat = 1 / (prod1 * R)
    t11 = -drdn * ((1 - 2ν) + 2 * dr[1]^2) * fat
    t22 = -drdn * ((1 - 2ν) + 2 * dr[2]^2) * fat
    t12 = -(drdn * 2 * dr[1] * dr[2] - (1 - 2ν) * (dr[1] * n[2] - dr[2] * n[1])) * fat
    t21 = -(drdn * 2 * dr[1] * dr[2] - (1 - 2ν) * (dr[2] * n[1] - dr[1] * n[2])) * fat
    U = @SMatrix [u11 u12; u12 u22]
    T = @SMatrix [t11 t12; t21 t22]
    return U, T
end

"""D[k,i,j], S[k,i,j] as in `calcula_DeS.m` (field normal `n_field`)."""
function kelvin_DS(rvec::Point2D, n_field::Point2D, ν, μ)
    R = max(norm(rvec), 1e-30)
    dr = rvec / R
    drdn = dot(dr, n_field)
    fat1 = 4π * (1 - ν)
    fat2 = 1 - 2ν
    D = zeros(MArray{Tuple{2,2,2},Float64})
    S = zeros(MArray{Tuple{2,2,2},Float64})
    @inbounds for k in 1:2, i in 1:2, j in 1:2
        δki = i == k ? 1.0 : 0.0
        δkj = j == k ? 1.0 : 0.0
        δij = i == j ? 1.0 : 0.0
        d1 = fat2 * (δki * dr[j] + δkj * dr[i] - δij * dr[k])
        d2 = 2 * dr[i] * dr[j] * dr[k]
        D[k, i, j] = (d1 + d2) / (fat1 * R)
        t1 = 2 * drdn * (fat2 * δij * dr[k] + ν * (δki * dr[j] + δkj * dr[i]) -
                         4 * dr[i] * dr[j] * dr[k])
        t2 = 2ν * (n_field[i] * dr[j] * dr[k] + n_field[j] * dr[i] * dr[k])
        t3 = fat2 * (2 * n_field[k] * dr[i] * dr[j] + n_field[j] * δki + n_field[i] * δkj) -
             (1 - 4ν) * n_field[k] * δij
        S[k, i, j] = (t1 + t2 + t3) * 2μ / (fat1 * R^2)
    end
    return D, S
end

# pull D/S into 2×2 blocks as in gh_nsing_du
function _DS_blocks(D, S)
    D1 = @SMatrix [D[1,1,1] D[2,1,1]; D[1,1,2] D[2,1,2]]
    D2 = @SMatrix [D[1,2,1] D[2,2,1]; D[1,2,2] D[2,2,2]]
    S1 = @SMatrix [S[1,1,1] S[2,1,1]; S[1,1,2] S[2,1,2]]
    S2 = @SMatrix [S[1,2,1] S[2,2,1]; S[1,2,2] S[2,2,2]]
    return D1, D2, S1, S2
end

# =============================================================================
# Singular analytical kernels on crack (h_sing_du / g_sing_analitico)
# =============================================================================

function h_sing_crack(x1, y1, x3, y3, no_pf, tipo_eq, n_pf, ν, μ)
    h = zeros(2, 6)
    L = hypot(x3 - x1, y3 - y1)
    qsi = XI_DISC[no_pf]
    if tipo_eq == 2
        coef = @SMatrix [0.0 -(1-2ν)/(4π*(1-ν)); (1-2ν)/(4π*(1-ν)) 0.0]
        pf1 = 3/4 * (qsi * (3qsi - 2) / 2 * log(abs((1 - qsi) / (1 + qsi))) + 3qsi - 2)
        pf2 = 1/2 * (qsi * (3qsi - 2) * (3qsi + 2) / 2 * log(abs((1 + qsi) / (1 - qsi))) - 9qsi)
        pf3 = 3/4 * (qsi * (3qsi + 2) / 2 * log(abs((1 - qsi) / (1 + qsi))) + 3qsi + 2)
        pf = @SMatrix [pf1 0 pf2 0 pf3 0; 0 pf1 0 pf2 0 pf3]
        h = Matrix(coef * pf)
        diag = 2 * no_pf - 1
        h[1, diag] += 0.5
        h[2, diag+1] += 0.5
    else
        # traction BIE finite-part (tipo_eq == 3)
        coef = μ / (2π * (1 - ν)) * 2 / L
        n1 = (y3 - y1) / L
        n2 = -(x3 - x1) / L
        S1 = @SMatrix [n1*(2n2^2+1)  -n2*(-2n2^2+1); -n2*(2n1^2-1)  n1*(-2n2^2+1)]
        S2 = @SMatrix [-n2*(2n1^2-1)  n1*(-2n2^2+1);  n1*(2n1^2-1) -n2*(-2n1^2-1)]
        pf1 = 3/4 * ((3qsi - 1) * log(abs((1 - qsi) / (1 + qsi))) + (6qsi^2 - 2qsi - 3) / (qsi^2 - 1))
        pf2 = 1/2 * (9qsi * log(abs((1 + qsi) / (1 - qsi))) - (18qsi^2 - 13) / (qsi^2 - 1))
        pf3 = 3/4 * ((3qsi + 1) * log(abs((1 - qsi) / (1 + qsi))) + (6qsi^2 + 2qsi - 3) / (qsi^2 - 1))
        pf = @SMatrix [pf1 0 pf2 0 pf3 0; 0 pf1 0 pf2 0 pf3]
        h = Matrix(coef * (n_pf[1] * S1 * pf + n_pf[2] * S2 * pf))
    end
    return h
end

function g_sing_crack(x1, y1, x3, y3, no_pf, tipo_eq, n_pf, ν, μ)
    g = zeros(2, 6)
    L = hypot(x3 - x1, y3 - y1)
    rx, ry = (x3 - x1) / L, (y3 - y1) / L
    qsi = XI_DISC[no_pf]
    if tipo_eq == 2
        coef2 = (3 - 4ν) / (8π * μ * (1 - ν))
        coef1 = 1 / (8π * μ * (1 - ν))
        const_ = log(2 / L)
        int1 = ((-2 + 6qsi - 6qsi^2 - 3 * (-1 + qsi) * qsi^2 * log(abs(1 - qsi)) +
                 3 * (2 - qsi^2 + qsi^3) * log(abs(1 + qsi))) / 8)
        int2 = ((6 * (-1 + qsi^2) + (1 + 4qsi - 3qsi^3) * log(abs(-1 - qsi)) +
                 (1 - 4qsi + 3qsi^3) * log(abs(1 - qsi))) / 4)
        int3 = ((-2 - 6qsi - 6qsi^2 + 3qsi^2 * (1 + qsi) * log(abs(-1 - qsi)) -
                 3 * (-2 + qsi^2 + qsi^3) * log(abs(1 - qsi))) / 8)
        int_us = -coef2 * @SMatrix [int1 0 int2 0 int3 0; 0 int1 0 int2 0 int3]
        uns = @SMatrix [coef1*rx^2+coef2*const_  coef1*rx*ry; coef1*rx*ry  coef1*ry^2+coef2*const_]
        int_fi = @SMatrix [0.75 0 0.5 0 0.75 0; 0 0.75 0 0.5 0 0.75]
        g = Matrix((int_us + uns * int_fi) * L / 2)
    else
        coef1 = (1 - 2ν) / (4π * (1 - ν))
        coef2 = 1 / (2π * (1 - ν))
        D = zeros(2, 2, 2)
        D[1,1,1] = coef1*rx + coef2*rx^3
        D[2,1,1] = -coef1*ry + coef2*rx^2*ry
        D[1,2,2] = -coef1*rx + coef2*rx*ry^2
        D[2,2,2] = coef1*ry + coef2*ry^3
        D[1,2,1] = coef1*ry + coef2*ry*rx^2
        D[2,1,2] = coef1*rx + coef2*rx*ry^2
        D[1,1,2] = D[1,2,1]
        D[2,2,1] = D[2,1,2]
        D1 = @SMatrix [D[1,1,1] D[2,1,1]; D[1,1,2] D[2,1,2]]
        D2 = @SMatrix [D[1,1,2] D[2,1,2]; D[1,2,2] D[2,2,2]]
        pf1 = 3/4 * (qsi * (3qsi - 2) / 2 * log(abs((1 - qsi) / (1 + qsi))) + 3qsi - 2)
        pf2 = 1/2 * (qsi * (3qsi - 2) * (3qsi + 2) / 2 * log(abs((1 + qsi) / (1 - qsi))) - 9qsi)
        pf3 = 3/4 * (qsi * (3qsi + 2) / 2 * log(abs((1 - qsi) / (1 + qsi))) + 3qsi + 2)
        pf = @SMatrix [pf1 0 pf2 0 pf3 0; 0 pf1 0 pf2 0 pf3]
        g = Matrix(n_pf[1] * D1 * pf + n_pf[2] * D2 * pf)
        diag = 2 * no_pf - 1
        g[1, diag] -= 0.5
        g[2, diag+1] -= 0.5
    end
    return g
end

# =============================================================================
# Regular integration of one collocation × one element
# =============================================================================

function integrate_nsing!(h_el, g_el, mesh, el, xf, n_pf, tipo_eq, qsi_g, w_g)
    ν = _νeff(mesh)
    μ = _μ(mesh)
    fill!(h_el, 0)
    fill!(g_el, 0)
    @inbounds for (ig, ξ) in enumerate(qsi_g)
        x, J, n = elem_geometry(mesh, el, ξ)
        Nf = N_disc(ξ)
        ff = @SMatrix [
            Nf[1] 0 Nf[2] 0 Nf[3] 0
            0 Nf[1] 0 Nf[2] 0 Nf[3]
        ]
        wJ = J * w_g[ig]
        rvec = x - xf
        R = norm(rvec)
        R < 1e-14 && continue
        if tipo_eq == 1 || tipo_eq == 2
            U, T = kelvin_UT(rvec, n, ν, μ)
            g_el .+= U * ff * wJ
            h_el .+= T * ff * wJ
        else
            D, S = kelvin_DS(rvec, n, ν, μ)
            D1, D2, S1, S2 = _DS_blocks(D, S)
            g_el .+= (n_pf[1] * D1 + n_pf[2] * D2) * ff * wJ
            h_el .+= (n_pf[1] * S1 + n_pf[2] * S2) * ff * wJ
        end
    end
    return h_el, g_el
end

# =============================================================================
# Full assembly
# =============================================================================

"""
    assemble_dual!(mesh; npg=12)

Build `mesh.H` (2n×2n) and `mesh.G` (2n×6nel) for the dual BEM.
"""
function assemble_dual!(mesh::DualMesh; npg=12)
    n = length(mesh.nodes)
    nel = length(mesh.elements)
    H = zeros(2n, 2n)
    G = zeros(2n, 6nel)
    qsi_g, w_g = gausslegendre(npg)
    # denser for near-singular regularisation
    qsi_f, w_f = gausslegendre(max(npg, 20))

    h_el = zeros(2, 6)
    g_el = zeros(2, 6)

    for (ie, el) in enumerate(mesh.elements)
        g1, g2, g3 = el.geo
        xg1 = mesh.geo_nodes[g1]; xg3 = mesh.geo_nodes[g3]
        fis = el.fis
        for j in 1:n
            node = mesh.nodes[j]
            xf = node.pos
            tipo_eq = node.eq_type
            n_pf = node.normal
            twin = node.twin

            on_elem = (j == fis[1] || j == fis[2] || j == fis[3])
            twin_on = twin != 0 && (twin == fis[1] || twin == fis[2] || twin == fis[3])

            if !on_elem && !twin_on
                integrate_nsing!(h_el, g_el, mesh, el, xf, n_pf, tipo_eq, qsi_g, w_g)
            elseif tipo_eq == 1
                # outer singular: dense CPV for H/G; free term of H via RBM later
                integrate_nsing!(h_el, g_el, mesh, el, xf, n_pf, 1, qsi_f, w_f)
            else
                # crack / twin singular — analytical finite part (disc. elements)
                no_pf = j == fis[1] || twin == fis[1] ? 1 :
                        j == fis[2] || twin == fis[2] ? 2 : 3
                h_el .= h_sing_crack(xg1[1], xg1[2], xg3[1], xg3[2], no_pf, tipo_eq, n_pf,
                    _νeff(mesh), _μ(mesh))
                g_el .= g_sing_crack(xg1[1], xg1[2], xg3[1], xg3[2], no_pf, tipo_eq, n_pf,
                    _νeff(mesh), _μ(mesh))
            end

            rows = 2j-1:2j
            # scatter H into physical DOFs
            for a in 1:3
                ja = fis[a]
                H[rows, 2ja-1:2ja] .+= h_el[:, 2a-1:2a]
            end
            G[rows, 6ie-5:6ie] .+= g_el
        end
    end

    # Rigid-body motion free term for outer (eq_type==1) nodes
    for m in 1:n
        mesh.nodes[m].eq_type != 1 && continue
        rows = 2m-1:2m
        H[rows, rows] .= 0
        for nn in 1:n
            nn == m && continue
            H[rows, rows] .-= H[rows, 2nn-1:2nn]
        end
    end

    mesh.H = H
    mesh.G = G
    return mesh
end

# =============================================================================
# BCs and solve
# =============================================================================

"""
Apply mixed BCs by column exchange between H (displacements) and G (element tractions).
Returns `(A, b)` for `A x = b` with `x` packing unknown u-dofs and unknown t-dofs.
Also returns bookkeeping to reconstruct full `u` and `t_el`.
"""
function apply_bc_dual(mesh::DualMesh)
    n = length(mesh.nodes)
    nel = length(mesh.elements)
    H = copy(mesh.H)
    G = copy(mesh.G)
    μ = _μ(mesh)   # MATLAB multiplies swapped G columns by GEL (=μ)

    # Build map: each physical DOF (node,dir) → (element, local_node) for traction
    # Prefer first element that owns the node.
    owner = fill((0, 0), n)   # (el, loc)
    for (ie, el) in enumerate(mesh.elements)
        for loc in 1:3
            j = el.fis[loc]
            owner[j] == (0, 0) && (owner[j] = (ie, loc))
        end
    end

    # known traction vector (6*nel), default 0; known displacement defaults 0
    t_known = zeros(6nel)
    u_known = zeros(2n)
    is_dir = falses(2n)   # true → u prescribed (Dirichlet)

    for (ie, el) in enumerate(mesh.elements)
        for loc in 1:3, dir in 1:2
            j = el.fis[loc]
            dof = 2(j - 1) + dir
            gcol = 6(ie - 1) + 2(loc - 1) + dir
            if el.bc_type[loc, dir] == 0
                # Dirichlet: u known, t unknown — swap
                is_dir[dof] = true
                u_known[dof] = el.bc_val[loc, dir]
                # only swap once per physical DOF (first owning element)
                if owner[j] == (ie, loc)
                    colH = H[:, dof]
                    colG = G[:, gcol]
                    H[:, dof] = -colG .* μ
                    G[:, gcol] = -colH
                end
            else
                # Neumann: t known
                t_known[gcol] = el.bc_val[loc, dir]
            end
        end
    end

    b = G * t_known
    # move known Dirichlet contributions: for Dirichlet DOFs, the unknown is t,
    # and H column already holds the (scaled) G column; known u multiplies original H?
    # Standard: after swap, A = H, x has u at Neu dofs and t at Dir dofs,
    # b = G * t_prescribed. For prescribed u, they entered via the swap convention
    # of the dual code: b = G*cdc_val with cdc holding known values of the *original*
    # G-side quantity (tractions). Dirichlet values are applied by setting the
    # corresponding known on the swapped side — see aplica_cdc_du (only G*cdc).
    #
    # When u is prescribed, the RHS should include H_orig * u_known. After swap
    # H_new[:,dir] = -G_old, so we need to add H_old * u = contribution.
    # Reconstruct: before swap H_old[:,dof] = -G_new[:,gcol] (same gcol used).
    for dof in 1:2n
        if is_dir[dof] && abs(u_known[dof]) > 0
            # find gcol of owner
            j = (dof + 1) ÷ 2
            dir = dof - 2(j - 1)
            ie, loc = owner[j]
            ie == 0 && continue
            gcol = 6(ie - 1) + 2(loc - 1) + dir
            # H_old[:,dof] is currently stored as -G[:,gcol] after swap
            H_old_col = -G[:, gcol]
            b .-= H_old_col .* u_known[dof]
        end
    end

    A = H
    return A, b, is_dir, u_known, t_known, owner
end

"""Solve dual system and fill `mesh.u`, `mesh.t_el`."""
function solve_dual!(mesh::DualMesh)
    A, b, is_dir, u_known, t_known, owner = apply_bc_dual(mesh)
    x = A \ b
    n = length(mesh.nodes)
    nel = length(mesh.elements)
    u = zeros(2n)
    t_el = copy(t_known)
    for dof in 1:2n
        if is_dir[dof]
            u[dof] = u_known[dof]
            j = (dof + 1) ÷ 2
            dir = dof - 2(j - 1)
            ie, loc = owner[j]
            ie == 0 && continue
            gcol = 6(ie - 1) + 2(loc - 1) + dir
            t_el[gcol] = x[dof]
        else
            u[dof] = x[dof]
        end
    end
    mesh.u = u
    mesh.t_el = t_el
    return u, t_el
end

# =============================================================================
# Mesh builders
# =============================================================================

# =============================================================================
# Gmsh / format2d → DualMesh
# =============================================================================

"""
    dual_mesh_from_bemdata(dad::BEMdata{<:Elasticity}; plane_strain=true) -> DualMesh

Convert a mesh from [`format2d`](@ref) into a dual-BEM mesh.

Crack faces must be tagged with BC type [`CRACK_BC`](@ref) (`5`) in the Gmsh
physical name. The **value** slot selects the dual equation:

| Physical name   | Role |
|-----------------|------|
| `"5;2;5;2"`     | crack face, displacement BIE (`eq=2`) |
| `"5;3;5;3"`     | crack face, traction BIE (`eq=3`) |
| other           | outer boundary (`eq=1`) |

Twins are paired by nearest opposite-face node at the same geometric station.
"""
function dual_mesh_from_bemdata(dad; plane_strain=true)
    props = dad.properties
    E = props.E
    ν = props.nu
    plane_strain = hasproperty(props, :plane_strain) ? props.plane_strain : plane_strain

    geo = Point2D[Point2D(p) for p in dad.Nodes]
    nodes = DualNode[]
    elems = DualElement[]

    # one DualNode per collocation node
    for i in 1:dad.n
        bc_x = dad.BC[2i - 1]
        bv_x = dad.BV[2i - 1]
        if bc_x == CRACK_BC
            eq = Int(round(bv_x))
            eq in (2, 3) || (eq = 2)
        else
            eq = 1
        end
        push!(nodes, DualNode(i, Point2D(dad.Nodes[i]), eq, Point2D(dad.Normal[i]), 0))
    end

    for (ie, el) in enumerate(dad.elements)
        idx = el.index
        nn = length(idx)
        # map to 3 local nodes (pad linear elements)
        if nn == 1
            fis = (idx[1], idx[1], idx[1])
        elseif nn == 2
            # synthetic mid geo point
            mid = Point2D(0.5 * (geo[idx[1]] + geo[idx[2]]))
            push!(geo, mid)
            imid = length(geo)
            # mid collocation node (copy attributes from first)
            n0 = nodes[idx[1]]
            push!(nodes, DualNode(length(nodes) + 1, mid, n0.eq_type, n0.normal, 0))
            imid_n = length(nodes)
            fis = (idx[1], imid_n, idx[2])
            # geo indices: use collocation for ends + new mid
            g = (idx[1], imid, idx[2])
        else
            # take first, middle, last
            i1, i2, i3 = idx[1], idx[(nn + 1) ÷ 2], idx[end]
            fis = (i1, i2, i3)
            g = fis
        end
        eq = nodes[fis[1]].eq_type
        bc_t = fill(1, 3, 2)
        bc_v = zeros(3, 2)
        for loc in 1:3
            j = fis[loc]
            j > dad.n && continue   # synthetic mid: Neumann 0
            for dir in 1:2
                dof = 2(j - 1) + dir
                t = dad.BC[dof]
                v = dad.BV[dof]
                if t == CRACK_BC
                    # crack faces are traction-free unknowns (u free, t=0)
                    bc_t[loc, dir] = 1
                    bc_v[loc, dir] = 0.0
                elseif t == 0
                    bc_t[loc, dir] = 0
                    bc_v[loc, dir] = v
                else
                    bc_t[loc, dir] = 1
                    bc_v[loc, dir] = v
                end
            end
        end
        push!(elems, DualElement(ie, g, fis, eq, bc_t, bc_v))
    end

    mesh = DualMesh(geo, nodes, elems; E=E, ν=ν, plane_strain=plane_strain)

    # classify crack faces + pair twins
    faceA = Int[]
    faceB = Int[]
    for el in elems
        if el.eq_type == 2
            push!(mesh.crack_face_a, el.id)
            append!(faceA, el.fis)
        elseif el.eq_type == 3
            push!(mesh.crack_face_b, el.id)
            append!(faceB, el.fis)
        end
    end
    unique!(faceA)
    unique!(faceB)
    _pair_crack_twins!(nodes, faceA, faceB)

    # tips on face A
    if !isempty(faceA)
        xs = [nodes[i].pos[1] for i in faceA]
        push!(mesh.tip_nodes, faceA[argmin(xs)])
        push!(mesh.tip_nodes, faceA[argmax(xs)])
    end
    return mesh
end

function _pair_crack_twins!(nodes, faceA, faceB)
    (isempty(faceA) || isempty(faceB)) && return nothing
    used = falses(length(faceB))
    for ia in faceA
        best = 0
        bd = Inf
        pa = nodes[ia].pos
        for (k, ib) in enumerate(faceB)
            used[k] && continue
            d = norm(nodes[ib].pos - pa)
            if d < bd
                bd = d
                best = k
            end
        end
        best == 0 && continue
        ib = faceB[best]
        used[best] = true
        nodes[ia].twin = ib
        nodes[ib].twin = ia
        # opposite normals
        nodes[ib].normal = -nodes[ia].normal
    end
    return nothing
end

"""Pin rigid-body modes on the outer boundary of a dual mesh."""
function _pin_plate_rbm!(mesh::DualMesh; W=5.0, H=10.0)
    nodes = mesh.nodes
    function nearest(pred)
        best = 0
        bd = Inf
        for (i, nd) in enumerate(nodes)
            nd.eq_type == 1 || continue
            pred(nd.pos) || continue
            d = abs(nd.pos[1]) + abs(nd.pos[2])
            if d < bd
                bd = d
                best = i
            end
        end
        return best
    end
    i_left = nearest(p -> abs(p[1] + W) < 1e-6 * max(W, 1) && abs(p[2]) < 0.6H)
    i_bot = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.6W)
    i_bot2 = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.25W)
    function set_dir!(inode, dir, val=0.0)
        inode == 0 && return
        for el in mesh.elements
            for loc in 1:3
                if el.fis[loc] == inode
                    el.bc_type[loc, dir] = 0
                    el.bc_val[loc, dir] = val
                    return
                end
            end
        end
    end
    set_dir!(i_left, 1, 0.0)
    set_dir!(i_bot, 2, 0.0)
    if i_bot2 != 0 && i_bot2 != i_bot
        set_dir!(i_bot2, 1, 0.0)
    elseif i_left != 0
        set_dir!(i_left, 2, 0.0)
    end
    return mesh
end

"""
    build_center_crack_mesh(; ...) -> DualMesh

Build a centre-cracked plate via **Gmsh + `format2d`** (crack BC type 5).
"""
function build_center_crack_mesh(; W=5.0, H=10.0, a=1.0,
    n_bottom=8, n_right=16, n_top=8, n_left=16, n_crack=16,
    E=3000.0, ν=0.2, σ=1.0, plane_strain=true, ordem=2)

    # defined in data/elastico/iso/center_crack.jl — call mesh + format when available
    if isdefined(Main, :mesh_center_crack)
        msh = Main.mesh_center_crack(; W=W, H=H, a=a,
            ndiv_b=max(n_bottom, n_top), ndiv_h=max(n_right, n_left),
            ndiv_crack=n_crack, σ=σ, ordem=ordem, show=false)
    else
        # inline Gmsh (same as data/elastico/iso/center_crack.jl)
        msh = _gmsh_center_crack(; W=W, H=H, a=a,
            ndiv_b=max(n_bottom, n_top), ndiv_h=max(n_right, n_left),
            ndiv_crack=n_crack, σ=σ, ordem=ordem)
    end
    B = parentmodule(@__MODULE__)
    props = B.Elasticity(E, ν, 1.0; plane_strain=plane_strain)
    dad = B.format2d(msh, props; tipo=ordem, pontointerno=false)
    mesh = dual_mesh_from_bemdata(dad; plane_strain=plane_strain)
    return _pin_plate_rbm!(mesh; W=W, H=H)
end

function _gmsh_center_crack(; W=5.0, H=10.0, a=1.0,
    ndiv_b=10, ndiv_h=16, ndiv_crack=16, σ=1.0, ordem=2, nome="center_crack")
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)

    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)
    ptL = gmsh.model.geo.addPoint(-a, 0, 0, lc / 2)
    ptR = gmsh.model.geo.addPoint(a, 0, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    ll = gmsh.model.geo.addLine(p4, p1)
    c_low = gmsh.model.geo.addLine(ptL, ptR)
    c_up = gmsh.model.geo.addLine(ptR, ptL)

    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [c_low, c_up], 2, s)

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lr, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(ll, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(c_low, ndiv_crack)
    gmsh.model.mesh.setTransfiniteCurve(c_up, ndiv_crack)

    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")
    gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [c_low], -1, "5;2;5;2")
    gmsh.model.addPhysicalGroup(1, [c_up], -1, "5;3;5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

# =============================================================================
# Post-process — COD SIFs
# =============================================================================

"""Crack opening (and sliding) at a face-A node via its twin on face B."""
function crack_opening(mesh::DualMesh, inode::Int)
    tw = mesh.nodes[inode].twin
    tw == 0 && error("node $inode has no twin")
    uA = SVector(mesh.u[2inode-1], mesh.u[2inode])
    uB = SVector(mesh.u[2tw-1], mesh.u[2tw])
    return uA - uB
end

"""
    sif_cod_dual(mesh, tip_node; sample=2) -> (KI, KII)

COD correlation at the `sample`-th node behind the tip on face A.
"""
function sif_cod_dual(mesh::DualMesh, tip_node::Int; sample::Int=2)
    # gather face-A nodes ordered by distance to tip
    faceA = Int[]
    for elid in mesh.crack_face_a
        el = mesh.elements[elid]
        append!(faceA, el.fis)
    end
    unique!(faceA)
    tip = mesh.nodes[tip_node].pos
    sort!(faceA; by = i -> norm(mesh.nodes[i].pos - tip))
    # sample node behind tip
    isamp = faceA[min(1 + sample, length(faceA))]
    r = norm(mesh.nodes[isamp].pos - tip)
    r = max(r, 1e-14)
    Δu = crack_opening(mesh, isamp)

    # local frame from tip into crack
    # approximate tangent from tip to next node
    inext = faceA[min(2, length(faceA))]
    t̂ = mesh.nodes[inext].pos - tip
    t̂ = t̂ / (norm(t̂) + eps())
    n̂ = Point2D(-t̂[2], t̂[1])
    # choose n pointing from face B to face A roughly + opening
    Δun = abs(dot(Δu, n̂))
    Δut = dot(Δu, t̂)

    μ = _μ(mesh)
    κ = _κ(mesh)
    c = μ / (κ + 1) * sqrt(2π / r)
    return c * Δun, c * Δut
end

"""Build a `CrackPath` from dual mesh tips (for propagation geometry)."""
function dual_to_crack_path(mesh::DualMesh)
    tips_pos = [mesh.nodes[i].pos for i in mesh.tip_nodes]
    return tips_pos
end

