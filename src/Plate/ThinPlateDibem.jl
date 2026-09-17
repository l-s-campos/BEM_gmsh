# Kirchhoff DIBEM (`M`), port of BEM atual `Monta_M_RIMd` (`src/rim.jl`).
# Packed in this repo's layout: 2 DOF/node (w, ∂w/∂n), dummy internal slope,
# then corner w. Body force lives on w-columns only.

export dibem_plate!, plate_Mw

"""Unit distributed load so `compute_q_el` is RIM of `U*`."""
_unit_q_props(p::ThinPlateProps) =
    ThinPlateProps(; E=p.E, ν=p.ν, h=p.h, q_c=1.0, ρ=p.ρ)
function _unit_q_props(p::AnisoThinPlateProps)
    return AnisoThinPlateProps(; D11=p.D11, D22=p.D22, D12=p.D12, D16=p.D16,
        D26=p.D26, D66=p.D66, d=p.d, e=p.e, C1=p.C1, C2=p.C2, C3=p.C3,
        q_c=1.0, ρ=p.ρ, h=p.h)
end

"""RBF / collocation centres: boundary + internals + Kirchhoff corners."""
function _plate_dibem_centers(dad::BEMdata{<:AbstractThinPlate})
    corners = ThinPlate._plate_corners(dad)
    pts = Point2D[point(dad, i) for i in 1:dad.nt]
    for c in corners
        push!(pts, c.pos)
    end
    return pts
end

"""Packed w-column for DIBEM centre `j` (1:nt collocation, then corners)."""
@inline function _kirchhoff_wcol(j::Int, nt::Int)
    return j <= nt ? 2j - 1 : 2 * nt + (j - nt)
end

"""BIE rows at source centre `j` that receive `U*` (`w` and, on Γ, `∂w/∂m`)."""
function _kirchhoff_src_rows(j::Int, n::Int, nt::Int)
    if j <= n
        return (2j - 1):(2j)
    elseif j <= nt
        return (2j - 1):(2j - 1)
    else
        r = 2 * nt + (j - nt)
        return r:r
    end
end

"""RIM of `U*` (analytic primitives of `compute_q_el` with `q=1`)."""
function _plate_dibem_ID(dad::BEMdata{<:AbstractThinPlate}; npg::Int=10,
        threaded::Bool=true)
    p1 = _unit_q_props(dad.properties)
    corners = ThinPlate._plate_corners(dad)
    n, nt, nc = dad.n, dad.nt, length(corners)
    ID = zeros(2 * nt + nc)
    qsi, w = gausslegendre(npg)
    _collocation_loop!(threaded, nt) do i
        pf = point(dad, i)
        nf = i <= n ? dad.Normal[i] : zero(pf)
        acc1 = 0.0
        acc2 = 0.0
        for el in dad.elements
            qe = ThinPlate.compute_q_el(pf, nf, dad, el, qsi, w, p1)
            acc1 += qe[1]
            i <= n && (acc2 += qe[2])
        end
        ID[2i - 1] = acc1
        i <= n && (ID[2i] = acc2)
    end
    nf0 = zero(Point2D)
    @inbounds for (c, corner) in enumerate(corners)
        acc = 0.0
        pf = corner.pos
        for el in dad.elements
            acc += ThinPlate.compute_q_el(pf, nf0, dad, el, qsi, w, p1)[1]
        end
        ID[2 * nt + c] = acc
    end
    return ID
end

"""
    dibem_plate!(dad::BEMdata{<:AbstractThinPlate}; npg=10, rbf=PHS(),
                 threaded=true, apply_load=false) -> M

Kirchhoff DIBEM operator `M` (BEM atual `Monta_M_RIMd`).

```
∫_Ω q(X) U*(ξ, X) dΩ  ≈  (M q)(ξ)
```

Laplace-style remainder form: PHS Gram `F` on boundary + internals +
corners, `c` from CPD (`F c = IF`), off-diagonal
`A_{ij} = U*(ξ_i, x_j) c_j` (`U*` = `(w, ∂w/∂m)` on Γ, `w` only at
internals/corners), diagonal so `A 1 = ID`. Packed into the 2-DOF + dummy
internal slope + corner layout of `dad.H` (body force on **w-columns**).

`ID` is RIM of `U*` (same primitives as the particular integral at `q=1`).
`IF` is RIM of `∫_0^R φ ρ dρ`. Stores `dad.M`.

`apply_load=true` replaces `plate_q` with `M q_nodal` (linear `q_a x+q_b y+q_c`
sampled at the DIBEM centres). Default keeps the particular-integral `plate_q`
used by [`solve_plate!`](@ref).
"""
function dibem_plate!(dad::BEMdata{<:AbstractThinPlate}; npg::Int=10,
        rbf=PHS(), threaded::Bool=true, apply_load::Bool=false)
    n, nt = dad.n, dad.nt
    corners = ThinPlate._plate_corners(dad)
    nc = length(corners)
    ndof = 2 * nt + nc
    centers = _plate_dibem_centers(dad)
    npts = length(centers)
    props = dad.properties
    dummy = Point2D(1.0, 0.0)

    F = zeros(npts, npts)
    D = zeros(ndof, npts)
    _collocation_loop!(threaded, npts) do j
        xj = centers[j]
        @inbounds for i in 1:npts
            i == j && continue
            pf = centers[i]
            R = norm(xj - pf)
            R < 1e-30 && continue
            F[i, j] = rbf(R)
            nf = i <= n ? dad.Normal[i] : zero(pf)
            U, _ = ThinPlate.plate_kernels(xj, pf, dummy, nf, props)
            if i <= n
                D[2i - 1, j] = U[1, 1]
                D[2i, j] = U[2, 1]
            elseif i <= nt
                D[2i - 1, j] = U[1, 1]
            else
                D[2 * nt + (i - nt), j] = U[1, 1]
            end
        end
    end
    _dibem_ridge_F!(F)

    IF = _fsdt_IF(dad, centers, rbf; npg=npg)
    ID = _plate_dibem_ID(dad; npg=npg, threaded=threaded)
    IP = _fsdt_monomial_IP(dad, rbf; npg=npg)
    c = _dibem_poly_c(F, IF, centers, rbf; IP=IP)

    A = D .* c'
    @inbounds for j in 1:npts
        rows = _kirchhoff_src_rows(j, n, nt)
        rs = vec(sum(view(A, rows, :), dims=2))
        for (k, r) in enumerate(rows)
            A[r, j] = ID[r] - rs[k]
        end
    end
    M = zeros(ndof, ndof)
    @inbounds for j in 1:npts
        M[:, _kirchhoff_wcol(j, nt)] .= view(A, :, j)
    end
    set_cache!(dad; M, dibem_F=F, dibem_c=c, dibem_ID=ID, dibem_D=D,
        dibem_IF=IF, dibem_IP=IP, dibem_rbf=rbf, dibem_method=:dense)
    if apply_load
        p = dad.properties
        qw = zeros(ndof)
        @inbounds for j in 1:npts
            x = centers[j]
            qw[_kirchhoff_wcol(j, nt)] = p.q_a * x[1] + p.q_b * x[2] + p.q_c
        end
        set_cache!(dad; plate_q=M * qw)
    end
    _dibem_plate_dM!(dad, centers, c)
    return M
end

"""IBP `Mx, My` for Kirchhoff: `Mx_ij = -∂U*_w/∂X_x c_j`, `Mx 1 = 0`."""
function _dibem_plate_dM!(dad::BEMdata{<:AbstractThinPlate}, centers, c;
        hfd::Float64=1e-7)
    n, nt = dad.n, dad.nt
    npts = length(centers)
    ndof = size(dad.M, 1)
    dummy = Point2D(1.0, 0.0)
    props = dad.properties
    Mx = zeros(ndof, npts)
    My = zeros(ndof, npts)
    @inbounds for j in 1:npts
        cj = c[j]
        abs(cj) < 1e-30 && continue
        xj = centers[j]
        for i in 1:npts
            i == j && continue
            pf = centers[i]
            nf = i <= n ? dad.Normal[i] : zero(pf)
            Ufun = (pg, pfs, n̂) -> ThinPlate.plate_kernels(pg, pfs, dummy, nf, props)[1]
            dUx, dUy = _fd_dU_pg(Ufun, xj, pf, dummy, hfd)
            rows = _kirchhoff_src_rows(i, n, nt)
            for (k, r) in enumerate(rows)
                Mx[r, j] = -dUx[k, 1] * cj
                My[r, j] = -dUy[k, 1] * cj
            end
        end
    end
    @inbounds for i in 1:npts
        rows = _kirchhoff_src_rows(i, n, nt)
        Mx[rows, i] .= -vec(sum(view(Mx, rows, :), dims=2))
        My[rows, i] .= -vec(sum(view(My, rows, :), dims=2))
    end
    set_cache!(dad; Mx, My)
    return dad
end

"""
    plate_Mw(dad) -> Matrix

`ndof × nt` map from a scalar field at collocation (boundary + internals)
onto the Kirchhoff BIE: the w-columns of [`dibem_plate!`](@ref) `M`.
Same role as FSDT `M[:, 3j] / I0` in [`apply_shell_coupling!`](@ref).
"""
function plate_Mw(dad::BEMdata{<:AbstractThinPlate})
    has_cache(dad, :M) || error("plate_Mw: call dibem_plate! first")
    nt = dad.nt
    M = dad.M
    ndof = size(M, 1)
    Mw = zeros(ndof, nt)
    @inbounds for j in 1:nt
        Mw[:, j] .= view(M, :, 2j - 1)
    end
    return Mw
end
