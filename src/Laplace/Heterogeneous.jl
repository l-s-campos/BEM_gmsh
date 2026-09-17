# Smoothly inhomogeneous Laplace / Helmholtz via DIBEM
# (Barcelos, Loeffler & Lara, EABE 131:41–50, 2021; Laplace special case
#  Barcelos & Loeffler, EABE 105, 2019).
#
# Governing (static):  ∇·(K ∇u) = 0
# Poisson FS u* (package Laplace k=1). Regularized BIE:
#   Σ_j H_ij K_j (u_j − u_i) + Σ_j G_ij K_j qn_j + (A u)_i = 0
# with qn = ∂u/∂n and A the DIBEM interpolant of
#   [u(X)−u(ξ)] ∇K(X)·∇_X u*(ξ,X).
#
# Homogeneous K=K0, A=0 recovers H u = G q_phys (q_phys = −K0 qn).
#
# Helmholtz inertia is the existing DIBEM M acting on (ρ ∘ u).

export heterogeneous_K_operator, heterogeneous_system
export solve_heterogeneous!, eval_material_field
export HeterogeneousSector, heterogeneous_dst_operator

"""Evaluate `K` at every collocation node. `K` is a function `p -> Real` or a vector."""
function eval_material_field(dad::BEMdata, K)
    nt = dad.nt
    if K isa AbstractVector
        length(K) == nt || throw(DimensionMismatch("material field length $(length(K)) ≠ nt=$nt"))
        return collect(Float64, K)
    end
    Kv = zeros(nt)
    @inbounds for i in 1:nt
        p = point(dad, i)
        Kv[i] = float(K(p))
    end
    return Kv
end

"""`N_j = ∫_Ω φ(||X−X_j||) dΩ` via the same RIM as DIBEM `IF`."""
function _dibem_rbf_volume(dad::BEMdata{<:Laplace}, rbf)
    IF = zeros(dad.nt)
    ID = zeros(dad.nt)
    _dibem_accumulate_IF_ID!(IF, ID, dad, rbf)
    return IF, ID
end

"""RBF Gram `F_ij = φ(||x_i−x_j||)` plus a small ridge."""
function _dibem_gram(dad::BEMdata, rbf)
    nt = dad.nt
    pts = all_points(dad)
    F = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        F[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    _dibem_ridge_F!(F)
    return F, pts
end

"""Nodal `∇` of `vals` from the same RBF used for DIBEM. Returns `dim` vectors."""
function _grad_field(pts, vals, rbf)
    n = length(pts)
    n == 0 && return (Float64[], Float64[])
    dim = length(pts[1])
    F = zeros(n, n)
    Fd = ntuple(_ -> zeros(n, n), dim)
    @inbounds for j in 1:n, i in 1:n
        F[i, j] = rbf(norm(pts[i] - pts[j]))
        if i != j
            for α in 1:dim
                Fd[α][i, j] = ∂(rbf, α, pts[i], pts[j])
            end
        end
    end
    _dibem_ridge_F!(F)
    Finv = inv(F)
    return ntuple(α -> (Fd[α] * Finv) * vals, dim)
end

"""Cache key for geometry `dM` / RBF gradient maps (independent of `K`)."""
function _het_geom_key(dad::BEMdata, rbf)
    pd = hasproperty(rbf, :poly_deg) ? rbf.poly_deg : 0
    return (dad.nt, dad.dimension, float(dad.properties.k), typeof(rbf), pd)
end

"""Build and cache `dM^α_ij = S_j ∂u*/∂x_α(ξ_i, X_j)` and RBF `∇` maps.

`A p = B p − (B 1) ∘ p` with `B = Σ_α dM^α diag(∂K/∂x_α)`. `dM` and the
RBF gradient operators do not depend on `K`.
"""
function _ensure_het_geom!(dad::BEMdata{<:Laplace}, rbf)
    key = _het_geom_key(dad, rbf)
    if has_cache(dad, :het_dM) && dad.het_geom_key == key
        return dad
    end
    nt = dad.nt
    dim = dad.dimension
    k0 = float(dad.properties.k)
    F, pts = _dibem_gram(dad, rbf)
    N, _ = _dibem_rbf_volume(dad, rbf)
    S = F' \ N
    Fd = ntuple(_ -> zeros(nt, nt), dim)
    @inbounds for j in 1:nt, i in 1:nt
        if i != j
            for α in 1:dim
                Fd[α][i, j] = ∂(rbf, α, pts[i], pts[j])
            end
        end
    end
    Finv = inv(F)
    Ggrad = ntuple(α -> Fd[α] * Finv, dim)
    dM = ntuple(_ -> zeros(nt, nt), dim)
    @inbounds for i in 1:nt
        ξ = pts[i]
        for j in 1:nt
            i == j && continue
            dU = _poisson_grad_X(pts[j] - ξ, k0, dim)
            Sj = S[j]
            for α in 1:dim
                dM[α][i, j] = Sj * dU[α]
            end
        end
    end
    set_cache!(dad; het_dM=dM, het_Ggrad=Ggrad, het_S=S, het_geom_key=key,
        het_A_buf=zeros(nt, nt))
    return dad
end

"""`A` from cached `dM` and nodal `∇K` (`A = B − diag(B 1)`)."""
function _het_A_from_dM!(A::AbstractMatrix, dM, gK)
    nt = size(A, 1)
    A .= dM[1]
    v = gK[1]
    @inbounds for j in 1:nt, i in 1:nt
        A[i, j] *= v[j]
    end
    for α in 2:length(dM)
        d = dM[α]
        w = gK[α]
        @inbounds for j in 1:nt, i in 1:nt
            A[i, j] += d[i, j] * w[j]
        end
    end
    @inbounds for i in 1:nt
        s = 0.0
        for j in 1:nt
            s += A[i, j]
        end
        A[i, i] -= s
    end
    return A
end

"""`∇_X u*(ξ, X)` of the Poisson FS (`u* = −log R / (2πk)` or `1/(4π k R)`)."""
function _poisson_grad_X(r, k0, dim::Integer)
    R2 = dot(r, r)
    R2 < 1e-30 && return zero(r)
    if dim == 2
        return -r / (2π * k0 * R2)
    end
    R = sqrt(R2)
    return -r / (4π * k0 * R * R2)
end

"""
    heterogeneous_K_operator(dad, K; rbf=PHS(1)) -> A

DIBEM matrix `A` for the regularized conductivity-gradient kernel

```
[u(X) − u(ξ)] ∇K(X) · ∇_X u*(ξ, X)
```

so the domain integral is `A u`. `dad` must be `Laplace` (Poisson FS);
assemble `H,G` with `k=1`. Paper's "simple radial" is `PHS(1)` (`φ=r`).
"""
function heterogeneous_K_operator(dad::BEMdata{<:Laplace}, K; rbf=PHS(1; poly_deg=-1))
    Kv = eval_material_field(dad, K)
    _ensure_het_geom!(dad, rbf)
    gK = ntuple(α -> dad.het_Ggrad[α] * Kv, dad.dimension)
    _het_A_from_dM!(dad.het_A_buf, dad.het_dM, gK)
    return copy(dad.het_A_buf), Kv
end

"""
Weighted collocation operator for `∇·(K ∇u)=0`:

```
L u + G (K ∘ qn) = 0,    qn = ∂u/∂n
L = H diag(K) − diag(H K) + A
```

Homogeneous `K=K0` gives `H u = G q` with `q = −K0 qn`.
"""
function heterogeneous_L(dad::BEMdata{<:Laplace}, K; rbf=PHS(1; poly_deg=-1))
    has_cache(dad, :H) || error("call assemble!(dad) first")
    H = Matrix{Float64}(dad.H)
    A, Kv = heterogeneous_K_operator(dad, K; rbf=rbf)
    HK = H * Kv
    L = H * Diagonal(Kv) - Diagonal(HK) + A
    return L, Kv, A
end

"""Mixed BC system `A x = b` for unknown Neumann `u` and Dirichlet `qn`.

Optional `source` is `f` in `∇·(K ∇u) = f` (function, vector, or number).
The Poisson DIBEM mass gives `L u + G (K ∘ qn) = M f`.
"""
function heterogeneous_system(dad::BEMdata{<:Laplace}, K;
        rbf=PHS(1; poly_deg=-1), source=nothing,
        source_rbf=PHS(3; poly_deg=1))
    L, Kv, Aop = heterogeneous_L(dad, K; rbf=rbf)
    G = Matrix{Float64}(dad.G)
    n = dad.n
    nt = dad.nt
    BC = dad.BC
    BV = dad.BV
    Kb = Kv[1:n]
    Asys = copy(L)
    b = zeros(nt)
    @inbounds for j in 1:n
        if BC[j] == 0
            # unknown qn_j; known u_j = BV
            b .-= L[:, j] .* BV[j]
            Asys[:, j] .= G[:, j] .* Kb[j]
        else
            # known q_phys = BV = −K qn  ⇒  qn = −BV/K
            qn = -BV[j] / max(Kb[j], 1e-30)
            b .-= G[:, j] .* (Kb[j] * qn)
        end
    end
    if source !== nothing
        has_cache(dad, :M) || DIBEM(dad; rbf=source_rbf)
        fv = source isa Number ? fill(float(source), nt) : eval_material_field(dad, source)
        b .+= dad.M * fv
    end
    return Asys, b, Kv, L, Aop
end

"""
    solve_heterogeneous!(dad, K; rbf=PHS(1), source=nothing) -> T

Solve `∇·(K ∇u)=0`, or `∇·(K ∇u)=f` if `source` is set. Stores `dad.T`
and package flux `dad.q = −K ∂u/∂n` on the boundary.
"""
function solve_heterogeneous!(dad::BEMdata{<:Laplace}, K;
        rbf=PHS(1; poly_deg=-1), source=nothing, source_rbf=PHS(3; poly_deg=1))
    Asys, b, Kv, _, _ = heterogeneous_system(dad, K; rbf=rbf, source=source,
        source_rbf=source_rbf)
    x = bem_linsolve(Asys, b)
    n = dad.n
    T = zeros(dad.nt)
    qn = zeros(n)
    @inbounds for j in 1:n
        if dad.BC[j] == 0
            T[j] = dad.BV[j]
            qn[j] = x[j]
        else
            T[j] = x[j]
            qn[j] = -dad.BV[j] / max(Kv[j], 1e-30)
        end
    end
    if dad.nt > n
        T[n+1:end] .= x[n+1:end]
    end
    q = -Kv[1:n] .* qn
    set_cache!(dad; T=T, q=q, het_K=Kv, het_qn=qn)
    return T
end

# ---------------------------------------------------------------------------
# Domain superposition (DST): contrast on an interior sector
# ---------------------------------------------------------------------------

"""Internal sector: `K_int`, `ρ_int` inside `inside(p)`, contrast vs surrounding."""
struct HeterogeneousSector
    K::Function
    ρ::Function
    inside::Function
end

"""
Contrast DIBEM operator on nodes that lie in `sector`.

The 2021 paper models the body as surrounding `K^sur` on all of `Ω` plus
an internal problem with `K̄ = K^sur − K^int` (DST). Basis points of one
sector do not interpolate those of another; here that is a nodal mask.
Full DST also needs the inclusion boundary `Γ^int` for `H,G`; that term is
omitted until an inclusion mesh is supplied (smooth SIMP uses one sector).
"""
function heterogeneous_dst_operator(dad::BEMdata{<:Laplace}, Ksur, sector::HeterogeneousSector;
        rbf=PHS(1; poly_deg=-1))
    Ks = eval_material_field(dad, Ksur)
    Ki = eval_material_field(dad, p -> sector.inside(p) ? sector.K(p) : Ksur(p))
    # contrast K̄ = K^sur − K^int on the inclusion, 0 outside
    Kbar = zeros(dad.nt)
    mask = falses(dad.nt)
    @inbounds for i in 1:dad.nt
        p = point(dad, i)
        if sector.inside(p)
            mask[i] = true
            Kbar[i] = Ks[i] - Ki[i]
        end
    end
    A, _ = heterogeneous_K_operator(dad, Kbar; rbf=rbf)
    # kill rows/cols of nodes outside the sector (no cross-sector RBF)
    @inbounds for i in 1:dad.nt
        if !mask[i]
            A[i, :] .= 0
            A[:, i] .= 0
        end
    end
    return A, Kbar, mask
end
