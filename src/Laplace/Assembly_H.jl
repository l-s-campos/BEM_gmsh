export H_G_Hmat, corrige_diagonais!, MixedBCOperator

"""
    node_weights(dad::BEMdata) -> Vector{Float64}

Integration weights for each boundary collocation node (`Jacobian × quad weight`).
"""
function node_weights(dad::BEMdata)
    w = zeros(dad.n)
    for elem in dad.elements
        for (k, node) in enumerate(elem.index)
            w[node] = elem.Jacobian[k] * dad.elem_weight[k]
        end
    end
    return w
end

"""
    all_points(dad::BEMdata)

Boundary nodes followed by internal nodes.
"""
function all_points(dad::BEMdata)
    isempty(dad.internalNodes) && return dad.Nodes
    return vcat(dad.Nodes, dad.internalNodes)
end

# ---------------------------------------------------------------------------
# Kernel matrices (collocation / "direto" form, cf. calc_HeG_Hd)
# ---------------------------------------------------------------------------

"""
Double-layer kernel ``H``: entry `(i,j) = (∂G/∂n_j)(x_i,x_j) w_j`.
Square over all collocation points; internal columns are left at 0
(filled by [`corrige_diagonais!`](@ref)).
"""
struct LaplaceHKernel{P} <: AbstractMatrix{Float64}
    points::Vector{P}
    normals::Vector{P}
    weights::Vector{Float64}
    n_boundary::Int
    dim::Int
end

Base.size(K::LaplaceHKernel) = (length(K.points), length(K.points))

function Base.getindex(K::LaplaceHKernel, i::Int, j::Int)
    (i == j || j > K.n_boundary) && return 0.0
    r = K.points[j] - K.points[i]
    R2 = sum(abs2, r)
    R2 < 1e-30 && return 0.0
    Qast = if K.dim == 2
        dot(r, K.normals[j]) / (2π * R2)
    else
        R = sqrt(R2)
        dot(r, K.normals[j]) / (4π * R^3)
    end
    return Qast * K.weights[j]
end

"""
Single-layer kernel ``G``: size `(n_total × n_boundary)`.
"""
struct LaplaceGKernel{P} <: AbstractMatrix{Float64}
    points::Vector{P}
    weights::Vector{Float64}
    n_boundary::Int
    k::Float64
    dim::Int
end

Base.size(K::LaplaceGKernel) = (length(K.points), K.n_boundary)

function Base.getindex(K::LaplaceGKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    R = norm(r)
    R < 1e-15 && return 0.0
    # same sign convention as fundamental(::Laplace): q = -k ∂T/∂n
    Tast = if K.dim == 2
        -log(R) / (2π * K.k)
    else
        1 / (4π * K.k * R)
    end
    return Tast * K.weights[j]
end

"""
    H_G_Hmat(dad::BEMdata{<:Laplace}; atol=1e-6, nmax=32, eta=3.0, threads=true)

Assemble hierarchical approximations of ``H`` and ``G`` with partial ACA.

Follows the collocation strategy of legacy `calc_HeG_Hd`:
pointwise kernels + a posteriori diagonal correction.

Stores `H::HMatrix` (`nt×nt`) and `G::HMatrix` (`nt×n`) in `dad.cache`.
"""
function H_G_Hmat(
    dad::BEMdata{<:Laplace};
    atol=1e-6,
    nmax=32,
    eta=3.0,
    threads=true,
)
    points = collect(all_points(dad))
    weights = node_weights(dad)
    n = dad.n
    k = float(dad.properties.k)
    dim = dad.dimension

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    Xclt = ClusterTree(points, splitter)
    Yclt_H = ClusterTree(copy(points), splitter)
    Yclt_G = ClusterTree(collect(dad.Nodes), splitter)

    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol)

    KH = LaplaceHKernel(points, dad.Normal, weights, n, dim)
    KG = LaplaceGKernel(points, weights, n, k, dim)

    HH = assemble_hmatrix(KH, Xclt, Yclt_H; adm=adm, comp=comp, threads=threads)
    HG = assemble_hmatrix(KG, Xclt, Yclt_G; adm=adm, comp=comp, threads=threads)

    corrige_diagonais!(dad, HH, HG)
    set_cache!(dad; H=HH, G=HG)
    return HH, HG
end

"""
    corrige_diagonais!(dad, Hmat, Gmat)

Diagonal of ``H`` from the constant-field identity; internal free term → 1.
Diagonal of ``G`` from the linear-field identity ``H x + G (n·e) ≈ 0``.
"""
function corrige_diagonais!(dad::BEMdata{<:Laplace}, Hmat::HMatrix, Gmat::HMatrix)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    # Row-sum free term (boundary ≈ -1/2, interior ≈ -1 with this kernel sign)
    hsum = Hmat * ones(nt)
    _set_diagonal!(Hmat) do i
        -hsum[i]
    end

    pts = all_points(dad)
    # Linear field T = x·1 ;  ∂T/∂n = n·1 ;  q = -k ∂T/∂n
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    # H T - G q = 0  with G_ii currently 0 in the product:
    # (H T - G_off q)_i - G_ii q_i = 0  ⇒  G_ii = (H T - G q)_i / q_i
    resid = Hmat * xlin - Gmat * qlin
    _set_diagonal!(Gmat) do i
        i > n && return 0.0
        denom = qlin[i]
        abs(denom) < 1e-14 && return 0.0
        return resid[i] / denom
    end
    return nothing
end

function _set_diagonal!(fdiag, Hmat::HMatrix)
    piv = HMatrices.pivot(Hmat)
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue

        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        # only columns that exist (G may be rectangular)
        for (iloc, ig) in enumerate(irangeg)
            ig > size(Hmat, 2) && continue
            for (jloc, jg) in enumerate(jrangeg)
                if ig == jg && jloc <= size(data, 2) && iloc <= size(data, 1)
                    data[iloc, jloc] = fdiag(ig)
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Mixed-BC linear operator for H-matrix GMRES
# ---------------------------------------------------------------------------

"""
    MixedBCOperator

Matrix-free operator realizing the dense `applyBC` column swap for
hierarchical `H` (`nt×nt`) and `G` (`nt×n`):

- Dirichlet dof `j`: column is `-G[:, j]`, unknown is `q_j`
- Neumann dof `j`: column is `H[:, j]`, unknown is `T_j`
- Internal dof `j`: column is `H[:, j]`, unknown is `T_j`
"""
struct MixedBCOperator{TH,TG} <: AbstractMatrix{Float64}
    H::TH
    G::TG
    BC::Vector{Int}
    n::Int
    nt::Int
end

Base.size(A::MixedBCOperator) = (A.nt, A.nt)

function LinearAlgebra.mul!(y::AbstractVector, A::MixedBCOperator, x::AbstractVector)
    T = zeros(eltype(x), A.nt)
    q = zeros(eltype(x), A.n)
    @inbounds for j in 1:A.n
        if A.BC[j] == 0
            q[j] = x[j]
        else
            T[j] = x[j]
        end
    end
    @inbounds for j in (A.n+1):A.nt
        T[j] = x[j]
    end
    # y = H*T - G*q
    mul!(y, A.H, T)
    yg = A.G * q
    y .-= yg
    return y
end

Base.:*(A::MixedBCOperator, x::AbstractVector) = mul!(similar(x, size(A, 1)), A, x)

"""
Build RHS `b` consistent with [`MixedBCOperator`](@ref):
`b = -H * T_known + G * q_known`.
"""
function mixed_bc_rhs(H, G, dad::BEMdata{<:Laplace})
    Tknown = zeros(dad.nt)
    qknown = zeros(dad.n)
    @inbounds for j in 1:dad.n
        if dad.BC[j] == 0
            Tknown[j] = dad.BV[j]
        else
            qknown[j] = dad.BV[j]
        end
    end
    return -(H * Tknown) + (G * qknown)
end
