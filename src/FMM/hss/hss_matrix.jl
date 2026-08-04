# =============================================================================
# HSS matrix + matvec (nested up/down) — Martinsson / Xia–Gu format
# =============================================================================

"""
    mutable struct HSSMatrix{R,T}

Hierarchically semiseparable matrix on a binary [`ClusterTree`](@ref).

- leaves: dense diagonal block `D`, bases `U,V`
- parents: sibling couplings `B01,B10`, nested translators `U,V`
"""
mutable struct HSSMatrix{R,T}
    tree::R
    U::HSSBasisID{T}
    V::HSSBasisID{T}
    D::Union{Matrix{T},Nothing}
    B01::Union{Matrix{T},Nothing}
    B10::Union{Matrix{T},Nothing}
    children::Vector{HSSMatrix{R,T}}
    parentnode::HSSMatrix{R,T}
    function HSSMatrix{R,T}(tree, parent=nothing) where {R,T}
        H = new{R,T}(tree)
        m = length(tree)
        H.U = HSSBasisID{T}(m, 0)
        H.V = HSSBasisID{T}(m, 0)
        H.D = nothing
        H.B01 = nothing
        H.B10 = nothing
        H.children = HSSMatrix{R,T}[]
        H.parentnode = isnothing(parent) ? H : parent
        return H
    end
end

isleaf(H::HSSMatrix) = isempty(H.children)
isroot(H::HSSMatrix) = H.parentnode === H
children(H::HSSMatrix) = H.children
index_range(H::HSSMatrix) = index_range(H.tree)
Base.size(H::HSSMatrix) = (length(H.tree), length(H.tree))
Base.size(H::HSSMatrix, d::Integer) = d == 1 || d == 2 ? length(H.tree) : 1
Base.eltype(::HSSMatrix{<:Any,T}) where {T} = T
Base.length(H::HSSMatrix) = length(H.tree)
rowperm(H::HSSMatrix) = loc2glob(H.tree)
colperm(H::HSSMatrix) = loc2glob(H.tree)
U_rank(H::HSSMatrix) = ncols(H.U)
V_rank(H::HSSMatrix) = ncols(H.V)

function Base.show(io::IO, H::HSSMatrix)
    print(io, "HSSMatrix{$(eltype(H))} $(size(H,1))×$(size(H,2)) maxrank=$(maxrank(H))")
end

function maxrank(H::HSSMatrix)
    r = max(U_rank(H), V_rank(H))
    for c in children(H)
        r = max(r, maxrank(c))
    end
    return r
end

function _build_hss_tree(::Type{HSSMatrix{R,T}}, tree, parent=nothing) where {R,T}
    H = HSSMatrix{R,T}(tree, parent)
    if !isleaf(tree)
        ch = children(tree)
        length(ch) == 2 || error("HSS requires a binary ClusterTree")
        H.children = [
            _build_hss_tree(HSSMatrix{R,T}, ch[1], H),
            _build_hss_tree(HSSMatrix{R,T}, ch[2], H),
        ]
    end
    return H
end

# ---------- matvec ----------
function LinearAlgebra.mul!(y::AbstractVector, H::HSSMatrix, x::AbstractVector,
                            a::Number=1, b::Number=0; global_index::Bool=true)
    length(x) == size(H, 2) || throw(DimensionMismatch())
    length(y) == size(H, 1) || throw(DimensionMismatch())
    xp = global_index ? x[colperm(H)] : collect(x)
    a != 1 && rmul!(xp, a)
    iszero(b) ? fill!(y, 0) : rmul!(y, b)
    yp = _hss_matvec(H, xp)
    if global_index
        invpermute!(yp, rowperm(H))
        y .+= yp
    else
        y .+= yp
    end
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, H::HSSMatrix, X::AbstractMatrix,
                            a::Number=1, b::Number=0; kwargs...)
    @inbounds for k in 1:size(X, 2)
        mul!(view(Y, :, k), H, view(X, :, k), a, b; kwargs...)
    end
    return Y
end

Base.:*(H::HSSMatrix, x::AbstractVector) = mul!(similar(x, eltype(H)), H, x)
Base.:*(H::HSSMatrix, X::AbstractMatrix) = mul!(similar(X, eltype(H), size(H, 1), size(X, 2)), H, X)

function _hss_matvec(H::HSSMatrix{R,T}, x::Vector{T}) where {R,T}
    y = zeros(T, length(x))
    zmap = IdDict{Any,Vector{T}}()
    _hss_up!(H, x, zmap)
    _hss_down!(H, y, x, zmap, T[])
    return y
end

function _hss_up!(H::HSSMatrix{R,T}, x, zmap) where {R,T}
    if isleaf(H)
        z = applyC(H.V, collect(view(x, index_range(H))))
    else
        z0 = _hss_up!(H.children[1], x, zmap)
        z1 = _hss_up!(H.children[2], x, zmap)
        z = applyC(H.V, vcat(z0, z1))
    end
    zmap[H] = z
    return z
end

function _hss_down!(H::HSSMatrix{R,T}, y, x, zmap, gin::Vector{T}) where {R,T}
    if isleaf(H)
        I = index_range(H)
        yloc = H.D * view(x, I)
        isempty(gin) || (yloc = yloc + apply(H.U, gin))
        view(y, I) .+= yloc
    else
        c0, c1 = H.children[1], H.children[2]
        z0, z1 = zmap[c0]::Vector{T}, zmap[c1]::Vector{T}
        g0 = isempty(H.B01) ? zeros(T, size(H.B01, 1)) : H.B01 * z1
        g1 = isempty(H.B10) ? zeros(T, size(H.B10, 1)) : H.B10 * z0
        if !isempty(gin)
            uin = apply(H.U, gin)
            r0u = size(H.B01, 1)
            g0 = g0 + uin[1:r0u]
            g1 = g1 + uin[(r0u + 1):end]
        end
        _hss_down!(c0, y, x, zmap, g0)
        _hss_down!(c1, y, x, zmap, g1)
    end
    return y
end

function Base.Matrix(H::HSSMatrix{R,T}; global_index=true) where {R,T}
    n = size(H, 1)
    M = zeros(T, n, n)
    ej = zeros(T, n)
    @inbounds for j in 1:n
        fill!(ej, 0)
        ej[j] = one(T)
        M[:, j] = _hss_matvec(H, ej)
    end
    global_index || return M
    p = rowperm(H)
    ip = invperm(p)
    return M[ip, ip]
end

function compression_ratio(H::HSSMatrix)
    n = length(H)
    return (n * n * sizeof(eltype(H))) / max(Base.summarysize(H), 1)
end
