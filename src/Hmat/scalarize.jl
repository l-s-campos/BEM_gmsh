# =============================================================================
# Reinterpret matrices of tensors as larger matrices of scalars
# =============================================================================

"""
    tensor_blocksize(T)

For a scalar type return `(1,1)`. For `SMatrix{p,q}` return `(p,q)`.
"""
tensor_blocksize(::Type{<:Number}) = (1, 1)
tensor_blocksize(::Type{<:SMatrix{p, q}}) where {p, q} = (p, q)
tensor_blocksize(::T) where {T} = tensor_blocksize(T)

"""
    is_tensor_eltype(T) -> Bool

`true` when `T` is a static matrix used as a tensor-valued kernel entry.
"""
is_tensor_eltype(::Type{<:Number}) = false
is_tensor_eltype(::Type{<:SMatrix}) = true
is_tensor_eltype(::Type) = false
is_tensor_eltype(::T) where {T} = is_tensor_eltype(T)

# ---- range / index maps -----------------------------------------------------

"""
    expand_range(r::UnitRange, p::Int) -> UnitRange

Map a block-index range `r` to the corresponding scalar-index range when each
block contributes `p` scalar rows/columns: `i ↦ (p*(i-1)+1):(p*i)`.
"""
function expand_range(r::UnitRange{<:Integer}, p::Integer)
    p == 1 && return Int(first(r)):Int(last(r))
    a, b = Int(first(r)), Int(last(r))
    return (p * (a - 1) + 1):(p * b)
end

"""
    block_index(i_scalar, p) -> (i_block, i_local)

Convert a 1-based scalar index to `(block index, local index in 1:p)`.
"""
function block_index(i::Integer, p::Integer)
    p == 1 && return Int(i), 1
    i0 = Int(i) - 1
    return div(i0, p) + 1, (i0 % p) + 1
end

"""
    scalar_index(i_block, i_local, p) -> Int
"""
scalar_index(i_block::Integer, i_local::Integer, p::Integer) =
    p * (Int(i_block) - 1) + Int(i_local)

function _expand_perm(perm::Vector{Int}, p::Int)
    n = length(perm)
    out = Vector{Int}(undef, p * n)
    @inbounds for (i, g) in enumerate(perm)
        for a in 1:p
            out[p * (i - 1) + a] = p * (g - 1) + a
        end
    end
    return out
end

# ---- vector scalarize / descalarize ----------------------------------------

"""
    scalarize(v)

Flatten a vector of tensor entries into a contiguous scalar vector.
- `Vector{<:Number}` → copy
- `Vector{SVector{p,T}}` → `Vector{T}` of length `p*n`
"""
scalarize(v::AbstractVector{<:Number}) = collect(v)

function scalarize(v::AbstractVector{SVector{p, T}}) where {p, T}
    n = length(v)
    out = Vector{T}(undef, p * n)
    @inbounds for i in 1:n
        base = p * (i - 1)
        vi = v[i]
        for a in 1:p
            out[base + a] = vi[a]
        end
    end
    return out
end

"""
    descalarize(v, ::Type{SVector{p,T}})

Inverse of [`scalarize`](@ref) for `SVector{p}` densities.
"""
function descalarize(v::AbstractVector{T}, ::Type{SVector{p, T}}) where {p, T}
    length(v) % p == 0 || throw(DimensionMismatch("length $(length(v)) not divisible by $p"))
    n = length(v) ÷ p
    out = Vector{SVector{p, T}}(undef, n)
    @inbounds for i in 1:n
        base = p * (i - 1)
        out[i] = SVector{p, T}(ntuple(a -> v[base + a], p))
    end
    return out
end

descalarize(v::AbstractVector{T}, ::Val{p}) where {T, p} = descalarize(v, SVector{p, T})

# ---- ScalarizedMatrix -------------------------------------------------------

"""
    struct ScalarizedMatrix{K,T,p,q} <: AbstractMatrix{T}

View of an `m×n` matrix `K` whose entries are `SMatrix{p,q,T}` as a plain
scalar matrix of size `(p*m) × (q*n)`.

Layout (block row-major / component order):
```
K_scalar[p*(i-1)+a, q*(j-1)+b] == K[i,j][a,b]
```

# Example
```julia
K = KernelMatrix(stokes_kernel, X, Y)   # eltype SMatrix{3,3,Float64}
S = ScalarizedMatrix(K)                 # (3m)×(3n) Float64 matrix
rt = expand_tree(Xclt, 3)
ct = expand_tree(Yclt, 3)
H = assemble_hmatrix(S, rt, ct; rtol=1e-4)
y = descalarize(H * scalarize(σ), Val(3))
```

See also [`assemble_hmatrix_scalarized`](@ref), [`expand_tree`](@ref).
"""
struct ScalarizedMatrix{K, T, p, q} <: AbstractMatrix{T}
    parent::K
    function ScalarizedMatrix(parent::AbstractMatrix)
        Te = eltype(parent)
        is_tensor_eltype(Te) || throw(
            ArgumentError("ScalarizedMatrix expects SMatrix eltype, got $Te"),
        )
        p, q = tensor_blocksize(Te)
        T = eltype(Te)
        return new{typeof(parent), T, p, q}(parent)
    end
end

Base.parent(S::ScalarizedMatrix) = S.parent
tensor_blocksize(S::ScalarizedMatrix{<:Any, <:Any, p, q}) where {p, q} = (p, q)

function Base.size(S::ScalarizedMatrix{<:Any, <:Any, p, q}) where {p, q}
    m, n = size(S.parent)
    return (p * m, q * n)
end

function Base.getindex(
        S::ScalarizedMatrix{<:Any, T, p, q},
        i::Int,
        j::Int,
    ) where {T, p, q}
    ib, il = block_index(i, p)
    jb, jl = block_index(j, q)
    return S.parent[ib, jb][il, jl]
end

function getblock!(
        out::AbstractMatrix{T},
        S::ScalarizedMatrix{<:Any, T, p, q},
        irange_,
        jrange_,
    ) where {T, p, q}
    irange = irange_ isa Colon ? axes(S, 1) : irange_
    jrange = jrange_ isa Colon ? axes(S, 2) : jrange_
    @inbounds for (jloc, j) in enumerate(jrange)
        jb, jl = block_index(j, q)
        for (iloc, i) in enumerate(irange)
            ib, il = block_index(i, p)
            out[iloc, jloc] = S.parent[ib, jb][il, jl]
        end
    end
    return out
end

# ---- Expanded cluster tree --------------------------------------------------

"""
    mutable struct ExpandedClusterTree

Point [`ClusterTree`](@ref) with index ranges expanded by a block size `p`:
point `i` owns scalar indices `(p*(i-1)+1):(p*i)`. Geometry and admissibility
use the underlying point tree; only index bookkeeping changes.
"""
mutable struct ExpandedClusterTree{N, T}
    tree::ClusterTree{N, T}
    p::Int
    children::Vector{ExpandedClusterTree{N, T}}
    parentnode::ExpandedClusterTree{N, T}
    function ExpandedClusterTree{N, T}(
            tree::ClusterTree{N, T},
            p::Int,
            children,
            parentnode,
        ) where {N, T}
        node = new{N, T}(tree, p)
        node.children = children
        node.parentnode = isnothing(parentnode) ? node : parentnode
        return node
    end
end

"""
    expand_tree(tree::ClusterTree, p::Int) -> ExpandedClusterTree

Expand a point cluster tree so each point corresponds to `p` contiguous scalar
indices. If `p == 1`, returns `tree` unchanged.
"""
function expand_tree(tree::ClusterTree{N, T}, p::Integer) where {N, T}
    p = Int(p)
    p >= 1 || throw(ArgumentError("block size p must be ≥ 1"))
    p == 1 && return tree
    return _expand_tree(tree, p, nothing)
end

function _expand_tree(
        tree::ClusterTree{N, T},
        p::Int,
        parent::Union{ExpandedClusterTree{N, T}, Nothing},
    ) where {N, T}
    ch = ExpandedClusterTree{N, T}[]
    node = ExpandedClusterTree{N, T}(tree, p, ch, parent)
    for c in children(tree)
        push!(node.children, _expand_tree(c, p, node))
    end
    return node
end

isleaf(t::ExpandedClusterTree) = isempty(t.children)
isroot(t::ExpandedClusterTree) = t.parentnode === t
children(t::ExpandedClusterTree) = t.children
parentnode(t::ExpandedClusterTree) = t.parentnode
container(t::ExpandedClusterTree) = container(t.tree)
index_range(t::ExpandedClusterTree) = expand_range(index_range(t.tree), t.p)
Base.length(t::ExpandedClusterTree) = length(index_range(t))
loc2glob(t::ExpandedClusterTree) = _expand_perm(loc2glob(t.tree), t.p)
glob2loc(t::ExpandedClusterTree) = invperm(loc2glob(t))
diameter(t::ExpandedClusterTree) = diameter(t.tree)
distance(a::ExpandedClusterTree, b::ExpandedClusterTree) = distance(a.tree, b.tree)
root_elements(t::ExpandedClusterTree) = root_elements(t.tree)
elements(t::ExpandedClusterTree) = elements(t.tree)
center(t::ExpandedClusterTree) = center(container(t))

# ---- high-level API ---------------------------------------------------------

"""
    scalarize_kernel(K) -> (K_scalar, p, q)

Return a scalar view of `K` together with the tensor block sizes.
If `K` is already scalar, returns `(K, 1, 1)`.
"""
function scalarize_kernel(K::AbstractMatrix)
    Te = eltype(K)
    if is_tensor_eltype(Te)
        p, q = tensor_blocksize(Te)
        return ScalarizedMatrix(K), p, q
    else
        return K, 1, 1
    end
end

"""
    assemble_hmatrix_scalarized(K, rowtree, coltree; kwargs...)
    assemble_hmatrix_scalarized(K::AbstractKernelMatrix; kwargs...)

Assemble an [`HMatrix`](@ref) of **scalars** by reinterpreting a tensor-valued
kernel matrix as a larger scalar matrix ([`ScalarizedMatrix`](@ref)) and
expanding the cluster trees ([`expand_tree`](@ref)).

All keyword arguments are forwarded to [`assemble_hmatrix`](@ref).
"""
function assemble_hmatrix_scalarized(
        K::AbstractMatrix,
        rowtree,
        coltree;
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        comp = PartialACA(; atol, rank, rtol),
        kwargs...,
    )
    S, p, q = scalarize_kernel(K)
    return assemble_hmatrix(
        S, expand_tree(rowtree, p), expand_tree(coltree, q);
        comp, kwargs...,
    )
end

function assemble_hmatrix_scalarized(K::AbstractKernelMatrix; kwargs...)
    X = map(center, rowelements(K))
    Y = map(center, colelements(K))
    return assemble_hmatrix_scalarized(K, ClusterTree(X), ClusterTree(Y); kwargs...)
end

"""
    apply_scalarized(H, σ::Vector{SVector{p}})

Apply a scalarized H-matrix / matrix of size `(p m)×(p n)` to a blocked density
`σ`, returning a `Vector{SVector{p}}`.
"""
function apply_scalarized(
        H::AbstractMatrix,
        σ::AbstractVector{SVector{p, T}},
    ) where {p, T}
    return descalarize(H * scalarize(σ), SVector{p, T})
end
