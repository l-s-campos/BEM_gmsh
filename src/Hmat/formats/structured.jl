"""
    assemble_structured([T,], K, tree; format=:H, kwargs...)
    assemble_structured([T,], K, rowtree, coltree; format=:H, kwargs...)

Unified factory for rank-structured formats.

# Arguments
- `K`: matrix-like object (`AbstractMatrix`, [`KernelMatrix`](@ref), …)
- `tree` / `rowtree,coltree`: [`ClusterTree`](@ref) partition(s). A single `tree`
  is used for both sides when only one tree is supplied.

# Keywords
- `format`: one of
  - `:H` / `:HMatrix` — classical H-matrix ([`assemble_hmatrix`](@ref))
  - `:BLR` / `:BLRMatrix` — block low-rank ([`assemble_blr`](@ref))
  - `:H2` / `:NNCA` / `:H2Matrix` — NNCA H² ([`assemble_h2`](@ref))
- remaining keywords are forwarded only when accepted by the target assembler
  (`adm`, `comp`, `rtol`, `threads`, `global_index`, `rank`, `distributed`, …)

# Examples
```julia
tree = ClusterTree(points)
K = KernelMatrix(kernel, points, points)
H  = assemble_structured(K, tree; format=:H, comp=PartialACA(;rtol=1e-6))
B  = assemble_structured(K, tree; format=:BLR)
H2 = assemble_structured(K, tree; format=:H2, rtol=1e-6)
```
"""
function assemble_structured(::Type{T}, K, tree; format = :H, kwargs...) where {T}
    return assemble_structured(T, K, tree, tree; format, kwargs...)
end

function assemble_structured(K::AbstractMatrix, tree; kwargs...)
    return assemble_structured(eltype(K), K, tree; kwargs...)
end

function assemble_structured(K::AbstractMatrix, rowtree, coltree; kwargs...)
    return assemble_structured(eltype(K), K, rowtree, coltree; kwargs...)
end

function assemble_structured(
        ::Type{T},
        K,
        rowtree,
        coltree;
        format = :H,
        kwargs...,
    ) where {T}
    fmt = _normalize_format(format)
    kw = Dict{Symbol, Any}(kwargs)
    if fmt === :H
        return assemble_hmatrix(
            T, K, rowtree, coltree;
            _pick(kw, (:adm, :comp, :global_index, :threads, :distributed, :device))...,
        )
    elseif fmt === :BLR
        return assemble_blr(
            T, K, rowtree, coltree;
            _pick(kw, (:adm, :comp, :global_index, :threads, :rowperm, :colperm))...,
        )
    elseif fmt === :H2
        if !haskey(kw, :rtol) && haskey(kw, :comp) && kw[:comp] isa PartialACA
            kw[:rtol] = kw[:comp].rtol
        end
        return assemble_h2(
            T, K, rowtree;
            _pick(kw, (:rtol, :rank, :global_index, :threads, :device))...,
        )
    else
        throw(ArgumentError(
            "unknown structured format $(repr(format)); use :H, :BLR, or :H2",
        ))
    end
end

function _normalize_format(format)
    f = Symbol(format)
    f in (:H, :HMatrix) && return :H
    f in (:BLR, :BLRMatrix) && return :BLR
    f in (:H2, :H2Matrix, :NNCA, :nnca) && return :H2
    return f
end

function _pick(kw::Dict{Symbol, Any}, keys)
    pairs_out = Pair{Symbol, Any}[]
    for k in keys
        haskey(kw, k) && push!(pairs_out, k => kw[k])
    end
    return (; pairs_out...)
end
