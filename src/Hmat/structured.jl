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
  - `:HODLR` / `:HODLRMatrix` — hierarchically off-diagonal low-rank
  - `:HSS` / `:HSSMatrix` — hierarchically semi-separable
  - `:HBS` / `:HBSMatrix` — hierarchically block separable (alias of HSS;
    Martinsson Ch. 14–18 nested weak-admissibility format)
  - `:H2` / `:H2Matrix` — H^2 matrix with proxy-point bases
- remaining keywords are forwarded only when accepted by the target assembler
  (`adm`, `comp`, `rtol`, `method`, `threads`, `global_index`, `rank`,
  `oversampling`, `distributed`, …)

# Examples
```julia
tree = ClusterTree(points)
K = KernelMatrix(kernel, points, points)
H  = assemble_structured(K, tree; format=:H, comp=PartialACA(;rtol=1e-6))
B  = assemble_structured(K, tree; format=:BLR)
Hd = assemble_structured(K, tree; format=:HODLR)
S  = assemble_structured(K, tree; format=:HSS, method=:dense)
Hb = assemble_structured(K, tree; format=:HBS, method=:dense)  # ≡ HSS
H2 = assemble_structured(K, tree; format=:H2, rtol=1e-6, alpha=1.0)
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
            _pick(kw, (:adm, :comp, :global_index, :threads, :distributed))...,
        )
    elseif fmt === :BLR
        return assemble_blr(
            T, K, rowtree, coltree;
            _pick(kw, (:adm, :comp, :global_index, :threads, :rowperm, :colperm))...,
        )
    elseif fmt === :HODLR
        # optional: map rtol into PartialACA compressor
        if !haskey(kw, :comp) && (haskey(kw, :rtol) || haskey(kw, :atol) || haskey(kw, :rank))
            kw[:comp] = PartialACA(;
                rtol = get(kw, :rtol, 0.0),
                atol = get(kw, :atol, 0.0),
                rank = get(kw, :rank, typemax(Int)),
            )
            # fix default rtol when only rank/atol given — match PartialACA defaults
            if !haskey(kwargs, :rtol) && !haskey(kwargs, :atol) && !haskey(kwargs, :rank)
                nothing
            elseif !haskey(kwargs, :rtol)
                c = kw[:comp]
                kw[:comp] = PartialACA(; atol = c.atol, rank = c.rank, rtol = c.rtol)
            end
        end
        return assemble_hodlr(T, K, rowtree; _pick(kw, (:comp, :global_index))...)
    elseif fmt === :HSS || fmt === :HBS
        if !haskey(kw, :rtol) && haskey(kw, :comp) && kw[:comp] isa PartialACA
            kw[:rtol] = kw[:comp].rtol
        end
        return assemble_hss(
            T, K, rowtree;
            _pick(kw, (:rtol, :rank, :method, :global_index, :oversampling))...,
        )
    elseif fmt === :H2
        if !haskey(kw, :rtol) && haskey(kw, :comp) && kw[:comp] isa PartialACA
            kw[:rtol] = kw[:comp].rtol
        end
        return assemble_h2(
            T, K, rowtree;
            _pick(kw, (:alpha, :rtol, :rank, :nsample, :adm, :global_index, :symmetric))...,
        )
    else
        throw(ArgumentError(
            "unknown structured format $(repr(format)); use :H, :BLR, :HODLR, :HSS, :HBS, or :H2",
        ))
    end
end

function _normalize_format(format)
    f = Symbol(format)
    f in (:H, :HMatrix) && return :H
    f in (:BLR, :BLRMatrix) && return :BLR
    f in (:HODLR, :HODLRMatrix) && return :HODLR
    f in (:HSS, :HSSMatrix) && return :HSS
    f in (:HBS, :HBSMatrix) && return :HBS
    f in (:H2, :H2Matrix) && return :H2
    return f
end

function _pick(kw::Dict{Symbol, Any}, keys)
    pairs_out = Pair{Symbol, Any}[]
    for k in keys
        haskey(kw, k) && push!(pairs_out, k => kw[k])
    end
    return (; pairs_out...)
end
