# H² LR via H2Lib's convert_h2matrix_hmatrix path: expand nested NNCA
# bases to an H-matrix, then the existing hierarchical LU.
# Native nested-basis lrdecomp (h2arith lrdecomp_h2matrix) needs truncated
# H² products and cluster-basis updates we do not store.

"""
    hmatrix(A::NNCAMatrix; adm=nothing)

Convert a **square** NNCA H² to an [`HMatrix`](@ref) (H2Lib
`convert_h2matrix_hmatrix`): M2L couplings become `RkMatrix` with expanded
cluster bases; neighbor/self blocks stay dense.

`adm` defaults to treating NNCA interaction-list pairs as admissible.
"""
function hmatrix(A::NNCAMatrix{T}; adm = nothing) where {T}
    A.n == A.n_col || throw(ArgumentError(
        "HMatrix(::NNCAMatrix) needs a square operator; got $(size(A))"))
    A.p == 1 || throw(ArgumentError(
        "HMatrix(::NNCAMatrix) is implemented for scalar NNCA (p=1)"))
    tree = A.tree
    adm_fun = adm === nothing ? _NNCABlockAdm(A) : adm
    H = HMatrix{T}(tree, tree, adm_fun)
    cache = Dict{Int, Matrix{T}}()
    _fill_h_from_nnca!(H, A, cache)
    return H
end

struct _NNCABlockAdm{T}
    A::NNCAMatrix{T}
end

function (adm::_NNCABlockAdm)(X, Y)
    (isleaf(X) || isleaf(Y)) && return false
    rid = node_id(X)
    cid = node_id(Y)
    il = adm.A.il
    1 <= rid <= length(il) || return false
    return cid in il[rid]
end

function _nnca_expand_V(A::NNCAMatrix{T}, id::Int, cache::Dict{Int, Matrix{T}}) where {T}
    got = get(cache, id, nothing)
    got !== nothing && return got
    b = A.boxes[id]
    node = A.id2node[id]
    if isleaf(node) || isempty(b.L2P)
        V = isempty(b.L2P) ? zeros(T, A.p * length(index_range(node)), 0) : copy(b.L2P)
        cache[id] = V
        return V
    end
    chunks = [_nnca_expand_V(A, node_id(c), cache) for c in children(node)]
    nr = sum(size(C, 1) for C in chunks; init=0)
    nc = sum(size(C, 2) for C in chunks; init=0)
    Vch = zeros(T, nr, nc)
    i0 = 0
    j0 = 0
    for C in chunks
        m, n = size(C)
        @inbounds Vch[(i0 + 1):(i0 + m), (j0 + 1):(j0 + n)] = C
        i0 += m
        j0 += n
    end
    V = Vch * b.L2P
    cache[id] = V
    return V
end

function _nnca_coupling(A::NNCAMatrix{T}, rid::Int, cid::Int) where {T}
    1 <= rid <= length(A.m2l_dsptr) - 1 || return nothing
    a = A.m2l_dsptr[rid]
    stop = A.m2l_dsptr[rid + 1]
    @inbounds while a < stop
        if A.m2l_src[a] == cid
            m = A.m2l_m[a]
            n = A.m2l_n[a]
            p0 = A.m2l_ptr[a]
            return reshape(A.m2l_data[p0:(p0 + m * n - 1)], m, n)
        end
        a += 1
    end
    return nothing
end

function _nnca_near_copy(A::NNCAMatrix{T}, rid::Int, cid::Int) where {T}
    if rid == cid
        S = A.boxes[rid].self
        isempty(S) || return S
    end
    1 <= rid <= length(A.near_dsptr) - 1 || return nothing
    a = A.near_dsptr[rid]
    stop = A.near_dsptr[rid + 1]
    @inbounds while a < stop
        if A.near_src[a] == cid
            m = A.near_m[a]
            n = A.near_n[a]
            p0 = A.near_ptr[a]
            return reshape(A.near_data[p0:(p0 + m * n - 1)], m, n)
        end
        a += 1
    end
    return nothing
end

function _fill_h_from_nnca!(H::HMatrix, A::NNCAMatrix{T}, cache) where {T}
    if !isleaf(H)
        for child in children(H)
            _fill_h_from_nnca!(child, A, cache)
        end
        return H
    end
    rid = node_id(rowtree(H))
    cid = node_id(coltree(H))
    if isadmissible(H)
        S = _nnca_coupling(A, rid, cid)
        Vr = _nnca_expand_V(A, rid, cache)
        Vc = _nnca_expand_V(A, cid, cache)
        if S === nothing
            S = zeros(T, size(Vr, 2), size(Vc, 2))
        end
        size(S, 1) == size(Vr, 2) && size(S, 2) == size(Vc, 2) ||
            (S = zeros(T, size(Vr, 2), size(Vc, 2)))
        H.data = RkMatrix(Vr * S, copy(Vc))
    else
        M = _nnca_near_copy(A, rid, cid)
        if M === nothing
            ir = rowrange(H)
            jr = colrange(H)
            M = zeros(T, length(ir), length(jr))
            S = _nnca_coupling(A, rid, cid)
            if S !== nothing
                Vr = _nnca_expand_V(A, rid, cache)
                Vc = _nnca_expand_V(A, cid, cache)
                if size(Vr, 1) == size(M, 1) && size(Vc, 1) == size(M, 2) &&
                        size(S, 1) == size(Vr, 2) && size(S, 2) == size(Vc, 2)
                    M = Vr * S * transpose(Vc)
                end
            end
        end
        H.data = Matrix{T}(M)
    end
    return H
end

function LinearAlgebra.lu!(::NNCAMatrix, args...; kwargs...)
    throw(ArgumentError(
        "lu! on NNCAMatrix is not in-place; use lu(A; method=:nested)"))
end
