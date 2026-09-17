# Dual-threshold incomplete LU (Saad ILUT) and near-field sparse extract.

"""
    ILUTFactor

Incomplete LU with threshold (`ldiv!` / `\\`). `L` is unit lower, `U` is upper.
"""
struct ILUTFactor{T}
    L::SparseMatrixCSC{T, Int}
    U::SparseMatrixCSC{T, Int}
end

Base.size(F::ILUTFactor) = size(F.L)
Base.size(F::ILUTFactor, d::Integer) = size(F.L, d)
Base.eltype(::ILUTFactor{T}) where {T} = T

function LinearAlgebra.ldiv!(F::ILUTFactor{T}, x::AbstractVector{T}) where {T}
    length(x) == size(F, 1) || throw(DimensionMismatch())
    ldiv!(UnitLowerTriangular(F.L), x)
    ldiv!(UpperTriangular(F.U), x)
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::ILUTFactor, x::AbstractVector)
    y === x || copyto!(y, x)
    return ldiv!(F, y)
end

function Base.:\(F::ILUTFactor, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

"""
    ilut(A; lfil=10, droptol=1e-4) -> ILUTFactor

Saad ILUT on a square sparse matrix: each row of `L` and of `U` keeps at most
`lfil` entries besides the diagonal; entries below `droptol * ‖row‖_∞` are
dropped. `A` may be `SparseMatrixCSC` or [`NNCAMatrix`](@ref) (near field).
"""
function ilut(A::SparseMatrixCSC{T}; lfil::Integer = 10, droptol::Real = 1e-4) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("ilut needs a square matrix"))
    lfil = max(Int(lfil), 0)
    τ = float(droptol)
    # CSR-like row lists
    rows = [Tuple{Int, T}[] for _ in 1:n]
    rowsI, rowsJ, rowsV = findnz(A)
    @inbounds for t in eachindex(rowsI)
        push!(rows[rowsI[t]], (rowsJ[t], rowsV[t]))
    end
    Li = Int[]; Lj = Int[]; Lv = T[]
    Ui = Int[]; Uj = Int[]; Uv = T[]
    w = zeros(T, n)
    inw = fill(false, n)
    nzcols = Int[]
    U_cols = [Int[] for _ in 1:n]
    U_vals = [T[] for _ in 1:n]
    @inbounds for i in 1:n
        empty!(nzcols)
        rownorm = zero(real(T))
        for (j, aij) in rows[i]
            w[j] += aij
            if !inw[j]
                inw[j] = true
                push!(nzcols, j)
            end
            rownorm = max(rownorm, abs(aij))
        end
        drop = τ * (rownorm + eps(real(T)))
        for k in 1:(i - 1)
            inw[k] || continue
            wk = w[k]
            abs(wk) <= drop && continue
            ukk = isempty(U_vals[k]) ? one(T) : U_vals[k][1]
            iszero(ukk) && continue
            wk /= ukk
            w[k] = wk
            cols = U_cols[k]
            vals = U_vals[k]
            for t in 2:length(cols)
                j = cols[t]
                w[j] -= wk * vals[t]
                if !inw[j]
                    inw[j] = true
                    push!(nzcols, j)
                end
            end
        end
        # collect L (j<i) and U (j>=i)
        Lcand = Tuple{real(T), Int, T}[]
        Ucand = Tuple{real(T), Int, T}[]
        diag = zero(T)
        for j in nzcols
            wij = w[j]
            w[j] = zero(T)
            inw[j] = false
            abs(wij) <= drop && j != i && continue
            if j < i
                push!(Lcand, (abs(wij), j, wij))
            elseif j == i
                diag = wij
            else
                push!(Ucand, (abs(wij), j, wij))
            end
        end
        empty!(nzcols)
        _ilut_keep!(Lcand, lfil)
        _ilut_keep!(Ucand, lfil)
        if abs(diag) < 1e-14 * (rownorm + 1)
            diag = T(rownorm + 1e-14)
        end
        push!(Li, i); push!(Lj, i); push!(Lv, one(T))
        for (_, j, v) in Lcand
            push!(Li, i); push!(Lj, j); push!(Lv, v)
        end
        push!(Ui, i); push!(Uj, i); push!(Uv, diag)
        push!(U_cols[i], i)
        push!(U_vals[i], diag)
        for (_, j, v) in Ucand
            push!(Ui, i); push!(Uj, j); push!(Uv, v)
            push!(U_cols[i], j)
            push!(U_vals[i], v)
        end
    end
    L = sparse(Li, Lj, Lv, n, n)
    U = sparse(Ui, Uj, Uv, n, n)
    return ILUTFactor{T}(L, U)
end

function _ilut_keep!(cand::Vector{<:Tuple}, lfil::Int)
    length(cand) <= lfil && return cand
    partialsort!(cand, 1:lfil; rev = true)
    resize!(cand, lfil)
    return cand
end

"""Left preconditioner: ILUT of the H² near field, mapped to **global** index order."""
struct PermutedILUT{T}
    F::ILUTFactor{T}
    perm::Vector{Int}   # loc2glob
    tmp::Vector{T}
end

Base.size(P::PermutedILUT) = size(P.F)
Base.size(P::PermutedILUT, d::Integer) = size(P.F, d)
Base.eltype(::PermutedILUT{T}) where {T} = T

function LinearAlgebra.ldiv!(P::PermutedILUT{T}, x::AbstractVector{T}) where {T}
    n = length(x)
    n == length(P.perm) || throw(DimensionMismatch())
    tmp = P.tmp
    length(tmp) == n || resize!(tmp, n)
    perm = P.perm
    @inbounds for i in 1:n
        tmp[i] = x[perm[i]]
    end
    ldiv!(P.F, tmp)
    @inbounds for i in 1:n
        x[perm[i]] = tmp[i]
    end
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::PermutedILUT, x::AbstractVector)
    y === x || copyto!(y, x)
    return ldiv!(P, y)
end

function ilut(A::NNCAMatrix{T}; kwargs...) where {T}
    F = ilut(near_sparse(A); kwargs...)
    perm = A.p == 1 ? copy(A.perm) : _expand_perm(A.perm, A.p)
    return PermutedILUT{T}(F, perm, zeros(T, length(perm)))
end

"""
    near_sparse(A::NNCAMatrix)

Sparse matrix of packed near-field (and self) blocks in **tree-local** order,
same as `mul!(y, A, x; global_index=false)`.
"""
function near_sparse(A::NNCAMatrix{T}) where {T}
    n = A.p * A.n
    I = Int[]
    J = Int[]
    V = T[]
    p = A.p
    id2 = A.id2node
    @inbounds for id in A.leaf_ids
        node = id2[id]
        rows = collect(expand_range(index_range(node), p))
        S = A.boxes[id].self
        if !isempty(S) && size(S, 1) == length(rows)
            for jj in eachindex(rows), ii in eachindex(rows)
                push!(I, rows[ii])
                push!(J, rows[jj])
                push!(V, S[ii, jj])
            end
        end
        a = A.near_dsptr[id]
        stop = A.near_dsptr[id + 1]
        while a < stop
            src = A.near_src[a]
            src == id && continue
            cols = collect(expand_range(index_range(id2[src]), p))
            m = A.near_m[a]
            nb = A.near_n[a]
            M = reshape(view(A.near_data, A.near_ptr[a]:(A.near_ptr[a] + m * nb - 1)), m, nb)
            for jj in 1:nb, ii in 1:m
                push!(I, rows[ii])
                push!(J, cols[jj])
                push!(V, M[ii, jj])
            end
            a += 1
        end
    end
    S = sparse(I, J, V, n, n)
    d = Vector(diag(S))
    @inbounds for i in 1:n
        if iszero(d[i])
            push!(I, i)
            push!(J, i)
            push!(V, one(T))
        end
    end
    return sparse(I, J, V, n, n)
end
