# Weak recursive skeletonization factorization (Ho–Ying RSF / FLAM rskelf).
# A ≈ product of sparse E^{-1}, L, D, U, F^{-1} with column ID sparsifiers.

"""One box (or root) factor in a [`RSKELFFactor`](@ref)."""
struct RSKELFStep{T}
    sk::Vector{Int}
    rd::Vector{Int}
    nbr::Vector{Int}
    T::Matrix{T}             # |sk| × |rd|
    Frr::LU{T, Matrix{T}}
    E::Matrix{T}             # |sk| × |rd|
    F::Matrix{T}             # |rd| × |sk|  (G)
    C::Matrix{T}             # |nbr| × |rd|
    D::Matrix{T}             # |rd| × |nbr|
    p::Vector{Int}
end

"""
    RSKELFFactor

Weak RS / HIF / strong RS factorization. `mul!` applies `A`, `ldiv!` applies
`A⁻¹` (Krylov `Pl`). Neighbor `C`/`D` blocks are empty for weak RS.
"""
struct RSKELFFactor{T}
    steps::Vector{RSKELFStep{T}}
    root_idx::Vector{Int}
    n::Int
    symm::Symbol
end

Base.size(F::RSKELFFactor) = (F.n, F.n)
Base.size(F::RSKELFFactor, d::Integer) = d == 1 || d == 2 ? F.n : 1
Base.eltype(::RSKELFFactor{T}) where {T} = T

function Base.show(io::IO, F::RSKELFFactor)
    nelim = sum(s -> length(s.rd), F.steps; init = 0)
    return print(io, "RSKELFFactor{", eltype(F), "} n=", F.n,
        " steps=", length(F.steps), " eliminated=", nelim,
        " root=", length(F.root_idx), " symm=:", F.symm)
end
Base.show(io::IO, ::MIME"text/plain", F::RSKELFFactor) = show(io, F)

# ---- blocks ------------------------------------------------------------------

function _rskelf_block(A, I::AbstractVector{Int}, J::AbstractVector{Int})
    T = eltype(A)
    M = Matrix{T}(undef, length(I), length(J))
    @inbounds for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        M[ii, jj] = T(A[i, j])
    end
    return M
end

function _rskelf_block(K::KernelMatrix{Tf, Tx, Ty, T}, I::AbstractVector{Int},
        J::AbstractVector{Int}) where {Tf, Tx, Ty, T}
    m, n = length(I), length(J)
    M = Matrix{T}(undef, m, n)
    X = K.X
    Y = K.Y
    f = K.f
    if m * n >= 16_384 && Threads.nthreads() > 1
        Threads.@threads for jj in 1:n
            @inbounds begin
                yj = Y[J[jj]]
                for ii in 1:m
                    M[ii, jj] = f(X[I[ii]], yj)
                end
            end
        end
    else
        @inbounds for jj in 1:n
            yj = Y[J[jj]]
            for ii in 1:m
                M[ii, jj] = f(X[I[ii]], yj)
            end
        end
    end
    return M
end

# ---- one-box elimination from Kid + Kself ------------------------------------

function _rskelf_factor_box(
        slf::Vector{Int},
        Kid::AbstractMatrix{T},
        Kself::Matrix{T},
        Knbr_slf::Matrix{T},
        Kslf_nbr::Matrix{T},
        nbr::Vector{Int};
        rtol,
        rank,
        Tmax,
        symm::Symbol,
        strong::Bool = false,
    ) where {T}
    nslf = length(slf)
    nslf < 2 && return nothing
    size(Kid, 2) == nslf || throw(DimensionMismatch("Kid columns vs slf"))
    sk, rd, Tm = interpolative_decomp(Kid; rtol = rtol, rank = rank, Tmax = Tmax)
    isempty(rd) && return nothing
    # sparsify self
    if symm === :s
        Kself[rd, :] .-= transpose(Tm) * Kself[sk, :]
    else
        Kself[rd, :] .-= adjoint(Tm) * Kself[sk, :]
    end
    Kself[:, rd] .-= Kself[:, sk] * Tm
    nbr_f = strong ? nbr : Int[]
    Knbr_slf = strong ? Knbr_slf : zeros(T, 0, nslf)
    Kslf_nbr = strong ? Kslf_nbr : zeros(T, nslf, 0)
    if !isempty(nbr_f)
        Knbr_slf[:, rd] .-= Knbr_slf[:, sk] * Tm
        if symm !== :s && !isempty(Kslf_nbr)
            if symm === :n
                Kslf_nbr[rd, :] .-= adjoint(Tm) * Kslf_nbr[sk, :]
            else
                Kslf_nbr[rd, :] .-= transpose(Tm) * Kslf_nbr[sk, :]
            end
        end
    end
    Arr = Kself[rd, rd]
    Frr = try
        lu(Arr)
    catch
        lu(Arr + T(1e-12) * I)
    end
    Uf = UpperTriangular(Frr.factors)
    Lf = UnitLowerTriangular(Frr.factors)
    E = Kself[sk, rd] / Uf
    G = Lf \ Kself[rd[Frr.p], sk]
    if isempty(nbr_f)
        C = zeros(T, 0, length(rd))
        Dv = zeros(T, length(rd), 0)
    else
        C = Knbr_slf[:, rd] / Uf
        Dv = if size(Kslf_nbr, 2) == length(nbr_f)
            Lf \ Kslf_nbr[rd[Frr.p], :]
        else
            Matrix(adjoint(C))
        end
    end
    Xsch = E * G
    step = RSKELFStep{T}(slf[sk], slf[rd], copy(nbr_f), Tm, Frr, E, G, C, Dv,
        Vector{Int}(Frr.p))
    return step, Xsch, slf[sk]
end

function _rskelf_root_step(idx::Vector{Int}, D::AbstractMatrix{T}) where {T}
    nrd = length(idx)
    nrd == 0 && return nothing
    Dd = Matrix{T}(D) + T(1e-14) * I
    Frr = lu(Dd)
    return RSKELFStep{T}(Int[], idx, Int[], zeros(T, 0, nrd), Frr,
        zeros(T, 0, nrd), zeros(T, nrd, 0), zeros(T, 0, nrd), zeros(T, nrd, 0),
        Vector{Int}(Frr.p))
end

# ---- apply -------------------------------------------------------------------

function _rskelf_sv_up!(s::RSKELFStep, x::AbstractVector)
    isempty(s.rd) && return x
    if !isempty(s.sk)
        mul!(view(x, s.rd), adjoint(s.T), view(x, s.sk), -one(eltype(x)), true)
    end
    tmp = x[s.rd[s.p]]
    ldiv!(UnitLowerTriangular(s.Frr.factors), tmp)
    x[s.rd] = tmp
    if !isempty(s.sk)
        mul!(view(x, s.sk), s.E, view(x, s.rd), -one(eltype(x)), true)
    end
    if !isempty(s.nbr) && size(s.C, 1) == length(s.nbr)
        mul!(view(x, s.nbr), s.C, view(x, s.rd), -one(eltype(x)), true)
    end
    return x
end

function _rskelf_sv_down!(s::RSKELFStep, x::AbstractVector)
    isempty(s.rd) && return x
    if !isempty(s.sk) && size(s.F, 2) == length(s.sk)
        mul!(view(x, s.rd), s.F, view(x, s.sk), -one(eltype(x)), true)
    end
    if !isempty(s.nbr) && size(s.D, 2) == length(s.nbr)
        mul!(view(x, s.rd), s.D, view(x, s.nbr), -one(eltype(x)), true)
    end
    ldiv!(UpperTriangular(s.Frr.factors), view(x, s.rd))
    if !isempty(s.sk)
        mul!(view(x, s.sk), s.T, view(x, s.rd), -one(eltype(x)), true)
    end
    return x
end

function _rskelf_mv_up!(s::RSKELFStep, x::AbstractVector)
    isempty(s.rd) && return x
    if !isempty(s.sk)
        mul!(view(x, s.sk), s.T, view(x, s.rd), one(eltype(x)), true)
    end
    copyto!(view(x, s.rd), UpperTriangular(s.Frr.factors) * x[s.rd])
    if !isempty(s.sk) && size(s.F, 2) == length(s.sk)
        mul!(view(x, s.rd), s.F, view(x, s.sk), one(eltype(x)), true)
    end
    return x
end

function _rskelf_mv_down!(s::RSKELFStep, x::AbstractVector)
    isempty(s.rd) && return x
    if !isempty(s.sk)
        mul!(view(x, s.sk), s.E, view(x, s.rd), one(eltype(x)), true)
    end
    if !isempty(s.nbr) && size(s.C, 1) == length(s.nbr)
        mul!(view(x, s.nbr), s.C, view(x, s.rd), one(eltype(x)), true)
    end
    tmp = UnitLowerTriangular(s.Frr.factors) * x[s.rd]
    x[s.rd[s.p]] = tmp
    if !isempty(s.sk)
        mul!(view(x, s.rd), adjoint(s.T), view(x, s.sk), one(eltype(x)), true)
    end
    return x
end

function LinearAlgebra.ldiv!(F::RSKELFFactor{T}, x::AbstractVector{T}) where {T}
    length(x) == F.n || throw(DimensionMismatch())
    for s in F.steps
        _rskelf_sv_up!(s, x)
    end
    for s in Iterators.reverse(F.steps)
        _rskelf_sv_down!(s, x)
    end
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::RSKELFFactor, x::AbstractVector)
    y === x || copyto!(y, x)
    return ldiv!(F, y)
end

function Base.:\(F::RSKELFFactor, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, F::RSKELFFactor{T}, x::AbstractVector;
        ) where {T}
    length(x) == F.n && length(y) == F.n || throw(DimensionMismatch())
    y === x || copyto!(y, x)
    for s in F.steps
        _rskelf_mv_up!(s, y)
    end
    for s in Iterators.reverse(F.steps)
        _rskelf_mv_down!(s, y)
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, F::RSKELFFactor, x::AbstractVector,
        a::Number, b::Number)
    if iszero(b)
        mul!(y, F, x)
        a != 1 && rmul!(y, a)
    else
        tmp = similar(y)
        mul!(tmp, F, x)
        y .= b .* y .+ a .* tmp
    end
    return y
end

Base.:*(F::RSKELFFactor, x::AbstractVector) = mul!(similar(x, eltype(F)), F, x)

# ---- proxy -------------------------------------------------------------------

"""
    circle_proxy(kernel, pts; npts=64, radius=1.5)

FLAM-style 2D proxy: `kernel(x,y)` on a circle of radius `radius` in the
reference box `[-1,1]²`, scaled by the **full** box side (not the half-width).
Neighbors with `||(x-ctr)/ℓ|| < radius` stay near. Returns
`pxyfun(slf, nbr, box) -> (Kpxy, nbr′)`.
"""
function circle_proxy(kernel, pts::AbstractVector; npts::Int = 64, radius = 1.5)
    θ = range(0.0, 2π; length = npts + 1)[1:(end - 1)]
    pref = [SVector(radius * cos(t), radius * sin(t)) for t in θ]
    r2 = float(radius)^2
    function pxyfun(slf::Vector{Int}, nbr::Vector{Int}, box::ClusterTree,
            ctr = nothing)
        c = ctr === nothing ? center(container(box)) : ctr
        ℓ = high_corner(container(box)) - low_corner(container(box))
        T = typeof(float(kernel(pts[slf[1]], pts[slf[1]])))
        Kpxy = Matrix{T}(undef, npts, length(slf))
        @inbounds for (j, i) in enumerate(slf)
            yi = pts[i]
            for (ii, p) in enumerate(pref)
                q = SVector(c[1] + p[1] * ℓ[1], c[2] + p[2] * ℓ[2])
                Kpxy[ii, j] = T(kernel(q, yi))
            end
        end
        nbr2 = Int[]
        @inbounds for i in nbr
            d = (pts[i] - SVector(c[1], c[2])) ./ SVector(ℓ[1], ℓ[2])
            sum(abs2, d) < r2 && push!(nbr2, i)
        end
        return Kpxy, nbr2
    end
    return pxyfun
end

function circle_proxy(K::KernelMatrix; kwargs...)
    return circle_proxy(kernel(K), rowelements(K); kwargs...)
end

"""
    sphere_proxy(kernel, pts; npts=64, radius=1.5)

FLAM-style 3D proxy: Fibonacci samples of a sphere of radius `radius` in the
reference cube `[-1,1]³`, scaled by the **full** box side. Neighbors with
`||(x-ctr)/ℓ|| < radius` stay near. Returns `pxyfun(slf, nbr, box) -> (Kpxy, nbr′)`.
"""
function sphere_proxy(kernel, pts::AbstractVector; npts::Int = 64, radius = 1.5)
    pref = Vector{SVector{3, Float64}}(undef, npts)
    φ = π * (3 - sqrt(5))
    @inbounds for i in 0:(npts - 1)
        y = 1 - 2 * (i + 0.5) / npts
        rxy = sqrt(max(0.0, 1 - y^2))
        th = φ * i
        pref[i + 1] = radius * SVector(rxy * cos(th), y, rxy * sin(th))
    end
    r2 = float(radius)^2
    function pxyfun(slf::Vector{Int}, nbr::Vector{Int}, box::ClusterTree,
            ctr = nothing)
        c = ctr === nothing ? center(container(box)) : ctr
        ℓ = high_corner(container(box)) - low_corner(container(box))
        T = typeof(float(kernel(pts[slf[1]], pts[slf[1]])))
        Kpxy = Matrix{T}(undef, npts, length(slf))
        @inbounds for (j, i) in enumerate(slf)
            yi = pts[i]
            for (ii, p) in enumerate(pref)
                q = SVector(c[1] + p[1] * ℓ[1], c[2] + p[2] * ℓ[2],
                    c[3] + p[3] * ℓ[3])
                Kpxy[ii, j] = T(kernel(q, yi))
            end
        end
        nbr2 = Int[]
        @inbounds for i in nbr
            cc = SVector(c[1], c[2], c[3])
            d = (pts[i] - cc) ./ ℓ
            sum(abs2, d) < r2 && push!(nbr2, i)
        end
        return Kpxy, nbr2
    end
    return pxyfun
end

function sphere_proxy(K::KernelMatrix; kwargs...)
    return sphere_proxy(kernel(K), rowelements(K); kwargs...)
end

# ---- tree walk ---------------------------------------------------------------

function _rskelf_init_skel(tree::ClusterTree)
    node_id(tree) == 0 && assign_node_ids!(tree)
    skel = Dict{Int, Vector{Int}}()
    for box in nodes(tree)
        id = node_id(box)
        skel[id] = isempty(children(box)) ? _srs_glob(box) : Int[]
    end
    return skel
end

function _rskelf_pull_children!(skel, lev)
    for box in lev
        id = node_id(box)
        for c in children(box)
            append!(skel[id], skel[node_id(c)])
        end
        unique!(skel[id])
    end
    return skel
end

function _rskelf_active_nbr(skel, neigh, id, slf, active)
    Nidx = Int[]
    for qid in neigh[id]
        haskey(skel, qid) || continue
        for i in skel[qid]
            active[i] && i ∉ slf && push!(Nidx, i)
        end
    end
    unique!(Nidx)
    return Nidx
end

"""
    rskelf(A, tree; rtol=1e-6, rank=typemax(Int), pxyfun=nothing, Tmax=2, symm=:n)

Weak recursive skeletonization factorization of square `A` (kernel entries
`A[i,j]` in global point order of `tree`). `pxyfun(slf, nbr, box) -> (Kpxy, nbr′)`
captures the far field; omit it to ID against all remaining off-diagonal.
`symm = :n` (unsymmetric) or `:s` (symmetric, skip the adjoint sample).
"""
function rskelf(
        A::AbstractMatrix{T},
        tree::ClusterTree;
        rtol = 1e-6,
        rank = typemax(Int),
        pxyfun = nothing,
        Tmax = 2,
        symm::Symbol = :n,
    ) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("rskelf needs a square matrix"))
    neigh, _, id2 = neighbor_il_lists(tree)
    levels = nodes_by_depth(tree)
    skel = _rskelf_init_skel(tree)
    active = trues(n)
    steps = RSKELFStep{T}[]
    Mblk = Dict{Int, Matrix{T}}()
    Midx = Dict{Int, Vector{Int}}()

    for lev in Iterators.reverse(levels)
        isempty(lev) && continue
        depth(first(lev)) == 0 && continue
        _rskelf_pull_children!(skel, lev)
        for box in lev
            id = node_id(box)
            slf = filter(i -> active[i], skel[id])
            length(slf) < 2 && continue
            nslf = length(slf)
            pos = Dict{Int, Int}(g => i for (i, g) in enumerate(slf))
            Mlocal = zeros(T, nslf, nslf)
            for c in children(box)
                cid = node_id(c)
                haskey(Midx, cid) || continue
                idx = [pos[g] for g in Midx[cid] if haskey(pos, g)]
                length(idx) == length(Midx[cid]) || continue
                Mlocal[idx, idx] = Mblk[cid]
                delete!(Mblk, cid)
                delete!(Midx, cid)
            end
            nbr = _rskelf_active_nbr(skel, neigh, id, slf, active)
            Kpxy = zeros(T, 0, nslf)
            if depth(box) >= 2
                if pxyfun === nothing
                    nbr = Int[i for i in 1:n if active[i] && i ∉ slf]
                else
                    Kpxy, nbr = pxyfun(slf, nbr, box)
                end
            end
            Anbr = isempty(nbr) ? zeros(T, 0, nslf) : _rskelf_block(A, nbr, slf)
            if symm === :n && !isempty(nbr)
                Anbr = vcat(Anbr, adjoint(_rskelf_block(A, slf, nbr)))
            end
            Kid = isempty(Kpxy) ? Anbr : vcat(Anbr, Kpxy)
            Kself = _rskelf_block(A, slf, slf) + Mlocal
            Kn = isempty(nbr) ? zeros(T, 0, nslf) : _rskelf_block(A, nbr, slf)
            Ks = (symm === :n && !isempty(nbr)) ? _rskelf_block(A, slf, nbr) :
                zeros(T, nslf, 0)
            got = _rskelf_factor_box(slf, Kid, Kself, Kn, Ks, nbr;
                rtol = rtol, rank = rank, Tmax = Tmax, symm = symm)
            if got === nothing
                skel[id] = slf
                Mblk[id] = Mlocal
                Midx[id] = slf
                continue
            end
            step, Xsch, skg = got
            push!(steps, step)
            for i in step.rd
                active[i] = false
            end
            skel[id] = skg
            skloc = [pos[g] for g in skg]
            Mblk[id] = Mlocal[skloc, skloc] - Xsch
            Midx[id] = skg
        end
    end

    root_idx = findall(active)
    isempty(root_idx) && (root_idx = Int[1])
    Droot = _rskelf_block(A, root_idx, root_idx)
    # leftover child Schur
    pos = Dict{Int, Int}(g => i for (i, g) in enumerate(root_idx))
    for (id, idx) in Midx
        isempty(idx) && continue
        all(haskey(pos, g) for g in idx) || continue
        loc = [pos[g] for g in idx]
        Droot[loc, loc] .+= Mblk[id]
    end
    rst = _rskelf_root_step(root_idx, Droot)
    rst !== nothing && push!(steps, rst)
    return RSKELFFactor{T}(steps, root_idx, n, symm)
end
