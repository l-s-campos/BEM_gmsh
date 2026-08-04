"""
    HalfSpaceBEM

2D/3D elastic half-space contact operators with multiple acceleration backends,
following `calc_halfspace.jl` (legacy BEM.jl) and Pohrt–Li influence kernels.

# Acceleration methods
| Symbol | Description |
|--------|-------------|
| `:dense` | Full influence matrix |
| `:fft` | Circulant embedding + FFT convolution |
| `:hmatrix` | Hierarchical ACA (`HMatrices`) |
| `:fmm` | Fast multipole (`BEM.FMM` / `rfmm2d`) for the log kernel (2D/3D) |

# Wear
[`wear_2d`](@ref) / `desgaste_2D` — Archard wear stepping under constant load.
"""
module HalfSpaceBEM

using LinearAlgebra
using Statistics: mean
using SparseArrays
using FFTW
using StaticArrays
using ProgressMeter
using NearestNeighbors

# parent modules (loaded by BEM before this module)
using ..HMatrices
using ..FMM

export HalfSpace2D, HalfSpace3D
export build_operator, build_dense, build_fft, build_hmatrix, build_fmm
export contact_pressure_force, wear_2d, desgaste_2D
export OpBackend

# =============================================================================
# Problem data
# =============================================================================

"""
    HalfSpace2D(x0, xf, n; E=1.0, h0=nothing)

Uniform 1-D surface mesh on ``[x0,xf]`` with `n` panels.
`E` is the **plane-strain contact modulus** ``E/(1-ν²)`` (as in legacy `calc_K_2d`).
`h0` is the initial gap (default flat zero).
"""
struct HalfSpace2D{T}
    y::Vector{T}              # panel endpoints (n+1)
    x::Vector{T}              # collocation (panel centres, n)
    node::Vector{SVector{2,Int}}
    al::Vector{T}             # panel lengths
    h0::Vector{T}             # initial gap
    E::T
end

function HalfSpace2D(x0::Real, xf::Real, n::Int; E::Real=1.0, h0=nothing)
    T = float(promote_type(typeof(x0), typeof(xf), typeof(E)))
    y = collect(range(T(x0), stop=T(xf), length=n + 1))
    x = (y[1:end-1] .+ y[2:end]) ./ 2
    node = [SVector(i, i + 1) for i in 1:n]
    al = abs.(y[2:end] .- y[1:end-1])
    h = h0 === nothing ? zeros(T, n) : T.(h0)
    length(h) == n || throw(DimensionMismatch("h0 length"))
    return HalfSpace2D{T}(y, x, node, al, h, T(E))
end

Base.length(dad::HalfSpace2D) = length(dad.x)

# =============================================================================
# Dense kernel  (legacy calc_K_2d)
# =============================================================================

"""
Influence entry ``K_{ij}`` for 2D half-plane (Flamant integrated on a panel).

```math
K_{ij} = -\\frac{4}{π E}\\big(ℓ_j + x_1\\log|x_1| - x_2\\log|x_2|\\big)
```
with ``x_{1,2}`` = endpoint distances from collocation ``i``.
"""
function kernel_entry(dad::HalfSpace2D, i::Int, j::Int)
    x1 = dad.y[dad.node[j][1]] - dad.x[i]
    x2 = dad.y[dad.node[j][2]] - dad.x[i]
    r1 = abs(x1)
    r2 = abs(x2)
    # continuous extension: s log|s| → 0 as s→0
    t1 = r1 > 0 ? x1 * log(r1) : zero(x1)
    t2 = r2 > 0 ? x2 * log(r2) : zero(x2)
    return (dad.al[j] + t1 - t2) * (-4 / (π * dad.E))
end

function build_dense(dad::HalfSpace2D)
    n = length(dad)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = kernel_entry(dad, i, j)
    end
    return K
end

# AbstractMatrix wrapper for H-matrix / generic matvec
struct HS2DKernel{T} <: AbstractMatrix{Float64}
    dad::HalfSpace2D{T}
end
Base.size(K::HS2DKernel) = (length(K.dad), length(K.dad))
Base.getindex(K::HS2DKernel, i::Int, j::Int) = kernel_entry(K.dad, i, j)

# =============================================================================
# FFT backend
# =============================================================================

struct FFTOp{T}
    dad::HalfSpace2D{T}
    Khat::Vector{ComplexF64}
    M::Int
    n::Int
end

function build_fft(dad::HalfSpace2D)
    n = length(dad)
    M = 2n
    # Toeplitz first column/row via kernel_entry(i,1) and kernel_entry(1,j)
    col = [kernel_entry(dad, i, 1) for i in 1:n]
    row = [kernel_entry(dad, 1, j) for j in 1:n]
    # circulant embed
    c = zeros(Float64, M)
    c[1:n] .= col
    c[n+2:M] .= reverse(row[2:n])
    return FFTOp(dad, rfft(c), M, n)
end

function LinearAlgebra.mul!(y::AbstractVector, op::FFTOp, x::AbstractVector)
    pad = zeros(Float64, op.M)
    pad[1:op.n] .= x
    full = irfft(op.Khat .* rfft(pad), op.M)
    copyto!(y, @view full[1:op.n])
    return y
end
Base.:*(op::FFTOp, x::AbstractVector) = mul!(similar(x), op, x)
Base.size(op::FFTOp) = (op.n, op.n)
Base.eltype(::FFTOp) = Float64

# =============================================================================
# H-matrix backend
# =============================================================================

function build_hmatrix(dad::HalfSpace2D; nmax=32, atol=1e-8, eta=3.0)
    n = length(dad)
    # embed 1-D points as 2-D for the cluster tree
    pts = [SVector(dad.x[i], 0.0) for i in 1:n]
    splitter = HMatrices.PrincipalComponentSplitter(; nmax=nmax)
    clt = ClusterTree(pts, splitter)
    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol)
    K = HS2DKernel(dad)
    return assemble_hmatrix(K, clt, clt; adm=adm, comp=comp, threads=false)
end

# =============================================================================
# FMM operator (BEM.FMM / rfmm2d) for 2D half-space log kernel
# =============================================================================

"""
    FMMOp

Fast multipole matvec for the 2D half-plane influence operator.

Far field uses `FMM.rfmm2d` on the point log kernel
``c * sum_j L_j p_j log|x_i-x_j|`` with ``c = -4/(pi E)``.
Near field replaces FMM contributions by the exact panel integral
[`kernel_entry`](@ref).
"""
struct FMMOp{T}
    dad::HalfSpace2D{T}
    sources::Matrix{Float64}       # 2 × n  (collocation on x-axis)
    near::Vector{Vector{Int}}
    eps::Float64
    nmax::Int
    n::Int
end

function build_fmm(dad::HalfSpace2D; eps=1e-8, nmax=50, near_factor=12.0, θ=0.8)
    # θ kept for API compatibility with old Barnes–Hut call sites (ignored)
    n = length(dad)
    sources = zeros(Float64, 2, n)
    @inbounds for i in 1:n
        sources[1, i] = dad.x[i]
        sources[2, i] = 0.0
    end
    bt = BallTree(dad.x')
    near = Vector{Vector{Int}}(undef, n)
    @inbounds for i in 1:n
        r = near_factor * dad.al[i]
        near[i] = inrange(bt, [dad.x[i]], r)
    end
    return FMMOp(dad, sources, near, float(eps), Int(nmax), n)
end

Base.size(op::FMMOp) = (op.n, op.n)
Base.eltype(::FMMOp) = Float64
Base.:*(op::FMMOp, x::AbstractVector) = mul!(similar(x, Float64), op, x)

function LinearAlgebra.mul!(y::AbstractVector, op::FMMOp, p::AbstractVector)
    dad = op.dad
    n = op.n
    # kernel_entry = c * (ℓ + x1 log|x1| - x2 log|x2|), c = -4/(πE)
    # Far asymptotics: (ℓ + x1 log|x1| - x2 log|x2|) ~ -ℓ log|r|
    # so K_ij ~ (-c) * ℓ_j * log|r| = 4/(πE) * ℓ_j * log|r|
    c = -4 / (π * dad.E)
    q = dad.al .* p                     # panel lengths × pressure
    far = -c                            # +4/(πE)

    # 1) FMM: pot_i = Σ_{j≠i} q_j log|x_i-x_j|
    vals = FMM.rfmm2d(op.eps, op.sources; charges=q, pg=1, nmax=op.nmax)
    @inbounds for i in 1:n
        y[i] = far * vals.pot[i]
    end

    # 2) Near-field correction: remove FMM log, add exact panel kernel
    @inbounds for i in 1:n
        xi = dad.x[i]
        for j in op.near[i]
            if j != i
                r = abs(xi - dad.x[j])
                if r > 0
                    y[i] -= far * q[j] * log(r)
                end
            end
            y[i] += kernel_entry(dad, i, j) * p[j]
        end
    end
    return y
end

# =============================================================================
# Unified builder
# =============================================================================

const OpBackend = Union{Matrix{Float64},FFTOp,HMatrix,FMMOp}

"""
    build_operator(dad::HalfSpace2D, method::Symbol; kwargs...)

`method ∈ (:dense, :fft, :hmatrix, :fmm)`.
"""
function build_operator(dad::HalfSpace2D, method::Symbol=:dense; kwargs...)
    method === :dense && return build_dense(dad)
    method === :fft && return build_fft(dad)
    method === :hmatrix && return build_hmatrix(dad; kwargs...)
    method === :fmm && return build_fmm(dad; kwargs...)
    throw(ArgumentError("unknown method $method — use :dense, :fft, :hmatrix, :fmm"))
end

# generic matvec
_matvec(K::AbstractMatrix, p) = K * p
_matvec(K::FFTOp, p) = K * p
_matvec(K::FMMOp, p) = K * p
_matvec(K::HMatrix, p) = begin
    y = zeros(size(K, 1))
    mul!(y, K, p)
    y
end

# =============================================================================
# Contact pressure (force-controlled, Polonsky–Keer style)
# =============================================================================

"""
    contact_pressure_force(dad, K, W; kwargs...) -> (p, g)

Force-controlled frictionless contact on a half-space operator `K`
(dense / FFT / H-matrix / FMM). `W` is the target normal load.
"""
function contact_pressure_force(
    dad::HalfSpace2D,
    K,
    W::Real;
    p_min=0.0,
    Hmax=1e30,
    err_tol=1e-8,
    it_max=200,
    h0=dad.h0,
    p_con_ini=nothing,
)
    n = length(dad)
    area = sum(dad.al)
    p = p_con_ini === nothing ? fill(float(W) / area, n) : copy(p_con_ini)
    # ensure load
    p .*= (W / max(dot(p, dad.al), eps()))

    g = zeros(n)
    t = zeros(n)   # search direction
    err_hist = Float64[]

    u = _matvec(K, p)
    @. g = h0 - u
    # shift so min gap in contact is 0
    Ael = findall(p .> p_min)
    isempty(Ael) && (Ael = collect(1:n))
    g .-= minimum(g[Ael])

    for it in 1:it_max
        Ael = findall((p .> p_min) .& (p .< Hmax))
        isempty(Ael) && break

        # residual on contact: G = g - mean(g on Ael)
        ḡ = mean(g[Ael])
        r = zeros(n)
        @inbounds for i in Ael
            r[i] = g[i] - ḡ
        end
        # search direction ≈ residual (CG-like; simple gradient)
        if it == 1
            t .= r
        else
            # Polonsky: t = r + β t
            β = dot(r, r) / max(err_hist[end], eps())  # rough
            @. t = r  # simplified steepest descent (robust)
        end
        # matvec
        dt = _matvec(K, t)
        dt .-= mean(dt[Ael])
        num = dot(r[Ael], r[Ael])
        den = dot(t[Ael], dt[Ael])
        abs(den) < eps() && break
        α = num / den
        @. p = p - α * t
        # projections
        @inbounds for i in 1:n
            p[i] = clamp(p[i], p_min, Hmax)
        end
        # enforce load
        curW = dot(p, dad.al)
        curW > 0 && (p .*= W / curW)

        u = _matvec(K, p)
        @. g = h0 - u
        Ael = findall(p .> p_min)
        isempty(Ael) && break
        g .-= minimum(g[Ael])

        # relative error
        err = sqrt(sum(abs2, g[i] for i in Ael if g[i] > 0; init=0.0)) /
              max(abs(minimum(g[Ael])), eps())
        push!(err_hist, num)
        err < err_tol && break
    end
    return p, g
end

# =============================================================================
# Archard wear  (desgaste_2D)
# =============================================================================

"""
    wear_2d(dad, K, W; k_ar1, k_ar2, δ, nsteps) -> (wear1, wear2, p_hist)

Archard wear under constant load `W`. Each step slides distance `δ` and updates
gap ``h ← h + k_{ar} p δ`` for each body.

Alias: [`desgaste_2D`](@ref).
"""
function wear_2d(
    dad::HalfSpace2D,
    K,
    W::Real;
    k_ar1=1e-13,
    k_ar2=0.0,
    δ=1e-5,
    nsteps=100,
    p_min=0.0,
    Hmax=1e30,
    err_tol=1e-8,
    it_max=100,
)
    n = length(dad)
    wear1 = zeros(n, nsteps)
    wear2 = zeros(n, nsteps)
    p_hist = zeros(n, nsteps)
    w1 = zeros(n)
    w2 = zeros(n)
    p_con = nothing
    h_work = copy(dad.h0)

    @showprogress "wear steps" for i in 1:nsteps
        # temporary dad gap
        h_work .= dad.h0 .+ w1 .+ w2
        p_con, _ = contact_pressure_force(
            dad, K, W;
            p_min=p_min, Hmax=Hmax, err_tol=err_tol, it_max=it_max,
            h0=h_work, p_con_ini=p_con,
        )
        @. w1 += p_con * δ * k_ar1
        @. w2 += p_con * δ * k_ar2
        wear1[:, i] .= w1
        wear2[:, i] .= w2
        p_hist[:, i] .= p_con
    end
    return wear1, wear2, p_hist
end

const desgaste_2D = wear_2d

# =============================================================================
# 3D half-space (uniform rectangular grid, Love normal kernel)
# =============================================================================

"""
    HalfSpace3D(x0, xf, nx, y0, yf, ny; E=1.0, h0=nothing)

Uniform rectangular surface mesh on ``[x0,xf]×[y0,yf]``.
`E` = contact modulus ``E/(1-ν²)``.
"""
struct HalfSpace3D{T}
    x::Vector{SVector{2,T}}          # collocation centres (nx*ny)
    hx::T
    hy::T
    nx::Int
    ny::Int
    h0::Vector{T}
    E::T
end

function HalfSpace3D(x0, xf, nx::Int, y0=x0, yf=xf, ny::Int=nx; E=1.0, h0=nothing)
    T = float(promote_type(typeof(x0), typeof(xf), typeof(E)))
    hx = T(xf - x0) / nx
    hy = T(yf - y0) / ny
    xs = range(T(x0) + hx/2, stop=T(xf) - hx/2, length=nx)
    ys = range(T(y0) + hy/2, stop=T(yf) - hy/2, length=ny)
    pts = SVector{2,T}[SVector(xs[i], ys[j]) for j in 1:ny for i in 1:nx]
    n = nx * ny
    h = h0 === nothing ? zeros(T, n) : T.(h0)
    return HalfSpace3D{T}(pts, hx, hy, nx, ny, h, T(E))
end

Base.length(dad::HalfSpace3D) = length(dad.x)

"""Love kernel for normal displacement under uniform pressure on a rectangle."""
function kernel_entry_3d(dad::HalfSpace3D, i::Int, j::Int)
    # relative offsets in cell units
    ix = ((i - 1) % dad.nx) + 1
    iy = ((i - 1) ÷ dad.nx) + 1
    jx = ((j - 1) % dad.nx) + 1
    jy = ((j - 1) ÷ dad.nx) + 1
    di = ix - jx
    dj = iy - jy
    hx, hy, E = dad.hx, dad.hy, dad.E
    k = (di + 0.5) * hx
    m = (dj + 0.5) * hy
    l = (di - 0.5) * hx
    n = (dj - 0.5) * hy
    s(a, b) = sqrt(a * a + b * b)
    F = (
        k * log((m + s(k, m)) / (n + s(k, n))) +
        l * log((n + s(l, n)) / (m + s(l, m))) +
        m * log((k + s(k, m)) / (l + s(l, m))) +
        n * log((l + s(l, n)) / (k + s(k, n)))
    )
    # (1-ν²)/(π E_young) = 1/(π E*) with E*=E/(1-ν²); Love uses (1-ν)/(2πG)=1/(π E*)
    return F / (π * E)
end

function build_dense(dad::HalfSpace3D)
    n = length(dad)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = kernel_entry_3d(dad, i, j)
    end
    return K
end

struct HS3DKernel <: AbstractMatrix{Float64}
    dad::HalfSpace3D
end
Base.size(K::HS3DKernel) = (length(K.dad), length(K.dad))
Base.getindex(K::HS3DKernel, i::Int, j::Int) = kernel_entry_3d(K.dad, i, j)

function build_fft(dad::HalfSpace3D)
    nx, ny = dad.nx, dad.ny
    # 2D Toeplitz → circulant embed
    Mx, My = 2nx, 2ny
    C = zeros(Float64, Mx, My)
    @inbounds for dj in -(ny - 1):(ny - 1), di in -(nx - 1):(nx - 1)
        # map (di,dj) to kernel between cells
        i = 1 + max(di, 0) + max(dj, 0) * nx   # dummy — build via relative
        # direct: K[di,dj] using synthetic indices
        k = (di + 0.5) * dad.hx
        m = (dj + 0.5) * dad.hy
        l = (di - 0.5) * dad.hx
        n = (dj - 0.5) * dad.hy
        s(a, b) = sqrt(a * a + b * b)
        F = (
            k * log(max(m + s(k, m), eps()) / max(n + s(k, n), eps())) +
            l * log(max(n + s(l, n), eps()) / max(m + s(l, m), eps())) +
            m * log(max(k + s(k, m), eps()) / max(l + s(l, m), eps())) +
            n * log(max(l + s(l, n), eps()) / max(k + s(k, n), eps()))
        )
        val = F / (π * dad.E)
        ii = di >= 0 ? di + 1 : Mx + di + 1
        jj = dj >= 0 ? dj + 1 : My + dj + 1
        C[ii, jj] = val
    end
    return (; dad, Khat=rfft(C), Mx, My, nx, ny)
end

function Base.:*(op::NamedTuple{(:dad, :Khat, :Mx, :My, :nx, :ny), T}, p::AbstractVector) where {T}
    nx, ny, Mx, My = op.nx, op.ny, op.Mx, op.My
    P = reshape(p, nx, ny)
    pad = zeros(Float64, Mx, My)
    pad[1:nx, 1:ny] .= P
    full = irfft(op.Khat .* rfft(pad), Mx)
    return vec(full[1:nx, 1:ny])
end

function build_hmatrix(dad::HalfSpace3D; nmax=32, atol=1e-8, eta=3.0)
    pts = [SVector(p[1], p[2]) for p in dad.x]
    splitter = HMatrices.PrincipalComponentSplitter(; nmax=nmax)
    clt = ClusterTree(pts, splitter)
    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol)
    return assemble_hmatrix(HS3DKernel(dad), clt, clt; adm=adm, comp=comp, threads=false)
end


"""
3D half-space FMM via `FMM.lfmm3d` on collocation points (Boussinesq-like 1/r
far field). Near field uses exact [`kernel_entry`] for the 3D influence.
"""
struct FMMOp3D{T}
    dad::HalfSpace3D{T}
    sources::Matrix{Float64}   # 3 × n
    near::Vector{Vector{Int}}
    eps::Float64
    nmax::Int
    n::Int
    scale::Float64             # 1/(π E) factor matching dense kernel scale
end

function build_fmm(dad::HalfSpace3D; eps=1e-8, nmax=40, near_factor=3.0, kwargs...)
    n = length(dad)
    sources = zeros(Float64, 3, n)
    @inbounds for i in 1:n
        sources[1, i] = dad.x[i][1]
        sources[2, i] = dad.x[i][2]
        sources[3, i] = 0.0
    end
    # near neighbors in plane
    xy = zeros(2, n)
    @inbounds for i in 1:n
        xy[1, i] = dad.x[i][1]
        xy[2, i] = dad.x[i][2]
    end
    bt = BallTree(xy)
    near = Vector{Vector{Int}}(undef, n)
    r0 = near_factor * max(dad.hx, dad.hy)
    @inbounds for i in 1:n
        near[i] = inrange(bt, xy[:, i], r0)
    end
    scale = 1 / (π * dad.E)
    return FMMOp3D(dad, sources, near, float(eps), Int(nmax), n, float(scale))
end

Base.size(op::FMMOp3D) = (op.n, op.n)
Base.eltype(::FMMOp3D) = Float64
Base.:*(op::FMMOp3D, x::AbstractVector) = mul!(similar(x, Float64), op, x)

function LinearAlgebra.mul!(y::AbstractVector, op::FMMOp3D, p::AbstractVector)
    dad = op.dad
    n = op.n
    area = dad.hx * dad.hy
    q = area .* p
    # far: scaled 1/r potential (lfmm3d uses 1/(4πr); half-space ~ 1/(π E r))
    vals = FMM.lfmm3d(op.eps, op.sources; charges=q, pg=1, nmax=op.nmax)
    fac = 4 / dad.E
    @inbounds for i in 1:n
        y[i] = fac * vals.pot[i]
    end
    # near correction: replace FMM 1/r with exact 3D panel kernel
    @inbounds for i in 1:n
        for j in op.near[i]
            if j != i
                dx = dad.x[i][1] - dad.x[j][1]
                dy = dad.x[i][2] - dad.x[j][2]
                r = hypot(dx, dy)
                if r > 0
                    y[i] -= fac * q[j] * (1 / (4 * π)) / r
                end
            end
            y[i] += kernel_entry_3d(dad, i, j) * p[j]
        end
    end
    return y
end


# include 3D FMM in backend union
const OpBackend = Union{Matrix{Float64},FFTOp,HMatrix,FMMOp,FMMOp3D}
_matvec(K::FMMOp3D, p) = K * p

function build_operator(dad::HalfSpace3D, method::Symbol=:dense; kwargs...)
    method === :dense && return build_dense(dad)
    method === :fft && return build_fft(dad)
    method === :hmatrix && return build_hmatrix(dad; kwargs...)
    method === :fmm && return build_fmm(dad; kwargs...)
    throw(ArgumentError("unknown method $method"))
end

function contact_pressure_force(dad::HalfSpace3D, K, W::Real; kwargs...)
    # reuse 2D algorithm with area weights hx*hy
    n = length(dad)
    area_el = fill(dad.hx * dad.hy, n)
    # thin wrapper as HalfSpace2D-like
    fake = (
        al = area_el,
        h0 = dad.h0,
        x = dad.x,
    )
    # local copy of algorithm
    area = sum(area_el)
    p = fill(float(W) / area, n)
    h0 = dad.h0
    err_tol = get(kwargs, :err_tol, 1e-8)
    it_max = get(kwargs, :it_max, 200)
    p_min = get(kwargs, :p_min, 0.0)
    for it in 1:it_max
        u = K isa AbstractMatrix ? (K * p) : (K * p)
        g = h0 .- u
        Ael = findall(p .> p_min)
        isempty(Ael) && break
        g .-= minimum(g[Ael])
        r = zero(g)
        ḡ = mean(g[Ael])
        @inbounds for i in Ael
            r[i] = g[i] - ḡ
        end
        t = r
        dt = K isa AbstractMatrix ? (K * t) : (K * t)
        dt .-= mean(dt[Ael])
        num = dot(r[Ael], r[Ael])
        den = dot(t[Ael], dt[Ael])
        abs(den) < eps() && break
        α = num / den
        @. p = max(p - α * t, p_min)
        curW = dot(p, area_el)
        curW > 0 && (p .*= W / curW)
        num < err_tol && break
    end
    u = K isa AbstractMatrix ? (K * p) : (K * p)
    g = h0 .- u
    return p, g
end

end # module
