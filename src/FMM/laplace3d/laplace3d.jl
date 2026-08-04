# =============================================================================
# 3D Laplace FMM — kernel 1/(4π r)
#
# Spherical multipoles + full dual-tree FMM:
#   P2M → M2M → M2L → L2L → L2P (+ P2P near field)
#
# Translations use equivalent charges on a Fibonacci sphere (stable, O(p⁴)
# per shift with precomputed unit-sphere matrix).
# =============================================================================

const INV4PI = 1 / (4π)

"""Associated Legendre P_n^m(x), m ≥ 0, Condon–Shortley phase."""
function _legendre_pm(n::Int, m::Int, x::Float64)
    m > n && return 0.0
    pmm = 1.0
    if m > 0
        somx2 = sqrt(max(0.0, 1 - x * x))
        fact = 1.0
        for _ in 1:m
            pmm *= -fact * somx2
            fact += 2
        end
    end
    n == m && return pmm
    pmmp1 = x * (2m + 1) * pmm
    n == m + 1 && return pmmp1
    pnm = pmmp1
    for ll in (m + 2):n
        pnn = (x * (2ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pnn
        pnm = pnn
    end
    return pnm
end

# Precomputed (n-m)!/(n+m)! table: FR[n+1, m+1]
const _FR_CACHE = Dict{Int,Matrix{Float64}}()

function _fact_ratio_table(p::Int)
    return get!(_FR_CACHE, p) do
        FR = ones(Float64, p + 1, p + 1)
        for n in 0:p
            for m in 1:n
                r = 1.0
                @inbounds for k in (n - m + 1):(n + m)
                    r /= k
                end
                FR[n + 1, m + 1] = r
            end
        end
        FR
    end
end

@inline function _fact_ratio(n::Int, m::Int)
    m == 0 && return 1.0
    r = 1.0
    @inbounds for k in (n - m + 1):(n + m)
        r /= k
    end
    return r
end

@inline function _cart2sph(v::SVector{3,Float64})
    r = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    r < 1e-30 && return 0.0, 0.0, 0.0
    θ = acos(clamp(v[3] / r, -1.0, 1.0))
    φ = atan(v[2], v[1])
    return r, θ, φ
end

"""All P_n^m(x) for 0≤m≤n≤p into table P[n+1, m+1]."""
function _legendre_table!(P::Matrix{Float64}, p::Int, x::Float64)
    # P[n+1, m+1] = P_n^m(x)
    fill!(P, 0.0)
    P[1, 1] = 1.0  # P_0^0
    p == 0 && return P
    # P_m^m recurrence
    somx2 = sqrt(max(0.0, 1 - x * x))
    fact = 1.0
    @inbounds for m in 1:p
        P[m + 1, m + 1] = -fact * somx2 * P[m, m]
        fact += 2
    end
    # P_{m+1}^m
    @inbounds for m in 0:(p - 1)
        P[m + 2, m + 1] = x * (2m + 1) * P[m + 1, m + 1]
    end
    # upward in n
    @inbounds for m in 0:p
        for n in (m + 2):p
            P[n + 1, m + 1] =
                (x * (2n - 1) * P[n, m + 1] - (n + m - 1) * P[n - 1, m + 1]) / (n - m)
        end
    end
    return P
end

function laplace3d_nterms(eps::Real; full_fmm::Bool=false)
    # Treecode (default): moderate p is enough; full FMM translations cost O(p^4).
    pmax = full_fmm ? 10 : 14
    return min(max(laplace_nterms(Float64(eps)) + 1, 4), pmax)
end

@inline _mp_offset(n::Int) = n * n
@inline _ncoeff_sph(p::Int) = (p + 1)^2

function _coeff_degree(i::Int)
    n = 0
    while (n + 1)^2 < i
        n += 1
    end
    return n
end

# ---------- form multipole / local ----------

function form_mpole3d!(
    mpole::Vector{Float64},
    center::SVector{3,Float64},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    dipoles::Union{Nothing,Vector{SVector{3,Float64}}},
    irange,
    p::Int;
    reset::Bool=true,
)
    reset && fill!(mpole, 0.0)
    FR = _fact_ratio_table(p)
    Ptab = zeros(Float64, p + 1, p + 1)
    @inbounds for i in irange
        d = sources[i] - center
        x, y, z = d[1], d[2], d[3]
        rs = sqrt(x * x + y * y + z * z)
        q = Float64(charges[i])
        if rs < 1e-30
            mpole[1] += q  # only monopole
            continue
        end
        ct = clamp(z / rs, -1.0, 1.0)
        φs = atan(y, x)
        _legendre_table!(Ptab, p, ct)
        # cos/sin multiples of φ
        c0, s0 = 1.0, 0.0
        c1, s1 = cos(φs), sin(φs)
        rsn = 1.0
        for n in 0:p
            off = _mp_offset(n)
            mpole[off + 1] += q * rsn * Ptab[n + 1, 1]
            cm, sm = c1, s1  # cos(mφ), sin(mφ) for m=1
            for m in 1:n
                nmf = FR[n + 1, m + 1]
                Pm = Ptab[n + 1, m + 1]
                c = q * rsn * nmf * Pm
                # e^{-imφ} = cos(-mφ)+i sin(-mφ) = cos(mφ) - i sin(mφ)
                mpole[off + 2m] += c * cm
                mpole[off + 2m + 1] += c * (-sm)
                # next angle: e^{i(m+1)φ} = e^{imφ} e^{iφ}
                cm2 = cm * c1 - sm * s1
                sm2 = sm * c1 + cm * s1
                cm, sm = cm2, sm2
            end
            rsn *= rs
        end
        if dipoles !== nothing
            # keep FD path (less common)
            δ = 1e-7
            dip = dipoles[i]
            for (ehat, dval) in (
                (SVector(1.0, 0.0, 0.0), dip[1]),
                (SVector(0.0, 1.0, 0.0), dip[2]),
                (SVector(0.0, 0.0, 1.0), dip[3]),
            )
                abs(dval) < 1e-30 && continue
                for sgn in (1.0, -1.0)
                    ds = sources[i] + (sgn * δ) * ehat - center
                    rs2, θ2, φ2 = _cart2sph(ds)
                    ct2 = cos(θ2)
                    w = dval / (2δ) * sgn
                    _legendre_table!(Ptab, p, ct2)
                    rsn = 1.0
                    c1, s1 = cos(φ2), sin(φ2)
                    for n in 0:p
                        off = _mp_offset(n)
                        mpole[off + 1] += w * rsn * Ptab[n + 1, 1]
                        cm, sm = c1, s1
                        for m in 1:n
                            c = w * rsn * FR[n + 1, m + 1] * Ptab[n + 1, m + 1]
                            mpole[off + 2m] += c * cm
                            mpole[off + 2m + 1] += c * (-sm)
                            cm2 = cm * c1 - sm * s1
                            sm2 = sm * c1 + cm * s1
                            cm, sm = cm2, sm2
                        end
                        rsn *= rs2
                    end
                end
            end
        end
    end
    return mpole
end

"""Form local expansion (inner) about center from far charges."""
function form_local3d!(
    localexp::Vector{Float64},
    center::SVector{3,Float64},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    irange,
    p::Int;
    reset::Bool=true,
)
    reset && fill!(localexp, 0.0)
    for i in irange
        d = sources[i] - center
        ρ, θs, φs = _cart2sph(d)
        ρ < 1e-30 && continue
        cts = cos(θs)
        q = Float64(charges[i])
        for n in 0:p
            off = _mp_offset(n)
            rpow = 1 / ρ^(n + 1)
            localexp[off + 1] += q * rpow * _legendre_pm(n, 0, cts)
            for m in 1:n
                nmf = _fact_ratio(n, m)
                P = _legendre_pm(n, m, cts)
                c = q * rpow * nmf * P
                localexp[off + 2m] += c * cos(-m * φs)
                localexp[off + 2m + 1] += c * sin(-m * φs)
            end
        end
    end
    return localexp
end

function _mpole_series(d::SVector{3,Float64}, mpole::Vector{Float64}, p::Int,
                       Ptab::Matrix{Float64}=zeros(p + 1, p + 1))
    x, y, z = d[1], d[2], d[3]
    rt = sqrt(x * x + y * y + z * z)
    rt < 1e-30 && return 0.0
    ct = clamp(z / rt, -1.0, 1.0)
    φt = atan(y, x)
    _legendre_table!(Ptab, p, ct)
    s = 0.0
    rpow = 1 / rt
    c1, s1 = cos(φt), sin(φt)
    @inbounds for n in 0:p
        off = _mp_offset(n)
        s += mpole[off + 1] * Ptab[n + 1, 1] * rpow
        cm, sm = c1, s1
        for m in 1:n
            Mr, Mi = mpole[off + 2m], mpole[off + 2m + 1]
            # 2 Re(M e^{imφ}) = 2(Mr cos - Mi sin) wait e^{imφ}=cm+i sm
            # Re((Mr+iMi)(cm+ism)) = Mr*cm - Mi*sm
            s += 2 * (Mr * cm - Mi * sm) * Ptab[n + 1, m + 1] * rpow
            cm2 = cm * c1 - sm * s1
            sm2 = sm * c1 + cm * s1
            cm, sm = cm2, sm2
        end
        rpow /= rt
    end
    return s
end

function _local_series(d::SVector{3,Float64}, localexp::Vector{Float64}, p::Int)
    r, θ, φ = _cart2sph(d)
    ct = cos(θ)
    s = 0.0
    rn = 1.0
    for n in 0:p
        off = _mp_offset(n)
        s += localexp[off + 1] * _legendre_pm(n, 0, ct) * rn
        for m in 1:n
            Pt = _legendre_pm(n, m, ct)
            Lr, Li = localexp[off + 2m], localexp[off + 2m + 1]
            s += 2 * (Lr * cos(m * φ) - Li * sin(m * φ)) * Pt * rn
        end
        rn *= r
    end
    return s
end

function eval_mpole3d!(
    pot::AbstractVector{<:Real},
    grad::Union{Nothing,AbstractMatrix{<:Real}},
    center::SVector{3,Float64},
    mpole::Vector{Float64},
    targets::Vector{SVector{3,Float64}},
    irange,
    p::Int,
)
    δ = 1e-7
    e1, e2, e3 = SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)
    Ptab = zeros(Float64, p + 1, p + 1)
    @inbounds for (jt, j) in enumerate(irange)
        d = targets[j] - center
        s = _mpole_series(d, mpole, p, Ptab)
        pot[jt] += s * INV4PI
        if grad !== nothing
            gx = (_mpole_series(d + δ * e1, mpole, p, Ptab) - _mpole_series(d - δ * e1, mpole, p, Ptab)) / (2δ)
            gy = (_mpole_series(d + δ * e2, mpole, p, Ptab) - _mpole_series(d - δ * e2, mpole, p, Ptab)) / (2δ)
            gz = (_mpole_series(d + δ * e3, mpole, p, Ptab) - _mpole_series(d - δ * e3, mpole, p, Ptab)) / (2δ)
            grad[1, jt] += gx * INV4PI
            grad[2, jt] += gy * INV4PI
            grad[3, jt] += gz * INV4PI
        end
    end
    return pot
end

function eval_local3d!(
    pot::AbstractVector{<:Real},
    grad::Union{Nothing,AbstractMatrix{<:Real}},
    center::SVector{3,Float64},
    localexp::Vector{Float64},
    targets::Vector{SVector{3,Float64}},
    irange,
    p::Int,
)
    δ = 1e-7
    e1, e2, e3 = SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)
    for (jt, j) in enumerate(irange)
        d = targets[j] - center
        s = _local_series(d, localexp, p)
        pot[jt] += s * INV4PI
        if grad !== nothing
            gx = (_local_series(d + δ * e1, localexp, p) - _local_series(d - δ * e1, localexp, p)) / (2δ)
            gy = (_local_series(d + δ * e2, localexp, p) - _local_series(d - δ * e2, localexp, p)) / (2δ)
            gz = (_local_series(d + δ * e3, localexp, p) - _local_series(d - δ * e3, localexp, p)) / (2δ)
            grad[1, jt] += gx * INV4PI
            grad[2, jt] += gy * INV4PI
            grad[3, jt] += gz * INV4PI
        end
    end
    return pot
end

# ---------- translation cache (equivalent charges) ----------

"""
Precomputed unit-sphere charge→multipole map for order `p`.
`A[:, j]` = multipole of unit charge at Fibonacci point j on the unit sphere.
"""
struct SphericalTranslateCache
    p::Int
    unit_pts::Vector{SVector{3,Float64}}
    A::Matrix{Float64}       # ncoeff × K, unit sphere
    deg::Vector{Int}         # degree of each coefficient row
    F::Factorization{Float64} # LU of regularized A
end

const _STCACHE = Dict{Int,SphericalTranslateCache}()

function get_spherical_cache(p::Int)
    return get!(_STCACHE, p) do
        K = _ncoeff_sph(p)
        origin = SVector(0.0, 0.0, 0.0)
        unit_pts = fibonacci_sphere(origin, 1.0, K)
        A = zeros(Float64, K, K)
        mp = zeros(Float64, K)
        ch1 = [1.0]
        for j in 1:K
            fill!(mp, 0)
            form_mpole3d!(mp, origin, [unit_pts[j]], ch1, nothing, 1:1, p; reset=true)
            A[:, j] .= mp
        end
        deg = [_coeff_degree(i) for i in 1:K]
        F = lu(A + 1e-12 * I)
        SphericalTranslateCache(p, unit_pts, A, deg, F)
    end
end

"""Solve A_R q = mpole for charges on sphere of radius R.

Uses A_R = D(R) * A_unit with D = diag(R.^n), so q = A \\ (mpole ./ R.^n).
"""
function _equiv_charges(mpole::Vector{Float64}, R::Float64, p::Int)
    cache = get_spherical_cache(p)
    K = length(mpole)
    R = max(R, 1e-12)
    b = similar(mpole)
    @inbounds for i in 1:K
        b[i] = mpole[i] / (R^cache.deg[i])
    end
    q = cache.F \ b
    pts = Vector{SVector{3,Float64}}(undef, K)
    @inbounds for j in 1:K
        u = cache.unit_pts[j]
        pts[j] = SVector(R * u[1], R * u[2], R * u[3])
    end
    return pts, q
end

function m2m_laplace3d!(
    parent_mp::Vector{Float64},
    parent_ctr::SVector{3,Float64},
    child_mp::Vector{Float64},
    child_ctr::SVector{3,Float64},
    child_R::Float64,
    p::Int,
)
    R = max(child_R * 1.05, 1e-12)
    pts_rel, q = _equiv_charges(child_mp, R, p)
    pts = [child_ctr + pts_rel[j] for j in eachindex(pts_rel)]
    form_mpole3d!(parent_mp, parent_ctr, pts, q, nothing, eachindex(pts), p; reset=false)
    return parent_mp
end

function m2l_laplace3d!(
    localexp::Vector{Float64},
    tctr::SVector{3,Float64},
    mpole::Vector{Float64},
    sctr::SVector{3,Float64},
    sR::Float64,
    p::Int,
)
    R = max(sR * 1.05, 1e-12)
    pts_rel, q = _equiv_charges(mpole, R, p)
    pts = [sctr + pts_rel[j] for j in eachindex(pts_rel)]
    form_local3d!(localexp, tctr, pts, q, eachindex(pts), p; reset=false)
    return localexp
end

function l2l_laplace3d!(
    child_local::Vector{Float64},
    child_ctr::SVector{3,Float64},
    parent_local::Vector{Float64},
    parent_ctr::SVector{3,Float64},
    child_R::Float64,
    p::Int,
)
    # Sample parent local on sphere about child, fit child local coeffs
    K = _ncoeff_sph(p)
    R = max(child_R * 0.5, 1e-12)
    pts = fibonacci_sphere(child_ctr, R, K)
    pots = zeros(Float64, K)
    @inbounds for j in 1:K
        pots[j] = _local_series(pts[j] - parent_ctr, parent_local, p)
    end
    # B maps local coeffs → pot at unit-sphere-scaled points about child
    # pot_j = sum_i B_ji L_i  where B_ji = basis_i(pts_j - child_ctr)
    B = zeros(Float64, K, K)
    origin = SVector(0.0, 0.0, 0.0)
    basis = zeros(Float64, K)
    for i in 1:K
        fill!(basis, 0)
        basis[i] = 1.0
        for j in 1:K
            B[j, i] = _local_series(pts[j] - child_ctr, basis, p)
        end
    end
    L = B \ pots
    child_local .+= L
    return child_local
end

# ---------- real multipole storage on tree (Float64 coeffs) ----------

mutable struct L3Exp
    multipole::Vector{Float64}
    localexp::Vector{Float64}
    center::SVector{3,Float64}
    R::Float64
end

function _allocate_l3(root::ClusterTree{3}, p::Int)
    nc = _ncoeff_sph(p)
    data = Dict{UInt,L3Exp}()
    for node in nodes(root)
        ctr = SVector{3,Float64}(center(container(node)))
        R = max(diameter(node) / 2, 1e-30)
        data[objectid(node)] = L3Exp(zeros(Float64, nc), zeros(Float64, nc), ctr, R)
    end
    return data
end

# ---------- public API ----------

"""
```julia
vals = lfmm3d(eps, sources; charges=nothing, dipvecs=nothing,
              targets=nothing, pg=0, pgt=0, nmax=40, η=1.0,
              threaded=false, full_fmm=false)
```

3D Laplace FMM for ``1/(4\\pi r)``.

- `full_fmm=false` (default): leaf spherical multipoles + dual-tree M2P
- `full_fmm=true`: P2M/M2M/M2L/L2L/L2P via equivalent-sphere translations
- `threaded=true`: parallel interaction list (set JULIA_NUM_THREADS)
"""
function lfmm3d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipvecs=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=40,
    η::Real=1.0,
    threaded::Bool=false,
    full_fmm::Bool=false,
)
    @assert size(sources, 1) == 3
    ns = size(sources, 2)
    has_c = charges !== nothing
    has_d = dipvecs !== nothing
    @assert has_c || has_d
    @assert pg > 0 || pgt > 0

    p = laplace3d_nterms(eps; full_fmm=full_fmm)
    ncoeff = _ncoeff_sph(p)
    adm = StrongAdmissibility(η=Float64(η))
    spl = GeometricSplitter(nmax=Int(nmax))
    stree, sl2g, src = build_point_tree(sources, spl)

    ch = has_c ? permute_to_local(vec(charges), sl2g) : zeros(Float64, ns)
    dips = nothing
    if has_d
        dips = Vector{SVector{3,Float64}}(undef, ns)
        @inbounds for i in 1:ns
            g = sl2g[i]
            dips[i] = SVector{3,Float64}(dipvecs[1, g], dipvecs[2, g], dipvecs[3, g])
        end
    end

    # only need sphere-translation cache for full FMM
    full_fmm && get_spherical_cache(p)
    # warm factorial table
    _fact_ratio_table(p)

    sdata = _allocate_l3(stree, p)

    function upward!(node)
        ed = sdata[objectid(node)]
        if isleaf(node)
            form_mpole3d!(ed.multipole, ed.center, src, ch, dips, index_range(node), p; reset=true)
        elseif full_fmm
            # M2M only needed for full FMM path
            chs = children(node)
            if threaded && length(chs) > 1
                Threads.@threads for c in chs
                    upward!(c)
                end
            else
                for c in chs
                    upward!(c)
                end
            end
            fill!(ed.multipole, 0)
            for c in chs
                cd = sdata[objectid(c)]
                m2m_laplace3d!(ed.multipole, ed.center, cd.multipole, cd.center, cd.R, p)
            end
        else
            # treecode: only form multipoles at leaves
            chs = children(node)
            if threaded && length(chs) > 1
                Threads.@threads for c in chs
                    upward!(c)
                end
            else
                for c in chs
                    upward!(c)
                end
            end
        end
    end

    vals = FMMVals()
    want_g_src = pg >= 2
    want_g_trg = pgt >= 2

    function run(tmat, is_src, want_grad)
        nt = size(tmat, 2)
        if is_src
            ttree, tl2g, tpts = stree, sl2g, src
            same = true
            tdata = sdata
        else
            ttree, tl2g, tpts = build_point_tree(tmat, spl)
            same = false
            tdata = _allocate_l3(ttree, p)
        end
        pot_loc = zeros(Float64, nt)
        grad_loc = want_grad ? zeros(Float64, 3, nt) : nothing

        # zero locals
        for ed in values(tdata)
            fill!(ed.localexp, 0)
        end

        upward!(stree)

        # shared near-field kernel (allocation-free)
        function p2p_near!(tnode, snode)
            @inbounds for j in index_range(tnode)
                tj = tpts[j]
                s = 0.0
                gx = gy = gz = 0.0
                for i in index_range(snode)
                    same && i == j && continue
                    si = src[i]
                    rx = tj[1] - si[1]
                    ry = tj[2] - si[2]
                    rz = tj[3] - si[3]
                    r2 = rx * rx + ry * ry + rz * rz
                    r2 < 1e-30 && continue
                    invr = 1 / sqrt(r2)
                    invr3 = invr / r2
                    qi = ch[i]
                    s += qi * INV4PI * invr
                    if want_grad
                        c = -qi * INV4PI * invr3
                        gx += c * rx; gy += c * ry; gz += c * rz
                    end
                    if dips !== nothing
                        di = dips[i]
                        ddot = di[1] * rx + di[2] * ry + di[3] * rz
                        s += ddot * INV4PI * invr3
                        if want_grad
                            invr5 = invr3 / r2
                            gx += INV4PI * (di[1] * invr3 - 3 * ddot * rx * invr5)
                            gy += INV4PI * (di[2] * invr3 - 3 * ddot * ry * invr5)
                            gz += INV4PI * (di[3] * invr3 - 3 * ddot * rz * invr5)
                        end
                    end
                end
                pot_loc[j] += s
                if want_grad
                    grad_loc[1, j] += gx
                    grad_loc[2, j] += gy
                    grad_loc[3, j] += gz
                end
            end
        end

        if full_fmm
            function m2l_job!(tnode, snode, _, __)
                td = tdata[objectid(tnode)]
                sd = sdata[objectid(snode)]
                m2l_laplace3d!(td.localexp, td.center, sd.multipole, sd.center, sd.R, p)
            end
            dummy_s = allocate_expdata(stree, 1)
            dummy_t = same ? dummy_s : allocate_expdata(ttree, 1)
            dualtree_interact!(
                ttree, stree, dummy_t, dummy_s, adm;
                m2l! = m2l_job!,
                p2p! = p2p_near!,
                leaf_mpole_only=false,
                threaded=threaded,
            )

            # downward L2L + L2P
            function downward!(node)
                ed = tdata[objectid(node)]
                if isleaf(node)
                    idx = index_range(node)
                    nloc = length(idx)
                    buf = zeros(Float64, nloc)
                    gbuf = want_grad ? zeros(Float64, 3, nloc) : nothing
                    eval_local3d!(buf, gbuf, ed.center, ed.localexp, tpts, idx, p)
                    for (jt, j) in enumerate(idx)
                        pot_loc[j] += buf[jt]
                        if want_grad
                            grad_loc[1, j] += gbuf[1, jt]
                            grad_loc[2, j] += gbuf[2, jt]
                            grad_loc[3, j] += gbuf[3, jt]
                        end
                    end
                else
                    for c in children(node)
                        cd = tdata[objectid(c)]
                        l2l_laplace3d!(cd.localexp, cd.center, ed.localexp, ed.center, cd.R, p)
                    end
                    chs = children(node)
                    if threaded && length(chs) > 1
                        Threads.@threads for c in chs
                            downward!(c)
                        end
                    else
                        for c in chs
                            downward!(c)
                        end
                    end
                end
            end
            downward!(ttree)
        else
            # leaf M2P treecode path
            # evaluate multipole directly into pot_loc (no temp buffers when no grad)
            Ptab_tls = zeros(Float64, p + 1, p + 1)
            function m2p_job!(tnode, snode, _, __)
                sd = sdata[objectid(snode)]
                ctr = sd.center
                mp = sd.multipole
                if !want_grad
                    @inbounds for j in index_range(tnode)
                        pot_loc[j] += INV4PI * _mpole_series(tpts[j] - ctr, mp, p, Ptab_tls)
                    end
                else
                    idx = index_range(tnode)
                    nloc = length(idx)
                    buf = zeros(Float64, nloc)
                    gbuf = zeros(Float64, 3, nloc)
                    eval_mpole3d!(buf, gbuf, ctr, mp, tpts, idx, p)
                    @inbounds for (jt, j) in enumerate(idx)
                        pot_loc[j] += buf[jt]
                        grad_loc[1, j] += gbuf[1, jt]
                        grad_loc[2, j] += gbuf[2, jt]
                        grad_loc[3, j] += gbuf[3, jt]
                    end
                end
            end
            dummy_s = allocate_expdata(stree, 1)
            dummy_t = same ? dummy_s : allocate_expdata(ttree, 1)
            dualtree_interact!(
                ttree, stree, dummy_t, dummy_s, adm;
                m2p! = m2p_job!,
                p2p! = p2p_near!,
                leaf_mpole_only=true,
                threaded=threaded,
            )
        end

        pot_g = zeros(Float64, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        grad_g = nothing
        if want_grad
            grad_g = zeros(Float64, 3, nt)
            @inbounds for i in 1:nt
                g = tl2g[i]
                grad_g[1, g] = grad_loc[1, i]
                grad_g[2, g] = grad_loc[2, i]
                grad_g[3, g] = grad_loc[3, i]
            end
        end
        return pot_g, grad_g
    end

    if pg > 0
        pot, grad = run(sources, true, want_g_src)
        vals.pot = pot
        want_g_src && (vals.grad = grad)
    end
    if pgt > 0 && targets !== nothing
        pot, grad = run(targets, false, want_g_trg)
        vals.pottarg = pot
        want_g_trg && (vals.gradtarg = grad)
    end
    vals.ier = 0
    return vals
end

function l3ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipvecs=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    ns = size(sources, 2)
    nt = size(targets, 2)
    pot = zeros(Float64, nt)
    has_c = charges !== nothing
    has_d = dipvecs !== nothing
    @inbounds for j in 1:nt
        s = 0.0
        for i in 1:ns
            rx = targets[1, j] - sources[1, i]
            ry = targets[2, j] - sources[2, i]
            rz = targets[3, j] - sources[3, i]
            r2 = rx * rx + ry * ry + rz * rz
            r = sqrt(r2)
            r <= thresh && continue
            has_c && (s += charges[i] * INV4PI / r)
            has_d && (s += (dipvecs[1, i] * rx + dipvecs[2, i] * ry + dipvecs[3, i] * rz) * INV4PI / (r2 * r))
        end
        pot[j] = s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
