# =============================================================================
# 3D Laplace FMM — kernel 1/(4π r)
#
# Spherical multipoles + full dual-tree FMM:
#   P2M → M2M → M2L → L2L → L2P (+ P2P near field)
#
# Translations: spherical-harmonic P2M/M2M/M2L/L2L/L2P on a cubic octree
# (same `DyadicSplitter(tight=false); cube=true` tree as H²). Lists follow
# Flatiron FMM3D (same-level IL + adaptive W-list).
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

# Precomputed (n-m)!/(n+m)! and sqrt of that (FMM3D ylgndru scale).
const _FR_CACHE = Dict{Int,Matrix{Float64}}()
const _SNM_CACHE = Dict{Int,Matrix{Float64}}()

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

function _sph_norm_table(p::Int)
    return get!(_SNM_CACHE, p) do
        FR = _fact_ratio_table(p)
        SNM = ones(Float64, p + 1, p + 1)
        @inbounds for n in 0:p, m in 0:n
            SNM[n + 1, m + 1] = sqrt(FR[n + 1, m + 1])
        end
        SNM
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

"""FMM3D `ylgndrini` recurrence weights for `ylgndruf` (no extra √(2n+1))."""
struct _YlgndrRec
    rat1::Matrix{Float64}
    rat2::Matrix{Float64}
end
const _YLGNDR_CACHE = Dict{Int,_YlgndrRec}()
function _ylgndr_rec(p::Int)
    return get!(_YLGNDR_CACHE, p) do
        rat1 = zeros(Float64, p + 1, p + 1)
        rat2 = ones(Float64, p + 1, p + 1)
        rat1[1, 1] = 1.0
        @inbounds for m in 0:p
            if m > 0
                rat1[m + 1, m + 1] = sqrt((2m - 1) / (2m))
            end
            if m < p
                rat1[m + 2, m + 1] = sqrt(2m + 1)
            end
            for n in (m + 2):p
                den = sqrt(Float64(n - m) * Float64(n + m))
                rat1[n + 1, m + 1] = (2n - 1) / den
                rat2[n + 1, m + 1] = sqrt((n + m - 1.0) * (n - m - 1.0)) / den
            end
        end
        _YlgndrRec(rat1, rat2)
    end
end

"""FMM3D `ylgndruf`: Y[n+1,m+1] = sqrt((n-m)!/(n+m)!) P_n^m(x)."""
function _legendre_table!(P::Matrix{Float64}, p::Int, x::Float64)
    P[1, 1] = 1.0
    p == 0 && return P
    rec = _ylgndr_rec(p)
    rat1, rat2 = rec.rat1, rec.rat2
    u = -sqrt(max(0.0, (1 - x) * (1 + x)))
    @inbounds for m in 0:p
        if m > 0
            P[m + 1, m + 1] = P[m, m] * u * rat1[m + 1, m + 1]
        end
        if m < p
            P[m + 2, m + 1] = x * P[m + 1, m + 1] * rat1[m + 2, m + 1]
        end
        for n in (m + 2):p
            P[n + 1, m + 1] = rat1[n + 1, m + 1] * x * P[n, m + 1] - rat2[n + 1, m + 1] * P[n - 1, m + 1]
        end
    end
    return P
end

function laplace3d_nterms(eps::Real; full_fmm::Bool=true)
    return min(max(laplace3d_nterms_flatiron(Float64(eps)), 4), 16)
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
    Ptab::Union{Nothing,Matrix{Float64}}=nothing,
)
    reset && fill!(mpole, 0.0)
    if Ptab === nothing
        Ptab = zeros(Float64, p + 1, p + 1)
    end
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
                c = q * rsn * Ptab[n + 1, m + 1]
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
                            c = w * rsn * Ptab[n + 1, m + 1]
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
    localexp::AbstractVector{Float64},
    center::SVector{3,Float64},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    irange,
    p::Int;
    reset::Bool=true,
    Ptab::Union{Nothing,Matrix{Float64}}=nothing,
)
    reset && fill!(localexp, 0.0)
    if Ptab === nothing
        Ptab = zeros(Float64, p + 1, p + 1)
    end
    @inbounds for i in irange
        d = sources[i] - center
        x, y, z = d[1], d[2], d[3]
        ρ = sqrt(x * x + y * y + z * z)
        ρ < 1e-30 && continue
        ct = clamp(z / ρ, -1.0, 1.0)
        φs = atan(y, x)
        q = Float64(charges[i])
        _legendre_table!(Ptab, p, ct)
        c1, s1 = cos(φs), sin(φs)
        rpow = 1 / ρ
        for n in 0:p
            off = _mp_offset(n)
            localexp[off + 1] += q * rpow * Ptab[n + 1, 1]
            cm, sm = c1, s1
            for m in 1:n
                c = q * rpow * Ptab[n + 1, m + 1]
                localexp[off + 2m] += c * cm
                localexp[off + 2m + 1] += c * (-sm)
                cm2 = cm * c1 - sm * s1
                sm2 = sm * c1 + cm * s1
                cm, sm = cm2, sm2
            end
            rpow /= ρ
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

function _local_series(d::SVector{3,Float64}, localexp::Vector{Float64}, p::Int,
                       Ptab::Matrix{Float64}=zeros(p + 1, p + 1))
    x, y, z = d[1], d[2], d[3]
    r = sqrt(x * x + y * y + z * z)
    # ρ=0: only L₀⁰ survives (P_n^0(1)=1, P_n^m(1)=0 for m>0, r^n=0 for n>0).
    r < 1e-30 && return localexp[1]
    ct = clamp(z / r, -1.0, 1.0)
    φ = atan(y, x)
    _legendre_table!(Ptab, p, ct)
    s = 0.0
    rn = 1.0
    c1, s1 = cos(φ), sin(φ)
    @inbounds for n in 0:p
        off = _mp_offset(n)
        s += localexp[off + 1] * Ptab[n + 1, 1] * rn
        cm, sm = c1, s1
        for m in 1:n
            Lr, Li = localexp[off + 2m], localexp[off + 2m + 1]
            s += 2 * (Lr * cm - Li * sm) * Ptab[n + 1, m + 1] * rn
            cm2 = cm * c1 - sm * s1
            sm2 = sm * c1 + cm * s1
            cm, sm = cm2, sm2
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
    B1::Matrix{Float64}      # local-basis samples on unit sphere (L2L)
    F_B::Factorization{Float64}
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
        B1 = zeros(Float64, K, K)
        basis = zeros(Float64, K)
        @inbounds for i in 1:K
            fill!(basis, 0)
            basis[i] = 1.0
            for j in 1:K
                B1[j, i] = _local_series(unit_pts[j], basis, p)
            end
        end
        F_B = lu(B1 + 1e-12 * I)
        SphericalTranslateCache(p, unit_pts, A, deg, F, B1, F_B)
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
    return _m2l_laplace3d!(localexp, tctr, mpole, sctr, sR, p, Laplace3DTransWS(p))
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

function _allocate_l3(root::ClusterTree{3}, p::Int; gumerov::Bool=false)
    nc = _ncoeff_sph(p)
    if node_id(root) == 0
        assign_node_ids!(root)
    end
    nn = nnodes(root)
    data = Vector{L3Exp}(undef, nn)
    for node in nodes(root)
        ctr = SVector{3,Float64}(center(container(node)))
        R = max(diameter(node) / 2, 1e-30)
        data[node_id(node)] = L3Exp(
            zeros(Float64, nc), zeros(Float64, nc),
            ctr, R,
        )
    end
    return data
end

_l3get(data::AbstractVector, node) = data[node_id(node)]
_l3get(data::AbstractDict, node) = data[objectid(node)]

# ---------- public API ----------

"""
```julia
vals = lfmm3d(eps, sources; charges=nothing, dipvecs=nothing,
              targets=nothing, pg=0, pgt=0, nmax=-1, η=1.0,
              threaded=false)
```

3D Laplace FMM for ``1/(4\\pi r)``. Charge-only source→source (`pg=1` or `2`)
uses a cubic octree and spherical-harmonic FMM via [`build_laplace3d_plan`](@ref).
`threaded=true`: parallel interaction list on the dipole/target fallback.
"""
function lfmm3d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipvecs=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=-1,
    η::Real=1.0,
    threaded::Bool=false,
    full_fmm::Bool=true,
    plan=nothing,   # optional cache from build_laplace3d_plan
)
    @assert size(sources, 1) == 3
    ns = size(sources, 2)
    has_c = charges !== nothing
    has_d = dipvecs !== nothing
    @assert has_c || has_d
    @assert pg > 0 || pgt > 0

    # octree spherical-harmonic apply (potential, optionally gradient)
    if targets === nothing && !has_d && has_c && Int(pgt) == 0 && Int(pg) in (1, 2)
        if !(plan isa Laplace3DFMMPlan)
            plan = build_laplace3d_plan(sources; eps=Float64(eps), nmax=Int(nmax),
                η=Float64(η))
        end
        pot = Vector{Float64}(undef, plan.n)
        vals = FMMVals()
        if Int(pg) == 2
            g = zeros(Float64, 3, plan.n)
            apply_laplace3d!(plan, pot; charges=charges, grad=g)
            vals.grad = g
        else
            apply_laplace3d!(plan, pot; charges=charges)
        end
        vals.pot = pot
        vals.ier = 0
        return vals
    end

    p = laplace3d_nterms(eps)
    ncoeff = _ncoeff_sph(p)
    adm = StrongAdmissibility(η=Float64(η))
    nmax_i = Int(nmax) < 0 ? laplace3d_ndiv(eps) : Int(nmax)
    if plan !== nothing
        stree, sl2g, src, sdata = plan.stree, plan.sl2g, plan.src, plan.sdata
        for ed in sdata
            fill!(ed.multipole, 0)
            fill!(ed.localexp, 0)
        end
    else
        spl = DyadicSplitter(nmax=nmax_i)
        stree, sl2g, src = build_point_tree(sources, spl)
        sdata = _allocate_l3(stree, p)
    end

    ch = has_c ? permute_to_local(vec(charges), sl2g) : zeros(Float64, ns)
    dips = nothing
    if has_d
        dips = Vector{SVector{3,Float64}}(undef, ns)
        @inbounds for i in 1:ns
            g = sl2g[i]
            dips[i] = SVector{3,Float64}(dipvecs[1, g], dipvecs[2, g], dipvecs[3, g])
        end
    end

    full_fmm && get_spherical_cache(p)
    _ylgndr_rec(p)

    function upward!(node)
        ed = _l3get(sdata, node)
        if isleaf(node)
            form_mpole3d!(ed.multipole, ed.center, src, ch, dips, index_range(node), p; reset=true)
        else
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
                cd = _l3get(sdata, c)
                m2m_laplace3d!(ed.multipole, ed.center, cd.multipole, cd.center, cd.R, p)
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

        function m2l_job!(tnode, snode, _, __)
            td = _l3get(tdata, tnode)
            sd = _l3get(sdata, snode)
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

        function downward!(node)
            ed = _l3get(tdata, node)
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
                    cd = _l3get(tdata, c)
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

"""Scratch for 3D equivalent-charge M2M / M2L / L2L."""
mutable struct Laplace3DTransWS
    b::Vector{Float64}
    q::Vector{Float64}
    pts::Vector{SVector{3,Float64}}
    pots::Vector{Float64}
    B::Matrix{Float64}
    basis::Vector{Float64}
    Ptab::Matrix{Float64}
end
function Laplace3DTransWS(p::Int)
    K = _ncoeff_sph(p)
    return Laplace3DTransWS(
        zeros(Float64, K), zeros(Float64, K),
        Vector{SVector{3,Float64}}(undef, K),
        zeros(Float64, K), zeros(Float64, K, K), zeros(Float64, K),
        zeros(Float64, p + 1, p + 1),
    )
end

function _equiv_charges!(pts, q, b, mpole, R::Float64, p::Int)
    cache = get_spherical_cache(p)
    K = length(mpole)
    R = max(R, 1e-12)
    @inbounds for i in 1:K
        b[i] = mpole[i] / (R^cache.deg[i])
    end
    copyto!(q, b)
    ldiv!(cache.F, q)
    @inbounds for j in 1:K
        u = cache.unit_pts[j]
        pts[j] = SVector(R * u[1], R * u[2], R * u[3])
    end
    return pts, q
end

"""In-place equivalent-sphere M2L into `localexp` (incremental)."""
function _m2l_laplace3d!(
    localexp::AbstractVector{Float64},
    tctr::SVector{3,Float64},
    mpole::Vector{Float64},
    sctr::SVector{3,Float64},
    sR::Float64,
    p::Int,
    ws::Laplace3DTransWS,
)
    R = max(sR * 1.05, 1e-12)
    _equiv_charges!(ws.pts, ws.q, ws.b, mpole, R, p)
    @inbounds for j in eachindex(ws.pts)
        ws.pts[j] = sctr + ws.pts[j]
    end
    form_local3d!(localexp, tctr, ws.pts, ws.q, eachindex(ws.pts), p;
        reset=false, Ptab=ws.Ptab)
    return localexp
end

function _upward3d!(node, sdata, src, ch, p, Ptab, ws; full_fmm::Bool=false)
    ed = _l3get(sdata, node)
    if isleaf(node)
        form_mpole3d!(ed.multipole, ed.center, src, ch, nothing, index_range(node), p;
            reset=true, Ptab=Ptab)
    else
        for c in children(node)
            _upward3d!(c, sdata, src, ch, p, Ptab, ws; full_fmm=full_fmm)
        end
        if full_fmm
            fill!(ed.multipole, 0)
            for c in children(node)
                cd = _l3get(sdata, c)
                R = max(cd.R * 1.05, 1e-12)
                _equiv_charges!(ws.pts, ws.q, ws.b, cd.multipole, R, p)
                @inbounds for j in eachindex(ws.pts)
                    ws.pts[j] = cd.center + ws.pts[j]
                end
                form_mpole3d!(ed.multipole, ed.center, ws.pts, ws.q, nothing,
                    eachindex(ws.pts), p; reset=false, Ptab=Ptab)
            end
        end
    end
    return nothing
end

function _p2p_laplace3d!(pot, tpts, src, ch, tnode, snode, grad=nothing)
    irj = index_range(tnode)
    iri = index_range(snode)
    skip_self = tnode === snode
    if grad === nothing && !skip_self
        @inbounds for j in irj
            tj = tpts[j]
            s = 0.0
            for i in iri
                si = src[i]
                rx = tj[1] - si[1]
                ry = tj[2] - si[2]
                rz = tj[3] - si[3]
                r2 = rx * rx + ry * ry + rz * rz
                r2 < 1e-30 && continue
                s += ch[i] * INV4PI / sqrt(r2)
            end
            pot[j] += s
        end
        return nothing
    end
    @inbounds for j in irj
        tj = tpts[j]
        s = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for i in iri
            skip_self && i == j && continue
            si = src[i]
            rx = tj[1] - si[1]
            ry = tj[2] - si[2]
            rz = tj[3] - si[3]
            r2 = rx * rx + ry * ry + rz * rz
            r2 < 1e-30 && continue
            invr = 1 / sqrt(r2)
            qi = ch[i] * INV4PI
            s += qi * invr
            if grad !== nothing
                c = -qi * invr / r2
                gx += c * rx
                gy += c * ry
                gz += c * rz
            end
        end
        pot[j] += s
        if grad !== nothing
            grad[1, j] += gx
            grad[2, j] += gy
            grad[3, j] += gz
        end
    end
    return nothing
end

# Source tile fits in L1 with 4 target coords + charges (FMM3D/SCTL-style).
const _P2P_STILE = 64

@inline _rsqrt(r2::Float64) = @fastmath 1 / sqrt(r2)

@inline function _r2eps(tx, ty, tz, sx, sy, sz)
    rx = tx - sx
    ry = ty - sy
    rz = tz - sz
    return muladd(rz, rz, muladd(ry, ry, muladd(rx, rx, 1e-300)))
end

"""Accumulate `Σ q/r` for targets `jlo:jhi` from sources `ilo:ihi` (no self)."""
function _p2p_invr_block!(pot, xs, ys, zs, ch, jlo::Int, jhi::Int, ilo::Int, ihi::Int)
    ilo > ihi && return nothing
    jlo > jhi && return nothing
    j = jlo
    @inbounds while j + 3 <= jhi
        tx0, ty0, tz0 = xs[j], ys[j], zs[j]
        tx1, ty1, tz1 = xs[j + 1], ys[j + 1], zs[j + 1]
        tx2, ty2, tz2 = xs[j + 2], ys[j + 2], zs[j + 2]
        tx3, ty3, tz3 = xs[j + 3], ys[j + 3], zs[j + 3]
        s0 = s1 = s2 = s3 = 0.0
        ib = ilo
        while ib + _P2P_STILE - 1 <= ihi
            lim = ib + _P2P_STILE - 1
            @fastmath for i in ib:lim
                sx = xs[i]
                sy = ys[i]
                sz = zs[i]
                q = ch[i]
                s0 += q * _rsqrt(_r2eps(tx0, ty0, tz0, sx, sy, sz))
                s1 += q * _rsqrt(_r2eps(tx1, ty1, tz1, sx, sy, sz))
                s2 += q * _rsqrt(_r2eps(tx2, ty2, tz2, sx, sy, sz))
                s3 += q * _rsqrt(_r2eps(tx3, ty3, tz3, sx, sy, sz))
            end
            ib += _P2P_STILE
        end
        if ib <= ihi
            @fastmath for i in ib:ihi
                sx = xs[i]
                sy = ys[i]
                sz = zs[i]
                q = ch[i]
                s0 += q * _rsqrt(_r2eps(tx0, ty0, tz0, sx, sy, sz))
                s1 += q * _rsqrt(_r2eps(tx1, ty1, tz1, sx, sy, sz))
                s2 += q * _rsqrt(_r2eps(tx2, ty2, tz2, sx, sy, sz))
                s3 += q * _rsqrt(_r2eps(tx3, ty3, tz3, sx, sy, sz))
            end
        end
        pot[j] = muladd(INV4PI, s0, pot[j])
        pot[j + 1] = muladd(INV4PI, s1, pot[j + 1])
        pot[j + 2] = muladd(INV4PI, s2, pot[j + 2])
        pot[j + 3] = muladd(INV4PI, s3, pot[j + 3])
        j += 4
    end
    while j <= jhi
        tx, ty, tz = xs[j], ys[j], zs[j]
        s = 0.0
        ib = ilo
        while ib + _P2P_STILE - 1 <= ihi
            lim = ib + _P2P_STILE - 1
            @fastmath @simd ivdep for i in ib:lim
                s += ch[i] * _rsqrt(_r2eps(tx, ty, tz, xs[i], ys[i], zs[i]))
            end
            ib += _P2P_STILE
        end
        if ib <= ihi
            @fastmath @simd ivdep for i in ib:ihi
                s += ch[i] * _rsqrt(_r2eps(tx, ty, tz, xs[i], ys[i], zs[i]))
            end
        end
        pot[j] = muladd(INV4PI, s, pot[j])
        j += 1
    end
    return nothing
end

"""Charge-only 3D P2P on SoA coords (Flatiron `l3d_directcp`: skip-self only on the same leaf)."""
function _p2p_laplace3d_soa!(pot, xs, ys, zs, ch, tnode, snode, grad=nothing)
    irj = index_range(tnode)
    iri = index_range(snode)
    skip_self = tnode === snode
    jlo, jhi = Int(first(irj)), Int(last(irj))
    ilo, ihi = Int(first(iri)), Int(last(iri))
    if grad === nothing
        if !skip_self
            _p2p_invr_block!(pot, xs, ys, zs, ch, jlo, jhi, ilo, ihi)
            return nothing
        end
        # Same leaf: 4-target tiles with a scalar diagonal block so i=j is skipped
        # without a per-pair branch in the long SIMD ranges.
        j = jlo
        @inbounds while j + 3 <= jhi
            if ilo < j
                _p2p_invr_block!(pot, xs, ys, zs, ch, j, j + 3, ilo, j - 1)
            end
            if j + 4 <= ihi
                _p2p_invr_block!(pot, xs, ys, zs, ch, j, j + 3, j + 4, ihi)
            end
            for jj in j:(j + 3)
                tx, ty, tz = xs[jj], ys[jj], zs[jj]
                s = 0.0
                @fastmath for i in j:(j + 3)
                    i == jj && continue
                    s += ch[i] * _rsqrt(_r2eps(tx, ty, tz, xs[i], ys[i], zs[i]))
                end
                pot[jj] = muladd(INV4PI, s, pot[jj])
            end
            j += 4
        end
        @inbounds while j <= jhi
            if ilo <= j - 1
                _p2p_invr_block!(pot, xs, ys, zs, ch, j, j, ilo, j - 1)
            end
            if j + 1 <= ihi
                _p2p_invr_block!(pot, xs, ys, zs, ch, j, j, j + 1, ihi)
            end
            j += 1
        end
        return nothing
    end
    @inbounds for j in jlo:jhi
        tx, ty, tz = xs[j], ys[j], zs[j]
        s = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for i in ilo:ihi
            skip_self && i == j && continue
            rx = tx - xs[i]
            ry = ty - ys[i]
            rz = tz - zs[i]
            r2 = muladd(rz, rz, muladd(ry, ry, rx * rx))
            r2 < 1e-30 && continue
            invr = _rsqrt(r2)
            qi = ch[i] * INV4PI
            s += qi * invr
            c = -qi * invr / r2
            gx += c * rx
            gy += c * ry
            gz += c * rz
        end
        pot[j] += s
        grad[1, j] += gx
        grad[2, j] += gy
        grad[3, j] += gz
    end
    return nothing
end

function _l2l_laplace3d!(child_local, child_ctr, parent_local, parent_ctr, child_R, p, ws)
    cache = get_spherical_cache(p)
    K = _ncoeff_sph(p)
    R = max(child_R * 0.5, 1e-12)
    @inbounds for j in 1:K
        u = cache.unit_pts[j]
        pt = child_ctr + SVector(R * u[1], R * u[2], R * u[3])
        ws.pots[j] = _local_series(pt - parent_ctr, parent_local, p, ws.Ptab)
    end
    copyto!(ws.q, ws.pots)
    ldiv!(cache.F_B, ws.q)
    @inbounds for i in 1:K
        child_local[i] += ws.q[i] / (R^cache.deg[i])
    end
    return child_local
end

function _downward3d!(node, sdata, src, pot, p, ws)
    ed = _l3get(sdata, node)
    if isleaf(node)
        @inbounds for j in index_range(node)
            pot[j] += INV4PI * _local_series(src[j] - ed.center, ed.localexp, p, ws.Ptab)
        end
    else
        for c in children(node)
            cd = _l3get(sdata, c)
            _l2l_laplace3d!(cd.localexp, cd.center, ed.localexp, ed.center, cd.R, p, ws)
            _downward3d!(c, sdata, src, pot, p, ws)
        end
    end
    return nothing
end

include("planewave.jl")
include("octree_fmm.jl")
