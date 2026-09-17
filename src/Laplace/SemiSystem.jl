# SLIPPY IterSemiSystem equivalent: staggered EHL on BEM operators only.
#
# Fluid:  ∇·((h³/μ) ∇p) = 12 ū ∂h/∂x     heterogeneous DIBEM (cached dM)
# Solid:  u = Kzz * p                     Pohrt–Li FFT on Reynolds interiors
# Gap:    h = max(h_min, h0 + u − δ)
# Load:   nudge δ until ∑ p ΔA = W
# Clip:   p ∈ [0, p_yield],  K=0 where h ≤ h_min  (SLIPPY mixed switch)
#
# Reference: Azam et al., Tribol. Int. 131:520–529 (2019); SLIPPY
# `IterSemiSystem` + `UnifiedReynoldsSolver`.

export roelands_viscosity, dowson_higginson
export mesh_ehl_square, interior_rect_map
export solve_semi_system!

"""Roelands η(p) (SLIPPY `nd_roelands`, dimensional)."""
function roelands_viscosity(p::Real; η0=0.096, p0=1 / 5.1e-9, z=0.68)
    pp = max(float(p), 0.0)
    return η0 * exp((log(η0) + 9.67) * (-1 + (1 + pp / p0)^z))
end

"""Dowson–Higginson ρ(p)/ρ0 (not used in the incompressible inner solve)."""
dowson_higginson(p::Real; C=5.9e8) = (C + 1.34 * max(float(p), 0.0)) / (C + max(float(p), 0.0))

"""Square pad `[−L/2, L/2]²` with transfinite sides (boundary only)."""
function mesh_ehl_square(; L=1.0, nside=17, ordem=1, nome="ehl_square")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    a = L / 2
    lc = L / 5
    p1 = gmsh.model.geo.addPoint(-a, -a, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(a, -a, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(a, a, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(-a, a, 0.0, lc)
    bot = gmsh.model.geo.addLine(p1, p2)
    rgt = gmsh.model.geo.addLine(p2, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    lft = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([bot, rgt, top, lft])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for c in (bot, rgt, top, lft)
        gmsh.model.mesh.setTransfiniteCurve(c, nside)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [bot, rgt, top, lft], -1, "0;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "pad")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

"""Map Reynolds interiors to a uniform `nix × niy` lattice for FFT `Kzz`."""
function interior_rect_map(dad::BEMdata)
    n = dad.n
    ni = dad.ni
    ni >= 4 || throw(ArgumentError("need interior collocation for the contact grid"))
    xs = Float64[]; ys = Float64[]
    @inbounds for k in 1:ni
        p = point(dad, n + k)
        push!(xs, p[1]); push!(ys, p[2])
    end
    ux = sort(unique(round.(xs; digits=12)))
    uy = sort(unique(round.(ys; digits=12)))
    nix, niy = length(ux), length(uy)
    nix * niy == ni || throw(ArgumentError(
        "interiors are not a tensor product ($(ni) points, $(nix)×$(niy) unique)"))
    xmap = Dict(ux[i] => i for i in 1:nix)
    ymap = Dict(uy[j] => j for j in 1:niy)
    idx = zeros(Int, nix, niy)
    @inbounds for k in 1:ni
        i = xmap[round(xs[k]; digits=12)]
        j = ymap[round(ys[k]; digits=12)]
        idx[i, j] = n + k
    end
    hx = nix > 1 ? ux[2] - ux[1] : 1.0
    hy = niy > 1 ? uy[2] - uy[1] : 1.0
    return (; xs=ux, ys=uy, nix, niy, idx, hx, hy)
end

function _scatter_u!(u::AbstractVector, um::AbstractMatrix, gmap, dad)
    fill!(u, 0.0)
    @inbounds for j in 1:gmap.niy, i in 1:gmap.nix
        u[gmap.idx[i, j]] = um[i, j]
    end
    @inbounds for b in 1:dad.n
        pt = point(dad, b)
        i = clamp(searchsortedfirst(gmap.xs, pt[1]), 1, gmap.nix)
        j = clamp(searchsortedfirst(gmap.ys, pt[2]), 1, gmap.niy)
        if i > 1 && abs(gmap.xs[i - 1] - pt[1]) < abs(gmap.xs[i] - pt[1])
            i -= 1
        end
        if j > 1 && abs(gmap.ys[j - 1] - pt[2]) < abs(gmap.ys[j] - pt[2])
            j -= 1
        end
        u[b] = um[i, j]
    end
    return u
end

function _gather_p!(pmat::AbstractMatrix, p::AbstractVector, gmap)
    @inbounds for j in 1:gmap.niy, i in 1:gmap.nix
        pmat[i, j] = p[gmap.idx[i, j]]
    end
    return pmat
end

"""`∂h/∂x` by central differences on constant-`y` bins."""
function _d_h_dx(pts, h)
    n = length(pts)
    dh = zeros(n)
    bins = Dict{Float64,Vector{Int}}()
    @inbounds for i in 1:n
        push!(get!(bins, round(pts[i][2]; digits=8), Int[]), i)
    end
    for idxs in values(bins)
        perm = sort(idxs; by=i -> pts[i][1])
        m = length(perm)
        m < 2 && continue
        dh[perm[1]] = (h[perm[2]] - h[perm[1]]) / (pts[perm[2]][1] - pts[perm[1]][1] + 1e-30)
        dh[perm[m]] = (h[perm[m]] - h[perm[m - 1]]) / (pts[perm[m]][1] - pts[perm[m - 1]][1] + 1e-30)
        @inbounds for k in 2:(m - 1)
            dh[perm[k]] = (h[perm[k + 1]] - h[perm[k - 1]]) /
                (pts[perm[k + 1]][1] - pts[perm[k - 1]][1] + 1e-30)
        end
    end
    return dh
end

"""
    solve_semi_system!(dad, gmap, hs, prep; h0, W, ū, η0, R, kwargs...)

Staggered EHL: DIBEM Reynolds + FFT half-space on `gmap` interiors.
`h0` is the undeformed gap (just-touching). Returns a named tuple.
"""
function solve_semi_system!(dad::BEMdata{<:Laplace}, gmap, hs, prep;
        h0::AbstractVector,
        W::Real,
        ū::Real,
        η0::Real=0.096,
        R::Real=0.01905,
        p_fft_scale::Real=1.0,
        u_scale::Real=1.0,
        δ0::Real=0.0,
        p0::Union{Nothing,AbstractVector}=nothing,
        μfun=nothing,
        h_min::Real=0.47e-9,
        h_kfloor::Real=1e-4,
        μ_cap::Real=1e3,
        p_yield::Real=Inf,
        ωp::Real=0.05,
        ωδ::Real=0.05,
        maxiter::Int=400,
        rtol_p::Real=2e-4,
        rtol_W::Real=1e-3,
        rbf=PHS(1; poly_deg=-1),
        source_rbf=PHS(3; poly_deg=1),
        npg::Int=8,
        verbose::Bool=true)
    n = dad.n
    nt = dad.nt
    length(h0) == nt || throw(DimensionMismatch("h0"))
    apply_ambient_pressure!(dad)
    has_cache(dad, :H) || assemble!(dad; npg=npg, threaded=false)
    has_cache(dad, :M) || DIBEM(dad; rbf=source_rbf)
    pts = all_points(dad)
    p = p0 === nothing ? zeros(nt) : copy(p0)
    u = zeros(nt)
    h = copy(h0)
    pmat = zeros(gmap.nix, gmap.niy)
    δ = float(δ0)
    ΔA = gmap.hx * gmap.hy
    δ0 = float(δ0)
    δlo, δhi = -2.0 * max(abs(δ0), 1.0), 4 * max(abs(δ0), 1.0)
    μfun_ = μfun === nothing ? (pp -> roelands_viscosity(pp; η0=η0)) : μfun
    hist = NamedTuple[]
    p_prev = copy(p)
    um = zeros(gmap.nix, gmap.niy)
    _gather_p!(pmat, p, gmap)
    pmat .*= p_fft_scale
    Contact.fc_forward!(um, pmat, Contact.Kzz, prep)
    um .*= u_scale
    _scatter_u!(u, um, gmap, dad)

    local load = 0.0
    local er_p = 1.0
    local er_W = 1.0
    it_done = 0
    for it in 1:maxiter
        @inbounds for i in 1:nt
            h[i] = h0[i] + u[i] - δ
            h[i] < h_min && (h[i] = h_min)
        end
        Kv = zeros(nt)
        @inbounds for i in 1:nt
            μi = clamp(μfun_(p[i]), η0, μ_cap * η0)
            hi = max(h[i], h_kfloor)
            Kv[i] = hi^3 / μi
        end
        L, _, _ = heterogeneous_L(dad, Kv; rbf=rbf)
        Asys, b0 = _het_mixed_from_L(dad, L, Kv)
        @inbounds for i in 1:nt
            aii = abs(Asys[i, i])
            aii > 0 && (Asys[i, i] += 1e-14 * aii)
        end
        dh = _d_h_dx(pts, h)
        f = (12 * ū) .* dh
        rhs = b0 + dad.M * f
        F = lu(Asys, check=false)
        ok = LinearAlgebra.issuccess(F) && all(isfinite, rhs)
        x = ok ? (F \ rhs) : p_prev
        ok = ok && all(isfinite, x)
        if it <= 2 && verbose
            pint = ok ? maximum(abs, @view x[(n + 1):nt]) : NaN
            println("  it", it, "  max|p_int|=", pint, "  hmin=", minimum(h), "  δ=", δ)
        end
        if !ok
            verbose && it <= 3 && println("  EHL it=", it, "  singular Reynolds — keep p")
            copyto!(p, p_prev)
        else
            p_new = _split_het_sol!(dad, x, Kv)
            ok = all(isfinite, p_new)
            if ok
                @inbounds for i in 1:nt
                    p_new[i] = clamp(p_new[i], 0.0, p_yield)
                    p[i] = p_prev[i] + ωp * (p_new[i] - p_prev[i])
                    p[i] = clamp(p[i], 0.0, p_yield)
                end
            else
                copyto!(p, p_prev)
            end
        end
        _gather_p!(pmat, p, gmap)
        load = 0.0
        @inbounds for j in 1:gmap.niy, i in 1:gmap.nix
            load += pmat[i, j]
        end
        load *= ΔA
        pmat .*= p_fft_scale
        Contact.fc_forward!(um, pmat, Contact.Kzz, prep)
        um .*= u_scale
        _scatter_u!(u, um, gmap, dad)
        pmat ./= p_fft_scale  # restore nd pressure for output / next gather
        er_W = load / W - 1
        num = 0.0
        den = 0.0
        @inbounds for i in 1:nt
            num += abs(p[i] - p_prev[i])
            den += abs(p_prev[i])
        end
        er_p = num / max(den, 1e-30)
        copyto!(p_prev, p)
        if ok
            δ = clamp(δ - ωδ * er_W * δ0, δlo, δhi)
        end
        it_done = it
        verbose && it % 10 == 0 && println("  EHL it=", it, "  W=", load, "  er_W=", er_W,
            "  er_p=", er_p, "  pmax=", maximum(p), "  hmin=", minimum(h), "  δ=", δ)
        push!(hist, (; it, load, er_W, er_p, pmax=maximum(p), hmin=minimum(h), δ))
        if it >= 8 && abs(er_W) < rtol_W && er_p < rtol_p
            verbose && println("  EHL converged at it=", it)
            break
        end
    end
    set_cache!(dad; T=p, q=dad.q, ehl_h=h, ehl_u=u, ehl_δ=δ)
    return (; p, h, u, δ, load, er_W, er_p, iters=it_done,
        pmax=maximum(p), hmin=minimum(h), hist, pmat=copy(pmat), um=copy(um), gmap)
end
