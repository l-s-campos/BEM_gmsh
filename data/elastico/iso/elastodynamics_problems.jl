# 2-D BEM models for the analytical elastodynamics suite + sudden bar.
# Internals = Gmsh cell centroids (`pontointerno=true`).
#
# include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
# include(datadir("elastico", "iso", "bar_sudden.jl"))
# include(this file)

const ELASTO_MESH_NDIV = Dict(200 => 13, 500 => 21, 1000 => 31)

# Short beam: L/h = 4 (Timoshenko shear is first-order). Plane-stress thickness b = 1.
const SHORT_BEAM = (L=1.0, h=0.25, b=1.0, E=1.0, ρ=1.0, ν=0.3, F0=1.0)

function _elasto_period_eb(; L=1.0, EI=1.0, ρA=1.0)
    ω1 = (π / L)^2 * sqrt(EI / ρA)
    return 2π / ω1
end

function _elasto_period_bar(; L=1.0, c=1.0)
    return 4L / c
end

function _elasto_period_hole(; a=1.0, E=1.0, ν=0.3, ρ=1.0, plane_stress=true)
    cp = plane_stress ? sqrt(E / (ρ * (1 - ν^2))) : sqrt((E * (1 - ν)) / (ρ * (1 + ν) * (1 - 2ν)))
    return 2π * a / cp
end

"""Nearest collocation to `probe`; return (index, point)."""
function _elasto_probe_id(dad, probe)
    pts = all_points(dad)
    i = argmin(norm(p - probe) for p in pts)
    return i, pts[i]
end

function _elasto_hist(U, dad, ip; comp=1)
    return U[2 * (ip - 1) + comp, :]
end

function _nodal_lengths(dad)
    n = dad.n
    lens = zeros(n)
    for el in dad.elements
        for i in el.index
            i <= n || continue
            lens[i] += el.Length / length(el.index)
        end
    end
    return lens
end

function _coord_mask(dad, dim, val)
    n = dad.n
    mask = falses(n)
    @inbounds for i in 1:n
        abs(dad.Nodes[i][dim] - val) < 1e-8 * (1 + abs(val)) && (mask[i] = true)
    end
    return mask
end

function _top_mask(dad)
    ymax = maximum(p[2] for p in dad.Nodes)
    return _coord_mask(dad, 2, ymax)
end

function _bottom_mask(dad)
    ymin = minimum(p[2] for p in dad.Nodes)
    return _coord_mask(dad, 2, ymin)
end

"""Pin at (0,0): ``u_x=u_y=0``. Roller at (L,0): ``u_y=0``. Ends can rotate."""
function _apply_ss_supports!(dad, L)
    n = dad.n
    iL = argmin(norm(dad.Nodes[i] - Point2D(0.0, 0.0)) for i in 1:n)
    iR = argmin(norm(dad.Nodes[i] - Point2D(L, 0.0)) for i in 1:n)
    dad.BC[2iL-1] = 0; dad.BV[2iL-1] = 0.0
    dad.BC[2iL]   = 0; dad.BV[2iL]   = 0.0
    dad.BC[2iR-1] = 1; dad.BV[2iR-1] = 0.0
    dad.BC[2iR]   = 0; dad.BV[2iR]   = 0.0
    return iL, iR
end

function _apply_top_patch_ty!(dad, F0, xmid, halfw)
    lens = _nodal_lengths(dad)
    mask = _top_mask(dad)
    @inbounds for i in eachindex(mask)
        mask[i] || continue
        abs(dad.Nodes[i][1] - xmid) <= halfw + 1e-12 || (mask[i] = false)
    end
    Lpatch = sum(lens[mask])
    Lpatch <= 0 && error("top traction patch empty")
    ty = -F0 / Lpatch
    @inbounds for i in eachindex(mask)
        mask[i] || continue
        dad.BC[2i] = 1
        dad.BV[2i] = ty
    end
    return ty, Lpatch
end

"""Top-edge traction. `:point` mid-span patch, `:uniform` ``q=F0/L``, `:sine` ``q0=F0 π/(2L)``."""
function _apply_top_load!(dad, load::Symbol, F0, L; halfw=nothing)
    load === :point && return _apply_top_patch_ty!(dad, F0, L / 2, something(halfw, L / 20))
    lens = _nodal_lengths(dad)
    mask = _top_mask(dad)
    if load === :uniform
        q = F0 / L
        @inbounds for i in eachindex(mask)
            mask[i] || continue
            dad.BC[2i] = 1
            dad.BV[2i] = -q
        end
        return -q, sum(lens[mask])
    elseif load === :sine
        q0 = F0 * π / (2L)
        @inbounds for i in eachindex(mask)
            mask[i] || continue
            dad.BC[2i] = 1
            dad.BV[2i] = -q0 * sin(π * dad.Nodes[i][1] / L)
        end
        return -q0, sum(lens[mask])
    else
        throw(ArgumentError("load must be :point, :uniform, or :sine"))
    end
end

function _timoshenko_kn_T1(kn, b, h, E, ν, ρ)
    A = b * h
    I = b * h^3 / 12
    G = E / (2 * (1 + ν))
    κ = 5 / 6
    μA, μI = ρ * A, ρ * I
    α = (κ * G * A) / μI + (E * I) / μI * kn^2 + (κ * G * A) / μA * kn^2
    βp = (κ * G * A * E * I) / (μA * μI) * kn^4
    ω1 = sqrt((α - sqrt(max(α^2 - 4 * βp, 0.0))) / 2)
    return 2π / ω1
end

_timoshenko_ss_T1(L, b, h, E, ν, ρ) = _timoshenko_kn_T1(π / L, b, h, E, ν, ρ)
_timoshenko_cantilever_T1(L, b, h, E, ν, ρ) =
    _timoshenko_kn_T1(1.875104068 / L, b, h, E, ν, ρ)

"""1-D static probe deflection and fundamental period. Amplitude target is ``2 δ``."""
function beam_expect(kind::Symbol, load::Symbol; L, h, b, E, ν, ρ, F0)
    I = b * h^3 / 12
    A = b * h
    EI, ρA = E * I, ρ * A
    G = E / (2 * (1 + max(ν, 0.0)))
    κAG = (5 / 6) * G * A
    shear = kind === :eb_ss || kind === :eb_cantilever ? 0.0 : 1.0
    if kind === :eb_ss || kind === :timoshenko_ss
        T = kind === :eb_ss ? _elasto_period_eb(; L=L, EI=EI, ρA=ρA) :
            _timoshenko_ss_T1(L, b, h, E, ν, ρ)
        if load === :point
            δ = F0 * L^3 / (48 * EI) + shear * F0 * L / (4 * κAG)
        elseif load === :uniform
            q = F0 / L
            δ = 5 * q * L^4 / (384 * EI) + shear * q * L^2 / (8 * κAG)
        else
            q0 = F0 * π / (2L)
            δ = q0 * L^4 / (π^4 * EI) + shear * q0 * L^2 / (π^2 * κAG)
        end
        return (; δ, T, amp=2δ, t_peak=T / 2)
    elseif kind === :eb_cantilever || kind === :timoshenko_cantilever
        T = 2π / (1.875104068^2 * sqrt(EI / (ρA * L^4)))
        if load === :point
            δ = F0 * L^3 / (3 * EI) + shear * F0 / κAG
        elseif load === :uniform
            q = F0 / L
            δ = q * L^4 / (8 * EI) + shear * q * L^2 / (2 * κAG)
        else
            q0 = F0 * π / (2L)
            δ = _cantilever_sine_tip(q0, L, EI) + shear * q0 * L^2 / (π * κAG)
        end
        return (; δ, T, amp=2δ, t_peak=T / 2)
    else
        throw(ArgumentError("unknown beam kind $kind"))
    end
end

function _cantilever_sine_tip(q0, L, EI; n::Int=400)
    # w(L) = ∫_0^L M(ξ)(L-ξ)/EI dξ,  M(ξ)=q0[(L/π)(L-ξ)-(L/π)^2 sin(πξ/L)]
    acc = 0.0
    dξ = L / n
    k = π / L
    @inbounds for j in 1:n
        ξ = (j - 0.5) * dξ
        M = q0 * ((L / π) * (L - ξ) - (L / π)^2 * sin(k * ξ))
        acc += M * (L - ξ) * dξ
    end
    return acc / EI
end

"""First peak and period of a rest-to-step history (``1-\\cosωt``).

Peaks with ``u ≤ minfrac * max(u)`` are ignored (start-up ringing).
"""
function beam_amp_period(t, u; minfrac=0.0)
    n = length(u)
    n < 4 && return (; u_peak=NaN, t_peak=NaN, T_num=NaN, npeak=0)
    umax = maximum(u)
    thresh = minfrac * umax
    pks = Int[]
    @inbounds for i in 2:(n - 1)
        if u[i] >= u[i - 1] && u[i] >= u[i + 1] && u[i] > thresh
            push!(pks, i)
        end
    end
    isempty(pks) && return (; u_peak=umax, t_peak=t[argmax(u)], T_num=NaN, npeak=0)
    u_peak = u[pks[1]]
    t_peak = t[pks[1]]
    T_num = length(pks) >= 2 ? t[pks[2]] - t[pks[1]] : 2 * t_peak
    return (; u_peak, t_peak, T_num, npeak=length(pks))
end

# ---------------------------------------------------------------------------
# bar_sudden
# ---------------------------------------------------------------------------
function elastodynamics_problem(::Val{:bar_sudden}; mesh_tag::Int=200, L=1.0)
    ndiv = ELASTO_MESH_NDIV[mesh_tag]
    msh = mesh_elasticity_bar(; ndiv=ndiv, L=L, P=1.0,
        nome="elasto_bar_$(mesh_tag)")
    props = Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    T = _elasto_period_bar(; L=L, c=1.0)
    probe = Point2D(L, L / 2)
    ana = ana_bar_sudden(; N=400, c=1.0, L=L)
    return dad, (; name=:bar_sudden, mesh_tag, T, tf=4T, probe, comp=1,
        ana_kind=:disp, ana=(pt, t) -> ana.u(pt; t=t), notes="ν=0 plane stress vs 1D rod")
end

# ---------------------------------------------------------------------------
# Short beams (rectangle L×h, L/h = 4)
# ---------------------------------------------------------------------------
function mesh_short_beam(L, h; ndivx, ndivy, nome, leftbc, rightbc, botbc, topbc)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(L, h) / max(min(ndivx, ndivy), 4)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(L, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(L, h, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, h, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l2, ndivy)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndivy)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    groups = Dict{String,Vector{Int}}()
    for (l, tag) in ((l1, botbc), (l2, rightbc), (l3, topbc), (l4, leftbc))
        push!(get!(groups, tag, Int[]), l)
    end
    for (tag, curves) in groups
        gmsh.model.addPhysicalGroup(1, curves, -1, tag)
    end
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function _short_beam_divs(mesh_tag, L, h)
    ndivx = ELASTO_MESH_NDIV[mesh_tag]
    ndivy = max(4, round(Int, (ndivx - 1) * h / L) + 1)
    return ndivx, ndivy
end

function _make_ss_beam(kind::Symbol, load::Symbol, mesh_tag, L, h, b, E, ν, ρ, F0)
    ndivx, ndivy = _short_beam_divs(mesh_tag, L, h)
    nome = "elasto_$(kind)_$(load)_$(mesh_tag)"
    msh = mesh_short_beam(L, h; ndivx=ndivx, ndivy=ndivy, nome=nome,
        leftbc="1;0;1;0", rightbc="1;0;1;0", botbc="1;0;1;0", topbc="1;0;1;0")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    _apply_ss_supports!(dad, L)
    _apply_top_load!(dad, load, F0, L; halfw=max(L / ndivx, h / 8))
    ex = beam_expect(kind, load; L=L, h=h, b=b, E=E, ν=ν, ρ=ρ, F0=F0)
    probe = Point2D(L / 2, h)
    return dad, (; name=Symbol("$(kind)_$(load)"), mesh_tag, T=ex.T, tf=4 * ex.T,
        probe, comp=2, ana_kind=:amp_period, load=load, kind=kind,
        δ=ex.δ, amp=ex.amp, t_peak=ex.t_peak,
        ana=(pt, t) -> NaN, notes="SS pin-roller; compare 2δ and T1 only; L/h=$(L/h) load=$load")
end

function _make_cantilever_beam(kind::Symbol, load::Symbol, mesh_tag, L, h, b, E, ν, ρ, F0)
    ndivx, ndivy = _short_beam_divs(mesh_tag, L, h)
    nome = "elasto_$(kind)_$(load)_$(mesh_tag)"
    msh = mesh_short_beam(L, h; ndivx=ndivx, ndivy=ndivy, nome=nome,
        leftbc="0;0;0;0", rightbc="1;0;1;0", botbc="1;0;1;0", topbc="1;0;1;0")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    if load === :point
        _apply_right_ty!(dad, -F0)
    else
        _apply_top_load!(dad, load, F0, L; halfw=max(L / ndivx, h / 8))
    end
    ex = beam_expect(kind, load; L=L, h=h, b=b, E=E, ν=ν, ρ=ρ, F0=F0)
    probe = Point2D(L, h / 2)
    return dad, (; name=Symbol("$(kind)_$(load)"), mesh_tag, T=ex.T,
        tf=4 * ex.T, probe, comp=2, ana_kind=:amp_period, load=load, kind=kind,
        δ=ex.δ, amp=ex.amp, t_peak=ex.t_peak,
        ana=(pt, t) -> NaN, notes="clamped left; 2δ and T1; L/h=$(L/h) load=$load")
end

"""Parabolic end shear ``t_y = P(c²-y_c²)/(2I)``, ``t_x=0`` on ``x=L``."""
function _apply_parabolic_end_load!(dad, P, L, D)
    I = D^3 / 12
    c2 = (D / 2)^2
    xmax = maximum(p[1] for p in dad.Nodes)
    @inbounds for i in 1:dad.n
        abs(dad.Nodes[i][1] - xmax) < 1e-8 * (1 + abs(xmax)) || continue
        yc = dad.Nodes[i][2] - D / 2
        ty = P * (c2 - yc^2) / (2I)
        dad.BC[2i-1] = 1
        dad.BV[2i-1] = 0.0
        dad.BC[2i] = 1
        dad.BV[2i] = ty
    end
    return nothing
end

"""Root ``x=0``: exact TOE displacements (warping), not a rigid clamp."""
function _apply_toe_root_u!(dad, L, D, E, ν, P)
    @inbounds for i in 1:dad.n
        abs(dad.Nodes[i][1]) < 1e-8 * (1 + L) || continue
        u1, u2 = timoshenko_elasticity_cantilever_u(dad.Nodes[i][1], dad.Nodes[i][2];
            L=L, D=D, E=E, ν=ν, P=P)
        dad.BC[2i-1] = 0
        dad.BV[2i-1] = u1
        dad.BC[2i] = 0
        dad.BV[2i] = u2
    end
    return nothing
end

function _apply_right_ty!(dad, Fy)
    xmax = maximum(p[1] for p in dad.Nodes)
    lens = _nodal_lengths(dad)
    mask = [abs(dad.Nodes[i][1] - xmax) < 1e-8 * (1 + abs(xmax)) for i in 1:dad.n]
    Lr = sum(lens[mask])
    Lr <= 0 && error("right face empty")
    ty = Fy / Lr
    @inbounds for i in 1:dad.n
        mask[i] || continue
        dad.BC[2i] = 1
        dad.BV[2i] = ty
    end
    return ty
end

function elastodynamics_problem(::Val{:eb_ss}; mesh_tag::Int=200, load::Symbol=:point,
        L=SHORT_BEAM.L, h=SHORT_BEAM.h, b=SHORT_BEAM.b,
        E=SHORT_BEAM.E, ρ=SHORT_BEAM.ρ, F0=SHORT_BEAM.F0)
    return _make_ss_beam(:eb_ss, load, mesh_tag, L, h, b, E, 0.0, ρ, F0)
end

function elastodynamics_problem(::Val{:timoshenko_ss}; mesh_tag::Int=200, load::Symbol=:point,
        L=SHORT_BEAM.L, b=SHORT_BEAM.b, h=SHORT_BEAM.h,
        E=SHORT_BEAM.E, ν=SHORT_BEAM.ν, ρ=SHORT_BEAM.ρ, F0=SHORT_BEAM.F0)
    return _make_ss_beam(:timoshenko_ss, load, mesh_tag, L, h, b, E, ν, ρ, F0)
end

function elastodynamics_problem(::Val{:eb_cantilever}; mesh_tag::Int=200,
        load::Symbol=:point, L=SHORT_BEAM.L, h=SHORT_BEAM.h, b=SHORT_BEAM.b,
        E=SHORT_BEAM.E, ρ=SHORT_BEAM.ρ, F0=SHORT_BEAM.F0)
    return _make_cantilever_beam(:eb_cantilever, load, mesh_tag, L, h, b, E, 0.0, ρ, F0)
end

function elastodynamics_problem(::Val{:timoshenko_cantilever}; mesh_tag::Int=200,
        load::Symbol=:point, L=SHORT_BEAM.L, b=SHORT_BEAM.b, h=SHORT_BEAM.h,
        E=SHORT_BEAM.E, ν=SHORT_BEAM.ν, ρ=SHORT_BEAM.ρ, F0=SHORT_BEAM.F0)
    return _make_cantilever_beam(:timoshenko_cantilever, load, mesh_tag, L, h, b, E, ν, ρ, F0)
end

elastodynamics_problem(::Val{:eb_ss_point}; kwargs...) =
    elastodynamics_problem(Val(:eb_ss); load=:point, kwargs...)
elastodynamics_problem(::Val{:eb_ss_uniform}; kwargs...) =
    elastodynamics_problem(Val(:eb_ss); load=:uniform, kwargs...)
elastodynamics_problem(::Val{:eb_ss_sine}; kwargs...) =
    elastodynamics_problem(Val(:eb_ss); load=:sine, kwargs...)
elastodynamics_problem(::Val{:timoshenko_ss_point}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_ss); load=:point, kwargs...)
elastodynamics_problem(::Val{:timoshenko_ss_uniform}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_ss); load=:uniform, kwargs...)
elastodynamics_problem(::Val{:timoshenko_ss_sine}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_ss); load=:sine, kwargs...)
elastodynamics_problem(::Val{:eb_cantilever_point}; kwargs...) =
    elastodynamics_problem(Val(:eb_cantilever); load=:point, kwargs...)
elastodynamics_problem(::Val{:eb_cantilever_uniform}; kwargs...) =
    elastodynamics_problem(Val(:eb_cantilever); load=:uniform, kwargs...)
elastodynamics_problem(::Val{:eb_cantilever_sine}; kwargs...) =
    elastodynamics_problem(Val(:eb_cantilever); load=:sine, kwargs...)

"""Uniformly loaded strip with 1-D SS kinematics: axis pin–roller, traction-free ends."""
function elastodynamics_problem(::Val{:toe_beam_uniform}; mesh_tag::Int=200,
        l=SHORT_BEAM.L / 2, c=SHORT_BEAM.h / 2, E=SHORT_BEAM.E, ν=SHORT_BEAM.ν,
        q=1.0, ρ=SHORT_BEAM.ρ)
    L, D = 2l, 2c
    ndivx, ndivy = _short_beam_divs(mesh_tag, L, D)
    msh = mesh_short_beam(L, D; ndivx=ndivx, ndivy=ndivy, nome="elasto_toe_uni_$(mesh_tag)",
        leftbc="1;0;1;0", rightbc="1;0;1;0", botbc="1;0;1;0", topbc="1;0;1;0")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    _apply_bottom_qy!(dad, q)
    iL, iR = _pin_ss_axis!(dad, L, D)
    ex = beam_expect(:timoshenko_ss, :uniform; L=L, h=D, b=1.0, E=E, ν=ν, ρ=ρ, F0=q * L)
    I = toe_beam_I(c)
    δ_eb = 5 * q * l^4 / (24 * E * I)
    δ_airy = toe_beam_uniform_δ(; l=l, c=c, E=E, ν=ν, q=q)
    T = ex.T
    probe = Point2D(l, 2c)  # top mid-span, on Γ
    ana = (pt, tt=0.0) -> toe_beam_uniform_u_mesh(pt[1], pt[2]; l=l, c=c, E=E, ν=ν, q=q)
    return dad, (; name=:toe_beam_uniform, mesh_tag, T, tf=4T, probe, comp=2,
        ana_kind=:amp_period, load=:uniform, kind=:toe_beam_uniform,
        δ=ex.δ, amp=2 * ex.δ, t_peak=T / 2, δ_eb=δ_eb, δ_airy=δ_airy,
        ana=ana, l=l, c=c, E=E, ν=ν, q=q, iL=iL, iR=iR,
        notes="1-D SS: axis pin–roller; bottom σ_y=-q; traction-free ends")
end

"""Bottom face ``σ_y=-q``, ``τ=0``; all other Neumann DOFs stay ``t=0``."""
function _apply_bottom_qy!(dad, q)
    mask = _bottom_mask(dad)
    @inbounds for i in eachindex(mask)
        mask[i] || continue
        ny = dad.Normal[i][2]
        dad.BC[2i-1] = 1
        dad.BV[2i-1] = 0.0
        dad.BC[2i] = 1
        dad.BV[2i] = -q * ny
    end
    return nothing
end

"""Pin–roller on the neutral axis: ``u=v=0`` at left mid-height, ``v=0`` at right."""
function _pin_ss_axis!(dad, L, D)
    n = dad.n
    xmin = minimum(p[1] for p in dad.Nodes)
    xmax = maximum(p[1] for p in dad.Nodes)
    left = Int[i for i in 1:n if abs(dad.Nodes[i][1] - xmin) < 1e-8 * (1 + abs(xmin))]
    right = Int[i for i in 1:n if abs(dad.Nodes[i][1] - xmax) < 1e-8 * (1 + abs(xmax))]
    isempty(left) && error("left face empty")
    isempty(right) && error("right face empty")
    iL = left[argmin(norm(dad.Nodes[i] - Point2D(0.0, D / 2)) for i in left)]
    iR = right[argmin(norm(dad.Nodes[i] - Point2D(L, D / 2)) for i in right)]
    dad.BC[2iL-1] = 0; dad.BV[2iL-1] = 0.0
    dad.BC[2iL]   = 0; dad.BV[2iL]   = 0.0
    dad.BC[2iR]   = 0; dad.BV[2iR]   = 0.0
    return iL, iR
end

"""2-D Timoshenko *Theory of Elasticity* cantilever: warped-root Dirichlet + parabolic tip shear."""
function elastodynamics_problem(::Val{:toe_cantilever}; mesh_tag::Int=200,
        L=SHORT_BEAM.L, D=SHORT_BEAM.h, E=SHORT_BEAM.E, ν=SHORT_BEAM.ν,
        P=SHORT_BEAM.F0, ρ=SHORT_BEAM.ρ)
    ndivx, ndivy = _short_beam_divs(mesh_tag, L, D)
    msh = mesh_short_beam(L, D; ndivx=ndivx, ndivy=ndivy, nome="elasto_toe_c_$(mesh_tag)",
        leftbc="1;0;1;0", rightbc="1;0;1;0", botbc="1;0;1;0", topbc="1;0;1;0")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    _apply_toe_root_u!(dad, L, D, E, ν, P)
    _apply_parabolic_end_load!(dad, P, L, D)
    δ = timoshenko_elasticity_cantilever_tip(; L=L, D=D, E=E, ν=ν, P=P)
    T = _timoshenko_cantilever_T1(L, 1.0, D, E, ν, ρ)
    probe = Point2D(L, D / 2)
    ana = (pt, t=0.0) -> timoshenko_elasticity_cantilever_u(pt[1], pt[2]; L=L, D=D, E=E, ν=ν, P=P)
    return dad, (; name=:toe_cantilever, mesh_tag, T, tf=4T, probe, comp=2,
        ana_kind=:toe, load=:parabolic, kind=:toe_cantilever, δ=δ, amp=2δ, t_peak=T / 2,
        ana=ana, L=L, D=D, E=E, ν=ν, P=P,
        notes="TOE cantilever; parabolic ty on x=L; exact u on x=0")
end
elastodynamics_problem(::Val{:timoshenko_cantilever_point}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_cantilever); load=:point, kwargs...)
elastodynamics_problem(::Val{:timoshenko_cantilever_uniform}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_cantilever); load=:uniform, kwargs...)
elastodynamics_problem(::Val{:timoshenko_cantilever_sine}; kwargs...) =
    elastodynamics_problem(Val(:timoshenko_cantilever); load=:sine, kwargs...)

# ---------------------------------------------------------------------------
# cylinder (full annulus)
# ---------------------------------------------------------------------------
function mesh_full_annulus(a, b; nθ, nr, nome="annulus", p_inner=1.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (b - a) / max(nr, 3)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    pi1 = gmsh.model.geo.addPoint(a, 0.0, 0.0, lc)
    pi2 = gmsh.model.geo.addPoint(0.0, a, 0.0, lc)
    pi3 = gmsh.model.geo.addPoint(-a, 0.0, 0.0, lc)
    pi4 = gmsh.model.geo.addPoint(0.0, -a, 0.0, lc)
    po1 = gmsh.model.geo.addPoint(b, 0.0, 0.0, lc)
    po2 = gmsh.model.geo.addPoint(0.0, b, 0.0, lc)
    po3 = gmsh.model.geo.addPoint(-b, 0.0, 0.0, lc)
    po4 = gmsh.model.geo.addPoint(0.0, -b, 0.0, lc)
    ai1 = gmsh.model.geo.addCircleArc(pi1, c, pi2)
    ai2 = gmsh.model.geo.addCircleArc(pi2, c, pi3)
    ai3 = gmsh.model.geo.addCircleArc(pi3, c, pi4)
    ai4 = gmsh.model.geo.addCircleArc(pi4, c, pi1)
    ao1 = gmsh.model.geo.addCircleArc(po1, c, po2)
    ao2 = gmsh.model.geo.addCircleArc(po2, c, po3)
    ao3 = gmsh.model.geo.addCircleArc(po3, c, po4)
    ao4 = gmsh.model.geo.addCircleArc(po4, c, po1)
    cli = gmsh.model.geo.addCurveLoop([ai1, ai2, ai3, ai4])
    clo = gmsh.model.geo.addCurveLoop([ao1, ao2, ao3, ao4])
    s = gmsh.model.geo.addPlaneSurface([clo, cli])
    gmsh.model.geo.synchronize()
    nseg = max(nθ ÷ 4, 4)
    for crv in (ai1, ai2, ai3, ai4, ao1, ao2, ao3, ao4)
        gmsh.model.mesh.setTransfiniteCurve(crv, nseg)
    end
    # Inner traction overwritten in `_apply_inner_pressure!` (not constant Cartesian).
    gmsh.model.addPhysicalGroup(1, [ai1, ai2, ai3, ai4], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ao1, ao2, ao3, ao4], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function _apply_inner_pressure!(dad, p0, a; rtol=0.05)
    @inbounds for i in 1:dad.n
        r = norm(dad.Nodes[i])
        abs(r - a) / a < rtol || continue
        n = dad.Normal[i]
        # n is BEM outward of Ω. On inner wall n ≈ -ê_r. t = p0 * ê_r = -p0 * n
        dad.BC[2i-1] = 1
        dad.BC[2i] = 1
        dad.BV[2i-1] = -p0 * n[1]
        dad.BV[2i] = -p0 * n[2]
    end
    return nothing
end

function elastodynamics_problem(::Val{:cylinder}; mesh_tag::Int=200,
        a=1.0, b=2.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0)
    nθ = mesh_tag == 200 ? 32 : mesh_tag == 500 ? 48 : 72
    nr = mesh_tag == 200 ? 4 : mesh_tag == 500 ? 6 : 8
    msh = mesh_full_annulus(a, b; nθ=nθ, nr=nr, nome="elasto_cyl_$(mesh_tag)")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_strain=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    _apply_inner_pressure!(dad, p0, a)
    μ = E / (2 * (1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    cp = sqrt((λ + 2μ) / ρ)
    T = 2π * (b - a) / cp
    probe = Point2D(a, 0.0)
    return dad, (; name=:cylinder, mesh_tag, T, tf=4T, probe, comp=1,
        ana_kind=:disp, ana=(pt, t) -> cylinder_step_pressure(norm(pt), t; a=a, b=b, E=E, ν=ν, ρ=ρ, p0=p0),
        notes="plane strain annulus vs Ding series")
end

function elastodynamics_problem(::Val{:plate_hole}; mesh_tag::Int=200,
        a=1.0, b=4.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0)
    nθ = mesh_tag == 200 ? 32 : mesh_tag == 500 ? 48 : 72
    msh = mesh_full_annulus(a, b; nθ=nθ, nr=mesh_tag == 200 ? 4 : 6,
        nome="elasto_hole_$(mesh_tag)")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    _apply_inner_pressure!(dad, p0, a)
    T = _elasto_period_hole(; a=a, E=E, ν=ν, ρ=ρ, plane_stress=true)
    probe = Point2D(1.5 * a, 0.0)
    return dad, (; name=:plate_hole, mesh_tag, T, tf=4T, probe, comp=1,
        ana_kind=:disp,
        ana=(pt, t) -> plate_hole_pressure(norm(pt), t; a=a, E=E, ν=ν, ρ=ρ, p0=p0),
        notes="truncated outer radius $(b); infinite-domain formula")
end

function elastodynamics_problem(::Val{:kirsch}; mesh_tag::Int=200,
        a=50.0, E=1.0e5, ν=0.25, ρ=1.0, σ∞=1.0)
    ndiv = mesh_tag == 200 ? 10 : mesh_tag == 500 ? 16 : 24
    msh = mesh_plate_with_hole(; ndiv=ndiv, nome="elasto_kirsch_$(mesh_tag)")
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    T = _elasto_period_hole(; a=a, E=E, ν=ν, ρ=ρ, plane_stress=true)
    probe = Point2D(0.0, a)
    return dad, (; name=:kirsch, mesh_tag, T, tf=4T, probe, comp=1,
        ana_kind=:stress,
        ana=(pt, t) -> transient_kirsch(norm(pt), atan(pt[2], pt[1] + 1e-30), t;
            a=a, E=E, ν=ν, ρ=ρ, σ∞=σ∞),
        notes="illustrative Kirsch; finite plate; u_num is ux not hoop stress")
end

function elastodynamics_problem(name::Symbol; kwargs...)
    return elastodynamics_problem(Val(name); kwargs...)
end

const BEAM_PROBLEMS = (
    :eb_ss_point, :eb_ss_uniform, :eb_ss_sine,
    :timoshenko_ss_point, :timoshenko_ss_uniform, :timoshenko_ss_sine,
    :eb_cantilever_point, :eb_cantilever_uniform, :eb_cantilever_sine,
    :timoshenko_cantilever_point, :timoshenko_cantilever_uniform, :timoshenko_cantilever_sine,
)
const TOE_BEAM_PROBLEMS = (:toe_cantilever, :toe_beam_uniform)
const ELASTO_PROBLEMS = (
    :bar_sudden, BEAM_PROBLEMS...,
    :cylinder, :plate_hole, :kirsch,
)
