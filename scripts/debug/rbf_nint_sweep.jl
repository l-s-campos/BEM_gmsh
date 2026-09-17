# Does more DIBEM interiors fix RBF ∇w / flux on SS sine?
# julia --project=. scripts/debug/rbf_nint_sweep.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate

const a = 1.0
const W = 0.013077  # Table 3 L/h=100 P̄=300 linear w_c
const RBF_G = PHS(3; poly_deg=1)

E2 = 1.0
E1 = 25 * E2
t = (a / 100) / 4
pl = [(E1, E2, 0.25, 0.5 * E2, Float64(θ), t) for θ in (0, 90, 90, 0)]
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=0.5 * E2, G23=0.2 * E2)

function flux_of(wx, wy)
    vx = similar(wx)
    @inbounds for i in eachindex(wx)
        εx, εy, γ = 0.5 * wx[i]^2, 0.5 * wy[i]^2, wx[i] * wy[i]
        Nxx = A[1, 1] * εx + A[1, 2] * εy + A[1, 3] * γ
        Nyy = A[1, 2] * εx + A[2, 2] * εy + A[2, 3] * γ
        Nxy = A[1, 3] * εx + A[2, 3] * εy + A[3, 3] * γ
        vx[i] = Nxx * wx[i] + Nxy * wy[i]
    end
    return vx
end

function fields(pts)
    w = [W * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
    wx = [W * (π / a) * cos(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
    wy = [W * (π / a) * sin(π * p[1] / a) * cos(π * p[2] / a) for p in pts]
    return w, wx, wy, flux_of(wx, wy)
end

rel(u, v) = norm(u - v) / (norm(v) + 1e-30)

function eval_cloud(pts, nΓ, rbf)
    n = length(pts)
    ib, ii = 1:nΓ, (nΓ + 1):n
    w, wx_a, wy_a, vx_a = fields(pts)
    ops = rbf_gradient_ops(pts; rbf=rbf)
    wx, wy = ops.Fx * w, ops.Fy * w
    vx = flux_of(wx, wy)
    cF = try
        cond(ops.Fx)
    catch
        Inf
    end
    return (cond=cF,
        wx=rel(wx, wx_a), wxΩ=rel(wx[ii], wx_a[ii]), wxΓ=rel(wx[ib], wx_a[ib]),
        vx=rel(vx, vx_a), vxΩ=rel(vx[ii], vx_a[ii]), vxΓ=rel(vx[ib], vx_a[ib]),
        wxmid=begin
            i0 = argmin(norm(p - Point2D(0.0, a / 2)) for p in pts[1:nΓ])
            wx[i0] / (wx_a[i0] + 1e-30)
        end)
end

function print_row(nel, ni, r)
    @printf("  %3d %4d %5d  %9.2e  %5.3f %5.3f %5.3f  %5.3f %5.3f %5.3f  %5.3f\n",
        nel, ni, nel == 0 ? 0 : 0, r.cond, r.wx, r.wxΩ, r.wxΓ, r.vx, r.vxΩ, r.vxΓ,
        r.wxmid)
end

println("PHS3+lin, sine samples, NEL fixed, ni swept")
@printf("  %-3s %-4s %-5s  %-9s  %-5s %-5s %-5s  %-5s %-5s %-5s  %s\n",
    "NEL", "ni", "nt", "cond(Dx)", "wx", "wxΩ", "wxΓ", "vx", "vxΩ", "vxΓ", "wx_mid/true")

for nel in (4, 6)
    dummy = FSDTProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0)
    dad = build_square_fsdt(; a=a, n_el=nel, bc="SSSS", props=dummy, n_internal=1)
    Γ = Point2D[dad.Nodes;]
    nΓ = length(Γ)
    println("  --- NEL=$nel  nΓ=$nΓ ---")
    for ni in (1, 4, 9, 16, 25, 36, 49, 81, 121, 169)
        Ω = BEM.Plate._fsdt_cell_centroids(a, a, ni)
        pts = Point2D[Γ; Ω]
        r = eval_cloud(pts, nΓ, RBF_G)
        @printf("  %3d %4d %5d  %9.2e  %5.3f %5.3f %5.3f  %5.3f %5.3f %5.3f  %5.3f\n",
            nel, length(Ω), length(pts), r.cond, r.wx, r.wxΩ, r.wxΓ, r.vx, r.vxΩ,
            r.vxΓ, r.wxmid)
    end
end

println("\nPHS5+lin on NEL=4")
dummy = FSDTProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0)
dad = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=dummy, n_internal=1)
Γ = Point2D[dad.Nodes;]
nΓ = length(Γ)
for ni in (9, 25, 49, 81)
    Ω = BEM.Plate._fsdt_cell_centroids(a, a, ni)
    pts = Point2D[Γ; Ω]
    r = eval_cloud(pts, nΓ, PHS(5; poly_deg=1))
    @printf("  %3d %4d %5d  %9.2e  %5.3f %5.3f %5.3f  %5.3f %5.3f %5.3f  %5.3f\n",
        4, length(Ω), length(pts), r.cond, r.wx, r.wxΩ, r.wxΓ, r.vx, r.vxΩ,
        r.vxΓ, r.wxmid)
end

println("\nInterior-only cloud (no Γ): can PHS recover ∇sine without Dirichlet ring?")
for ni in (9, 25, 49, 81, 169)
    Ω = BEM.Plate._fsdt_cell_centroids(a, a, ni)
    w, wx_a, wy_a, vx_a = fields(Ω)
    ops = rbf_gradient_ops(Ω; rbf=RBF_G)
    wx = ops.Fx * w
    vx = flux_of(wx, ops.Fy * w)
    @printf("  ni=%3d  cond=%.2e  wx=%.3f  vx=%.3f  wx_c/true=%.3f\n",
        ni, cond(ops.Fx), rel(wx, wx_a), rel(vx, vx_a),
        wx[1] / (wx_a[1] + 1e-30))
end
println("done")
