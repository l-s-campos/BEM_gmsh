# Trials for MAT1 P3 HBIE: transpose, no jump, z-sinh, high npg.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
msh = datadir("elastico", "cordeiro_p3.msh")

function p3dad()
    format2d(msh, props; tipo=2, pontointerno=false)
end

dad = p3dad()
assemble!(dad; npg=16, threaded=false); solve(dad)
uc = copy(dad.u)
println("CBIE max|u|=$(maximum(abs, uc))")

function report(lab, dad, uc)
    solve(dad)
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    @printf("%-28s max|u|=%10.3f  relCBIE=%.3e  cond=%.3e\n",
        lab, maximum(abs, dad.u), rel, cond(Matrix(dad.A)))
end

# baseline SST
dad = p3dad()
H_G_hyper(dad; npg=16, threaded=false)
report("HBIE baseline", dad, uc)

# transpose H' and G'
dad = p3dad()
H_G_hyper(dad; npg=16, threaded=false)
H = Matrix(dad.H)'; G = Matrix(dad.G)'
set_cache!(dad; H=H, G=G, H_hyper=H, G_hyper=G)
report("HBIE H',G' transposed", dad, uc)

# no 1/2 jump on G
dad = p3dad()
H_G_hyper(dad; npg=16, threaded=false)
G = copy(dad.G)
n = dad.n
for i in 1:n
    G[2i-1, 2i-1] += 0.5
    G[2i, 2i] += 0.5
end
set_cache!(dad; G=G, G_hyper=G)
report("HBIE no G jump", dad, uc)

# no rigid row-sum: rebuild without the H diagonal replace
# (already tried earlier; skip if hard)

# characteristic sinh: min |z_m|
μ = props.params.mi
function z_closest(poly, nodes, pf)
    # coarse scan + Newton on |z_m|^2
    ξbest = 0.0; dbest = Inf
    for ξ in range(-1, 1; length=21)
        N, _ = BEM.shapefun(poly, ξ)
        pg = (N * nodes)[1]
        r = pg - pf
        d = min(abs(r[1]+μ[1]*r[2]), abs(r[1]+μ[2]*r[2]))
        if d < dbest
            dbest = d; ξbest = ξ
        end
    end
    return ξbest, dbest
end

@eval BEM function transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type)
    if has_cache(dad, :nearfield) && dad.nearfield === :zsinh
        μ = dad.properties.params.mi
        ξbest = 0.0; dbest = Inf
        for ξ in range(-1.0, 1.0; length=21)
            N, _ = shapefun(poly, ξ)
            pg = (N * nodes)[1]
            r = pg - pf
            d = min(abs(r[1]+μ[1]*r[2]), abs(r[1]+μ[2]*r[2]))
            if d < dbest
                dbest = Float64(d); ξbest = Float64(ξ)
            end
        end
        b = dbest / max(elem.Length, eps())
        return nearfield_1d(ξbest, max(b, 1e-8); qsi=dad.qsi, w=dad.w)
    end
    if has_cache(dad, :nearfield) && dad.nearfield === :plain
        return dad.qsi, dad.w
    end
    a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
    b = dist / max(elem.Length, eps())
    return nearfield_1d(a, b; qsi=dad.qsi, w=dad.w)
end

dad = p3dad()
set_cache!(dad; nearfield=:zsinh)
H_G_hyper(dad; npg=16, threaded=false)
report("HBIE z-sinh npg=16", dad, uc)

dad = p3dad()
set_cache!(dad; nearfield=:zsinh)
H_G_hyper(dad; npg=50, threaded=false)
report("HBIE z-sinh npg=50", dad, uc)

# kernel FD near MAT1 characteristic (z2 small)
r = Point2D(0.216, -1.0)  # r1 + 0.067 r2 ≈ 0.216-0.067=0.15, not tiny
# r1 + Re(μ2) r2 = 0 → r1 = -0.067 r2. Take r2=1, r1=-0.067
r = Point2D(-0.067, 1.0)
n = Point2D(0.0, 1.0); nf = Point2D(1.0, 0.0)
println("\nnear-char r=$r  |z2|=$(abs(r[1]+μ[2]*r[2]))  |r|=$(norm(r))")
h = 1e-6
Uh, Th = let kp = fundamental_hyper(props, r, zero(r), n, nf)
    BEM._to_smat(kp.U), BEM._to_smat(kp.T)
end
function fd_col(which)
    ξ = zero(r)
    function UTξ(δ)
        kp = fundamental(props, r, ξ+δ, n)
        BEM._to_smat(kp.U), BEM._to_smat(kp.T)
    end
    U0, T0 = UTξ(Point2D(0,0))
    Ux, Tx = UTξ(Point2D(h,0))
    Uy, Ty = UTξ(Point2D(0,h))
    C = props.params.C
    out = zeros(2, 2)
    Kx, Ky = which === :U ? (Ux-U0, Uy-U0) ./ h : (Tx-T0, Ty-T0) ./ h
    for k in 1:2
        ε = SVector(Kx[1,k], Ky[2,k], Ky[1,k]+Kx[2,k])
        σ = C * ε
        out[1,k] = nf[1]*σ[1] + nf[2]*σ[3]
        out[2,k] = nf[1]*σ[3] + nf[2]*σ[2]
    end
    return out
end
UhFD = fd_col(:U); ThFD = fd_col(:T)
println("  near-char FD Uh rel=$(norm(Uh-UhFD)/(norm(Uh)+1e-30))")
println("  near-char FD Th rel=$(norm(Th-ThFD)/(norm(Th)+1e-30))")
println("done")
