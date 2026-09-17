using LinearAlgebra, StaticArrays, FastGaussQuadrature, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=10)
pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
nh = SVector(1.0, 0.0)
nξ = SVector(0.0, 1.0)
RX, RY = pg[1] - pf[1], pg[2] - pf[2]
R = hypot(RX, RY)
At = BEM.Plate._hsu_At(p.AT)

# --- single-θ: Y' vs FD of Y, and dN vs FD of N ---
θ = atan(RY, RX)  # ρ = |r| here (ω ∥ r)
Ω, ω = BEM.Plate._unsym_Ωω(θ)
ρ = ω[1] * RX + ω[2] * RY
Z, L2, λd = BEM.Plate._unsym_Z_L2(θ, p.A, p.B, p.D, At)
println("θ=", θ, " ρ=", ρ, " condZ=", cond(Z), " λd=", λd)
Zi = inv(Z)
L2inv = inv(L2)
c = Zi[:, 6:10] * L2inv

function state(ρ)
    F = zeros(10, 10)
    BEM.Plate._unsym_F!(F, ρ)
    BEM.Plate._unsym_Fd!(F, ρ, λd)
    return Z * (F * c)
end
function statep(ρ)
    Fp = zeros(10, 10)
    BEM.Plate._unsym_Fp!(Fp, ρ)
    BEM.Plate._unsym_Fdp!(Fp, ρ, λd)
    return Z * (Fp * c)
end

h = 1e-6
Y = state(ρ)
Yp = statep(ρ)
Yfd = (state(ρ + h) - state(ρ - h)) / (2h)
println("Y' vs FD rel = ", maximum(abs.(Yp .- Yfd)) / maximum(abs, Yfd))
println("max|Y|=", maximum(abs, Y), " max|Yp|=", maximum(abs, Yp), " max|Yfd|=", maximum(abs, Yfd))
println("Y[1:5] vs Yp wait: max|Y[6:10] - Yfd wait analytic Yp[1:5] vs Y[6:10] rel = ",
    maximum(abs.(Yp[1:5, :] .- Y[6:10, :])) / max(maximum(abs, Y[6:10, :]), 1e-30))

function NMQ(ρ)
    Y = state(ρ)
    v = Y[1:5, :]
    vp = Y[6:10, :]
    N = p.A * (Ω * vp[1:2, :]) + p.B * (Ω * vp[3:4, :])
    M = p.B * (Ω * vp[1:2, :]) + p.D * (Ω * vp[3:4, :])
    Q = At * (v[3:4, :] .+ ω * vp[5:5, :])
    return N, M, Q
end
function dNMQ(ρ)
    Y = state(ρ)
    Yp = statep(ρ)
    vρ = Y[6:10, :]
    vρρ = Yp[6:10, :]
    dN = p.A * (Ω * vρρ[1:2, :]) + p.B * (Ω * vρρ[3:4, :])
    dM = p.B * (Ω * vρρ[1:2, :]) + p.D * (Ω * vρρ[3:4, :])
    dQ = At * (vρ[3:4, :] .+ ω * vρρ[5:5, :])
    return dN, dM, dQ
end
N, M, Q = NMQ(ρ)
dN, dM, dQ = dNMQ(ρ)
Np, Mp, Qp = NMQ(ρ + h)
Nm, Mm, Qm = NMQ(ρ - h)
println("dN vs FD rel = ", maximum(abs.(dN .- (Np - Nm) / 2h)) / maximum(abs, (Np - Nm) / 2h))
println("dM vs FD rel = ", maximum(abs.(dM .- (Mp - Mm) / 2h)) / maximum(abs, (Mp - Mm) / 2h))
println("dQ vs FD rel = ", maximum(abs.(dQ .- (Qp - Qm) / 2h)) / maximum(abs, (Qp - Qm) / 2h))
println("max|dN|=", maximum(abs, dN), " max|N|=", maximum(abs, N))

# --- S by FD of P (reference) ---
hfd = 1e-5
_, P0 = unsym_fsdt_kernels(pg, pf, nh, p)
_, Pp = unsym_fsdt_kernels(pg, pf + hfd * nξ, nh, p)
_, Pm = unsym_fsdt_kernels(pg, pf - hfd * nξ, nh, p)
Sfd = (Pp - Pm) / (2 * hfd)
_, Sana = unsym_hbie_kernels(pg, pf, nh, nξ, p)
println("\nanalytic S vs Sfd rel = ", maximum(abs.(Sana .- Sfd)) / maximum(abs, Sfd))

# --- rebuild S with gauss, more θ, skip small ρ ---
function S_quad(; nθ=40, map=:gauss, ρmin=0.0)
    p2 = UnsymFSDTProps(p.A, p.B, p.D, p.AT; h=p.h, nθ=nθ, map=map)
    eg, wg0 = gausslegendre(nθ)
    Sast = zeros(5, 5)
    F = zeros(10, 10)
    Fp = zeros(10, 10)
    θ0 = atan(-RX, RY)
    n1, n2 = nh[1], nh[2]
    skipped = 0
    used = 0
    maxdN = 0.0
    for iq in 1:4
        eet = (iq == 1 || iq == 3) ? -1.0 : 1.0
        et, wg = BEM.Plate._cluster_rule(eg, wg0, eet, p2.map; b=p2.sinh_b)
        for i in 1:nθ
            ξ = clamp(et[i], -1.0, 1.0)
            θ = θ0 + (ξ + 1) * π / 4
            Ω, ω = BEM.Plate._unsym_Ωω(θ)
            ρ = ω[1] * RX + ω[2] * RY
            if abs(ρ) < max(1e-14, ρmin)
                skipped += 1
                continue
            end
            got = BEM.Plate._unsym_Z_L2(θ, p.A, p.B, p.D, At)
            got === nothing && continue
            Z, L2, λd = got
            cond(Z) > 1e12 && continue
            fill!(F, 0); fill!(Fp, 0)
            BEM.Plate._unsym_F!(F, ρ)
            BEM.Plate._unsym_Fd!(F, ρ, λd)
            BEM.Plate._unsym_Fp!(Fp, ρ)
            BEM.Plate._unsym_Fdp!(Fp, ρ, λd)
            Zi = inv(Z)
            L2inv = inv(L2)
            c = Zi[:, 6:10] * L2inv
            Yp = Z * (Fp * c)
            vρ = (Z * (F * c))[6:10, :]
            vρρ = Yp[6:10, :]
            dN = (p.A * (Ω * vρρ[1:2, :]) + p.B * (Ω * vρρ[3:4, :]))
            dM = (p.B * (Ω * vρρ[1:2, :]) + p.D * (Ω * vρρ[3:4, :]))
            dQ = At * (vρ[3:4, :] .+ ω * vρρ[5:5, :])
            maxdN = max(maxdN, maximum(abs, dN))
            nξω = nξ[1] * ω[1] + nξ[2] * ω[2]
            wθ = wg[i] * (π / 4)
            s = -nξω * wθ
            for j in 1:5
                Sast[j, 1] += (dN[1, j] * n1 + dN[3, j] * n2) * s
                Sast[j, 2] += (dN[3, j] * n1 + dN[2, j] * n2) * s
                Sast[j, 3] += (dM[1, j] * n1 + dM[3, j] * n2) * s
                Sast[j, 4] += (dM[3, j] * n1 + dM[2, j] * n2) * s
                Sast[j, 5] += (dQ[1, j] * n1 + dQ[2, j] * n2) * s
            end
            used += 1
        end
        θ0 += π / 2
    end
    Sast .*= 1 / (4 * π^2)
    rel = maximum(abs.(Sast .- Sfd)) / maximum(abs, Sfd)
    println("map=", map, " nθ=", nθ, " ρmin=", ρmin,
        " used=", used, " skip=", skipped,
        " max|S|=", maximum(abs, Sast), " maxdN=", maxdN, " rel=", rel)
    return Sast
end

S_quad(; nθ=10, map=:telles, ρmin=0.0)
S_quad(; nθ=24, map=:gauss, ρmin=0.0)
S_quad(; nθ=48, map=:gauss, ρmin=0.0)
S_quad(; nθ=48, map=:gauss, ρmin=0.05 * R)
S_quad(; nθ=48, map=:gauss, ρmin=0.2 * R)

# S by FD of P inside (complex-step along nξ)
ε = 1e-20
# real FD already Sfd
println("\nSfd max=", maximum(abs, Sfd))
display(Sfd)
