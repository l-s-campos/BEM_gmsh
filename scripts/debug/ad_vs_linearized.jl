# AD Jacobian vs linearized (J=A and frozen-N geometric stiffness)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, Statistics
using BEM.Plate
using .ThinPlate

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
q0 = 40 * E * h^4
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
plate = build_square_plate(; a=a, n_el=2, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=6, singular=:analytic)

include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=3, show=false, nome="ad_lin_pe", Lx=a, Ly=a)
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0)
fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=6, threaded=false)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=6, npg_pe=6)
n = length(plate.nodes)
nq = norm(prob.q_load) + eps()
nA = norm(prob.A_pl) + eps()

function geo_weight(prob)
    D = bending_stiffness(prob.plate.props)
    xs = getindex.(prob.pts, 1)
    ys = getindex.(prob.pts, 2)
    Lref = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
    return (Lref^2) / (8 * π * D + eps())
end

"""Frozen-N geometric tangent: J = A − ∂(N:∇∇w)/∂x with N held from current x."""
function lin_jacobian(prob, x)
    u = BEM.Plate.pack_plate_u(prob, x)
    w = BEM.Plate.extract_w_field(prob, u)
    Nxx_nl, Nyy_nl, Nxy_nl = BEM.Plate.membrane_N_from_w(prob, w)
    Nxx, Nyy, Nxy = BEM.Plate.solve_membrane_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
    wt = geo_weight(prob)
    Fxx = prob.Fx * prob.Fx
    Fyy = prob.Fy * prob.Fy
    Fxy = prob.Fx * prob.Fy
    nw = length(w)
    L = zeros(nw, nw)
    @inbounds for j in 1:nw
        ej = zeros(nw)
        ej[j] = 1
        L[:, j] .= Nxx .* (Fxx * ej) .+ 2 .* Nxy .* (Fxy * ej) .+ Nyy .* (Fyy * ej)
    end
    J = copy(prob.A_pl)
    @inbounds for (kj, dofj) in enumerate(prob.w_index)
        prob.is_kin[dofj] && continue
        for (ki, dofi) in enumerate(prob.w_index)
            prob.is_kin[dofi] && continue
            J[dofi, dofj] -= wt * L[ki, kj]
        end
    end
    return J
end

function wh(prob, x)
    u = BEM.Plate.pack_plate_u(prob, x)
    return abs(u[2n + 1]) / h
end

function newton_hist!(Jfun, x0, λ; maxiters=5)
    x = copy(x0)
    hist = Float64[]
    finite = true
    for _ in 1:maxiters
        R = BEM.Plate._residual_xλ(prob, x, λ)
        push!(hist, norm(R) / nq)
        if !all(isfinite, R)
            finite = false
            break
        end
        J = Jfun(x, λ)
        dx = try
            J \ R
        catch
            finite = false
            break
        end
        any(!isfinite, dx) && (finite = false; break)
        x .-= dx
        if !all(isfinite, x)
            x .+= dx
            finite = false
            break
        end
        if norm(x) > 1e6 * (norm(x0) + 1)
            finite = false
            break
        end
    end
    if finite
        push!(hist, norm(BEM.Plate._residual_xλ(prob, x, λ)) / nq)
    end
    return x, hist, finite
end

Random.seed!(1)
println("ndof = ", length(prob.is_kin), "   ||q|| = ", nq)
println()
println("Jacobian at Picard state  (Pala: Q=5,10,20 → w/h=0.521,0.852,1.255)")
@printf("%6s %8s %10s %10s %10s %10s %10s\n",
    "Q", "w/h", "||Jad-A||", "||Jad-G||", "FD-AD", "FD-G", "FD-A")

for Q in (5.0, 10.0, 20.0)
    λ = Q / 40
    x = BEM.Plate._picard_plate_step(prob, λ, zeros(length(prob.is_kin)); niter=8, e=0.5)
    Jad = BEM.Plate._ad_jacobian(prob, x, λ)
    JG = lin_jacobian(prob, x)
    R = BEM.Plate._residual_xλ(prob, x, λ)
    s = randn(length(x))
    s ./= norm(s)
    ε = 1e-6 * (1 + norm(x))
    dR = (BEM.Plate._residual_xλ(prob, x .+ ε .* s, λ) .- R) ./ ε
    ndR = norm(dR) + eps()
    @printf("%6.0f %8.4f %10.3e %10.3e %10.3e %10.3e %10.3e\n",
        Q, wh(prob, x),
        norm(Jad - prob.A_pl) / nA,
        norm(Jad - JG) / nA,
        norm(dR - Jad * s) / ndR,
        norm(dR - JG * s) / ndR,
        norm(dR - prob.A_pl * s) / ndR)
end

println()
println("Newton from linear guess: rel residual ||R||/||q|| per iteration")
for Q in (5.0, 10.0, 20.0)
    λ = Q / 40
    xlin = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
    println("-"^60)
    @printf("Q=%g  λ=%.3f  linear w/h=%.4f\n", Q, λ, wh(prob, xlin))
    xA, hA, okA = newton_hist!((x, λ) -> prob.A_pl, xlin, λ)
    xG, hG, okG = newton_hist!((x, λ) -> lin_jacobian(prob, x), xlin, λ)
    xD, hD, okD = newton_hist!((x, λ) -> BEM.Plate._ad_jacobian(prob, x, λ), xlin, λ)
    niter = maximum(length.((hA, hG, hD)))
    @printf("  %4s  %12s %12s %12s\n", "it", "J=A", "J=A-G(N)", "J=AD")
    for i in 1:niter
        a = i <= length(hA) ? @sprintf("%.4e", hA[i]) : "—"
        g = i <= length(hG) ? @sprintf("%.4e", hG[i]) : "—"
        d = i <= length(hD) ? @sprintf("%.4e", hD[i]) : "—"
        @printf("  %4d  %12s %12s %12s\n", i - 1, a, g, d)
    end
    @printf("  final w/h   %12.4f %12.4f %12.4f\n",
        okA ? wh(prob, xA) : NaN,
        okG ? wh(prob, xG) : NaN,
        okD ? wh(prob, xD) : NaN)
    @printf("  finite      %12s %12s %12s\n", okA, okG, okD)
end
println("Done.")
