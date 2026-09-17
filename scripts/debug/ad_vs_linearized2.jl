# Jacobian norms at the linear mixed solve (finite). Reuses same mesh as part 1.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random
using BEM.Plate
using .ThinPlate

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
q0 = 40 * E * h^4
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
plate = build_square_plate(; a=a, n_el=2, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=6, singular=:analytic)
include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=3, show=false, nome="ad_lin_pe2", Lx=a, Ly=a)
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0); fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=6, threaded=false)
prob = build_large_plate_problem(plate, dad_pe; npg_plate=6, npg_pe=6)
nA = norm(prob.A_pl) + eps()
n = length(plate.nodes)

function geo_weight(prob)
    D = bending_stiffness(prob.plate.props)
    xs = getindex.(prob.pts, 1); ys = getindex.(prob.pts, 2)
    Lref = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
    return (Lref^2) / (8 * π * D + eps())
end

function lin_jacobian(prob, x)
    u = BEM.Plate.pack_plate_u(prob, x)
    w = BEM.Plate.extract_w_field(prob, u)
    Nxx_nl, Nyy_nl, Nxy_nl = BEM.Plate.membrane_N_from_w(prob, w)
    Nxx, Nyy, Nxy = BEM.Plate.solve_membrane_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
    wt = geo_weight(prob)
    Fxx = prob.Fx * prob.Fx; Fyy = prob.Fy * prob.Fy; Fxy = prob.Fx * prob.Fy
    nw = length(w)
    L = zeros(nw, nw)
    for j in 1:nw
        ej = zeros(nw); ej[j] = 1
        L[:, j] .= Nxx .* (Fxx * ej) .+ 2 .* Nxy .* (Fxy * ej) .+ Nyy .* (Fyy * ej)
    end
    J = copy(prob.A_pl)
    for (kj, dofj) in enumerate(prob.w_index)
        prob.is_kin[dofj] && continue
        for (ki, dofi) in enumerate(prob.w_index)
            prob.is_kin[dofi] && continue
            J[dofi, dofj] -= wt * L[ki, kj]
        end
    end
    return J
end

Random.seed!(1)
@printf("%6s %8s %10s %10s %10s %10s %10s %10s %10s\n",
    "Q", "w/h", "||Jad-A||", "||JG-A||", "||Jad-G||", "fracG", "FD-AD", "FD-G", "FD-A")
for Q in (5.0, 10.0, 20.0, 40.0)
    λ = Q / 40
    x = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
    u = BEM.Plate.pack_plate_u(prob, x)
    wh = abs(u[2n+1]) / h
    Jad = BEM.Plate._ad_jacobian(prob, x, λ)
    JG = lin_jacobian(prob, x)
    dAD = norm(Jad - prob.A_pl)
    dG = norm(JG - prob.A_pl)
    dAG = norm(Jad - JG)
    R = BEM.Plate._residual_xλ(prob, x, λ)
    s = randn(length(x)); s ./= norm(s)
    ε = 1e-6 * (1 + norm(x))
    dR = (BEM.Plate._residual_xλ(prob, x .+ ε .* s, λ) .- R) ./ ε
    ndR = norm(dR) + eps()
    @printf("%6.0f %8.4f %10.3e %10.3e %10.3e %10.3f %10.3e %10.3e %10.3e\n",
        Q, wh, dAD / nA, dG / nA, dAG / nA,
        dG / (dAD + eps()),
        norm(dR - Jad * s) / ndR,
        norm(dR - JG * s) / ndR,
        norm(dR - prob.A_pl * s) / ndR)
end
println("Done.")
