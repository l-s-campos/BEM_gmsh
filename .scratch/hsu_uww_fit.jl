using BEM, BEM.Plate, LinearAlgebra, Printf, StaticArrays, FastGaussQuadrature
const Plate = BEM.Plate

function kβ(pg, pf, n, p; β=-1.0)
    RX, RY = pg[1]-pf[1], pg[2]-pf[2]
    At = Plate._hsu_At(p.AT)
    nθ = p.nθ
    eg, wg0 = gausslegendre(nθ)
    Uast = zeros(5, 5)
    F = zeros(10, 10)
    θ0 = atan(-RX, RY)
    for iq in 1:4
        eet = (iq == 1 || iq == 3) ? -1.0 : 1.0
        et, wg = Plate._cluster_rule(eg, wg0, eet, p.map; b=p.sinh_b)
        for i in 1:nθ
            ξ = clamp(et[i], -1.0, 1.0)
            θ = θ0 + (ξ + 1) * π / 4
            Ω, ω = Plate._unsym_Ωω(θ)
            ρ = ω[1]*RX + ω[2]*RY
            abs(ρ) < 1e-14 && continue
            got = Plate._unsym_Z_L2(θ, p.A, p.B, p.D, At)
            got === nothing && continue
            Z, L2, λd = got
            cond(Z) > 1e12 && continue
            Plate._unsym_F!(F, ρ; β=β)
            Plate._unsym_Fd!(F, ρ, λd)
            Yθ = (Z * (F / Z))[:, 6:10] * inv(L2)
            Uast .+= Yθ[1:5, :] .* (wg[i] * (π / 4))
        end
        θ0 += π / 2
    end
    return Uast ./ (4 * π^2)
end

E, ν, h = 1e5, 0.3, 0.05
D = E * h^3 / (12 * (1 - ν^2))
Gsh = E / (2 * (1 + ν))
A11 = E * h / (1 - ν^2)
A = @SMatrix [A11 ν*A11 0; ν*A11 A11 0; 0 0 Gsh*h]
Dmat = @SMatrix [D ν*D 0; ν*D D 0; 0 0 (1-ν)*D/2]
AT = @SMatrix [5/6*Gsh*h 0; 0 5/6*Gsh*h]
pH = UnsymFSDTProps(A, 1e-6*Dmat, Dmat, AT; h=h, nθ=16)
pR = FSDTProps(; E=E, ν=ν, h=h)
pf = SVector(0.0, 0.0)
nh = SVector(1.0, 0.0)

Rs = [0.08, 0.12, 0.18, 0.28, 0.40, 0.55, 0.75, 1.00, 1.30]
function collectU(β)
    UH = Float64[]; UW = Float64[]; UR = Float64[]
    for R in Rs
        pg = SVector(R, 0.0)
        Uh = kβ(pg, pf, nh, pH; β=β)
        Uw, _, _ = wang_kernels(pg, pf, nh, Dmat, AT; nθ=16)
        Ur, _, _ = fsdt_kernels(pg, pf, nh, D, ν, reissner_lambda(pR))
        push!(UH, Uh[5,5]); push!(UW, Uw[3,3]); push!(UR, Ur[3,3])
    end
    return UH, UW, UR
end
UH, UW, UR = collectU(-1.0)
println("β=-1")
println("     R         UwwH         UwwW         UwwR     H/W")
for (R,h,w,r) in zip(Rs,UH,UW,UR)
    @printf("%8.3f %12.4e %12.4e %12.4e %7.3f\n", R, h, w, r, h/w)
end

# fit a R²lnR + b R² + c lnR + d
function fit4(R, U)
    X = hcat((R .^ 2) .* log.(R), R .^ 2, log.(R), ones(length(R)))
    return X \ U
end
cH, cW, cR = fit4(Rs, UH), fit4(Rs, UW), fit4(Rs, UR)
println("\ncoeffs  [R²lnR, R², lnR, 1]  β=-1")
@printf("Hsu    %12.4e %12.4e %12.4e %12.4e\n", cH...)
@printf("Wang   %12.4e %12.4e %12.4e %12.4e\n", cW...)
@printf("Reiss  %12.4e %12.4e %12.4e %12.4e\n", cR...)

UH0, _, _ = collectU(0.0)
c0 = fit4(Rs, UH0)
println("\nHsu β=0 R² coeff $(c0[2])  β=-1 $(cH[2])  Wang $(cW[2])")
# b(β) = b(0) + β*(b(-1)-b(0))/(-1)
# want b(β)=cW[2]
db = cH[2] - c0[2]          # change from β=0 to β=-1
# b(β) = c0[2] + β*(cH[2]-c0[2])/(-1) = c0[2] - β*db
βstar = (c0[2] - cW[2]) / db
@printf("β* for Wang R² = %.6f\n", βstar)

UHs, _, _ = collectU(βstar)
cs = fit4(Rs, UHs)
@printf("Hsu β* coeffs %12.4e %12.4e %12.4e %12.4e\n", cs...)
println("     R      UwwH*        UwwW     H/W")
for (R,h,w) in zip(Rs, UHs, UW)
    @printf("%8.3f %12.4e %12.4e %7.4f\n", R, h, w, h/w)
end
@printf("theory r²lnr /8πD = %.4e\n", 1/(8*π*D))

