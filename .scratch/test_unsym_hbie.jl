using Test, LinearAlgebra, StaticArrays, BEM, BEM.Plate

@testset "unsymmetric FSDT Hsu–Hwu kernels (8.3.1)" begin
    E, ν, h = 1e5, 0.3, 0.05
    D = E * h^3 / (12 * (1 - ν^2))
    Gsh = E / (2 * (1 + ν))
    A11 = E * h / (1 - ν^2)
    A = @SMatrix [A11 ν * A11 0; ν * A11 A11 0; 0 0 Gsh * h]
    Dmat = @SMatrix [D ν * D 0; ν * D D 0; 0 0 (1 - ν) * D / 2]
    AT = @SMatrix [5 / 6 * Gsh * h 0; 0 5 / 6 * Gsh * h]
    pH = UnsymFSDTProps(A, 1e-8 * Dmat, Dmat, AT; h=h, nθ=12)
    pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
    nh = SVector(1.0, 0.0)
    UW, PW, _ = wang_kernels(pg, pf, nh, Dmat, AT; nθ=12)
    UH, PH = unsym_fsdt_kernels(pg, pf, nh, pH)
    @test abs(UH[5, 5] / UW[3, 3] - 1) < 0.05
    @test abs(PH[5, 5] / PW[3, 3] - 1) < 0.05
    @test maximum(abs.(UH[3:5, 3:5] .- UW)) / maximum(abs, UW) < 0.05
    @test maximum(abs.(PH[3:5, 3:5] .- PW)) / maximum(abs, PW) < 0.05

    plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
    p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=10)
    @test maximum(abs, p.B) > 1.0
    U, P = unsym_fsdt_kernels(pg, pf, nh, p)
    @test size(U) == (5, 5) && size(P) == (5, 5)
    @test all(isfinite, U) && all(isfinite, P)
    @test maximum(abs, U) > 0
    U2, _ = unsym_fsdt_kernels(SVector(0.8, 0.4), pf, nh, p)
    @test all(isfinite, U2)

    nξ = SVector(0.0, 1.0)
    W, S = unsym_hbie_kernels(pg, pf, nh, nξ, p)
    @test size(W) == (5, 5) && size(S) == (5, 5)
    @test all(isfinite, W) && all(isfinite, S)
    @test maximum(abs, S) > maximum(abs, P)
    _, Wswap = unsym_fsdt_kernels(pf, pg, nξ, p)
    @test W ≈ Wswap rtol = 1e-10
    hfd = 2e-5
    _, Pp = unsym_fsdt_kernels(pg, pf + hfd * nξ, nh, p)
    _, Pm = unsym_fsdt_kernels(pg, pf - hfd * nξ, nh, p)
    Sfd = (Pp - Pm) / (2 * hfd)
    rel = maximum(abs.(S .- Sfd)) / maximum(abs, Sfd)
    println("S vs FD rel = ", rel)
    println("max|S| = ", maximum(abs, S), "  max|Sfd| = ", maximum(abs, Sfd))
    @test rel < 0.05

    F = zeros(10, 10)
    Fp = zeros(10, 10)
    ρ = 0.3
    BEM.Plate._unsym_F!(F, ρ)
    BEM.Plate._unsym_Fp!(Fp, ρ)
    Fh = zeros(10, 10)
    BEM.Plate._unsym_F!(Fh, ρ + 1e-6)
    Fm = zeros(10, 10)
    BEM.Plate._unsym_F!(Fm, ρ - 1e-6)
    @test maximum(abs.(Fp .- (Fh - Fm) / 2e-6)) < 1e-4
    λd = 2.5
    BEM.Plate._unsym_Fd!(F, ρ, λd)
    BEM.Plate._unsym_Fdp!(Fp, ρ, λd)
    BEM.Plate._unsym_Fd!(Fh, ρ + 1e-6, λd)
    BEM.Plate._unsym_Fd!(Fm, ρ - 1e-6, λd)
    errEi = maximum(abs.(Fp .- (Fh - Fm) / 2e-6))
    println("F' (Jordan+Ei) vs FD maxabs = ", errEi)
    @test errEi < 1e-4
end
