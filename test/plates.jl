# Kirchhoff thin plate: Navier w_max on SS square.
using Test
using LinearAlgebra
using StaticArrays
using BEM
using BEM.Plate

@testset "SS square Navier w_max" begin
    E, ν, h, a, q0 = 1e5, 0.3, 0.01, 1.0, 1.0
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    D = bending_stiffness(props)
    w_ana = analytical_wmax_ss_square(; a=a, q=q0, D=D)
    @test abs(w_ana * D / (q0 * a^4) - 0.004062) < 5e-5
    mesh = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    @test mesh isa BEMdata{<:BEM.ThinPlate}
    @test mesh.elements[1] isa Element
    @test !isempty(mesh.elements[1].geo)
    @test length(mesh.elements[1].index) == 3
    assemble!(mesh; npg=10, threaded=false)
    solve(mesh)
    rel = abs(plate_w_int(mesh, 1) - w_ana) / w_ana
    @test rel < 0.10

    mesh_l = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    mesh_f = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    assemble!(mesh_l; npg=10, near_factor=1.5, threaded=false)
    assemble!(mesh_f; npg=10, near_factor=Inf, threaded=false)
    solve(mesh_l)
    solve(mesh_f)
    @test abs(plate_w_int(mesh_l, 1) - plate_w_int(mesh_f, 1)) / w_ana < 0.05
    @test abs(plate_w_int(mesh_l, 1) - w_ana) / w_ana < 0.10
end

@testset "CCCC linear: analytic and Guiggiani vs Levy 0.001263" begin
    E, ν, h, a, q0 = 1e6, 0.316, 0.01, 1.0, 17.79 * 1e6 * 0.01^4
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    D = bending_stiffness(props)
    w_ana = 0.001263 * q0 * a^4 / D
    for sing in (:auto, :guiggiani)
        dad = build_square_plate(; a=a, n_el=6, bc="CCCC", props=props,
            corner_bc='C', n_internal=1)
        assemble_plate!(dad; npg=10, threaded=false, singular=sing)
        solve_plate!(dad)
        rel = abs(plate_w_int(dad, 1) - w_ana) / w_ana
        @test rel < 0.02
    end
end

@testset "anisotropic Kirchhoff Useche 7.5.1" begin
    # Repeated μ (isotropic D) must not use the aniso kernels.
    D = 1.0
    ν = 0.3
    @test_throws ErrorException aniso_thin_plate_props(;
        D11=D, D22=D, D12=ν * D, D66=(1 - ν) * D / 2)

    Ex, Ey, νxy, Gxy, h, a, q = 2.068e11, 2.068e11 / 15, 0.3, 6.055e8, 0.01, 1.0, 1e4
    den = 1 - νxy^2 * Ey / Ex
    D11 = Ex * h^3 / (12 * den)
    D22 = Ey * h^3 / (12 * den)
    D12 = νxy * Ey * h^3 / (12 * den)
    D66 = Gxy * h^3 / 12
    wA = navier_w_ss_ortho(a / 2, a / 2; a=a, q=q, D11=D11, D22=D22, D12=D12, D66=D66)
    @test abs(wA - 8.1258e-3) / 8.1258e-3 < 0.01

    props = aniso_thin_plate_props(; D11=D11, D22=D22, D12=D12, D66=D66, q_c=q, h=h)
    @test props.D16 == 0 && props.D26 == 0
    @test props.e[1] > 0 && props.e[2] > 0
    pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
    nh = SVector(1.0, 0.0)
    U, P = plate_kernels(pg, pf, nh, nh, props)
    @test all(isfinite, U) && all(isfinite, P)

    mesh = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    assemble_plate!(mesh; npg=10, singular=:guiggiani)
    solve_plate!(mesh)
    wc = plate_w_int(mesh, 1)
    @test isfinite(wc)
    @test abs(wc - wA) / abs(wA) < 0.05
end

@testset "Kirchhoff DIBEM remainder M 1_w = ID" begin
    E, ν, h, a, q0 = 1e5, 0.3, 0.01, 1.0, 1.0
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    dad = build_square_plate(; a=a, n_el=4, bc="SSSS", props=props,
        corner_bc='F', n_internal=4)
    assemble_plate!(dad; npg=8, threaded=false)
    M = dibem_plate!(dad; npg=8, rbf=PHS(2; poly_deg=1), threaded=false)
    @test has_cache(dad, :M)
    @test all(isfinite, M)
    ndof = size(dad.H, 1)
    @test size(M) == (ndof, ndof)
    e = zeros(ndof)
    @inbounds for j in 1:dad.nt
        e[2j - 1] = 1
    end
    nc = length(dad.plate_corners)
    @inbounds for c in 1:nc
        e[2 * dad.nt + c] = 1
    end
    ID = dad.dibem_ID
    @test norm(M * e - ID) / (norm(ID) + 1e-14) < 1e-8
    Mw = plate_Mw(dad)
    @test size(Mw) == (ndof, dad.nt)
    q_dib = Mw * fill(q0, dad.nt)
    @test all(isfinite, q_dib)
    @test norm(q_dib) > 0

    props_a = aniso_thin_plate_props(; D11=1.0, D22=0.5, D12=0.1, D66=0.2,
        q_c=q0, h=h)
    dada = build_square_plate(; a=a, n_el=3, bc="SSSS", props=props_a,
        corner_bc='F', n_internal=1)
    assemble_plate!(dada; npg=8, threaded=false)
    Ma = dibem_plate!(dada; npg=8, rbf=PHS(2; poly_deg=1), threaded=false)
    @test all(isfinite, Ma)
    ea = zeros(size(Ma, 1))
    @inbounds for j in 1:dada.nt
        ea[2j - 1] = 1
    end
    nca = length(dada.plate_corners)
    @inbounds for c in 1:nca
        ea[2 * dada.nt + c] = 1
    end
    @test norm(Ma * ea - dada.dibem_ID) / (norm(dada.dibem_ID) + 1e-14) < 1e-8
end

@testset "FSDT IBP Mx 1 + Γ(e_x) = 0" begin
    props = FSDTProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0, ρ=1.0)
    dad = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=4)
    assemble_fsdt!(dad; npg=8, nsub=4)
    dibem_fsdt!(dad; npg=8, rbf=PHS(2; poly_deg=1))
    @test has_cache(dad, :Mx)
    Mx, My = dad.Mx, dad.My
    nt = dad.nt
    γx = BEM.Plate.ibp_gamma_vn(dad.G, dad.Normal, ones(nt), zeros(nt), 3)
    γy = BEM.Plate.ibp_gamma_vn(dad.G, dad.Normal, zeros(nt), ones(nt), 3)
    e = ones(nt)
    @test norm(Mx * e .+ γx) / (norm(γx) + 1e-14) < 1e-8
    @test norm(My * e .+ γy) / (norm(γy) + 1e-14) < 1e-8
    r0 = BEM.Plate.ibp_div_Uv(Mx, My, dad.G, dad.Normal, e, zero(e), 3)
    @test norm(r0) / (norm(γx) + 1e-14) < 1e-8
end

@testset "membrane t_L = -N_vk·n on SSSS1 free DOFs" begin
    E2, a, h = 1.0, 1.0, 0.1
    plies = [(25.0, E2, 0.25, 0.5, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=0.5, G23=0.2)
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=0.5, G23=0.2, q_c=1.0, ρ=1.0)
    mesh = build_square_fsdt(; a=a, n_el=3, bc="SSSS", props=props, n_internal=4)
    n = mesh.n
    Nxx, Nyy, Nxy = ones(n + 4), zeros(n + 4), zeros(n + 4)
    sh_c = LaminatedShell(mesh, A, FlatShell(); mem_bc=:clamped)
    t_c = BEM.Plate._membrane_t_from_Nvk(sh_c, Nxx, Nyy, Nxy)
    @test t_c == zeros(2n)
    sh_n = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
    t_n = BEM.Plate._membrane_t_from_Nvk(sh_n, Nxx, Nyy, Nxy)
    @test any(!iszero, t_n)
    # traction-known u-slot gets -Nxx nx
    nrm = mesh.Normal
    for i in 1:n
        if sh_n.BCm[2i - 1] == 1
            @test t_n[2i - 1] ≈ -(Nxx[i] * nrm[i][1] + Nxy[i] * nrm[i][2])
        else
            @test t_n[2i - 1] == 0
        end
    end
    Nxxp, Nyyp, Nxyp = copy(Nxx), copy(Nyy), copy(Nxy)
    BEM.Plate._project_Nnn_free!(sh_n, Nxxp, Nyyp, Nxyp)
    for i in 1:n
        nx, ny = nrm[i][1], nrm[i][2]
        free_n = abs(nx) >= abs(ny) ? sh_n.BCm[2i - 1] == 1 : sh_n.BCm[2i] == 1
        Nn = Nxxp[i] * nx * nx + Nyyp[i] * ny * ny + 2 * Nxyp[i] * nx * ny
        if free_n
            @test abs(Nn) < 1e-14
        end
    end
    Nxxc, Nyyc, Nxyc = copy(Nxx), copy(Nyy), copy(Nxy)
    BEM.Plate._project_Nnn_free!(sh_c, Nxxc, Nyyc, Nxyc)
    @test Nxxc == Nxx && Nyyc == Nyy && Nxyc == Nxy
end

@testset "FSDT SS square vs Kirchhoff Navier (thin)" begin
    E, ν, h, a, q0 = 1e5, 0.3, 0.01, 1.0, 1.0
    Dk = bending_stiffness(ThinPlateProps(; E=E, ν=ν, h=h))
    wK = analytical_wmax_ss_square(; a=a, q=q0, D=Dk)
    props = FSDTProps(; E=E, ν=ν, h=h, q_c=q0, ρ=1.0)
    mesh = build_square_fsdt(; a=a, n_el=5, bc="SSSS", props=props, n_internal=1)
    @test mesh isa BEMdata{<:BEM.FSDT}
    assemble_fsdt!(mesh; npg=8, nsub=6)
    dibem_fsdt!(mesh)
    solve_fsdt!(mesh)
    wc = abs(fsdt_w_int(mesh, 1))
    @test isfinite(wc)
    @test abs(wc - wK) / wK < 0.15
    κGh = shear_stiffness(props)
    wF = navier_w_ss_fsdt(a / 2, a / 2; a=a, q=q0, D=Dk, ν=ν, κGh=κGh)
    @test abs(wc - wF) / wF < 0.20
end

@testset "FSDT Houbolt DIBEM (centroids)" begin
    pd = FSDTProps(; E=200e3, ν=0.3, h=0.1, q_c=1e3, ρ=0.7853)
    mesh = build_square_fsdt(; a=2.0, n_el=4, bc="SSSS", props=pd, n_internal=9)
    @test mesh isa BEMdata{<:BEM.FSDT}
    @test length(mesh.internalNodes) == 9
    @test hypot(mesh.internalNodes[1][1] - 1.0, mesh.internalNodes[1][2] - 1.0) < 1e-12
    assemble_fsdt!(mesh; npg=8, nsub=6)
    dibem_fsdt!(mesh)
    solve_fsdt!(mesh)
    wstat = abs(fsdt_w_int(mesh, 1))
    @test isfinite(wstat) && wstat > 0
    res = solve_fsdt_houbolt!(mesh; dt=5e-3, tmax=0.15)
    @test all(isfinite, res.w_center)
    peak = maximum(abs.(res.w_center))
    @test 0.7 < peak / (2 * wstat) < 1.3
end

@testset "FSDT 8.5 resultants and 8.6.2 Houbolt laminate" begin
    E1, E2, ν12 = 4e6, 2e6, 0.25
    plies = [(E1, E2, ν12, 1e6, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, ρ=4000.0, nθ=8)
    gold = navier_ss_fsdt_MQ(0.5, 0.5, props; a=1.0, q=1.0)
    @test gold.w > 0 && isfinite(gold.Mx)
    mesh = build_square_fsdt(; a=1.0, n_el=4, bc="SSSS", props=props, n_internal=9)
    assemble_fsdt!(mesh; npg=8, nsub=6)
    dibem_fsdt!(mesh)
    solve_fsdt!(mesh)
    wc = fsdt_w_int(mesh, 1)
    @test abs(wc - gold.w) / abs(gold.w) < 0.25
    rq = fsdt_resultants(mesh)
    ic = length(mesh.Nodes) + 1
    @test all(isfinite, rq.Mx) && all(isfinite, rq.Qx)
    @test abs(rq.Mx[ic] - gold.Mx) / max(abs(gold.Mx), 1e-12) < 0.50
    res = solve_fsdt_houbolt!(mesh; dt=5e-3, tmax=0.35, mass=:raw)
    @test all(isfinite, res.w_center)
    peak = maximum(abs.(res.w_center))
    @test 1.3 < peak / abs(wc) < 2.6
end

@testset "Wang laminate kernels" begin
    # Octave KernelP.m gold (isotropic D, AT from FSDTProps E=1e5 ν=0.3 h=0.05)
    p = FSDTProps(; E=1e5, ν=0.3, h=0.05)
    lp = LaminateFSDTProps(p; nθ=12)
    pg = SVector(0.4, 0.2)
    pf = SVector(0.0, 0.0)
    nh = SVector(1.0, 0.0)
    Uw, Pw, _ = wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=12, map=:telles)
    UM = [-0.020706 -0.027410 -0.013844
          -0.027410  0.020409 -0.006922
           0.013844  0.006922 -0.006937]
    PM = [-0.2518  0.0317  0.4777
          -0.0161 -0.0665  0.6367
          -0.0169 -0.0223 -0.3183]
    @test maximum(abs.(Uw .- UM)) / maximum(abs.(UM)) < 1e-3
    @test maximum(abs.(Pw .- PM)) / maximum(abs.(PM)) < 1e-3
    Us, Ps, _ = wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=16, map=:sinh, sinh_b=1e-4)
    @test maximum(abs.(Us .- Uw)) / maximum(abs.(Uw)) < 0.05

    plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
    lam = laminate_fsdt_props(plies; Ks=5 / 6, G13=1e6, G23=5e5)
    @test abs(lam.D[1, 1] - 322.5806) / 322.5806 < 1e-5
    @test abs(lam.D[2, 2] - 193.5484) / 193.5484 < 1e-5
    @test abs(lam.AT[1, 1] - 62500) < 1.0
    @test abs(lam.AT[2, 2] - 62500) < 1.0
    wN = navier_w_ss_fsdt(0.5, 0.5, lam; a=1.0, q=1.0)
    @test isfinite(wN) && wN > 0
end

@testset "unsymmetric FSDT Hsu–Hwu kernels (8.3.1)" begin
    # tiny-B Hsu vs Wang: F(ρ) scale (not the OCR of 8.38–8.42)
    E, ν, h = 1e5, 0.3, 0.05
    D = E * h^3 / (12 * (1 - ν^2))
    Gsh = E / (2 * (1 + ν))
    A11 = E * h / (1 - ν^2)
    A = @SMatrix [A11 ν*A11 0; ν*A11 A11 0; 0 0 Gsh*h]
    Dmat = @SMatrix [D ν*D 0; ν*D D 0; 0 0 (1-ν)*D/2]
    AT = @SMatrix [5/6*Gsh*h 0; 0 5/6*Gsh*h]
    pH = UnsymFSDTProps(A, 1e-8 * Dmat, Dmat, AT; h=h, nθ=12)
    pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
    nh = SVector(1.0, 0.0)
    UW, PW, _ = wang_kernels(pg, pf, nh, Dmat, AT; nθ=12)
    UH, PH = unsym_fsdt_kernels(pg, pf, nh, pH)
    @test abs(UH[5, 5] / UW[3, 3] - 1) < 0.05
    @test abs(PH[5, 5] / PW[3, 3] - 1) < 0.05
    @test maximum(abs.(UH[3:5, 3:5] .- UW)) / maximum(abs, UW) < 0.05
    @test maximum(abs.(PH[3:5, 3:5] .- PW)) / maximum(abs, PW) < 0.05

    # two-ply [0/90] has B ≠ 0 (no iso smear)
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
    # Hsu EABE 156: S = L_{nξ}(T*, ∇_ξ T*), not nξ·∇P
    hfd = 1e-5
    _, P0, _, _, _, _ = BEM.Plate.unsym_fsdt_dkernels(pg, pf, nh, p)
    _, Pp = unsym_fsdt_kernels(pg, pf + SVector(hfd, 0.0), nh, p)
    _, Pm = unsym_fsdt_kernels(pg, pf - SVector(hfd, 0.0), nh, p)
    Px = (Pp - Pm) / (2 * hfd)
    _, Pp = unsym_fsdt_kernels(pg, pf + SVector(0.0, hfd), nh, p)
    _, Pm = unsym_fsdt_kernels(pg, pf - SVector(0.0, hfd), nh, p)
    Py = (Pp - Pm) / (2 * hfd)
    Sfd = BEM.Plate._hsu_L_mat(P0, Px, Py, nξ, p)
    @test maximum(abs.(S .- Sfd)) / maximum(abs, Sfd) < 0.25

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
    @test maximum(abs.(Fp .- (Fh - Fm) / 2e-6)) < 1e-4

    p.nθ = 8
    p.q_c = 1.0
    wN = navier_w_ss_unsym(0.5, 0.5, p; a=1.0, q=1.0)
    @test isfinite(wN) && wN > 0
    mesh = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=p, n_internal=9)
    @test mesh isa BEMdata{<:UnsymFSDTProps}
    @test n_dof(mesh) == 5
    @test length(mesh.BC) == 5 * mesh.n
    assemble_fsdt!(mesh; npg=6, nsub=4)
    @test size(mesh.H, 1) == 5 * (mesh.n + 9)
    @test all(isfinite, mesh.H) && all(isfinite, mesh.G)
    dibem_fsdt!(mesh; npg=6)
    @test all(isfinite, mesh.M) && all(isfinite, mesh.q)
    @test norm(mesh.q) > 0
    solve_fsdt!(mesh)
    wc = fsdt_w_int(mesh, 1)
    @test isfinite(wc) && wc * wN > 0
    @test abs(wc - wN) / abs(wN) < 0.25

    # Von Kármán: membrane ½∇w⊗∇w + ∇·(N∇w) on the same 5-DOF BEM.
    h = p.h
    λvk = min(1.0, 0.4 * h / max(abs(wc), eps()))
    resvk = solve_fsdt!(mesh; large=true, nsteps=2, λ_max=λvk,
        nonlinear=:picard, maxiters=4)
    @test all(isfinite, resvk.w_center)
    @test abs(resvk.w_center[end]) < 5 * λvk * abs(wc) + 1e-12
    @test abs(resvk.w_center[end]) > 0.05 * λvk * abs(wc)

    # Guiggiani CBIE on the same mesh (self-element CPV)
    p.nθ = 8
    meshG = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=p, n_internal=9)
    assemble_fsdt!(meshG; npg=6, nsub=4, singular=:guiggiani, ninterp=12)
    dibem_fsdt!(meshG; npg=6)
    solve_fsdt!(meshG)
    wcG = fsdt_w_int(meshG, 1)
    @test isfinite(wcG) && wcG * wN > 0
    @test abs(wcG - wN) / abs(wN) < 0.25
end

@testset "unsymmetric FSDT HBIE Guiggiani vs CBIE" begin
    plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
    p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
    wN = navier_w_ss_unsym(0.5, 0.5, p; a=1.0, q=1.0)
    meshC = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=p, n_internal=1)
    assemble_fsdt!(meshC; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
    dibem_fsdt!(meshC; npg=4)
    solve_fsdt!(meshC)
    wcC = fsdt_w_int(meshC, 1)
    @test isfinite(wcC) && wcC * wN > 0
    @test abs(wcC - wN) / abs(wN) < 0.25
    meshH = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=p, n_internal=1)
    assemble_unsym_fsdt_hbie!(meshH; npg=4, nsub=4, ninterp=12)
    @test all(==(3), meshH.eq_type)
    solve_fsdt!(meshH)
    wcH = fsdt_w_int(meshH, 1)
    @test isfinite(wcH) && all(isfinite, meshH.H)
    @test wcH * wN > 0
end

@testset "unsym interior HBIE moments vs FD" begin
    plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
    p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
    mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=p, n_internal=1)
    assemble_fsdt!(mesh; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
    dibem_fsdt!(mesh; npg=4)
    solve_fsdt!(mesh)
    pf, nx = SVector(0.5, 0.5), SVector(1.0, 0.0)
    tH = unsym_interior_t(mesh, pf, nx; npg=4, nsub=4)
    tF, aux = unsym_interior_t_fd(mesh, pf, nx; h=1e-3, npg=4, nsub=4)
    @test all(isfinite, tH) && all(isfinite, tF)
    @test isfinite(aux.M[1]) && aux.M[1] != 0
    @test tF[3] ≈ aux.M[1] rtol = 1e-10
    @test tH[3] ≈ tF[3] rtol = 0.05
    @test abs(tH[5]) + abs(tF[5]) < 1e-2 * max(abs(aux.M[1]), 1e-30)
end

@testset "FSDT Maxima Fρ vs Gauss" begin
    p = FSDTProps(; E=1e5, ν=0.3, h=0.05)
    R = 0.05
    n1, n2 = 0.8, 0.6
    n1, n2 = n1 / hypot(n1, n2), n2 / hypot(n1, n2)
    D, ν, λ = bending_stiffness(p), p.ν, reissner_lambda(p)
    F = BEM.Plate._reissner_Fρ(R, n1, n2, D, ν, λ)
    xs, ws = gausslegendre(24)
    acc = zeros(3, 3)
    pf = SVector(0.0, 0.0)
    for (g, w) in zip(xs, ws)
        ξ, Jt = BEM.Plate._telles(g, -1.0)
        ρ = (ξ + 1) / 2 * R
        ρ < 1e-16 && continue
        pg = pf + ρ * SVector(n1, n2)
        U, _, _ = fsdt_kernels(p, pg, pf, SVector(1.0, 0.0))
        acc .+= Matrix(U) .* (ρ * (R / 2) * w * Jt)
    end
    @test maximum(abs.(Matrix(F) .- acc)) / maximum(abs.(acc)) < 1e-6

    lp = LaminateFSDTProps(p; nθ=10)
    pgR = pf + R * SVector(n1, n2)
    Fw = fsdt_Fρ(lp, pgR, pf)
    accw = zeros(3, 3)
    for (g, w) in zip(xs, ws)
        ξ, Jt = BEM.Plate._telles(g, -1.0)
        ρ = (ξ + 1) / 2 * R
        ρ < 1e-16 && continue
        pg = pf + ρ * SVector(n1, n2)
        U, _, _ = wang_kernels(pg, pf, SVector(1.0, 0.0), lp.D, lp.AT; nθ=10)
        accw .+= Matrix(U) .* (ρ * (R / 2) * w * Jt)
    end
    @test maximum(abs.(Matrix(Fw) .- accw)) / maximum(abs.(accw)) < 1e-6
end

@testset "laminated shell DIBEM vs 5-DOF Navier (9.6.1)" begin
    E1, E2, ν12 = 25.0, 1.0, 0.25
    G12 = 0.5 * E2
    a, h, R, q = 1.0, 0.01, 1.0, 1.0
    κ = 1 / R
    plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=1.0)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    gold = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κ, κ2=κ, A=A, D=D, As=As)
    mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=81)
    shell = LaminatedShell(mesh, A, κ, κ; mem_bc=:navier_ss)
    assemble_laminated_shell!(shell; npg=8, nsub=6)
    solve_laminated_shell!(shell)
    nv = shell_navier_centre(shell)
    wc = nv.w_bem
    @test isfinite(wc) && wc > 0
    # Donnell-on-w is ~20× too stiff; 16 BE + 81 DIBEM centres ~ MATLAB 98 RIM.
    @test wc > 30.0
    @test nv.relerr < 0.15
    @test abs(nv.w - gold.w) / abs(gold.w) < 1e-10
    # Von Kármán only after linear Navier matches, at λ so λ w_lin ~ h/2.
    h = props.h
    λvk = min(1.0, 0.5 * h / max(abs(wc), eps()))
    resvk = solve_laminated_shell!(shell; large=true, nsteps=2, λ_max=λvk,
        nonlinear=:picard, maxiters=4)
    @test all(isfinite, resvk.w_center)
    @test abs(resvk.w_center[end]) < 5 * λvk * abs(wc) + 1e-12
    # Fig. 9.3: N, M along y=a/2 from SS-1 DST of interior w (not raw RBF ∇u).
    rq = shell_resultants(shell; method=:ss1)
    ic = shell.plate.n + 1
    @test abs(rq.Nx[ic] - gold.Nx) / max(abs(gold.Nx), 1e-12) < 0.25
    @test abs(rq.Ny[ic] - gold.Ny) / max(abs(gold.Ny), 1e-12) < 0.25
    @test abs(rq.Mx[ic] - gold.Mx) / max(abs(gold.Mx), 1e-12) < 0.25
    cl = shell_centreline(shell; dir=:x)
    @test length(cl.s) >= 5
    eN = eM = 0.0
    nmid = 0
    for i in eachindex(cl.s)
        abs(cl.s[i] - a / 2) > 0.2 && continue
        g = navier_ss_laminate_shell(cl.pts[i][1], cl.pts[i][2];
            a=a, q=q, κ1=κ, κ2=κ, A=A, D=D, As=As)
        eN += abs(cl.Nx[i] - g.Nx) / max(abs(g.Nx), 1e-12)
        eM += abs(cl.Mx[i] - g.Mx) / max(abs(g.Mx), 1e-12)
        nmid += 1
    end
    @test nmid >= 1
    @test eN / nmid < 0.30
    @test eM / nmid < 0.35
end

@testset "laminated cylindrical shell DIBEM vs 5-DOF Navier (7.7.1)" begin
    E1, E2, ν12 = 25.0e9, 1.0e9, 0.25
    G12 = 0.5 * E2
    a, h, R, q = 10.0, 1.0, 50.0, 1.0
    κ1, κ2 = 1 / R, 0.0
    angs = [90.0, 0.0, 0.0, 90.0, 90.0, 0.0, 0.0, 90.0]
    plies = [(E1, E2, ν12, G12, θ, h / 8) for θ in angs]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=1.0)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    gold = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κ1, κ2=κ2, A=A, D=D, As=As)
    mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=81)
    shell = LaminatedShell(mesh, A, κ1, κ2; mem_bc=:navier_ss)
    assemble_laminated_shell!(shell; npg=8, nsub=6)
    solve_laminated_shell!(shell)
    wc = fsdt_w_int(shell.plate, 1)
    @test isfinite(wc) && wc > 0
    # Donnell-on-w misses u,v relief; cylinder a/R=0.2 is only ~1% below flat.
    @test wc > 0.4 * gold.w
    @test abs(wc - gold.w) / abs(gold.w) < 0.30
end

@testset "Ch.9 laminated shell Houbolt DIBEM" begin
    E1, E2, ν12 = 25.0, 1.0, 0.25
    G12 = 0.5 * E2
    a, h, R, q, ρ = 1.0, 0.01, 1.0, 1.0, 1.0
    κ = 1 / R
    plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=ρ)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    T11 = navier_ss_laminate_T11(; a=a, κ1=κ, κ2=κ, A=A, D=D, As=As, ρ=ρ, h=h)
    @test isfinite(T11) && 4.0 < T11 < 6.0
    mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=25)
    shell = LaminatedShell(mesh, A, κ, κ; mem_bc=:navier_ss)
    assemble_laminated_shell!(shell; npg=6, nsub=4)
    solve_laminated_shell!(shell)
    wstat = abs(fsdt_w_int(shell.plate, 1))
    @test wstat > 0
    res = solve_laminated_shell_houbolt!(shell; dt=T11 / 30, tmax=0.65 * T11, mass=:raw)
    @test all(isfinite, res.w_center)
    peak = maximum(abs.(res.w_center))
    @test 1.3 < peak / wstat < 2.6
end

@testset "shell geometry κ and von Kármán" begin
    E1, E2, ν12 = 25.0, 1.0, 0.25
    G12 = 0.5 * E2
    a, h, R, q, ρ = 1.0, 0.01, 1.0, 1.0, 1.0
    plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=ρ)
    A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=9)
    sph = SphericalShell(R)
    cyl = CylindricalShell(R; hoop=:x)
    @test curvature_at(sph, SVector(0.5, 0.5)) == (1.0, 1.0, 0.0)
    @test curvature_at(cyl, SVector(0.5, 0.5)) == (1.0, 0.0, 0.0)
    @test curvature_at(FlatShell(), SVector(0.5, 0.5)) == (0.0, 0.0, 0.0)
    zg = HeightGraph((x, y) -> x^2 / (2R))
    κg = curvature_at(zg, SVector(0.5, 0.5))
    @test abs(κg[1] - 1 / R) < 1e-10 && abs(κg[2]) < 1e-10
    shellκ = LaminatedShell(mesh, A, 1 / R, 1 / R; mem_bc=:navier_ss)
    shellG = LaminatedShell(mesh, A, sph; mem_bc=:navier_ss)
    @test shellG.κ1 ≈ shellκ.κ1 && shellG.κ2 ≈ shellκ.κ2
    shellC = LaminatedShell(mesh, A, cyl; mem_bc=:navier_ss)
    @test shellC.κ1 ≈ 1 / R && abs(shellC.κ2) < 1e-12
    shellZ = LaminatedShell(mesh, A, zg; mem_bc=:navier_ss)
    @test shellZ.κ1 ≈ 1 / R && abs(shellZ.κ2) < 1e-8
    assemble_laminated_shell!(shellG; npg=6, nsub=4)
    solve_laminated_shell!(shellG)
    wlin = abs(fsdt_w_int(shellG.plate, 1))
    @test isfinite(wlin) && wlin > 0
    # n_el=2 / 9 internals is too coarse for Navier (that is 9.6.1). VK here
    # only checks a finite moderate-w/h Picard on this discrete operator.
    λvk = min(0.25, 0.5 * h / max(wlin, eps()))
    res = solve_laminated_shell!(shellG; large=true, nsteps=2, λ_max=λvk,
        maxiters=4, nonlinear=:picard)
    @test all(isfinite, res.w_center)
    @test length(res.λ) == 2
    @test res.λ[end] ≈ λvk
    @test abs(res.w_center[end]) > 0
end

@testset "shell Crisfield arc-length" begin
    # Solver smoke on a cheap mesh. Not a Navier / Sabir–Lock accuracy test
    # (9.6.1 uses 16 BE + 81 centres).
    E1, E2, ν12 = 25.0, 1.0, 0.25
    G12 = 0.5 * E2
    a, h, q, ρ = 1.0, 0.02, 1.0, 1.0
    rise = 2h
    R = a^2 / (8 * rise)
    plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=ρ)
    A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    mesh = build_square_fsdt(; a=a, n_el=2, bc="SSSS", props=props, n_internal=4)
    shell = LaminatedShell(mesh, A, SphericalShell(R); mem_bc=:clamped)
    assemble_laminated_shell!(shell; npg=4, nsub=3)
    res = solve_laminated_shell!(shell; large=true, nonlinear=:arclength,
        nsteps=12, λ_max=0.0, maxiters=8, atol=1e-4)
    @test length(res.λ) >= 3
    @test all(isfinite, res.λ) && all(isfinite, res.w_center)
    @test res.λ[1] == 0
    @test maximum(abs.(res.w_center)) > 0
    @test maximum(abs.(res.λ)) > 0
    @test maximum(abs.(res.w_center[2:end])) > abs(res.w_center[2])
    # Discrete load limit of this operator (not continuum Navier).
    @test argmax(res.λ) < length(res.λ)
    @test any(<(0), diff(res.λ))

    # Inward curvature: crown-w control traces past the soft point (dλ dip).
    mesh2 = build_square_fsdt(; a=a, n_el=2, bc="SSSS", props=props, n_internal=4)
    shell2 = LaminatedShell(mesh2, A, SphericalShell(-R); mem_bc=:clamped)
    assemble_laminated_shell!(shell2; npg=4, nsub=3)
    resw = solve_laminated_shell!(shell2; large=true, nonlinear=:wcontrol,
        nsteps=10, w_max=4h, maxiters=8, atol=1e-4)
    @test all(isfinite, resw.λ) && all(isfinite, resw.w_center)
    @test length(resw.λ) >= 4
    @test resw.w_center[end] > resw.w_center[2]
    dλ = diff(resw.λ)
    @test 1 < argmin(dλ) < length(dλ)
end

@testset "Reissner Dual BEM centre crack (10.5.1)" begin
    E, ν, h = 2.1e5, 0.3, 0.5
    props = FSDTProps(; E=E, ν=ν, h=h)
    pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
    n = SVector(1.0, 0.0)
    nξ = SVector(0.0, 1.0)
    D = bending_stiffness(props)
    λ = reissner_lambda(props)
    Wk, Pk = reissner_hbie_kernels(pg, pf, n, nξ, D, ν, λ)
    @test size(Wk) == (3, 3) && size(Pk) == (3, 3)
    @test all(isfinite, Wk) && all(isfinite, Pk)
    @test maximum(abs, Pk) > 0

    mesh = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=8, nome="test_1051")
    @test mesh isa BEMdata{<:BEM.FSDT}
    @test n_dof(mesh) == 3
    @test count(==(2), mesh.eq_type) > 0 && count(==(3), mesh.eq_type) > 0
    @test any(!=(0), mesh.twin)
    assemble_fsdt_dual!(mesh; npg=8, nsub=6)
    @test all(isfinite, mesh.H) && all(isfinite, mesh.G)
    solve_fsdt!(mesh)
    @test all(isfinite, mesh.u)
    K1b, K2b, K3b, rA, rB, Le = sif_ctod_fsdt(mesh; tip=:right)
    @test all(isfinite, (K1b, K2b, K3b, rA, rB, Le))
    @test isapprox(rB / Le, 0.5; atol=0.02)          # GL mid node, geometric tip
    @test isapprox(rA / Le, 0.5 * (1 + sqrt(3 / 5)); atol=0.02)  # GL far, not 5/6
    F = K1b / sqrt(π * 0.2)
    @test 0.3 < F < 2.5
end

@testset "unsym Hsu T* Dual BEM centre crack" begin
    plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
    props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=6)
    mesh = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=3, ndiv_h=3, ndiv_crack=4, nome="test_unsym_dual")
    @test mesh isa BEMdata{<:UnsymFSDTProps}
    @test n_dof(mesh) == 5
    @test count(==(2), mesh.eq_type) > 0 && count(==(3), mesh.eq_type) > 0
    @test any(!iszero, mesh.twin)
    assemble_fsdt_dual!(mesh; npg=4, nsub=4, ninterp=8)
    @test all(isfinite, mesh.H) && all(isfinite, mesh.G)
    @test size(mesh.H, 1) == 5 * mesh.n
    solve_fsdt!(mesh)
    @test all(isfinite, mesh.u)
    iA = findfirst(==(2), mesh.eq_type)
    tw = mesh.twin[iA]
    Δw = mesh.u[5iA] - mesh.u[5tw]
    @test isfinite(Δw) && Δw != 0
end

@testset "Reissner XBEM Hui–Zehnder (Dolbow 36)" begin
    E, ν, h = 2.1e5, 0.3, 0.5
    ρ = 0.01
    Mp = hui_zehnder_M_local(E, ν, h, ρ, π)
    Mm = hui_zehnder_M_local(E, ν, h, ρ, -π)
    @test size(Mp) == (3, 3)
    @test all(isfinite, Mp) && all(isfinite, Mm)
    Δψ2 = Mp[2, 1] - Mm[2, 1]
    @test Δψ2 ≈ 48 * sqrt(2ρ) / (E * h^3) rtol = 1e-10
    Crot = E * h^3 / 48 * sqrt(π / 2)
    K1b = Crot * Δψ2 / sqrt(ρ)
    @test K1b ≈ sqrt(π) rtol = 1e-8          # Dirgantara K1b = √π K1_dolbow
    Δw3 = Mp[3, 3] - Mm[3, 3]
    @test Δw3 > 0
    props = FSDTProps(; E=E, ν=ν, h=h)
    mesh = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=8, nome="test_1051_xbem")
    assemble_fsdt_dual!(mesh; npg=8, nsub=6)
    _, K1, K2, K3, tips, _ = solve_fsdt_xbem!(mesh; n_enr=2, npg=8, nsub=6, n_v=3)
    @test length(tips) == 2 && length(K1) == 2
    @test all(isfinite, K1) && all(isfinite, K2) && all(isfinite, K3)
    ir = argmax(p[1] for p in tips)
    Fx = abs(K1[ir]) / sqrt(0.2)             # extra-DOF Dolbow F = K1/(Mo√a)
    @test 0.3 < Fx < 3.0
    K1b, _, _, _, _, _ = sif_ctod_fsdt(mesh; tip=:right)
    F = K1b / sqrt(π * 0.2)
    @test 0.3 < F < 2.5
end

@testset "Dolbow 5.2 angled centre-crack mesh" begin
    props = FSDTProps(; E=2e5, ν=0.3, h=1.0)
    mesh = build_rect_fsdt_crack(; W=5.0, H=5.0, a=0.5, α=π / 4, props=props, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=6, nome="test_dolbow52")
    tips = BEM.Plate._fsdt_geometric_tips(mesh)
    @test length(tips) == 2
    @test any(p -> abs(p[1] - 0.5 / sqrt(2)) < 0.05 && abs(p[2] - 0.5 / sqrt(2)) < 0.05, tips)
    @test count(==(2), mesh.eq_type) > 0 && count(==(3), mesh.eq_type) > 0
end

@testset "Kirchhoff Dual (iso + aniso)" begin
    E, ν, h = 2.1e5, 0.3, 0.5
    props = ThinPlateProps(; E=E, ν=ν, h=h)
    pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
    n = SVector(1.0, 0.0)
    nξ = SVector(0.0, 1.0)
    Uh, Ph = plate_hbie_kernels(pg, pf, n, nξ, props)
    @test size(Uh) == (2, 2) && size(Ph) == (2, 2)
    @test all(isfinite, Uh) && all(isfinite, Ph)
    @test maximum(abs, Ph) > 0

    D22 = E * h^3 / (12 * (1 - ν^2))
    props_a = aniso_thin_plate_props(; D11=2 * D22, D22=D22, D12=0.3 * D22,
        D66=0.4 * D22, h=h)
    Ua, Pa = plate_hbie_kernels(pg, pf, n, nξ, props_a)
    @test all(isfinite, Ua) && all(isfinite, Pa)

    mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=4, nome="test_k_1051")
    @test mesh isa BEMdata{<:AbstractThinPlate}
    @test count(==(2), mesh.eq_type) > 0 && count(==(3), mesh.eq_type) > 0
    @test any(!=(0), mesh.twin)
    assemble!(mesh; npg=6, nsub=4)
    @test all(isfinite, mesh.H) && all(isfinite, mesh.G)
    solve(mesh)
    @test all(isfinite, mesh.u)
    K1, K2, rA, rB, Le = sif_ctod_plate(mesh; tip=:right, method=:band)
    @test all(isfinite, (K1, K2, rA, rB, Le))
    F = abs(K1) / sqrt(0.2)
    @test isfinite(F)
    # Dual must open (Sih F=1). Mix w+Mn clamped F~0.005; traction Dual
    # of Useche 10.2 is the Portela analogue.
    @test 0.15 < F < 6.0
    K1t, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:tip)
    @test isfinite(K1t)

    mesh_a = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props_a, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=4, nome="test_k_aniso")
    assemble_plate_dual!(mesh_a; npg=6, nsub=4)
    @test all(isfinite, mesh_a.H)
    solve_plate!(mesh_a)
    @test all(isfinite, mesh_a.u)
end
