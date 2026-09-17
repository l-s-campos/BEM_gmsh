# Steady Laplace: dense T=x and hierarchical T=x.
using Test
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using BEM
using BEM.HMatrices
using BEM.Topology: interior_grad_T

const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset "square Laplace dense T=x" begin
    dad = format2d(quadrado(ndiv=10, show=false, nome="t_lap_d"), Laplace(1.0);
        pontointerno=true)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    assemble!(dad, 12)
    solve(dad)
    @test rel_error(dad) < 0.05
end

@testset "square Laplace mixed HSS+ULV" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_lap_ulv"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    assemble!(dad, 12)
    solve(dad; blocks=true, factor=:ulv)
    @test dad.block_ulv isa BlockULV
    @test rel_error(dad) < 0.05
end

function _interior_T(dad, pf)
    n = dad.n
    h = zeros(n)
    g = zeros(n)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        nn = length(el.index)
        hloc = zeros(nn)
        gloc = zeros(nn)
        integrate_element(dad, el, xj, pf, hloc, gloc)
        for (a, j) in enumerate(el.index)
            h[j] += hloc[a]
            g[j] += gloc[a]
        end
    end
    c = -sum(h)
    return (dot(g, view(dad.q, 1:n)) - dot(h, view(dad.T, 1:n))) / c
end

@testset "near-field maps: interior point → boundary" begin
    # Finer than the 10-div patch test. Approach y=0 at x=0.5 (T_ana=0.5).
    dad = format2d(quadrado(ndiv=20, show=false, nome="t_nf_approach"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    set_cache!(dad; nearfield=:sinhsinh)
    assemble!(dad, 16; threaded=false)
    solve(dad)
    @test rel_error(dad) < 0.03
    maps = (:euclid, :csinh, :sinhsinh, :tangent, :p3c, :tanp3c)
    ds = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    errT = Dict(nf => Float64[] for nf in (maps..., :plain))
    errG = Dict(nf => Float64[] for nf in (maps..., :plain))
    gana = SA[1.0, 0.0]
    for d in ds
        pf = Point2D(0.5, d)
        for nf in (maps..., :plain)
            set_cache!(dad; nearfield=nf)
            push!(errT[nf], abs(_interior_T(dad, pf) - 0.5))
            push!(errG[nf], norm(interior_grad_T(dad, [pf])[1] - gana))
        end
    end
    for nf in maps
        # T: transformed maps stay at mesh error as d/L → 0 (L=1/20)
        @test maximum(errT[nf]) < 5e-3
    end
    # ∇T ~ 1/r²: tangent family holds; plain Gauss blows by d=1e-3
    @test errG[:tangent][4] < 0.05    # d=1e-4
    @test errG[:tanp3c][4] < 0.05
    @test errG[:sinhsinh][3] < 0.05   # d=1e-3
    @test errG[:plain][3] > 1.0
end

@testset "square Laplace H-matrix T=x" begin
    dad = format2d(quadrado(ndiv=10, show=false, nome="t_lap_h"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    assemble!(dad; method=:hmatrix, atol=1e-5, nmax=16, threads=false)
    @test dad.H isa ColWeightedOp
    solve(dad)
    @test rel_error(dad) < 0.08
    @test compression_ratio(dad.H.K) > 0
end

@testset "square Laplace H² T=x" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_lap_h2"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    H_G_Hmat(dad; format=:H2, atol=1e-5, rtol=1e-5, nmax=16, threads=false)
    @test dad.H isa ColWeightedOp
    @test dad.H.K isa NNCAMatrix
    solve(dad)
    @test rel_error(dad) < 0.15
end

@testset "cube 3D Laplace T=z" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_cube(; L=1.0, ndiv=2, nome="t_lap3d")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0]))
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 1e-4
    @test rel_error_flux(dad) < 1e-4
end

@testset "cube 3D interior approach" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_cube(; L=1.0, ndiv=2, nome="t_lap3d_near")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0]))
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 1e-3
    function _intT(dad, pf)
        n = dad.n
        h = zeros(n); g = zeros(n)
        @inbounds for el in dad.elements
            xj = dad.Nodes[el.index]
            nn = length(el.index)
            hloc = zeros(nn); gloc = zeros(nn)
            integrate_element(dad, el, xj, pf, hloc, gloc)
            for (a, j) in enumerate(el.index)
                h[j] += hloc[a]; g[j] += gloc[a]
            end
        end
        c = -sum(h)
        return (dot(g, view(dad.q, 1:n)) - dot(h, view(dad.T, 1:n))) / c
    end
    # Face-interior (not a mesh vertex): z=d, T_ana=d.
    d = 1e-3
    pf = Point3D(0.25, 0.25, d)
    set_cache!(dad; nearfield=:plain)
    eplain = abs(_intT(dad, pf) - d)
    set_cache!(dad; nearfield=:tensor)
    etens = abs(_intT(dad, pf) - d)
    set_cache!(dad; nearfield=:polar)
    epol = abs(_intT(dad, pf) - d)
    set_cache!(dad; nearfield=:auto)
    eauto = abs(_intT(dad, pf) - d)
    @test etens < 5e-3
    @test epol < 5e-3
    @test eauto < 5e-3
    @test epol < eplain
    set_cache!(dad; nearfield=:dibem)
    edib = abs(_intT(dad, pf) - d)
    @test edib < 5e-3
end

# ---------------------------------------------------------------------------
# Anisotropic Laplace (strategy 2 FS + strategy 1 DIBEM) vs FEniCS plate
# ---------------------------------------------------------------------------

function _right_edge_T(dad; x0=1.0, atol=0.03, zmid=nothing)
    ys = Float64[]; Ts = Float64[]
    for i in 1:dad.n
        p = dad.Nodes[i]
        abs(p[1] - x0) < atol || continue
        if zmid !== nothing && abs(p[3] - zmid) > atol
            continue
        end
        push!(ys, p[2]); push!(Ts, dad.T[i])
    end
    isempty(ys) && return ys, Ts
    perm = sortperm(ys)
    return ys[perm], Ts[perm]
end

function _interp_lin(x, y, xq)
    n = length(x)
    out = similar(xq)
    @inbounds for (k, z) in enumerate(xq)
        if z <= x[1]
            out[k] = y[1]
        elseif z >= x[end]
            out[k] = y[end]
        else
            j = searchsortedlast(x, z)
            t = (z - x[j]) / (x[j+1] - x[j] + eps())
            out[k] = (1 - t) * y[j] + t * y[j+1]
        end
    end
    return out
end

function _rms_right_vs_fenics(dad; csv=joinpath(@__DIR__, "..", "data", "Laplace",
        "fenics_ortho_plate_Tright.csv"), zmid=nothing)
    ys, Ts = _right_edge_T(dad; zmid=zmid)
    data = readdlm(csv, ',')
    # skip header if present
    i0 = data[1, 1] isa AbstractString ? 2 : 1
    yref = Float64.(data[i0:end, 1])
    Tref = Float64.(data[i0:end, 2])
    Tb = _interp_lin(ys, Ts, yref)
    return norm(Tb .- Tref) / (norm(Tref) + eps())
end

@testset "anisotropic Laplace FS isotropic limit" begin
    props = AnisotropicLaplace(@SMatrix [1.0 0; 0 1.0])
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_ani_iso"), props;
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    assemble!(dad, 12)
    solve(dad)
    @test rel_error(dad) < 1e-4
end

@testset "anisotropic DIBEM K=I matches Laplace" begin
    msh = quadrado(ndiv=8, show=false, nome="t_ani_dibem_I", ordem=2)
    dad0 = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(dad0; npg=10, threaded=false)
    solve(dad0)
    dad1 = format2d(quadrado(ndiv=8, show=false, nome="t_ani_dibem_Ib", ordem=2), Laplace(1.0);
        pontointerno=true, tipo=2)
    assemble!(dad1; npg=10, threaded=false)
    DIBEM(dad1; rbf=PHS(3; poly_deg=1))
    solve_anisotropic_dibem!(dad1, I(2); rbf=PHS(3; poly_deg=1))
    @test median(abs.(dad1.T[1:dad1.n] .- dad0.T[1:dad0.n])) < 0.05
end

@testset "ortho plate FS vs FEniCS" begin
    K = @SMatrix [5.0 0; 0 0.5]
    msh = placa_furo_orto(; lc=0.06, nome="t_orto_fs", show=false, ordem=2)
    dad = format2d(msh, AnisotropicLaplace(K); pontointerno=false, tipo=2)
    assemble!(dad; npg=10, threaded=false)
    solve(dad)
    @test _rms_right_vs_fenics(dad) < 0.12
end

@testset "anisotropic DIBEM no-hole T=x" begin
    K = @SMatrix [5.0 0; 0 0.5]
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_s1_Tx", ordem=2), Laplace(1.0);
        pontointerno=true, tipo=2)
    ana = AnalyticalSolution("Tx", (x; t=0) -> x[1];
        q=(x, n; t=0) -> -dot(n, K * SVector(1.0, 0.0)), description="")
    neu = [i for i in 1:dad.n if dad.Nodes[i][1] > 1e-8]
    apply_analytical_bc!(dad, ana, neu)
    assemble!(dad; npg=10, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    solve_anisotropic_dibem!(dad, K; rbf=PHS(3; poly_deg=1))
    @test rel_error(dad) < 0.05
end

@testset "anisotropic wave shift IBP K=I is a no-op" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_wave_ibp_I", ordem=2), Laplace(1.0);
        pontointerno=true, tipo=2)
    assemble!(dad; npg=8, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    H0, M0 = copy(dad.H), copy(dad.M)
    anisotropic_wave_shift!(dad, I(2); strategy=:ibp, rbf=PHS(3; poly_deg=1))
    @test dad.H ≈ H0
    @test dad.M ≈ M0
    @test dad.aniso_strategy === :ibp
end

@testset "anisotropic IBP K=I matches Laplace" begin
    msh = quadrado(ndiv=8, show=false, nome="t_ani_ibp_I", ordem=2)
    dad0 = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(dad0; npg=10, threaded=false)
    solve(dad0)
    dad3 = format2d(quadrado(ndiv=8, show=false, nome="t_ani_ibp_Ib", ordem=2), Laplace(1.0);
        pontointerno=true, tipo=2)
    assemble!(dad3; npg=10, threaded=false)
    DIBEM(dad3; rbf=PHS(3; poly_deg=1))
    solve_anisotropic_ibp!(dad3, I(2); rbf=PHS(3; poly_deg=1))
    @test median(abs.(dad3.T[1:dad3.n] .- dad0.T[1:dad0.n])) < 0.05
end

@testset "anisotropic IBP no-hole T=x" begin
    K = @SMatrix [5.0 0; 0 0.5]
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_s3_Tx", ordem=2), Laplace(1.0);
        pontointerno=true, tipo=2)
    ana = AnalyticalSolution("Tx", (x; t=0) -> x[1];
        q=(x, n; t=0) -> -dot(n, K * SVector(1.0, 0.0)), description="")
    neu = [i for i in 1:dad.n if dad.Nodes[i][1] > 1e-8]
    apply_analytical_bc!(dad, ana, neu)
    assemble!(dad; npg=10, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    solve_anisotropic_ibp!(dad, K; rbf=PHS(3; poly_deg=1))
    @test rel_error(dad) < 0.05
end

@testset "anisotropic DIBEM plate vs FEniCS" begin
    K = @SMatrix [5.0 0; 0 0.5]
    dad = format2d(placa_furo_orto(; lc=0.06, nome="t_orto_s1", show=false, ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(dad; npg=10, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    solve_anisotropic_dibem!(dad, K; rbf=PHS(3; poly_deg=1))
    @test _rms_right_vs_fenics(dad) < 0.15
end

@testset "anisotropic quadratic known solution" begin
    K = @SMatrix [5.0 0; 0 0.5]
    ana = ana_aniso_quadratic(K)
    rbf = PHS(3; poly_deg=2)
    function _interior_rel(dad)
        ni = dad.nt - dad.n
        ni == 0 && return 0.0
        Ti = [ana.u(p) for p in dad.internalNodes]
        return norm(dad.T[dad.n+1:dad.nt] .- Ti) / (norm(Ti) + eps())
    end
    dadfs = format2d(quadrado(ndiv=8, show=false, nome="t_aq_fs", ordem=2),
        AnisotropicLaplace(K); pontointerno=true, tipo=2)
    apply_analytical_bc!(dadfs, ana)
    assemble!(dadfs; npg=12, threaded=false)
    solve(dadfs)
    @test _interior_rel(dadfs) < 0.01
    dad1 = format2d(quadrado(ndiv=8, show=false, nome="t_aq_s1", ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    apply_analytical_bc!(dad1, ana)
    assemble!(dad1; npg=12, threaded=false)
    DIBEM(dad1; rbf=rbf)
    solve_anisotropic_dibem!(dad1, K; rbf=rbf)
    @test _interior_rel(dad1) < 0.02
    dad3 = format2d(quadrado(ndiv=8, show=false, nome="t_aq_s3", ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    apply_analytical_bc!(dad3, ana)
    assemble!(dad3; npg=12, threaded=false)
    DIBEM(dad3; rbf=rbf)
    solve_anisotropic_ibp!(dad3, K; rbf=rbf)
    @test _interior_rel(dad3) < 0.05
end

@testset "anisotropic Poisson sin known solution" begin
    K = @SMatrix [5.0 0; 0 0.5]
    ana, bsrc = ana_aniso_poisson_sin(K)
    rbf = PHS(3; poly_deg=2)
    function _interior_rel(dad)
        Ti = [ana.u(p) for p in dad.internalNodes]
        return norm(dad.T[dad.n+1:dad.nt] .- Ti) / (norm(Ti) + eps())
    end
    dad1 = format2d(quadrado(ndiv=8, show=false, nome="t_ap_s1", ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    apply_analytical_bc!(dad1, ana)
    assemble!(dad1; npg=12, threaded=false)
    DIBEM(dad1; rbf=rbf)
    solve_anisotropic_dibem!(dad1, K; b=bsrc, rbf=rbf)
    @test _interior_rel(dad1) < 0.08
    dad3 = format2d(quadrado(ndiv=8, show=false, nome="t_ap_s3", ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    apply_analytical_bc!(dad3, ana)
    assemble!(dad3; npg=12, threaded=false)
    DIBEM(dad3; rbf=rbf)
    solve_anisotropic_ibp!(dad3, K; b=bsrc, rbf=rbf)
    @test _interior_rel(dad3) < 0.08
end

@testset "anisotropic Laplace 3D isotropic cube" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = AnisotropicLaplace(Matrix{Float64}(I, 3, 3))
    msh = mesh_cube(; L=1.0, ndiv=2, nome="t_ani3d_iso")
    dad = format3d(msh, props; pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0]))
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 1e-3
end

@testset "anisotropic Laplace 3D FS T=z" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    K = @SMatrix [5.0 0 0; 0 0.5 0; 0 0 1.0]
    props = AnisotropicLaplace(K)
    msh = mesh_cube(; L=1.0, ndiv=1, nome="t_ani3d_Tz")
    dad = format3d(msh, props; pontointerno=false)
    ana = AnalyticalSolution("Tz", (x; t=0) -> x[3];
        q=(x, n; t=0) -> -dot(n, K * SVector(0.0, 0.0, 1.0)), description="")
    neu = [i for i in 1:dad.n if abs(dad.Normal[i][3]) < 0.5]
    apply_analytical_bc!(dad, ana, neu)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 0.02
end

@testset "format3d interior centroids" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    dad = format3d(mesh_cube(; L=1.0, ndiv=2, nome="t_fmt3d_ni"), Laplace(1.0);
        pontointerno=true)
    @test dad.ni > 0
    @test all(p -> all(0.0 .< p .< 1.0), dad.internalNodes)
    dad0 = format3d(mesh_cube(; L=1.0, ndiv=2, nome="t_fmt3d_n0"), Laplace(1.0);
        pontointerno=false)
    @test dad0.ni == 0
    dadt = format3d(mesh_cube(; L=1.0, ndiv=2, nome="t_fmt3d_tri", recombine=false),
        Laplace(1.0); pontointerno=true)
    @test dadt.ni > 0
    @test all(p -> all(0.0 .< p .< 1.0), dadt.internalNodes)
end

@testset "square Laplace KA CPU kernel T=x" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_lap_gpu_cpu"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    dad_ref = format2d(quadrado(ndiv=8, show=false, nome="t_lap_gpu_ref"), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad_ref, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    assemble!(dad_ref, 12; threaded=false)
    H_G_gpu(dad; T=Float64, npg=12, near=:cpu, device=:cpu, threaded=false)
    @test eltype(dad.H) === Float64
    @test norm(dad.H - dad_ref.H) / norm(dad_ref.H) < 1e-10
    @test norm(dad.G - dad_ref.G) / norm(dad_ref.G) < 1e-10
    solve(dad)
    @test rel_error(dad) < 0.05
end

@testset "assemble! method=:gpu without CUDA throws" begin
    dad = format2d(quadrado(ndiv=4, show=false, nome="t_lap_gpu_throw"), Laplace(1.0);
        pontointerno=false)
    if !haskey(Base.loaded_modules, Base.PkgId(
            Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        @test_throws ArgumentError assemble!(dad; method=:gpu, npg=8)
    else
        @test_nowarn gpu_float_support()
    end
end

@testset "square Laplace CUDA T=x" begin
    if !HAVE_CUDA
        @info "CUDA not functional; skipping GPU Laplace assembly test"
    else
        info = gpu_float_support()
        @test info.available
        @test info.Float32
        @test info.Float64
        dad = format2d(quadrado(ndiv=8, show=false, nome="t_lap_gpu"), Laplace(1.0);
            pontointerno=false)
        attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
        dad_ref = format2d(quadrado(ndiv=8, show=false, nome="t_lap_gpu_d"), Laplace(1.0);
            pontointerno=false)
        attach_analytical!(dad_ref, ana_laplace_linear(; direction=SA[1.0, 0.0]))
        assemble!(dad_ref, 12; threaded=false)
        assemble!(dad; method=:gpu, T=Float64, npg=12, near=:cpu, threaded=false)
        @test norm(dad.H - dad_ref.H) / norm(dad_ref.H) < 1e-10
        solve(dad)
        @test rel_error(dad) < 0.05
        dad32 = format2d(quadrado(ndiv=8, show=false, nome="t_lap_gpu32"), Laplace(1.0);
            pontointerno=false)
        attach_analytical!(dad32, ana_laplace_linear(; direction=SA[1.0, 0.0]))
        assemble!(dad32; method=:gpu, T=Float32, npg=12, near=:cpu, threaded=false)
        @test norm(Float64.(dad32.H) - dad_ref.H) / norm(dad_ref.H) < 1e-6
        @test norm(Float64.(dad32.G) - dad_ref.G) / norm(dad_ref.G) < 1e-6
        solve(dad32)
        @test rel_error(dad32) < 0.05
    end
end

@testset "anisotropic DIBEM/IBP 3D cube" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    K = @SMatrix [5.0 0 0; 0 0.5 0; 0 0 1.0]
    rbf = PHS(3; poly_deg=2)
    ana = AnalyticalSolution("Tz", (x; t=0) -> x[3];
        q=(x, n; t=0) -> -dot(n, K * SVector(0.0, 0.0, 1.0)), description="")
    dad1 = format3d(mesh_cube(; L=1.0, ndiv=1, nome="t_s1_3d"), Laplace(1.0);
        pontointerno=true)
    neu1 = [i for i in 1:dad1.n if abs(dad1.Normal[i][3]) < 0.5]
    apply_analytical_bc!(dad1, ana, neu1)
    assemble!(dad1; npg=6, threaded=false)
    DIBEM(dad1; rbf=rbf)
    solve_anisotropic_dibem!(dad1, K; rbf=rbf)
    @test dad1.ni > 0
    @test rel_error(dad1) < 0.05
    dad3 = format3d(mesh_cube(; L=1.0, ndiv=1, nome="t_s3_3d"), Laplace(1.0);
        pontointerno=true)
    neu3 = [i for i in 1:dad3.n if abs(dad3.Normal[i][3]) < 0.5]
    apply_analytical_bc!(dad3, ana, neu3)
    assemble!(dad3; npg=6, threaded=false)
    DIBEM(dad3; rbf=rbf)
    solve_anisotropic_ibp!(dad3, K; rbf=rbf)
    @test rel_error(dad3) < 0.08
end

@testset "Guiggiani films and 1-D Reynolds" begin
    a = 2.0
    films = guiggiani_films(; a=a, hi=a)
    @test films.h1.h(0.0) ≈ a
    @test films.h1.h(1.0) ≈ 1 atol=1e-12
    @test films.h2.h(0.0) ≈ a
    @test films.h2.h(1.0) ≈ 1 atol=1e-12
    @test films.h4.h(0.0) ≈ a
    @test films.h4.h(1.0) ≈ 1 atol=1e-12
    @test films.linear.h(0.0) ≈ a
    @test films.linear.h(1.0) ≈ 1
    xs = 0.1:0.1:0.9
    @test all(films.h1.h(x) > films.h2.h(x) > films.h3.h(x) >
              films.h4.h(x) > films.h5.h(x) for x in xs)
    @test films.h2.k == 0
    @test films.h1.k > 0
    @test films.h5.k < 0
    @test abs(films.h3.h(0.5) - (a + 1) / 2) < 1e-8
    ξ = 0.4
    p_gl = infinite_bearing_pressure(films.linear, ξ)
    p_cf = linear_wedge_pressure(ξ, a)
    @test abs(p_gl - p_cf) / abs(p_cf) < 1e-5
    pmax_lin = maximum(infinite_bearing_pressure(films.linear, x) for x in 0.05:0.05:0.95)
    @test abs(pmax_lin - 0.25) < 0.01
end

@testset "Guiggiani pad Laplace+DIBEM h2" begin
    film = film_h2(; a=2.0, hi=2.0)
    msh = mesh_guiggiani_pad(; nome="t_guiggiani_pad", show=false)
    dad = format2d(msh, Laplace(1.0); pontointerno=false, tipo=2)
    @test length(dad.elements) == 16
    internal_grid!(dad, 9, 5; d_min=0.02, layout=:cell)
    apply_ambient_pressure!(dad)
    assemble!(dad; npg=10, threaded=false)
    dad_pi = deepcopy(dad)
    solve_reynolds_dibem!(dad, film; npg=10, rbf=PHS(3; poly_deg=1))
    solve_reynolds_particular!(dad_pi, film; npg=10)
    p_inf = maximum(infinite_bearing_pressure(film, x) for x in 0.1:0.05:0.9)
    pint = dad.T[(dad.n + 1):end]
    @test !isempty(pint)
    @test all(p -> p > -1e-4, pint)
    @test maximum(pint) > 0.02
    @test maximum(pint) < p_inf
    pnd = reynolds_pressure.(pint, Ref(film))
    @test maximum(pnd) < 0.25
    rel = norm(dad.T - dad_pi.T) / (norm(dad_pi.T) + 1e-14)
    @test rel < 0.25
end

@testset "Elrod–Adams 1-D linear wedge (no cavitation)" begin
    a = 2.0
    L = 1e-2
    ho = 1e-6
    hi = a * ho
    μ = 0.01
    U = 1.0
    n = 201
    x = collect(range(0.0, L; length=n))
    h = hi .- (hi - ho) .* (x ./ L)
    dx = x[2] - x[1]
    rheo = ConstRheology(; ρ=850.0, μ=μ)
    p, θ = solve_elrod_1d(h, dx; U=U, rheo=rheo, pleft=0.0, pright=0.0,
        opt=ElrodOptions(; pcav=-1e6, maxiter=8_000, tol=1e-10))
    @test all(θ .> 0.999)
    pexact = [linear_wedge_pressure(ξ, a; μ=μ, U=U, L=L, ho=ho) for ξ in x ./ L]
    @test maximum(p) > 0
    @test abs(maximum(p) - maximum(pexact)) / maximum(pexact) < 0.08
end

