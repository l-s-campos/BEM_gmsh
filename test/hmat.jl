# H-matrix matvec vs dense log kernel.
using Test
using LinearAlgebra
using SparseArrays
using Random
using StaticArrays
using BEM
using BEM.HMatrices

@testset "H matvec vs dense" begin
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        return d < 1e-14 ? 0.0 : log(d)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    x = randn(length(pts))
    rel = norm(H * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test rel < 5e-4
    @test compression_ratio(H) > 0
end

@testset "geometric quadtree IL (2D uniform)" begin
    xs = range(-1.0, 1.0; length=16)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    neigh, il, id2 = HMatrices.neighbor_il_lists(tree)
    @test !isempty(leaves(tree))
    # interior-ish leaf: some neighbors, and IL at depth ≥ 2
    n_touch = 0
    n_il = 0
    d2 = 0
    for L in leaves(tree)
        d = HMatrices.depth(L)
        d < 2 && continue
        d2 += 1
        id = node_id(L)
        n_touch = max(n_touch, length(neigh[id]))
        n_il = max(n_il, length(il[id]))
        @test length(neigh[id]) <= 8
    end
    @test d2 > 0
    @test n_touch >= 3
    @test n_il >= 1
end

@testset "H-matrix quadtree LU / matvec" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do x, y
        d = hypot(x[1] - y[1], x[2] - y[2])
        return d < 1e-14 ? 2.0 : log(d)
    end
    tree = ClusterTree(pts, hmatrix_splitter(; nmax=12); cube=true)
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    x = randn(length(pts))
    rel = norm(H * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test rel < 5e-3
    Kd = Matrix(K)
    Kd = Kd + Kd' + 5.0 * I
    tree2 = ClusterTree(pts, hmatrix_splitter(; nmax=16); cube=true)
    Hs = assemble_hmatrix(Kd, tree2, tree2;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-8),
        threads=false)
    b = randn(length(pts))
    F = lu(Hs; threads=false)
    u = F \ copy(b)
    @test norm(Hs * u - b) / (norm(b) + 1e-14) < 5e-3
end

@testset "NNCA 2D vs dense" begin
    Random.seed!(1)
    xs = range(-1.0, 1.0; length=12)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do x, y
        d = hypot(x[1] - y[1], x[2] - y[2])
        return d < 1e-14 ? 0.0 : log(d)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H = assemble_h2(K, tree; rtol=1e-6, threads=false)
    x = randn(length(pts))
    rel = norm(H * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test rel < 5e-3
    @test maxrank(H) < 80
    Hc = assemble_h2(K, tree; method=:cheb, order=8, threads=false, rtol=0)
    relc = norm(Hc * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test relc < 1e-5
    Hr = assemble_h2(K, tree; method=:cheb, order=8, threads=false, rtol=1e-6)
    relr = norm(Hr * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test relr < 5e-5
    @test maxrank(Hr) <= maxrank(Hc)
end

@testset "rectangular NNCA vs dense" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    X = [SVector(float(x), float(y)) for y in xs for x in xs]
    Y = [SVector(float(t), 0.0) for t in range(0.0, 1.0; length=12)]
    K = KernelMatrix(X, Y) do a, b
        d = hypot(a[1] - b[1], a[2] - b[2])
        return d < 1e-14 ? 0.0 : log(d)
    end
    tx = ClusterTree(X, DyadicSplitter(; nmax=12, tight=false); cube=true)
    ty = ClusterTree(Y, DyadicSplitter(; nmax=8, tight=false); cube=true,
        container=HMatrices.container(tx))
    A = assemble_h2(K, tx, ty; rtol=1e-6, threads=false)
    @test size(A) == (length(X), length(Y))
    x = randn(length(Y))
    rel = norm(A * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test rel < 5e-3
end

@testset "nested rkupdate R2-R3" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do a, b
        d2 = sum(abs2, a - b)
        return exp(-6 * d2) + (d2 < 1e-30 ? 3.0 : 0.0)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H2 = assemble_h2(K, tree; rtol=1e-5, threads=false)
    root = h2node(H2)
    n = length(pts)
    Xl = randn(n, 2)
    Yl = randn(n, 2)
    x = randn(n)
    Md = Matrix(root; global_index=false)
    mul!(Md, Xl, adjoint(Yl), true, true)
    G = h2_clone(root)
    h2_rkupdate_nested!(G, Xl, Yl; rtol=1e-8, recompress=true)
    yg = zeros(n)
    mul!(yg, G, x; global_index=false)
    @test norm(yg - Md * x) / (norm(Md * x) + 1e-14) < 0.15
    w = prepare_h2_weights(G.pack; side=:row)
    @test w isa H2ClusterOperator
    @test length(w.C) == length(G.pack.U)
end

@testset "SRS factor as GMRES preconditioner" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    K = KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    F = srs_factor(K, tree; rtol=1e-6, rank=24)
    @test F isa SRSFactor
    b = randn(n)
    u = F \ copy(b)
    rel = norm(K * u - b) / (norm(b) + 1e-14)
    @test rel < 0.5
    H2 = assemble_h2(K, tree; rtol=1e-6, threads=false)
    _, st0 = gmres_h(H2, b; rtol=1e-8, itmax=80, history=true)
    _, st1 = gmres_h(H2, b; Pl=F, rtol=1e-8, itmax=80, history=true)
    @test st1.niter < st0.niter || st1.solved

    # Matrix-free RSRS (sketches of H2), same tree.
    Fm = srs_factor_matvec(H2, tree; rtol=1e-6, rank=12, p=12)
    _, stm = gmres_h(H2, b; Pl=Fm, rtol=1e-8, itmax=80, history=true)
    @test stm.solved || stm.niter <= st0.niter
end

@testset "matrix-free RSRS eliminates and preconditions" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=16)
    pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=8, tight=false); cube=true)
    H2 = assemble_h2(K, tree; rtol=1e-6, threads=false)
    Fm = srs_factor_matvec(H2, tree; rtol=1e-5, rank=6, p=6)
    @test Fm isa SRSFactor
    @test !isempty(Fm.steps)
    @test sum(s -> length(s.R), Fm.steps) > 0
    b = randn(n)
    u = Fm \ copy(b)
    rdir = norm(H2 * u - b) / (norm(b) + 1e-14)
    @test rdir < 0.5
    x, st = gmres_h(H2, b; Pl=Fm, rtol=1e-8, itmax=40, history=true)
    r = norm(H2 * x - b) / (norm(b) + 1e-14)
    @test st.solved || r < 1e-6
    @test r < 1e-5
end

@testset "column ID and rskelf" begin
    Random.seed!(1)
    U = randn(20, 4)
    V = randn(12, 4)
    A = U * V'
    sk, rd, T = interpolative_decomp(A; rtol=1e-12)
    @test norm(A[:, rd] - A[:, sk] * T) / (norm(A) + 1e-14) < 1e-10
    _, _, T2 = interpolative_decomp(randn(24, 16); rtol=1e-3, Tmax=2)
    @test isempty(T2) || maximum(abs, T2) <= 2 + 1e-6
    As = randn(800, 25) * randn(25, 200)
    sks, rds, Ts = interpolative_decomp(As; rtol=1e-10)
    @test norm(As[:, rds] - As[:, sks] * Ts) / (norm(As) + 1e-14) < 1e-8
    @test isempty(Ts) || maximum(abs, Ts) <= 2 + 1e-5
    ske, rde, Te = interpolative_decomp(As; rtol=1e-10, sketch=false)
    @test norm(As[:, rde] - As[:, ske] * Te) / (norm(As) + 1e-14) < 1e-8

    xs = range(0.0, 1.0; length=16)
    pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=8, tight=false); cube=true)
    b = randn(n)
    Kd = Matrix(K)

    pxy = circle_proxy(kf, pts; npts=32)
    Fr = rskelf(K, tree; rtol=1e-8, rank=24, pxyfun=pxy, Tmax=2, symm=:s)
    @test Fr isa RSKELFFactor
    @test sum(s -> length(s.rd), Fr.steps) > 0
    ur = Fr \ copy(b)
    @test norm(Kd * ur - b) / (norm(b) + 1e-14) < 0.05
    yr = Fr * ur
    @test norm(yr - b) / (norm(b) + 1e-14) < 0.05

    Fp = rskelf(K, tree; rtol=1e-6, rank=16, pxyfun=pxy, Tmax=2, symm=:s)
    up = Fp \ copy(b)
    @test norm(Kd * up - b) / (norm(b) + 1e-14) < 0.15
end

@testset "ILUT" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=16)
    pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H2 = assemble_h2(K, tree; rtol=1e-4, threads=false)
    Snear = near_sparse(H2)
    @test size(Snear) == (n, n)
    @test nnz(Snear) > n
    nL = 48
    SL = I + 0.15 * sprandn(nL, nL, 0.25)
    SL = SL + SL' + 3I
    FL = ilut(sparse(SL); lfil=20, droptol=1e-10)
    @test FL isa ILUTFactor
    bL = randn(nL)
    uL = FL \ copy(bL)
    @test norm(SL * uL - bL) / (norm(bL) + 1e-14) < 0.05
end

@testset "HSS PCA matvec and ULV" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hss(K, tree; rtol=1e-8)
    @test H isa HSSMatrix
    Kd = Matrix(K)
    x = randn(n)
    @test norm(H * x - Kd * x) / (norm(Kd * x) + 1e-14) < 1e-5
    F = ulv(H)
    @test F isa ULVFactor
    b = randn(n)
    u = F \ copy(b)
    @test norm(Kd * u - b) / (norm(b) + 1e-14) < 1e-6
    @test norm(H * u - b) / (norm(b) + 1e-14) < 1e-6
    Haca = assemble_hss(K, tree; rtol=1e-8, method=:aca)
    @test Haca isa HSSMatrix
    @test norm(Haca * x - Kd * x) / (norm(Kd * x) + 1e-14) < 1e-4
    Faca = ulv(Haca)
    uaca = Faca \ copy(b)
    @test norm(Kd * uaca - b) / (norm(b) + 1e-14) < 1e-4
    H2 = assemble_hss(K, tree; rtol=1e-8)
    Hs = H + H2
    @test Hs isa HSSMatrix
    @test norm(Hs * x - (H * x .+ H2 * x)) / (norm(H * x) + 1e-14) < 1e-4
    @test norm(Hs * x - 2 .* (Kd * x)) / (norm(Kd * x) + 1e-14) < 1e-4
    @test norm((H - H2) * x) / (norm(H * x) + 1e-14) < 1e-4
end

@testset "rectangular HSS matvec" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    ys = range(0.0, 1.0; length=6)
    Xt = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    Yt = [SVector{2, Float64}(float(x), float(y)) for y in ys for x in ys]
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(Xt), typeof(Yt), Float64}(kf, Xt, Yt)
    rt = ClusterTree(Xt, PrincipalComponentSplitter(; nmax=12))
    ct = ClusterTree(Yt, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hss(K, rt, ct; rtol=1e-6, method=:id)
    @test size(H) == (length(Xt), length(Yt))
    Kd = Matrix(K)
    x = randn(length(Yt))
    @test norm(H * x - Kd * x) / (norm(Kd * x) + 1e-14) < 5e-4
    @test_throws ArgumentError ulv(H)
end

@testset "assemble HSS of B D^{-1} C" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    ys = range(0.0, 1.0; length=6)
    Xu = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    Xq = [SVector{2, Float64}(float(x), float(y)) for y in ys for x in ys]
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 4.0 : log(r)
    end
    nu_, nq_ = length(Xu), length(Xq)
    Bd = [kf(Xu[i], Xq[j]) for i in 1:nu_, j in 1:nq_]
    Cd = [kf(Xq[i], Xu[j]) for i in 1:nq_, j in 1:nu_]
    Dd = [kf(Xq[i], Xq[j]) for i in 1:nq_, j in 1:nq_]
    Dd += 4.0 * I
    tu = ClusterTree(Xu, PrincipalComponentSplitter(; nmax=16))
    tq = ClusterTree(Xq, PrincipalComponentSplitter(; nmax=12))
    B = assemble_hss(Bd, tu, tq; rtol=1e-8, method=:id)
    C = assemble_hss(Cd, tq, tu; rtol=1e-8, method=:id)
    D = assemble_hss(Dd, tq; rtol=1e-8, method=:id)
    Fd = ulv(D)
    P = HMatrices.assemble_hss_BDC(B, Fd, C, tu; rtol=1e-8, method=:id)
    Pd = Bd * (Dd \ Cd)
    v = randn(nu_)
    @test P isa HSSMatrix
    @test size(P) == (nu_, nu_)
    @test norm(P * v - Pd * v) / (norm(Pd * v) + 1e-14) < 5e-3
    Ad = [kf(Xu[i], Xu[j]) for i in 1:nu_, j in 1:nu_]
    Ad += 4.0 * I
    Ah = assemble_hss(Ad, tu; rtol=1e-8, method=:id)
    S = HMatrices.assemble_hss_schur(Ah, B, Fd, C, tu; rtol=1e-8, method=:id)
    Sd = Ad - Pd
    @test norm(S * v - Sd * v) / (norm(Sd * v) + 1e-14) < 5e-3
    Fs = ulv(S)
    b = Sd * v
    @test norm(Sd * (Fs \ b) - b) / (norm(b) + 1e-14) < 5e-3
    Md = [Ad Bd; Cd Dd]
    z = randn(nu_ + nq_)
    Fh = hodlr_lu_2x2(Ad, Bd, Cd, Dd; rtol=1e-8)
    @test size(Fh) == (nu_ + nq_, nu_ + nq_)
    @test norm(Md * (Fh \ z) - z) / (norm(z) + 1e-14) < 1e-6
    Fu = hodlr_ulv_2x2(Ad, Bd, Cd, Dd, Xu, Xq; rtol=1e-8, nmax=64)
    b = Md * z
    @test norm(Md * (Fu \ b) - b) / (norm(b) + 1e-14) < 1e-6
end

@testset "NNCA nested LR (H2Lib lrdecomp)" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do a, b
        d = hypot(a[1] - b[1], a[2] - b[2])
        return d < 1e-14 ? 4.0 : log(d)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H2 = assemble_h2(K, tree; rtol=1e-6, threads=false)
    root = h2node(H2)
    x = randn(length(pts))
    @test norm(root * x - H2 * x) / (norm(H2 * x) + 1e-14) < 1e-10
    Kd = Matrix(K)
    Kd = Kd + Kd' + 8.0 * I
    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    A2 = assemble_h2(Kd, tree2; rtol=1e-8, threads=false)
    b = randn(length(pts))
    F = lu(A2; method=:nested, rtol=1e-6)
    @test F isa H2NodeLU
    u = F \ copy(b)
    @test norm(A2 * u - b) / (norm(b) + 1e-14) < 1e-10

    # Deeper tree (16×16 grid, nmax=16): truncation at rtol, not a structural bug.
    xs16 = range(0.0, 1.0; length=16)
    pts16 = [SVector(float(x), float(y)) for y in xs16 for x in xs16]
    K16 = KernelMatrix(pts16, pts16) do a, b
        d = hypot(a[1] - b[1], a[2] - b[2])
        return d < 1e-14 ? 4.0 : log(d)
    end
    Kd16 = Matrix(K16)
    Kd16 = Kd16 + Kd16' + 8.0 * I
    tree16 = ClusterTree(pts16, DyadicSplitter(; nmax=16, tight=false); cube=true)
    A16 = assemble_h2(Kd16, tree16; rtol=1e-8, threads=false)
    b16 = randn(length(pts16))
    F16 = lu(A16; method=:nested, rtol=1e-8)
    u16 = F16 \ copy(b16)
    @test norm(A16 * u16 - b16) / (norm(b16) + 1e-14) < 1e-4

    # Structured addmul (no ACA of C+αAB): C ← C - A A vs dense.
    Cadd = h2_clone(root)
    Ad = Matrix(root; global_index=false)
    h2_addmul!(Cadd, root, root, -1.0; rtol=1e-8)
    Cd = Matrix(Cadd; global_index=false)
    @test norm(Cd - (Ad - Ad * Ad)) / (norm(Ad - Ad * Ad) + 1e-14) < 5e-6
end

@testset "NNCA 3D vs dense" begin
    xs = range(0.0, 1.0; length=6)
    pts = [SVector(float(x), float(y), float(z)) for z in xs for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        return d < 1e-14 ? 0.0 : 1 / (4π * d)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H = assemble_h2(K, tree; rtol=1e-5, threads=false)
    x = randn(length(pts))
    rel = norm(H * x - Matrix(K) * x) / (norm(Matrix(K) * x) + 1e-14)
    @test rel < 5e-2
end

function _flatten_tensor(K)
    n = size(K, 1)
    Te = eltype(K)
    p, q = HMatrices.tensor_blocksize(Te)
    T = eltype(Te)
    D = zeros(T, p * n, q * n)
    @inbounds for j in 1:n, i in 1:n
        Bij = K[i, j]
        i0, j0 = p * (i - 1), q * (j - 1)
        for b in 1:q, a in 1:p
            D[i0 + a, j0 + b] = Bij[a, b]
        end
    end
    return D
end

@testset "tensor H-matrix flat and SVector matvec" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    G = SMatrix{2, 2, Float64, 4}(1.0, 0.2, 0.2, 1.0)
    K = KernelMatrix(pts, pts) do x, y
        return log(norm(x - y) + 0.15) * G
    end
    tree = ClusterTree(pts, hmatrix_splitter(; nmax=12); cube=true)
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    @test eltype(H) <: SMatrix
    n = length(pts)
    xflat = randn(2 * n)
    xsv = [SVector(xflat[2 * i - 1], xflat[2 * i]) for i in 1:n]
    ysv = H * xsv
    yflat = H * xflat
    @test yflat ≈ HMatrices.scalarize(ysv)
    D = _flatten_tensor(K)
    @test norm(yflat - D * xflat) / (norm(D * xflat) + 1e-14) < 5e-3
    ythr = similar(yflat)
    mul!(ythr, H, xflat; threads=true)
    @test ythr ≈ yflat
end

@testset "block NNCA vs dense" begin
    Random.seed!(1)
    xs = range(-1.0, 1.0; length=10)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    G = SMatrix{2, 2, Float64, 4}(1.0, 0.15, 0.15, 1.0)
    K = KernelMatrix(pts, pts) do x, y
        return log(hypot(x[1] - y[1], x[2] - y[2]) + 0.2) * G
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H = assemble_h2(K, tree; rtol=1e-6, threads=false)
    @test H.p == 2
    @test size(H) == (2 * length(pts), 2 * length(pts))
    n = length(pts)
    xflat = randn(2 * n)
    D = _flatten_tensor(K)
    rel = norm(H * xflat - D * xflat) / (norm(D * xflat) + 1e-14)
    @test rel < 5e-3
    xsv = [SVector(xflat[2 * i - 1], xflat[2 * i]) for i in 1:n]
    ysv = H * xsv
    @test HMatrices.scalarize(ysv) ≈ H * xflat
end

@testset "GPU H-matrix / NNCA matvec (KA CPU)" begin
    Random.seed!(1)
    xs = range(0.0, 1.0; length=8)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    K = KernelMatrix(pts, pts) do x, y
        d = hypot(x[1] - y[1], x[2] - y[2])
        return d < 1e-14 ? 2.0 : log(d)
    end
    tree = ClusterTree(pts, hmatrix_splitter(; nmax=12); cube=true)
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    Hg = gpu(H; device=:cpu)
    x = randn(length(pts))
    @test Hg isa GPUHMatrix
    @test norm(Hg * x - H * x) / (norm(H * x) + 1e-14) < 1e-10
    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    H2 = assemble_h2(K, tree2; rtol=1e-6, threads=false)
    G2 = gpu(H2; device=:cpu)
    @test G2 isa GPUNNCAMatrix
    @test norm(G2 * x - H2 * x) / (norm(H2 * x) + 1e-14) < 1e-8
    G = SMatrix{2, 2, Float64, 4}(1.0, 0.2, 0.2, 1.0)
    Kt = KernelMatrix(pts, pts) do x, y
        return log(norm(x - y) + 0.15) * G
    end
    Ht = assemble_hmatrix(Kt, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    Htg = gpu(Ht; device=:cpu)
    xt = randn(2 * length(pts))
    @test size(Htg) == (2 * length(pts), 2 * length(pts))
    @test norm(Htg * xt - Ht * xt) / (norm(Ht * xt) + 1e-14) < 1e-8
    Ht2 = assemble_h2(Kt, tree2; rtol=1e-6, threads=false)
    Gt2 = gpu(Ht2; device=:cpu)
    @test Gt2 isa GPUNNCAMatrix
    @test size(Gt2) == (2 * length(pts), 2 * length(pts))
    @test norm(Gt2 * xt - Ht2 * xt) / (norm(Ht2 * xt) + 1e-14) < 1e-8
    yg = similar(x)
    mul!(yg, Hg, x, 2, 0)
    @test yg ≈ 2 .* (Hg * x)
    xs64 = range(0.0, 1.0; length=64)
    pts64 = [SVector(float(x), float(y)) for y in xs64 for x in xs64]
    K64 = KernelMatrix(pts64, pts64) do a, b
        d = hypot(a[1] - b[1], a[2] - b[2])
        return d < 1e-14 ? 2.0 : log(d)
    end
    tree64 = ClusterTree(pts64, hmatrix_splitter(; nmax=32); cube=true)
    H64 = assemble_hmatrix(K64, tree64, tree64;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    Hg64 = gpu(H64; device=:cpu)
    x64 = randn(length(pts64))
    @test norm(Hg64 * x64 - H64 * x64) / (norm(H64 * x64) + 1e-14) < 1e-10
end
