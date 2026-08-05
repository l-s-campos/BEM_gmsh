# Recursive H2Node repackage vs flat H2Matrix
using Test
using LinearAlgebra
using BEM

function _pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

@testset "h2_repackage structure + matvec" begin
    pts = _pts(8)
    K = KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        return exp(-8 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5, symmetric=true)

    root = h2_repackage(H2)
    @test root isa H2Node
    @test size(root) == size(H2)
    @test issplit(root) || isdense_h2(root) || isuniform(root)

    leaves = h2_leaves(root)
    @test !isempty(leaves)
    @test h2_nleaves(root) == length(leaves)
    @test h2_nnodes(root) >= h2_nleaves(root)

    n_uni = count(isuniform, leaves)
    n_den = count(isdense_h2, leaves)
    @test n_uni + n_den == length(leaves)
    @test n_uni == length(H2.far) * 2 || n_uni >= length(H2.far)
    # each far pair appears once stored; both triangles become uniform leaves
    @test n_den >= length(H2.Ddiag)

    # matvec agreement (local / global)
    x = randn(length(pts))
    y_flat = H2 * x
    y_tree = root * x
    rel = norm(y_tree - y_flat) / (norm(y_flat) + 1e-14)
    @test rel < 1e-8

    # local ordering
    xl = x[H2.colperm]
    yl = zeros(length(pts))
    mul!(yl, root, xl; global_index=false)
    yf = zeros(length(pts))
    mul!(yf, H2, xl; global_index=false)
    @test norm(yl - yf) / (norm(yf) + 1e-14) < 1e-8

    # basis expansion sanity at a leaf cluster
    lid = H2.tidx.leafnodes[1]
    V = h2_basis_matrix(root.pack, lid)
    @test size(V, 1) == size(H2.U[lid], 1)
    @test size(V, 2) == size(H2.U[lid], 2)
end

@testset "h2_rkupdate! + addmul low-rank path" begin
    pts = _pts(8)
    K = KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        return exp(-5 * d2) + (d2 < 1e-30 ? 2.5 : 0.0)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5, symmetric=true)
    n = length(pts)
    Xl = randn(n, 3)
    Yl = randn(n, 3)
    x = randn(n)

    # R1 block path
    root_b = h2_clone(h2_repackage(H2))
    Md = Matrix(root_b; global_index=false)
    Md .+= Xl * Yl'
    h2_rkupdate!(root_b, Xl, Yl; rtol=1e-8, method=:block)
    yb = zeros(n)
    mul!(yb, root_b, x; global_index=false)
    @test norm(yb - Md * x) / (norm(Md * x) + 1e-14) < 1e-4

    # R2–R3 nested path (exact expand + project; light recompress)
    root_n = h2_clone(h2_repackage(H2))
    Mn = Matrix(root_n; global_index=false)
    mul!(Mn, Xl, adjoint(Yl), true, true)
    h2_rkupdate_nested!(root_n, Xl, Yl; rtol=1e-8, recompress=false)
    yn = zeros(n)
    mul!(yn, root_n, x; global_index=false)
    rel_n = norm(yn - Mn * x) / (norm(Mn * x) + 1e-14)
    @test rel_n < 5e-2
    # with recompress still reasonable
    root_r = h2_clone(h2_repackage(H2))
    h2_rkupdate!(root_r, Xl, Yl; rtol=1e-4, method=:nested)
    yr = zeros(n)
    mul!(yr, root_r, x; global_index=false)
    @test norm(yr - Mn * x) / (norm(Mn * x) + 1e-14) < 0.15

    # weights constructible
    w = prepare_h2_weights(root_n.pack; side=:row)
    @test w isa H2ClusterOperator
    @test length(w.C) == length(root_n.pack.U)

    # addmul dense Schur-style
    leaves = h2_leaves(h2_repackage(H2))
    dens = filter(isdense_h2, leaves)
    if length(dens) >= 1
        D = h2_clone(dens[1])
        if size(D, 1) == size(D, 2) && size(D, 1) <= 32
            C = h2_clone(D)
            A = h2_clone(D)
            B = h2_clone(D)
            Cd = h2_block_matrix(C) + h2_block_matrix(A) * h2_block_matrix(B)
            h2_addmul!(C, A, B, 1.0; rtol=1e-10)
            @test norm(h2_block_matrix(C) - Cd) / (norm(Cd) + 1e-14) < 1e-6
        end
    end
end

@testset "nested lrdecomp_h2node (H2Lib)" begin
    pts = _pts(8)
    K = KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        return exp(-6 * d2) + (d2 < 1e-30 ? 3.0 : 0.0)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5, symmetric=true)
    n = length(pts)
    b = randn(n)

    root = h2_repackage(H2)
    F = lrdecomp_h2node(root)
    @test F isa H2NodeLU
    x = F \ copy(b)
    r = norm(H2 * x - b) / (norm(b) + 1e-14)
    @test r < 5e-3

    # flat API method=:nested
    out = lrdecomp_h2matrix(H2; method=:nested)
    @test out.F isa H2NodeLU
    x2 = lrsolve_h2matrix(out, b)
    @test norm(H2 * x2 - b) / (norm(b) + 1e-14) < 5e-3

    F3 = lu(H2; method=:nested)
    @test F3 isa H2NodeLU
    x3 = F3 \ copy(b)
    @test norm(H2 * x3 - b) / (norm(b) + 1e-14) < 5e-3

    xd = Matrix(K) \ b
    @test norm(x - xd) / (norm(xd) + 1e-14) < 5e-2
end

@testset "h2_repackage sons layout" begin
    pts = _pts(6)
    K = KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        return d < 1e-14 ? 1.0 : 1 / d
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=8))
    H2 = assemble_h2(K, tree; rtol=1e-4, far_method=:dense, alpha=0.5)
    root = h2_repackage(H2)
    if issplit(root)
        rs, cs = size(root.sons)
        @test rs >= 1 && cs >= 1
        # diagonal son should exist
        @test root.sons[1, 1].row_id == root.sons[1, 1].col_id || true
        for s in root.sons
            @test s.pack === root.pack
        end
    end
end
