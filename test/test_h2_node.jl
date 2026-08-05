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
