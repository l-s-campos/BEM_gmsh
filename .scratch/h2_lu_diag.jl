# Diagnose nested H² LU residual / structure.
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using BEM.HMatrices
import BEM.HMatrices:
    h2node, h2_clone, h2_block_matrix, h2_foreach, isdense_h2, isuniform, issplit,
    h2_nnodes, h2_nleaves, lrdecomp_h2node!, H2NodeLU, rowrange, colrange

function pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function spd_kernel(pts)
    K = KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 4.0 : log(r)
    end
    Kd = Matrix(K)
    return Kd + Kd' + 8.0 * I
end

function count_kinds(N)
    n_d = n_u = n_sf = n_s = 0
    h2_foreach(N) do node
        if isdense_h2(node)
            n_d += 1
        elseif isuniform(node)
            n_u += 1
            node.s_full && (n_sf += 1)
        else
            n_s += 1
        end
    end
    return (; n_d, n_u, n_sf, n_s)
end

function check_ranges(N; parent_ir=nothing, parent_jr=nothing)
    issues = String[]
    ir, jr = rowrange(N), colrange(N)
    m, n = size(N)
    if length(ir) != m || length(jr) != n
        push!(issues, "size vs range $(size(N)) vs $(length(ir))×$(length(jr)) row=$(N.row_id)")
    end
    if isdense_h2(N) && size(N.F) != (m, n)
        push!(issues, "dense F $(size(N.F)) vs $(m)×$(n) row=$(N.row_id) col=$(N.col_id)")
    end
    if issplit(N)
        rs, cs = size(N.sons)
        covered_r = falses(m)
        covered_c = falses(n)
        i0, j0 = first(ir), first(jr)
        for i in 1:rs, j in 1:cs
            s = N.sons[i, j]
            append!(issues, check_ranges(s; parent_ir=ir, parent_jr=jr))
            sir, sjr = rowrange(s), colrange(s)
            rows = (first(sir) - i0 + 1):(last(sir) - i0 + 1)
            cols = (first(sjr) - j0 + 1):(last(sjr) - j0 + 1)
            if !(first(sir) >= first(ir) && last(sir) <= last(ir))
                push!(issues, "son row range $sir not in parent $ir")
            end
            if 1 <= first(rows) <= last(rows) <= m
                covered_r[rows] .= true
            end
            if 1 <= first(cols) <= last(cols) <= n
                covered_c[cols] .= true
            end
        end
        if !all(covered_r)
            push!(issues, "row holes at row=$(N.row_id) missing=$(count(.!covered_r))/$m sons=$(rs)×$(cs)")
        end
        if !all(covered_c)
            push!(issues, "col holes at col=$(N.col_id) missing=$(count(.!covered_c))/$n")
        end
        # diagonal sons should be square same-cluster
        if N.row_id == N.col_id
            rs == cs || push!(issues, "non-square sons on diagonal $(rs)×$(cs)")
            for i in 1:min(rs, cs)
                s = N.sons[i, i]
                s.row_id == s.col_id || push!(issues, "diag son $i row=$(s.row_id) col=$(s.col_id)")
            end
        end
    end
    return issues
end

function run(n1d; nmax=16, rtol=1e-6)
    Random.seed!(1)
    P = pts(n1d)
    n = length(P)
    Kd = spd_kernel(P)
    tree = ClusterTree(P, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    A = assemble_h2(Kd, tree; rtol=1e-8, threads=false)
    root = h2node(A)
    x = randn(n)
    yA = A * x
    yN = root * x
    rel_h2 = norm(yN - yA) / (norm(yA) + 1e-14)
    Md = Matrix(root; global_index=false)
    yloc = zeros(n)
    mul!(yloc, root, x[A.colperm_pts]; global_index=false)
    rel_loc = norm(Md * (x[A.colperm_pts]) - yloc) / (norm(yloc) + 1e-14)
    Mb = try
        h2_block_matrix(root)
    catch e
        @warn "h2_block_matrix failed" exception=e
        nothing
    end
    rel_blk = Mb === nothing ? NaN : norm(Mb - Md) / (norm(Md) + 1e-14)
    issues = check_ranges(root)
    c0 = count_kinds(root)
    b = randn(n)
    b_loc = b[A.colperm_pts]
    # dense reference in local order
    u_dense_loc = Md \ b_loc
    rel_dense = norm(Md * u_dense_loc - b_loc) / (norm(b_loc) + 1e-14)

    t_lu = @elapsed F = lu(A; method=:nested, rtol=rtol)
    c1 = count_kinds(F.factors)
    u = F \ copy(b)
    rel_A = norm(A * u - b) / (norm(b) + 1e-14)
    u_loc = copy(b_loc)
    ldiv!(F, u_loc; global_index=false)
    rel_loc_solve = norm(Md * u_loc - b_loc) / (norm(b_loc) + 1e-14)

    # also H-matrix path
    rel_hmat = NaN
    t_hmat = NaN
    try
        t_hmat = @elapsed FH = lu(A; method=:hmatrix, rtol=rtol, threads=false)
        uH = FH \ copy(b)
        rel_hmat = norm(A * uH - b) / (norm(b) + 1e-14)
    catch e
        @warn "hmatrix lu failed" exception=e
    end

    println("="^72)
    @printf("N=%d nmax=%d rtol=%.1e  nodes=%d leaves=%d\n", n, nmax, rtol, h2_nnodes(root), h2_nleaves(root))
    println("  before: dense=$(c0.n_d) unif=$(c0.n_u) s_full=$(c0.n_sf) split=$(c0.n_s)")
    println("  after : dense=$(c1.n_d) unif=$(c1.n_u) s_full=$(c1.n_sf) split=$(c1.n_s)")
    @printf("  h2node vs NNCA matvec      %.3e\n", rel_h2)
    @printf("  Matrix(root) vs mul local  %.3e\n", rel_loc)
    @printf("  h2_block_matrix vs Matrix  %.3e\n", rel_blk)
    @printf("  dense LU residual          %.3e\n", rel_dense)
    @printf("  nested LU vs NNCA (global) %.3e   (t=%.3fs)\n", rel_A, t_lu)
    @printf("  nested LU vs Matrix local  %.3e\n", rel_loc_solve)
    @printf("  H-matrix LU vs NNCA        %.3e   (t=%.3fs)\n", rel_hmat, t_hmat)
    if !isempty(issues)
        println("  RANGE ISSUES ($(length(issues))):")
        for s in issues[1:min(8, length(issues))]
            println("    ", s)
        end
    else
        println("  range check: ok")
    end
    return (; n, rel_h2, rel_blk, rel_A, rel_loc_solve, rel_hmat, t_lu, c0, c1, issues)
end

println("threads=$(Threads.nthreads())")
run(8; nmax=16)
run(16; nmax=16)
println("\n--- rtol sweep N=256 nmax=16 ---")
for rt in (0.0, 1e-10, 1e-8, 1e-6, 1e-4)
    run(16; nmax=16, rtol=rt)
end
println("\n--- N=1024 nmax=16 rtol=1e-6 ---")
run(32; nmax=16, rtol=1e-6)
nothing
