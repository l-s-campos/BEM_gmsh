# Collect all paper-figure series into results/paper_data.json
include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

function square_path(nel; tag="p")
    ndiv = nel ÷ 4 + 1
    return quadrado(; nome=@sprintf("paper_%s_n%d", tag, nel), ndiv=ndiv,
        show=false, ordem=1)
end

function run_square(path, fs, K, nx; setbc, rbf=RBF, kiso=nothing, layout=:cell)
    dad = make_dad(path, fs, K)
    internal_grid!(dad, nx, nx; d_min=0, layout=layout)
    setbc(dad)
    t0 = time()
    if fs === :iso
        assemble!(dad; npg=NPG, threaded=true)
        DIBEM(dad; rbf=rbf, threaded=true)
        nl = min(21, dad.nt - 1)
        if kiso === nothing
            solve_anisotropic_ibp!(dad, K; rbf=rbf, npg=NPG, nlocal=nl)
        else
            solve_anisotropic_dibem!(dad, K; rbf=rbf, npg=NPG, nlocal=nl, kiso=kiso)
        end
    else
        solve_aniso_fs!(dad)
    end
    return dad, time() - t0
end

function edge_xy(dad, edge)
    idx = edge_idx(dad, edge)
    s = [edge in (:left, :right) ? dad.Nodes[i][2] : dad.Nodes[i][1] for i in idx]
    perm = sortperm(s)
    return idx[perm], s[perm]
end

function collect_paper()
    data = Dict{String,Any}()

    # ---- Ex1 mixed, cell lattice ----
    println("=== Ex1 grid ===")
    r1 = run_ex1_grid()
    data["ex1"] = r1

    # ---- Ex1 k-ratio (nel=80, 20×20 cell) ----
    println("=== Ex1 kx/ky ===")
    path80 = square_path(80; tag="k")
    k2 = 0.5
    kratio = Dict{String,Any}[]
    for k1 in 1.0:1.0:5.0
        K = @SMatrix [k1 0.0; 0.0 k2]
        ufun = (x, y) -> example1_u(x, y; k1=k1, k2=k2)
        gfun = (x, y) -> example1_grad(x, y; k1=k1, k2=k2)
        for fs in FS_BOTH
            dad, dt = run_square(path80, fs, K, 20;
                setbc=d -> apply_mixed_ex1!(d; k1=k1, k2=k2))
            e = mrpe_ex1(dad, ufun, gfun, K)
            ir, yr = edge_xy(dad, :right)
            il, yl = edge_xy(dad, :left)
            knn(i) = dot(dad.Normal[i], K * dad.Normal[i])
            rec = merge(e, Dict(
                "fs" => String(fs), "k1" => k1, "k2" => k2, "ratio" => k1 / k2,
                "y_right" => yr,
                "T_right" => [dad.T[i] for i in ir],
                "T_right_ana" => [ufun(dad.Nodes[i][1], dad.Nodes[i][2]) for i in ir],
                "y_left" => yl,
                "dudn_left" => [-dad.q[i] / knn(i) for i in il],
                "dudn_left_ana" => [dot(gfun(dad.Nodes[i][1], dad.Nodes[i][2]), dad.Normal[i]) for i in il],
                "time" => dt,
            ))
            @printf("  k1/k2=%4.1f fs=%-5s  Tint=%.3f\n", k1 / k2, fs, e["eT_int_pct"])
            push!(kratio, rec)
        end
    end
    data["ex1k"] = kratio

    # ---- Ex2A sinusoidal ----
    println("=== Ex2A ===")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example2_sinusoidal(x, y; kx=k1, ky=k2)
    gfun = (x, y) -> example2_sinusoidal_grad(x, y; kx=k1, ky=k2)
    r2a = Dict{String,Any}[]
    for nel in (80, 160)
        path = square_path(nel; tag="2a")
        nxs = nel <= 80 ? (8, 13, 20) : (13, 20, 32)
        for nx in nxs, fs in FS_BOTH
            dad, dt = run_square(path, fs, K, nx; setbc=d -> apply_dirichlet!(d, ufun))
            e = mrpe_ex1(dad, ufun, gfun, K)
            rec = merge(e, Dict("fs" => String(fs), "nel" => nel, "nx" => nx,
                "k1" => k1, "k2" => k2, "eq_right_pct" => begin
                    idx = edge_idx(dad, :right)
                    qb = dad.q[idx]
                    qa = [pkg_flux(dad.Normal[i], K, gfun(dad.Nodes[i][1], dad.Nodes[i][2])) for i in idx]
                    pct_rel(qb, qa)
                end, "time" => dt))
            @printf("  2A nel=%d nx=%d fs=%s Tint=%.3f qR=%.3f\n",
                nel, nx, fs, rec["eT_int_pct"], rec["eq_right_pct"])
            push!(r2a, rec)
        end
    end
    data["ex2a"] = r2a

    # ---- Ex2B k contrast ----
    println("=== Ex2B ===")
    path80 = square_path(80; tag="2b")
    r2b = Dict{String,Any}[]
    for k1 in 1.0:1.0:5.0
        K = @SMatrix [k1 0.0; 0.0 0.5]
        ufun = (x, y) -> example2_sinusoidal(x, y; kx=k1, ky=0.5)
        gfun = (x, y) -> example2_sinusoidal_grad(x, y; kx=k1, ky=0.5)
        for fs in FS_BOTH
            dad, dt = run_square(path80, fs, K, 20; setbc=d -> apply_dirichlet!(d, ufun))
            e = mrpe_ex1(dad, ufun, gfun, K)
            rec = merge(e, Dict("fs" => String(fs), "k1" => k1, "k2" => 0.5,
                "dk" => k1 - 0.5, "time" => dt))
            @printf("  2B k1=%g fs=%s Tint=%.3f\n", k1, fs, e["eT_int_pct"])
            push!(r2b, rec)
        end
    end
    data["ex2b"] = r2b

    # ---- Ex2C anisotropic ----
    println("=== Ex2C ===")
    Kc = @SMatrix [1.0 1.0; 1.0 1.0]
    r2c = Dict{String,Any}[]
    botprof = nothing
    for nel in (80, 160)
        path = square_path(nel; tag="2c")
        nxs = nel <= 80 ? (8, 13, 20) : (13, 20, 32)
        for nx in nxs
            dad, dt = run_square(path, :iso, Kc, nx;
                setbc=d -> apply_dirichlet!(d, example2_anisotropic),
                rbf=RBF_QUAD, kiso=1.0)
            e = mrpe_ex1(dad, example2_anisotropic, example2_anisotropic_grad, Kc)
            rec = merge(e, Dict("fs" => "iso", "nel" => nel, "nx" => nx, "time" => dt))
            rec["eq_bot_pct"] = e["eq_bot_pct"]
            rec["edudn_bot_pct"] = begin
                idx = edge_idx(dad, :bottom)
                knn(i) = dot(dad.Normal[i], Kc * dad.Normal[i])
                dn = [-dad.q[i] / knn(i) for i in idx]
                da = [dot(example2_anisotropic_grad(dad.Nodes[i][1], dad.Nodes[i][2]), dad.Normal[i]) for i in idx]
                pct_rel(dn, da)
            end
            @printf("  2C nel=%d nx=%d Tint=%.3f qB=%.3f\n", nel, nx, e["eT_int_pct"], e["eq_bot_pct"])
            if nel == 80 && nx == 20
                ib, xb = edge_xy(dad, :bottom)
                knn(i) = dot(dad.Normal[i], Kc * dad.Normal[i])
                rec["x_bot"] = xb
                rec["dudn_bot"] = [-dad.q[i] / knn(i) for i in ib]
                rec["dudn_bot_ana"] = [dot(example2_anisotropic_grad(dad.Nodes[i][1], dad.Nodes[i][2]), dad.Normal[i]) for i in ib]
                botprof = rec
            end
            push!(r2c, rec)
        end
    end
    data["ex2c"] = r2c

    # ---- Ex2D discontinuous ----
    println("=== Ex2D ===")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example2_discontinuous(x, y; k1=k1, k2=k2)
    gfun = (x, y) -> example2_discontinuous_grad(x, y; k1=k1, k2=k2)
    path = square_path(80; tag="2d")
    r2d = Dict{String,Any}[]
    for nx in (8, 13, 20), fs in FS_BOTH
        dad, dt = run_square(path, fs, K, nx;
            setbc=d -> apply_dirichlet!(d, (x, y) -> y >= 1 - 1e-8 ? 1.0 : 0.0))
        e = mrpe_ex1(dad, ufun, gfun, K)
        rec = merge(e, Dict("fs" => String(fs), "nx" => nx, "k1" => k1, "k2" => k2, "time" => dt))
        @printf("  2D nx=%d fs=%s Tint=%.3f\n", nx, fs, e["eT_int_pct"])
        push!(r2d, rec)
    end
    data["ex2d"] = r2d

    save_json(joinpath(RESULTDIR, "paper_data.json"), data)
    println("wrote paper_data.json")
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    collect_paper()
end
