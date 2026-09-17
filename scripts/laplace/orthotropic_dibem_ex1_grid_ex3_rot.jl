# Example 1: three boundary meshes + internal_grid internals, MRPE tables.
# Example 3: three Gmsh meshes, gmsh internals, K = R diag(5,0.5) R' at 30°.
#
#   julia --project=. scripts/laplace/orthotropic_dibem_ex1_grid_ex3_rot.jl
using LinearAlgebra
using StaticArrays
using Statistics
using Printf

include(joinpath(@__DIR__, "orthotropic_dibem_examples.jl"))

pct_rel(num, ana) = begin
    m = maximum(abs, ana)
    m == 0 && return 100 * mean(abs.(num .- ana))
    100 * mean(abs.(num .- ana)) / m
end

function edge_idx(dad, edge; skip_corners=true)
    idx = Int[]
    @inbounds for i in 1:dad.n
        classify_edge(dad.Nodes[i]) === edge || continue
        skip_corners && near_corner(dad.Nodes[i]) && continue
        push!(idx, i)
    end
    return idx
end

function rotate_K(k1, k2, θ)
    c, s = cos(θ), sin(θ)
    R = @SMatrix [c -s; s c]
    Kd = @SMatrix [k1 0.0; 0.0 k2]
    return R * Kd * transpose(R)
end

function apply_ex3_K!(dad, K)
    for i in 1:dad.n
        e = classify_edge(dad.Nodes[i])
        n = dad.Normal[i]
        if e === :left
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif e === :right
            dad.BC[i] = 1
            # ∂u/∂n = 1  →  take ∇u = n  ⇒  q = −n·K n
            dad.BV[i] = -dot(n, K * n)
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    return dad
end

function mrpe_ex1(dad, ufun, gfun, K)
    n = dad.n
    T = dad.T
    q = dad.q
    Ti = T[n+1:n+dad.ni]
    Tia = [ufun(p[1], p[2]) for p in dad.internalNodes]
    function Tedge(edge)
        idx = edge_idx(dad, edge)
        Tb = T[idx]
        Ta = [ufun(dad.Nodes[i][1], dad.Nodes[i][2]) for i in idx]
        return pct_rel(Tb, Ta)
    end
    function qedge(edge)
        idx = edge_idx(dad, edge)
        qb = q[idx]
        qa = [pkg_flux(dad.Normal[i], K, gfun(dad.Nodes[i][1], dad.Nodes[i][2])) for i in idx]
        return pct_rel(qb, qa)
    end
    function dudn_edge(edge)
        idx = edge_idx(dad, edge)
        dn = Float64[]
        da = Float64[]
        for i in idx
            nrm = dad.Normal[i]
            knn = dot(nrm, K * nrm)
            push!(dn, abs(knn) < 1e-15 ? NaN : -dad.q[i] / knn)
            push!(da, dot(gfun(dad.Nodes[i][1], dad.Nodes[i][2]), nrm))
        end
        return pct_rel(dn, da)
    end
    return Dict{String,Any}(
        "eT_int_pct" => pct_rel(Ti, Tia),
        "eT_left_pct" => Tedge(:left),
        "eT_top_pct" => Tedge(:top),
        "eT_right_pct" => Tedge(:right),
        "eq_left_pct" => qedge(:left),
        "eq_bot_pct" => qedge(:bottom),
        "edudn_left_pct" => dudn_edge(:left),
        "ni" => dad.ni,
        "n" => n,
        "nel" => length(dad.elements),
    )
end

function run_ex1_grid(; nels=[80, 160, 320],
        # h_int ≲ 2 h_Γ. A fixed Cartesian grid (same nx on 80 and 320)
        # makes the fine boundary worse: IBP ∇u at Γ Gauss points never
        # sees the interior. nx ≈ nel_edge recovered 320 to the 80-element
        # error level (Tint 0.007 %).
        grids=nothing)
    println("\n=== Example 1  internal_grid  k1=2 k2=0.5 ===")
    println("MRPE % = 100 * mean|num-ana| / max|ana|")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example1_u(x, y; k1=k1, k2=k2)
    gfun = (x, y) -> example1_grad(x, y; k1=k1, k2=k2)
    rows = Dict{String,Any}[]
    @printf("%-5s %-5s %-5s %-5s %8s %8s %8s %8s %8s %8s\n",
        "nel", "nx", "ny", "fs", "ni", "T_int%", "T_top%", "T_right%", "q_left%", "q_bot%")
    for nel in nels
        nel_edge = nel ÷ 4
        ndiv = nel_edge + 1
        # keep h_int ≲ 2 h_Γ so IBP stencils at Γ reach the lattice
        nx_list = if grids !== nothing
            grids
        elseif nel <= 80
            [(8, 8), (13, 13), (20, 20)]
        elseif nel <= 160
            [(13, 13), (20, 20), (40, 40)]
        else
            [(20, 20), (32, 32), (40, 40)]
        end
        path = quadrado(; nome=@sprintf("orto_e1g_n%d", nel), ndiv=ndiv,
            show=false, ordem=1)
        for (nx, ny) in nx_list
            for fs in FS_BOTH
                dad = make_dad(path, fs, K)
                internal_grid!(dad, nx, ny; d_min=0, layout=:cell)
                apply_mixed_ex1!(dad; k1=k1, k2=k2)
                t0 = time()
                if fs === :iso
                    solve_iso_dibem!(dad, K; hole=false)
                else
                    solve_aniso_fs!(dad)
                end
                dt = time() - t0
                e = mrpe_ex1(dad, ufun, gfun, K)
                @printf("%-5d %-5d %-5d %-5s %8d %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                    nel, nx, ny, fs, e["ni"],
                    e["eT_int_pct"], e["eT_top_pct"], e["eT_right_pct"],
                    e["eq_left_pct"], e["eq_bot_pct"])
                flush(stdout)
                push!(rows, merge(e, Dict("example" => "1grid", "fs" => String(fs),
                    "nel" => nel, "nx" => nx, "ny" => ny, "k1" => k1, "k2" => k2,
                    "time" => dt)))
            end
        end
    end
    return rows
end

function run_ex3_rotated(; lcs=[0.10, 0.06, 0.04], k1=5.0, k2=0.5, θdeg=30.0)
    θ = deg2rad(θdeg)
    K = rotate_K(k1, k2, θ)
    println("\n=== Example 3  rotated K  k1=$k1 k2=$k2  θ=$(θdeg)° ===")
    println("  K = ", K)
    println("  n·K n on x=1  (n=e_x) = ", K[1, 1])
    rows = Dict{String,Any}[]
    @printf("%-8s %-5s %6s %6s %8s %10s\n", "lc", "fs", "n", "ni", "t[s]", "T(1,0.5)")
    for lc in lcs
        path = mesh_square_hole(; nome=@sprintf("orto_e3r_lc%.3f", lc), lc=lc)
        for fs in FS_BOTH
            dad = make_dad(path, fs, K)
            apply_ex3_K!(dad, K)
            t0 = time()
            if fs === :iso
                solve_iso_dibem!(dad, K; hole=true)
            else
                solve_aniso_fs!(dad)
            end
            dt = time() - t0
            ys, Ts, qs = right_edge(dad)
            Tmid = isempty(Ts) ? NaN : Ts[argmin(abs.(ys .- 0.5))]
            @printf("%-8.3f %-5s %6d %6d %8.2f %10.4f\n",
                lc, fs, dad.n, dad.ni, dt, Tmid)
            flush(stdout)
            push!(rows, Dict{String,Any}(
                "example" => "3rot", "fs" => String(fs), "lc" => lc,
                "n" => dad.n, "ni" => dad.ni, "nel" => length(dad.elements),
                "k1" => k1, "k2" => k2, "theta_deg" => θdeg,
                "K11" => K[1, 1], "K12" => K[1, 2], "K22" => K[2, 2],
                "time" => dt, "T_mid" => Tmid,
                "y_right" => ys, "T_right" => Ts, "q_right" => qs,
            ))
        end
    end
    return rows
end

function write_ex1_tables(rows)
    series_of(key, fs) = begin
        out = []
        for nel in sort(unique(Int(r["nel"]) for r in rows))
            sub = filter(r -> Int(r["nel"]) == nel && r["fs"] == fs, rows)
            isempty(sub) && continue
            perm = sortperm([Int(r["ni"]) for r in sub])
            sub = sub[perm]
            push!(out, (label="n=$(nel) $(fs)",
                x=[Int(r["ni"]) for r in sub],
                y=[float(r[key]) for r in sub]))
        end
        return out
    end
    for (key, ylab, fname) in (
            ("eT_int_pct", "MRPE T int. [%]", "conv-ex1-Tint"),
            ("eT_top_pct", "MRPE T top [%]", "conv-ex1-Ttop"),
            ("eT_right_pct", "MRPE T right [%]", "conv-ex1-Tright"),
            ("eq_left_pct", "MRPE q left [%]", "conv-ex1-qleft"),
            ("eq_bot_pct", "MRPE q bottom [%]", "conv-ex1-qbot"),
        )
        ser = vcat(series_of(key, "iso"), series_of(key, "aniso"))
        write_conv_typ(joinpath(FIGDIR, fname * ".typ");
            title="", xlabel="\$N_\"int\"\$", ylabel=ylab, series=ser)
    end
end

function write_ex3rot_plot(rows)
    series = []
    lcs = sort(unique(float(r["lc"]) for r in rows))
    lc = lcs[1]
    for fs in ("iso", "aniso")
        sub = filter(r -> r["fs"] == fs && abs(float(r["lc"]) - lc) < 1e-12, rows)
        isempty(sub) && continue
        r = sub[1]
        push!(series, (label="$(fs) lc=$(lc)",
            x=float.(r["y_right"]), y=float.(r["T_right"])))
    end
    isempty(series) || write_xy_typ(joinpath(FIGDIR, "ex3-rot-right.typ");
        title="Example 3 rotated 30°", xlabel="y", ylabel="T(x=1)", series=series)
end

function main_grid()
    r1 = run_ex1_grid()
    save_json(joinpath(RESULTDIR, "example_1_grid.json"),
        Dict("created" => string(Dates.now()),
            "error" => "MRPE % = 100 * mean|num-ana| / max|ana|",
            "grids" => "internal_grid nx×ny, d_min=0.01",
            "runs" => r1))
    write_ex1_tables(r1)
    r3 = run_ex3_rotated()
    save_json(joinpath(RESULTDIR, "example_3_rot.json"),
        Dict("created" => string(Dates.now()),
            "note" => "K = R(30°) diag(5, 0.5) R'; gmsh internals; IBP on hole",
            "runs" => r3))
    write_ex3rot_plot(r3)
    println("\ndone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_grid()
end
