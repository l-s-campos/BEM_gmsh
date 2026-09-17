# Isolate HBIE error: simple exact (u,t) fields, residual H u − G t, Dirichlet vs Neumann.
# Rigid rotation (t=0) tests H only. Linear patch tests both; all-Dirichlet recovers t (G),
# all-Neumann recovers u (H).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props_an = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
props_iso = Elasticity(MAT1.E1, MAT1.ν12, 1.0; plane_stress=true)
D = inv(props_an.params.C)
pσ = 100.0

ana_rigid_x = AnalyticalSolution("rigid-ux",
    (x; t=0.0) -> SVector(1.0, 0.0);
    q = (x, n; t=0.0) -> SVector(0.0, 0.0))
ana_rot = AnalyticalSolution("rigid-rot",
    (x; t=0.0) -> SVector(-x[2], x[1]);
    q = (x, n; t=0.0) -> SVector(0.0, 0.0))
ana_patch = AnalyticalSolution("patch-σx",
    (x; t=0.0) -> SVector(D[1, 1] * pσ * x[1] + D[1, 3] * pσ * x[2],
                          D[1, 2] * pσ * x[2]);
    q = (x, n; t=0.0) -> SVector(pσ * n[1], 0.0))
# plane-stress uniaxial σx = pσ  (matches props_iso)
ana_iso = AnalyticalSolution("patch-σx-iso",
    (x; t=0.0) -> SVector(pσ / MAT1.E1 * x[1], -MAT1.ν12 * pσ / MAT1.E1 * x[2]);
    q = (x, n; t=0.0) -> SVector(pσ * n[1], 0.0))

function ut_exact(dad, ana)
    n = dad.n
    u = zeros(2n); t = zeros(2n)
    @inbounds for i in 1:n
        ui = ana.u(dad.Nodes[i])
        ti = ana.q(dad.Nodes[i], dad.Normal[i])
        u[2i-1] = ui[1]; u[2i] = ui[2]
        t[2i-1] = ti[1]; t[2i] = ti[2]
    end
    return u, t
end

function group_of(p)
    r = hypot(p[1], p[2])
    p[1] < 1.0 && return :left
    p[2] < 1.0 && return :bottom
    r < 450 && return :inner
    r > 450 && return :outer
    return :other
end

function residual_report(label, H, G, u, t, dad)
    Hu = H * u
    Gt = G * t
    r = Hu .- Gt
    nHu, nGt, nr = norm(Hu), norm(Gt), norm(r)
    scale = nHu + nGt + 1e-30
    @printf("  %-22s  ||Hu||=%9.3e  ||Gt||=%9.3e  ||r||/scale=%8.2e  max|r|=%9.3e\n",
        label, nHu, nGt, nr / scale, maximum(abs, r))
    # residual by geometric group (P3-style)
    acc = Dict{Symbol,Float64}(:left=>0.0, :bottom=>0.0, :inner=>0.0, :outer=>0.0, :other=>0.0)
    cnt = Dict{Symbol,Int}(:left=>0, :bottom=>0, :inner=>0, :outer=>0, :other=>0)
    @inbounds for i in 1:dad.n
        g = group_of(dad.Nodes[i])
        acc[g] += hypot(r[2i-1], r[2i])
        cnt[g] += 1
    end
    parts = String[]
    for g in (:left, :bottom, :inner, :outer)
        cnt[g] == 0 && continue
        push!(parts, @sprintf("%s=%.2e", g, acc[g] / cnt[g]))
    end
    isempty(parts) || println("                         mean|r|_node  ", join(parts, "  "))
    return r, Hu, Gt
end

function solve_dirichlet!(dad, ana)
    apply_analytical_bc!(dad, ana, Int[])
    solve(dad)
    uana, tana = ut_exact(dad, ana)
    relu = norm(dad.u .- uana) / (norm(uana) + 1e-30)
    nt = norm(tana)
    relt = nt < 1e-14 ? maximum(abs, dad.traction) : norm(dad.traction .- tana) / nt
    return relu, relt, cond(Matrix(dad.A))
end

function solve_neumann!(dad, ana)
    n = dad.n
    xs = [p[1] for p in dad.Nodes]
    i1 = argmin(xs)
    i2 = argmax(xs)
    i2 == i1 && (i2 = mod1(i1 + 1, n))
    pins = sort(unique([i1, i2]))
    neu = setdiff(1:n, pins)
    apply_analytical_bc!(dad, ana, neu)
    # force the three rigid pins (ux,uy at i1, ux at i2)
    dad.BC[2i1-1] = 0; dad.BC[2i1] = 0
    dad.BV[2i1-1] = ana.u(dad.Nodes[i1])[1]
    dad.BV[2i1]   = ana.u(dad.Nodes[i1])[2]
    dad.BC[2i2-1] = 0
    dad.BV[2i2-1] = ana.u(dad.Nodes[i2])[1]
    solve(dad)
    uana, _ = ut_exact(dad, ana)
    relu = norm(dad.u .- uana) / (norm(uana) + 1e-30)
    return relu, cond(Matrix(dad.A)), pins
end

function svd_tail(M; k=4)
    S = svd(M).S
    return S[1], S[end-k+1:end]
end

function run_mesh(name, msh, props, fields; tipo=2, near=:euclid, npg_h=50)
    println("\n", "="^72)
    println(name, "  near=", near, "  npg_h=", npg_h)
    dad0 = format2d(msh, props; tipo=tipo, pontointerno=false)
    println("  n=$(dad0.n)  nelem=$(length(dad0.elements))")

    assemble!(dad0; npg=16, threaded=false)
    Hc, Gc = copy(dad0.H), copy(dad0.G)
    set_cache!(dad0; nearfield=near)
    H_G_hyper(dad0; npg=npg_h, threaded=false)
    Hh, Gh = copy(dad0.H), copy(dad0.G)

    σH1, σHt = svd_tail(Hh)
    σG1, σGt = svd_tail(Gh)
    @printf("  ||H'||=%.3e  ||G'||=%.3e  σmax(H')=%.2e  σmin4(H')=%s\n",
        norm(Hh), norm(Gh), σH1, string(σHt))
    @printf("  σmax(G')=%.2e  σmin4(G')=%s\n", σG1, string(σGt))

    for (flab, ana) in fields
        println("  -- field ", flab, " --")
        u, t = ut_exact(dad0, ana)
        residual_report("CBIE", Hc, Gc, u, t, dad0)
        residual_report("HBIE", Hh, Gh, u, t, dad0)

        dad = format2d(msh, props; tipo=tipo, pontointerno=false)
        set_cache!(dad; H=copy(Hh), G=copy(Gh), H_hyper=copy(Hh), G_hyper=copy(Gh))
        relu_d, relt_d, κd = solve_dirichlet!(dad, ana)
        @printf("  HBIE all-Dirichlet  rel(u)=%.3e  rel(t)=%.3e  cond=%.2e   [tests G]\n",
            relu_d, relt_d, κd)

        dad = format2d(msh, props; tipo=tipo, pontointerno=false)
        set_cache!(dad; H=copy(Hh), G=copy(Gh), H_hyper=copy(Hh), G_hyper=copy(Gh))
        relu_n, κn, pins = solve_neumann!(dad, ana)
        @printf("  HBIE all-Neumann    rel(u)=%.3e  cond=%.2e  pins=%s   [tests H]\n",
            relu_n, κn, string(pins))

        # mixed like P3: Dirichlet on x≈0, Neumann elsewhere
        dad = format2d(msh, props; tipo=tipo, pontointerno=false)
        set_cache!(dad; H=copy(Hh), G=copy(Gh), H_hyper=copy(Hh), G_hyper=copy(Gh))
        vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1.0]
        if !isempty(vert) && length(vert) < dad.n
            apply_analytical_bc!(dad, ana, setdiff(1:dad.n, vert))
            solve(dad)
            uana, _ = ut_exact(dad, ana)
            rel = norm(dad.u .- uana) / (norm(uana) + 1e-30)
            @printf("  HBIE mixed x=0 Dir  rel(u)=%.3e  max|u|=%.4f  cond=%.2e\n",
                rel, maximum(abs, dad.u), cond(Matrix(dad.A)))
        end
    end
end

# --- square isotropic (straight, quadratic disc.) ---
msh_sq = quadrado_elasticity(; nome="hbie_HvsG_sq", Lx=1.0, Ly=1.0, ordem=2, ndiv=5, show=false)
run_mesh("square iso quadratic", msh_sq, props_iso,
    [("rigid-ux", ana_rigid_x), ("rigid-rot", ana_rot), ("patch-εxx", ana_iso)];
    npg_h=16)

# --- P1 rectangle MAT1 ---
run_mesh("P1 MAT1", datadir("elastico", "cordeiro_p1.msh"), props_an,
    [("rigid-ux", ana_rigid_x), ("rigid-rot", ana_rot), ("patch-σx", ana_patch)];
    npg_h=16)

# --- P3 on-circle (the failing mesh) ---
run_mesh("P3 on-circle q2 MAT1", datadir("elastico", "p3_cmp_q2.msh"), props_an,
    [("rigid-ux", ana_rigid_x), ("rigid-rot", ana_rot), ("patch-σx", ana_patch)];
    near=:euclid, npg_h=50)

run_mesh("P3 on-circle q2 MAT1 tangent", datadir("elastico", "p3_cmp_q2.msh"), props_an,
    [("rigid-rot", ana_rot), ("patch-σx", ana_patch)];
    near=:tangent, npg_h=50)

# --- P3 chordal (HBIE overlay was OK) ---
run_mesh("P3 chordal lin MAT1", datadir("elastico", "p3_cmp_lin.msh"), props_an,
    [("rigid-rot", ana_rot), ("patch-σx", ana_patch)];
    near=:euclid, npg_h=50)

println("\ndone")
