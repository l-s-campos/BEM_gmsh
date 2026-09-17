# Exact Lekhnitskii source outside the mesh: non-polynomial (u,t). Dirichlet tests G, Neumann tests H.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function source_ana(xs, dir::Int)
    AnalyticalSolution("lekh-src",
        (x; t=0.0) -> begin
            kp = fundamental(props, SVector(x[1], x[2]), xs, SVector(1.0, 0.0))
            U = BEM._to_smat(kp.U)
            SVector(U[1, dir], U[2, dir])
        end;
        q = (x, n; t=0.0) -> begin
            kp = fundamental(props, SVector(x[1], x[2]), xs, SVector(n[1], n[2]))
            T = BEM._to_smat(kp.T)
            SVector(T[1, dir], T[2, dir])
        end)
end

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

function run(msh, near, xs, dir, tag)
    ana = source_ana(xs, dir)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; nearfield=near)
    H_G_hyper(dad; npg=50, threaded=false)
    H, G = dad.H, dad.G
    u, t = ut_exact(dad, ana)
    Hu, Gt = H * u, G * t
    r = Hu .- Gt
    scale = norm(Hu) + norm(Gt) + 1e-30
    @printf("\n%s  near=%s  src=%s dir=%d  n=%d\n", tag, near, xs, dir, dad.n)
    @printf("  ||Hu||=%.3e  ||Gt||=%.3e  ||r||/scale=%.3e  max|r|=%.3e\n",
        norm(Hu), norm(Gt), norm(r)/scale, maximum(abs, r))

    # all-Dirichlet → t  (G)
    apply_analytical_bc!(dad, ana, Int[])
    solve(dad)
    relt = norm(dad.traction .- t) / (norm(t) + 1e-30)
    @printf("  all-Dirichlet  rel(t)=%.3e  cond=%.2e   [G]\n", relt, cond(Matrix(dad.A)))

    # all-Neumann with 2 pinned nodes → u  (H)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; H=H, G=G, H_hyper=H, G_hyper=G, nearfield=near)
    xspos = [p[1] for p in dad.Nodes]
    i1 = argmin(xspos); i2 = argmax(xspos)
    neu = setdiff(1:dad.n, unique([i1, i2]))
    apply_analytical_bc!(dad, ana, neu)
    solve(dad)
    relu = norm(dad.u .- u) / (norm(u) + 1e-30)
    @printf("  all-Neumann    rel(u)=%.3e  cond=%.2e   [H]\n", relu, cond(Matrix(dad.A)))

    # mixed: Dirichlet on x=0 (paper wall)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; H=H, G=G, H_hyper=H, G_hyper=G, nearfield=near)
    vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1.0]
    apply_analytical_bc!(dad, ana, setdiff(1:dad.n, vert))
    solve(dad)
    relm = norm(dad.u .- u) / (norm(u) + 1e-30)
    @printf("  mixed x=0 Dir  rel(u)=%.3e  max|u|=%.4e  cond=%.2e\n",
        relm, maximum(abs, dad.u), cond(Matrix(dad.A)))
end

msh_q2 = datadir("elastico", "p3_cmp_q2.msh")
msh_lin = datadir("elastico", "p3_cmp_lin.msh")
msh_p1 = datadir("elastico", "cordeiro_p1.msh")

# P1 control: source left of the rectangle
run(msh_p1, :euclid, SVector(-100.0, 100.0), 1, "P1 MAT1 far-left")

# P3: source in the hole (near inner arc) and far left
run(msh_q2, :euclid, SVector(150.0, 150.0), 1, "P3 q2 hole-src")
run(msh_q2, :euclid, SVector(-200.0, 450.0), 1, "P3 q2 far-left")
run(msh_lin, :euclid, SVector(150.0, 150.0), 1, "P3 chordal hole-src")
run(msh_q2, :tangent, SVector(150.0, 150.0), 1, "P3 q2 hole-src tangent")
println("\ndone")
