# 1-stick residuals + specimen-only compression test.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const MR = BEM.MultiRegion

function print_n(dad, label, i)
    n = dad.Normal[i]; p = dad.Nodes[i]
    @printf("  %s i=%d x=(%.3f,%.3f) n=(%+.3f,%+.3f) BC=(%d,%d) BV=(%.3f,%.3f)\n",
        label, i, p[1], p[2], n[1], n[2], dad.BC[2i-1], dad.BC[2i], dad.BV[2i-1], dad.BV[2i])
end

prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    gap=:euclidean, nome="juyfix")
dad1, dad2 = prob.regions
ymax = maximum(pt[2] for pt in dad1.Nodes)
ymin = minimum(pt[2] for pt in dad2.Nodes)
itop = argmin(i -> (abs(dad1.Nodes[i][2] - ymax), abs(dad1.Nodes[i][1])), 1:dad1.n)
ibot = argmin(i -> (abs(dad2.Nodes[i][2] - ymin), abs(dad2.Nodes[i][1])), 1:dad2.n)
ic1 = argmin(i -> abs(dad1.Nodes[i][1]) + 10*abs(dad1.Normal[i][2] + 1), 1:dad1.n)
ic2 = argmin(i -> abs(dad2.Nodes[i][1]) + 10*abs(dad2.Normal[i][2] - 1), 1:dad2.n)
println("=== normals / BC ===")
print_n(dad1, "pad top", itop)
print_n(dad1, "pad contact", ic1)
print_n(dad2, "spec contact", ic2)
print_n(dad2, "spec bottom", ibot)

# --- specimen alone: pressure on contact, uy=0 on bottom ---
println("\n=== specimen-only: ty=-1 on contact (global), one local convert ===")
dadS = deepcopy(dad2)
let
    for i in 1:dadS.n
        if abs(dadS.Normal[i][2] - 1) < 0.05
            dadS.BC[2i-1] = 1; dadS.BV[2i-1] = 0.0
            dadS.BC[2i]     = 1; dadS.BV[2i]     = -1.0
        end
    end
end
H_G_full_direct(dadS; npg=20, near_factor=Inf)
BC = copy(dadS.BC); BV = copy(dadS.BV)
MR._exterior_bc_to_local!(BC, BV, dadS)
dadS.BC .= BC; dadS.BV .= BV
ndir = count(==(0), dadS.BC)
@printf("spec Dirichlet dofs after local=%d  sample contact BV=(%.3f,%.3f)\n",
    ndir, dadS.BV[2ic2-1], dadS.BV[2ic2])
BEM.applyBC_local!(dadS)
xS = BEM.bem_linsolve(dadS.A, dadS.b)
nd = 2*dadS.n
u_loc = zeros(nd); t_loc = zeros(nd)
BEM._split_sol_local!(dadS, xS, u_loc, t_loc)
u_g = BEM.local_to_global_field(dadS, u_loc)
@printf("spec contact i=%d uy=%.4e un_loc=%.4e  (want uy<0)\n",
    ic2, u_g[2ic2], u_loc[2ic2-1])
@printf("spec bottom  i=%d uy=%.4e ty=%.4e\n", ibot, u_g[2ibot],
    BEM.local_to_global_field(dadS, t_loc)[2ibot])

println("\n=== specimen-only GLOBAL frame: ty=-1 on contact ===")
dadG = deepcopy(dad2)
for i in 1:dadG.n
    if abs(dadG.Normal[i][2] - 1) < 0.05
        dadG.BC[2i-1] = 1; dadG.BV[2i-1] = 0.0
        dadG.BC[2i]     = 1; dadG.BV[2i]     = -1.0
    end
end
H_G_full_direct(dadG; npg=20, near_factor=Inf)
solve(dadG)  # global
uys = Float64[]
for i in 1:dadG.n
    abs(dadG.Normal[i][2] - 1) < 0.05 && push!(uys, dadG.u[2i])
end
@printf("GLOBAL spec contact uy center=%.4e  min=%.4e max=%.4e mean=%.4e  (want all <0)\n",
    dadG.u[2ic2], minimum(uys), maximum(uys), sum(uys)/length(uys))
@printf("  bottom uy=%.4e ty=%.4e  n_contact_uy=%d\n",
    dadG.u[2ibot], dadG.traction[2ibot], length(uys))

# --- pad alone: ty=-1 on top, tn=0 on contact, pin one ut ---
println("\n=== pad-only: ty=-1 on top, contact free, pin top-centre ut ===")
dadP = deepcopy(dad1)
H_G_full_direct(dadP; npg=20, near_factor=Inf)
BC = copy(dadP.BC); BV = copy(dadP.BV)
MR._exterior_bc_to_local!(BC, BV, dadP)
dadP.BC .= BC; dadP.BV .= BV
BEM.applyBC_local!(dadP)
xP = BEM.bem_linsolve(dadP.A, dadP.b)
uP = zeros(2dadP.n); tP = zeros(2dadP.n)
BEM._split_sol_local!(dadP, xP, uP, tP)
ugP = BEM.local_to_global_field(dadP, uP)
@printf("pad top uy=%.4e  contact uy=%.4e  (top load down → both should be <0 if free drop)\n",
    ugP[2itop], ugP[2ic1])

# --- coupled 1-stick residual ---
println("\n=== coupled 1-stick ===")
ctx = MR._contact_friction_setup(prob; method=:ntn, npg=20, common_normal=false,
    near_factor=Inf)
println("pad Dirichlet dofs=", count(==(0), ctx.prep[1].BC_ext),
    " spec Dirichlet dofs=", count(==(0), ctx.prep[2].BC_ext))
prep, pairs = ctx.prep, ctx.pairs
h = [cp.gap0 for cp in pairs]
nx = sum(p.ndof for p in prep)
nsteps = 50
x = zeros(ctx.N)
MR._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
A, b = MR._assemble_contact_system(prep, pairs, h, x)
b[1:nx] ./= nsteps
for (k, cp) in enumerate(pairs)
    abs(cp.state) == 1 && continue
    b[nx+4(k-1)+1] = h[k]
end
x = A \ b
MR._scatter_contact_solution!(prob, prep, pairs, x)
pr1, pr2 = prep[1], prep[2]
@printf("pad top uy=%.4e  pad contact uy=%.4e\n", dad1.u[2itop], dad1.u[2ic1])
@printf("spec contact uy=%.4e  spec bottom uy=%.4e  spec bottom ty=%.4e\n",
    dad2.u[2ic2], dad2.u[2ibot], dad2.traction[2ibot])
for (name, pr) in (("pad", pr1), ("spec", pr2))
    dad = pr.dad
    Hl, Gl = BEM.transform_HG_local(dad.H, dad.G, dad)
    ul = dad.u_local; tl = dad.traction_local
    r = Hl[1:pr.ndof, 1:pr.ndof] * ul - Gl[1:pr.ndof, 1:pr.ndof] * tl
    @printf("%s ||H u - G t||=%.3e  max|res|=%.3e\n", name, norm(r), maximum(abs, r))
end
println("Done.")
