using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

# bar with P=0, constant body force b_x=1, no end load
dad, meta = elasticity_bar_sudden(; ndiv=8, n_int=0, ν=0.0, P=0.0)
H_G_full_direct(dad; npg=8, threaded=false)

# DRM (with MQ F from the edit)
drm = build_drm_matrices(dad; npg=8)
M = dad.M
H = dad.H
G = dad.G

# constant body force b = (1,0) at all collocation points
bvec = zeros(2 * dad.nt)
for i in 1:dad.nt
    bvec[2*i-1] = 1.0   # b_x =1
end

# Known particular for constant b (ν=0)
up_b = zeros(2 * dad.n)
for i in 1:dad.n
    x = dad.Nodes[i][1]
    up_b[2*i-1] = -0.5 * x^2 + x
end
tp = zeros(2 * dad.n)
for i in 1:dad.n
    n = dad.Normal[i]
    x = dad.Nodes[i][1]
    dux = -x + 1
    if n[1] > 0.5
        tp[2*i-1] = 0.0
    elseif n[1] < -0.5
        tp[2*i-1] = -1.0 * dux
    else
        tp[2*i-1] = 0.0
    end
end
domain_term = H[1:2*dad.n, 1:2*dad.n] * up_b - G * tp   # exact domain for constant b (boundary block approx)

# rhs for solve = exact domain
applyBC(dad)
dad.b .+= domain_term

x = bem_linsolve(dad.A, dad.b)

u = zeros(2 * dad.nt)
traction = zeros(2 * dad.n)
split_sol!(dad, x, u, traction)

# right end nodes (x≈1)
right_idx = findall(p -> p[1] > 0.99, dad.Nodes)
ux_right = [u[2*(i-1)+1] for i in right_idx]
uy_right = [u[2*(i-1)+2] for i in right_idx]

E = dad.properties.E
@printf("E=%.4f G=%.4f\n", E, shear_modulus(dad.properties))
@printf("u_x at right (analytic 0.5): avg=%.4f min=%.4f max=%.4f\n", mean(ux_right), minimum(ux_right), maximum(ux_right))
@printf("u_y at right (should ~0): avg=%.4e\n", mean(uy_right))

# Test the DRM mass for constant b by comparing M * b with the exact domain term
domain_from_m = M * bvec
diff = norm(domain_term - domain_from_m)
@printf("Domain term diff (exact vs M*b with MQ F): %.4e\n", diff)
@printf("Max |domain_term| = %.4e\n", maximum(abs, domain_term))
