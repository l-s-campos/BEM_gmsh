# Quadrature check for elasticity mass: current (npg=8 / nodal DIBEM) vs 100 GL.
# julia --project=. scripts/compare_M_npg.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

function rel(A, B)
    return norm(A - B) / (norm(B) + 1e-30)
end

function make_dad()
    msh = mesh_elasticity_bar(; ndiv=8, L=1.0, P=1.0, nome="mnpg8")
    dad = format2d(msh, Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true);
        tipo=1, pontointerno=true)
    return dad
end

function report(tag, A, B)
    @printf("  %-18s  relF=%.3e  max|Δ|=%.3e  |A|=%.3e  |B|=%.3e\n",
        tag, rel(A, B), maximum(abs, A - B), norm(A), norm(B))
end

println("========== DRM M: H,G at npg=8 (scripts) vs npg=100 ==========")
dad8 = make_dad()
H_G_full_direct(dad8; npg=8, threaded=false)
drm8 = build_drm_matrices(dad8; npg=8, kernel=:r)

dad100 = make_dad()
H_G_full_direct(dad100; npg=100, threaded=false)
drm100 = build_drm_matrices(dad100; npg=100, kernel=:r)

H8, G8, M8 = Matrix(dad8.H), Matrix(dad8.G), drm8.M
H1, G1, M1 = Matrix(dad100.H), Matrix(dad100.G), drm100.M
report("H", H8, H1)
report("G", G8, G1)
report("DRM M (f=r)", M8, M1)
@printf("  cos(M8,M100)=%.6f  |tr8-tr100|/|tr100|=%.3e\n",
    dot(vec(M8), vec(M1)) / (norm(M8) * norm(M1) + 1e-30),
    abs(tr(M8) - tr(M1)) / (abs(tr(M1)) + 1e-30))
x = ones(size(M1, 1))
@printf("  ||(M8-M100)x|| / ||M100 x||  (x=1) = %.3e\n",
    norm(M8 * x - M1 * x) / (norm(M1 * x) + 1e-30))
x = randn(size(M1, 1))
@printf("  ||(M8-M100)x|| / ||M100 x||  (randn) = %.3e\n",
    norm(M8 * x - M1 * x) / (norm(M1 * x) + 1e-30))

println("\n========== DRM also f=1+r ==========")
dad8b = make_dad(); H_G_full_direct(dad8b; npg=8, threaded=false)
M8b = build_drm_matrices(dad8b; npg=8, kernel=:one_plus_r).M
dad1b = make_dad(); H_G_full_direct(dad1b; npg=100, threaded=false)
M1b = build_drm_matrices(dad1b; npg=100, kernel=:one_plus_r).M
report("DRM M (1+r)", M8b, M1b)

println("\n========== DIBEM M: nodal lumping (current) vs Gauss-100 IF/ID ==========")
# Current DIBEM uses dad.elem_weight (collocation lumping), independent of npg.
dadN = make_dad()
H_G_full_direct(dadN; npg=8, threaded=false)
MN = DIBEM_dense(dadN; rbf=PHS(1; poly_deg=-1))

# Rebuild IF, ID with npg=100 Gauss on every element (same F, D as dense DIBEM).
function dibem_M_gauss(dad; npg=100, rbf=PHS(1; poly_deg=-1))
    dim = 2
    nt = dad.nt
    props = dad.properties
    n0 = dad.Normal[1]
    ηs, ws = gausslegendre(npg)
    F = zeros(nt, nt)
    D = zeros(2nt, 2nt)
    IF = zeros(nt)
    ID = zeros(2nt, 2)
    for j in 1:nt, i in 1:nt
        x = point(dad, i)
        xj = point(dad, j)
        r = norm(x - xj)
        F[i, j] = rbf(r)
        if r > 0
            U, _ = fundamental(dad, xj - x, n0)
            D[2i-1:2i, 2j-1:2j] .= U
        end
    end
    ε = 1e-12 * (sum(abs, F) / max(nt * (nt - 1), 1) + 1)
    for i in 1:nt
        F[i, i] += ε
    end
    for i in 1:nt
        x = point(dad, i)
        for el in dad.elements
            nodes = dad.Nodes[el.index]
            N, dN = BEM.shapefun(dad.element_type, ηs)
            pg = N * nodes
            dx = dN * nodes
            nref = dad.Normal[el.index[1]]
            for q in eachindex(ηs)
                Jv = dx[q]
                J = norm(Jv)
                J < 1e-16 && continue
                n = BEM.tan2normal(Jv / J)
                n ⋅ nref < 0 && (n = -n)
                y = pg[q]
                r = y - x
                R = norm(r)
                R < 1e-14 && continue
                wJn = ws[q] * J * (n ⋅ r) / R^2
                IF[i] += BEM.int(rbf, x, y) * wJn
                e = r / R
                ID[2i-1:2i, :] .+= BEM._galerkin_Ustar(props, R, e) * wJn
            end
        end
    end
    c = vec(IF' / F)
    M = zeros(2nt, 2nt)
    for j in 1:nt
        a = c[j]
        M[:, 2j-1] .= a .* D[:, 2j-1]
        M[:, 2j] .= a .* D[:, 2j]
    end
    for i in 1:nt
        rows = 2i-1:2i
        M[rows, rows] .= 0
        s1 = sum(view(M, rows, 1:2:2nt); dims=2)
        s2 = sum(view(M, rows, 2:2:2nt); dims=2)
        M[rows, rows] .= .-hcat(s1, s2) .+ ID[rows, :]
    end
    return M, IF, ID
end

MG, IFg, IDg = dibem_M_gauss(dadN; npg=100)
IDn = dadN.dibem_ID
report("DIBEM M", MN, MG)
@printf("  cos=%.6f  |ID_nodal - ID_G100|/|ID_G100|=%.3e\n",
    dot(vec(MN), vec(MG)) / (norm(MN) * norm(MG) + 1e-30),
    rel(IDn, IDg))
x = ones(size(MG, 1))
@printf("  ||(Mn-Mg)x|| / ||Mg x||  (x=1) = %.3e\n",
    norm(MN * x - MG * x) / (norm(MG * x) + 1e-30))
x = randn(size(MG, 1))
@printf("  ||(Mn-Mg)x|| / ||Mg x||  (randn) = %.3e\n",
    norm(MN * x - MG * x) / (norm(MG * x) + 1e-30))
nneg(A) = count(<( -1e-8), real.(eigvals(A)))
@printf("  nneg nodal=%d  nneg G100=%d  tr nodal=%.4f  tr G100=%.4f\n",
    nneg(MN), nneg(MG), tr(MN), tr(MG))
