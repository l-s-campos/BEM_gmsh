# Validate DIBEM heterogeneous Laplace (Barcelos–Loeffler).
#
#   julia --project=. scripts/dibem/heterogeneous_laplace.jl
#
# 1) K=1 on quadrado (T=x) vs homogeneous assemble!/solve
# 2) Manufactured ∇·((1+y)∇x)=0 with Dirichlet T=x on vertical sides

using DrWatson
@quickactivate :BEM
using Statistics
using LinearAlgebra

function _report(label, T, Tex)
    e = abs.(T .- Tex)
    println(label, "  n=", length(T),
        "  med=", round(median(e); sigdigits=3),
        "  max=", round(maximum(e); sigdigits=3))
end

# --- homogeneous limit ---
dad = format2d(quadrado(ndiv=12, show=false, nome="het_k1"), Laplace(1.0);
    pontointerno=true, tipo=1)
assemble!(dad; npg=12, threaded=false)
solve(dad)
T0 = copy(dad.T)
solve_heterogeneous!(dad, p -> 1.0; rbf=PHS(1; poly_deg=-1))
_report("K=1 vs homogeneous", dad.T, T0)

# --- manufactured u=x, K=1+y ---
dad = format2d(quadrado(ndiv=12, show=false, nome="het_ky"), Laplace(1.0);
    pontointerno=true, tipo=1)
for i in 1:dad.n
    p = dad.Nodes[i]
    if p[1] < 1e-8
        dad.BC[i] = 0; dad.BV[i] = 0.0
    elseif p[1] > 1 - 1e-8
        dad.BC[i] = 0; dad.BV[i] = 1.0
    else
        dad.BC[i] = 1; dad.BV[i] = 0.0
    end
end
assemble!(dad; npg=12, threaded=false)
solve_heterogeneous!(dad, p -> 1 + p[2]; rbf=PHS(1; poly_deg=-1))
ux = [p[1] for p in all_points(dad)]
_report("K=1+y, u=x", dad.T, ux)
println("done")
