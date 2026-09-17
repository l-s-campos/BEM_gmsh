# Hypersingular H: interpolation Laurent coefficients vs Richardson.
# Laplace and Kelvin HBIE on a square; dual-BEM H on a small centre crack.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Crack

const NPG = 12
const NINTERP = 20

rel(A, B) = norm(A - B) / max(norm(B), 1e-16)
amax(A, B) = maximum(abs, A - B)

"""Frobenius relative error of on-element (self) blocks of `H`."""
function rel_self(ΔH, Href, dad; dim::Int=1)
    accΔ = 0.0
    accH = 0.0
    nR, nC = size(ΔH)
    for el in dad.elements
        cols = Int[]
        for j in el.index
            append!(cols, dim == 1 ? (j,) : collect(BEM.expand(j, dim)))
        end
        rows = filter(i -> 1 <= i <= nR, cols)
        cols = filter(j -> 1 <= j <= nC, cols)
        isempty(rows) && continue
        accΔ += sum(abs2, view(ΔH, rows, cols))
        accH += sum(abs2, view(Href, rows, cols))
    end
    return sqrt(accΔ) / max(sqrt(accH), 1e-16)
end

"""||H[source, twin]|| over crack-face pairs."""
function twin_coupling(H, dad; dim::Int=1)
    has_cache(dad, :twin) || return 0.0
    acc = 0.0
    for i in 1:dad.n
        tw = dad.twin[i]
        tw == 0 && continue
        ii = dim == 1 ? (i:i) : BEM.expand(i, dim)
        tt = dim == 1 ? (tw:tw) : BEM.expand(tw, dim)
        acc += sum(abs2, view(H, ii, tt))
    end
    return sqrt(acc)
end

function report(label, Hi, Gi, Hr, Gr, Ha, Ga, dad; dim=1)
    println("  ", label, "   n=", dad.n, "  elems=", length(dad.elements),
        "  size(H)=", size(Hi))
    @printf "    ||H||  interp=%9.3e  Rich.=%9.3e  auto=%9.3e\n" norm(Hi) norm(Hr) norm(Ha)
    @printf "    rel H  interp vs Rich.  = %.3e    max|Δ|=%.3e\n" rel(Hi, Hr) amax(Hi, Hr)
    @printf "    rel H  interp vs auto   = %.3e    max|Δ|=%.3e\n" rel(Hi, Ha) amax(Hi, Ha)
    @printf "    rel H  Rich.  vs auto   = %.3e    max|Δ|=%.3e\n" rel(Hr, Ha) amax(Hr, Ha)
    @printf "    self-block rel  I/R=%.3e  I/auto=%.3e  R/auto=%.3e\n" rel_self(Hi - Hr, Hr, dad; dim=dim) rel_self(Hi - Ha, Ha, dad; dim=dim) rel_self(Hr - Ha, Ha, dad; dim=dim)
    @printf "    rel G  interp vs Rich.  = %.3e    interp vs auto = %.3e\n" rel(Gi, Gr) rel(Gi, Ga)
    if has_cache(dad, :twin) && any(!iszero, dad.twin)
        ti, tr, ta = twin_coupling(Hi, dad; dim=dim), twin_coupling(Hr, dad; dim=dim),
            twin_coupling(Ha, dad; dim=dim)
        @printf "    twin ||H_i,twin||  interp=%.3e  Rich.=%.3e  auto=%.3e\n" ti tr ta
    end
end

function assemble_hyper(build, hyper!; dim=1)
    d_i = build()
    d_r = build()
    d_a = build()
    Hi, Gi = hyper!(d_i, :interp)
    Hr, Gr = hyper!(d_r, :richardson)
    Ha, Ga = hyper!(d_a, :auto)
    return Hi, Gi, Hr, Gr, Ha, Ga, d_i
end

println("="^78)
println("Hypersingular H: interpolation vs Richardson vs closed-form (:auto)")
println("  npg=$NPG   ninterp=$NINTERP")
println("="^78)

# -----------------------------------------------------------------------------
println()
println("1) Laplace HBIE  H_G_hyper  (square, T=x mesh)")
build_lap() = format2d(quadrado(ndiv=6, show=false, nome="hcmp_lap"), Laplace(1.0);
    pontointerno=false, tipo=1)
Hi, Gi, Hr, Gr, Ha, Ga, dad = assemble_hyper(build_lap,
    (d, m) -> H_G_hyper(d; npg=NPG, threaded=false, laurent=m, ninterp=NINTERP))
report("Laplace H′", Hi, Gi, Hr, Gr, Ha, Ga, dad)

# -----------------------------------------------------------------------------
println()
println("2) Kelvin HBIE  H_G_hyper  (square, plane strain)")
build_el() = format2d(quadrado_elasticity(ndiv=6, show=false, nome="hcmp_el"),
    Elasticity(1.0, 0.3, 1.0; plane_strain=true); pontointerno=false, tipo=1)
Hi, Gi, Hr, Gr, Ha, Ga, dad = assemble_hyper(build_el,
    (d, m) -> H_G_hyper(d; npg=NPG, threaded=false, laurent=m, ninterp=NINTERP);
    dim=2)
report("Kelvin H′", Hi, Gi, Hr, Gr, Ha, Ga, dad; dim=2)

# -----------------------------------------------------------------------------
println()
println("3) Laplace dual BEM H  (centre crack, HBIE on face B)")
build_dl() = dual_laplace_problem(; W=5.0, H=10.0, a=1.0, ndiv_b=4, ndiv_h=6,
    ndiv_crack=6, ordem=1, nome="hcmp_dlap", pontointerno=false)
Hi, Gi, Hr, Gr, Ha, Ga, dad = assemble_hyper(build_dl,
    (d, m) -> assemble_dual_laplace!(d; npg=NPG, threaded=false, laurent=m,
        ninterp=NINTERP))
report("Laplace dual H", Hi, Gi, Hr, Gr, Ha, Ga, dad)
ih = findall(==(3), dad.eq_type)
if !isempty(ih)
    @printf "    HBIE rows only  rel I/R=%.3e  I/auto=%.3e  R/auto=%.3e\n" rel(Hi[ih, :], Hr[ih, :]) rel(Hi[ih, :], Ha[ih, :]) rel(Hr[ih, :], Ha[ih, :])
end

# -----------------------------------------------------------------------------
println()
println("4) Elasticity dual BEM H  (centre crack, HBIE on face B)")
build_de() = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0,
    E=1.0, ν=0.3, n_bottom=4, n_right=6, n_top=4, n_left=6, n_crack=6)
Hi, Gi, Hr, Gr, Ha, Ga, dad = assemble_hyper(build_de,
    (d, m) -> assemble_dual!(d; npg=NPG, threaded=false, laurent=m,
        ninterp=NINTERP); dim=2)
report("Kelvin dual H", Hi, Gi, Hr, Gr, Ha, Ga, dad; dim=2)
ih = findall(==(3), dad.eq_type)
if !isempty(ih)
    rows = reduce(vcat, [collect(BEM.expand(i, 2)) for i in ih])
    @printf "    HBIE rows only  rel I/R=%.3e  I/auto=%.3e  R/auto=%.3e\n" rel(Hi[rows, :], Hr[rows, :]) rel(Hi[rows, :], Ha[rows, :]) rel(Hr[rows, :], Ha[rows, :])
end

println()
println("done.")
