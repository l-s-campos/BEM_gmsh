# 3D Laplace collocation: mesh refinement vs analytical fields on the unit cube.
#
#   T = z              (linear, mixed Dirichlet/Neumann as in mesh_cube)
#   T = z              (linear, Dirichlet all faces)
#   T = x² + y² − 2z²  (quadratic harmonic, Dirichlet all faces)
#
using DrWatson
@quickactivate :BEM
using Printf
using LinearAlgebra
include(datadir("Laplace", "cube_mesh.jl"))

const L = 1.0
const NPG = 8
const NDIVS = (2, 3, 4, 6)

function _rate(h, e)
    length(e) < 2 && return fill(NaN, length(e))
    r = fill(NaN, length(e))
    for i in 2:length(e)
        (e[i-1] > 0 && e[i] > 0 && h[i-1] != h[i]) || continue
        r[i] = log(e[i-1] / e[i]) / log(h[i-1] / h[i])
    end
    return r
end

function _print_table(title, nds, ns, hs, eT, eQ)
    rT = _rate(hs, eT)
    rQ = _rate(hs, eQ)
    println("\n", title)
    @printf("  %4s  %6s  %8s  %9s  %6s  %9s  %6s\n",
        "ndiv", "n", "h", "rel T", "rateT", "rel q", "rateQ")
    for i in eachindex(nds)
        rTs = isnan(rT[i]) ? "—" : @sprintf("%.2f", rT[i])
        rQs = isnan(rQ[i]) ? "—" : @sprintf("%.2f", rQ[i])
        @printf("  %4d  %6d  %8.4f  %9.2e  %6s  %9.2e  %6s\n",
            nds[i], ns[i], hs[i], eT[i], rTs, eQ[i], rQs)
    end
end

function _side_nodes(dad)
    return [i for i in 1:dad.n if abs(dad.Normal[i][3]) < 0.5]
end

function run_case(name, ndivs; ana, bc::Symbol=:mixed_z, interior::Bool=false)
    hs = Float64[]; eT = Float64[]; eQ = Float64[]; ns = Int[]; nds = Int[]
    for ndiv in ndivs
        msh = bc === :mixed_z ?
            mesh_cube(; L=L, ndiv=ndiv, nome="cv3d_mix_$ndiv") :
            mesh_unit_cube(; L=L, ndiv=ndiv, nome="cv3d_d_$ndiv", bc="0;0")
        dad = format3d(msh, Laplace(1.0); pontointerno=false)
        if interior
            set_internal_nodes!(dad, vec(cube_interior_grid(L, 2)))
        end
        if bc === :mixed_z
            attach_analytical!(dad, ana)
        elseif bc === :dirichlet
            apply_analytical_bc!(dad, ana)
        elseif bc === :mixed_sides
            apply_analytical_bc!(dad, ana, _side_nodes(dad))
        else
            error("unknown bc=$bc")
        end
        assemble!(dad; npg=NPG, threaded=true)
        solve(dad)
        push!(nds, ndiv)
        push!(ns, dad.n)
        push!(hs, L / ndiv)
        push!(eT, rel_error(dad))
        push!(eQ, rel_error_flux(dad))
    end
    _print_table(name, nds, ns, hs, eT, eQ)
    return nothing
end

println("="^72)
println(" 3D Laplace BEM — unit cube convergence (npg=$NPG, tipo=1 quads)")
println("="^72)

ana_z = ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0], k=1.0)
ana_q = ana_laplace_quadratic(; k=1.0, dim=3)

run_case("Linear T=z, mixed BC (Dirichlet top/bottom, q=0 sides)", NDIVS;
    ana=ana_z, bc=:mixed_z)
run_case("Quadratic T=x²+y²−2z², Dirichlet top/bottom, Neumann sides", NDIVS;
    ana=ana_q, bc=:mixed_sides)
run_case("Linear T=z, Dirichlet all faces + 8 interior probes (rel T; rel q is dual recovery)", NDIVS;
    ana=ana_z, bc=:dirichlet, interior=true)

println("\nDone.")
