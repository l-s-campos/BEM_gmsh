# 3-D DIBEM-SIMP / DT-ρ on a fixed cube (heat) or box cantilever (elasticity).
#
#   julia --project=. scripts/topology/dibem_simp_3d.jl
#   julia --project=. scripts/topology/dibem_simp_3d.jl --physics=heat --nsimp=6
#   julia --project=. scripts/topology/dibem_simp_3d.jl --physics=elast --nsimp=4

using DrWatson
@quickactivate :BEM
using BEM.Topology
using Printf
using Statistics

const OUT = datadir("topology", "3d")
mkpath(OUT)

physics = :heat
n_simp = 4
ndiv = 2
nint = 2
volfrac = 0.45
do_vtk = true

for a in ARGS
    if startswith(a, "--physics=")
        global physics = Symbol(split(a, "=")[2])
    elseif startswith(a, "--nsimp=")
        global n_simp = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ndiv=")
        global ndiv = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--nint=")
        global nint = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--volfrac=")
        global volfrac = parse(Float64, split(a, "=")[2])
    elseif a == "--no-vtk"
        global do_vtk = false
    end
end

opt = DibemSimpOptions(; volfrac=volfrac, n_simp=n_simp, rmin=0.25, cut=true,
    pacheco=false, verbose=true, npg=8, p_start=1.0, p=3.0, ngrid=21)

println("="^72)
println(" 3-D DIBEM-SIMP  physics=$physics  ndiv=$ndiv  nint=$nint  n_simp=$n_simp")
println("="^72)

if physics === :heat
    dad = heat_cube_3d(; ndiv=ndiv, nint=nint, degree=1)
    assemble!(dad; npg=opt.npg, threaded=false)
    dad, ρ, hist = solve_dibem_simp!(dad, opt)
    C = thermal_conductance(dad)
    @printf("  C = %.4g   V = %.3f   gray = %.3f   nt = %d\n",
        C, hist[end].V, hist[end].gray, dad.nt)
    if do_vtk
        vtk = joinpath(OUT, "heat_cube_simp.vtk")
        export_vtk_density(dad, ρ, vtk)
        println("  wrote $vtk")
        if has_cache(dad, :simp_iso)
            iso = joinpath(OUT, "heat_cube_iso.vtk")
            export_vtk_isosurface(dad.simp_iso, iso)
            println("  wrote $iso")
        end
    end
elseif physics === :elast || physics === :elasticity
    dad = cantilever_cube_3d(; ndiv=ndiv, nint=nint, degree=1, E=1.0, ν=0.3)
    assemble!(dad; npg=opt.npg, threaded=false)
    dad, ρ, hist = solve_dibem_simp!(dad, opt)
    J = elastic_compliance(dad)
    @printf("  J = %.4g   V = %.3f   gray = %.3f   nt = %d\n",
        J, hist[end].V, hist[end].gray, dad.nt)
    if do_vtk
        vtk = joinpath(OUT, "cantilever_simp.vtk")
        export_vtk_density(dad, ρ, vtk)
        println("  wrote $vtk")
        if has_cache(dad, :simp_iso)
            iso = joinpath(OUT, "cantilever_iso.vtk")
            export_vtk_isosurface(dad.simp_iso, iso)
            println("  wrote $iso")
        end
    end
elseif physics === :dt
    dad = heat_cube_3d(; ndiv=ndiv, nint=nint, degree=1)
    assemble!(dad; npg=opt.npg, threaded=false)
    opt.method = :dt
    dad, ρ, hist = solve_dibem_simp!(dad, opt)
    @printf("  DT-ρ  C = %.4g   V = %.3f   nt = %d\n", hist[end].C, hist[end].V, dad.nt)
    if do_vtk
        vtk = joinpath(OUT, "heat_cube_dt.vtk")
        export_vtk_density(dad, ρ, vtk; DT=nodal_DT(dad))
        println("  wrote $vtk")
        if has_cache(dad, :simp_iso)
            iso = joinpath(OUT, "heat_cube_dt_iso.vtk")
            export_vtk_isosurface(dad.simp_iso, iso)
            println("  wrote $iso")
        end
    end
else
    error("unknown --physics=$physics (heat | elast | dt)")
end
