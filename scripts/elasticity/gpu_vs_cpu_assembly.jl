# Compare 2-D Kelvin assembly + DIBEM: CPU vs KernelAbstractions GPU.
#
#   julia --project=. scripts/elasticity/gpu_vs_cpu_assembly.jl

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

const NPG = parse(Int, get(ENV, "GPU_NPG", "12"))
const OUTPATH = get(ENV, "GPU_OUT", projectdir("plots", "gpu_vs_cpu_elasticity.tsv"))
_parse_ints(s) = parse.(Int, split(s, ','; keepempty=false))
const NDIVS = _parse_ints(get(ENV, "GPU_NDIVS", "8,16,32"))
const RBF = PHS(3; poly_deg=1)

const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

function _rel(A, B)
    nb = norm(B)
    return nb > 0 ? norm(A .- B) / nb : (norm(A) == 0 ? 0.0 : Inf)
end

function _make_dad(ndiv, nome)
    props = Elasticity(1.0, 0.3, 1.0)
    dad = format2d(quadrado_elasticity(ndiv=ndiv, show=false, nome=nome), props;
        pontointerno=false)
    apply_analytical_bc!(dad, ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01))
    return dad
end

function main()
    println("="^72)
    println(" Kelvin 2-D: CPU vs KernelAbstractions GPU")
    println(" ", Dates.now(), "  threads=", Threads.nthreads())
    println("="^72)
    info = gpu_float_support()
    println("GPU: available=", info.available, "  name=", info.name,
        "  cap=", info.capability)
    wdev = HAVE_CUDA ? :cuda : :cpu
    HAVE_CUDA || println("CUDA not functional — using device=:cpu")

    w = _make_dad(4, "el_gpu_warmup")
    assemble!(w, 8; threaded=false)
    H_G_gpu(_make_dad(4, "el_gpu_warmup_k"); T=Float64, npg=8, device=wdev,
        threaded=false)

    devices = HAVE_CUDA ? (:cuda,) : (:cpu,)
    types = HAVE_CUDA ? (Float32, Float64) : (Float64,)
    rows = Any[]
    @printf("%6s %8s %6s %10s %10s %9s %9s %10s %10s\n",
        "ndiv", "n", "T", "t_asm_cpu", "t_asm_gpu", "relH", "relG",
        "err_cpu", "err_gpu")
    for ndiv in NDIVS, gpuT in types, device in devices
        dad_cpu = _make_dad(ndiv, "el_cmp_cpu_$(ndiv)_$(gpuT)")
        t_cpu = @elapsed assemble!(dad_cpu, NPG; threaded=true)
        solve(dad_cpu)
        err_cpu = rel_error(dad_cpu)

        dad_gpu = _make_dad(ndiv, "el_cmp_gpu_$(ndiv)_$(gpuT)")
        t_gpu = @elapsed H_G_gpu(dad_gpu; T=gpuT, npg=NPG, near=:cpu,
            device=device, threaded=true)
        relH = _rel(Float64.(dad_gpu.H), dad_cpu.H)
        relG = _rel(Float64.(dad_gpu.G), dad_cpu.G)
        solve(dad_gpu)
        err_gpu = rel_error(dad_gpu)
        @printf("%6d %8d %6s %10.4f %10.4f %9.2e %9.2e %10.3e %10.3e\n",
            ndiv, dad_cpu.n, string(gpuT), t_cpu, t_gpu, relH, relG, err_cpu, err_gpu)
        push!(rows, (; ndiv, n=dad_cpu.n, T=string(gpuT), t_cpu, t_gpu, relH, relG,
            err_cpu, err_gpu))
    end

    println("\nDIBEM M (Float64)")
    @printf("%6s %8s %10s %10s %10s\n", "ndiv", "nt", "t_cpu", "t_gpu", "relM")
    for ndiv in NDIVS
        d0 = _make_dad(ndiv, "el_dibem_cpu_$ndiv")
        t0 = @elapsed DIBEM(d0; rbf=RBF, npg=NPG)
        d1 = _make_dad(ndiv, "el_dibem_gpu_$ndiv")
        t1 = @elapsed DIBEM_gpu(d1; rbf=RBF, T=Float64, npg=NPG, device=wdev)
        relM = _rel(d1.M, d0.M)
        @printf("%6d %8d %10.4f %10.4f %10.2e\n", ndiv, d0.nt, t0, t1, relM)
    end

    mkpath(dirname(OUTPATH))
    open(OUTPATH, "w") do io
        println(io, "ndiv\tn\tT\tt_asm_cpu\tt_asm_gpu\trelH\trelG\terr_cpu\terr_gpu")
        for r in rows
            @printf(io, "%d\t%d\t%s\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\n",
                r.ndiv, r.n, r.T, r.t_cpu, r.t_gpu, r.relH, r.relG, r.err_cpu, r.err_gpu)
        end
    end
    println("\nWrote ", OUTPATH)
    return nothing
end

main()
