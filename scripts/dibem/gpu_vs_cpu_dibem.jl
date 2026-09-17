# Compare 2-D Laplace DIBEM: CPU (`DIBEM_dense`) vs GPU (`DIBEM_gpu`).
#
#   julia --project=. scripts/dibem/gpu_vs_cpu_dibem.jl
#
# ENV:
#   GPU_NDIVS=12,24,40
#   GPU_NPG=12
#   GPU_OUT=plots/gpu_vs_cpu_dibem.tsv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

const NPG = parse(Int, get(ENV, "GPU_NPG", "12"))
const OUTPATH = get(ENV, "GPU_OUT", projectdir("plots", "gpu_vs_cpu_dibem.tsv"))
_parse_ints(s) = parse.(Int, split(s, ','; keepempty=false))
const NDIVS = _parse_ints(get(ENV, "GPU_NDIVS", "12,24,40"))
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
    return format2d(quadrado(ndiv=ndiv, show=false, nome=nome), Laplace(1.0);
        pontointerno=true)
end

function _run_one(ndiv, gpuT, device)
    dad_cpu = _make_dad(ndiv, "dibem_cmp_cpu_$(ndiv)_$(gpuT)")
    t_cpu = @elapsed DIBEM(dad_cpu; rbf=RBF, npg=NPG)
    onesv = ones(dad_cpu.nt)
    res_cpu = norm(dad_cpu.M * onesv - dad_cpu.dibem_ID) /
        (norm(dad_cpu.dibem_ID) + 1e-14)

    dad_gpu = _make_dad(ndiv, "dibem_cmp_gpu_$(ndiv)_$(gpuT)_$(device)")
    t_gpu = @elapsed DIBEM_gpu(dad_gpu; rbf=RBF, T=gpuT, npg=NPG,
        device=device, threaded=true)
    relM = _rel(dad_gpu.M, dad_cpu.M)
    res_gpu = norm(dad_gpu.M * onesv - dad_gpu.dibem_ID) /
        (norm(dad_gpu.dibem_ID) + 1e-14)
    return (
        n = dad_cpu.n, nt = dad_cpu.nt,
        t_cpu = t_cpu, t_gpu = t_gpu, relM = relM,
        res_cpu = res_cpu, res_gpu = res_gpu,
    )
end

function main()
    println("="^72)
    println(" Laplace 2-D DIBEM: CPU vs KernelAbstractions GPU")
    println(" ", Dates.now())
    println(" threads = ", Threads.nthreads(), "   npg = ", NPG, "   rbf = PHS3+lin")
    println("="^72)
    info = gpu_float_support()
    println("GPU: available=", info.available, "  name=", info.name,
        "  cap=", info.capability, "  recommended=", info.recommended)
    HAVE_CUDA || println("CUDA not functional — GPU rows use device=:cpu.\n")

    wdev = HAVE_CUDA ? :cuda : :cpu
    DIBEM(_make_dad(6, "dibem_cmp_warmup"); rbf=RBF, npg=8)
    DIBEM_gpu(_make_dad(6, "dibem_cmp_warmup_k"); rbf=RBF, T=Float32, npg=8,
        device=wdev, threaded=false)

    devices = HAVE_CUDA ? (:cuda,) : (:cpu,)
    types = HAVE_CUDA ? (Float32, Float64) : (Float64,)
    rows = Any[]
    @printf("%6s %8s %6s %10s %10s %10s %10s %10s\n",
        "ndiv", "nt", "T", "t_cpu", "t_gpu", "relM", "res_cpu", "res_gpu")
    for ndiv in NDIVS, gpuT in types, device in devices
        r = _run_one(ndiv, gpuT, device)
        @printf("%6d %8d %6s %10.4f %10.4f %10.2e %10.2e %10.2e\n",
            ndiv, r.nt, string(gpuT), r.t_cpu, r.t_gpu, r.relM, r.res_cpu, r.res_gpu)
        push!(rows, (ndiv=ndiv, nt=r.nt, T=string(gpuT), device=string(device),
            t_cpu=r.t_cpu, t_gpu=r.t_gpu, relM=r.relM,
            res_cpu=r.res_cpu, res_gpu=r.res_gpu))
    end

    mkpath(dirname(OUTPATH))
    open(OUTPATH, "w") do io
        println(io, "ndiv\tnt\tT\tdevice\tt_cpu\tt_gpu\trelM\tres_cpu\tres_gpu")
        for r in rows
            @printf(io, "%d\t%d\t%s\t%s\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\n",
                r.ndiv, r.nt, r.T, r.device, r.t_cpu, r.t_gpu, r.relM,
                r.res_cpu, r.res_gpu)
        end
    end
    println("\nWrote ", OUTPATH)
    return nothing
end

main()
