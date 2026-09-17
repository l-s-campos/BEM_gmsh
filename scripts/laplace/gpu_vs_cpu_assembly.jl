# Compare 2-D Laplace dense assembly: CPU (`H_G_full_direct`) vs GPU
# (KernelAbstractions kernel in `src/Laplace/Assembly_GPU.jl`).
#
#   julia --project=. scripts/laplace/gpu_vs_cpu_assembly.jl
#
# ENV:
#   GPU_NDIVS=20,40,80
#   GPU_NPG=12
#   GPU_OUT=plots/gpu_vs_cpu_assembly.tsv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

const NPG = parse(Int, get(ENV, "GPU_NPG", "12"))
const OUTPATH = get(ENV, "GPU_OUT", projectdir("plots", "gpu_vs_cpu_assembly.tsv"))
_parse_ints(s) = parse.(Int, split(s, ','; keepempty=false))
const NDIVS = _parse_ints(get(ENV, "GPU_NDIVS", "20,40,80"))

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

function _print_floats(info)
    println("GPU floating-point support")
    println("  available      = ", info.available)
    println("  name           = ", info.name)
    println("  capability     = ", info.capability)
    println("  Float16        = ", info.Float16, "  (native CUDA cores; not used for BEM kernels)")
    println("  Float32        = ", info.Float32, "  (default GPU assembly)")
    println("  Float64        = ", info.Float64, "  (IEEE; GeForce rate ", info.fp64_fp32_ratio, " of FP32)")
    println("  BFloat16       = ", info.BFloat16, "  (Ampere+)")
    println("  TensorFloat32  = ", info.TensorFloat32, "  (Ampere+)")
    println("  Float8         = ", info.Float8, "  (Ada/Hopper+)")
    println("  recommended    = ", info.recommended)
    if info.memory_bytes !== nothing
        @printf("  memory         = %.2f GiB\n", info.memory_bytes / 2^30)
    end
    info.message === nothing || println("  message        = ", info.message)
    println()
end

function _make_dad(ndiv, nome)
    dad = format2d(quadrado(ndiv=ndiv, show=false, nome=nome), Laplace(1.0);
        pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    return dad
end

function _run_one(ndiv, gpuT, device)
    dad_cpu = _make_dad(ndiv, "gpu_cmp_cpu_$(ndiv)_$(gpuT)")
    t_asm_cpu = @elapsed assemble!(dad_cpu, NPG; threaded=true)
    t_sol_cpu = @elapsed solve(dad_cpu)
    err_cpu = rel_error(dad_cpu)

    dad_gpu = _make_dad(ndiv, "gpu_cmp_gpu_$(ndiv)_$(gpuT)_$(device)")
    t_asm_gpu = @elapsed H_G_gpu(dad_gpu; T=gpuT, npg=NPG, near=:cpu,
        device=device, threaded=true)
    relH = _rel(Float64.(dad_gpu.H), dad_cpu.H)
    relG = _rel(Float64.(dad_gpu.G), dad_cpu.G)
    t_sol_gpu = @elapsed solve(dad_gpu)
    err_gpu = rel_error(dad_gpu)
    return (
        n = dad_cpu.n,
        t_asm_cpu = t_asm_cpu,
        t_asm_gpu = t_asm_gpu,
        relH = relH,
        relG = relG,
        t_sol_cpu = t_sol_cpu,
        t_sol_gpu = t_sol_gpu,
        err_cpu = err_cpu,
        err_gpu = err_gpu,
    )
end

function main()
    println("="^72)
    println(" Laplace 2-D assembly: CPU vs KernelAbstractions GPU")
    println(" ", Dates.now())
    println(" threads = ", Threads.nthreads(), "   npg = ", NPG)
    println("="^72)
    info = gpu_float_support()
    _print_floats(info)
    HAVE_CUDA || println("CUDA not functional — GPU rows use device=:cpu (same kernel).\n")

    # Warmup (compile KA + CUDA + LU)
    wdev = HAVE_CUDA ? :cuda : :cpu
    wdad = _make_dad(6, "gpu_cmp_warmup")
    assemble!(wdad, 8; threaded=false)
    solve(wdad)
    H_G_gpu(_make_dad(6, "gpu_cmp_warmup_k"), T=Float32, npg=8, near=:cpu,
        device=wdev, threaded=false)

    rows = Any[]
    devices = HAVE_CUDA ? (:cuda,) : (:cpu,)
    types = HAVE_CUDA ? (Float32, Float64) : (Float64,)
    @printf("%6s %8s %6s %10s %10s %9s %9s %10s %10s %10s %10s\n",
        "ndiv", "n", "T", "t_asm_cpu", "t_asm_gpu", "relH", "relG",
        "t_sol_cpu", "t_sol_gpu", "err_cpu", "err_gpu")
    for ndiv in NDIVS, gpuT in types, device in devices
        r = _run_one(ndiv, gpuT, device)
        @printf("%6d %8d %6s %10.4f %10.4f %9.2e %9.2e %10.4f %10.4f %10.3e %10.3e\n",
            ndiv, r.n, string(gpuT), r.t_asm_cpu, r.t_asm_gpu, r.relH, r.relG,
            r.t_sol_cpu, r.t_sol_gpu, r.err_cpu, r.err_gpu)
        push!(rows, (ndiv=ndiv, n=r.n, T=string(gpuT), device=string(device),
            t_asm_cpu=r.t_asm_cpu, t_asm_gpu=r.t_asm_gpu, relH=r.relH, relG=r.relG,
            t_sol_cpu=r.t_sol_cpu, t_sol_gpu=r.t_sol_gpu,
            err_cpu=r.err_cpu, err_gpu=r.err_gpu))
    end

    mkpath(dirname(OUTPATH))
    open(OUTPATH, "w") do io
        println(io, "ndiv\tn\tT\tdevice\tt_asm_cpu\tt_asm_gpu\trelH\trelG\t",
            "t_sol_cpu\tt_sol_gpu\terr_cpu\terr_gpu")
        for r in rows
            @printf(io, "%d\t%d\t%s\t%s\t%.6f\t%.6f\t%.6e\t%.6e\t%.6f\t%.6f\t%.6e\t%.6e\n",
                r.ndiv, r.n, r.T, r.device, r.t_asm_cpu, r.t_asm_gpu, r.relH, r.relG,
                r.t_sol_cpu, r.t_sol_gpu, r.err_cpu, r.err_gpu)
        end
    end
    println("\nWrote ", OUTPATH)
    return nothing
end

main()
