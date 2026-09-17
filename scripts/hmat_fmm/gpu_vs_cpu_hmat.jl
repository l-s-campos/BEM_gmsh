# CPU vs GPU apply for H-matrix and NNCA H².
#
#   julia --project=. -t auto scripts/hmat_fmm/gpu_vs_cpu_hmat.jl
#
# ENV:
#   HMAT_NS=32,64,96
#   HMAT_NWARM=3
#   HMAT_NRUN=10
#   HMAT_RTOL=1e-6
#   HMAT_NMAX=32

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Statistics
using StaticArrays
using BEM.HMatrices

const NS = parse.(Int, split(get(ENV, "HMAT_NS", "32,64,96"), ','; keepempty=false))
const NWARM = parse(Int, get(ENV, "HMAT_NWARM", "3"))
const NRUN = parse(Int, get(ENV, "HMAT_NRUN", "10"))
const RTOL = parse(Float64, get(ENV, "HMAT_RTOL", "1e-6"))
const NMAX = parse(Int, get(ENV, "HMAT_NMAX", "32"))

const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _kernel(pts)
    return KernelMatrix(pts, pts) do x, y
        d = hypot(x[1] - y[1], x[2] - y[2])
        return d < 1e-14 ? 2.0 : log(d)
    end
end

function _bench(f; nwarm=NWARM, nrun=NRUN)
    for _ in 1:nwarm
        f()
    end
    ts = Vector{Float64}(undef, nrun)
    @inbounds for i in 1:nrun
        ts[i] = @elapsed f()
    end
    return minimum(ts), median(ts)
end

function _rel(a, b)
    nb = norm(b)
    return nb > 0 ? norm(a .- b) / nb : (norm(a) == 0 ? 0.0 : Inf)
end

function _print_gpu()
    info = gpu_float_support()
    println("GPU")
    println("  available   = ", info.available)
    println("  name        = ", info.name)
    println("  capability  = ", info.capability)
    println("  FP64/FP32   = ", info.fp64_fp32_ratio)
    if info.memory_bytes !== nothing
        @printf("  memory      = %.2f GiB\n", info.memory_bytes / 2^30)
    end
    println("  threads     = ", Threads.nthreads())
    println()
end

function _run_h(n1d)
    Random.seed!(1)
    pts = _pts(n1d)
    n = length(pts)
    K = _kernel(pts)
    x = randn(n)
    tree = ClusterTree(pts, hmatrix_splitter(; nmax=NMAX); cube=true)
    t_asm = @elapsed H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=RTOL),
        threads=true)
    ycpu = H * x
    t_cpu_min, t_cpu_med = _bench(() -> mul!(similar(ycpu), H, x))
    t_up64 = NaN
    t_g64_min = t_g64_med = NaN
    err64 = NaN
    t_up32 = NaN
    t_g32_min = t_g32_med = NaN
    err32 = NaN
    t_d64_min = t_d64_med = t_d32_min = t_d32_med = NaN
    err_d64 = err_d32 = NaN
    if HAVE_CUDA
        t_up64 = @elapsed H64 = gpu(H; device=:cuda, T=Float64)
        y64 = H64 * x
        err64 = _rel(y64, ycpu)
        t_g64_min, t_g64_med = _bench(() -> mul!(similar(y64), H64, x))
        xd64 = CUDA.CuArray(x)
        yd64 = similar(xd64)
        mul!(yd64, H64, xd64)
        err_d64 = _rel(Vector(yd64), ycpu)
        t_d64_min, t_d64_med = _bench(() -> mul!(yd64, H64, xd64))
        t_up32 = @elapsed H32 = gpu(H; device=:cuda, T=Float32)
        x32 = Float32.(x)
        y32 = Float64.(H32 * x32)
        err32 = _rel(y32, ycpu)
        t_g32_min, t_g32_med = _bench(() -> mul!(similar(x32), H32, x32))
        xd32 = CUDA.CuArray(x32)
        yd32 = similar(xd32)
        mul!(yd32, H32, xd32)
        err_d32 = _rel(Float64.(Vector(yd32)), ycpu)
        t_d32_min, t_d32_med = _bench(() -> mul!(yd32, H32, xd32))
    end
    yref = n <= 2500 ? Matrix(K) * x : nothing
    err_cpu_dense = yref === nothing ? NaN : _rel(ycpu, yref)
    return (; kind=:H, n, t_asm, t_cpu_min, t_cpu_med, t_up64, t_g64_min, t_g64_med,
        err64, t_up32, t_g32_min, t_g32_med, err32, err_cpu_dense,
        t_d64_min, t_d64_med, err_d64, t_d32_min, t_d32_med, err_d32)
end

function _run_h2(n1d)
    Random.seed!(1)
    pts = _pts(n1d)
    n = length(pts)
    K = _kernel(pts)
    x = randn(n)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    t_asm = @elapsed H = assemble_h2(K, tree; rtol=RTOL, threads=true)
    ycpu = H * x
    t_cpu_min, t_cpu_med = _bench(() -> mul!(similar(ycpu), H, x))
    t_up64 = NaN
    t_g64_min = t_g64_med = NaN
    err64 = NaN
    t_up32 = NaN
    t_g32_min = t_g32_med = NaN
    err32 = NaN
    t_d64_min = t_d64_med = t_d32_min = t_d32_med = NaN
    err_d64 = err_d32 = NaN
    if HAVE_CUDA
        t_up64 = @elapsed H64 = gpu(H; device=:cuda, T=Float64)
        y64 = H64 * x
        err64 = _rel(y64, ycpu)
        t_g64_min, t_g64_med = _bench(() -> mul!(similar(y64), H64, x))
        xd64 = CUDA.CuArray(x)
        yd64 = similar(xd64)
        mul!(yd64, H64, xd64)
        err_d64 = _rel(Vector(yd64), ycpu)
        t_d64_min, t_d64_med = _bench(() -> mul!(yd64, H64, xd64))
        t_up32 = @elapsed H32 = gpu(H; device=:cuda, T=Float32)
        x32 = Float32.(x)
        y32 = Float64.(H32 * x32)
        err32 = _rel(y32, ycpu)
        t_g32_min, t_g32_med = _bench(() -> mul!(similar(x32), H32, x32))
        xd32 = CUDA.CuArray(x32)
        yd32 = similar(xd32)
        mul!(yd32, H32, xd32)
        err_d32 = _rel(Float64.(Vector(yd32)), ycpu)
        t_d32_min, t_d32_med = _bench(() -> mul!(yd32, H32, xd32))
    end
    yref = n <= 2500 ? Matrix(K) * x : nothing
    err_cpu_dense = yref === nothing ? NaN : _rel(ycpu, yref)
    return (; kind=:H2, n, t_asm, t_cpu_min, t_cpu_med, t_up64, t_g64_min, t_g64_med,
        err64, t_up32, t_g32_min, t_g32_med, err32, err_cpu_dense,
        t_d64_min, t_d64_med, err_d64, t_d32_min, t_d32_med, err_d32)
end

function _row(r)
    spd_c64 = r.t_g64_med > 0 ? r.t_cpu_med / r.t_g64_med : NaN
    spd_d64 = r.t_d64_med > 0 ? r.t_cpu_med / r.t_d64_med : NaN
    spd_d32 = r.t_d32_med > 0 ? r.t_cpu_med / r.t_d32_med : NaN
    @printf("%-4s %6d  %7.2f  %7.2f  %7.2f  %7.2f  %8.1e  %5.2f  %5.2f  %7.2f  %7.2f  %8.1e  %5.2f\n",
        r.kind, r.n, 1e3 * r.t_asm, 1e3 * r.t_cpu_med,
        1e3 * r.t_g64_med, 1e3 * r.t_d64_med, r.err_d64, spd_c64, spd_d64,
        1e3 * r.t_g32_med, 1e3 * r.t_d32_med, r.err_d32, spd_d32)
end

function main()
    println("CPU vs GPU H-matrix / NNCA apply  (log kernel on unit square)")
    println("rtol=", RTOL, "  nmax=", NMAX, "  nwarm=", NWARM, "  nrun=", NRUN)
    println("times are median of ", NRUN, " (ms). copy = host x/y; dev = CuArray x/y")
    println()
    _print_gpu()
    HAVE_CUDA || println("CUDA not functional — CPU only.\n")
    println("kind      N  assemble     CPU  copy64   dev64     εdev  spC  spD  copy32   dev32     εdev   spD")
    println("-"^112)
    for n1d in NS
        _row(_run_h(n1d))
        _row(_run_h2(n1d))
    end
    println()
    println("εdev is ||y_gpu − y_cpu|| / ||y_cpu|| on the device-resident path.")
    println("spC = t_cpu / t_copy;  spD = t_cpu / t_dev  (>1 means GPU faster).")
    return
end

main()
