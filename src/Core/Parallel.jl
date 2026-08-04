# Parallelism helpers and optional GPU far-field evaluation
export nthreads_bem, gpu_available, farfield_gpu!

"""Number of Julia threads available to BEM assembly."""
nthreads_bem() = Threads.nthreads()

"""True if CUDA.jl is loaded and a device is present."""
function gpu_available()
    try
        @eval Main begin
            using CUDA
            return CUDA.functional()
        end
    catch
        return false
    end
end

"""
    farfield_gpu!(H, G, dad, collocation_idx, elem_data)

Optional GPU kernel for the **far-field** collocation update
``H_{i j} = Q(r_{ij}, n_j) w_j``, ``G_{i j} = U(r_{ij}, n_j) w_j``.

Requires CUDA.jl. Near-field (Newton + sinh-quad) stays on CPU.
This is a scaffold — enable by setting `BEM_USE_GPU=1` and calling from
assembly when `gpu_available()`.
"""
function farfield_gpu!(H, G, dad, pairs)
    error("farfield_gpu!: compile with CUDA and implement kernel for your platform")
end

function _maybe_gpu_farfield()
    return get(ENV, "BEM_USE_GPU", "0") == "1" && gpu_available()
end
