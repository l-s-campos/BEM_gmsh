# =============================================================================
# Linear solves via LinearSolve.jl
# Prefer this over bare `A \ b` for dense/sparse BEM systems.
# Caching interface follows:
#   https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface/
# =============================================================================

export bem_linsolve, bem_linfactor, bem_linsolve!

"""
Default algorithm: `nothing` → LinearSolve picks (LU for square dense, QR for
least-squares, etc.). Pass e.g. `alg=LUFactorization()` to force.
"""
const BEM_DEFAULT_LS_ALG = nothing

function _ls_solve(prob; alg=nothing, kwargs...)
    if alg === nothing
        return LinearSolve.solve(prob; kwargs...)
    else
        return LinearSolve.solve(prob, alg; kwargs...)
    end
end

function _ls_init(prob; alg=nothing, kwargs...)
    if alg === nothing
        return init(prob; kwargs...)
    else
        return init(prob, alg; kwargs...)
    end
end

"""
    bem_linsolve(A, b; alg=nothing, kwargs...) -> x

Solve ``A x = b`` with LinearSolve.jl.

- `b` may be an `AbstractVector` or `AbstractMatrix` (multiple RHS; one factorization)
- Precomputed `Factorization` objects use `ldiv!` (no re-factor)
- Hierarchical H-matrix factors should keep using `ldiv!` on the H-factor type
- Square dense → LU by default; tall/skinny → QR / least-squares

Calls `LinearSolve.solve` explicitly so it does not clash with BEM's [`solve`](@ref).
"""
function bem_linsolve(A, b::AbstractVector; alg=BEM_DEFAULT_LS_ALG, kwargs...)
    if A isa Factorization
        return ldiv!(A, collect(b))
    end
    return _ls_solve(LinearProblem(A, b); alg=alg, kwargs...).u
end

function bem_linsolve(A, B::AbstractMatrix; alg=BEM_DEFAULT_LS_ALG, kwargs...)
    size(A, 1) == size(B, 1) || throw(DimensionMismatch("A and B row mismatch"))
    nrhs = size(B, 2)
    nrhs == 0 && return similar(B, size(A, 2), 0)
    if A isa Factorization
        return ldiv!(A, collect(B))
    end
    # Official caching interface: init once, swap b, solve!
    # https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface/
    cache = bem_linfactor(A, @view(B[:, 1]); alg=alg, kwargs...)
    X = similar(B, size(A, 2), nrhs)
    @inbounds for j in 1:nrhs
        X[:, j] = bem_linsolve!(cache, @view(B[:, j]))
    end
    return X
end

"""
    bem_linfactor(A, b_proto=zeros(size(A,1)); alg=nothing, kwargs...) -> cache

Build a LinearSolve cache for repeated solves with fixed `A` (and later new `b`).

```julia
cache = bem_linfactor(A, b1)
x1 = bem_linsolve!(cache)          # uses b1
x2 = bem_linsolve!(cache, b2)      # reuse factorization
cache.A = A2                       # triggers refactor on next solve
x3 = bem_linsolve!(cache, b3)
```

See https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface/
"""
function bem_linfactor(A, b_proto::AbstractVector=zeros(eltype(A), size(A, 1));
        alg=BEM_DEFAULT_LS_ALG, kwargs...)
    return _ls_init(LinearProblem(A, collect(b_proto)); alg=alg, kwargs...)
end

"""
    bem_linsolve!(cache) -> x
    bem_linsolve!(cache, b) -> x

Reuse a cache from [`bem_linfactor`](@ref).

Matches the LinearSolve pattern:

```julia
cache.b = b2
sol = solve!(cache)
```
"""
function bem_linsolve!(cache)
    return solve!(cache).u
end

function bem_linsolve!(cache, b::AbstractVector)
    # Official API: assign new RHS, then solve! (factorization reused)
    if length(cache.b) == length(b)
        if cache.b === b
            # already set
        elseif axes(cache.b) == axes(b)
            copyto!(cache.b, b)
        else
            cache.b = collect(b)
        end
    else
        cache.b = collect(b)
    end
    return solve!(cache).u
end
