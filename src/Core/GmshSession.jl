# =============================================================================
# Gmsh session helpers — shared initialize / finalize refcounting
# =============================================================================
# Multi-step pipelines need Gmsh to stay alive across mesh load → solve →
# view export. Plain `gmsh.initialize/finalize` pairs fight that. These helpers
# keep a process-wide refcount so nested calls are safe.

export with_gmsh, gmsh_is_alive, gmsh_ensure!, gmsh_release!

const _GMSH_REFS = Ref(0)
const _GMSH_OWNED = Ref(false)

"""True if the Gmsh API is currently initialized."""
gmsh_is_alive() = _GMSH_REFS[] > 0

"""
    gmsh_ensure!(; verbosity=1, terminal=nothing)

Initialize Gmsh if needed and bump the refcount. Pair with [`gmsh_release!`](@ref).
Adopts an already-initialized session without calling `gmsh.initialize` again.
"""
function gmsh_ensure!(; verbosity::Integer = 1, terminal = nothing)
    if _GMSH_REFS[] == 0
        already = try
            Bool(gmsh.isInitialized())
        catch
            false
        end
        if already
            _GMSH_OWNED[] = true
        else
            gmsh.initialize()
            _GMSH_OWNED[] = true
        end
        try
            gmsh.option.setNumber("General.Verbosity", Float64(verbosity))
        catch
        end
        if terminal !== nothing
            try
                gmsh.option.setNumber("General.Terminal", Float64(terminal))
            catch
            end
        end
    end
    _GMSH_REFS[] += 1
    return nothing
end

"""
    gmsh_release!(; force=false)

Drop one refcount from [`gmsh_ensure!`](@ref). Finalizes Gmsh when the count
hits zero (unless the session was not owned by us).
"""
function gmsh_release!(; force::Bool = false)
    if force
        if _GMSH_OWNED[] && _GMSH_REFS[] > 0
            try
                gmsh.finalize()
            catch
            end
        end
        _GMSH_REFS[] = 0
        _GMSH_OWNED[] = false
        return nothing
    end
    _GMSH_REFS[] <= 0 && return nothing
    _GMSH_REFS[] -= 1
    if _GMSH_REFS[] == 0 && _GMSH_OWNED[]
        try
            gmsh.finalize()
        catch
        end
        _GMSH_OWNED[] = false
    end
    return nothing
end

"""
    with_gmsh(f; verbosity=1, terminal=nothing)

Run `f()` inside a refcounted Gmsh session. Nested `with_gmsh` calls share one
API instance; finalize runs only when the outermost block exits.
"""
function with_gmsh(f; kwargs...)
    gmsh_ensure!(; kwargs...)
    try
        return f()
    finally
        gmsh_release!()
    end
end
