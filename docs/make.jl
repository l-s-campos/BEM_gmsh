# Build HTML docs with Documenter `@docs` bound to the BEM module.
# Load BEM from the package env (avoids pinning Documenter's JSON against BEM).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BEM

Pkg.activate(@__DIR__)
Pkg.instantiate()
using Documenter

const REPO = "github.com/l-s-campos/BEM_gmsh.git"
const PAGES_URL = "https://l-s-campos.github.io/BEM_gmsh/"
const BUILD = joinpath(@__DIR__, "build")

# Documenter starts by `rm(build; recursive=true)`. On Windows/OneDrive that
# often fails with EACCES if a browser, editor preview, or sync client holds a
# handle. Try a best-effort unlock: rename old build out of the way first.
function _clear_build_dir!(dir::AbstractString=BUILD)
    parent = dirname(dir)
    bas = basename(dir)
    # Drop leftover renamed builds from previous runs (OneDrive locks).
    for name in readdir(parent)
        startswith(name, bas * "_old_") || continue
        try
            rm(joinpath(parent, name); force=true, recursive=true)
        catch e
            @warn "Could not remove stale docs build trash" path=name exception=e
        end
    end

    isdir(dir) || return
    trash = dir * "_old_" * string(time_ns())
    try
        mv(dir, trash; force=true)
    catch
        try
            rm(dir; force=true, recursive=true)
            return nothing
        catch e
            @warn "Could not remove docs/build (close browser/preview or pause OneDrive)" exception=e
            # last resort: empty contents so Documenter's rm is less likely to fail
            for (root, dirs, files) in walkdir(dir; topdown=false)
                for f in files
                    try
                        rm(joinpath(root, f); force=true)
                    catch e2
                        @debug "docs build file locked" file=f exception=e2
                    end
                end
                for d in dirs
                    try
                        rm(joinpath(root, d); force=true)
                    catch e2
                        @debug "docs build dir locked" dir=d exception=e2
                    end
                end
            end
            return nothing
        end
    end
    # Synchronous cleanup of the just-renamed tree (no @async / OneDrive race).
    if isdir(trash)
        try
            rm(trash; force=true, recursive=true)
        catch e
            @warn "Left docs build trash for next run" path=trash exception=e
        end
    end
    return nothing
end
_clear_build_dir!()

makedocs(;
    modules = [BEM, BEM.Crack, BEM.Contact, BEM.Plate, BEM.Topology,
               BEM.MultiRegion, BEM.HMatrices, BEM.FMM, BEM.Examples],
    authors = "BEM_gmsh contributors",
    sitename = "BEM_gmsh",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = PAGES_URL,
        repolink = "https://github.com/l-s-campos/BEM_gmsh",
        assets = String[],
        sidebar_sitename = true,
        footer = "BEM_gmsh · [GitHub](https://github.com/l-s-campos/BEM_gmsh)",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Recipes" => "recipes.md",
        "Theory notes" => "theory.md",
        "API" => [
            "Data structures" => "api/structures.md",
            "Fundamentals" => "api/fundamentals.md",
            "Mesh I/O" => "api/input.md",
            "Assembly" => "api/assembly.md",
            "Solvers" => "api/solvers.md",
            "DIBEM" => "api/dibem.md",
            "Local BEM" => "api/local_bem.md",
            "SBM / DRM" => "api/sbm.md",
            "Helmholtz" => "api/helmholtz.md",
            "Elasticity" => "api/elasticity.md",
            "Analytical" => "api/analytical.md",
            "Visualization" => "api/visualization.md",
            "H-matrices" => "api/hmatrices.md",
            "FMM" => "api/fmm.md",
            "Crack" => "api/crack.md",
            "Cohesive" => "api/cohesive.md",
            "Contact" => "api/contact.md",
            "Multi-region" => "api/multiregion.md",
            "Plates" => "api/plates.md",
            "Topology" => "api/topology.md",
            "Examples module" => "api/examples.md",
        ],
        "Examples" => "examples.md",
        "Performance" => "performance.md",
        "Architecture" => "architecture.md",
        "Português (BR)" => [
            "Início" => "pt-br/index.md",
            "Começando" => "pt-br/getting_started.md",
            "Receitas" => "pt-br/recipes.md",
            "Teoria" => "pt-br/theory.md",
            "API" => "pt-br/api.md",
            "Exemplos" => "pt-br/examples.md",
            "Desempenho" => "pt-br/performance.md",
            "Arquitetura" => "pt-br/architecture.md",
        ],
    ],
    checkdocs = :none,
    warnonly = true,
)

@info "Docs built → docs/build (open index.html)"
