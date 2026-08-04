using Documenter
using DrWatson
@quickactivate :BEM

# Multilingual docs with Documenter.jl
# ------------------------------------
# Documenter has **no** built-in i18n (unlike Jekyll + jekyll-polyglot).
# Practical approach used here:
#   • English sources:  docs/src/*.md          (default sidebar)
#   • Portuguese BR:    docs/src/pt-br/*.md    (section "Português (BR)")
#   • One `makedocs` build serves both; pages cross-link via 🌐 banners.
# API `@docs` pages stay in English (Julia symbol names).

makedocs(;
    modules=[BEM],
    authors="BEM.jl contributors",
    sitename="BEM.jl",
    remotes=nothing,  # local / no git origin required
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://example.com/BEM.jl",
        assets=String[],
        lang="en",
        footer="BEM.jl · English + Português (BR)",
    ),
    pages=[
        # --- English (default) ---
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Theory notes" => "theory.md",
        "API" => [
            "Data structures" => "api/structures.md",
            "Fundamental solutions" => "api/fundamentals.md",
            "Mesh I/O" => "api/input.md",
            "Assembly" => "api/assembly.md",
            "Solvers" => "api/solvers.md",
            "Analytical solutions" => "api/analytical.md",
            "Visualization" => "api/visualization.md",
            "H-matrices" => "api/hmatrices.md",
            "Half-space contact" => "api/contact.md",
            "Crack propagation" => "api/crack.md",
            "Cohesive-contact DBEM" => "api/cohesive.md",
        ],
        "Examples" => "examples.md",
        "Performance" => "performance.md",
        # --- Português (BR) ---
        "Português (BR)" => [
            "Início" => "pt-br/index.md",
            "Começando" => "pt-br/getting_started.md",
            "Notas de teoria" => "pt-br/theory.md",
            "Exemplos" => "pt-br/examples.md",
            "Desempenho" => "pt-br/performance.md",
            "API (links EN)" => "pt-br/api.md",
        ],
    ],
    checkdocs=:none,
    warnonly=true,
)

# Optional:
# deploydocs(repo = "github.com/USER/BEM_gmsh.git")
