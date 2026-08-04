# Build HTML docs (markdown sources; no full BEM precompile required).
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter

const REPO = "github.com/l-s-campos/BEM_gmsh.git"
const PAGES_URL = "https://l-s-campos.github.io/BEM_gmsh/"

makedocs(;
    modules = Module[],
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
        "API guide" => [
            "Data structures" => "api/structures.md",
            "Fundamental solutions" => "api/fundamentals.md",
            "Mesh I/O" => "api/input.md",
            "Assembly" => "api/assembly.md",
            "Solvers" => "api/solvers.md",
            "DIBEM & diffuse–advective" => "api/dibem.md",
            "Analytical solutions" => "api/analytical.md",
            "Visualization" => "api/visualization.md",
            "H-matrices" => "api/hmatrices.md",
            "Half-space contact" => "api/contact.md",
            "Crack propagation" => "api/crack.md",
            "Cohesive-contact DBEM" => "api/cohesive.md",
        ],
        "Examples" => "examples.md",
        "Performance" => "performance.md",
        "Architecture" => "architecture.md",
        "Português (BR)" => [
            "Início" => "pt-br/index.md",
            "Começando" => "pt-br/getting_started.md",
            "Notas de teoria" => "pt-br/theory.md",
            "Exemplos" => "pt-br/examples.md",
            "Desempenho" => "pt-br/performance.md",
            "API (links EN)" => "pt-br/api.md",
        ],
    ],
    checkdocs = :none,
    warnonly = true,
)

@info "Docs built → docs/build (open index.html)"
