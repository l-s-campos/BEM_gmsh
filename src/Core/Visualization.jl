export plot_geo, plot_nodes, plot_hmatrix, export_results_to_gmsh

"""
    plot_geo(dad::BEMdata; kwargs...) -> Figure

Visualize the BEM geometry, collocation nodes, and boundary conditions.

# Boundary-condition glyphs
| Type | Scalar | Vectorial |
|------|--------|-----------|
| **Dirichlet** (`BC == 0`) | filled triangle on the node, colored by prescribed value | filled triangle; color by ``\\|u\\|`` |
| **Neumann** (`BC == 1`) | arrow along the outward normal, length ∝ ``\\|q\\|`` | arrow of the traction vector ``t`` |

Arrow lengths are scaled so the largest ``|q|`` (or ``\\|t\\|``) maps to
`arrow_scale` (default ``0.15 ×`` the geometry characteristic length).
Zero Neumann values are shown as open circles.

# Keyword arguments
- `figsize=(850, 700)`: figure size in pixels
- `show_nodes=true`: scatter all boundary collocation nodes
- `show_internal=true`: scatter internal points
- `show_elements=true`: draw element edges
- `show_normals=false`: draw unit normals (debug)
- `show_bc=true`: draw Dirichlet / Neumann glyphs
- `arrow_scale=nothing`: max arrow length in domain units
- `markersize=8`: base marker size for nodes
- `tri_scale=nothing`: Dirichlet triangle size (domain units)
- `colormap=:viridis`: colormap for BC values
- `title=nothing`: axis title (auto if `nothing`)
- `legend=true`: show legend
"""
function plot_geo(
    dad::BEMdata;
    figsize=(850, 700),
    show_nodes=true,
    show_internal=true,
    show_elements=true,
    show_normals=false,
    show_bc=true,
    arrow_scale=nothing,
    markersize=8,
    tri_scale=nothing,
    colormap=:viridis,
    title=nothing,
    legend=true,
    node_color=:gray55,
    internal_node_color=:indianred,
)
    fig = Figure(size=figsize)
    dim = dad.dimension
    L = _char_length(dad)
    tri_s = something(tri_scale, 0.045 * L)
    arr_s = something(arrow_scale, 0.15 * L)

    # layout: main axis + optional colorbar column
    ax = if dim == 2
        Axis(
            fig[1, 1];
            xlabel="x",
            ylabel="y",
            title=something(title, _default_title(dad)),
            aspect=DataAspect(),
        )
    else
        Axis3(
            fig[1, 1];
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=something(title, _default_title(dad)),
            aspect=:data,
        )
    end

    # ---- elements ---------------------------------------------------------
    if show_elements
        _plot_elements!(ax, dad)
    end

    # ---- nodes ------------------------------------------------------------
    if show_nodes && !isempty(dad.Nodes)
        _scatter_points!(
            ax,
            dad.Nodes;
            color=node_color,
            markersize=markersize,
            label="collocation",
        )
    end
    if show_internal && !isempty(dad.internalNodes)
        _scatter_points!(
            ax,
            dad.internalNodes;
            color=internal_node_color,
            markersize=markersize,
            marker=:xcross,
            label="internal",
        )
    end

    # ---- unit normals (optional) ------------------------------------------
    if show_normals
        nlen = 0.05 * L
        _arrows_from_points!(
            ax,
            dad.Nodes,
            [nlen * n for n in dad.Normal];
            color=:gray65,
            linewidth=0.8,
            arrowsize=8,
            label="normal",
        )
    end

    # ---- boundary conditions ----------------------------------------------
    cb = nothing
    if show_bc
        if dad.properties isa Scalar
            cb = _plot_bc_scalar!(fig, ax, dad, tri_s, arr_s, colormap)
        elseif dad.properties isa Vectorial
            cb = _plot_bc_vectorial!(fig, ax, dad, tri_s, arr_s, colormap)
        end
    end

    if legend && dim == 2
        axislegend(ax; position=:rt, framevisible=true, labelsize=11)
    end

    return fig
end

"""Deprecated alias for [`plot_geo`](@ref)."""
plot_nodes(dad; kwargs...) = plot_geo(dad; kwargs...)

# =============================================================================
# helpers
# =============================================================================

function _default_title(dad::BEMdata)
    p = dad.properties
    if p isa Laplace
        return "Geometry & BCs — Laplace (k=$(p.k))"
    elseif p isa Elasticity
        return "Geometry & BCs — Elasticity (E=$(p.E), ν=$(p.nu))"
    end
    return "Geometry & BCs — $(dad.name)"
end

function _char_length(dad::BEMdata)
    isempty(dad.Nodes) && return 1.0
    xs = (p[1] for p in dad.Nodes)
    ys = (p[2] for p in dad.Nodes)
    Lx = maximum(xs) - minimum(xs)
    Ly = maximum(ys) - minimum(ys)
    if dad.dimension == 3
        zs = (p[3] for p in dad.Nodes)
        Lz = maximum(zs) - minimum(zs)
        return max(Lx, Ly, Lz, 1e-12)
    end
    return max(Lx, Ly, 1e-12)
end

function _plot_elements!(ax, dad::BEMdata)
    if dad.dimension == 2
        ξ = collect(range(-1.0, 1.0; length=9))
        for elem in dad.elements
            pts = dad.Nodes[elem.index]
            P = reduce(hcat, pts)'          # (nnodes × 2)
            N, _ = shapefun(dad.element_type, ξ)  # (nξ × nnodes)
            C = N * P                         # (nξ × 2)
            lines!(ax, C[:, 1], C[:, 2]; color=:black, linewidth=1.5)
        end
    else
        for elem in dad.elements
            pts = dad.Nodes[elem.index]
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            zs = [p[3] for p in pts]
            # close the loop loosely
            push!(xs, xs[1]); push!(ys, ys[1]); push!(zs, zs[1])
            lines!(ax, xs, ys, zs; color=:black, linewidth=1.0)
        end
    end
    return nothing
end

function _scatter_points!(ax, pts::Vector{<:Point2D}; kwargs...)
    scatter!(ax, [p[1] for p in pts], [p[2] for p in pts]; kwargs...)
end

function _scatter_points!(ax, pts::Vector{<:Point3D}; kwargs...)
    scatter!(ax, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]; kwargs...)
end

function _arrows_from_points!(ax, pts::Vector{<:Point2D}, dirs; kwargs...)
    isempty(pts) && return nothing
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    u = [d[1] for d in dirs]
    v = [d[2] for d in dirs]
    arrows!(ax, x, y, u, v; kwargs...)
    return nothing
end

function _arrows_from_points!(ax, pts::Vector{<:Point3D}, dirs; kwargs...)
    isempty(pts) && return nothing
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    z = [p[3] for p in pts]
    u = [d[1] for d in dirs]
    v = [d[2] for d in dirs]
    w = [d[3] for d in dirs]
    arrows!(ax, x, y, z, u, v, w; kwargs...)
    return nothing
end

# ---------- scalar BC -------------------------------------------------------

function _plot_bc_scalar!(fig, ax, dad::BEMdata{<:Scalar}, tri_s, arr_s, colormap)
    dim = dad.dimension
    n = min(dad.n, length(dad.BC))
    BC = dad.BC
    BV = dad.BV
    cb = nothing

    idir = findall(i -> BC[i] == 0, 1:n)
    ineu = findall(i -> BC[i] == 1, 1:n)

    # ---- Dirichlet: triangles ---------------------------------------------
    if !isempty(idir)
        pts = dad.Nodes[idir]
        vals = Float64.(BV[idir])
        vrange = _sym_range(vals)

        if dim == 2
            for (p, nrm, v) in zip(pts, dad.Normal[idir], vals)
                _draw_triangle_2d!(ax, p, nrm, tri_s, v, vrange, colormap)
            end
            # proxy scatter for colorbar + legend
            sc = scatter!(
                ax,
                [p[1] for p in pts],
                [p[2] for p in pts];
                marker=:utriangle,
                markersize=0.1,
                color=vals,
                colormap=colormap,
                colorrange=vrange,
                label="Dirichlet",
            )
            cb = Colorbar(fig[1, 2], sc; label="Dirichlet value", width=14)
        else
            sc = scatter!(
                ax,
                [p[1] for p in pts],
                [p[2] for p in pts],
                [p[3] for p in pts];
                marker=:utriangle,
                markersize=16,
                color=vals,
                colormap=colormap,
                colorrange=vrange,
                label="Dirichlet",
            )
            cb = Colorbar(fig[1, 2], sc; label="Dirichlet value", width=14)
        end
    end

    # ---- Neumann: arrows along normal -------------------------------------
    if !isempty(ineu)
        pts = dad.Nodes[ineu]
        vals = Float64.(BV[ineu])
        nrms = dad.Normal[ineu]
        maxabs = maximum(abs, vals; init=0.0)
        scale = maxabs > 0 ? arr_s / maxabs : 0.0
        dirs = [scale * v * nrm for (v, nrm) in zip(vals, nrms)]

        nz = findall(d -> norm(d) > 1e-14 * max(arr_s, 1.0), dirs)
        zidx = findall(i -> abs(vals[i]) <= 1e-14, eachindex(vals))

        if !isempty(nz)
            _arrows_from_points!(
                ax,
                pts[nz],
                dirs[nz];
                color=:dodgerblue,
                linewidth=1.8,
                arrowsize=dim == 2 ? 14 : 10,
                label="Neumann q",
            )
        end
        if !isempty(zidx) && dim == 2
            pz = pts[zidx]
            scatter!(
                ax,
                [p[1] for p in pz],
                [p[2] for p in pz];
                marker=:circle,
                markersize=9,
                color=:white,
                strokecolor=:dodgerblue,
                strokewidth=1.8,
                label="Neumann q = 0",
            )
        end
    end
    return cb
end

# ---------- vectorial BC ----------------------------------------------------

function _plot_bc_vectorial!(fig, ax, dad::BEMdata{<:Vectorial}, tri_s, arr_s, colormap)
    dim = dad.dimension
    n = dad.n
    BC = dad.BC
    BV = dad.BV
    ndof = dim * n
    length(BC) >= ndof || return nothing
    cb = nothing

    dir_nodes = Int[]
    dir_vals = Float64[]
    neu_nodes = Int[]
    neu_vecs = Vector{SVector{dim,Float64}}()

    for i in 1:n
        dofs = (dim * (i - 1) + 1):(dim * i)
        bc_i = BC[dofs]
        bv_i = BV[dofs]

        if any(==(0), bc_i)
            push!(dir_nodes, i)
            ucomp = [bc_i[k] == 0 ? bv_i[k] : 0.0 for k in 1:dim]
            push!(dir_vals, norm(ucomp))
        end
        if any(==(1), bc_i)
            t = SVector{dim,Float64}(ntuple(k -> (bc_i[k] == 1 ? bv_i[k] : 0.0), dim))
            push!(neu_nodes, i)
            push!(neu_vecs, t)
        end
    end

    # ---- Dirichlet triangles ----------------------------------------------
    if !isempty(dir_nodes)
        pts = dad.Nodes[dir_nodes]
        vrange = _sym_range(dir_vals)
        if dim == 2
            for (p, nrm, v) in zip(pts, dad.Normal[dir_nodes], dir_vals)
                _draw_triangle_2d!(ax, p, nrm, tri_s, v, vrange, colormap)
            end
            sc = scatter!(
                ax,
                [p[1] for p in pts],
                [p[2] for p in pts];
                marker=:utriangle,
                markersize=0.1,
                color=dir_vals,
                colormap=colormap,
                colorrange=vrange,
                label="Dirichlet u",
            )
            cb = Colorbar(fig[1, 2], sc; label="|u| prescribed", width=14)
        else
            sc = scatter!(
                ax,
                [p[1] for p in pts],
                [p[2] for p in pts],
                [p[3] for p in pts];
                marker=:utriangle,
                markersize=16,
                color=dir_vals,
                colormap=colormap,
                colorrange=vrange,
                label="Dirichlet u",
            )
            cb = Colorbar(fig[1, 2], sc; label="|u| prescribed", width=14)
        end
    end

    # ---- Neumann traction arrows ------------------------------------------
    if !isempty(neu_nodes)
        pts = dad.Nodes[neu_nodes]
        maxabs = maximum(norm, neu_vecs; init=0.0)
        scale = maxabs > 0 ? arr_s / maxabs : 0.0
        dirs = [scale * t for t in neu_vecs]

        nz = findall(d -> norm(d) > 1e-14 * max(arr_s, 1.0), dirs)
        zidx = findall(i -> norm(neu_vecs[i]) <= 1e-14, eachindex(neu_vecs))

        if !isempty(nz)
            _arrows_from_points!(
                ax,
                pts[nz],
                dirs[nz];
                color=:dodgerblue,
                linewidth=1.8,
                arrowsize=dim == 2 ? 14 : 10,
                label="Neumann t",
            )
        end
        if !isempty(zidx) && dim == 2
            pz = pts[zidx]
            scatter!(
                ax,
                [p[1] for p in pz],
                [p[2] for p in pz];
                marker=:circle,
                markersize=9,
                color=:white,
                strokecolor=:dodgerblue,
                strokewidth=1.8,
                label="Neumann t = 0",
            )
        end
    end
    return cb
end

# ---------- geometry primitives ---------------------------------------------

"""
Draw a filled triangle at `p`. Base is along the tangent; tip points
**inward** (opposite the outward normal `nrm`).
"""
function _draw_triangle_2d!(ax, p::Point2D, nrm::Point2D, size, val, vrange, colormap)
    nn = norm(nrm)
    n̂ = nn > 0 ? nrm / nn : Point2D(0.0, 1.0)
    t̂ = Point2D(-n̂[2], n̂[1])
    tip = p - 0.90 * size * n̂
    b1 = p + 0.40 * size * n̂ + 0.55 * size * t̂
    b2 = p + 0.40 * size * n̂ - 0.55 * size * t̂
    col = _color_from_value(val, vrange, colormap)
    poly!(
        ax,
        [Point2f(tip...), Point2f(b1...), Point2f(b2...)];
        color=col,
        strokecolor=:black,
        strokewidth=0.7,
    )
    return nothing
end

function _sym_range(vals)
    isempty(vals) && return (0.0, 1.0)
    lo, hi = extrema(vals)
    if abs(hi - lo) < 1e-14
        δ = max(abs(lo) * 0.05, 1e-6)
        return (lo - δ, hi + δ)
    end
    return (float(lo), float(hi))
end

function _color_from_value(v, vrange, colormap)
    lo, hi = vrange
    t = clamp((v - lo) / (hi - lo + eps()), 0.0, 1.0)
    cm = Makie.to_colormap(colormap)
    idx = round(Int, 1 + t * (length(cm) - 1))
    return cm[clamp(idx, 1, length(cm))]
end

# =============================================================================
# Gmsh export
# =============================================================================

"""
    export_results_to_gmsh(dad, filename, result_name::Symbol; viewer=true)

Export a nodal field stored in `dad.cache` (e.g. `:T`) to a Gmsh `.msh` view,
interpolating from BEM nodes onto the mesh with an RBF.
"""
function export_results_to_gmsh(dad::BEMdata, filename::String, result_name::Symbol, viewer=true)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.clear()
    gmsh.open(filename)

    tag = gmsh.view.add(string(result_name), 0)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()

    if dad.dimension == 2
        gmsh_points = [Point2D(coords[3i-2], coords[3i-1]) for i in 1:length(node_tags)]
    else
        gmsh_points = [Point3D(coords[3i-2], coords[3i-1], coords[3i]) for i in 1:length(node_tags)]
    end

    data = getproperty(dad, result_name)
    rbf = RBF([dad.Nodes; dad.internalNodes], PHS(3, poly_deg=2))
    gmshdata = rbf(gmsh_points, data)
    gmshdata = [[gmshdata[i]] for i in eachindex(gmsh_points)]
    gmsh.view.addModelData(tag, 0, dad.name, "NodeData", node_tags, gmshdata)
    gmsh.write(filename)
    println("Results exported to $filename using Gmsh API")

    if viewer
        gmsh.fltk.run()
    end
    gmsh.finalize()
    return nothing
end
