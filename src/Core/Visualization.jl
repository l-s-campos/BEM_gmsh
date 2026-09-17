export plot_geo, export_results_to_gmsh, export_vtk

"""
    plot_geo(dad::BEMdata; kwargs...) -> Plots.Plot

Visualize the BEM geometry, collocation nodes, and boundary conditions
(Plots.jl + GR backend).

# Boundary-condition glyphs
| Type | Scalar | Vectorial |
|------|--------|-----------|
| **Dirichlet** (`BC == 0`) | triangle markers colored by value | triangles; color by ``\\|u\\|`` |
| **Neumann** (`BC == 1`) | arrow along outward normal | traction vector arrow |

# Keyword arguments
- `figsize=(850, 700)`: figure size in pixels
- `show_nodes=true`: scatter boundary collocation nodes
- `show_internal=true`: scatter internal points
- `show_elements=true`: draw element edges
- `show_normals=false`: draw unit normals
- `show_bc=true`: Dirichlet / Neumann glyphs
- `arrow_scale=nothing`: max arrow length in domain units
- `markersize=6`: base marker size
- `colormap=:viridis`: colormap for BC values
- `title=nothing`: plot title
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
    markersize=6,
    colormap=:viridis,
    title=nothing,
    legend=true,
    node_color=:gray,
    internal_node_color=:indianred,
)
    dim = dad.dimension
    L = _char_length(dad)
    arr_s = something(arrow_scale, 0.15 * L)
    ttl = something(title, _default_title(dad))

    plt = if dim == 2
        plot(;
            size=figsize,
            xlabel="x",
            ylabel="y",
            title=ttl,
            aspect_ratio=:equal,
            legend=legend ? :topright : false,
            framestyle=:box,
        )
    else
        plot(;
            size=figsize,
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=ttl,
            legend=legend ? :topright : false,
            camera=(30, 30),
        )
    end

    if show_elements
        _plot_elements!(plt, dad)
    end

    if show_nodes && !isempty(dad.Nodes)
        _scatter_points!(plt, dad.Nodes;
            color=node_color, markersize=markersize, label="collocation", marker=:circle)
    end
    if show_internal && !isempty(dad.internalNodes)
        _scatter_points!(plt, dad.internalNodes;
            color=internal_node_color, markersize=markersize, label="internal", marker=:x)
    end

    if show_normals
        nlen = 0.05 * L
        dirs = [nlen * n for n in dad.Normal]
        _arrows_from_points!(plt, dad.Nodes, dirs; color=:gray, label="normal")
    end

    if show_bc
        if dad.properties isa Scalar
            _plot_bc_scalar!(plt, dad, arr_s, colormap)
        elseif dad.properties isa Vectorial
            _plot_bc_vectorial!(plt, dad, arr_s, colormap)
        end
    end

    return plt
end

# =============================================================================
# helpers
# =============================================================================

function _default_title(dad::BEMdata)
    p = dad.properties
    if p isa Laplace
        return "Geometry & BCs — Laplace (k=$(p.k))"
    elseif p isa Helmholtz
        return "Geometry & BCs — Helmholtz (ω=$(p.ω), c=$(p.c))"
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

function _plot_elements!(plt, dad::BEMdata)
    if dad.dimension == 2
        ξ = collect(range(-1.0, 1.0; length=9))
        first = true
        for elem in dad.elements
            pts = dad.Nodes[elem.index]
            P = reduce(hcat, pts)'
            N, _ = shapefun(dad.element_type, ξ)
            C = N * P
            plot!(plt, C[:, 1], C[:, 2];
                color=:black, linewidth=1.5, label=first ? "elements" : "")
            first = false
        end
    else
        first = true
        for elem in dad.elements
            pts = dad.Nodes[elem.index]
            xs = [p[1] for p in pts]; push!(xs, xs[1])
            ys = [p[2] for p in pts]; push!(ys, ys[1])
            zs = [p[3] for p in pts]; push!(zs, zs[1])
            plot!(plt, xs, ys, zs; color=:black, linewidth=1.0, label=first ? "elements" : "")
            first = false
        end
    end
    return nothing
end

function _scatter_points!(plt, pts::Vector{<:Point2D}; kwargs...)
    scatter!(plt, [p[1] for p in pts], [p[2] for p in pts]; kwargs...)
end

function _scatter_points!(plt, pts::Vector{<:Point3D}; kwargs...)
    scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]; kwargs...)
end

function _arrows_from_points!(plt, pts::Vector{<:Point2D}, dirs; color=:dodgerblue, label="", kwargs...)
    isempty(pts) && return nothing
    # quiver: Plots expects u,v as displacements
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    u = [d[1] for d in dirs]
    v = [d[2] for d in dirs]
    quiver!(plt, x, y; quiver=(u, v), color=color, label=label, kwargs...)
    return nothing
end

function _arrows_from_points!(plt, pts::Vector{<:Point3D}, dirs; color=:dodgerblue, label="", kwargs...)
    isempty(pts) && return nothing
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    z = [p[3] for p in pts]
    u = [d[1] for d in dirs]
    v = [d[2] for d in dirs]
    w = [d[3] for d in dirs]
    # 3D quiver support varies; fall back to line segments
    first = true
    for i in eachindex(pts)
        plot!(plt, [x[i], x[i] + u[i]], [y[i], y[i] + v[i]], [z[i], z[i] + w[i]];
            color=color, label=(first ? label : ""), linewidth=1.5, kwargs...)
        first = false
    end
    return nothing
end

# ---------- scalar BC -------------------------------------------------------

function _plot_bc_scalar!(plt, dad::BEMdata{<:Scalar}, arr_s, colormap)
    dim = dad.dimension
    n = min(dad.n, length(dad.BC))
    BC = dad.BC
    BV = dad.BV

    idir = findall(i -> BC[i] == 0, 1:n)
    ineu = findall(i -> BC[i] == 1, 1:n)

    if !isempty(idir)
        pts = dad.Nodes[idir]
        vals = Float64.(BV[idir])
        if dim == 2
            scatter!(plt, [p[1] for p in pts], [p[2] for p in pts];
                marker=:utriangle, markersize=8, zcolor=vals, c=colormap,
                label="Dirichlet", colorbar_title="Dirichlet value")
        else
            scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts];
                marker=:utriangle, markersize=8, zcolor=vals, c=colormap,
                label="Dirichlet", colorbar_title="Dirichlet value")
        end
    end

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
            _arrows_from_points!(plt, pts[nz], dirs[nz]; color=:dodgerblue, label="Neumann q")
        end
        if !isempty(zidx) && dim == 2
            pz = pts[zidx]
            scatter!(plt, [p[1] for p in pz], [p[2] for p in pz];
                marker=:circle, markersize=6, color=:white,
                markerstrokecolor=:dodgerblue, markerstrokewidth=1.5,
                label="Neumann q = 0")
        end
    end
    return nothing
end

# ---------- vectorial BC ----------------------------------------------------

function _plot_bc_vectorial!(plt, dad::BEMdata{<:Vectorial}, arr_s, colormap)
    dim = dad.dimension
    n = dad.n
    BC = dad.BC
    BV = dad.BV
    ndof = dim * n
    length(BC) >= ndof || return nothing

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

    if !isempty(dir_nodes)
        pts = dad.Nodes[dir_nodes]
        if dim == 2
            scatter!(plt, [p[1] for p in pts], [p[2] for p in pts];
                marker=:utriangle, markersize=8, zcolor=dir_vals, c=colormap,
                label="Dirichlet u", colorbar_title="|u| prescribed")
        else
            scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts];
                marker=:utriangle, markersize=8, zcolor=dir_vals, c=colormap,
                label="Dirichlet u", colorbar_title="|u| prescribed")
        end
    end

    if !isempty(neu_nodes)
        pts = dad.Nodes[neu_nodes]
        maxabs = maximum(norm, neu_vecs; init=0.0)
        scale = maxabs > 0 ? arr_s / maxabs : 0.0
        dirs = [scale * t for t in neu_vecs]
        nz = findall(d -> norm(d) > 1e-14 * max(arr_s, 1.0), dirs)
        zidx = findall(i -> norm(neu_vecs[i]) <= 1e-14, eachindex(neu_vecs))
        if !isempty(nz)
            _arrows_from_points!(plt, pts[nz], dirs[nz]; color=:dodgerblue, label="Neumann t")
        end
        if !isempty(zidx) && dim == 2
            pz = pts[zidx]
            scatter!(plt, [p[1] for p in pz], [p[2] for p in pz];
                marker=:circle, markersize=6, color=:white,
                markerstrokecolor=:dodgerblue, markerstrokewidth=1.5,
                label="Neumann t = 0")
        end
    end
    return nothing
end

# =============================================================================
# Gmsh export
# =============================================================================

"""
    export_results_to_gmsh(dad, filename, result_name::Symbol; viewer=true)

Export a nodal field stored in `dad.cache` (e.g. `:T`) to a Gmsh `.msh` view,
interpolating from BEM nodes onto the mesh with an RBF.
"""
function export_results_to_gmsh(dad::BEMdata, filename::String, result_name::Symbol; viewer=true)
    has_cache(dad, result_name) ||
        error("cache.$result_name is not set — assemble/solve first?")
    return with_gmsh() do
        gmsh.clear()
        gmsh.open(filename)

        tag = gmsh.view.add(string(result_name), 0)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()

        if dad.dimension == 2
            gmsh_points = [Point2D(coords[3i - 2], coords[3i - 1]) for i in 1:length(node_tags)]
        else
            gmsh_points = [Point3D(coords[3i - 2], coords[3i - 1], coords[3i]) for i in 1:length(node_tags)]
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
        return nothing
    end
end

# =============================================================================
# VTK (legacy ASCII POLYDATA) — no extra dependency
# =============================================================================

"""
    export_vtk(dad, path; fields=(:u, :traction, :strain, :stress))

Write boundary collocation as VTK POLYDATA. 2D is placed on ``z=0``.
Cells: lines (2D) or quads/triangles (3D). Missing cache fields are skipped.
"""
function export_vtk(dad::BEMdata, path::AbstractString;
        fields=(:u, :traction, :strain, :stress))
    n = dad.n
    dim = dad.dimension
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, dad.name)
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS ", n, " float")
        @inbounds for i in 1:n
            p = dad.Nodes[i]
            if dim == 2
                @printf(io, "%.10g %.10g 0\n", p[1], p[2])
            else
                @printf(io, "%.10g %.10g %.10g\n", p[1], p[2], p[3])
            end
        end
        _vtk_cells(io, dad)
        nfields = 0
        buf = IOBuffer()
        for f in fields
            has_cache(dad, f) || continue
            data = getproperty(dad, f)
            nf = _vtk_point_data!(buf, f, data, n, dim)
            nfields += nf
        end
        if nfields > 0
            println(io, "POINT_DATA ", n)
            write(io, take!(buf))
        end
    end
    return path
end

function _vtk_cells(io, dad::BEMdata)
    ne = length(dad.elements)
    if dad.dimension == 2
        # LINES: each element is a 2-point segment (first and last collocation)
        nbytes = 3 * ne
        println(io, "LINES ", ne, " ", nbytes)
        for elem in dad.elements
            i0 = elem.index[1] - 1
            i1 = elem.index[end] - 1
            println(io, "2 ", i0, " ", i1)
        end
    else
        nbytes = sum(1 + length(elem.index) for elem in dad.elements)
        println(io, "POLYGONS ", ne, " ", nbytes)
        for elem in dad.elements
            ids = elem.index
            print(io, length(ids))
            for j in ids
                print(io, " ", j - 1)
            end
            println(io)
        end
    end
    return nothing
end

function _vtk_point_data!(io, name::Symbol, data, n::Int, dim::Int)
    if name in (:u, :traction, :T) && data isa AbstractVector && length(data) >= dim * n
        println(io, "VECTORS ", name, " float")
        @inbounds for i in 1:n
            if dim == 2
                @printf(io, "%.10g %.10g 0\n", data[2i-1], data[2i])
            else
                @printf(io, "%.10g %.10g %.10g\n", data[3i-2], data[3i-1], data[3i])
            end
        end
        return 1
    elseif name in (:strain, :stress) && data isa AbstractMatrix && size(data, 1) >= n
        println(io, "TENSORS ", name, " float")
        @inbounds for i in 1:n
            t = _voigt_to_tensor9(view(data, i, :))
            @printf(io, "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", t...)
        end
        return 1
    elseif data isa AbstractVector && length(data) >= n
        println(io, "SCALARS ", name, " float 1")
        println(io, "LOOKUP_TABLE default")
        @inbounds for i in 1:n
            @printf(io, "%.10g\n", data[i])
        end
        return 1
    end
    return 0
end

function _voigt_to_tensor9(v)
    if length(v) >= 6
        # 11 22 33 23 13 12
        return (v[1], v[6], v[5], v[6], v[2], v[4], v[5], v[4], v[3])
    else
        # 11 22 12 in xy, z=0
        return (v[1], v[3], 0.0, v[3], v[2], 0.0, 0.0, 0.0, 0.0)
    end
end

