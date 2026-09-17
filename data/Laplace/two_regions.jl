using DrWatson: datadir

# Two rectangular subregions sharing a vertical interface (type-3 BC)
# Left:  [0,0.5]×[0,1], Right: [0.5,1]×[0,1]
# Exterior: left wall Dirichlet 0, right wall Dirichlet 1, top/bottom insulated

function mesh_two_regions(; ndiv=8, nome="two_regions", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1

    # points
    p = Dict(
        1 => gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc),
        2 => gmsh.model.geo.addPoint(0.5, 0.0, 0.0, lc),
        3 => gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc),
        4 => gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc),
        5 => gmsh.model.geo.addPoint(0.5, 1.0, 0.0, lc),
        6 => gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc),
    )
    # left rectangle edges
    l1 = gmsh.model.geo.addLine(p[1], p[2])   # bottom L
    l2 = gmsh.model.geo.addLine(p[2], p[5])   # interface (L side, up)
    l3 = gmsh.model.geo.addLine(p[5], p[6])   # top L
    l4 = gmsh.model.geo.addLine(p[6], p[1])   # left wall
    # right rectangle
    l5 = gmsh.model.geo.addLine(p[2], p[3])   # bottom R
    l6 = gmsh.model.geo.addLine(p[3], p[4])   # right wall
    l7 = gmsh.model.geo.addLine(p[4], p[5])   # top R
    l8 = gmsh.model.geo.addLine(p[5], p[2])   # interface (R side, down)

    clL = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    clR = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    sL = gmsh.model.geo.addPlaneSurface([clL])
    sR = gmsh.model.geo.addPlaneSurface([clR])
    gmsh.model.geo.synchronize()

    for ℓ in (l1, l2, l3, l4, l5, l6, l7, l8)
        gmsh.model.mesh.setTransfiniteCurve(ℓ, ndiv)
    end
    gmsh.model.mesh.setTransfiniteSurface(sL)
    gmsh.model.mesh.setTransfiniteSurface(sR)
    gmsh.model.mesh.setRecombine(2, sL)
    gmsh.model.mesh.setRecombine(2, sR)

    # Physical BCs
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0")       # left Dirichlet 0
    gmsh.model.addPhysicalGroup(1, [l6], -1, "0;1")       # right Dirichlet 1
    gmsh.model.addPhysicalGroup(1, [l1, l3, l5, l7], -1, "1;0")  # insulated
    gmsh.model.addPhysicalGroup(1, [l2, l8], -1, "3;0")   # interface type 3
    gmsh.model.addPhysicalGroup(2, [sL], -1, "RegionL")
    gmsh.model.addPhysicalGroup(2, [sR], -1, "RegionR")

    gmsh.model.mesh.generate(2)
    outL = datadir("Laplace", nome * "_L.msh")
    outR = datadir("Laplace", nome * "_R.msh")
    mkpath(dirname(outL))

    # write full mesh then we'll extract per-region by filtering — simpler: write once
    out = datadir("Laplace", nome * ".msh")
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
Piecewise-linear exact temperature for two perfectly bonded slabs:

- left ``[0, xif]`` conductivity `kL`, right ``[xif, 1]`` conductivity `kR`
- ``T(0)=0``, ``T(1)=1``, insulated top/bottom (1-D conduction)

```
Tif = kR xif / (kL (1-xif) + kR xif)
T(x) = Tif (x/xif)                          x ≤ xif
     = Tif + (1-Tif) (x-xif)/(1-xif)        x ≥ xif
```
"""
function ana_two_layer_T(x, kL, kR; xif=0.5)
    Tif = kR * xif / (kL * (1 - xif) + kR * xif)
    return x <= xif + 1e-14 ? Tif * (x / xif) :
           Tif + (1 - Tif) * (x - xif) / (1 - xif)
end

"""
Build two BEMdata regions from a single two-region mesh by selecting elements
whose midpoint lies in x≤0.5 (left) or x≥0.5 (right).

Pass two [`Laplace`](@ref) properties for a contrast interface; a single
`props` is used on both sides.
"""
load_two_regions(msh, props::Laplace; ndiv_hint=8) =
    load_two_regions(msh, props, props; ndiv_hint=ndiv_hint)

function load_two_regions(msh, propsL::Laplace, propsR::Laplace; ndiv_hint=8)
    # Load full boundary once, then split elements by region.
    # Interface (BC type 3) is assigned by outward normal: +x → left region, −x → right.
    dad_all = format2d(msh, propsL; pontointerno=false)
    left_elems = Element[]
    right_elems = Element[]
    for e in dad_all.elements
        c = mean(dad_all.Nodes[e.index])
        n0 = mean(dad_all.Normal[e.index])
        is_if = any(dad_all.BC[i] == 3 for i in e.index)
        if is_if
            if n0[1] >= 0
                push!(left_elems, e)    # outward +x = left body
            else
                push!(right_elems, e)
            end
        elseif c[1] < 0.5 - 1e-12
            push!(left_elems, e)
        else
            push!(right_elems, e)
        end
    end
    function _subset(elems, name, props)
        # renumber nodes compactly
        old = sort(unique(vcat([e.index for e in elems]...)))
        map_n = Dict(old[i] => i for i in eachindex(old))
        Nodes = dad_all.Nodes[old]
        Normal = dad_all.Normal[old]
        BC = dad_all.BC[old]
        BV = dad_all.BV[old]
        new_elems = Element[]
        for e in elems
            idx = [map_n[i] for i in e.index]
            push!(new_elems, Element(idx, e.Jacobian, e.Length, e.Region))
        end
        n = length(Nodes)
        return BEMdata(name, 2, new_elems, dad_all.element_type, dad_all.elem_weight,
                       Nodes, Normal, props, BC, BV, n, 0, n, BEMCache())
    end
    dadL = _subset(left_elems, "left", propsL)
    dadR = _subset(right_elems, "right", propsR)
    return MultiRegionProblem([dadL, dadR]; name="two_regions")
end
