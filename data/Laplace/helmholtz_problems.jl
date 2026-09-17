# Helmholtz benchmark meshes + analytics (port of BEM.jl atual `data/dadhelmholtz.jl`)
#
# Frequency-domain acoustics via the **Laplace correlato**:
#   ∇²T + κ² T = 0   with   κ = ω / c
# is solved as  (H + κ² M) T = G q  using Laplace H,G and DIBEM mass M
# (same convention as BEM.jl atual `helmholtz_H_paper.jl`).
#
# BC tags: "0;T" Dirichlet, "1;q" Neumann with q = −k ∂T/∂n.

using SpecialFunctions: besselj0

# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

"""
    helm1d_mesh(; ndiv=16, ordem=1, nome="helm1d", show=false)

Unit square, 1D wave along x (legacy `helm1d`):
- bottom / top: insulated (q=0)
- right: Neumann q = −1
- left:  Dirichlet T = 0
Exact (c=1): ``T = sin(κ x) / (κ cos κ)``.
"""
function helm1d_mesh(; ndiv=16, ordem=1, Lx=1.0, Ly=1.0, nome="helm1d", show=false)
    return _square_mesh_bc(;
        nome=nome, Lx=Lx, Ly=Ly, ndiv=ndiv, ordem=ordem, show=show,
        bottom="1;0", right="1;-1", top="1;0", left="0;0",
    )
end

"""
    helmdirichlet_mesh(; ndiv=16, …)

Unit square, all-Dirichlet placeholder tags. Non-homogeneous values are filled
by [`apply_helmdirichlet_bc!`](@ref) after `format2d`.
Exact (c=1, κ>π): ``T = sin(γ x) sin(π y) / sin(γ)``, ``γ = √(κ²−π²)``.
"""
function helmdirichlet_mesh(; ndiv=16, ordem=1, Lx=1.0, Ly=1.0, nome="helmdirichlet", show=false)
    return _square_mesh_bc(;
        nome=nome, Lx=Lx, Ly=Ly, ndiv=ndiv, ordem=ordem, show=show,
        bottom="0;0", right="0;0", top="0;0", left="0;0",
    )
end

"""
    helmcirculo_mesh(; ndiv=16, …)

Quarter disk of radius 1 (legacy `helmcirculo`):
- radial edges: Neumann q=0
- arc: Dirichlet T=1
Exact: ``T = J₀(κ r) / J₀(κ)``.
"""
function helmcirculo_mesh(; ndiv=16, ordem=1, R=1.0, nome="helmcirculo", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = R / max(ndiv, 4)

    c  = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(R, 0.0, 0.0, lc)   # θ=0
    p2 = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)   # θ=π/2

    l_bot = gmsh.model.geo.addLine(c, p1)            # x-axis
    a_arc = gmsh.model.geo.addCircleArc(p1, c, p2)   # quarter arc
    l_left = gmsh.model.geo.addLine(p2, c)           # y-axis

    cl = gmsh.model.geo.addCurveLoop([l_bot, a_arc, l_left])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    nθ = max(ndiv, 4)
    nr = max(ndiv ÷ 2, 4)
    gmsh.model.mesh.setTransfiniteCurve(l_bot, nr)
    gmsh.model.mesh.setTransfiniteCurve(a_arc, nθ)
    gmsh.model.mesh.setTransfiniteCurve(l_left, nr)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

    gmsh.model.addPhysicalGroup(1, [l_bot, l_left], -1, "1;0")  # insulated
    gmsh.model.addPhysicalGroup(1, [a_arc], -1, "0;1")          # arc Dirichlet
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

# ---------------------------------------------------------------------------
# BC fill / analytics
# ---------------------------------------------------------------------------

"""Fill right-edge Dirichlet ``T = sin(π y)`` (legacy `corrigeCDC_helmdirechlet`)."""
function apply_helmdirichlet_bc!(dad::BEMdata{<:Laplace}; tol=1e-8)
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        if abs(p[1] - 1.0) < tol
            dad.BC[i] = 0
            dad.BV[i] = sin(π * p[2])
        else
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        end
    end
    return dad
end

"""Analytical ``T = sin(κ x)/(κ cos κ)`` (c = 1)."""
function ana_helm1d(κ::Real)
    κ = float(κ)
    abs(cos(κ)) < 1e-14 && error("helm1d resonance: cos(κ)≈0 at κ=$κ")
    u = (p; t=0.0) -> sin(κ * p[1]) / (κ * cos(κ))
    return AnalyticalSolution("helm1d", u; description="1D Helmholtz bar, κ=$κ")
end

"""Analytical Dirichlet cavity mode (needs κ > π)."""
function ana_helmdirichlet(κ::Real)
    κ = float(κ)
    κ > π || error("helmdirichlet needs κ > π (got $κ)")
    γ = sqrt(κ^2 - π^2)
    abs(sin(γ)) < 1e-14 && error("helmdirichlet resonance: sin(γ)≈0")
    u = (p; t=0.0) -> sin(γ * p[1]) * sin(π * p[2]) / sin(γ)
    return AnalyticalSolution("helmdirichlet", u;
        description="2D Dirichlet cavity, κ=$κ, γ=$γ")
end

"""Analytical quarter-disk: ``J₀(κ r)/J₀(κ)``."""
function ana_helmcirculo(κ::Real)
    κ = float(κ)
    j0 = besselj0(κ)
    abs(j0) < 1e-14 && error("helmcirculo resonance: J0(κ)≈0 at κ=$κ")
    u = (p; t=0.0) -> besselj0(κ * hypot(p[1], p[2])) / j0
    return AnalyticalSolution("helmcirculo", u;
        description="quarter-disk J0, κ=$κ")
end

helmholtz_problem_names() = (:helm1d, :helmdirichlet, :helmcirculo)
