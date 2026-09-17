using DrWatson: datadir

# Contato MATLAB dad_Contato_Bulk.m (pad on flat specimen).

function dad_contato_bulk_params()
    w = 6.5
    R = 70.0
    θ = asin(w / R)
    y = R * (1 - cos(θ))
    L = 2 * θ * R
    E = 73.4e3
    ν = 0.33
    μ = 0.2
    cargav = 17.0
    P = 2 * w * cargav
    East = E / (1 - ν^2)
    E_eq = East / 2
    a_H = sqrt(4 * R * P / (π * E_eq))
    p0_H = 2 * P / (π * a_H)
    return (; w, R, θ, y, L, E, ν, μ, cargav, P, East, E_eq, a_H, p0_H)
end

function mesh_contato_bulk_pad(; ndiv_c=45, ndiv_s=15, ndiv_top=15, μ=0.2,
        cargav=17.0, nome="contato_bulk_pad", ordem=2)
    par = dad_contato_bulk_params()
    w, R, y = par.w, par.R, par.y
    ytop = 1.5 * w
    μs = string(float(μ)); cvs = string(-float(cargav))
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2w / ndiv_c
    c = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(-w, y, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(w, y, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(w, ytop, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(-w, ytop, 0.0, lc)
    a1 = gmsh.model.geo.addCircleArc(p1, c, p2)
    s2 = gmsh.model.geo.addLine(p2, p3)
    s3 = gmsh.model.geo.addLine(p3, p4)
    s4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([a1, s2, s3, s4])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    # Gmsh transfinite count is nodes; MATLAB MALHA is elements → n_el+1.
    gmsh.model.mesh.setTransfiniteCurve(a1, ndiv_c + 1)
    gmsh.model.mesh.setTransfiniteCurve(s2, ndiv_s + 1)
    gmsh.model.mesh.setTransfiniteCurve(s3, ndiv_top + 1)
    gmsh.model.mesh.setTransfiniteCurve(s4, ndiv_s + 1)
    gmsh.model.addPhysicalGroup(1, [a1], -1, "4;$μs;4;$μs;8;11")
    gmsh.model.addPhysicalGroup(1, [s2], -1, "1;0;1;0;8;12")
    gmsh.model.addPhysicalGroup(1, [s3], -1, "1;0;1;$cvs;8;13")
    gmsh.model.addPhysicalGroup(1, [s4], -1, "1;0;1;0;8;14")
    gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(Int(ordem))
    out = datadir("elastico", nome * ".msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function mesh_contato_bulk_spec(; ndiv_c=45, ndiv_out=15, ndiv_s=15, ndiv_bot=15,
        μ=0.2, nome="contato_bulk_spec", ordem=2)
    par = dad_contato_bulk_params()
    w, L = par.w, par.L
    W = w + 3L / 4
    H = 2w
    μs = string(float(μ))
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2W / ndiv_bot
    t1 = gmsh.model.geo.addPoint(-W, 0.0, 0.0, lc)
    t2 = gmsh.model.geo.addPoint(-L / 2, 0.0, 0.0, lc)
    t3 = gmsh.model.geo.addPoint(L / 2, 0.0, 0.0, lc)
    t4 = gmsh.model.geo.addPoint(W, 0.0, 0.0, lc)
    bR = gmsh.model.geo.addPoint(W, -H, 0.0, lc)
    bL = gmsh.model.geo.addPoint(-W, -H, 0.0, lc)
    topL = gmsh.model.geo.addLine(t1, t2)
    cC = gmsh.model.geo.addLine(t2, t3)
    topR = gmsh.model.geo.addLine(t3, t4)
    right = gmsh.model.geo.addLine(t4, bR)
    bot = gmsh.model.geo.addLine(bR, bL)
    left = gmsh.model.geo.addLine(bL, t1)
    cl = gmsh.model.geo.addCurveLoop([topL, cC, topR, right, bot, left])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(topL, ndiv_out + 1)
    gmsh.model.mesh.setTransfiniteCurve(cC, ndiv_c + 1)
    gmsh.model.mesh.setTransfiniteCurve(topR, ndiv_out + 1)
    gmsh.model.mesh.setTransfiniteCurve(right, ndiv_s + 1)
    gmsh.model.mesh.setTransfiniteCurve(bot, ndiv_bot + 1)
    gmsh.model.mesh.setTransfiniteCurve(left, ndiv_s + 1)
    gmsh.model.addPhysicalGroup(1, [topL], -1, "1;0;1;0;8;31")
    gmsh.model.addPhysicalGroup(1, [cC], -1, "4;$μs;4;$μs;8;32")
    gmsh.model.addPhysicalGroup(1, [topR], -1, "1;0;1;0;8;33")
    gmsh.model.addPhysicalGroup(1, [right], -1, "1;0;1;0;8;34")
    gmsh.model.addPhysicalGroup(1, [bot], -1, "1;0;0;0;8;35")   # uy=0
    gmsh.model.addPhysicalGroup(1, [left], -1, "1;0;1;0;8;36")
    gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(Int(ordem))
    out = datadir("elastico", nome * ".msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

"""Pin `u_x=0` on the horizontal-face node closest to `x=0` (Contato `NOS_RES` gdl=2)."""
function pin_face_center_ux!(dad; face::Symbol=:top)
    ys = [pt[2] for pt in dad.Nodes]
    yref = face === :top ? maximum(ys) : minimum(ys)
    thr = 1e-9 * max(abs(yref), maximum(abs, ys), 1.0)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - yref) <= thr || continue
        ax = abs(dad.Nodes[i][1]); ax < bx && (bx = ax; best = i)
    end
    best == 0 && return 0
    dad.BC[2best - 1] = 0
    dad.BV[2best - 1] = 0.0
    return best
end

function load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, μ=nothing, tipo=2,
        collocation=:legendre, gap=:euclidean, nome="contato_bulk")
    par = dad_contato_bulk_params()
    μv = μ === nothing ? par.μ : float(μ)
    props = Elasticity(par.E, par.ν, 1.0; plane_strain=true)
    msh_p = mesh_contato_bulk_pad(; ndiv_c=ndiv_c, ndiv_s=ndiv_s, ndiv_top=ndiv_s,
        μ=μv, cargav=par.cargav, nome=nome * "_pad", ordem=tipo)
    msh_s = mesh_contato_bulk_spec(; ndiv_c=ndiv_c, ndiv_out=ndiv_s, ndiv_s=ndiv_s,
        ndiv_bot=ndiv_s, μ=μv, nome=nome * "_spec", ordem=tipo)
    dad_p = format2d(msh_p, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_s = format2d(msh_s, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_p.name = "pad"; dad_s.name = "specimen"
    # Contato NOS_RES: pad top-center u_t=0 and specimen bottom-center u_t=0.
    pin_face_center_ux!(dad_p; face=:top)
    pin_face_center_ux!(dad_s; face=:bottom)
    prob = MultiRegionProblem([dad_p, dad_s]; name="dad_contato_bulk")
    pair_contacts!(prob; method=:ntn, slave_reg=1, master_reg=2, gap=gap)
    for cp in prob.contacts
        cp.μ = μv; cp.ut_lock = 0.0; cp.state = 1
    end
    return prob, par
end
