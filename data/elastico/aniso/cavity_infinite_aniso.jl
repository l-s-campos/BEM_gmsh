# Pressurized circular cavity in infinite anisotropic medium (plane strain)
# include(datadir("elastico", "aniso", "cavity_infinite_aniso.jl"))
#
# BC on cavity wall: tn = -P, tt = 0
# Analytical (isotropic limit): ur = (1+ν) P ra / E

ra = 3.0             # cavity radius (same length unit as E)
E1 = 2.07            # GPa longitudinal
E2 = 1.01            # GPa transverse
# Thesis G12 printed as 94499.9 GPa is unphysical — use orthotropic estimate:
# G12 ≈ E1 E2 / (E1 + E2 + 2 ν12 E2) style; set mild shear:
G12 = 0.945          # GPa (likely intended ~944.99 MPa scale; adjust if needed)
ν12 = 0.1
P = 100.0            # pressure
C_layers = 1
plane_strain = true

# Isotropic comparison (E≈E1, ν=ν12): ur = (1+ν) P ra² / (E r)
ur_iso(r; E=E1, ν=ν12) = (1 + ν) * P * ra^2 / (E * r)

# External sample radii (thesis Table 8.2 style, ra=3)
external_radii = (4.0, 6.0, 10.0, 20.0)

function mesh_cavity_aniso(; ndiv=32, nome="cavity_infinite_aniso", show=false, b=20 * ra)
    # truncated annulus as finite model of infinite medium
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (b - ra) / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    θs = (0.0, π / 2, π, 3π / 2)
    pin = [gmsh.model.geo.addPoint(ra * cos(θ), ra * sin(θ), 0.0, lc) for θ in θs]
    pout = [gmsh.model.geo.addPoint(b * cos(θ), b * sin(θ), 0.0, lc) for θ in θs]
    ain = ntuple(i -> gmsh.model.geo.addCircleArc(pin[i], c, pin[mod1(i + 1, 4)]), 4)
    aout = ntuple(i -> gmsh.model.geo.addCircleArc(pout[i], c, pout[mod1(i + 1, 4)]), 4)
    cl_out = gmsh.model.geo.addCurveLoop(collect(aout))
    cl_in = gmsh.model.geo.addCurveLoop([ain[4], ain[3], ain[2], ain[1]])
    s = gmsh.model.geo.addPlaneSurface([cl_out, cl_in])
    gmsh.model.geo.synchronize()
    nseg = max(ndiv ÷ 4, 4)
    for crv in (ain..., aout...)
        gmsh.model.mesh.setTransfiniteCurve(crv, nseg)
    end
    gmsh.model.addPhysicalGroup(1, collect(ain), -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, collect(aout), -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
