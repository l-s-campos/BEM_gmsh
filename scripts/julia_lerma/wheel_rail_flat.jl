# Chalmers wheel-flat (CONTACT wheelflat.inp) — variable wheel + FZ=125 kN.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, FFTW

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA = joinpath(ROOT, "data", "contact", "vollebregt")
const CEX  = get(ENV, "CONTACT_EXAMPLES",
                 raw"C:\Users\ufesl\Downloads\CONTACT-main\CONTACT-main\examples")

slcw = joinpath(DATA, "S1002_flat.slcw")
railp = joinpath(DATA, "r300_wide.prr")
isfile(joinpath(CEX, "S1002_flat", "Wheel_section_208.txt")) ||
    error("slice files not found under $CEX/S1002_flat")

println("reading profiles...")
rail = read_rail_profile(railp)
wheel = read_wheel_profile(slcw; scale=1.0, mirror_z=-1)  # inp MIRRORZ=-1
println("  rail n=", length(rail.spl), "  wheel slices=", length(wheel.slices))

trk = TrackGeom(cant=0.020)
ws = WheelsetGeom(nom_radius=490.0, z=0.3729, pitch=-25π/180, vs=2000.0,
                  vpitch=-4.08190679, fz=125000.0)
println("locate at CONTACT z_ws, pitch=-25° ...")
res = solve_wheel_rail(trk, ws, rail, wheel; side=:right, force=false, dx=0.4, ds=0.4)
if isempty(res.patches)
    println("NO CONTACT")
else
    p = res.patches[1]
    println(@sprintf("  ours    FN=%.0f  FX=%.0f  pmax=%.0f  YCP=%.2f  ξx=%.3e",
                     p.FN, p.FX, p.pmax, p.YCP_tr, p.ξx))
    println("  CONTACT FN=125000  FX=7499  pmax=1096  YCP=759.33  ξx=-4.16e-4")
end
