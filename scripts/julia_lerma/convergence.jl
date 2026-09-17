# Print mesh-convergence table for every Ch. 3 example.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

println("="^72)
println(" Juliá Lerma Ch.3 mesh convergence")
println("="^72)

c = pin_hertz(13; L=0.6); f = pin_hertz(21; L=0.6)
@printf "pin Hertz     errP  %.4f → %.4f   errp %.4f → %.4f   errVM %.4f → %.4f\n" c.errP f.errP c.errp f.errp c.errVM f.errVM

c = pin_wear(13; Δs=4.0, nsteps=3); f = pin_wear(21; Δs=4.0, nsteps=3)
@printf "pin Argatov   errw  %.4f → %.4f   w/w_arg=%.3f\n" c.errw f.errw f.w / f.w_arg

c = pin_friction(13); f = pin_friction(21)
@printf "pin friction  P     %.4f → %.4f   Qx %.4f → %.4f\n" c.P f.P c.Qx f.Qx

c = pin_orthotropic(13; β=π/4, nsteps=2); f = pin_orthotropic(21; β=π/4, nsteps=2)
@printf "pin β=45°     Qy    %.3e → %.3e   w %.3e → %.3e\n" c.Qy f.Qy c.w f.w

c = spherical_fretting(13); f = spherical_fretting(21)
@printf "fretting β=0  P     %.4f → %.4f   n_slip %d → %d\n" c.P f.P c.n_slip f.n_slip

c = flat_punch_static(17); f = flat_punch_static(25)
@printf "flat Sneddon  |P-P*| /P  %.4f → %.4f   pmax/pcen %.2f\n" abs(c.P-PUNCH.P)/PUNCH.P abs(f.P-PUNCH.P)/PUNCH.P f.pmax/f.pcen

c = rolling_spheres(13); f = rolling_spheres(21)
@printf "rolling iso   errP  %.4f → %.4f   Qx/μP %.3f  Qy %.2e\n" c.errP f.errP f.Qx_over_μP f.Qy
o = rolling_spheres(17; β=π/4)
@printf "rolling β=45  Qx    %.4f   Qy %.4e\n" o.Qx o.Qy

c = twin_discs(9; nx=9, ny=27); f = twin_discs(13; nx=13, ny=39)
@printf "twin discs    errP  %.4f → %.4f   w %.3e → %.3e\n" c.errP f.errP c.w f.w
println("="^72)
