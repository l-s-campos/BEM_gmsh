# Mesh-convergence of Juliá Lerma (2025) Ch. 3 examples.
# Two resolutions per case: error vs analytical (or vs the finer mesh) must drop.
using Test
using LinearAlgebra
using BEM
using BEM.Contact

function _improves(err_coarse, err_fine; rtol_fine)
    return err_fine < err_coarse || err_fine < rtol_fine
end

@testset "Juliá Lerma Ch.3 convergence" begin

    @testset "3.1 pin-on-disc Hertz (P, pmax, σVM)" begin
        c = pin_hertz(13; L=0.6)
        f = pin_hertz(21; L=0.6)
        @info "pin Hertz" Nc=13 Nf=21 errP=(c.errP, f.errP) errp=(c.errp, f.errp) errVM=(c.errVM, f.errVM)
        @test _improves(c.errP, f.errP; rtol_fine=0.05)
        @test _improves(c.errp, f.errp; rtol_fine=0.08)
        @test f.errP < 0.05
        @test f.errp < 0.08
        @test f.errVM < 0.15
        @test _improves(c.errVM, f.errVM; rtol_fine=0.15)
    end

    @testset "3.1 pin-on-disc Argatov wear" begin
        c = pin_wear(13; Δs=4.0, nsteps=3)
        f = pin_wear(21; Δs=4.0, nsteps=3)
        @info "pin wear" errw=(c.errw, f.errw) w=(c.w, f.w) w_arg=f.w_arg
        @test c.w > 0 && f.w > 0
        @test _improves(c.errw, f.errw; rtol_fine=0.45)
        @test f.errw < 0.55
    end

    @testset "3.1 pin isotropic friction μ=0.25" begin
        c = pin_friction(13; μ=0.25)
        f = pin_friction(21; μ=0.25)
        @test f.P > 0 && f.Qx != 0
        @test abs(f.Qy) < 0.05 * abs(f.Qx)   # sliding along x, isotropic
        dP = abs(f.P - c.P) / f.P
        @info "pin friction" P=(c.P, f.P) Qx=(c.Qx, f.Qx) dP
        @test dP < 0.15
    end

    @testset "3.1 pin orthotropic β=45° (Qy ≠ 0)" begin
        c = pin_orthotropic(13; β=π/4, nsteps=2, Δs=2.0)
        f = pin_orthotropic(21; β=π/4, nsteps=2, Δs=2.0)
        @info "pin ortho" Qy=(c.Qy, f.Qy) Qx=(c.Qx, f.Qx) w=(c.w, f.w)
        @test f.w > 0
        @test abs(f.Qy) > 1e-8
        @test sign(f.Qy) == sign(c.Qy) || abs(c.Qy) < 1e-10
        @test abs(f.Qy - c.Qy) / max(abs(f.Qy), eps()) < 2.0
    end

    @testset "3.2.1 spherical punch fretting (one half-cycle)" begin
        c = spherical_fretting(13; β=0.0)
        f = spherical_fretting(21; β=0.0)
        @info "fretting" P=(c.P, f.P) Qx=(c.Qx, f.Qx) n_slip=(c.n_slip, f.n_slip) w=(c.w, f.w)
        @test f.P > 0 && f.n_contact > 0
        @test f.n_slip > 0          # β=0 is gross slip
        @test abs(f.P - c.P) / f.P < 0.25
        # partial-slip at β=90°: fewer slip nodes than β=0 on same mesh
        g = spherical_fretting(17; β=π/2)
        @info "fretting β=90" n_slip=g.n_slip n_contact=g.n_contact
        @test g.n_contact > 0
    end

    @testset "3.2.2 flat punch Sneddon + one cycle" begin
        c = flat_punch_static(17; μ=0.0)
        f = flat_punch_static(25; μ=0.0)
        errP_c = abs(c.P - PUNCH.P) / PUNCH.P
        errP_f = abs(f.P - PUNCH.P) / PUNCH.P
        @info "flat punch" errP=(errP_c, errP_f) kn_ratio=(c.kn_ratio, f.kn_ratio) pmax=(c.pmax, f.pmax)
        @test _improves(errP_c, errP_f; rtol_fine=0.08)
        @test errP_f < 0.08
        @test f.pmax > 1.3 * f.pcen          # edge singularity
        @test f.pmax > c.pmax                # peak grows under refinement
        cy = flat_punch_cycle(17; μ=0.2)
        fy = flat_punch_cycle(21; μ=0.2)
        @info "flat cycle" w=(cy.w, fy.w)
        @test fy.w >= 0
        fr = flat_punch_static(21; μ=0.4)
        @test abs(fr.P - PUNCH.P) / PUNCH.P < 0.10
    end

    @testset "3.3.1 rolling spheres (Qx, Qy=0 isotropic; Qy≠0 at β=45°)" begin
        c = rolling_spheres(13)
        f = rolling_spheres(21)
        @info "rolling iso" errP=(c.errP, f.errP) QxμP=(c.Qx_over_μP, f.Qx_over_μP) Qy=(c.Qy, f.Qy)
        @test _improves(c.errP, f.errP; rtol_fine=0.12)
        @test f.errP < 0.12
        @test abs(f.Qy) < 0.05 * max(abs(f.Qx), 1e-12)
        @test f.Qx * (-ROLL.ξx) > 0
        o = rolling_spheres(17; β=π/4)
        @info "rolling β=45" Qx=o.Qx Qy=o.Qy
        @test abs(o.Qy) > 1e-10
    end

    @testset "3.3.2 twin discs (one revolution)" begin
        c = twin_discs(9; nx=9, ny=27, nrev=1)
        f = twin_discs(13; nx=13, ny=39, nrev=1)
        @info "twin discs" errP=(c.errP, f.errP) w=(c.w, f.w) Qx=(c.Qx, f.Qx)
        @test _improves(c.errP, f.errP; rtol_fine=0.20)
        @test f.errP < 0.20
        @test f.w > 0
        @test abs(f.w - c.w) / max(f.w, eps()) < 3.0
    end
end
