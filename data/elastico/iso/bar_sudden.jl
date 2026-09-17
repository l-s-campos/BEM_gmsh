# Plane-stress bar, sudden unit end traction — elastodynamic analog of
# Laplace wave_problem(:bar_sudden).
# 1D rod: c = √(E/ρ), u_static = (P/E) x. With E=ρ=P=L=1 and ν=0 this
# matches ana_bar_sudden. ν>0 adds Poisson contraction (uy ≠ 0).

function mesh_elasticity_bar(; ndiv=12, L=1.0, P=1.0, nome="elast_bar_sudden",
        show=false, ordem=1)
    return _square_mesh_bc(;
        nome=nome, Lx=L, Ly=L, ndiv=ndiv, ordem=ordem, show=show,
        bottom="1;0;1;0",   # tx=0, ty=0
        right="1;$P;1;0",   # tx=P, ty=0  (tension)
        top="1;0;1;0",
        left="0;0;0;0",     # ux=uy=0
    )
end

"""
    elasticity_bar_sudden(; ndiv, n_int, E=1, ν=0, ρ=1, P=1, L=1)

Clamped left, sudden traction `P` on the right, traction-free sides.
`ν=0` plane stress reduces to the 1D rod (same series as Laplace).
"""
function elasticity_bar_sudden(; ndiv=12, n_int=8, E=1.0, ν=0.0, ρ=1.0,
        P=1.0, L=1.0, plane_stress=true, tipo=1, ordem=nothing, show=false,
        pad=1e-3)
    pgeo = something(ordem, tipo)
    msh = mesh_elasticity_bar(; ndiv=ndiv, L=L, P=P, ordem=pgeo,
        nome="elast_bar_n$(ndiv)_o$(pgeo)", show=show)
    props = Elasticity(E=E, nu=ν, rho=ρ; plane_stress=plane_stress)
    dad = format2d(msh, props; tipo=tipo, pontointerno=false)
    ni = something(n_int, max(ndiv ÷ 2, 4))
    set_internal_grid!(dad; nx=ni, ny=ni, x=(0, L), y=(0, L), pad=pad)
    c = sqrt(E / ρ)
    ana = ana_bar_sudden(; N=400, c=c, L=L)
    attach_analytical!(dad, ana)
    return dad, (; name=:elasticity_bar_sudden, ana, E, ν, ρ, P, L, c,
        probe=Point2D(L, L / 2), Δt=0.04, tf=4.0)
end
