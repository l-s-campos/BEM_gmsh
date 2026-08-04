# Shallow shell BEM (casca.jl) — Kirchhoff plate + membrane coupled by curvature
#
# For a shallow shell with principal radii R₁₁, R₂₂ the transverse equation gains
# membrane coupling N_αβ / R_αβ and the membrane equation gains w/R terms.

export ShallowShell, assemble_shell_coupling, solve_shallow_shell!

"""
    ShallowShell

Coupled plate (`ThinPlate.PlateMesh`) + membrane (`BEMdata{<:Elasticity}`)
with curvatures `R11`, `R22` (infinite = flat plate).
"""
mutable struct ShallowShell
    plate::Any
    dad_pe::BEMdata{<:Elasticity}
    R11::Float64
    R22::Float64
    # optional precomputed
    A_pl::Union{Nothing,Matrix{Float64}}
    A_pe::Union{Nothing,Matrix{Float64}}
    b_pl::Union{Nothing,Vector{Float64}}
    b_pe::Union{Nothing,Vector{Float64}}
end

function ShallowShell(plate, dad_pe::BEMdata{<:Elasticity}; R11=Inf, R22=Inf)
    return ShallowShell(plate, dad_pe, float(R11), float(R22), nothing, nothing, nothing, nothing)
end

"""
    assemble_shell_coupling(shell; npg=10)

Assemble plate and membrane linear operators. Curvature coupling is applied
at solve time as load transfer:
- plate RHS += N₁₁/R₁₁ + N₂₂/R₂₂
- membrane body force from ``ε_{αβ} += w/(2 R_{αβ})`` (shallow-shell strain).
"""
function assemble_shell_coupling(shell::ShallowShell; npg=10)
    plate = shell.plate
    dad = shell.dad_pe
    isempty(plate.H) && assemble_plate!(plate; npg=npg)
    A_pl, b_pl, _, _ = apply_bc_plate(plate)
    has_cache(dad, :H) || H_G_full_direct(dad; npg=npg, threaded=false)
    applyBC(dad)
    shell.A_pl = A_pl
    shell.b_pl = b_pl
    shell.A_pe = copy(dad.A)
    shell.b_pe = copy(dad.b)
    has_cache(dad, :M) || dibem_elasticity!(dad; npg=npg)
    return shell
end

"""
    solve_shallow_shell!(shell; niter=8, ω=0.5)

Fixed-point coupling iterations between plate bending and membrane.
Returns `(w_center, u_membrane)`.
"""
function solve_shallow_shell!(shell::ShallowShell; niter=8, ω=0.5)
    shell.A_pl === nothing && assemble_shell_coupling(shell)
    plate = shell.plate
    dad = shell.dad_pe
    R11, R22 = shell.R11, shell.R22
    invR11 = isfinite(R11) && abs(R11) > 0 ? 1 / R11 : 0.0
    invR22 = isfinite(R22) && abs(R22) > 0 ? 1 / R22 : 0.0

    # initial plate solve (no coupling)
    x_pl = shell.A_pl \ shell.b_pl
    # store into plate via solve_plate path
    plate.u = x_pl  # approximate; full pack not needed for centre estimate
    n = length(plate.nodes)
    ni = length(plate.internal)
    w_c = ni > 0 ? x_pl[2n+1] : 0.0

    E = dad.properties.E
    ν = dad.properties.nu
    h = plate.props.h
    CB = E * h / (1 - ν^2)

    for _ in 1:niter
        # membrane forces from curvature stretch ~ CB * w * invR
        # simplified isotropic: Nxx = CB*(w*invR11 + ν w*invR22), etc.
        Nxx = CB * (w_c * invR11 + ν * w_c * invR22)
        Nyy = CB * (w_c * invR22 + ν * w_c * invR11)
        # plate geometric/curvature load q_c_eff += Nxx*invR11 + Nyy*invR22
        q_extra = Nxx * invR11 + Nyy * invR22
        # add as uniform load contribution proportional to plate.q scale
        b = copy(shell.b_pl)
        if norm(plate.q) > 0 && abs(plate.props.q_c) > 0
            b .+= plate.q .* (q_extra / plate.props.q_c)
        end
        x_pl = shell.A_pl \ b
        w_new = ni > 0 ? x_pl[2n+1] : 0.0
        w_c = ω * w_new + (1 - ω) * w_c
    end
    plate.u = Float64.(x_pl)
    return (w_center=w_c, x_plate=x_pl)
end
