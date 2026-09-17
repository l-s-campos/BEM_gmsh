# Projected Coulomb frictional contact — GNM & accelerated Newton
#
# Algorithms after Rodríguez-Tembleque & Abascal (IJNME 2013), specialised to
# 2D Contato unknowns (traction projection P / P_g, frozen-state Newton):
#
#   A_α x_α − G_cα t_α = b_α ,   α = 1,2
#   equilibrium  t₁ + R t₂ = 0
#   contact      P t + P_g g(u) = 0
#
# Public solvers (via solve_contact_friction*!):
#   :proj_gnm    — residual generalised Newton with paper P / P_g
#   :proj_newton — accelerated Newton (§6.1): frozen-state solve +
#                  quasi-complementarity reduction + line search
#   :gnmls       — Comput. Struct. 2010: one Λ per pair (p¹=Λ, p²=−R^{-1}Λ),
#                  frozen slip, quasi-complementarity, Pang line search
#
# Public projections:
#   project_contact_traction, project_contact_multipliers


# Projection operators (paper §3.3)


"""Paper ``P_R``: ``P_R(x) = min(x, 0)`` (compression = negative traction)."""
project_contact_normal(tn_star::Real) = min(float(tn_star), 0.0)

"""Paper ``P_E`` (2D isotropic): project onto disk of radius ``radius = |t_n|``."""
function project_contact_tangent(tt_star::Real, radius::Real)
    r = max(float(radius), 0.0)
    τ = float(tt_star)
    abs(τ) <= r + 1e-15 && return τ
    return τ == 0.0 ? 0.0 : sign(τ) * r
end

"""Paper (21): ``t_n^★ = t_n + r_n g_n``, ``t_t^★ = t_t − r_t g_t``."""
function augmented_contact_tractions(tn, tt, gn, gt, rn, rt)
    return tn + rn * gn, tt - rt * gt
end

"""
Full paper contact operator ``t ← P_{C_f}(t^★)`` (eq. 17–20), isotropic Coulomb.
Traction signs: ``t_n ≤ 0`` in contact.
"""
function project_contact_traction(tn, tt, gn, gt, μ, rn, rt)
    tn★, tt★ = augmented_contact_tractions(tn, tt, gn, gt, rn, rt)
    tn_new = project_contact_normal(tn★)
    bound = float(μ) * abs(tn_new)
    tt_new = project_contact_tangent(tt★, bound)
    if tn_new >= -1e-15
        return 0.0, 0.0, :open
    elseif abs(tt★) <= bound + 1e-15
        return tn_new, tt_new, :stick
    else
        return tn_new, tt_new, :slip
    end
end

"""
Multiplier form (Contato / Alart–Curnier dual): ``λ = −t``.

```
λn ← max(0, λn − rn gn)
λt ← proj_{|·|≤μ λn}(λt − rt gt)
```
"""
function project_contact_multipliers(gn, gt, λn, λt, μ, rn, rt)
    λn_new = max(0.0, float(λn) - float(rn) * float(gn))
    τt = float(λt) - float(rt) * float(gt)
    bound = float(μ) * λn_new
    if abs(τt) <= bound + 1e-15
        λt_new = τt
        regime = λn_new <= 1e-15 ? :open : :stick
    else
        s = τt == 0.0 ? 1.0 : sign(τt)
        λt_new = s * bound
        regime = λn_new <= 1e-15 ? :open : :slip
    end
    if λn_new <= 1e-15
        λt_new = 0.0
        regime = :open
    end
    return λn_new, λt_new, regime
end

# -----------------------------------------------------------------------------
# State classification from augmented tractions (paper §6.1)
# -----------------------------------------------------------------------------

"""
Classify contact state from augmented tractions (paper §3.3 / §6.1).
Returns `(regime, tn★, tt★, ωt)` with ``ωt = t_t^★ / |t_t^★|`` on slip.
"""
function augmented_contact_state(tn, tt, gn, gt, μ, rn, rt)
    tn★, tt★ = augmented_contact_tractions(tn, tt, gn, gt, rn, rt)
    if tn★ > 0
        return :open, tn★, tt★, 0.0
    end
    bound = μ * abs(min(tn★, 0.0))
    if abs(tt★) <= bound + 1e-15
        return :stick, tn★, tt★, 0.0
    end
    ωt = tt★ == 0.0 ? 1.0 : sign(tt★)
    return :slip, tn★, tt★, ωt
end

# -----------------------------------------------------------------------------
# Residual GNM with paper P / P_g (paper §5, eq. 31–33 / 44)
# -----------------------------------------------------------------------------

function _assemble_proj_gnm_R_J(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 4 * np
    R = zeros(N)
    J = zeros(N, N)

    for pr in prep
        o, nd = pr.off, pr.ndof
        xr = @view x[o+1:o+nd]
        J[o+1:o+nd, o+1:o+nd] .= pr.A
        mul!(@view(R[o+1:o+nd]), pr.A, xr)
        R[o+1:o+nd] .-= pr.b
    end
    for (k, cp) in enumerate(pairs)
        pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)
        t1 = SVector(x[ot+1], x[ot+2])
        t2 = SVector(x[ot+3], x[ot+4])
        if haskey(pr1.Gc_cols, na)
            G1 = pr1.G_local[:, pr1.Gc_cols[na]]
            R[pr1.off+1:pr1.off+pr1.ndof] .-= G1 * t1
            J[pr1.off+1:pr1.off+pr1.ndof, ot+1:ot+2] .-= G1
        end
        if haskey(pr2.Gc_cols, nb)
            G2 = pr2.G_local[:, pr2.Gc_cols[nb]]
            R[pr2.off+1:pr2.off+pr2.ndof] .-= G2 * t2
            J[pr2.off+1:pr2.off+pr2.ndof, ot+3:ot+4] .-= G2
        end
    end

    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx; ht_k=ht[k])
        Rm, ot, iu1, iu2 = kin.R, kin.ot, kin.iu1, kin.iu2
        tn1, tt1 = kin.tn1, kin.tt1
        r1, r2, r3, r4 = ot + 1, ot + 2, ot + 3, ot + 4

        # equilibrium (always)
        R[r3] = tn1 + Rm[1, 1] * kin.tn2 + Rm[1, 2] * kin.tt2
        R[r4] = tt1 + Rm[2, 1] * kin.tn2 + Rm[2, 2] * kin.tt2
        J[r3, ot+1] = 1.0; J[r3, ot+3] = Rm[1, 1]; J[r3, ot+4] = Rm[1, 2]
        J[r4, ot+2] = 1.0; J[r4, ot+3] = Rm[2, 1]; J[r4, ot+4] = Rm[2, 2]

        regime, _, _, ωt = augmented_contact_state(tn1, tt1, kin.gn, kin.gt, cp.μ, rn, rt)
        if regime === :open
            # P = I  →  t = 0
            R[r1] = tn1; R[r2] = tt1
            J[r1, ot+1] = 1.0; J[r2, ot+2] = 1.0
            cp.state = 1
        elseif regime === :stick
            # P = 0, P_g g = 0  →  g = 0
            R[r1] = kin.gn; R[r2] = kin.gt
            _add_gn_row!(J, r1, iu1, iu2, Rm, 1.0)
            _add_gt_row!(J, r2, iu1, iu2, Rm, 1.0)
            cp.state = 3
        else
            # simplified slip: gn = 0, tt − μ |tn| ωt = 0  (ωt frozen)
            μ = cp.μ
            R[r1] = kin.gn
            R[r2] = tt1 - μ * abs(tn1) * ωt
            _add_gn_row!(J, r1, iu1, iu2, Rm, 1.0)
            J[r2, ot+2] = 1.0
            if abs(tn1) > 1e-15
                J[r2, ot+1] = -μ * ωt * sign(tn1)
            end
            cp.state = Int(2 * ωt)
        end
        cp.tn = tn1; cp.tt = tt1
    end
    return R, J
end

_assemble_proj_gnm_R(prep, pairs, h, x; ht=nothing, rn=1.0, rt=1.0) =
    first(_assemble_proj_gnm_R_J(prep, pairs, h, x; ht=ht, rn=rn, rt=rt))

"""Residual GNM (paper P/P_g) with Armijo line search."""
function _contact_proj_gnm!(prep, pairs, h, x_init;
        ht=nothing, tol=1e-8, maxiter=40, verbose=false,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        ls_max::Int=8)
    x = collect(Float64, x_init)
    rn0, rt0 = _default_contact_r(prep)
    rn_ = rn === nothing ? rn0 : float(rn)
    rt_ = rt === nothing ? rt0 : float(rt)
    ok = false
    nR = Inf
    for it in 1:maxiter
        R, J = _assemble_proj_gnm_R_J(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
        nR = norm(R)
        verbose && @info "contact proj-GNM" it nR n_closed=count(cp -> abs(cp.state) != 1, pairs)
        nR < tol && (ok = true; break)
        dx = J \ (-R)
        α = 1.0
        x_trial = similar(x)
        nR_new = nR
        for _ in 1:ls_max
            x_trial .= x .+ α .* dx
            nR_new = norm(_assemble_proj_gnm_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_))
            (nR_new < (1 - 1e-4 * α) * nR || nR_new < tol) && break
            α *= 0.5
        end
        x .= x_trial
        nR_new < tol && (ok = true; break)
        if α * norm(dx) < tol * max(1.0, norm(x))
            ok = nR_new < max(tol, 1e-6 * max(1.0, nR))
            break
        end
    end
    R, _ = _assemble_proj_gnm_R_J(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
    nR = norm(R)
    ok = ok || nR < tol
    verbose && @info "contact proj-GNM done" ok nR
    return x, ok
end

# -----------------------------------------------------------------------------
# Accelerated Newton (paper §6.1) — frozen-state solve + reduction + LS
# -----------------------------------------------------------------------------

"""
Build the **frozen-state linear system** ``A z = b`` of paper eqs. (31)–(33)/(44)
on Contato unknowns (BIE + equilibrium + P/P_g contact rows).

This is the matrix ``R^{(n)}`` in paper eq. (45): one active-set linearisation
with simplified slip Jacobian (frozen ``ω_t``).
"""
function _assemble_proj_frozen_Ab(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0)
    # Same structure as residual Jacobian of the GNM form evaluated as
    # A z = b  ⇔  R(z) = A z - b = 0  with frozen regime pieces that are affine.
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 4 * np
    A = zeros(N, N)
    b = zeros(N)
    regimes = Vector{Symbol}(undef, np)
    ωts = zeros(np)

    # BIE: A_r x_r - G_c t = b_r
    for pr in prep
        o, nd = pr.off, pr.ndof
        A[o+1:o+nd, o+1:o+nd] .= pr.A
        b[o+1:o+nd] .= pr.b
    end
    for (k, cp) in enumerate(pairs)
        pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)
        if haskey(pr1.Gc_cols, na)
            G1 = pr1.G_local[:, pr1.Gc_cols[na]]
            A[pr1.off+1:pr1.off+pr1.ndof, ot+1:ot+2] .-= G1
        end
        if haskey(pr2.Gc_cols, nb)
            G2 = pr2.G_local[:, pr2.Gc_cols[nb]]
            A[pr2.off+1:pr2.off+pr2.ndof, ot+3:ot+4] .-= G2
        end
    end

    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx; ht_k=ht[k])
        Rm, ot, iu1, iu2 = kin.R, kin.ot, kin.iu1, kin.iu2
        r1, r2, r3, r4 = ot + 1, ot + 2, ot + 3, ot + 4

        regime, _, _, ωt = augmented_contact_state(
            kin.tn1, kin.tt1, kin.gn, kin.gt, cp.μ, rn, rt)
        regimes[k] = regime
        ωts[k] = ωt

        if regime === :open
            # pin all four tractions (needed when reduce=false)
            A[r1, ot+1] = 1.0
            A[r2, ot+2] = 1.0
            A[r3, ot+3] = 1.0
            A[r4, ot+4] = 1.0
            cp.state = 1
        elseif regime === :stick
            # equilibrium: t1 + R t2 = 0
            A[r3, ot+1] = 1.0; A[r3, ot+3] = Rm[1, 1]; A[r3, ot+4] = Rm[1, 2]
            A[r4, ot+2] = 1.0; A[r4, ot+3] = Rm[2, 1]; A[r4, ot+4] = Rm[2, 2]
            # gn = 0 ⇒ -un1 + R[1,:]·u2 = -h
            A[r1, iu1] = -1.0
            A[r1, iu2] = Rm[1, 1]
            A[r1, iu2+1] = Rm[1, 2]
            b[r1] = -h[k]
            # gt = dut - ht - ut_lock = 0  ⇒ dut = ut_lock + ht
            A[r2, iu1+1] = 1.0
            A[r2, iu2] = -Rm[2, 1]
            A[r2, iu2+1] = -Rm[2, 2]
            b[r2] = cp.ut_lock + ht[k]
            cp.state = 3
        else
            # equilibrium: t1 + R t2 = 0
            A[r3, ot+1] = 1.0; A[r3, ot+3] = Rm[1, 1]; A[r3, ot+4] = Rm[1, 2]
            A[r4, ot+2] = 1.0; A[r4, ot+3] = Rm[2, 1]; A[r4, ot+4] = Rm[2, 2]
            # gn = 0
            A[r1, iu1] = -1.0
            A[r1, iu2] = Rm[1, 1]
            A[r1, iu2+1] = Rm[1, 2]
            b[r1] = -h[k]
            # Simplified slip: tt − μ |tn| ωt = 0.  tn ≤ 0 ⇒ tt + μ ωt tn = 0.
            A[r2, ot+2] = 1.0
            if kin.tn1 <= 0
                A[r2, ot+1] = cp.μ * ωt
            end
            cp.state = Int(2 * (ωt == 0 ? 1.0 : ωt))
        end
        cp.tn = kin.tn1
        cp.tt = kin.tt1
    end
    return A, b, regimes, ωts
end

"""
Quasi-complementarity reduction (paper §6.1, eqs. 45–46).

Eliminates known contact DOFs:
- **open** — ``t_n = t_t = 0`` on both bodies (4 DOFs fixed)
- **stick** — no elimination (``t`` free, ``g = 0``)
- **slip** — substitute ``t_t = -ω_t t_n`` (paper simplified slip), drop ``t_t`` column

Returns reduced ``(A_red, b_red, free)`` and a scatter map.
"""
function _proj_reduce_system(A, b, prep, pairs, regimes, ωts)
    nx = sum(p.ndof for p in prep)
    N = size(A, 1)
    fixed = falses(N)
    fixed_val = zeros(N)

    for (k, cp) in enumerate(pairs)
        ot = nx + 4(k - 1)
        reg = regimes[k]
        if reg === :open
            for j in 0:3
                fixed[ot + 1 + j] = true
                fixed_val[ot + 1 + j] = 0.0
            end
        elseif reg === :slip
            # eliminate tt1; reconstruct after solve as tt1 = -μ ωt tn1
            fixed[ot + 2] = true
            fixed_val[ot + 2] = 0.0
        end
    end

    # Fold slip columns before extracting free set
    A2 = copy(A)
    b2 = copy(b)
    for (k, cp) in enumerate(pairs)
        regimes[k] === :slip || continue
        ot = nx + 4(k - 1)
        ωt = ωts[k]
        # tt1 = -μ ωt tn1  ⇒  fold column tt1 into tn1
        A2[:, ot+1] .-= (cp.μ * ωt) .* A2[:, ot+2]
    end

    free = findall(!, fixed)
    # A_ff y = b_f - A_fc x_c
    b_red = b2[free]
    for j in findall(fixed)
        if abs(fixed_val[j]) > 0
            b_red .-= A2[free, j] .* fixed_val[j]
        end
    end
    A_red = A2[free, free]
    return A_red, b_red, free, fixed, fixed_val, regimes, ωts
end

function _proj_scatter_reduced!(x, y, free, fixed, fixed_val, prep, pairs, regimes, ωts)
    fill!(x, 0.0)
    x[free] .= y
    x[fixed] .= fixed_val[fixed]
    # reconstruct slip tt1 = -ωt * tn1 and body-2 tractions from equilibrium
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        ot = nx + 4(k - 1)
        if regimes[k] === :slip
            ωt = ωts[k]
            tn1 = x[ot + 1]
            x[ot + 2] = -(cp.μ * ωt) * tn1
        elseif regimes[k] === :open
            x[ot+1:ot+4] .= 0.0
        end
        # enforce equilibrium on body 2: t2 = -R \\ t1
        pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
        R1 = Matrix(node_rotation2d(pr1.dad.Normal[cp.node_a]))
        R2 = Matrix(node_rotation2d(pr2.dad.Normal[cp.node_b]))
        R = R2 * R1'
        t2 = -(R \ SVector(x[ot+1], x[ot+2]))
        x[ot+3] = t2[1]
        x[ot+4] = t2[2]
        cp.tn = x[ot+1]
        cp.tt = x[ot+2]
    end
    return x
end

"""
    _contact_proj_newton!(prep, pairs, h, x_init; kwargs...) -> (x, ok)

**Accelerated Newton** algorithm of Rodríguez-Tembleque & Abascal (2013) §6.1.

```
for n = 0,1,…
    classify pairs from augmented tractions t★ (eq. 21)
    build frozen-state system  R^{(n)} z = F   (P / P_g of eqs. 31–33, 44)
    reduce by quasi-complementarity (half contact DOFs known)
    solve reduced system → z̃
    line search:  z^{n+1} = α z̃ + (1−α) z^n
```

The reduction is the paper's main acceleration for BEM (high contact/total DOF
ratio). Line search follows Pang's GNMls as in the paper.
"""
function _contact_proj_newton!(prep, pairs, h, x_init;
        ht=nothing, tol=1e-8, maxiter=40, verbose=false,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        ls_max::Int=10,
        β_ls::Float64=0.5,          # paper β ∈ (0, 1/2) Armijo
        reduce::Bool=true)
    x = collect(Float64, x_init)
    rn0, rt0 = _default_contact_r(prep)
    rn_ = rn === nothing ? rn0 : float(rn)
    rt_ = rt === nothing ? rt0 : float(rt)
    ok = false
    nR = Inf

    for it in 1:maxiter
        # residual of current point (for LS merit and stop)
        R0 = _assemble_proj_gnm_R(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
        nR = norm(R0)
        verbose && @info "contact proj-Newton" it nR n_closed=count(cp -> abs(cp.state) != 1, pairs)
        if nR < tol
            ok = true
            break
        end

        # frozen-state linear system R^{(n)} z̃ = F
        A, bvec, regimes, ωts = _assemble_proj_frozen_Ab(
            prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)

        if reduce
            A_red, b_red, free, fixed, fval, regimes, ωts =
                _proj_reduce_system(A, bvec, prep, pairs, regimes, ωts)
            if length(free) == 0
                verbose && @warn "proj-Newton: no free DOFs"
                break
            end
            y = A_red \ b_red
            x_tilde = similar(x)
            _proj_scatter_reduced!(x_tilde, y, free, fixed, fval, prep, pairs, regimes, ωts)
        else
            x_tilde = A \ bvec
        end

        # GNMls line search: z ← α z̃ + (1−α) z   (paper step 3–4)
        # merit ψ = ½‖R‖², accept if ψ(z+αΔ) ≤ (1 − 2β α) ψ(z)
        ψ0 = 0.5 * nR^2
        Δ = x_tilde .- x
        α = 1.0
        x_trial = similar(x)
        nR_new = nR
        accepted = false
        for _ in 1:ls_max
            x_trial .= x .+ α .* Δ
            nR_new = norm(_assemble_proj_gnm_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_))
            ψ = 0.5 * nR_new^2
            if ψ <= (1.0 - 2 * β_ls * α) * ψ0 || nR_new < tol
                accepted = true
                break
            end
            α *= 0.5
        end
        if !accepted
            x_trial .= x .+ α .* Δ
            nR_new = norm(_assemble_proj_gnm_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_))
        end
        x .= x_trial

        if nR_new < tol
            ok = true
            break
        end
        if α * norm(Δ) < tol * max(1.0, norm(x))
            ok = nR_new < max(tol, 1e-6 * max(1.0, nR))
            break
        end
    end

    # final state tags
    _assemble_proj_gnm_R_J(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
    nR = norm(_assemble_proj_gnm_R(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_))
    ok = ok || nR < tol
    verbose && @info "contact proj-Newton done" ok nR
    return x, ok
end

# =============================================================================
# Rodríguez-Tembleque & Abascal, Comput. Struct. 88 (2010) 924–937
# BEM–BEM GNM with line search, single contact traction Λ, quasi-complementarity
#
# Paper eq. (22):  p¹ = C¹ Λ,  p² = −C² Λ   (one Λ per pair, not 4 tractions)
# Paper §9.3: freeze slip direction ω, drop known complementary contact DOFs.
# =============================================================================

"""Map public 4-traction `x` → 2-traction Λ layout (`nx + 2 np`)."""
function _gnmls_pack(prep, pairs, x4)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    x = zeros(nx + 2 * np)
    x[1:nx] .= view(x4, 1:nx)
    @inbounds for k in 1:np
        x[nx + 2(k - 1) + 1] = x4[nx + 4(k - 1) + 1]
        x[nx + 2(k - 1) + 2] = x4[nx + 4(k - 1) + 2]
    end
    return x
end

"""Slave Λ → body-2 local traction via paper Newton III: ``t² = −R^{-1} t¹``."""
function _gnmls_t2(prep, cp, tn, tt)
    R1 = node_rotation2d(prep[cp.reg_a].dad.Normal[cp.node_a])
    R2 = node_rotation2d(prep[cp.reg_b].dad.Normal[cp.node_b])
    R = R2 * R1'
    t2 = -(R' * SVector(tn, tt))   # R rotation ⇒ R^{-1}=R'
    return t2, R
end

"""Unpack Λ layout to public 4-traction `x4` (hard action–reaction)."""
function _gnmls_unpack!(x4, x, prep, pairs)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    x4[1:nx] .= view(x, 1:nx)
    @inbounds for k in 1:np
        tn = x[nx + 2(k - 1) + 1]
        tt = x[nx + 2(k - 1) + 2]
        t2, _ = _gnmls_t2(prep, pairs[k], tn, tt)
        ot = nx + 4(k - 1)
        x4[ot + 1] = tn
        x4[ot + 2] = tt
        x4[ot + 3] = t2[1]
        x4[ot + 4] = t2[2]
        pairs[k].tn = tn
        pairs[k].tt = tt
    end
    return x4
end

"""BIE residual/Jacobian contribution of Λ (paper A_p C)."""
function _gnmls_apply_G!(R, J, prep, pairs, x, nx; fillJ::Bool)
    @inbounds for (k, cp) in enumerate(pairs)
        ot = nx + 2(k - 1)
        tn, tt = x[ot + 1], x[ot + 2]
        t1 = SVector(tn, tt)
        t2, Rmat = _gnmls_t2(prep, cp, tn, tt)
        pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
        if haskey(pr1.Gc_cols, cp.node_a)
            G1 = pr1.G_local[:, pr1.Gc_cols[cp.node_a]]
            r1 = pr1.off+1:pr1.off+pr1.ndof
            R[r1] .-= G1 * t1
            fillJ && (J[r1, ot+1:ot+2] .-= G1)
        end
        if haskey(pr2.Gc_cols, cp.node_b)
            G2 = pr2.G_local[:, pr2.Gc_cols[cp.node_b]]
            r2 = pr2.off+1:pr2.off+pr2.ndof
            R[r2] .-= G2 * t2
            if fillJ
                # t2 = −R' t1  ⇒  ∂(−G2 t2)/∂t1 = G2 R'
                J[r2, ot+1:ot+2] .+= G2 * Rmat'
            end
        end
    end
    return nothing
end

"""
Frozen-state linear system of Comput. Struct. 2010 eqs. (32)–(33) on
``(u, Λ)`` with ``p¹=Λ``, ``p²=−R^{-1}Λ``.

`force_regime` overrides the augmented-traction classification (used to
kick-start load-controlled problems: all-stick, Λ=0 is otherwise open and
the Neumann pad is singular).
"""
function _assemble_gnmls_Ab(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0,
        force_regime::Union{Nothing,Symbol}=nothing)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 2 * np
    A = zeros(N, N)
    b = zeros(N)
    regimes = Vector{Symbol}(undef, np)
    ωts = zeros(np)

    for pr in prep
        o, nd = pr.off, pr.ndof
        A[o+1:o+nd, o+1:o+nd] .= pr.A
        b[o+1:o+nd] .= pr.b
    end
    Rtmp = zeros(N)
    _gnmls_apply_G!(Rtmp, A, prep, pairs, x, nx; fillJ=true)

    x4 = zeros(nx + 4 * np)
    _gnmls_unpack!(x4, x, prep, pairs)
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x4, k, nx; ht_k=ht[k])
        if force_regime === nothing
            regime, _, _, ωt = augmented_contact_state(
                kin.tn1, kin.tt1, kin.gn, kin.gt, cp.μ, rn, rt)
        else
            regime, ωt = force_regime, 0.0
        end
        regimes[k] = regime
        ωts[k] = ωt
        ot = nx + 2(k - 1)
        r1, r2 = ot + 1, ot + 2
        iu1, iu2, Rm = kin.iu1, kin.iu2, kin.R
        if regime === :open
            A[r1, ot+1] = 1.0
            A[r2, ot+2] = 1.0
            cp.state = 1
        elseif regime === :stick
            # gn = 0 ⇒ −un1 + R[1,:]·u2 = −h   (A z − b = gn)
            A[r1, iu1] = -1.0
            A[r1, iu2] = Rm[1, 1]
            A[r1, iu2+1] = Rm[1, 2]
            b[r1] = -h[k]
            A[r2, iu1+1] = 1.0
            A[r2, iu2] = -Rm[2, 1]
            A[r2, iu2+1] = -Rm[2, 2]
            b[r2] = cp.ut_lock + ht[k]
            cp.state = 3
        else
            A[r1, iu1] = -1.0
            A[r1, iu2] = Rm[1, 1]
            A[r1, iu2+1] = Rm[1, 2]
            b[r1] = -h[k]
            # tt + μ ω tn = 0   (tn ≤ 0 ⇒ tt = μ |tn| ω)
            A[r2, ot+2] = 1.0
            A[r2, ot+1] = cp.μ * ωt
            cp.state = Int(2 * (ωt == 0 ? 1.0 : ωt))
        end
        cp.tn = kin.tn1
        cp.tt = kin.tt1
    end
    return A, b, regimes, ωts
end

"""Paper P / P_g residual on the Λ layout."""
function _assemble_gnmls_R(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 2 * np
    R = zeros(N)
    for pr in prep
        o, nd = pr.off, pr.ndof
        mul!(@view(R[o+1:o+nd]), pr.A, @view x[o+1:o+nd])
        R[o+1:o+nd] .-= pr.b
    end
    _gnmls_apply_G!(R, R, prep, pairs, x, nx; fillJ=false)

    x4 = zeros(nx + 4 * np)
    _gnmls_unpack!(x4, x, prep, pairs)
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x4, k, nx; ht_k=ht[k])
        ot = nx + 2(k - 1)
        regime, _, _, ωt = augmented_contact_state(
            kin.tn1, kin.tt1, kin.gn, kin.gt, cp.μ, rn, rt)
        if regime === :open
            R[ot+1] = kin.tn1
            R[ot+2] = kin.tt1
            cp.state = 1
        elseif regime === :stick
            R[ot+1] = kin.gn
            R[ot+2] = kin.gt
            cp.state = 3
        else
            R[ot+1] = kin.gn
            R[ot+2] = kin.tt1 - cp.μ * abs(kin.tn1) * ωt
            cp.state = Int(2 * (ωt == 0 ? 1.0 : ωt))
        end
        cp.tn = kin.tn1
        cp.tt = kin.tt1
    end
    return R
end

function _gnmls_reduce(A, b, prep, pairs, regimes, ωts)
    nx = sum(p.ndof for p in prep)
    N = size(A, 1)
    fixed = falses(N)
    for (k, cp) in enumerate(pairs)
        ot = nx + 2(k - 1)
        if regimes[k] === :open
            fixed[ot+1] = true
            fixed[ot+2] = true
        elseif regimes[k] === :slip
            fixed[ot+2] = true
        end
    end
    A2 = copy(A)
    for (k, cp) in enumerate(pairs)
        regimes[k] === :slip || continue
        ot = nx + 2(k - 1)
        # tt = −μ ω tn  ⇒  fold
        A2[:, ot+1] .-= (cp.μ * ωts[k]) .* A2[:, ot+2]
    end
    free = findall(!, fixed)
    return A2[free, free], b[free], free, fixed
end

function _gnmls_scatter_reduced!(x, y, free, fixed, prep, pairs, regimes, ωts)
    fill!(x, 0.0)
    x[free] .= y
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        ot = nx + 2(k - 1)
        if regimes[k] === :open
            x[ot+1] = 0.0
            x[ot+2] = 0.0
        elseif regimes[k] === :slip
            x[ot+2] = -(cp.μ * ωts[k]) * x[ot+1]
        end
    end
    return x
end

function _gnmls_frozen_point(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0,
        force_regime::Union{Nothing,Symbol}=nothing)
    A, bvec, regimes, ωts = _assemble_gnmls_Ab(prep, pairs, h, x; ht=ht, rn=rn, rt=rt,
        force_regime=force_regime)
    A_red, b_red, free, fixed = _gnmls_reduce(A, bvec, prep, pairs, regimes, ωts)
    x_tilde = similar(x)
    if isempty(free)
        x_tilde .= x
        return x_tilde, regimes, ωts, false
    end
    y = try
        A_red \ b_red
    catch
        (A_red + 1e-12 * I) \ b_red
    end
    _gnmls_scatter_reduced!(x_tilde, y, free, fixed, prep, pairs, regimes, ωts)
    return x_tilde, regimes, ωts, true
end

"""
    _contact_gnmls!(prep, pairs, h, x_init; kwargs...) -> (x4, ok)

GNM with line search of Rodríguez-Tembleque & Abascal, *Comput. Struct.* **88**
(2010). BEM–BEM form (eq. 22): one contact traction ``Λ`` per pair, hard
action–reaction, frozen slip direction, quasi-complementarity reduction
(§9.3, §9.8), Pang line search (§9.1).

Public `x` is the Contato 4-traction layout (for scatter); internally ``Λ``
has 2 components per pair.
"""
function _contact_gnmls!(prep, pairs, h, x_init;
        ht=nothing, tol=1e-8, maxiter=40, verbose=false,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        ls_max::Int=10,
        β_ls::Float64=0.5)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    x4 = collect(Float64, x_init)
    length(x4) == nx + 4 * np || (x4 = zeros(nx + 4 * np))
    x = _gnmls_pack(prep, pairs, x4)
    rn0, rt0 = _default_contact_r(prep)
    rn_ = rn === nothing ? rn0 : float(rn)
    rt_ = rt === nothing ? rt0 : float(rt)
    ok = false
    nR = Inf

    # Load-control kick-start: Λ=0 classifies every pair as open (g_n=g₀>0)
    # and a Neumann-loaded pad is singular. One all-stick frozen solve is the
    # same first iteration Contato uses on Loyola §9.3.1/§9.3.2.
    if np > 0 && maximum(abs, view(x, nx+1:length(x))) < 1e-14
        x_stick, _, _, solved = _gnmls_frozen_point(prep, pairs, h, x;
            ht=ht, rn=rn_, rt=rt_, force_regime=:stick)
        if solved
            x .= x_stick
            verbose && @info "contact GNM-ls (2010) stick kick-start"
        end
    end

    for it in 1:maxiter
        R0 = _assemble_gnmls_R(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
        nR = norm(R0)
        verbose && @info "contact GNM-ls (2010)" it nR n_closed=count(cp -> abs(cp.state) != 1, pairs)
        if nR < tol
            ok = true
            break
        end
        x_tilde, _, _, solved = _gnmls_frozen_point(prep, pairs, h, x;
            ht=ht, rn=rn_, rt=rt_)
        if !solved
            verbose && @warn "GNM-ls: no free DOFs"
            break
        end

        ψ0 = 0.5 * nR^2
        Δ = x_tilde .- x
        α = 1.0
        x_trial = similar(x)
        nR_new = nR
        for _ in 1:ls_max
            x_trial .= x .+ α .* Δ
            nR_new = norm(_assemble_gnmls_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_))
            ψ = 0.5 * nR_new^2
            (ψ <= (1.0 - 2 * β_ls * α) * ψ0 || nR_new < tol) && break
            α *= 0.5
        end
        x .= x_trial
        if nR_new < tol
            ok = true
            break
        end
        if α * norm(Δ) < tol * max(1.0, norm(x))
            ok = nR_new < max(tol, 1e-6 * max(1.0, nR))
            break
        end
    end
    nR = norm(_assemble_gnmls_R(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_))
    ok = ok || nR < tol
    verbose && @info "contact GNM-ls (2010) done" ok nR
    _gnmls_unpack!(x4, x, prep, pairs)
    return x4, ok
end


