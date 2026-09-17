# =============================================================================
# Semi-smooth Newton (Alart–Curnier) on Contato unknowns
# =============================================================================

function alart_curnier(gn::Real, gt::Real, tn::Real, tt::Real, μ::Real,
        rn::Real, rt::Real)
    λn = -float(tn)
    λt = -float(tt)
    τn = λn - rn * gn
    τt = λt - rt * gt
    λn⁺ = max(0.0, τn)
    if τn <= 0.0
        Cn = λn
        Ct = λt
        return Cn, Ct, :open, 0.0, τn, τt, λn⁺
    end
    bound = μ * λn⁺
    if abs(τt) <= bound + 1e-15
        # stick
        Cn = rn * gn          # = λn - τn
        Ct = rt * gt          # = λt - τt
        return Cn, Ct, :stick, 0.0, τn, τt, λn⁺
    end
    # slip
    s = τt == 0.0 ? 1.0 : sign(τt)
    λt_hat = s * bound
    Cn = rn * gn
    Ct = λt - λt_hat         # = λt - s μ (λn - rn gn)
    return Cn, Ct, :slip, s, τn, τt, λn⁺
end
"""
Assemble SSN residual ``R`` and generalized Jacobian ``J`` on Contato layout.

Unknowns: mixed BIE DOFs then per pair ``(t_n¹,t_t¹,t_n²,t_t²)``.
Contact rows: ``(C_n, C_t, E_n, E_t)`` with Alart–Curnier + traction equilibrium.
"""
function _assemble_contact_R_J(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 4 * np
    R = zeros(N)
    J = zeros(N, N)

    # --- BIE blocks ---
    for pr in prep
        o = pr.off
        nd = pr.ndof
        xr = @view x[o+1:o+nd]
        J[o+1:o+nd, o+1:o+nd] .= pr.A
        mul!(@view(R[o+1:o+nd]), pr.A, xr)
        R[o+1:o+nd] .-= pr.b
    end
    for (k, cp) in enumerate(pairs)
        pr1 = prep[cp.reg_a]
        pr2 = prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)
        t1 = SVector(x[ot+1], x[ot+2])
        t2 = SVector(x[ot+3], x[ot+4])
        if haskey(pr1.Gc_cols, na)
            cols = pr1.Gc_cols[na]
            G1 = pr1.G_local[:, cols]
            R[pr1.off+1:pr1.off+pr1.ndof] .-= G1 * t1
            J[pr1.off+1:pr1.off+pr1.ndof, ot+1:ot+2] .-= G1
        end
        if haskey(pr2.Gc_cols, nb)
            cols = pr2.Gc_cols[nb]
            G2 = pr2.G_local[:, cols]
            R[pr2.off+1:pr2.off+pr2.ndof] .-= G2 * t2
            J[pr2.off+1:pr2.off+pr2.ndof, ot+3:ot+4] .-= G2
        end
    end

    # --- contact NCF + equilibrium ---
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx; ht_k=ht[k])
        Rmat = kin.R
        ot, iu1, iu2 = kin.ot, kin.iu1, kin.iu2
        tn1, tt1, tn2, tt2 = kin.tn1, kin.tt1, kin.tn2, kin.tt2
        Cn, Ct, regime, s, _, _, _ = alart_curnier(kin.gn, kin.gt, tn1, tt1, cp.μ, rn, rt)

        r1, r2, r3, r4 = ot + 1, ot + 2, ot + 3, ot + 4
        R[r1] = Cn
        R[r2] = Ct
        R[r3] = tn1 + Rmat[1, 1] * tn2 + Rmat[1, 2] * tt2
        R[r4] = tt1 + Rmat[2, 1] * tn2 + Rmat[2, 2] * tt2

        # equilibrium Jacobian (always)
        J[r3, ot+1] = 1.0
        J[r3, ot+3] = Rmat[1, 1]
        J[r3, ot+4] = Rmat[1, 2]
        J[r4, ot+2] = 1.0
        J[r4, ot+3] = Rmat[2, 1]
        J[r4, ot+4] = Rmat[2, 2]

        μ = cp.μ
        if regime === :open
            # Cn = λn = -tn1, Ct = λt = -tt1
            J[r1, ot+1] = -1.0
            J[r2, ot+2] = -1.0
        elseif regime === :stick
            # Cn = rn gn, Ct = rt gt
            _add_gn_row!(J, r1, iu1, iu2, Rmat, rn)
            _add_gt_row!(J, r2, iu1, iu2, Rmat, rt)
        else
            # slip: Cn = rn gn
            # Ct = -tt + s μ tn + s μ rn gn   (λ form chained through λ=-t)
            _add_gn_row!(J, r1, iu1, iu2, Rmat, rn)
            J[r2, ot+2] = -1.0
            J[r2, ot+1] = s * μ
            _add_gn_row!(J, r2, iu1, iu2, Rmat, s * μ * rn)
        end

        # Contato state tags for diagnostics (verify overwrites later)
        if regime === :open
            cp.state = 1
        elseif regime === :stick
            cp.state = 3
        else
            # slip sign from traction (Contato: ±2)
            cp.state = Int((tt1 == 0.0 ? s : sign(tt1)) * 2)
            cp.state == 0 && (cp.state = Int(s * 2))
        end
        cp.tn = tn1
        cp.tt = tt1
    end
    return R, J
end

"""Residual-only evaluation (for line search)."""
function _assemble_contact_R(prep, pairs, h, x; ht=nothing, rn::Real=1.0, rt::Real=1.0)
    R, _ = _assemble_contact_R_J(prep, pairs, h, x; ht=ht, rn=rn, rt=rt)
    return R
end

"""
Semi-smooth Newton on Alart–Curnier contact residual.

```text
for it
    R, J ← assemble_R_J(x)
    solve J Δx = -R
    line-search α on ‖R‖
    x ← x + α Δx
```
"""
function _contact_ssn!(prep, pairs, h, x_init;
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
        R, J = _assemble_contact_R_J(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
        nR = norm(R)
        verbose && @info "contact SSN" it nR rn=rn_ n_closed=count(cp -> abs(cp.state) != 1, pairs)
        if nR < tol
            ok = true
            break
        end
        dx = J \ (-R)
        # Armijo-like backtracking on ‖R‖
        α = 1.0
        nR_new = nR
        x_trial = similar(x)
        accepted = false
        for _ls in 1:ls_max
            x_trial .= x .+ α .* dx
            R_try = _assemble_contact_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_)
            nR_new = norm(R_try)
            if nR_new < (1.0 - 1e-4 * α) * nR || nR_new < tol
                accepted = true
                break
            end
            α *= 0.5
        end
        if !accepted
            # take smallest trial anyway (damped progress)
            x_trial .= x .+ α .* dx
            nR_new = norm(_assemble_contact_R(prep, pairs, h, x_trial; ht=ht, rn=rn_, rt=rt_))
        end
        x .= x_trial
        if nR_new < tol
            ok = true
            break
        end
        # also stop on tiny step
        if α * norm(dx) < tol * max(1.0, norm(x))
            ok = nR_new < max(tol, 1e-6 * max(1.0, nR))
            break
        end
    end
    # final residual / state tags
    R, _ = _assemble_contact_R_J(prep, pairs, h, x; ht=ht, rn=rn_, rt=rt_)
    nR = norm(R)
    ok = ok || nR < tol
    verbose && @info "contact SSN done" ok nR
    return x, ok
end
