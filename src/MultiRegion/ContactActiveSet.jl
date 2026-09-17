# Contato active-set frictional contact (frozen open/stick/±slip)

"""
Contato MATLAB active-set (`newton_*_multicorpos`):

```text
x = 0
for it = 1:maxiter
    state ← verify(x)                 # first iterate included (not all-stick)
    A, b  ← assemble(state, h)
    x_new ← A \\ b
    stop if ‖x_new - x‖ small and no state flips
    x ← x_new
end
```

Do **not** skip the first verify or force every pair to stick — that is the
path Contato uses from `x0=0` (`verfica_contato_*` then `aplica_contato_*`).
"""
function _contact_activeset!(prep, pairs, h, x_init;
        ht=nothing, tol=1e-8, maxiter=40, verbose=false, epsc=1e-7,
        nsteps::Int=1)
    nsteps > 1 && return _contact_contato_incremental!(prep, pairs, h, x_init;
        ht=ht, tol=tol, maxiter=maxiter, verbose=verbose, epsc=epsc, nsteps=nsteps)
    x = collect(Float64, x_init)
    ok = false
    prev_states = fill(0, length(pairs))
    stable = 0
    for it in 1:maxiter
        _verify_contact_states!(pairs, prep, h, x; ht=ht, epsc=epsc)
        states = [cp.state for cp in pairs]
        A, b = _assemble_contact_system(prep, pairs, h, x; ht=ht)
        x_new = try
            A \ b
        catch
            (A + 1e-12 * I) \ b
        end
        dist = norm(x_new - x) / max(1.0, norm(x_new))
        n_flip = count(i -> states[i] != prev_states[i], eachindex(states))
        verbose && @info "contact active-set" it dist n_flip n_closed=count(s -> abs(s) != 1, states)
        x .= x_new
        prev_states .= states
        if dist < tol && n_flip == 0
            ok = true
            break
        elseif n_flip == 0
            stable += 1
            if stable >= 3 && dist < max(tol * 100, 1e-5)
                ok = true
                break
            end
        else
            stable = 0
        end
    end
    _verify_contact_states!(pairs, prep, h, x; ht=ht, epsc=epsc)
    return x, ok
end

"""Remaining geometric gap after accumulated local displacement (Contato `h-deltaun`)."""
function _contato_deltaun(prep, cp, xtot, nx)
    pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
    iu1 = pr1.off + (2cp.node_a - 1)
    iu2 = pr2.off + (2cp.node_b - 1)
    R1 = node_rotation2d(pr1.dad.Normal[cp.node_a])
    R2 = node_rotation2d(pr2.dad.Normal[cp.node_b])
    R = R2 * R1'
    un1, ut1 = xtot[iu1], xtot[iu1 + 1]
    un2, ut2 = xtot[iu2], xtot[iu2 + 1]
    return un1 - (R[1, 1] * un2 + R[1, 2] * ut2)
end

"""
Contato `newton_incremental_multicorpos`: `nsteps` load increments, each
Newton on **Δx** with frozen previous totals (`desl`,`trac`). BIE rhs is
`b/nsteps`. Gap row: ``Δu_n = h - (u_n^{old})``. Slip Coulomb on totals.
"""
function _contact_contato_incremental!(prep, pairs, h, x_init;
        ht=nothing, tol=1e-8, maxiter=40, verbose=false, epsc=1e-7,
        nsteps::Int=50)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 4 * np
    xtot = zeros(N)
    x = zeros(N)                    # increment; warm-started like MATLAB x0
    ok = true
    for s in 1:nsteps
        ok_s = false
        prev = fill(0, np)
        for it in 1:maxiter
            x_tot = xtot .+ x
            _verify_contact_states!(pairs, prep, h, x_tot; ht=ht, epsc=epsc)
            st = [cp.state for cp in pairs]
            A, b = _assemble_contact_system(prep, pairs, h, x_tot; ht=ht)
            b[1:nx] ./= nsteps
            for (k, cp) in enumerate(pairs)
                abs(cp.state) == 1 && continue
                ot = nx + 4(k - 1)
                b[ot + 1] = h[k] - _contato_deltaun(prep, cp, xtot, nx)
                if abs(cp.state) == 2
                    sμ = cp.μ * (cp.state >= 0 ? 1.0 : -1.0)
                    b[ot + 2] = -sμ * xtot[ot + 1] - xtot[ot + 2]
                end
            end
            x_new = try
                A \ b
            catch
                (A + 1e-12 * I) \ b
            end
            dist = norm(x_new - x)
            n_flip = count(i -> st[i] != prev[i], eachindex(st))
            verbose && it <= 3 && @info "contato inc" s it dist n_flip n_closed=count(s0 -> abs(s0) != 1, st)
            x .= x_new
            prev .= st
            if dist < tol && n_flip == 0
                ok_s = true
                break
            end
        end
        xtot .+= x
        ok &= ok_s
        verbose && @info "contato load step" s n_closed=count(cp -> abs(cp.state) != 1, pairs) ok=ok_s
    end
    _verify_contact_states!(pairs, prep, h, xtot; ht=ht, epsc=epsc)
    return xtot, ok
end
