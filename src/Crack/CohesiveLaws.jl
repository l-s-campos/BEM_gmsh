# =============================================================================
# Cohesive-zone constitutive models for DBEM
# =============================================================================
# References
# - Cordeiro et al., Theor. Appl. Fract. Mech. 130 (2024) 104249  (PPR + states)
# - Alfano & Sacco, Int. J. Numer. Meth. Engng 68 (2006) 542–582 (damage+friction)
# - Távara et al., Comput. Mech. 51 (2013) 535–551 (CZM + BEM)
#
# Local frame: δ = (δn, δt), tn ≥ 0 is tension (opening positive), compression
# handled by the contact branch of the surface-condition machine.

export CohesiveState, STATE_CONTACT, STATE_SOFTENING, STATE_UNLOAD, STATE_FAILED
export AbstractCohesiveLaw, BilinearCZM, PPRLaw, AlfanoSaccoLaw
export local_to_global_R, opening_local, traction_and_stiffness
export evaluate_surface!, CohesiveHistory

# ---------------------------------------------------------------------------
# Surface condition codes (Cordeiro 2024, Algorithm 1)
# ---------------------------------------------------------------------------

@enum CohesiveState begin
    STATE_CONTACT = 1
    STATE_SOFTENING = 2
    STATE_UNLOAD = 3
    STATE_FAILED = 4
end

"""Per-collocation history variables for unload/reload and damage."""
mutable struct CohesiveHistory
    δn_max::Float64          # max normal opening in history
    δt_max::Float64          # max |tangential| opening in history
    D::Float64               # damage (Alfano–Sacco), ∈ [0,1]
    state::CohesiveState
end
CohesiveHistory() = CohesiveHistory(0.0, 0.0, 0.0, STATE_SOFTENING)

# ---------------------------------------------------------------------------
# Local ↔ global
# ---------------------------------------------------------------------------

"""
    local_to_global_R(n̂) -> R (2×2)

Rotation from local (n, t) to global (x, y). Columns are ê_n, ê_t with
ê_t = (-n_y, n_x) (CCW 90° from outward normal of face +).
"""
function local_to_global_R(n̂::SVector{2, Float64})
    n = n̂ / (norm(n̂) + eps())
    t = SVector(-n[2], n[1])
    return SMatrix{2, 2, Float64}(n[1], n[2], t[1], t[2])  # columns n, t
end

"""Local openings δ = Rᵀ (u⁺ − u⁻)."""
function opening_local(R::SMatrix{2, 2, Float64}, Δu_global::SVector{2, Float64})
    return R' * Δu_global
end

# ---------------------------------------------------------------------------
# Law interface
# ---------------------------------------------------------------------------

abstract type AbstractCohesiveLaw end

"""
    traction_and_stiffness(law, δn, δt, hist; kn_pen) -> (tn, tt, kn, kt, state)

Evaluate local cohesive tractions and **tangent** stiffnesses for the current
openings and history. Contact uses penalty `kn_pen` when δn ≤ 0.
"""
function traction_and_stiffness end

# ---------------------------------------------------------------------------
# Bilinear mixed-mode CZM (robust default)
# ---------------------------------------------------------------------------

"""
    BilinearCZM(; σn, σt, Gn, Gt, η=1e-3)

Bilinear mixed-mode cohesive law with linear softening and linear unload/reload
to the origin through the historic peak (Cordeiro eqs. 10–11 with α=β=1).

- `σn, σt` — peak strengths
- `Gn, Gt` — fracture energies (area under t–δ curves)
- Critical openings: δn0 = 2 Gn / σn? Wait standard bilinear:
  δnc = σn * δnf / (σn) ... 
  Initial stiffness kn0 = σn / δn0 with δn0 small elastic limit.
  Final opening δnf = 2 Gn / σn for triangular bilinear.
"""
Base.@kwdef struct BilinearCZM <: AbstractCohesiveLaw
    σn::Float64 = 4e6
    σt::Float64 = 3e6
    Gn::Float64 = 100.0
    Gt::Float64 = 200.0
    δn0::Float64 = 1e-6          # elastic limit (normal)
    δt0::Float64 = 1e-6
    η_fail::Float64 = 1e-3       # residual stiffness factor at complete failure
    μ::Float64 = 0.0             # Coulomb friction in contact (0 = frictionless)
    kt_contact::Float64 = 0.0    # tangential penalty in contact (0 → use σt/δt0)
end

function _final_openings(law::BilinearCZM)
    # triangular bilinear: Gn = ½ σn δnf  ⇒  δnf = 2 Gn / σn
    δnf = 2 * law.Gn / max(law.σn, eps())
    δtf = 2 * law.Gt / max(law.σt, eps())
    return max(δnf, 2 * law.δn0), max(δtf, 2 * law.δt0)
end

function traction_and_stiffness(
        law::BilinearCZM,
        δn::Float64,
        δt::Float64,
        hist::CohesiveHistory;
        kn_pen::Float64 = 1e12,
    )
    δnf, δtf = _final_openings(law)
    δn0, δt0 = law.δn0, law.δt0
    kn0 = law.σn / δn0
    kt0 = law.σt / δt0
    abs_t = abs(δt)
    s_t = δt >= 0 ? 1.0 : -1.0

    # --- contact (unilateral + optional Coulomb) ---
    if δn <= 0
        return _contact_response(δn, δt, kn_pen, law.μ,
            law.kt_contact > 0 ? law.kt_contact : kt0)
    end

    # --- complete failure ---
    if δn >= δnf || abs_t >= δtf
        η = law.η_fail
        # failed + re-contact handled above; here pure separation failure
        return 0.0, 0.0, η * kn0, η * kt0, STATE_FAILED
    end

    # update candidate peaks
    δn_peak = max(hist.δn_max, δn)
    δt_peak = max(hist.δt_max, abs_t)

    # --- unloading/reloading if below historic peak ---
    unloading = (δn < hist.δn_max - 1e-16) || (abs_t < hist.δt_max - 1e-16)
    if unloading && (hist.δn_max > δn0 || hist.δt_max > δt0)
        # linear unload through peak: t = t_peak(δ_peak) * (δ / δ_peak)
        tn_p, kn_p = _bilinear_soft_branch(δn_peak, δn0, δnf, law.σn, kn0)
        tt_p, kt_p = _bilinear_soft_branch(δt_peak, δt0, δtf, law.σt, kt0)
        tn = δn_peak > 0 ? tn_p * (δn / δn_peak) : 0.0
        tt = δt_peak > 0 ? s_t * tt_p * (abs_t / δt_peak) : 0.0
        kn = δn_peak > 0 ? tn_p / δn_peak : kn0
        kt = δt_peak > 0 ? tt_p / δt_peak : kt0
        return tn, tt, kn, kt, STATE_UNLOAD
    end

    # --- softening / elastic ---
    tn, kn = _bilinear_soft_branch(δn, δn0, δnf, law.σn, kn0)
    tt_mag, kt = _bilinear_soft_branch(abs_t, δt0, δtf, law.σt, kt0)
    tt = s_t * tt_mag
    return tn, tt, kn, kt, STATE_SOFTENING
end

function _bilinear_soft_branch(δ, δ0, δf, σ, k0)
    if δ <= δ0
        return k0 * δ, k0
    elseif δ < δf
        # linear soft from (δ0,σ) to (δf,0)
        t = σ * (δf - δ) / (δf - δ0)
        k = -σ / (δf - δ0)   # tangent (negative)
        return t, k
    else
        return 0.0, 0.0
    end
end

# ---------------------------------------------------------------------------
# Park–Paulino–Roesler (PPR) potential-based model (Cordeiro / Park & Paulino)
# ---------------------------------------------------------------------------

"""
    PPRLaw(; Γn, Γt, σn, σt, α=5.0, β=1.6, λn=0.005, λt=0.005)

Park–Paulino–Roesler mixed-mode cohesive potential [Park & Paulino 2012],
as used in Cordeiro et al. (2024). Parameters:
- `Γn, Γt` — fracture energies
- `σn, σt` — cohesive strengths  
- `α, β` — shape parameters (>2 convex softening in mode I if α>2)
- `λn, λt` — ratios of critical to final openings (δnc/δnf)
"""
Base.@kwdef struct PPRLaw <: AbstractCohesiveLaw
    Γn::Float64 = 100.0
    Γt::Float64 = 200.0
    σn::Float64 = 4e6
    σt::Float64 = 3e6
    α::Float64 = 5.0
    β::Float64 = 1.6
    λn::Float64 = 0.005
    λt::Float64 = 0.005
    η_fail::Float64 = 1e-3
    μ::Float64 = 0.0
    kt_contact::Float64 = 0.0
end

"""Final and critical openings from PPR identities (triangular-like limits)."""
function _ppr_openings(law::PPRLaw)
    # δnf from energy/strength with shape correction (Gain/Park approximate)
    # For α→2, δnf → 2 Γn/σn; general: use Park formula
    α, β = law.α, law.β
    δnf = (α / (α - 1.0)) * (2 * law.Γn / law.σn) * (1 - law.λn)^(α - 1) *
          (α / 2 + law.λn * (1 - α / 2))   # simplified robust form
    # fallback if pathological
    δnf = max(δnf, 2 * law.Γn / law.σn)
    δtf = max((β / max(β - 1, 0.5)) * (2 * law.Γt / law.σt), 2 * law.Γt / law.σt)
    δnc = law.λn * δnf
    δtc = law.λt * δtf
    return δnc, δtc, δnf, δtf
end

function _ppr_softening_tn(law::PPRLaw, δn, δt, δnc, δtc, δnf, δtf)
    # Separable approximation of PPR gradient (mode-mix via energy weights)
    # tn(δn) with shape α; reduced by tangential damage
    if δn <= 0
        return 0.0, 0.0
    end
    if δn >= δnf
        return 0.0, 0.0
    end
    α = law.α
    # elastic ramp then power soft
    if δn <= δnc
        kn = law.σn / δnc
        return kn * δn, kn
    end
    ξ = (δnf - δn) / (δnf - δnc)
    tn = law.σn * ξ^(α - 1)
    # d tn / d δn
    kn = law.σn * (α - 1) * ξ^(α - 2) * (-1) / (δnf - δnc)
    # mix reduction
    if abs(δt) > 0 && δtf > 0
        mix = max(0.0, 1 - (abs(δt) / δtf)^2)
        tn *= mix
        kn *= mix
    end
    return tn, kn
end

function _ppr_softening_tt(law::PPRLaw, δn, δt, δnc, δtc, δnf, δtf)
    abs_t = abs(δt)
    s = δt >= 0 ? 1.0 : -1.0
    if abs_t <= 0
        return 0.0, law.σt / max(δtc, eps())
    end
    if abs_t >= δtf
        return 0.0, 0.0
    end
    β = law.β
    if abs_t <= δtc
        kt = law.σt / δtc
        return s * kt * abs_t, kt
    end
    η = (δtf - abs_t) / (δtf - δtc)
    tt = law.σt * η^(β - 1)
    kt = law.σt * (β - 1) * η^(β - 2) * (-1) / (δtf - δtc)
    if δn > 0 && δnf > 0
        mix = max(0.0, 1 - (δn / δnf)^2)
        tt *= mix
        kt *= mix
    end
    return s * tt, kt
end

function traction_and_stiffness(
        law::PPRLaw,
        δn::Float64,
        δt::Float64,
        hist::CohesiveHistory;
        kn_pen::Float64 = 1e12,
    )
    δnc, δtc, δnf, δtf = _ppr_openings(law)
    abs_t = abs(δt)
    kn0 = law.σn / max(δnc, eps())
    kt0 = law.σt / max(δtc, eps())

    if δn <= 0
        return _contact_response(δn, δt, kn_pen, law.μ,
            law.kt_contact > 0 ? law.kt_contact : kt0)
    end
    if δn >= δnf || abs_t >= δtf
        η = law.η_fail
        return 0.0, 0.0, η * kn0, η * kt0, STATE_FAILED
    end

    unloading = (δn < hist.δn_max - 1e-16) || (abs_t < hist.δt_max - 1e-16)
    if unloading && (hist.δn_max > δnc || hist.δt_max > δtc)
        δn_p = max(hist.δn_max, δnc)
        δt_p = max(hist.δt_max, δtc)
        tn_p, _ = _ppr_softening_tn(law, δn_p, 0.0, δnc, δtc, δnf, δtf)
        tt_p, _ = _ppr_softening_tt(law, 0.0, δt_p, δnc, δtc, δnf, δtf)
        tn = tn_p * (δn / δn_p)
        tt = (δt_p > 0 ? abs(tt_p) * (abs_t / δt_p) : 0.0) * (δt >= 0 ? 1 : -1)
        kn = tn_p / δn_p
        kt = δt_p > 0 ? abs(tt_p) / δt_p : kt0
        return tn, tt, kn, kt, STATE_UNLOAD
    end

    tn, kn = _ppr_softening_tn(law, δn, δt, δnc, δtc, δnf, δtf)
    tt, kt = _ppr_softening_tt(law, δn, δt, δnc, δtc, δnf, δtf)
    return tn, tt, kn, kt, STATE_SOFTENING
end

# ---------------------------------------------------------------------------
# Alfano–Sacco damage + friction on damaged fraction
# ---------------------------------------------------------------------------

"""
    AlfanoSaccoLaw(; kn, kt, σn, σt, δnf, δtf, μ=0.3, η=1e-8)

Damage–friction interface (Alfano & Sacco 2006):
- undamaged fraction `(1-D)`: linear elastic with stiffnesses `kn, kt`
- damaged fraction `D`: Coulomb friction only (no tension)
- damage evolution: Crisfield-like equivalent opening

``t = (1-D) K δ + D t^{fr}``
"""
Base.@kwdef struct AlfanoSaccoLaw <: AbstractCohesiveLaw
    kn::Float64 = 1e12
    kt::Float64 = 1e12
    σn::Float64 = 3e6
    σt::Float64 = 3e6
    δnf::Float64 = 1e-4
    δtf::Float64 = 1e-4
    μ::Float64 = 0.3
    η::Float64 = 1e-8
end

function _as_damage(law::AlfanoSaccoLaw, δn, δt, D_old)
    # equivalent opening (mode-mix)
    δeq = sqrt(max(δn, 0.0)^2 + (law.σn / max(law.σt, eps()) * δt)^2)
    δ0 = law.σn / law.kn
    δf = law.δnf
    if δeq <= δ0
        D = D_old
    elseif δeq >= δf
        D = 1.0
    else
        # Crisfield damage
        Dstar = (δf / δeq) * (δeq - δ0) / (δf - δ0)
        D = max(D_old, min(1.0, Dstar))
    end
    return D
end

function traction_and_stiffness(
        law::AlfanoSaccoLaw,
        δn::Float64,
        δt::Float64,
        hist::CohesiveHistory;
        kn_pen::Float64 = 1e12,
    )
    D = _as_damage(law, δn, δt, hist.D)
    # elastic undamaged response
    tn_e = law.kn * δn
    tt_e = law.kt * δt

    # friction on damaged part (only if compressed or sliding)
    tn_fr = 0.0
    tt_fr = 0.0
    kn_fr = 0.0
    kt_fr = 0.0
    if δn < 0
        tn_fr = kn_pen * δn
        kn_fr = kn_pen
        # Coulomb on damaged fraction: |tt| ≤ μ |tn|
        tt_trial = law.kt * δt
        τmax = law.μ * abs(tn_fr)
        if abs(tt_trial) <= τmax
            tt_fr = tt_trial
            kt_fr = law.kt
        else
            tt_fr = sign(δt) * τmax
            kt_fr = 0.0   # perfect plastic friction
        end
    end

    tn = (1 - D) * tn_e + D * tn_fr
    tt = (1 - D) * tt_e + D * tt_fr
    kn = (1 - D) * law.kn + D * kn_fr
    kt = (1 - D) * law.kt + D * kt_fr

    state = if δn <= 0
        STATE_CONTACT
    elseif D >= 1 - 1e-12
        STATE_FAILED
    elseif D > hist.D + 1e-16
        STATE_SOFTENING
    else
        STATE_UNLOAD
    end
    # stash D into hist via return path (caller updates)
    hist.D = D
    return tn, tt, max(kn, law.η * law.kn), max(kt, law.η * law.kt), state
end

# ---------------------------------------------------------------------------
# Shared unilateral contact + Coulomb friction
# ---------------------------------------------------------------------------

"""
Penalty contact: ``t_n = k_n^{pen} δ_n`` (δn≤0), Coulomb on shear:
stick if ``|k_t δ_t| ≤ μ |t_n|``, else slip ``t_t = ±μ |t_n|``.
"""
function _contact_response(δn::Float64, δt::Float64, kn_pen::Float64, μ::Float64, kt::Float64)
    tn = kn_pen * δn                 # ≤ 0 in compression
    kn = kn_pen
    tt_trial = kt * δt
    τmax = μ * abs(tn)
    if μ <= 0 || abs(tt_trial) <= τmax + 1e-30
        # stick (or frictionless with kt→0 if μ=0 and we zero kt)
        if μ <= 0
            return tn, 0.0, kn, 0.0, STATE_CONTACT
        end
        return tn, tt_trial, kn, kt, STATE_CONTACT
    end
    # slip
    tt = sign(δt == 0 ? tt_trial : δt) * τmax
    return tn, tt, kn, 0.0, STATE_CONTACT
end

# ---------------------------------------------------------------------------
# Fatigue damage evolution (Cordeiro outlook / Roe–Siegmund style)
# ---------------------------------------------------------------------------

"""
    FatigueCZM(base; C=1e-4, m=2.0, δth=0.0)

Wraps a cohesive law with cycle-by-cycle damage accumulation:
``ΔD = C ⟨δ_eq/δ_f - δth⟩_+^m`` per cycle (envelope-based).
Call [`fatigue_cycle!`](@ref) after each load cycle peak.
"""
Base.@kwdef struct FatigueCZM{L<:AbstractCohesiveLaw} <: AbstractCohesiveLaw
    base::L
    C::Float64 = 1e-4
    m::Float64 = 2.0
    δth::Float64 = 0.0
    δf_ref::Float64 = 1e-4
end

function traction_and_stiffness(
        law::FatigueCZM,
        δn::Float64,
        δt::Float64,
        hist::CohesiveHistory;
        kn_pen::Float64 = 1e12,
    )
    tn, tt, kn, kt, st = traction_and_stiffness(law.base, δn, δt, hist; kn_pen)
    # scale residual capacity by remaining integrity (1-D)
    s = max(1 - hist.D, 1e-6)
    return s * tn, s * tt, s * kn, s * kt, st
end

"""
    fatigue_cycle!(law::FatigueCZM, hist, δn_peak, δt_peak)

Accumulate one cycle of fatigue damage from peak openings.
"""
function fatigue_cycle!(law::FatigueCZM, hist::CohesiveHistory, δn_peak::Float64, δt_peak::Float64)
    δeq = sqrt(max(δn_peak, 0.0)^2 + δt_peak^2)
    ξ = δeq / max(law.δf_ref, eps()) - law.δth
    if ξ > 0
        hist.D = min(1.0, hist.D + law.C * ξ^law.m)
    end
    return hist.D
end

# ---------------------------------------------------------------------------
# History update helper
# ---------------------------------------------------------------------------

"""
    evaluate_surface!(law, δn, δt, hist; kn_pen) -> (tn, tt, kn, kt, state)

Evaluate law and update `hist` peaks / damage / state.
"""
function evaluate_surface!(
        law::AbstractCohesiveLaw,
        δn::Float64,
        δt::Float64,
        hist::CohesiveHistory;
        kn_pen::Float64 = 1e12,
    )
    tn, tt, kn, kt, state = traction_and_stiffness(law, δn, δt, hist; kn_pen)
    # history peaks only grow under tension/opening
    if δn > hist.δn_max
        hist.δn_max = δn
    end
    if abs(δt) > hist.δt_max
        hist.δt_max = abs(δt)
    end
    hist.state = state
    return tn, tt, kn, kt, state
end
