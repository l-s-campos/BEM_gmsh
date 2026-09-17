# =============================================================================
# Block admissibility criteria (geometry only — no FMM / multipole state)
#
# Three documented variants; do NOT merge formulas (different conventions):
#   StrongAdmissibilityStd — classic ℋ (min diam vs distance)
#   FMMStrongAdmissibility — multipole dual-tree (max diam vs distance)
#   WeakAdmissibilityStd   — positive distance only
# =============================================================================

"""
    StrongAdmissibilityStd(; eta=3.0)

ℋ-matrix strong admissibility: `min(diam(A),diam(B)) < eta * dist(A,B)`.
"""
Base.@kwdef struct StrongAdmissibilityStd
    eta::Float64 = 3.0
end

function (adm::StrongAdmissibilityStd)(left_node, right_node)
    diam_min = minimum(diameter, (left_node, right_node))
    dist = distance(left_node, right_node)
    return diam_min < adm.eta * dist
end

"""
    FMMStrongAdmissibility(; η=1.0)

FMM dual-tree strong admissibility: `dist(A,B) > 0` and
`dist(A,B) ≥ η * max(diam(A),diam(B))`.

Note the **max** diameter (and `≥`) — different from [`StrongAdmissibilityStd`].
Zero-diameter (singleton) boxes are not admissible at distance 0.
"""
Base.@kwdef struct FMMStrongAdmissibility
    η::Float64 = 1.0
end

function (adm::FMMStrongAdmissibility)(a, b)
    dist = distance(a, b)
    # `diam == 0` leaves (singletons) would satisfy `0 ≥ 0` and M2L a box
    # with itself / nested clusters. Require a positive gap.
    dist > 0 || return false
    return dist >= adm.η * max(diameter(a), diameter(b))
end

"""
    WeakAdmissibilityStd()

Admissible iff `distance(A,B) > 0`.
"""
struct WeakAdmissibilityStd end

(adm::WeakAdmissibilityStd)(left_node, right_node) = distance(left_node, right_node) > 0
