# Plate buckling & thermal buckling (placa_flambagem / placa_termoflamba)
#
# Geometric stiffness from membrane forces N leads to generalized eigenproblem
#   H_b w = λ M_g(N) w
# Critical load factor λ_cr from the smallest positive eigenvalue.

export geometric_stiffness_plate, plate_buckling, thermal_buckling
export analytical_Ncr_ss_uniaxial

"""
    geometric_stiffness_plate(prob, Nxx, Nyy, Nxy) -> Mg

Build geometric stiffness matrix on free DOFs of a [`LargePlateProblem`](@ref)
or plate operators `(A_pl, is_kin, Fx, Fy, w_index, pts)`.

Uses RBF second derivatives: ``M_g ∼ -∫ (N_{αβ} w_{,α} δw_{,β})``
discretized as free-DOF matrix from `qg = N:∇∇φ` operator linearized.
"""
function geometric_stiffness_plate(A_pl::AbstractMatrix, is_kin::AbstractVector{Bool},
    Fx, Fy, w_index::AbstractVector{Int}, Nxx, Nyy, Nxy)
    ndof = size(A_pl, 1)
    free = findall(!, is_kin)
    nfree = length(free)
    # Build operator: for unit free w modes, geometric load on free eqs
    # Mg * w_free ≈ -P * geo_rhs(N, w) projected on free dofs
    # Use finite columns of RBF Hessian contracted with N
    nw = length(w_index)
    # Map free-w subset
    free_w = Int[]       # indices into w_index that are free DOFs
    free_w_dof = Int[]   # corresponding global free indices into `free`
    for (k, dof) in enumerate(w_index)
        is_kin[dof] && continue
        push!(free_w, k)
        loc = findfirst(==(dof), free)
        loc === nothing && continue
        push!(free_w_dof, loc)
    end
    nf_w = length(free_w)
    Mg = zeros(nfree, nfree)
    nf_w == 0 && return Mg, free

    # Second-derivative ops
    Fxx = Fx * Fx
    Fyy = Fy * Fy
    Fxy = Fx * Fy
    # For each free-w basis e_j, qg = Nxx wxx + ... then scatter to free residual
    # Mg[i,j] = - geo contribution of mode j on free eq i
    # Use: geo_full = P * (Nxx .* (Fxx * w) + ...) with w = E * y (E embeds free-w)
    # Linear in w: qg = L * w where L = diag(N)·Hess
    # Build L (nw × nw): L = Nxx.*Fxx + 2Nxy.*Fxy + Nyy.*Fyy  (elementwise N * matrix)
    L = zeros(nw, nw)
    @inbounds for j in 1:nw
        ej = zeros(nw); ej[j] = 1
        qg = Nxx .* (Fxx * ej) .+ 2 .* Nxy .* (Fxy * ej) .+ Nyy .* (Fyy * ej)
        L[:, j] .= qg
    end
    # weight
    D = 1.0  # scaled later by caller if needed
    # project L onto free-w subspace → nfree block
    for (jj, jw) in enumerate(free_w)
        jfree = free_w_dof[jj]
        for (ii, iw) in enumerate(free_w)
            ifree = free_w_dof[ii]
            Mg[ifree, jfree] -= L[iw, jw]   # geo load on free residual
        end
    end
    return Mg, free
end

"""
    plate_buckling(plate; Nxx, Nyy=0, Nxy=0, nmodes=3) -> (; λ, modes, k_factor)

Buckling eigenvalues for a pre-assembled thin plate under constant membrane
forces `(Nxx, Nyy, Nxy)`.

Returns load factors ``λ`` such that ``N_{cr} = λ N_{ref}``, mode shapes on
free DOFs, and non-dimensional ``k = N_{cr} a² / (π² D)`` when `a` is provided.
"""
function plate_buckling(plate::ThinPlate.PlateMesh;
    Nxx=1.0, Nyy=0.0, Nxy=0.0, nmodes=3, a=nothing)

    isempty(plate.H) && assemble_plate!(plate)
    A, b, is_kin, known = apply_bc_plate(plate)
    free = findall(!, is_kin)
    # Build RBF ops on w-sample points
    n = length(plate.nodes)
    ni = length(plate.internal)
    nc = length(plate.corners)
    pts = SVector{2,Float64}[]
    w_index = Int[]
    for i in 1:n
        push!(pts, plate.nodes[i].pos)
        push!(w_index, 2i - 1)
    end
    for k in 1:ni
        push!(pts, plate.internal[k])
        push!(w_index, 2n + k)
    end
    for c in 1:nc
        push!(pts, plate.corners[c].pos)
        push!(w_index, 2n + ni + c)
    end
    ops = rbf_gradient_ops(pts; rbf=PHS(3; poly_deg=1))
    nw = length(w_index)
    Nxxv = fill(float(Nxx), nw)
    Nyyv = fill(float(Nyy), nw)
    Nxyv = fill(float(Nxy), nw)
    Mg, free = geometric_stiffness_plate(A, is_kin, ops.Fx, ops.Fy, w_index, Nxxv, Nyyv, Nxyv)

    # Condensed elastic operator on free dofs (A already BC-swapped)
    Hb = Matrix(A[free, free])
    nfree = length(free)
    # Generalized eigen Hb φ = λ Mg φ (Mg not necessarily SPD)
    Mg_reg = Matrix(Mg) + 1e-10 * I(nfree)
    # Standard form: (Hb \ Mg) φ = μ φ with μ = 1/λ  when Mg φ = μ Hb φ
    # Prefer QZ via eigen(A,B) without posdef requirement:
    vals = ComplexF64[]
    vecs = zeros(nfree, 0)
    try
        F = eigen(Hb, Mg_reg)
        vals = F.values
        vecs = F.vectors
    catch
        # fallback: invert elastic operator
        μs, V = eigen(Hb \ Mg_reg)
        vals = 1 ./ μs
        vecs = V
    end
    pair = Tuple{Float64,Vector{Float64}}[]
    for i in 1:length(vals)
        λi = real(vals[i])
        isfinite(λi) || continue
        λi > 1e-10 || continue
        vi = real.(vecs[:, i])
        nv = norm(vi)
        nv > 0 && (vi ./= nv)
        push!(pair, (λi, vi))
    end
    sort!(pair; by=first)
    m = min(nmodes, length(pair))
    λs = [pair[i][1] for i in 1:m]
    modes = [pair[i][2] for i in 1:m]
    if m == 0
        # no positive eigen — return placeholder
        λs = [Inf]
        modes = [zeros(nfree)]
        m = 1
    end
    D = bending_stiffness(plate.props)
    kfac = if a !== nothing && abs(Nxx) > 0
        # N_cr = λ * Nxx_ref; k = N_cr a² /(π² D)
        [λs[i] * abs(Nxx) * a^2 / (π^2 * D) for i in 1:m]
    else
        λs .* NaN
    end
    return (λ=λs, modes=modes, k_factor=kfac, free=free, Hb=Hb, Mg=Mg)
end

"""
    thermal_buckling(plate; α, ΔT=1.0, nmodes=3, a=nothing)

Thermal buckling under uniform temperature rise.
Membrane force ``N_{αβ} = -\\frac{E h α ΔT}{1-ν} δ_{αβ}`` (plane stress, constrained).
"""
function thermal_buckling(plate::ThinPlate.PlateMesh;
    α=1e-5, ΔT=1.0, nmodes=3, a=nothing)
    E = plate.props.E
    ν = plate.props.ν
    h = plate.props.h
    # isotropic constrained thermal force (compression for ΔT>0)
    Nth = -E * h * α * ΔT / (1 - ν)
    return plate_buckling(plate; Nxx=Nth, Nyy=Nth, Nxy=0.0, nmodes=nmodes, a=a)
end

"""
Analytical critical uniaxial load for SS square plate:
``N_{cr} = 4 π² D / a²`` (k=4), lowest mode.
"""
function analytical_Ncr_ss_uniaxial(; a=1.0, D=1.0)
    return 4 * π^2 * D / a^2
end
