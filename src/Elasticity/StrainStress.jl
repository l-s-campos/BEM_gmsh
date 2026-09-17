# Boundary strain / stress from ∇u + traction (Kane). Each discontinuous
# collocation node belongs to one element — no nodal averaging.

export recover_strain_stress!, voigt_stiffness

function voigt_stiffness(props::Elasticity, dim::Integer)
    λ, μ = props.lambda, props.mu
    if dim == 2
        return @SMatrix [λ+2μ  λ  0; λ  λ+2μ  0; 0  0  μ]
    elseif dim == 3
        c11 = λ + 2μ
        C = zeros(6, 6)
        C[1, 1] = C[2, 2] = C[3, 3] = c11
        C[1, 2] = C[2, 1] = C[1, 3] = C[3, 1] = C[2, 3] = C[3, 2] = λ
        C[4, 4] = C[5, 5] = C[6, 6] = μ
        return C
    end
    throw(ArgumentError("dim must be 2 or 3"))
end

voigt_stiffness(props::AnisotropicElasticity3D, ::Integer) = Matrix(props.C)

function voigt_stiffness(props::AnisotropicElasticity, dim::Integer)
    dim == 2 || throw(ArgumentError("Lekhnitskii stiffness is 2D"))
    return Matrix(props.params.C)
end

"""
    recover_strain_stress!(dad) -> (ε, σ)

Boundary strain and stress at collocation nodes from the interpolant ``∇u``
and the traction ``t = σ n`` (Kane). Stores `dad.strain`, `dad.stress`.
Voigt: 2D ``(ε_{11},ε_{22},γ_{12})`` / 3D ``(ε_{11},ε_{22},ε_{33},γ_{23},γ_{13},γ_{12})``.
"""
function recover_strain_stress!(dad::BEMdata)
    has_cache(dad, :u) || error("no displacement — call solve first")
    has_cache(dad, :traction) || error("no traction — call solve first")
    dim = dad.dimension
    nvoigt = dim == 2 ? 3 : 6
    n = dad.n
    ε = zeros(n, nvoigt)
    σ = zeros(n, nvoigt)
    C = voigt_stiffness(dad.properties, dim)
    u = dad.u
    t = dad.traction
    if dim == 2
        _recover_2d!(ε, σ, dad, u, t, C)
    else
        _recover_3d!(ε, σ, dad, u, t, C)
    end
    set_cache!(dad; strain=ε, stress=σ)
    return ε, σ
end

function _recover_2d!(ε, σ, dad, u, t, C)
    poly = dad.element_type
    ξs = poly.nodes
    λ = dad.properties isa Elasticity ? dad.properties.lambda : C[1, 2]
    μ = dad.properties isa Elasticity ? dad.properties.mu : C[3, 3]
    @inbounds for elem in dad.elements
        idx = elem.index
        X = dad.Nodes[idx]
        nn = length(idx)
        for k in 1:nn
            ξ = ξs[min(k, length(ξs))]
            N, dN = shapefun(poly, ξ)
            dx = zero(X[1])
            du = zero(SVector{2,Float64})
            for j in 1:nn
                dx += dN[1, j] * X[j]
                ja = idx[j]
                du += dN[1, j] * SVector(u[2ja-1], u[2ja])
            end
            i = idx[k]
            n̂, t̂ = local_basis2d(dad.Normal[i])
            Jproj = t̂[1] * dx[1] + t̂[2] * dx[2]
            abs(Jproj) < 1e-16 && continue
            dudt = du / Jproj
            εss = t̂[1] * dudt[1] + t̂[2] * dudt[2]
            tv = SVector(t[2i-1], t[2i])
            σnn = n̂[1] * tv[1] + n̂[2] * tv[2]
            σnt = t̂[1] * tv[1] + t̂[2] * tv[2]
            den = λ + 2μ
            abs(den) < 1e-30 && continue
            εnn = (σnn - λ * εss) / den
            εnt = σnt / (2μ)
            σss = λ * (εnn + εss) + 2μ * εss
            n1, n2 = n̂[1], n̂[2]
            t1, t2 = t̂[1], t̂[2]
            ε[i, 1] = εnn * n1 * n1 + εss * t1 * t1 + 2εnt * n1 * t1
            ε[i, 2] = εnn * n2 * n2 + εss * t2 * t2 + 2εnt * n2 * t2
            ε[i, 3] = 2 * (εnn * n1 * n2 + εss * t1 * t2 + εnt * (n1 * t2 + n2 * t1))
            σ[i, 1] = σnn * n1 * n1 + σss * t1 * t1 + 2σnt * n1 * t1
            σ[i, 2] = σnn * n2 * n2 + σss * t2 * t2 + 2σnt * n2 * t2
            σ[i, 3] = σnn * n1 * n2 + σss * t1 * t2 + σnt * (n1 * t2 + n2 * t1)
        end
    end
    return nothing
end

function _recover_3d!(ε, σ, dad, u, t, Cglob)
    poly = dad.element_type
    ξs = poly.nodes
    nξ = length(ξs)
    C4 = dad.properties isa AnisotropicElasticity3D ?
         dad.properties.C4 : voigt_to_tensor(Cglob)
    @inbounds for elem in dad.elements
        idx = elem.index
        X = dad.Nodes[idx]
        nn = length(idx)
        for k in 1:nn
            iξ = (k - 1) % nξ + 1
            iη = (k - 1) ÷ nξ + 1
            ξ = ξs[iξ]
            η = ξs[min(iη, nξ)]
            L, Lξ, Lη = shapefun2D(poly, poly, ξ, η)
            xξ = zero(X[1])
            xη = zero(X[1])
            uξ = zero(SVector{3,Float64})
            uη = zero(SVector{3,Float64})
            nN = size(L, 2)
            for j in 1:nN
                xξ += Lξ[1, j] * X[j]
                xη += Lη[1, j] * X[j]
                ja = idx[j]
                uj = SVector(u[3ja-2], u[3ja-1], u[3ja])
                uξ += Lξ[1, j] * uj
                uη += Lη[1, j] * uj
            end
            i = idx[k]
            nvec = dad.Normal[i]
            nlen = norm(nvec)
            nlen < 1e-16 && continue
            n̂ = nvec / nlen
            a = xξ - n̂ * dot(xξ, n̂)
            na = norm(a)
            if na < 1e-14
                a = xη - n̂ * dot(xη, n̂)
                na = norm(a)
            end
            na < 1e-14 && continue
            a = a / na
            b = n̂ × a
            function dirder(e)
                # e ≈ α xξ + β xη  (tangent)
                G = hcat(SVector(xξ[1], xξ[2], xξ[3]), SVector(xη[1], xη[2], xη[3]))
                GtG = G' * G
                det(GtG) < 1e-20 && return zero(SVector{3,Float64})
                αβ = GtG \ (G' * e)
                return αβ[1] * uξ + αβ[2] * uη
            end
            ua = dirder(a)
            ub = dirder(b)
            εaa = dot(a, ua)
            εbb = dot(b, ub)
            εab = (dot(a, ub) + dot(b, ua)) / 2
            tv = SVector(t[3i-2], t[3i-1], t[3i])
            σnn = dot(n̂, tv)
            σan = dot(a, tv)
            σbn = dot(b, tv)
            Q = [a[1] b[1] n̂[1]; a[2] b[2] n̂[2]; a[3] b[3] n̂[3]]
            Cloc = _rotate_C4(C4, Q)
            # ε_loc = [εaa, εbb, εnn, γ_bn, γ_an, γ_ab]
            # known k = (εaa, εbb, γ_ab); unknown x = (εnn, γ_bn, γ_an)
            # rows 3,4,5 of σ = Cloc * ε
            A = @SMatrix [
                Cloc[3, 3] Cloc[3, 4] Cloc[3, 5]
                Cloc[4, 3] Cloc[4, 4] Cloc[4, 5]
                Cloc[5, 3] Cloc[5, 4] Cloc[5, 5]
            ]
            rhs = SVector(
                σnn - (Cloc[3, 1] * εaa + Cloc[3, 2] * εbb + Cloc[3, 6] * 2εab),
                σbn - (Cloc[4, 1] * εaa + Cloc[4, 2] * εbb + Cloc[4, 6] * 2εab),
                σan - (Cloc[5, 1] * εaa + Cloc[5, 2] * εbb + Cloc[5, 6] * 2εab),
            )
            abs(det(A)) < 1e-18 && continue
            xunk = A \ rhs
            εnn, γbn, γan = xunk[1], xunk[2], xunk[3]
            εan, εbn = γan / 2, γbn / 2
            εloc = @SMatrix [
                εaa εab εan
                εab εbb εbn
                εan εbn εnn
            ]
            εv = SVector(εaa, εbb, εnn, γbn, γan, 2εab)
            σv = Cloc * εv
            σloc_t = @SMatrix [
                σv[1] σv[6] σv[5]
                σv[6] σv[2] σv[4]
                σv[5] σv[4] σv[3]
            ]
            εg = Q * εloc * Q'
            σg = Q * σloc_t * Q'
            ε[i, 1] = εg[1, 1]
            ε[i, 2] = εg[2, 2]
            ε[i, 3] = εg[3, 3]
            ε[i, 4] = 2εg[2, 3]
            ε[i, 5] = 2εg[1, 3]
            ε[i, 6] = 2εg[1, 2]
            σ[i, 1] = σg[1, 1]
            σ[i, 2] = σg[2, 2]
            σ[i, 3] = σg[3, 3]
            σ[i, 4] = σg[2, 3]
            σ[i, 5] = σg[1, 3]
            σ[i, 6] = σg[1, 2]
        end
    end
    return nothing
end

function _rotate_C4(C4::Array{Float64,4}, Q::AbstractMatrix)
    Cp = zeros(Float64, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        s = 0.0
        for p in 1:3, q in 1:3, r in 1:3, t in 1:3
            s += Q[i, p] * Q[j, q] * Q[k, r] * Q[l, t] * C4[p, q, r, t]
        end
        Cp[i, j, k, l] = s
    end
    return tensor_to_voigt(Cp)
end
