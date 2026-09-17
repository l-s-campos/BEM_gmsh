# Compare integral/Guiggiani OIFs with Chen–Gu IIT + SAB on the same mesh,
# then rerun SBM-DRM with each diagonal.
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Statistics
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))

sample_u(p) = p[1] - p[2]
sample_q(p, n) = -(n[1] - n[2])   # q_pkg = −k ∂u/∂n, k=1, ∇u=(1,-1)

function sab_qii(d; weighted=true, adjoint=false)
    n = length(d.nodes)
    L = d.lengths
    qii = zeros(n)
    @inbounds for i in 1:n
        xi, ni = d.nodes[i], d.normals[i]
        s = 0.0
        for j in 1:n
            j == i && continue
            if adjoint
                T = BEM._sbm_Q_field(d.nodes[j] - xi, d.normals[j])  # T(s,x)
            else
                T = BEM._sbm_Q_field(xi - d.nodes[j], ni)            # T(x,s)
            end
            s += weighted ? L[j] * T : T
        end
        qii[i] = weighted ? -s / L[i] : -s
    end
    return qii
end

"""IIT Dirichlet OIFs from sample ū=x−y and a Neumann SBM solve with given q_ii."""
function iit_uii(d, qii)
    n = length(d.nodes)
    k = d.k
    H = zeros(n, n)
    Goff = zeros(n, n)
    @inbounds for i in 1:n
        xi, ni = d.nodes[i], d.normals[i]
        for j in 1:n
            if i == j
                H[i, i] = qii[i]
                continue
            end
            r = xi - d.nodes[j]
            H[i, j] = BEM._sbm_Q_field(r, ni)
            Goff[i, j] = BEM._sbm_U(r, k)
        end
    end
    q̄ = [sample_q(d.nodes[i], d.normals[i]) for i in 1:n]
    A = copy(H)
    A[n, :] .= 1.0
    b = copy(q̄)
    b[n] = 0.0
    α = A \ b
    ū = [sample_u(p) for p in d.nodes]
    uii = zeros(n)
    @inbounds for i in 1:n
        abs(α[i]) < 1e-14 && (uii[i] = NaN; continue)
        uii[i] = (ū[i] - (Goff[i, :] ⋅ α)) / α[i]
    end
    return uii, α, H
end

function assemble_with!(d, uii, qii)
    n = length(d.nodes)
    k = d.k
    G = zeros(n, n)
    H = zeros(n, n)
    @inbounds for i in 1:n
        xi, ni = d.nodes[i], d.normals[i]
        for j in 1:n
            if i == j
                G[i, i] = uii[i]
                H[i, i] = qii[i]
                continue
            end
            r = xi - d.nodes[j]
            G[i, j] = BEM._sbm_U(r, k)
            H[i, j] = BEM._sbm_Q_field(r, ni)
        end
    end
    d.u_ii = uii
    d.q_ii = qii
    d.G = G
    d.H = H
    return d
end

function rel_vec(a, b)
    return norm(a .- b) / max(norm(b), eps())
end

function oifs_on(dad, which::Symbol)
    d = sbm_from_bemdata(dad; internal=false)
    uii_int, qii_int = origin_intensity_factors!(d)
    q_sab = sab_qii(d; weighted=true, adjoint=false)
    if which === :integral
        return uii_int, qii_int
    elseif which === :iit
        uii, _, _ = iit_uii(d, q_sab)
        return uii, q_sab
    elseif which === :iit_intq
        uii, _, _ = iit_uii(d, qii_int)
        return uii, qii_int
    elseif which === :emp_sab
        emp = [-log(max(L, 1e-12)) / (2π * d.k) for L in d.lengths]
        return emp, q_sab
    elseif which === :iit_unw
        q_u = sab_qii(d; weighted=false, adjoint=false)
        uii, _, _ = iit_uii(d, q_u)
        return uii, q_u
    else
        error(which)
    end
end

function drm_sine_with_oif(which::Symbol; ndiv=10, Δt=0.005, tf=0.05)
    msh = quadrado(ndiv=ndiv, show=false, nome="oif_drm_sin_$which", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.BC .= 0
    dad.BV .= 0
    u0 = [sin(pi * point(dad, i)[1]) * sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    u0[1:dad.n] .= 0
    uii, qii = oifs_on(dad, which)
    d = sbm_from_bemdata(dad; internal=false)
    assemble_with!(d, uii, qii)
    st = sbm_drm_setup(dad; κ=1.0, Δt=Δt, scheme=:houbolt)
    st.Gbb .= d.G
    st.Hbb .= d.H
    N, M = dad.n, dad.ni
    A = zeros(N + M, N + M)
    @inbounds for i in 1:N
        A[i, 1:N] .= view(st.Gbb, i, :)
        A[i, N+1:N+M] .= view(st.Φb, i, :)
    end
    invγ = 6 * st.κ * st.Δt / 11
    @inbounds for i in 1:M
        row = N + i
        A[row, 1:N] .= view(st.Gib, i, :)
        for j in 1:M
            A[row, N + j] = st.Φi[i, j] - invγ * st.φi[i, j]
        end
    end
    st.A = A
    AF_h = factorize(A)
    invγE = st.κ * Δt
    AE = zeros(N + M, N + M)
    @inbounds for i in 1:N
        AE[i, 1:N] .= view(st.Gbb, i, :)
        AE[i, N+1:N+M] .= view(st.Φb, i, :)
    end
    @inbounds for i in 1:M
        row = N + i
        AE[row, 1:N] .= view(st.Gib, i, :)
        for j in 1:M
            AE[row, N + j] = st.Φi[i, j] - invγE * st.φi[i, j]
        end
    end
    AF_e = factorize(AE)
    nT = length(0.0:Δt:tf)
    Ufull = zeros(N + M, nT)
    Ufull[:, 1] .= u0
    Ufull[:, 1] .= sbm_drm_project_ic!(st, view(Ufull, :, 1))
    st.AF = AF_e
    nT >= 2 && (Ufull[:, 2] .= sbm_drm_step!(st, view(Ufull, :, 1)))
    nT >= 3 && (Ufull[:, 3] .= sbm_drm_step!(st, view(Ufull, :, 2)))
    st.AF = AF_h
    @inbounds for i in 4:nT
        Ufull[:, i] .= sbm_drm_step!(st, view(Ufull, :, i - 1);
                                     u_nm1=view(Ufull, :, i - 2),
                                     u_nm2=view(Ufull, :, i - 3))
    end
    uex = [exp(-2 * pi^2 * tf) * sin(pi * point(dad, i)[1]) *
           sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    ii = (N + 1):(N + M)
    rmse = sqrt(mean(abs2, Ufull[ii, end] .- uex[ii]))
    return rmse, maximum(abs, Ufull), all(isfinite, Ufull)
end

function main()
    msh = quadrado(ndiv=12, show=false, nome="oif_cmp", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    d = sbm_from_bemdata(dad)
    uii_int, qii_int = origin_intensity_factors!(d)
    qii_sab_w = sab_qii(d; weighted=true, adjoint=false)
    qii_sab_a = sab_qii(d; weighted=true, adjoint=true)   # current integral (I=0)
    qii_sab_u = sab_qii(d; weighted=false, adjoint=false)
    uii_iit, αs, _ = iit_uii(d, qii_sab_w)
    uii_iit_a, _, _ = iit_uii(d, qii_int)

    emp = [-log(max(L, 1e-12)) / (2π * d.k) for L in d.lengths]

    println("n=$(length(d))  Lmean=$(mean(d.lengths))")
    @printf("u_ii  integral  mean=% .4f  min=% .4f  max=% .4f\n",
            mean(uii_int), minimum(uii_int), maximum(uii_int))
    @printf("u_ii  IIT(SAB)  mean=% .4f  min=% .4f  max=% .4f  finite=%s\n",
            mean(filter(isfinite, uii_iit)), minimum(filter(isfinite, uii_iit)),
            maximum(filter(isfinite, uii_iit)), all(isfinite, uii_iit))
    @printf("u_ii  IIT(int q) mean=% .4f\n", mean(filter(isfinite, uii_iit_a)))
    @printf("u_ii  emp -log(L)/2π  mean=% .4f\n", mean(emp))
    println("  ||u_int - u_IIT|| / ||u_IIT|| = $(rel_vec(uii_int, uii_iit))")
    println("  ||u_int - emp|| / ||emp||     = $(rel_vec(uii_int, emp))")
    println("  ||u_IIT - emp|| / ||emp||     = $(rel_vec(uii_iit, emp))")
    println("  corr(u_int, u_IIT) = $(cor(uii_int, uii_iit))")
    println()
    @printf("q_ii  integral     mean=% .4f\n", mean(qii_int))
    @printf("q_ii  SAB field w  mean=% .4f\n", mean(qii_sab_w))
    @printf("q_ii  SAB adjoint  mean=% .4f  (=integral if I=0)\n", mean(qii_sab_a))
    @printf("q_ii  SAB unweighted mean=% .4f\n", mean(qii_sab_u))
    println("  ||q_int - q_SABw|| / ||q_SABw|| = $(rel_vec(qii_int, qii_sab_w))")
    println("  ||q_int - q_adj||  / ||q_adj||  = $(rel_vec(qii_int, qii_sab_a))")
    println("  sample α IIT: mean=$(mean(αs))  min=$(minimum(αs)) max=$(maximum(αs))")

    # Γ_m length vs L_m
    Lm = d.lengths[1]
    geo, poly, ξm = BEM.sbm_element_geom(d, 1)
    a, b = BEM.sbm_xi_interval(d, 1)
    # physical length of Voronoi
    qsi, w = gausslegendre(20)
    len = 0.0
    sξ = (b - a) / 2
    cξ = (b + a) / 2
    for (η, ww) in zip(qsi, w)
        ξ = sξ * η + cξ
        _, J, _ = BEM.sbm_geom_at(geo, poly, ξ)
        len += J * sξ * ww
    end
    println()
    println("node 1: L_m=$Lm  |Γ_m|≈$len  |Γ|/L=$(len/Lm)  ξm=$ξm interval=($a,$b)")
    println("  IU/L_m=$(uii_int[1])  IU/|Γ|=$(uii_int[1]*Lm/len)")

    # reproduction of sample ū=x−y
    n = length(d.nodes)
    ū = [sample_u(p) for p in d.nodes]
    assemble_with!(d, uii_iit, qii_sab_w)
    uhat = d.G * αs
    println("IIT Gα vs ū  rel=$(rel_vec(uhat, ū))")
    assemble_with!(d, uii_int, qii_int)
    Hint = d.H
    A = copy(Hint); A[n, :] .= 1
    b = [sample_q(d.nodes[i], d.normals[i]) for i in 1:n]; b[n] = 0
    αint = A \ b
    uhat2 = d.G * αint
    println("integral Gα vs ū  rel=$(rel_vec(uhat2, ū))  (α from integral H)")

    println("\nSBM-DRM sine with swapped OIFs (+ IC projection)")
    for (name, which) in (("integral", :integral), ("IIT+SABw", :iit),
                          ("IIT+unwSAB", :iit_unw), ("emp+SAB", :emp_sab))
        rmse, mx, fin = drm_sine_with_oif(which)
        @printf("  %-12s RMSE=%.4e  max|u|=%.3f  finite=%s\n", name, rmse, mx, fin)
    end
end
main()
