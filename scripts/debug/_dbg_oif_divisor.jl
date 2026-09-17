# What if OIF integrals are divided by element length instead of neighbour L_m?
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Statistics
using Printf
using StaticArrays

include(datadir("Laplace", "Laplace_dad.jl"))

function patch_len(d, m; full=false)
    geo, poly, _ = BEM.sbm_element_geom(d, m)
    a, b = full ? (-1.0, 1.0) : BEM.sbm_xi_interval(d, m)
    sξ = (b - a) / 2
    cξ = (b + a) / 2
    qsi, w = gausslegendre(20)
    len = 0.0
    for (η, ww) in zip(qsi, w)
        ξ = sξ * η + cξ
        _, J, _ = BEM.sbm_geom_at(geo, poly, ξ)
        len += J * sξ * ww
    end
    return len
end

function integrals(d, m; full=false)
    geo, poly, ξm0 = BEM.sbm_element_geom(d, m)
    ξa, ξb = full ? (-1.0, 1.0) : BEM.sbm_xi_interval(d, m)
    sξ = (ξb - ξa) / 2
    cξ = (ξb + ξa) / 2
    ξm = clamp((ξm0 - cξ) / sξ, nextfloat(-1.0), prevfloat(1.0))
    xm, nm, k, dad = d.nodes[m], d.normals[m], d.k, d.dad
    qsi, w = gausslegendre(20)
    Ig, Ih = guiggiani_GH(ξm; order_G=0, order_H=-1, qsi=qsi, w=w) do η
        ξ = sξ * η + cξ
        pg, J, nrm = BEM.sbm_geom_at(geo, poly, ξ)
        J < 1e-30 && return 0.0, 0.0
        rxs = xm - pg
        hypot(rxs[1], rxs[2]) < 1e-30 && return 0.0, 0.0
        wJ = J * sξ
        U, Txs = fundamental(dad, rxs, nm)
        _, Tsx = fundamental(dad, pg - xm, nrm)
        return U * wJ, (Txs + Tsx) * wJ
    end
    return Ig, Ih
end

function oifs(d; divisor::Symbol, full::Bool=false)
    n = length(d.nodes)
    κ = d.kappa
    L = d.lengths
    dad = d.dad
    uii = zeros(n)
    qii = zeros(n)
    den = zeros(n)
    qsi, w = gausslegendre(20)
    @inbounds for m in 1:n
        IU, II = integrals(d, m; full=full)
        if divisor === :L
            den[m] = L[m]
        elseif divisor === :gamma
            den[m] = patch_len(d, m; full=full)
        elseif divisor === :elem
            den[m] = dad.elements[d.col_el[m]].Length
        else
            error(divisor)
        end
        sT = 0.0
        xm = d.nodes[m]
        for nn in 1:n
            nn == m && continue
            sT += L[nn] * BEM._sbm_Q_field(d.nodes[nn] - xm, d.normals[nn])
        end
        uii[m] = IU / den[m]
        qii[m] = (II - κ * sT) / den[m]
    end
    return uii, qii, den
end

function assemble_with!(d, uii, qii)
    n = length(d.nodes)
    k = d.k
    G = zeros(n, n); H = zeros(n, n)
    @inbounds for i in 1:n
        xi, ni = d.nodes[i], d.normals[i]
        for j in 1:n
            if i == j
                G[i, i] = uii[i]; H[i, i] = qii[i]
            else
                r = xi - d.nodes[j]
                G[i, j] = BEM._sbm_U(r, k)
                H[i, j] = BEM._sbm_Q_field(r, ni)
            end
        end
    end
    d.u_ii, d.q_ii, d.G, d.H = uii, qii, G, H
    return d
end

function static_mixed(uii, qii)
    msh = quadrado(ndiv=12, show=false, nome="div_mix", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    for i in 1:dad.n
        dad.BV[i] = dad.BC[i] == 0 ? float(ana.u(dad.Nodes[i])) :
                    float(ana.q(dad.Nodes[i], dad.Normal[i]))
    end
    attach_analytical!(dad, ana)
    d = sbm_from_bemdata(dad)
    assemble_with!(d, uii, qii)
    solve_sbm!(d)
    sbm_eval_internal!(d)
    uc = sbm_eval_u(d, Point2D(0.5, 0.5))
    return sbm_rel_error(d, ana), sbm_rel_error_internal(d, ana), uc
end

function main()
    msh = quadrado(ndiv=12, show=false, nome="div_oif", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    d = sbm_from_bemdata(dad)
    origin_intensity_factors!(d)

    cases = (
        (name="L_m neighbour, Voronoi Γ", divisor=:L, full=false),
        (name="|Γ_m| Voronoi length", divisor=:gamma, full=false),
        (name="element Length, Voronoi Γ", divisor=:elem, full=false),
        (name="element Length, full Γ", divisor=:elem, full=true),
        (name="|Γ_m| full element", divisor=:gamma, full=true),
    )

    println("n=$(dad.n)  mean L_m=$(mean(d.lengths))  mean el.Length=$(mean(el.Length for el in dad.elements))")
    println("node1  L_m=$(d.lengths[1])  |Γ_v|=$(patch_len(d,1; full=false))  |Γ_full|=$(patch_len(d,1; full=true))  el.L=$(dad.elements[d.col_el[1]].Length)")
    println()

    results = []
    for c in cases
        uii, qii, den = oifs(d; divisor=c.divisor, full=c.full)
        eb, ei, uc = static_mixed(uii, qii)
        @printf("%-36s  den̄=%.4f  ū=%.4f  q̄=%.3f  err_b=%.4f  err_i=%.4f  u(c)=%.4f\n",
                c.name, mean(den), mean(uii), mean(qii), eb, ei, uc)
        push!(results, (; c..., uii, qii, den, eb, ei, uc))
    end
    return results
end
main()
