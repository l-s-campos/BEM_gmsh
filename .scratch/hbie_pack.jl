using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(mesh; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(mesh; npg=4)
solve_fsdt!(mesh)
pf, nx = SVector(0.5, 0.5), SVector(1.0, 0.0)
tF, aux = unsym_interior_t_fd(mesh, pf, nx; h=1e-3, npg=4, nsub=4)
tH = unsym_interior_t(mesh, pf, nx; npg=4, nsub=4)
@printf("FD   Mx=%.4e  Nx=%.4e  Q=%.3e\n", tF[3], tF[1], tF[5])
@printf("HBIE Mx=%.4e  Nx=%.4e  Q=%.3e   Mx/FD=%.3f\n",
    tH[3], tH[1], tH[5], tH[3] / tF[3])

# rebuild interior t with transpose and/or no domain
function tint(pf, nξ; transp=false, domain=true)
    props = mesh.props
    poly = mesh.element_type
    qsi, w = BEM.Plate.gausslegendre(4)
    ub, tb = mesh.u, mesh.t
    acc = zeros(5)
    dummy = nξ
    for el in mesh.elements
        nN = length(el.index)
        He = zeros(5, 5nN)
        Ge = zeros(5, 5nN)
        Cije = zeros(5, 5)
        x1, x3 = el.geo[1], el.geo[end]
        Le = norm(x3 - x1)
        Rmin = minimum(norm(mesh.nodes[j] - pf) for j in el.index)
        if Rmin <= Le / 4
            BEM.Plate._unsym_telles_sub!(He, Ge, el, poly, pf, nξ, props, qsi, w;
                nsub=4, bie=:hbie)
        else
            BEM.Plate._add_unsym_el!(He, Ge, Cije, el, poly, pf, props, qsi, w;
                bie=:hbie, nξ=nξ)
        end
        for a in 1:nN
            ja = el.index[a]
            W = Ge[:, 5a-4:5a]
            S = He[:, 5a-4:5a]
            if transp
                acc .+= W' * tb[5ja-4:5ja]
                acc .-= S' * ub[5ja-4:5ja]
            else
                acc .+= W * tb[5ja-4:5ja]
                acc .-= S * ub[5ja-4:5ja]
            end
        end
    end
    if domain
        IDW = BEM.Plate._unsym_ID_W_point(mesh, pf, nξ; npg=4)
        acc .+= (transp ? IDW' : IDW)[:, 5] .* props.q_c
    end
    return acc
end

for (lab, kw) in (
        ("plain", (; transp=false, domain=true)),
        ("T", (; transp=true, domain=true)),
        ("plain no d", (; transp=false, domain=false)),
        ("T no d", (; transp=true, domain=false)),
    )
    t = tint(pf, nx; kw...)
    @printf("%-12s Mx=%.4e  Nx=%.4e  relMx=%.2f  relN=%.2f\n", lab, t[3], t[1],
        abs(t[3] - tF[3]) / abs(tF[3]), abs(t[1] - tF[1]) / max(abs(tF[1]), 1e-30))
end
