# Dual Galerkin (SGBEM) Laplace assembly
# Costabel / Steinbach mixed system — V and W symmetric.
#
# Far: nodal lumping. Near: test GL × trial sinh. Coincident: Guiggiani.
# Maue W: ⟨Wφ,ψ⟩ = ∬ (∂_ξ ψ) U (∂_η φ) dξ dη  (Jacobians cancel).

export H_G_galerkin, assemble_galerkin_calderon, assemble_galerkin_W
export applyBC_galerkin!, galerkin_mixed_system

"""
    assemble_galerkin_calderon(dad; npg=20, threaded=true)
        -> (; V, K, W, Mass, H, B)

`V` single-layer (symmetrized), `K` double-layer of the same `T_pkg` as
collocation, `W` Maue hypersingular (symmetrized), `Mass` Galerkin mass.
`H = K − ½ Mass` so `H u = V q` with package flux `q = -k ∂u/∂n`.
`B = H'` keeps the mixed Costabel matrix symmetric.
"""

function assemble_galerkin_calderon(dad::BEMdata{<:Laplace}; npg::Integer=20,
        threaded::Bool=true)
    dad.dimension == 2 || error("H_G_galerkin: 2D Laplace only")
    _init_quadrature!(dad, npg)
    n = dad.n
    V = zeros(n, n)
    K = zeros(n, n)
    W = zeros(n, n)
    Mass = _galerkin_mass(dad)
    elems = dad.elements
    nel = length(elems)
    if threaded && Threads.nthreads() > 1
        bufs = [(zeros(n, n), zeros(n, n), zeros(n, n)) for _ in 1:Threads.nthreads()]
        Threads.@threads for ei in 1:nel
            tid = Threads.threadid()
            Vt, Kt, Wt = bufs[tid]
            for ej in 1:nel
                _galerkin_pair!(Vt, Kt, Wt, dad, elems[ei], elems[ej])
            end
        end
        for (Vt, Kt, Wt) in bufs
            V .+= Vt
            K .+= Kt
            W .+= Wt
        end
    else
        for ei in 1:nel, ej in 1:nel
            _galerkin_pair!(V, K, W, dad, elems[ei], elems[ej])
        end
    end
    V .= (V .+ transpose(V)) ./ 2
    W .= (W .+ transpose(W)) ./ 2
    H = K .- 0.5 .* Mass

    B = transpose(H)
    return (; V, K, W, Mass, H, B)
end

"""
    H_G_galerkin(dad; npg=20, threaded=true) -> (H, G)

`H` is `n_t×n_t`, `G` is `n_t×n`. Boundary–boundary: dual Galerkin
(`H = K − ½M`, `G = V`, same kernels as collocation). Interior rows:
representation at internal points (regular far / sinh).

"""
function H_G_galerkin(dad::BEMdata{<:Laplace}; npg::Integer=20, threaded::Bool=true)
    ops = assemble_galerkin_calderon(dad; npg=npg, threaded=threaded)
    n, nt = dad.n, dad.nt
    H = zeros(nt, nt)
    G = zeros(nt, n)
    H[1:n, 1:n] .= ops.H
    G[1:n, 1:n] .= ops.V
    _galerkin_interior_rows!(H, G, dad; threaded=threaded)
    set_cache!(dad; H=H, G=G, galerkin_V=ops.V, galerkin_K=ops.K,
        galerkin_W=ops.W, galerkin_Mass=ops.Mass, galerkin_B=ops.B)
    return H, G
end

function _galerkin_interior_rows!(H, G, dad; threaded::Bool=true)
    n = dad.n
    nt = dad.nt
    nt > n || return nothing
    elems = dad.elements
    _collocation_loop!(threaded, nt - n) do k
        i = n + k
        pf = point(dad, i)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            if _near_element(pf, xj, el)
                nn = length(el)
                hloc = zeros(nn)
                gloc = zeros(nn)
                integrate_element(dad, el, xj, pf, hloc, gloc)
                for (a, j) in enumerate(el.index)
                    H[i, j] += hloc[a]
                    G[i, j] += gloc[a]
                end
            else
                _far_nodal_scalar!(H, G, dad, el, pf, i)
            end
        end
        H[i, i] = -sum(view(H, i, 1:n))
    end
    return nothing
end


assemble_galerkin_W(dad::BEMdata{<:Laplace}; kwargs...) =
    assemble_galerkin_calderon(dad; kwargs...).W

function _galerkin_mass(dad)
    n = dad.n
    M = zeros(n, n)
    poly = dad.element_type
    ξ, w = dad.qsi, dad.w
    N, dN = shapefun(poly, ξ)
    @inbounds for el in dad.elements
        x = dad.Nodes[el.index]
        dx = dN * x
        Jw = norm.(dx) .* w
        idx = el.index
        nn = length(idx)
        for q in eachindex(ξ)
            wq = Jw[q]
            for a in 1:nn, b in 1:nn
                M[idx[a], idx[b]] += N[q, a] * N[q, b] * wq
            end
        end
    end
    M .= (M .+ transpose(M)) ./ 2
    return M
end

function _pair_near(ei, ej, dad)
    Li = ei.Length
    Lj = ej.Length
    lim = 2 * (Li + Lj)
    xi = dad.Nodes[ei.index]
    xj = dad.Nodes[ej.index]
    @inbounds for a in eachindex(xi), b in eachindex(xj)
        norm(xi[a] - xj[b]) < lim && return true
    end
    return false
end

function _galerkin_pair!(V, K, W, dad, etest, etrial)
    if !_pair_near(etest, etrial, dad)
        _galerkin_pair_far!(V, K, W, dad, etest, etrial)
    else
        _galerkin_pair_near!(V, K, W, dad, etest, etrial)
    end
    return nothing
end

function _galerkin_pair_far!(V, K, W, dad, etest, etrial)
    @inbounds for a in eachindex(etest.index)
        i = etest.index[a]
        wi = etest.Jacobian[a] * dad.elem_weight[a]
        dξi = _dN_at_node(dad, etest, a)
        xi = dad.Nodes[i]
        for b in eachindex(etrial.index)
            j = etrial.index[b]
            wj = etrial.Jacobian[b] * dad.elem_weight[b]
            dξj = _dN_at_node(dad, etrial, b)
            U, T = fundamental(dad, dad.Nodes[j] - xi, dad.Normal[j])
            V[i, j] += wi * U * wj
            K[i, j] += wi * T * wj
            W[i, j] += dξi * U * dξj
        end
    end
    return nothing
end

function _dN_at_node(dad, el, loc)
    poly = dad.element_type
    ξn = poly.nodes
    loc > length(ξn) && return 0.0
    _, dN = shapefun(poly, ξn[loc])
    loc > size(dN, 2) && return 0.0
    return dN[1, loc]
end

function _galerkin_pair_near!(V, K, W, dad, etest, etrial)
    poly = dad.element_type
    ξ, w = dad.qsi, dad.w
    xt = dad.Nodes[etest.index]
    xs = dad.Nodes[etrial.index]
    Nt, dNt = shapefun(poly, ξ)
    dxt = dNt * xt
    Jt = norm.(dxt)
    xtg = Nt * xt
    nn_t = length(etest.index)
    nn_s = length(etrial.index)
    gloc = zeros(nn_s)
    hloc = zeros(nn_s)
    wloc = zeros(nn_s)
    @inbounds for q in eachindex(ξ)
        pf = xtg[q]
        fill!(gloc, 0)
        fill!(hloc, 0)
        integrate_element(dad, etrial, xs, pf, hloc, gloc)
        wt = Jt[q] * w[q]
        fill!(wloc, 0)
        _integrate_U_dxi!(wloc, dad, etrial, xs, pf)
        for a in 1:nn_t
            ia = etest.index[a]
            Na = Nt[q, a] * wt
            dξa = dNt[q, a]
            for b in 1:nn_s
                jb = etrial.index[b]
                V[ia, jb] += Na * gloc[b]
                K[ia, jb] += Na * hloc[b]
                W[ia, jb] += dξa * w[q] * wloc[b]
            end
        end
    end
    return nothing
end

"""Inner ∫ U(x, y(η)) (∂N_b/∂η) dη — log singular, Guiggiani / sinh."""
function _integrate_U_dxi!(g::AbstractVector, dad, elem, nodes, pf::Point2D)
    poly = dad.element_type
    a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
    if dist <= 1e-12
        Ig = guiggiani_integral(ξ -> begin
                samp = _sample_U_dN(dad, poly, nodes, pf, ξ)
                samp === nothing ? zeros(length(g)) : samp
            end, a, 0; qsi=dad.qsi, w=dad.w)
        g .= Ig
        return g
    end
    eta, ww = transform(dad, elem, nodes, pf)
    N, dN = shapefun(poly, eta)
    pg = N * nodes
    dx = dN * nodes
    @inbounds for q in eachindex(eta)
        r = pg[q] - pf
        R = norm(r)
        R < 1e-15 && continue
        U, _ = fundamental(dad, r, tan2normal(dx[q] / norm(dx[q])))
        for b in eachindex(g)
            g[b] += U * dN[q, b] * ww[q]
        end
    end
    return g
end


function _sample_U_dN(dad, poly, nodes, pf, ξ)
    N, dN = shapefun(poly, ξ)
    pg = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for k in eachindex(nodes)
        pg += N[1, k] * nodes[k]
        dx += dN[1, k] * nodes[k]
    end
    J = norm(dx)
    J < 1e-30 && return nothing
    r = pg - pf
    norm(r) < 1e-30 && return nothing
    U, _ = fundamental(dad, r, tan2normal(dx / J))
    nn = length(nodes)
    out = zeros(nn)
    @inbounds for b in 1:nn
        out[b] = U * dN[1, b]
    end
    return out
end

"""
    galerkin_mixed_system(dad; npg=20) -> (; A, b, D, N)

Symmetric Costabel blocks (unknowns `q_D`, `u_N`):

```
A = [ V_DD   −H_DN ]
    [ −H_DNᵀ  W_NN ]
```
"""
function galerkin_mixed_system(dad::BEMdata{<:Laplace}; npg::Integer=20, threaded::Bool=true)
    ops = assemble_galerkin_calderon(dad; npg=npg, threaded=threaded)
    V, H, W = ops.V, ops.H, ops.W
    BC = Int.(dad.BC)
    D = Int[i for i in 1:dad.n if BC[i] == 0]
    Nset = Int[i for i in 1:dad.n if BC[i] != 0]
    nD, nN = length(D), length(Nset)
    uD = zeros(dad.n)
    qN = zeros(dad.n)
    @inbounds for i in D
        uD[i] = dad.BV[i]
    end
    @inbounds for i in Nset
        qN[i] = dad.BV[i]
    end
    if nN == 0
        A = Matrix(V[D, D])
        b = Vector(H[D, :] * uD)
        A .= (A .+ A') ./ 2
        return (; A, b, D, N=Nset, ops..., mode=:dirichlet)
    end
    if nD == 0
        A = Matrix(W[Nset, Nset])
        b = Vector(ops.B[Nset, :] * qN)
        A .= (A .+ A') ./ 2
        e = ones(nN)
        A .+= e * e'
        return (; A, b, D, N=Nset, ops..., mode=:neumann)
    end
    m = nD + nN
    A = zeros(m, m)
    bb = zeros(m)
    A[1:nD, 1:nD] .= V[D, D]
    A[1:nD, nD+1:m] .= .-H[D, Nset]
    A[nD+1:m, 1:nD] .= .-transpose(H[D, Nset])
    A[nD+1:m, nD+1:m] .= W[Nset, Nset]
    bb[1:nD] .= H[D, :] * uD .- V[D, :] * qN
    bb[nD+1:m] .= .-W[Nset, :] * uD .+ ops.B[Nset, :] * qN
    A .= (A .+ A') ./ 2
    return (; A, b=bb, D, N=Nset, ops..., mode=:mixed)
end

function applyBC_galerkin!(dad::BEMdata{<:Laplace}; npg::Integer=20, threaded::Bool=true)
    sys = galerkin_mixed_system(dad; npg=npg, threaded=threaded)
    set_cache!(dad; A=sys.A, b=sys.b, galerkin_sys=sys)
    has_cache(dad, :H) || H_G_galerkin(dad; npg=npg, threaded=threaded)
    return sys
end
