# =============================================================================
# Dual-tree 2D Laplace FMM on ClusterTree
#
# Expansions are the complex log multipoles of Flatiron l2d*. Physical potential
# is Re(φ). Densities are therefore processed as real channels; complex charges
# run two real FMMs and recombine.
# =============================================================================

"""Output container (Flatiron-style)."""
mutable struct FMMVals
    pot::Any
    grad::Any
    hess::Any
    pottarg::Any
    gradtarg::Any
    hesstarg::Any
    pre::Any
    pretarg::Any
    ier::Any
end
FMMVals() = FMMVals(
    nothing, nothing, nothing,
    nothing, nothing, nothing,
    nothing, nothing,
    0,
)

"""Strong admissibility: `dist(A,B) ≥ η * max(diam(A), diam(B))`."""
Base.@kwdef struct StrongAdmissibility
    η::Float64 = 1.0
end

function (adm::StrongAdmissibility)(a::ClusterTree, b::ClusterTree)
    return distance(a, b) >= adm.η * max(diameter(a), diameter(b))
end

struct FMMNodeData
    multipole::Vector{ComplexF64}
    localexp::Vector{ComplexF64}
    rscale::Float64
    center::SVector{2,Float64}
end

function _node_center_scale(node::ClusterTree{2,T}) where {T}
    c = center(container(node))
    hs = max(maximum(high_corner(container(node)) - low_corner(container(node))) / 2, 1e-30)
    return SVector{2,Float64}(c), Float64(hs)
end

function _allocate_expdata(root::ClusterTree{2}, nterms::Int)
    data = Dict{UInt,FMMNodeData}()
    for node in nodes(root)
        ctr, rsc = _node_center_scale(node)
        data[objectid(node)] = FMMNodeData(
            zeros(ComplexF64, nterms + 1),
            zeros(ComplexF64, nterms + 1),
            rsc,
            ctr,
        )
    end
    return data
end

_get(data, node) = data[objectid(node)]

function _zero_locals!(data)
    for nd in values(data)
        fill!(nd.localexp, 0)
    end
end

function _zero_multipoles!(data)
    for nd in values(data)
        fill!(nd.multipole, 0)
    end
end

# ---------- upward ----------
function _upward!(node::ClusterTree{2}, data, sources, charges, dipoles, carray)
    nd = _get(data, node)
    if isleaf(node)
        fill!(nd.multipole, 0)
        charges !== nothing && form_multipole_charge!(
            nd.multipole, nd.rscale, nd.center, sources, charges, index_range(node)
        )
        dipoles !== nothing && form_multipole_dipole!(
            nd.multipole, nd.rscale, nd.center, sources, dipoles, index_range(node)
        )
    else
        fill!(nd.multipole, 0)
        for child in children(node)
            _upward!(child, data, sources, charges, dipoles, carray)
            cd = _get(data, child)
            m2m!(nd.multipole, nd.rscale, nd.center, cd.multipole, cd.rscale, cd.center, carray)
        end
    end
    return nothing
end

# ---------- dual-tree interaction ----------
function _interact!(
    tnode::ClusterTree{2},
    snode::ClusterTree{2},
    tdata,
    sdata,
    sources,
    charges,
    dipstr_loc,
    dipvec_loc,
    targets,
    pot_leaf,
    grad_leaf,
    carray,
    adm,
    thresh::Float64,
    same_tree::Bool,
)
    if adm(tnode, snode)
        td = _get(tdata, tnode)
        sd = _get(sdata, snode)
        m2l!(td.localexp, td.rscale, td.center, sd.multipole, sd.rscale, sd.center, carray)
        return nothing
    end

    if isleaf(tnode) && isleaf(snode)
        p = pot_leaf[objectid(tnode)]
        g = grad_leaf === nothing ? nothing : grad_leaf[objectid(tnode)]
        direct_laplace_sv!(
            p,
            g,
            sources,
            charges,
            dipstr_loc,
            dipvec_loc,
            index_range(snode),
            targets,
            index_range(tnode);
            thresh=thresh,
            exclude_self=same_tree,
        )
        return nothing
    end

    if isleaf(snode) || (!isleaf(tnode) && diameter(tnode) >= diameter(snode))
        for c in children(tnode)
            _interact!(
                c, snode, tdata, sdata, sources, charges, dipstr_loc, dipvec_loc, targets,
                pot_leaf, grad_leaf, carray, adm, thresh, same_tree,
            )
        end
    else
        for c in children(snode)
            _interact!(
                tnode, c, tdata, sdata, sources, charges, dipstr_loc, dipvec_loc, targets,
                pot_leaf, grad_leaf, carray, adm, thresh, same_tree,
            )
        end
    end
    return nothing
end

# ---------- downward ----------
function _downward!(node::ClusterTree{2}, data, targets, pot_leaf, grad_leaf, carray)
    nd = _get(data, node)
    if isleaf(node)
        p = pot_leaf[objectid(node)]
        g = grad_leaf === nothing ? nothing : grad_leaf[objectid(node)]
        eval_local!(p, g, nd.rscale, nd.center, nd.localexp, targets, index_range(node))
    else
        for child in children(node)
            cd = _get(data, child)
            l2l!(cd.localexp, cd.rscale, cd.center, nd.localexp, nd.rscale, nd.center, carray)
            _downward!(child, data, targets, pot_leaf, grad_leaf, carray)
        end
    end
    return nothing
end

# ---------- index helpers ----------
function _to_local_real(v::AbstractVector, l2g::Vector{Int})
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(real(v[l2g[i]]))  # real channel
    end
    return out
end

function _to_local_imag(v::AbstractVector, l2g::Vector{Int})
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(imag(complex(v[l2g[i]])))
    end
    return out
end

function _local_dipoles_converted(dipstr::AbstractVector, dipvec::AbstractMatrix, l2g, ::Val{:real})
    n = length(l2g)
    out = Vector{ComplexF64}(undef, n)
    @inbounds for i in 1:n
        g = l2g[i]
        s = Float64(real(dipstr[g]))
        out[i] = s * (-complex(dipvec[1, g], dipvec[2, g]))
    end
    return out
end

function _local_dipoles_converted(dipstr::AbstractVector, dipvec::AbstractMatrix, l2g, ::Val{:imag})
    n = length(l2g)
    out = Vector{ComplexF64}(undef, n)
    @inbounds for i in 1:n
        g = l2g[i]
        s = Float64(imag(complex(dipstr[g])))
        out[i] = s * (-complex(dipvec[1, g], dipvec[2, g]))
    end
    return out
end

function _local_dipstr_real(dipstr::AbstractVector, l2g)
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(real(dipstr[l2g[i]]))
    end
    return out
end

function _local_dipstr_imag(dipstr::AbstractVector, l2g)
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(imag(complex(dipstr[l2g[i]])))
    end
    return out
end

function _local_dipvec(dipvec::AbstractMatrix, l2g)
    out = Vector{SVector{2,Float64}}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        g = l2g[i]
        out[i] = SVector{2,Float64}(dipvec[1, g], dipvec[2, g])
    end
    return out
end

"""
    _fmm_real_channel(...) -> (pot_src, grad_src_dz, pot_trg, grad_trg_dz)

Single real-valued density channel. Potentials are `Float64`; gradients are
packed as complex `d/dz` (convert with `dz_to_physical`).
"""
function _fmm_real_channel(
    stree,
    sdata,
    src_local,
    sl2g,
    charges_loc::Union{Nothing,Vector{Float64}},
    dipoles_mp::Union{Nothing,Vector{ComplexF64}},
    dipstr_loc::Union{Nothing,Vector{Float64}},
    dipvec_loc::Union{Nothing,Vector{SVector{2,Float64}}},
    targets_mat,
    pg::Int,
    pgt::Int,
    nterms::Int,
    carray,
    adm,
    thresh::Float64,
    spl,
)
    ns = length(sl2g)
    _zero_multipoles!(sdata)
    _upward!(stree, sdata, src_local, charges_loc, dipoles_mp, carray)

    pot_src = nothing
    grad_src = nothing
    pot_trg = nothing
    grad_trg = nothing

    if pg > 0
        pot_local = zeros(Float64, ns)
        pot_leaf = Dict{UInt,Any}()
        grad_local = pg >= 2 ? zeros(ComplexF64, ns) : nothing
        grad_leaf = pg >= 2 ? Dict{UInt,Any}() : nothing
        for leaf in leaves(stree)
            r = index_range(leaf)
            pot_leaf[objectid(leaf)] = view(pot_local, r)
            grad_leaf !== nothing && (grad_leaf[objectid(leaf)] = view(grad_local, r))
        end
        _zero_locals!(sdata)
        _interact!(
            stree, stree, sdata, sdata, src_local, charges_loc, dipstr_loc, dipvec_loc,
            src_local, pot_leaf, grad_leaf, carray, adm, thresh, true,
        )
        _downward!(stree, sdata, src_local, pot_leaf, grad_leaf, carray)

        pot_src = zeros(Float64, ns)
        @inbounds for i in 1:ns
            pot_src[sl2g[i]] = pot_local[i]
        end
        if pg >= 2
            grad_src = zeros(ComplexF64, ns)  # d/dz in global order
            @inbounds for i in 1:ns
                grad_src[sl2g[i]] = grad_local[i]
            end
        end
    end

    if pgt > 0 && targets_mat !== nothing
        nt = size(targets_mat, 2)
        trg_pts = [SVector{2,Float64}(targets_mat[1, i], targets_mat[2, i]) for i in 1:nt]
        ttree = ClusterTree(trg_pts, spl; copy_elements=false)
        tl2g = loc2glob(ttree)
        trg_local = root_elements(ttree)
        tdata = _allocate_expdata(ttree, nterms)
        _zero_locals!(tdata)

        pot_tlocal = zeros(Float64, nt)
        pot_leaf = Dict{UInt,Any}()
        grad_tlocal = pgt >= 2 ? zeros(ComplexF64, nt) : nothing
        grad_leaf = pgt >= 2 ? Dict{UInt,Any}() : nothing
        for leaf in leaves(ttree)
            r = index_range(leaf)
            pot_leaf[objectid(leaf)] = view(pot_tlocal, r)
            grad_leaf !== nothing && (grad_leaf[objectid(leaf)] = view(grad_tlocal, r))
        end

        _interact!(
            ttree, stree, tdata, sdata, src_local, charges_loc, dipstr_loc, dipvec_loc,
            trg_local, pot_leaf, grad_leaf, carray, adm, thresh, false,
        )
        _downward!(ttree, tdata, trg_local, pot_leaf, grad_leaf, carray)

        pot_trg = zeros(Float64, nt)
        @inbounds for i in 1:nt
            pot_trg[tl2g[i]] = pot_tlocal[i]
        end
        if pgt >= 2
            grad_trg = zeros(ComplexF64, nt)
            @inbounds for i in 1:nt
                grad_trg[tl2g[i]] = grad_tlocal[i]
            end
        end
    end

    return pot_src, grad_src, pot_trg, grad_trg
end

function _pack_grad(gdz::Vector{ComplexF64})
    n = length(gdz)
    g = zeros(Float64, 2, n)
    @inbounds for i in 1:n
        gx, gy = dz_to_physical(gdz[i])
        # For real potential, gradient components are Re(φ_x)-style:
        # ∂x u = Re(φ'), ∂y u = -Im(φ') when u = Re(φ)
        g[1, i] = gx
        g[2, i] = gy
    end
    return g
end

function _densities_are_complex(charges, dipstr)
    if charges !== nothing
        eltype(charges) <: Complex && return true
        any(x -> imag(complex(x)) != 0, charges) && return true
    end
    if dipstr !== nothing
        eltype(dipstr) <: Complex && return true
        any(x -> imag(complex(x)) != 0, dipstr) && return true
    end
    return false
end

function _fmm2d_laplace(
    eps::Float64,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Int=0,
    pgt::Int=0,
    nmax::Int=50,
    η::Float64=1.0,
    splitter=nothing,
)
    @assert size(sources, 1) == 2 "sources must be 2×n"
    ns = size(sources, 2)
    has_charge = charges !== nothing
    has_dipole = dipstr !== nothing
    @assert has_charge || has_dipole "provide charges and/or dipoles"
    @assert pg > 0 || pgt > 0 "set pg and/or pgt to 1 or 2"
    if has_dipole
        @assert dipvec !== nothing "dipvec required with dipstr"
        @assert size(dipvec, 1) == 2 && size(dipvec, 2) == ns
    end
    if targets !== nothing
        @assert size(targets, 1) == 2
    end

    nterms = laplace_nterms(eps)
    carray = binomial_table(2 * nterms + 2)
    adm = StrongAdmissibility(η=η)
    spl = splitter === nothing ? GeometricSplitter(nmax=nmax) : splitter

    src_pts = [SVector{2,Float64}(sources[1, i], sources[2, i]) for i in 1:ns]
    stree = ClusterTree(src_pts, spl; copy_elements=false)
    sl2g = loc2glob(stree)
    src_local = root_elements(stree)
    sdata = _allocate_expdata(stree, nterms)

    L = diameter(container(stree))
    thresh = L * Base.eps(Float64)

    chg_vec = has_charge ? (charges isa AbstractMatrix ? vec(charges) : charges) : nothing
    dstr_vec = has_dipole ? (dipstr isa AbstractMatrix ? vec(dipstr) : dipstr) : nothing
    dipvec_loc = has_dipole ? _local_dipvec(dipvec, sl2g) : nothing

    complex_dens = _densities_are_complex(chg_vec, dstr_vec)

    # --- real channel ---
    ch_r = has_charge ? _to_local_real(chg_vec, sl2g) : nothing
    dp_r = has_dipole ? _local_dipoles_converted(dstr_vec, dipvec, sl2g, Val(:real)) : nothing
    ds_r = has_dipole ? _local_dipstr_real(dstr_vec, sl2g) : nothing

    pot_r, gdz_r, pt_r, gt_r = _fmm_real_channel(
        stree, sdata, src_local, sl2g, ch_r, dp_r, ds_r, dipvec_loc,
        targets, pg, pgt, nterms, carray, adm, thresh, spl,
    )

    if !complex_dens
        pot_src = pot_r === nothing ? nothing : complex.(pot_r)
        grad_src = gdz_r === nothing ? nothing : complex.(_pack_grad(gdz_r))
        pot_trg = pt_r === nothing ? nothing : complex.(pt_r)
        grad_trg = gt_r === nothing ? nothing : complex.(_pack_grad(gt_r))
        return pot_src, grad_src, pot_trg, grad_trg
    end

    # --- imag channel ---
    ch_i = has_charge ? _to_local_imag(chg_vec, sl2g) : nothing
    dp_i = has_dipole ? _local_dipoles_converted(dstr_vec, dipvec, sl2g, Val(:imag)) : nothing
    ds_i = has_dipole ? _local_dipstr_imag(dstr_vec, sl2g) : nothing

    pot_i, gdz_i, pt_i, gt_i = _fmm_real_channel(
        stree, sdata, src_local, sl2g, ch_i, dp_i, ds_i, dipvec_loc,
        targets, pg, pgt, nterms, carray, adm, thresh, spl,
    )

    pot_src =
        pot_r === nothing ? nothing : (complex.(pot_r) .+ im .* pot_i)
    pot_trg =
        pt_r === nothing ? nothing : (complex.(pt_r) .+ im .* pt_i)

    grad_src = nothing
    if gdz_r !== nothing
        gr = _pack_grad(gdz_r)
        gi = _pack_grad(gdz_i)
        grad_src = complex.(gr) .+ im .* gi
    end
    grad_trg = nothing
    if gt_r !== nothing
        gr = _pack_grad(gt_r)
        gi = _pack_grad(gt_i)
        grad_trg = complex.(gr) .+ im .* gi
    end

    return pot_src, grad_src, pot_trg, grad_trg
end

# =============================================================================
# Public API
# =============================================================================

"""
```julia
vals = lfmm2d(eps, sources; charges=nothing, dipstr=nothing, dipvec=nothing,
              targets=nothing, pg=0, pgt=0, nmax=50, η=1.0)
```

Pure-Julia 2D Laplace FMM with complex densities (Flatiron `lfmm2d` kernel):

```math
u(x) = \\sum_j c_j \\log\\|x-x_j\\|
     + d_j\\, v_j\\cdot\\nabla_{x_j}\\log\\|x-x_j\\|
```
"""
function lfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=50,
    η::Real=1.0,
    splitter=nothing,
)
    vals = FMMVals()
    pot, grad, pottarg, gradtarg = _fmm2d_laplace(
        Float64(eps),
        sources;
        charges=charges,
        dipstr=dipstr,
        dipvec=dipvec,
        targets=targets,
        pg=Int(pg),
        pgt=Int(pgt),
        nmax=Int(nmax),
        η=Float64(η),
        splitter=splitter,
    )
    vals.pot = pot
    vals.grad = grad
    vals.pottarg = pottarg
    vals.gradtarg = gradtarg
    vals.ier = 0
    return vals
end

"""
```julia
vals = rfmm2d(eps, sources; charges=nothing, dipstr=nothing, dipvec=nothing,
              targets=nothing, pg=0, pgt=0, nmax=50, η=1.0)
```

Real-valued 2D Laplace FMM.
"""
function rfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=50,
    η::Real=1.0,
    splitter=nothing,
)
    vals = lfmm2d(
        eps, sources;
        charges=charges, dipstr=dipstr, dipvec=dipvec, targets=targets,
        pg=pg, pgt=pgt, nmax=nmax, η=η, splitter=splitter,
    )
    vals.pot !== nothing && (vals.pot = real.(vals.pot))
    vals.grad !== nothing && (vals.grad = real.(vals.grad))
    vals.pottarg !== nothing && (vals.pottarg = real.(vals.pottarg))
    vals.gradtarg !== nothing && (vals.gradtarg = real.(vals.gradtarg))
    return vals
end

"""
```julia
vals = l2ddir(sources, targets; charges=nothing, dipstr=nothing, dipvec=nothing,
              pgt=1, thresh=0.0)
```

Direct ``O(N^2)`` complex Laplace evaluation (sources → targets only).
"""
function l2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    nt = size(targets, 2)
    vals = FMMVals()
    pot = zeros(ComplexF64, nt)
    grad = pgt >= 2 ? zeros(ComplexF64, 2, nt) : nothing
    direct_laplace!(pot, grad, sources, charges, dipstr, dipvec, targets; thresh=thresh)
    vals.pottarg = pot
    vals.gradtarg = grad
    vals.ier = 0
    return vals
end

"""Direct ``O(N^2)`` real Laplace evaluation."""
function r2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    vals = l2ddir(
        sources, targets; charges=charges, dipstr=dipstr, dipvec=dipvec, pgt=pgt, thresh=thresh
    )
    vals.pottarg = real.(vals.pottarg)
    vals.gradtarg !== nothing && (vals.gradtarg = real.(vals.gradtarg))
    return vals
end
