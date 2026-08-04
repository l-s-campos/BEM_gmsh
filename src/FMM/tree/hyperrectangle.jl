"""
    struct HyperRectangle{N,T}

Axis-aligned hyperrectangle in `N` dimensions given by `low_corner` and
`high_corner`.
"""
struct HyperRectangle{N,T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
end
HyperRectangle(l::Tuple, h::Tuple) = HyperRectangle(SVector(l), SVector(h))
HyperRectangle(l::SVector, h::SVector) = HyperRectangle(promote(l, h)...)
HyperRectangle(a::Number, b::Number) = HyperRectangle(SVector(a), SVector(b))

low_corner(r::HyperRectangle) = r.low_corner
high_corner(r::HyperRectangle) = r.high_corner

function distance(rec1::HyperRectangle{N}, rec2::HyperRectangle{N}) where {N}
    d2 = zero(eltype(rec1.low_corner))
    for i in 1:N
        d2 +=
            max(zero(d2), rec1.low_corner[i] - rec2.high_corner[i])^2 +
            max(zero(d2), rec2.low_corner[i] - rec1.high_corner[i])^2
    end
    return sqrt(d2)
end

diameter(r::HyperRectangle) = norm(high_corner(r) .- low_corner(r), 2)
radius(r::HyperRectangle) = diameter(r) / 2
center(r::HyperRectangle) = (low_corner(r) + high_corner(r)) / 2

Base.in(point, h::HyperRectangle) = all(low_corner(h) .<= point .<= high_corner(h))

function bounding_box(els, cube=false)
    isempty(els) && error("data cannot be empty")
    lb = center(first(els))
    ub = center(first(els))
    for el in els
        pt = center(el)
        lb = min.(lb, pt)
        ub = max.(ub, pt)
    end
    if cube
        w = maximum(ub - lb)
        xc = (ub + lb) / 2
        lb = min.(xc .- w / 2, lb)
        ub = max.(xc .+ w / 2, ub)
    end
    lb == ub && (lb = prevfloat.(lb); ub = nextfloat.(ub))
    return HyperRectangle(lb, ub)
end

center(x::SVector) = x
center(x::NTuple) = SVector(x)

function Base.split(rec::HyperRectangle{N}, axis, place) where {N}
    rec_low = low_corner(rec)
    rec_high = high_corner(rec)
    high1 = SVector(ntuple(n -> n == axis ? place : rec_high[n], N))
    low2 = SVector(ntuple(n -> n == axis ? place : rec_low[n], N))
    return HyperRectangle(rec_low, high1), HyperRectangle(low2, rec_high)
end
function Base.split(rec::HyperRectangle, axis)
    place = (high_corner(rec)[axis] + low_corner(rec)[axis]) / 2
    return split(rec, axis, place)
end
function Base.split(rec::HyperRectangle)
    axis = argmax(high_corner(rec) .- low_corner(rec))
    return split(rec, axis)
end
