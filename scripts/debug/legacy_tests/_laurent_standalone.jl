# Standalone verification of Laurent singular integration (no full BEM load)
using Test
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using Richardson
using Richardson: extrapolate

include(joinpath(@__DIR__, "..", "src", "Core", "Interpolation.jl"))

function divide(qsi, w, a)
    if -1 < a < 1
        J1 = (1 - a) / 2
        J2 = (1 + a) / 2
        qsi1 = qsi * J1 .+ (a + 1) / 2
        qsi2 = qsi * J2 .+ (a - 1) / 2
        w1 = w * J1
        w2 = w * J2
        return SVector{2 * length(qsi)}([qsi2; qsi1]), SVector{2 * length(w)}([w2; w1])
    end
    return SVector{length(qsi)}(qsi), SVector{length(w)}(w)
end

function laurent_coefficients(f, h, order::Integer; kwargs...)
    return laurent_coefficients(f, h, Val(Int(order)); kwargs...)
end

function _kw(kwargs)
    defaults = (; contract=1 / 2, atol=1.0e-12, rtol=1.0e-10)
    return (; defaults..., kwargs...)
end

function laurent_coefficients(f, h, ::Val{-2}; kwargs...)
    kw = _kw(kwargs)
    f2, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^2 * f(x)
    end
    f1, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x) - f2 / x
    end
    f0, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f2 / x^2 - f1 / x
    end
    return f2, f1, f0
end

function laurent_coefficients(f, h, ::Val{-1}; kwargs...)
    kw = _kw(kwargs)
    f1, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x)
    end
    f0, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f1 / x
    end
    z = zero(f0)
    return z, f1, f0
end

function laurent_coefficients(f, h, ::Val{0}; kwargs...)
    kw = _kw(kwargs)
    f0, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x)
    end
    z = zero(f0)
    return z, z, f0
end

function _laurent_shape_analytic(poly, a::T, s::T, order::Integer) where {T<:Real}
    Na, dNa = shapefun(poly, a)
    Na = vec(Na)
    dNa = vec(dNa)
    nN = length(Na)
    d2Na = vec(reshape(dNa, 1, nN) * poly.Dmat)
    F2 = zeros(T, nN)
    F1 = zeros(T, nN)
    F0 = zeros(T, nN)
    if order == 0
        F0 .= Na
    elseif order == 1
        invs = inv(s)
        @inbounds for j in 1:nN
            F1[j] = Na[j] * invs
            F0[j] = dNa[j]
        end
    else
        invs = inv(s)
        invs2 = invs * invs
        @inbounds for j in 1:nN
            F2[j] = Na[j] * invs2
            F1[j] = dNa[j] * invs
            F0[j] = d2Na[j] / 2
        end
    end
    return F2, F1, F0
end

function _laurent_shape_richardson(poly, a::T, s::T, order::Integer, h::T; kwargs...) where {T<:Real}
    nN = length(poly.nodes)
    lord = order == 0 ? Val(0) : order == 1 ? Val(-1) : Val(-2)
    F2 = zeros(T, nN)
    F1 = zeros(T, nN)
    F0 = zeros(T, nN)
    @inbounds for j in 1:nN
        fj = ρ -> begin
            ξ = a + s * ρ
            Nj = shapefun(poly, ξ)[1][j]
            if order == 0
                Nj * log(ρ)
            elseif order == 1
                Nj / (s * ρ)
            else
                Nj / (s * ρ)^2
            end
        end
        a2, a1, a0 = laurent_coefficients(fj, h, lord; kwargs...)
        F2[j] = a2
        F1[j] = a1
        F0[j] = a0
    end
    return F2, F1, F0
end

function laurent_shape_coefficients(poly, a, s, order; method=:auto, h=1e-3, kwargs...)
    s = float(s)
    meth = method === :auto ? :analytic : method
    if meth === :analytic
        return _laurent_shape_analytic(poly, float(a), s, order)
    elseif meth === :richardson
        return _laurent_shape_richardson(poly, float(a), s, order, float(h); kwargs...)
    else
        error("bad method $method")
    end
end

function singular(qsi, w, order::Integer=0, eet::Real=0.0; poly=nothing)
    ngp = length(qsi)
    T = float(eltype(w))
    p = poly === nothing ? Legendre(ngp - 1) : poly
    nN = length(p.nodes)
    a = T(eet)
    if a <= -1 || a >= 1
        a = clamp(a, nextfloat(T(-1)), prevfloat(T(1)))
    end
    Na, dNa = shapefun(p, a)
    Na = vec(Na)
    dNa = vec(dNa)
    q2, w2 = divide(qsi, w, a)
    N2, _ = shapefun(p, q2)
    n2 = length(q2)
    I = zeros(T, nN)
    if order == 0
        cte = (1 - a) * log(abs(1 - a)) + (1 + a) * log(abs(1 + a)) - T(2)
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = abs(q2[i] - a)
                δ < eps(T) && continue
                s += w2[i] * (N2[i, j] - Na[j]) * log(δ)
            end
            I[j] = s + Na[j] * cte
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            Kj = log(abs(qsi[j] - a))
            wn[j] = abs(Kj) > eps(T) ? I[j] / Kj : zero(T)
        end
        return wn
    elseif order == 1
        cte = log(abs((1 - a) / (1 + a)))
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = q2[i] - a
                abs(δ) < eps(T) && continue
                s += w2[i] * (N2[i, j] - Na[j]) / δ
            end
            I[j] = s + Na[j] * cte
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)
        end
        return wn
    else
        cte = -T(2) / (1 - a^2)
        ctel = log(abs((1 - a) / (1 + a)))
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = q2[i] - a
                abs(δ) < eps(T) && continue
                s += w2[i] * (N2[i, j] - Na[j] - dNa[j] * δ) / δ^2
            end
            I[j] = s + Na[j] * cte + dNa[j] * ctel
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)^2
        end
        return wn
    end
end

function _guiggiani_line_log!(I, poly, a::T, qsi, w) where {T}
    nN = length(poly.nodes)
    Na, _ = shapefun(poly, a)
    Na = vec(Na)
    q2, w2 = divide(qsi, w, a)
    N2, _ = shapefun(poly, q2)
    n2 = length(q2)
    cte = (1 - a) * log(abs(1 - a)) + (1 + a) * log(abs(1 + a)) - T(2)
    @inbounds for j in 1:nN
        s = zero(T)
        for i in 1:n2
            δ = abs(q2[i] - a)
            δ < eps(T) && continue
            s += w2[i] * (N2[i, j] - Na[j]) * log(δ)
        end
        I[j] = s + Na[j] * cte
    end
    return I
end

function guiggiani_line(poly::AbstractPolynomial, a::Real, order::Integer;
                        qsi=nothing, w=nothing, method=:auto, h=1e-3, kwargs...)
    T = Float64
    a = T(a)
    nN = length(poly.nodes)
    if qsi === nothing
        qsi, w = gausslegendre(max(2nN, 12))
    end
    qsi = collect(T, qsi)
    w = collect(T, w)
    I = zeros(T, nN)
    order == 0 && return _guiggiani_line_log!(I, poly, a, qsi, w)
    for (s, ρmax) in ((T(-1), a - T(-1)), (T(1), T(1) - a))
        ρmax ≤ eps(T) && continue
        F2, F1, _ = laurent_shape_coefficients(poly, a, s, order; method=method, h=h, kwargs...)
        @inbounds for i in eachindex(qsi)
            u = qsi[i]
            ρ = (u + 1) * ρmax / 2
            ρ ≤ eps(T) && continue
            dρ = w[i] * ρmax / 2
            ξ = a + s * ρ
            Nξ = vec(shapefun(poly, ξ)[1])
            for j in 1:nN
                Fval = order == 1 ? Nξ[j] / (s * ρ) : Nξ[j] / (s * ρ)^2
                Freg = order == 2 ? (Fval - F2[j] / ρ^2 - F1[j] / ρ) : (Fval - F1[j] / ρ)
                I[j] += Freg * dρ
            end
        end
        @inbounds for j in 1:nN
            I[j] += order == 2 ? (F1[j] * log(ρmax) - F2[j] / ρmax) : (F1[j] * log(ρmax))
        end
    end
    return I
end

function singular_laurent(qsi, w, order::Integer=0, eet::Real=0.0;
                          poly=nothing, method=:auto, kwargs...)
    ngp = length(qsi)
    T = float(eltype(w))
    p = poly === nothing ? Legendre(ngp - 1) : poly
    nN = length(p.nodes)
    a = T(eet)
    if a <= -1 || a >= 1
        a = clamp(a, nextfloat(T(-1)), prevfloat(T(1)))
    end
    I = guiggiani_line(p, a, order; qsi=qsi, w=w, method=method, kwargs...)
    wn = similar(I)
    if order == 0
        @inbounds for j in 1:nN
            Kj = log(abs(qsi[j] - a))
            wn[j] = abs(Kj) > eps(T) ? I[j] / Kj : zero(T)
        end
    elseif order == 1
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)
        end
    else
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)^2
        end
    end
    return wn
end

function guiggiani_integral(f, a::Real, order::Integer;
                        qsi=nothing, w=nothing, h=1e-3, kwargs...)
    T = Float64
    a = T(a)
    if qsi === nothing
        qsi, w = gausslegendre(16)
    end
    qsi = collect(T, qsi)
    w = collect(T, w)
    lord = Val(Int(order))
    probe = f(a + 1e-6)
    acc = zero(probe)
    for (s, ρmax) in ((T(-1), a - T(-1)), (T(1), T(1) - a))
        ρmax ≤ eps(T) && continue
        Fr = ρ -> f(a + s * ρ)
        F2, F1, _ = laurent_coefficients(Fr, T(h), lord; kwargs...)
        Iρ = zero(probe)
        @inbounds for i in eachindex(qsi)
            u = qsi[i]
            ρ = (u + 1) * ρmax / 2
            ρ ≤ eps(T) && continue
            dρ = w[i] * ρmax / 2
            Fval = Fr(ρ)
            Freg = if order == -2 || order == 2
                Fval - F2 / ρ^2 - F1 / ρ
            elseif order == -1 || order == 1
                Fval - F1 / ρ
            else
                Fval
            end
            Iρ += Freg * dρ
        end
        if order == -2 || order == 2
            acc += Iρ + F1 * log(ρmax) - F2 / ρmax
        elseif order == -1 || order == 1
            acc += Iρ + F1 * log(ρmax)
        else
            acc += Iρ
        end
    end
    return acc
end

@testset "laurent_coefficients" begin
    f = ρ -> ρ^2 + 2ρ + 1
    f2, f1, f0 = laurent_coefficients(f, 1e-2, Val(-2))
    @test norm((f2, f1, f0) .- (0, 0, 1)) < 1e-10

    f = ρ -> cos(ρ) / ρ^2 + exp(ρ) / ρ + exp(ρ)
    f2, f1, f0 = laurent_coefficients(f, 1.0, Val(-2); atol=1e-12, breaktol=2, contract=1 / 2)
    @test norm((f2, f1, f0) .- (1, 1, 1.5)) < 1e-9

    f = ρ -> SVector(cos(ρ), sin(ρ)) / ρ^2 + SVector(exp(ρ), 0.2) / ρ
    f2, f1, f0 = laurent_coefficients(f, 1e-1, Val(-2))
    @test f2 ≈ SVector(1.0, 0.0)
    @test f1 ≈ SVector(1.0, 1.2)
end

@testset "analytic vs richardson shape" begin
    poly = Legendre(2)
    a = 0.3
    for s in (-1.0, 1.0), order in (1, 2)
        Fa = laurent_shape_coefficients(poly, a, s, order; method=:analytic)
        Fr = laurent_shape_coefficients(poly, a, s, order; method=:richardson, h=1e-3)
        @test Fa[1] ≈ Fr[1] rtol = 1e-8 atol = 1e-10
        @test Fa[2] ≈ Fr[2] rtol = 1e-8 atol = 1e-10
        @test Fa[3] ≈ Fr[3] rtol = 1e-5 atol = 1e-7
    end
end

@testset "singular_laurent == singular" begin
    for deg in (1, 2, 3)
        poly = Legendre(deg)
        qsi, w = gausslegendre(deg + 1)
        for a in (-0.4, 0.0, 0.55), order in (0, 1, 2)
            wn0 = singular(qsi, w, order, a; poly=poly)
            wnL = singular_laurent(qsi, w, order, a; poly=poly, method=:analytic)
            @test wn0 ≈ wnL rtol = 1e-10 atol = 1e-12
        end
    end
end

@testset "richardson singular_laurent" begin
    poly = Legendre(2)
    qsi, w = gausslegendre(3)
    a = 0.2
    for order in (1, 2)
        wnA = singular_laurent(qsi, w, order, a; poly=poly, method=:analytic)
        wnR = singular_laurent(qsi, w, order, a; poly=poly, method=:richardson, h=5e-4)
        @test wnA ≈ wnR rtol = 1e-6 atol = 1e-8
    end
end
@testset "guiggiani CPV/HFP" begin
    qsi, w = gausslegendre(24)
    for s in (-0.4, 0.1, 0.7)
        exact = log(abs((1 - s) / (-1 - s)))
        I = guiggiani_integral(ξ -> 1 / (ξ - s), s, -1; qsi=qsi, w=w, h=1e-3)
        @test abs(I - exact) < 1e-8
    end
    qsi, w = gausslegendre(32)
    for a in (-0.3, 0.0, 0.5)
        exact = -2 / (1 - a^2)
        I = guiggiani_integral(ξ -> 1 / (ξ - a)^2, a, -2; qsi=qsi, w=w, h=1e-3)
        @test abs(I - exact) < 1e-7
    end
end

println("ALL OK")
