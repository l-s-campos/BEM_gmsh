# FastMultipole gravitational point-source wrapper (from FastMultipole.jl test/gravitational.jl).
# Names are qualified so this can load next to BEM.
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra

struct GravBody{TF}
    position::SVector{3,TF}
    radius::TF
    strength::TF
end

struct GravSystem{TF}
    bodies::Vector{GravBody{TF}}
    potential::Matrix{TF}
end

function GravSystem(bodies::Matrix)
    nbodies = size(bodies, 2)
    bodies2 = [GravBody(SVector{3}(bodies[1:3, i]), bodies[4, i], bodies[5, i]) for i in 1:nbodies]
    potential = zeros(eltype(bodies), 16, nbodies)
    return GravSystem(bodies2, potential)
end

Base.eltype(::GravSystem{TF}) where {TF} = TF

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::GravSystem, i_body)
    x, y, z = system.bodies[i_body].position
    buffer[1, i_buffer] = x
    buffer[2, i_buffer] = y
    buffer[3, i_buffer] = z
    buffer[4, i_buffer] = system.bodies[i_body].radius
    buffer[5, i_buffer] = system.bodies[i_body].strength
end

FastMultipole.data_per_body(::GravSystem) = 5
FastMultipole.get_position(g::GravSystem, i) = g.bodies[i].position
FastMultipole.strength_dims(::GravSystem) = 1
FastMultipole.get_n_bodies(g::GravSystem) = length(g.bodies)

function FastMultipole.body_to_multipole!(system::GravSystem, args...)
    FastMultipole.body_to_multipole!(FastMultipole.Point{FastMultipole.Source}, system, args...; scale_strength=-1.0)
end

FastMultipole.has_vector_potential(::GravSystem) = false
FastMultipole.metadata_per_body(::GravSystem) = 2
FastMultipole.previous_potential_metadata_index(::GravSystem) = 1
FastMultipole.previous_gradient_metadata_index(::GravSystem) = 2

function FastMultipole.metadata_to_buffer!(buffer, switch, i_buffer, system::GravSystem, i_body)
    previous_potential = system.potential[1, i_body]
    previous_gradient = hypot(system.potential[5, i_body], system.potential[6, i_body], system.potential[7, i_body])
    buffer[FastMultipole.metadata_index(switch, 1), i_buffer] = previous_potential
    buffer[FastMultipole.metadata_index(switch, 2), i_buffer] = previous_gradient
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
        source_system::GravSystem, source_buffer, source_index) where {PS,GS,HS}
    @inbounds for j_target in target_index
        target_x, target_y, target_z = FastMultipole.get_position(target_system, j_target)
        dϕ = zero(eltype(target_system))
        d∇ϕ = zero(SVector{3,eltype(target_system)})
        @inbounds for i_source in source_index
            source_x, source_y, source_z = FastMultipole.get_position(source_buffer, i_source)
            source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > 0
                r = sqrt(r2)
                tmp = source_strength / r * FastMultipole.ONE_OVER_4π
                if PS
                    dϕ += tmp
                end
                if GS
                    d∇ϕ -= SVector{3}(dx, dy, dz) * tmp / r2
                end
            end
        end
        PS && FastMultipole.set_scalar_potential!(target_system, derivatives_switch, j_target, dϕ)
        GS && FastMultipole.set_gradient!(target_system, derivatives_switch, j_target, d∇ϕ)
    end
end

function FastMultipole.buffer_to_target_system!(target_system::GravSystem, i_target,
        derivatives_switch::FastMultipole.DerivativesSwitch{PS,GS,HS}, target_buffer, i_buffer) where {PS,GS,HS}
    TF = eltype(target_buffer)
    scalar_potential = PS ? FastMultipole.get_scalar_potential(target_buffer, derivatives_switch, i_buffer) : zero(TF)
    gradient = GS ? FastMultipole.get_gradient(target_buffer, derivatives_switch, i_buffer) : zero(SVector{3,TF})
    hessian = HS ? FastMultipole.get_hessian(target_buffer, derivatives_switch, i_buffer) : zero(SMatrix{3,3,TF,9})
    target_system.potential[1, i_target] = scalar_potential
    target_system.potential[5:7, i_target] .= gradient
    for (jj, j) in enumerate(8:16)
        target_system.potential[j, i_target] = hessian[jj]
    end
end
