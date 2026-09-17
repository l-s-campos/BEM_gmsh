"""
    WheelRail

Planar wheel–rail contact (Vollebregt CONTACT module 1, D=2) in front of the
Pohrt–Uzawa rolling solver.

Profiles (SIMPACK `.prr/.prw`, slice catalogues `.slcw`) are placed with the
CONTACT track/wheelset markers, a prismatic contact locus is built, the
undeformed distance `h = n_rail − n_wheel` is sampled on a planar grid, and
each patch is solved with [`solve_rolling_step!`](@ref). N=1 iterates `z_ws`
until the track vertical force matches `F_z`.

Units: N, mm, MPa. Left/right use CONTACT's internal right-rail frame
(`sgn = ±1`).

Reference: E.A.H. Vollebregt, CONTACT (Apache-2.0, rev 2781).
"""
module WheelRail

using LinearAlgebra
using Printf
using ..ContactHalfSpace
using ..OrthotropicUzawa
using ..RollingContact

export Marker, WRProfile, TrackGeom, WheelsetGeom, ContactPatch, PatchResult, WRResult
export ox, oy, oz, vec_2glob, vec_2loc, marker_2glob, marker_2loc
export read_rail_profile, read_wheel_profile, is_varprof, profile_at_theta
export set_rail_marker, set_wheel_markers, gauge_meas_pt
export locate_patches, undeformed_distance, creepage_at_patch
export rigid_gap_min, rigid_gap_1d
export solve_wheel_rail, match_fz!
export vollebregt_data_dir

include("WheelRail/Markers.jl")
include("WheelRail/Profiles.jl")
include("WheelRail/Geometry.jl")
include("WheelRail/Locate.jl")
include("WheelRail/Undeformed.jl")
include("WheelRail/Creepage.jl")
include("WheelRail/Solve.jl")

function vollebregt_data_dir()
    d = normpath(joinpath(@__DIR__, "..", "..", "data", "contact", "vollebregt"))
    isdir(d) && return d
    env = get(ENV, "CONTACT_EXAMPLES", "")
    isempty(env) && error("Vollebregt profile data not found at $d; set CONTACT_EXAMPLES")
    return env
end

end # module
