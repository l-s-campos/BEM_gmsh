# Laplace centre-crack meshes (dual BEM twins + finite-width slit).
# Prefer the package API:
#   dad = dual_laplace_problem(; field=:y, bc=:insulated)
#   dad = finite_width_laplace_problem(; gap=0.05, field=:y)

using BEM.Crack: mesh_center_crack_laplace, mesh_finite_width_crack,
    dual_laplace_problem, finite_width_laplace_problem
