# Fretting contact with bulk tension (cylindrical pad on flat specimen)
# include(datadir("elastico", "iso", "bulk_fretting.jl"))

R = 70.0             # mm pad radius
w = 6.5              # mm
E = 73_400.0         # MPa
ν = 0.33
f = 0.3
P = 100.0            # N/mm normal
Q = 15.0             # N/mm tangential (in phase with bulk)
B = 15.0             # N/mm bulk tension
plane_strain = true

E_eq = E / (2 * (1 - ν^2))
R_eq = R / 2
a = sqrt(4 * P * R_eq / (π * E_eq))
p0 = 2 * P / (π * a)
cattaneo_c(Qval=Q) = a * sqrt(max(0.0, 1 - abs(Qval) / (f * P)))

NPc_list = (21, 41, 61, 81, 101)
