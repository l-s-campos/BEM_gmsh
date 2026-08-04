# Cattaneo–Mindlin frictional contact of two cylinders (half-space path)
# include(datadir("elastico", "iso", "cattaneo_mindlin.jl"))

R = 70.0             # mm cylinder radius
w = 6.5              # mm geometric scale
E = 73_400.0         # MPa
ν = 0.33
P = 100.0            # N/mm normal load
f = 0.3              # friction
plane_strain = true

# Two identical bodies, plane strain contact modulus
E_eq = E / (2 * (1 - ν^2))
R_eq = R / 2
a = sqrt(4 * P * R_eq / (π * E_eq))    # Hertz half-width
p0 = 2 * P / (π * a)                   # peak pressure

# Stick half-width c = a √(1 − |Q|/(f P))
cattaneo_c(Q) = a * sqrt(max(0.0, 1 - abs(Q) / (f * P)))

Qmax = f * P * 0.5

# Cyclic tangential steps (normal held at P): A→B→C→D→E
load_steps = (
    (name=:A, P=P, Q=0.0),
    (name=:B, P=P, Q=+Qmax),
    (name=:C, P=P, Q=0.0),
    (name=:D, P=P, Q=-Qmax),
    (name=:E, P=P, Q=0.0),
)

NPc_list = (21, 41, 61, 81, 101)
