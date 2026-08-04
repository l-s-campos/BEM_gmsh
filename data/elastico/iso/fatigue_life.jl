# Fretting fatigue life — material sets and baseline loads
# include(datadir("elastico", "iso", "fatigue_life.jl"))

aluminium = (name="Aluminium", E=73_400.0, ν=0.33)   # MPa
titanium = (name="Titanium", E=116_000.0, ν=0.32)

materials = (aluminium, titanium)

# Baseline fretting loads
R = 70.0
P = 100.0
Q = 15.0
B = 15.0
f = 0.3

life_methods = (:SWT, :FS, :DMT)
