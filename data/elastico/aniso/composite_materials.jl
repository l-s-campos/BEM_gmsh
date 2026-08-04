# Orthotropic / composite lamina constants used in anisotropic BEM examples
# include(datadir("elastico", "aniso", "composite_materials.jl"))
# Units as in source tables (GPa unless noted). ν12 = ν_LT.

# Graphite / epoxy (Wang; Qin; Lei 2017)
graphite_epoxy = (name="graphite_epoxy", E1=181.0, E2=10.3, G12=7.17, ν12=0.28)

# Boron / epoxy B4/5505 (Tsai; Hahn)
boron_epoxy = (name="boron_epoxy", E1=204.0, E2=18.8, G12=5.59, ν12=0.23)

# Thornel 300 / Narmco 5208 (often in ksi in older refs — converted if needed)
thornel_narmco = (name="thornel_narmco", E1=21.4, E2=1.6, G12=0.77, ν12=0.29, unit="ksi")

# T300 / Narmco 5208 graphite-epoxy (GN/m² = GPa)
t300_narmco = (name="t300_narmco", E1=141.0, E2=9.44, G12=5.18, ν12=0.31)

# Boron / aluminum B4/6061-Al
boron_aluminum = (name="boron_aluminum", E1=235.0, E2=137.0, G12=47.0, ν12=0.30)

# Carbon fabric / phenolic
carbon_phenolic = (name="carbon_phenolic", E1=20.0, E2=19.0, G12=6.8, ν12=0.23)

# Silicon carbide / aluminum
sic_aluminum = (name="sic_aluminum", E1=204.0, E2=118.0, G12=41.0, ν12=0.27)

# Carbon / epoxy (alternate listing)
carbon_epoxy = (name="carbon_epoxy", E1=20.0, E2=19.0, G12=6.8, ν12=0.23)

materials = (
    graphite_epoxy,
    boron_epoxy,
    thornel_narmco,
    t300_narmco,
    boron_aluminum,
    carbon_phenolic,
    sic_aluminum,
    carbon_epoxy,
)

# Build Lekhnitskii params:
# using BEM
# p = lekhnitskii_params(m.E1, m.E2, m.G12, m.ν12; θ_deg=0)
# props = AnisotropicElasticity(p)
