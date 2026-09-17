using DrWatson
@quickactivate :BEM
using BEM.Topology

println("loaded BEM")
d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
println("area = ", design_area(d))
dad = bemdata_from_loops(d)
println("n=", dad.n, " ni=", dad.ni, " nelem=", length(dad.elements))
H_G_full_direct(dad; npg=8, threaded=false)
solve(dad)
println("T finite ", all(isfinite, dad.T), " q finite ", all(isfinite, dad.q))
DTb, DTi = topological_derivative(dad)
println("DTb med=", median(DTb), " DTi med=", isempty(DTi) ? NaN : median(DTi))
println("J=", thermal_conductance(dad))
