# Multi-region interfaces and frictional contact on BEMdata.
# Not reexported by `using BEM` — use `BEM.MultiRegion` or `using BEM.MultiRegion`.
"""
    BEM.MultiRegion

Subregion coupling (perfect interface, type 3) and multibody frictional
contact (type 4): active-set, SSN, projected Newton, GNM-ls (2010).
"""
module MultiRegion

using LinearAlgebra
using StaticArrays
using Statistics
using SparseArrays
using Printf

let P = parentmodule(@__MODULE__)
    for n in names(P; all=true)
        s = String(n)
        (startswith(s, "#") || n === :eval || n === :include) && continue
        isdefined(P, n) || continue
        try
            Core.eval(@__MODULE__, Expr(:import, Expr(:., :., :., n)))
        catch
        end
    end
end

include("SubRegions.jl")

end
