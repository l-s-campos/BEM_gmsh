export DIBEM

function DIBEM(dad::BEMdata{<:Laplace}; rbf=PHS())

    npoly = binomial(dad.dimension + rbf.poly_deg, rbf.poly_deg)
    mon = MonomialBasis(dad.dimension, rbf.poly_deg)
    F = zeros(dad.nt, dad.nt)
    D = zeros(dad.nt, dad.nt)
    IF = zeros(dad.nt)
    ID = zeros(dad.nt)
    P = zeros(npoly, dad.nt)
    IP = zeros(npoly)
    k = dad.properties.k


    @showprogress "Assembling F and D" for j in 1:dad.nt, i in 1:dad.nt
        x = i <= dad.n ? dad.Nodes[i] : dad.internalNodes[i-dad.n]
        xj = j <= dad.n ? dad.Nodes[j] : dad.internalNodes[j-dad.n]
        r2 = sqeuclidean(x, xj)
        F[i, j] = rbf(r2)
        if r2 > 0
            D[i, j] = -log(r2) / (4π * k)
        end
    end

    @showprogress "Integrating fundamental solutions and radial basis functions" for i in 1:dad.nt
        x = i <= dad.n ? dad.Nodes[i] : dad.internalNodes[i-dad.n]
        P[:, i] = mon(x)
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = ind <= dad.n ? dad.Nodes[ind] : dad.internalNodes[ind-dad.n]
                r = xj - x
                R = norm(r)
                if R < 1e-10
                    continue
                    # R = 1e-16
                end
                # @infiltrate
                if i == 1
                    IP += int(mon, x, xj) * dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                end
                IF[i] += int(rbf, x, xj) * dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                ID[i] += -(2 * R^2 * log(R) - R^2) / (8 * π * k) * dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
            end

        end
    end
    # @infiltrate
    # @show F[1:5, 1:5], D[1:5, 1:5], IF[1:5], ID[1:5]

    # Z = zeros(npoly, npoly)
    # aux = [F P'; P Z]
    # # @infiltrate
    # M = ([IF; IP]'/aux)[1:end-npoly] .* D

    M = IF' / F .* D
    for i = 1:dad.nt #Laço dos pontos radiais
        M[i, i] = 0
        M[i, i] = -sum(M[i, :]) + ID[i]
    end
    set_cache!(dad; M)
    return nothing
end
