using UnicodePlots
using Printf

function upp(x, t; N=400, c=1.0, L=1.0)
    s = 0.0
    @inbounds for n in 1:N
        kn = (2n - 1) * π / (2L)
        ωn = c * kn
        a = 8 * L * (-1)^n / ((2n - 1) * π)^2
        s += a * (-ωn * ωn) * cos(ωn * t) * sin(kn * x)
    end
    return s
end

xs = collect(range(0.0, 1.0; length=801))
kw = (width=72, height=14, xlabel="x  (lower edge y=0)", ylabel="ü",
      border=:ascii, canvas=BrailleCanvas)

for t in (0.0, 0.25, 0.5, 1.0)
    ys = upp.(xs, t)
    mx, i = findmax(abs.(ys))
    plt = lineplot(xs, ys;
        title=@sprintf("ü(x) on y=0   t=%.2f   peak %.0f at x=%.3f", t, ys[i], xs[i]),
        kw...)
    println(plt)
    println()
end
