using DrWatson
@quickactivate :BEM

xs = range(-1, 1; length=25)
ys = range(-1, 1; length=25)
Z = [0.55 - hypot(x, y) for x in xs, y in ys]
println("size Z=", size(Z), " min=", minimum(Z), " max=", maximum(Z))
println("n>0=", count(>(0), Z), " n<0=", count(<(0), Z))
lines = marching_squares(xs, ys, Z, 0.0)
println("nlines=", length(lines))
for (i, ln) in enumerate(lines)
    println("  line $i n=$(length(ln)) closed=$(norm(ln[1]-ln[end]))")
end
