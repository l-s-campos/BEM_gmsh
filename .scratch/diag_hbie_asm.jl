using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)

function report(label, mesh)
    A, b, _, _ = apply_bc_fsdt(mesh)
    wc = try
        solve_fsdt!(mesh)
        fsdt_w_int(mesh, 1)
    catch e
        @printf("  SOLVE FAIL %s\n", e)
        NaN
    end
    @printf("%s\n", label)
    @printf("  ||H||=%.3e  ||G||=%.3e  ||q||=%.3e  ||b||=%.3e  cond(A)=%.3e\n",
        norm(mesh.H), norm(mesh.G), norm(mesh.q), norm(b), cond(A))
    n = length(mesh.nodes)
    @printf("  ||H_bd||=%.3e  ||H_int||=%.3e  ||q_bd||=%.3e  ||q_int||=%.3e\n",
        norm(mesh.H[1:5n, :]),
        size(mesh.H, 1) > 5n ? norm(mesh.H[5n+1:end, :]) : 0.0,
        norm(mesh.q[1:5n]),
        length(mesh.q) > 5n ? norm(mesh.q[5n+1:end]) : 0.0)
    @printf("  w_c=%.6e  rel Navier=%.2f%%\n", wc, 100 * abs(wc - wN) / abs(wN))
    return mesh
end

mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(mesh; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(mesh; npg=4)
report("CBIE Guiggiani", mesh)

mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_unsym_fsdt!(mesh; npg=4, nsub=4, singular=:guiggiani, bie=:hbie,
    ninterp=12, scale_hbie=true)
report("HBIE Guiggiani scaled", mesh)

mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_unsym_fsdt!(mesh; npg=4, nsub=4, singular=:guiggiani, bie=:hbie,
    ninterp=12, scale_hbie=false)
report("HBIE Guiggiani unscaled", mesh)

meshT = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_unsym_fsdt!(meshT; npg=4, nsub=4, singular=:telles, bie=:hbie,
    scale_hbie=false)
report("HBIE Telles unscaled", meshT)

# HBIE Guiggiani with CBIE domain load (q from a CBIE mesh)
meshH = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_unsym_fsdt_hbie!(meshH; npg=4, nsub=4, ninterp=12)
meshC = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshC; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(meshC; npg=4)
# replace HBIE q with CBIE q, re-scale? H already scaled. Just swap interior/boundary q from CBIE without matching scale — skip.

# q=0 HBIE: should give ~0
mesh0 = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_unsym_fsdt_hbie!(mesh0; npg=4, nsub=4, ninterp=12)
fill!(mesh0.q, 0)
report("HBIE Guiggiani q=0", mesh0)

# CBIE q=0
mesh0c = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(mesh0c; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
fill!(mesh0c.q, 0)
report("CBIE Guiggiani q=0", mesh0c)
