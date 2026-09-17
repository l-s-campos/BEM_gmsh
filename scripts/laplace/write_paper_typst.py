#!/usr/bin/env python3
"""Write fig03–fig23 Typst sources with the legend outside the axes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto")
RES = ROOT / "results"
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)


def arr(v):
    return "(" + ", ".join(f"{float(x):.8e}" for x in v) + ")"


def write_fig(num, xlabel, ylabel, series, logx=False, logy=False, title=""):
    lines = [
        '#import "/lib.typ": figure-page, lq, legend-out',
        "#figure-page({",
        "  show: lq.layout",
        "  lq.diagram(",
        "    width: 8.4cm, height: 5.5cm,",
    ]
    if title:
        lines.append(f"    title: [{title}],")
    lines += [
        f"    xlabel: [{xlabel}], ylabel: [{ylabel}],",
        "    legend: legend-out,",
    ]
    if logx:
        lines.append('    xscale: "log",')
    if logy:
        lines.append('    yscale: "log",')
    marks = ['"o"', '"s"', '"d"', '"*"', '"x"', '"+"']
    for i, s in enumerate(series):
        if not s["x"]:
            continue
        mk = s.get("mark", marks[i % len(marks)])
        st = s.get("stroke", "0.9pt")
        lab = s["label"]
        ys = s["y"]
        if logy:
            ys = [max(float(v), 1e-8) for v in ys]
        xs = s["x"]
        if logx:
            xs = [max(float(v), 1e-8) for v in xs]
        n = len(xs)
        every = ""
        if mk != "none" and n > 10:
            step = (n + 9) // 10  # at most 10 marks, full line
            every = f", every: {step}"
        lines.append(
            f'    lq.plot({arr(xs)}, {arr(ys)}, mark: {mk}, '
            f"label: [{lab}], stroke: {st}{every}),"
        )
    lines += ["  )", "})"]
    path = FIG / f"fig{num:02d}.typ"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path.name, "series", len(series))


def by(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if isinstance(v, float):
                ok = ok and abs(float(r[k]) - v) < 1e-9
            else:
                ok = ok and r[k] == v
        if ok:
            out.append(r)
    return out


def series_vs(rows, xkey, ykey, group=("nel", "fs"), lab=None):
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[g] if g in r else r.get(g) for g in group)
        groups[key].append(r)
    ser = []
    for key, sub in groups.items():
        sub = sorted(sub, key=lambda r: float(r[xkey]))
        if lab:
            label = lab(key, sub[0])
        else:
            parts = []
            for g, v in zip(group, key):
                parts.append(f"{g}={v}")
            label = " ".join(parts)
        ser.append({"label": label, "x": [r[xkey] for r in sub], "y": [r[ykey] for r in sub]})
    return ser


def main():
    paper = json.loads((RES / "paper_data.json").read_text())
    ex1 = paper["ex1"]
    ex1k = paper["ex1k"]
    ex2a = paper["ex2a"]
    ex2b = paper["ex2b"]
    ex2c = paper["ex2c"]
    ex2d = paper["ex2d"]
    ex3 = json.loads((RES / "example_3.json").read_text())["runs"]
    fen = json.loads((RES / "fenics_ex3.json").read_text())["runs"]
    rot = json.loads((RES / "example_3_rot.json").read_text())["runs"]
    fenr = json.loads((RES / "fenics_ex3_rot.json").read_text())["runs"]

    def lab_nel_fs(key, _r):
        nel, fs = key
        name = "iso IBP" if fs == "iso" else "aniso FS"
        return f"n={nel} {name}"

    # Fig 03 internal T
    write_fig(
        3,
        '$N_"int"$',
        "MRPE T int. [%]",
        series_vs(ex1, "ni", "eT_int_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 04 flux on the left (Dirichlet unknown)
    write_fig(
        4,
        '$N_"int"$',
        "MRPE q left [%]",
        series_vs(ex1, "ni", "eq_left_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 05 T top
    write_fig(
        5,
        '$N_"int"$',
        "MRPE T top [%]",
        series_vs(ex1, "ni", "eT_top_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 06 dudn left
    write_fig(
        6,
        '$N_"int"$',
        "MRPE $partial u \/ partial n$ left [%]",
        series_vs(ex1, "ni", "edudn_left_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 07 q bottom
    write_fig(
        7,
        '$N_"int"$',
        "MRPE q bottom [%]",
        series_vs(ex1, "ni", "eq_bot_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )

    # Fig 08 T int vs kx/ky
    write_fig(
        8,
        "$k_x \/ k_y$",
        "MRPE T int. [%]",
        series_vs(ex1k, "ratio", "eT_int_pct", group=("fs",), lab=lambda k, r: "iso IBP" if k[0] == "iso" else "aniso FS"),
        logx=False,
        logy=True,
    )
    # Fig 09 T(y) right vs kx/ky (iso)
    ser = []
    for r in by(ex1k, fs="iso"):
        ser.append({"label": f'kx/ky={r["ratio"]:.0f}', "x": r["y_right"], "y": r["T_right"]})
    write_fig(9, "$y$", "$T(x=1)$", ser, logx=False, logy=False)
    # Fig 10 dudn left vs kx/ky (iso)
    ser = []
    for r in by(ex1k, fs="iso"):
        ser.append({"label": f'kx/ky={r["ratio"]:.0f}', "x": r["y_left"], "y": r["dudn_left"]})
    write_fig(10, "$y$", "$partial u \/ partial n$  ($x=0$)", ser)

    # Fig 12 2A internal T
    write_fig(
        12,
        '$N_"int"$',
        "MRPE T int. [%]",
        series_vs(ex2a, "ni", "eT_int_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 13 2A q right
    write_fig(
        13,
        '$N_"int"$',
        "MRPE q right [%]",
        series_vs(ex2a, "ni", "eq_right_pct", lab=lab_nel_fs),
        logx=True,
        logy=True,
    )
    # Fig 14 2B T vs k1-k2
    write_fig(
        14,
        "$k_1 - k_2$",
        "MRPE T int. [%]",
        series_vs(ex2b, "dk", "eT_int_pct", group=("fs",), lab=lambda k, r: "iso IBP" if k[0] == "iso" else "aniso FS"),
        logy=True,
    )
    # Fig 15 2C T vs ni (drop Hessian blow-ups)
    ex2c_ok = [r for r in ex2c if float(r["eT_int_pct"]) < 50]
    write_fig(
        15,
        '$N_"int"$',
        "MRPE T int. [%]",
        series_vs(ex2c_ok, "ni", "eT_int_pct", group=("nel",), lab=lambda k, r: f"n={k[0]} iso"),
        logx=True,
        logy=True,
    )
    # Fig 16 2C flux (q = -n·K∇u ≡ 0 analytically)
    write_fig(
        16,
        '$N_"int"$',
        "mean $|q|$ on $y=0$  [$times 100$]",
        series_vs(ex2c_ok, "ni", "eq_bot_pct", group=("nel",), lab=lambda k, r: f"n={k[0]} iso"),
        logx=True,
        logy=True,
    )
    # Fig 17: q on y=0. For K=[1 1;1 1], K∇u=0 so q_ana=0.
    # Recovering ∂u/∂n from q/(n·K n) is meaningless (n·K n does not invert K).
    prof = [r for r in ex2c if "x_bot" in r]
    ser = []
    if prof:
        r = prof[0]
        # dudn_bot was -q/(n·Kn); q = -(n·Kn) dudn so q_num = -knn * dudn_bot
        # We stored dudn_bot; reconstruct q ≈ 0. Plot q if we have it... use
        # dudn_bot * 0 as numerical flux from the fact q_ana=0 and q_num ~ 0.
        # The stored dudn_bot is ~0 (from q~0). Plot that as q proxy: actually
        # plot numerical q via -dudn * (n·K n). On bottom n=(0,-1), n·K n = 1,
        # q = -dudn. So numerical q ≈ -dudn_bot ~ 0, analytical q = 0.
        qn = [-float(v) for v in r["dudn_bot"]]
        xa = list(r["x_bot"])
        # 2 Gauss points/element → sawtooth; subsample to ≤10 icons
        if len(xa) > 10:
            idx = [round(i * (len(xa) - 1) / 9) for i in range(10)]
            xa = [xa[i] for i in idx]
            qn = [qn[i] for i in idx]
        qa = [0.0 for _ in xa]
        ser.append({"label": "numerical $q$", "x": xa, "y": qn})
        ser.append({
            "label": "analytical $q=0$",
            "x": xa,
            "y": qa,
            "mark": "none",
            "stroke": "(dash: \"dashed\", thickness: 0.9pt)",
        })
    write_fig(17, "$x$", "$q$  ($y=0$)", ser)
    # Fig 18: same profile (q, not ∂u/∂n). Analytical q=0 for this K.
    write_fig(18, "$x$", "$q$  ($y=0$)", ser)

    # Fig 20 2D T int vs ni
    write_fig(
        20,
        '$N_"int"$',
        "MRPE T int. [%]",
        series_vs(ex2d, "ni", "eT_int_pct", group=("fs",), lab=lambda k, r: "iso IBP" if k[0] == "iso" else "aniso FS"),
        logx=True,
        logy=True,
    )

    # Fig 21 Ex3 MRPE T right vs Fenics, finest mesh, vs k1 (k2=0.5)
    def interp(y, T, yq):
        pts = sorted(zip((float(a) for a in y), (float(b) for b in T)))
        ys = [p[0] for p in pts]
        Ts = [p[1] for p in pts]
        out = []
        for z in yq:
            z = float(z)
            if z <= ys[0]:
                out.append(Ts[0])
            elif z >= ys[-1]:
                out.append(Ts[-1])
            else:
                i = 0
                while i < len(ys) - 2 and ys[i + 1] < z:
                    i += 1
                t = (z - ys[i]) / (ys[i + 1] - ys[i] + 1e-30)
                out.append((1 - t) * Ts[i] + t * Ts[i + 1])
        return out

    lcs = sorted({float(r["lc"]) for r in ex3})
    lc = lcs[0]  # finest BEM
    fen_f = [r for r in fen if abs(r["k2"] - 0.5) < 1e-12]
    fen_best = {}
    for r in fen_f:
        fen_best.setdefault(r["k1"], r)
        if r["h"] < fen_best[r["k1"]]["h"]:
            fen_best[r["k1"]] = r

    # Fig 21: T on the right (not error), 3 panels
    panels = []
    for k1 in (1.0, 3.0, 5.0):
        plots = []
        if k1 in fen_best:
            fr = fen_best[k1]
            plots.append(("FEniCS", fr["y_right"], fr["T_right"], "none", "1.2pt"))
        for fs, lab, mk, st in (
            ("iso", "iso IBP", '"o"', "0.8pt"),
            ("aniso", "aniso FS", '"s"', "(dash: \"dashed\", thickness: 0.8pt)"),
        ):
            sub = [r for r in by(ex3, fs=fs, k1=k1, k2=0.5)
                   if abs(float(r["lc"]) - lc) < 1e-12]
            if sub:
                r = sub[0]
                plots.append((lab, r["y_right"], r["T_right"], mk, st))
        panels.append((k1, plots))
    lines = [
        '#import "/lib.typ": figure-page, lq, legend-out',
        "#figure-page({",
        "  show: lq.layout",
        "  grid(columns: 3, column-gutter: 0.7em, row-gutter: 0.6em,",
    ]
    for k1, plots in panels:
        lines.append("    lq.diagram(")
        lines.append("      width: 6.2cm, height: 4.6cm,")
        lines.append(f"      title: [$k_1={k1:g}$, $k_2=0.5$],")
        lines.append("      xlabel: [$y$], ylabel: [$T(x=1)$],")
        lines.append("      legend: legend-out,")
        for lab, x, y, mk, st in plots:
            n = len(x)
            ev = f", every: {(n + 9) // 10}" if mk != "none" and n > 10 else ""
            lines.append(
                f'      lq.plot({arr(x)}, {arr(y)}, mark: {mk}, '
                f"label: [{lab}], stroke: {st}{ev}),"
            )
        lines.append("    ),")
    lines += ["  )", "})"]
    (FIG / "fig21.typ").write_text("\n".join(lines) + "\n")
    print("wrote fig21.typ panels", len(panels))

    # Fig 22: T(1,0.5) vs discretisation (not error)
    ser = []
    for k1, labk in ((1.0, "k1=1"), (3.0, "k1=3"), (5.0, "k1=5")):
        for fs, labf in (("iso", "iso IBP"), ("aniso", "aniso FS")):
            sub = sorted(by(ex3, fs=fs, k1=k1, k2=0.5), key=lambda r: int(r["n"]))
            ser.append({
                "label": f"{labk} {labf}",
                "x": [r["n"] for r in sub],
                "y": [r["T_mid"] for r in sub],
            })
        if k1 in fen_best:
            Tref = fen_best[k1]["T_mid"]
            n0, n1 = 112, 280
            ser.append({
                "label": f"{labk} FEniCS",
                "x": [n0, n1], "y": [Tref, Tref],
                "mark": "none",
                "stroke": "(dash: \"dashed\", thickness: 0.8pt)",
            })
    write_fig(22, "$n$", "$T(1,0.5)$", ser)

    # Fig 23 rotated K T(y)
    lc = min(float(r["lc"]) for r in rot)
    fenrf = min(fenr, key=lambda r: r["h"])
    ser = [
        {"label": "FEniCS", "x": fenrf["y_right"], "y": fenrf["T_right"], "mark": "none", "stroke": "1.2pt"},
    ]
    for fs, lab, mk in (("iso", "iso IBP", '"o"'), ("aniso", "aniso FS", '"s"')):
        r = [x for x in rot if x["fs"] == fs and abs(float(x["lc"]) - lc) < 1e-12][0]
        ser.append({"label": lab, "x": r["y_right"], "y": r["T_right"], "mark": mk})
    write_fig(23, "$y$", "$T(x=1)$", ser)


if __name__ == "__main__":
    main()
