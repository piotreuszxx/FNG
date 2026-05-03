#!/usr/bin/env python3
"""Plot membership functions from investment.fcl and save PNGs.

Usage:
  python plot_mfs.py

Creates directory `plots/` inside lab04 and writes one PNG per variable.
Requires: matplotlib, numpy
"""
import re
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


FCL_PATH = os.path.join(os.path.dirname(__file__), 'investment.fcl')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')


def parse_fcl(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = defaultdict(dict)

    block_re = re.compile(r"(FUZZIFY|DEFUZZIFY)\s+(\w+)\s+(.*?)END_\1", re.S)
    term_re = re.compile(r"TERM\s+(\w+)\s*:=\s*([^;]+);", re.I)
    point_re = re.compile(r"\(([^)]+)\)")

    for m in block_re.finditer(text):
        block_type = m.group(1)
        varname = m.group(2)
        body = m.group(3)
        for tm in term_re.finditer(body):
            term = tm.group(1)
            rhs = tm.group(2)
            points = []
            for p in point_re.finditer(rhs):
                coords = p.group(1).strip()
                parts = [s.strip() for s in coords.split(',')]
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append((x, y))
                    except ValueError:
                        continue
            if points:
                points.sort(key=lambda t: t[0])
                blocks[varname][term] = points

    return blocks


def piecewise_mf(points, x_vals):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    y_out = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        if x <= xs[0]:
            y_out[i] = ys[0]
        elif x >= xs[-1]:
            y_out[i] = ys[-1]
        else:
            for j in range(len(xs)-1):
                x0, x1 = xs[j], xs[j+1]
                if x0 <= x <= x1:
                    y0, y1 = ys[j], ys[j+1]
                    if x1 == x0:
                        y_out[i] = max(y0, y1)
                    else:
                        t = (x - x0) / (x1 - x0)
                        y_out[i] = y0 + t * (y1 - y0)
                    break
    return y_out


def plot_variable(varname, terms, out_dir):
    # determine x range from points
    all_x = [x for pts in terms.values() for x, _ in pts]
    xmin, xmax = min(all_x), max(all_x)
    if math.isclose(xmin, xmax):
        xmin -= 1
        xmax += 1
    margin = (xmax - xmin) * 0.05
    xs = np.linspace(xmin - margin, xmax + margin, 600)

    # Polish display names
    var_display_map = {
        'expected_return': 'Oczekiwana stopa zwrotu',
        'risk': 'Ryzyko',
        'liquidity': 'Płynność',
        'attractiveness': 'Atrakcyjność'
    }
    term_map = {
        'low': 'niska',
        'medium': 'średnia',
        'high': 'wysoka'
    }

    plt.figure(figsize=(7, 4))
    # colors for terms
    term_colors = {
        'low': '#d73027',     # red
        'medium': '#fdae61',  # orange
        'high': '#1a9850'     # green
    }
    for name, pts in sorted(terms.items()):
        ys = piecewise_mf(pts, xs)
        label = term_map.get(name, name)
        color = term_colors.get(name, None)
        # fill under curve to indicate membership area
        if color is not None:
            plt.fill_between(xs, ys, 0.0, where=ys > 0, facecolor=color, alpha=0.25, interpolate=True)
        plt.plot(xs, ys, label=label, color=color)
        # mark key points
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        plt.plot(px, py, 'o', markerfacecolor='none', markeredgecolor=(color if color is not None else 'black'))

    plt.ylim(-0.05, 1.05)
    plt.xlabel(var_display_map.get(varname, varname))
    plt.ylabel('przynależność')
    plt.title(f'Funkcje przynależności - {var_display_map.get(varname, varname)}')
    plt.grid(alpha=0.25)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{varname}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def main():
    blocks = parse_fcl(FCL_PATH)
    if not blocks:
        print(f"No FUZZIFY/DEFUZZIFY blocks found in {FCL_PATH}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for var, terms in blocks.items():
        if not terms:
            continue
        path = plot_variable(var, terms, OUT_DIR)
        written.append(path)

    print("Wrote plots:")
    for p in written:
        print(' -', p)


if __name__ == '__main__':
    main()
