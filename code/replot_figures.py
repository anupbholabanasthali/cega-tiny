"""Re-plot the figures that show CEGA-Lite (colour-blind / grayscale safe) from the saved result files.
Usage: python replot_figures.py RESULTS_DIR PARTB_AGG_DIR OUT_DIR
RESULTS_DIR : unzipped cega_tiny_results (raw/*.csv)
PARTB_AGG_DIR : folder with main6_agg.csv and sweeps_agg.csv (aggregates printed by the Part B notebook)
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, P, OUT = sys.argv[1:4]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Static": "#7f7f7f", "Online-SGD": "#1f77b4", "TinyOL-Head": "#17becf", "RandomSkip": "#bcbd22",
       "Periodic-Full": "#ff7f0e", "DriftTrig-Full": "#9467bd", "GovRetrain": "#8c564b", "CEGA-Tiny": "#d62728",
       "CEGA-Lite": "#000000", "CEGA-noGate": "#e377c2", "CEGA-noBurst": "#393b79", "CEGA-noSoC": "#aec7e8",
       "CEGA-noEntropy": "#98df8a", "CEGA-HeadOnly": "#2ca02c", "SB-Matched": "#9467bd", "Entropy-Matched": "#8c564b"}
STY = {m: dict(ls="-", marker="o") for m in COL}
STY["CEGA-Lite"] = dict(ls="--", marker="s"); STY["CEGA-Tiny"] = dict(ls="-", marker="o")
HATCH = {"CEGA-Tiny": "", "CEGA-Lite": "//", "CEGA-noGate": "..", "CEGA-noBurst": "\\\\", "CEGA-noSoC": "xx",
         "CEGA-noEntropy": "++", "CEGA-HeadOnly": "--", "SB-Matched": "//", "Entropy-Matched": "..", "RandomSkip": "xx", "Online-SGD": ""}
DS = ["GasDrift", "UCI-HAR", "Elec2", "CoverType"]; ND = 4
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=450, bbox_inches="tight"); plt.close(fig)
def rd(n): return pd.read_csv(f"{R}/raw/{n}.csv")
main = rd("main_results")

# Fig 2
fig, axs = plt.subplots(1, ND, figsize=(7.2, 2.5))
for ax, n in zip(axs, DS):
    for m in ["Static","Online-SGD","TinyOL-Head","RandomSkip","Periodic-Full","DriftTrig-Full","GovRetrain","CEGA-Tiny","CEGA-Lite"]:
        s = main[(main.dataset == n) & (main.method == m)]
        ax.errorbar(s.E_adapt_mJ.mean(), s.acc.mean(), xerr=s.E_adapt_mJ.std(), yerr=s.acc.std(), fmt=STY[m]["marker"],
                    color=COL[m], ms=4, capsize=2, label=m, mfc=("white" if m == "CEGA-Lite" else COL[m]))
    ax.set_xscale("symlog", linthresh=1.0); ax.set_title(n); ax.set_xlabel("Adapt. energy (mJ)")
axs[0].set_ylabel("Prequential accuracy")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.14))
save(fig, "Fig02")

def sweep_plot(df, metrics, fname, xlabel, logx, xvals, methods=None, size=(7.2, 2.0)):
    methods = methods or list(dict.fromkeys(df.method))
    fig, axs = plt.subplots(len(metrics), ND, figsize=(size[0], size[1]*len(metrics)), squeeze=False)
    for j, n in enumerate(DS):
        for i, (met, lab) in enumerate(metrics):
            for m in methods:
                g = df[(df.dataset == n) & (df.method == m)].groupby("value")[met].mean()
                g.index = g.index.astype(float)
                axs[i, j].plot(xvals, [g.get(float(v), np.nan) for v in xvals], color=COL[m], lw=1.6 if m=="CEGA-Tiny" else 1.1,
                               ms=3, label=m, **STY[m])
            if logx: axs[i, j].set_xscale("log")
            axs[0, j].set_title(n); axs[i, 0].set_ylabel(lab); axs[-1, j].set_xlabel(xlabel)
    axs[0, 0].legend(fontsize=6); save(fig, fname)
def withval(n):
    d = rd(n)
    if "value" not in d:
        key = {"sweep_budget": "beta", "sweep_labels": "p_label", "sweep_noise": "label_noise", "sweep_energy": "e_wr_nj"}.get(n)
        d["value"] = d[key]
    return d
sweep_plot(withval("sweep_budget"), [("acc","Accuracy"),("violations","Violations")], "Fig05", r"Energy budget $\beta$", True, [0.1,0.25,0.5,1.0,2.0])
sweep_plot(withval("sweep_labels"), [("acc","Accuracy"),("E_adapt_mJ","Adapt. energy (mJ)")], "Fig08", r"Label rate $\rho$", True, [0.05,0.1,0.2,0.5])
sweep_plot(withval("sweep_noise"), [("acc","Accuracy"),("E_adapt_mJ","Adapt. energy (mJ)")], "Fig09", r"Label noise $\eta$", False, [0.0,0.1,0.2,0.3])

# Fig 6 ablation
abl = ["CEGA-Tiny","CEGA-Lite","CEGA-noGate","CEGA-noBurst","CEGA-noSoC","CEGA-noEntropy","CEGA-HeadOnly"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8)); x = np.arange(ND); w = 0.8/len(abl)
for k, m in enumerate(abl):
    for ax, col in zip(axs, ["acc","E_adapt_mJ"]):
        g = main[main.method == m].groupby("dataset")[col]
        mu = [g.mean()[n] for n in DS]; sd = [g.std()[n] for n in DS]
        ax.bar(x + k*w, mu, w, yerr=sd, color=COL[m] if m != "CEGA-Lite" else "white", edgecolor="k" if m=="CEGA-Lite" else COL[m],
               hatch=HATCH[m], error_kw=dict(lw=0.6, capsize=1), label=m, lw=0.8)
for ax, t in zip(axs, ["Accuracy","Adaptation energy (mJ)"]):
    ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels(DS, fontsize=7); ax.set_title(t)
lo = main[main.method.isin(abl)].acc.min(); axs[0].set_ylim(max(0, lo-0.05), None)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.12))
save(fig, "Fig06")

# Fig 7 fraction back-propagated
fig, ax = plt.subplots(figsize=(7.2, 3.3)); w = 0.26
for k, m in enumerate(["CEGA-Tiny","CEGA-Lite","CEGA-noSoC"]):
    v = [100*main[(main.dataset==n)&(main.method==m)].frac_updated.mean() for n in DS]
    b = ax.bar(np.arange(ND)+(k-1)*w, v, w, color=COL[m] if m!="CEGA-Lite" else "white", edgecolor="k" if m=="CEGA-Lite" else COL[m],
               hatch=HATCH[m], label=m, lw=0.8)
    for xi, vi in zip(np.arange(ND)+(k-1)*w, v): ax.text(xi, vi+1, f"{vi:.0f}", ha="center", fontsize=7)
ax.set_xticks(range(ND)); ax.set_xticklabels(["Gas Drift","UCI-HAR","Elec2","CoverType"]); ax.set_ylim(0,100)
ax.set_ylabel("Labelled samples\nback-propagated (%)"); ax.legend(frameon=False, loc="upper right"); save(fig, "Fig07")

# Fig 12 model size with seed s.d. bands
md = rd("sweep_model"); order = ["(32, 16)","(64, 32)","(128, 64)"]
fig, axs = plt.subplots(2, ND, figsize=(7.2, 4.0), squeeze=False)
for j, n in enumerate(DS):
    for m in ["Online-SGD","TinyOL-Head","CEGA-Tiny","CEGA-Lite"]:
        s = md[(md.dataset==n)&(md.method==m)]
        for i, col in enumerate(["acc","E_adapt_mJ"]):
            g = s.groupby("value")[col]; mu = [g.mean()[o] for o in order]; sd = [g.std()[o] for o in order]
            axs[i,j].errorbar(range(3), mu, yerr=sd, color=COL[m], ms=3, lw=1.6 if m=="CEGA-Tiny" else 1.1, capsize=1.5, elinewidth=0.6,
                              label=m, **STY[m])
    for i in range(2):
        axs[i,j].set_xticks(range(3)); axs[i,j].set_xticklabels([o.replace(" ","") for o in order], fontsize=6)
    axs[0,j].set_title(n)
axs[0,0].set_ylabel("Accuracy"); axs[1,0].set_ylabel("Adapt. energy (mJ)"); axs[0,0].legend(fontsize=6); save(fig, "Fig12")

# Part B figures
m6 = pd.read_csv(f"{P}/main6_agg.csv"); sw = pd.read_csv(f"{P}/sweeps_agg.csv")
D6 = ["GasDrift","UCI-HAR","Elec2","CoverType","SEA","Weather"]; meths = ["Online-SGD","RandomSkip","SB-Matched","Entropy-Matched","CEGA-Tiny","CEGA-Lite"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7)); x = np.arange(6); w = 0.13
for k, me in enumerate(meths):
    for ax, c, s in zip(axs, ["acc","E"], ["acc_sd","E_sd"]):
        g = m6[m6.method==me].set_index("dataset").loc[D6]
        ax.bar(x+k*w, g[c], w, yerr=g[s], color=COL[me] if me!="CEGA-Lite" else "white", edgecolor="k" if me=="CEGA-Lite" else COL[me],
               hatch=HATCH.get(me,""), error_kw=dict(lw=0.5, capsize=1), label=me, lw=0.7)
for ax, t in zip(axs, ["Prequential accuracy","Adaptation energy (mJ)"]):
    ax.set_xticks(x+2.5*w); ax.set_xticklabels(D6, fontsize=7, rotation=20); ax.set_title(t)
axs[0].set_ylim(0.65, 0.92); axs[0].legend(fontsize=6, ncol=2, frameon=False); save(fig, "Fig13")

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
dl = sw[sw.sw=="delay"].copy(); dl["value"] = dl.value.astype(float)
for me in ["Online-SGD","RandomSkip","CEGA-Tiny","CEGA-Lite"]:
    g = dl[dl.method==me].sort_values("value")
    for ax, c in zip(axs[:2], ["acc","E_adapt_mJ"]):
        ax.plot(range(len(g)), g[c], color=COL[me], label=me, ms=3, lw=1.6 if me=="CEGA-Tiny" else 1.1, **STY[me])
for ax, t in zip(axs[:2], ["Accuracy vs label delay","Adapt. energy (mJ) vs delay"]):
    ax.set_xticks(range(4)); ax.set_xticklabels([0,1,3,5]); ax.set_xlabel("Label delay (windows)"); ax.set_title(t)
axs[0].legend(fontsize=6, frameon=False)
so = sw[sw.sw=="solar"].copy(); vals = ["('const', 0.1)","('const', 0.25)","('const', 0.5)","('solar', 0.1)","('solar', 0.25)","('solar', 0.5)"]
lab = [v.replace("'solar'","d/n").replace("'const'","const").replace("(","").replace(")","").replace(", ","\n") for v in vals]
mm = ["Online-SGD","Periodic-Full","CEGA-noSoC","CEGA-Tiny","CEGA-Lite"]; w = 0.16
for k, me in enumerate(mm):
    g = so[so.method==me].set_index("value").loc[vals]
    axs[2].bar(np.arange(6)+(k-2)*w, g.violations, w, color=COL[me] if me!="CEGA-Lite" else "white",
               edgecolor="k" if me=="CEGA-Lite" else COL[me], hatch=HATCH.get(me,""), label=me, lw=0.7)
axs[2].set_xticks(range(6)); axs[2].set_xticklabels(lab, fontsize=6); axs[2].set_title("Violations per stream"); axs[2].set_xlabel(r"harvest profile and $\beta$ (d/n = day/night)", fontsize=6.5)
axs[2].legend(fontsize=5.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.98)); save(fig, "Fig14")
print("done")
