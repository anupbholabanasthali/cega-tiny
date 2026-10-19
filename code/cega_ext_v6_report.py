# ============================== C3. Report (compact text summary)
pd.set_option("display.width", 250)
c1 = pd.read_csv(f"{OUT_C}/C1_fixed_rate.csv")
print("###C1_SUMMARY  (mean over datasets x seeds; viol = mean violations per run)")
s = c1.groupby(["beta", "method"]).agg(acc=("acc", "mean"), E=("E_adapt_mJ", "mean"), viol=("violations", "mean"),
                                       upd=("frac_updated", "mean")).reset_index()
print(s.round(4).to_csv(index=False))
print("###C1_VS_CEGA  (paired over dataset x seed: method minus CEGA-Tiny; acc in points)")
ref = c1[c1.method == "CEGA-Tiny"].set_index(["dataset", "seed", "beta"])
out = []
for (b, m), g in c1[c1.method != "CEGA-Tiny"].groupby(["beta", "method"]):
    g = g.set_index(["dataset", "seed", "beta"]); r = ref.loc[g.index]
    da = (g.acc - r.acc) * 100; dE = g.E_adapt_mJ - r.E_adapt_mJ
    p = stats.wilcoxon(da).pvalue if (da != 0).any() else 1.0
    # dominance: per dataset, the rule is at least as accurate (within 0.2 pts), no more energy, and no violations
    dd = pd.DataFrame(dict(da=da, dE=dE, v=g.violations)).groupby(level=0).mean()
    dom = int(((dd.da >= -0.2) & (dd.dE <= 0) & (dd.v == 0)).sum())
    out.append(dict(beta=b, method=m, d_acc=da.mean(), p=p, d_E=dE.mean(), viol=g.violations.mean(), datasets_matching_CEGA=dom))
print(pd.DataFrame(out).round(4).to_csv(index=False))
print("###C1_PER_DATASET_BEST_FIXED  (best guarded fixed rate per dataset & beta, chosen with hindsight)")
gd = c1[c1.guard == True].groupby(["dataset", "beta", "rule", "rate"]).agg(acc=("acc", "mean"), E=("E_adapt_mJ", "mean")).reset_index()
best = gd.loc[gd.groupby(["dataset", "beta", "rule"]).acc.idxmax()]
ct = c1[c1.method == "CEGA-Tiny"].groupby(["dataset", "beta"]).agg(cega_acc=("acc", "mean"), cega_E=("E_adapt_mJ", "mean")).reset_index()
print(best.merge(ct, on=["dataset", "beta"]).round(4).to_csv(index=False))

c2 = pd.read_csv(f"{OUT_C}/C2_cnn_har_raw.csv")
print("###C2_CNN_MAIN")
g = c2.groupby("method").agg(n=("acc", "size"), acc=("acc", "mean"), asd=("acc", "std"), f1=("macro_f1", "mean"),
                             E=("E_adapt_mJ", "mean"), Esd=("E_adapt_mJ", "std"), upd=("frac_updated", "mean"),
                             viol=("violations", "mean"), rec=("recovery_win", "mean")).reset_index()
print(g.round(4).to_csv(index=False))
print("###C2_CNN_TESTS  (CEGA-Tiny minus other, paired over seeds; acc in points)")
pv = c2.pivot_table(index="seed", columns="method", values=["acc", "E_adapt_mJ"])
rows = []
for m in [x for x in c2.method.unique() if x != "CEGA-Tiny"]:
    for met, sc in (("acc", 100), ("E_adapt_mJ", 1)):
        d = (pv[met]["CEGA-Tiny"] - pv[met][m]) * sc
        p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append(dict(comparison=f"CEGA-Tiny vs {m}", metric=met, mean_diff=d.mean(), wins=int((d > 0).sum()), p=p))
print(pd.DataFrame(rows).round(4).to_csv(index=False))
m0 = PRETRAINED[[k for k in PRETRAINED if k[0] == "HAR-raw-CNN"][0]][0]
print("###C2_MODEL", dict(params=m0.n_params(), macs_inf=m0.macs_fwd(), layer_macs=m0.layer_macs(),
                          E_inf_sample_uJ=m0.macs_fwd() * ENERGY["e_mac_nj"] * 1e-3,
                          E_full_retrain_mJ=planned_energy_mJ(m0, 3, CFG["buffer"], CFG["max_epochs"]),
                          E_head_retrain_mJ=planned_energy_mJ(m0, 1, CFG["buffer"], CFG["max_epochs"])))
print("###END")
