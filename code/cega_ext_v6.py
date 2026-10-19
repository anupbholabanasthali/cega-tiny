# ============================== C. Reviewer-response experiments (v6)
# C1: fixed-rate selection rules that must choose their own update rate (no oracle rate from CEGA-Tiny)
# C2: 1D CNN on raw UCI-HAR inertial signals (non-MLP model)
OUT_C = "cega_tiny_v6_results"; os.makedirs(OUT_C, exist_ok=True)

# ---------------------------------------------------------------- C1. fixed-rate rules
def run_fixed(ds, rule, seed, rate, guard, beta=None, keep_frac=None):
    """Online selection with a FIXED update rate chosen in advance (rate in (0,1]).
    rule: 'SB' (largest loss), 'Entropy' (largest predictive entropy) or 'Random'.
    guard=True adds the same affordability check as CEGA-Tiny (skip the update if the battery cannot pay).
    Energy accounting identical to run_stream4 (constant harvest, no delay, float inference)."""
    beta = CFG["beta"] if beta is None else beta
    base, _, _ = pretrained_model(ds, seed)
    model = base.copy(); rng = np.random.default_rng(10_000 + seed)
    Xs, ys, W, B = ds["Xs"], ds["ys"], CFG["window"], CFG["buffer"]
    nwin = len(ys) // W; n = nwin * W
    lab = np.random.default_rng(20_000 + seed).random(n) < CFG["p_label"]
    rs_rng = np.random.default_rng(40_000 + seed); mt_rng = np.random.default_rng(50_000 + seed)
    L = model.L
    E_full = planned_energy_mJ(model, L, B, CFG["max_epochs"])
    cap = CFG["cap_mult"] * E_full; batt = CFG["soc0"] * cap
    E_inf_win = W * model.macs_fwd() * ENERGY["e_mac_nj"] * 1e-6
    harvest = E_inf_win + beta * E_full / 20.0
    preds = np.empty(n, int); E_ad = 0.0; viol = 0; n_lab = n_upd = 0; skipped = 0
    for w in range(nwin):
        sl = slice(w * W, (w + 1) * W); Xw, yw = Xs[sl], ys[sl]
        P = model.proba(Xw); pw = P.argmax(1); preds[sl] = pw
        batt = min(cap, batt + harvest) - E_inf_win
        li = np.where(lab[sl])[0]; n_lab += len(li); E_step = 0.0
        if len(li):
            if rule == "Random":
                sel = li[rs_rng.random(len(li)) < rate]
            else:
                k = int(np.floor(rate * len(li) + mt_rng.random()))
                Pv = P[li]
                score = -np.log(Pv[np.arange(len(li)), yw[li]] + 1e-12) if rule == "SB" else -(Pv * np.log(Pv + 1e-12)).sum(1)
                sel = li[np.argsort(-score, kind="stable")[:k]]
            if len(sel):
                if guard and batt - planned_energy_mJ(model, L, len(sel), 1) < 0:
                    skipped += 1
                else:
                    ops = sgd_train(model, Xw[sel], yw[sel], L, 1, CFG["lr_online"], rng, CFG["momentum"], CFG["batch"])
                    E_step = train_energy_mJ(model, L, ops); n_upd += len(sel)
        if E_step > 0:
            E_ad += E_step; batt -= E_step
            if batt < 0: viol += 1; batt = 0.0
    y_true = ys[:n]
    return dict(dataset=ds["name"], seed=seed, beta=beta, method=f"{rule}-{'G' if guard else 'U'}-{rate}",
                rule=rule, guard=guard, rate=rate, acc=float((preds == y_true).mean()),
                macro_f1=float(f1_score(y_true, preds, average="macro")), E_adapt_mJ=E_ad, violations=viol,
                frac_updated=n_upd / max(n_lab, 1), skipped_unaffordable=skipped)

# regression check: SB rule at CEGA-Tiny's own rate, unguarded == SB-Matched in run_stream4 (same RNG streams)
_ds = get_dataset(DATASETS6[0], 0)
_c = run_stream4(_ds, "CEGA-Tiny", 0); _m = run_stream4(_ds, "SB-Matched", 0, keep=_c["frac_updated"])
_f = run_fixed(_ds, "SB", 0, _c["frac_updated"], False)
print("C1 REGRESSION CHECK:", "PASS" if (abs(_f["acc"] - _m["acc"]) < 1e-12 and abs(_f["E_adapt_mJ"] - _m["E_adapt_mJ"]) < 1e-9) else
      f"FAIL acc {_f['acc']} vs {_m['acc']}, E {_f['E_adapt_mJ']} vs {_m['E_adapt_mJ']}")

def exp_C1(seeds, betas=(0.1, 0.25, 0.5, 1.0), rates=(0.25, 0.5, 0.75), datasets=None):
    t0 = time.time(); rows = []
    for dname in (datasets or DATASETS6):
        for seed in seeds:
            ds = get_dataset(dname, seed)
            for beta in betas:
                for m in ["CEGA-Tiny", "CEGA-Lite", "Online-SGD"]:
                    r = run_stream4(ds, m, seed, beta=beta); r.update(rule=m, guard=None, rate=np.nan); rows.append(r)
                for rule in ("SB", "Entropy"):
                    for rate in rates:
                        for guard in (False, True):
                            rows.append(run_fixed(ds, rule, seed, rate, guard, beta=beta))
        print(f"C1 {dname} done ({time.time()-t0:.0f}s)", flush=True)
    keep = ["dataset", "seed", "beta", "method", "rule", "guard", "rate", "acc", "macro_f1", "E_adapt_mJ", "violations", "frac_updated"]
    df = pd.DataFrame(rows)[keep + ([c for c in ["skipped_unaffordable"] if c in pd.DataFrame(rows).columns])]
    df.to_csv(f"{OUT_C}/C1_fixed_rate.csv", index=False); return df

# ---------------------------------------------------------------- C2. 1D CNN on raw HAR
class TinyCNN1D:
    """conv(9->c1,k,s) - ReLU - conv(c1->c2,k,s) - ReLU - global average pool - dense(c2->C).
    Same interface as TinyMLP (W, b, L, proba, grads, layer_macs, ...); layer 0/1 = convs, layer 2 = head."""
    def __init__(self, cin, T, C, rng, c1=16, c2=32, k=5, s=2):
        self.cin, self.T, self.k, self.s, self.L = cin, T, k, s, 3
        self.T1 = (T - k) // s + 1; self.T2 = (self.T1 - k) // s + 1
        self.W = [rng.normal(0, np.sqrt(2 / (cin * k)), (cin * k, c1)).astype(np.float32),
                  rng.normal(0, np.sqrt(2 / (c1 * k)), (c1 * k, c2)).astype(np.float32),
                  rng.normal(0, np.sqrt(2 / c2), (c2, C)).astype(np.float32)]
        self.b = [np.zeros(c1, np.float32), np.zeros(c2, np.float32), np.zeros(C, np.float32)]
        self.sizes = [cin * T, c1, c2, C]

    def copy(self):
        m = TinyCNN1D.__new__(TinyCNN1D); m.__dict__.update(self.__dict__)
        m.W = [w.copy() for w in self.W]; m.b = [b.copy() for b in self.b]; m.sizes = list(self.sizes); return m

    def _im2col(self, h):            # h: (n, T, c) -> (n, Tout, k*c)
        n, T, c = h.shape; To = (T - self.k) // self.s + 1
        idx = np.arange(To)[:, None] * self.s + np.arange(self.k)[None, :]
        return h[:, idx, :].reshape(n, To, self.k * c)

    def _col2im(self, dcol, T, c):   # inverse scatter-add of _im2col
        n, To, _ = dcol.shape; d = dcol.reshape(n, To, self.k, c); out = np.zeros((n, T, c), np.float32)
        for j in range(self.k):
            out[:, j:j + self.s * (To - 1) + 1:self.s, :] += d[:, :, j, :]
        return out

    def _fwd(self, X):
        x = X.reshape(len(X), self.cin, self.T).transpose(0, 2, 1)       # (n, T, cin)
        c0 = self._im2col(x); z1 = c0 @ self.W[0] + self.b[0]; h1 = np.maximum(z1, 0)
        c1 = self._im2col(h1); z2 = c1 @ self.W[1] + self.b[1]; h2 = np.maximum(z2, 0)
        g = h2.mean(1); z3 = g @ self.W[2] + self.b[2]
        return dict(c0=c0, z1=z1, h1=h1, c1=c1, z2=z2, h2=h2, g=g, z3=z3)

    def forward(self, X): return [X, None, None, self._fwd(X)["z3"]]

    def proba(self, X):
        z = self._fwd(X)["z3"]; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def layer_macs(self):
        return [self.T1 * self.W[0].size, self.T2 * self.W[1].size, self.W[2].size]

    def macs_fwd(self): return sum(self.layer_macs())

    def macs_train_sample(self, depth):
        lm = self.layer_macs(); top = lm[self.L - depth:]
        return self.macs_fwd() + sum(top) + sum(top[1:])

    def n_params(self, depth=None):
        depth = self.L if depth is None else depth
        return sum(self.W[l].size + self.b[l].size for l in range(self.L - depth, self.L))

    def grads(self, X, y, depth):
        f = self._fwd(X); n = len(X)
        z = f["z3"] - f["z3"].max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
        d = p; d[np.arange(n), y] -= 1; d /= n
        gW, gb = {2: f["g"].T @ d}, {2: d.sum(0)}
        if depth >= 2:
            dg = d @ self.W[2].T                                              # (n, c2)
            dz2 = np.repeat(dg[:, None, :] / self.T2, self.T2, 1) * (f["z2"] > 0)
            gW[1] = f["c1"].reshape(-1, f["c1"].shape[2]).T @ dz2.reshape(-1, dz2.shape[2]); gb[1] = dz2.sum((0, 1))
            if depth >= 3:
                dh1 = self._col2im(dz2 @ self.W[1].T, self.T1, self.W[0].shape[1]); dz1 = dh1 * (f["z1"] > 0)
                gW[0] = f["c0"].reshape(-1, f["c0"].shape[2]).T @ dz1.reshape(-1, dz1.shape[2]); gb[0] = dz1.sum((0, 1))
        return gW, gb, loss

def _gradcheck():
    r = np.random.default_rng(0); m = TinyCNN1D(3, 20, 4, r, c1=4, c2=5, k=3, s=2)
    X = r.normal(size=(6, 60)).astype(np.float64); y = r.integers(0, 4, 6)
    m.W = [w.astype(np.float64) for w in m.W]; m.b = [b.astype(np.float64) for b in m.b]
    gW, gb, _ = m.grads(X, y, 3); worst = 0.0
    def L_():
        p = m.proba(X); return -np.log(p[np.arange(6), y]).mean()
    for l in range(3):
        for idx in [(0, 0), (1, 1), (-1, -1)]:
            o = m.W[l][idx]; m.W[l][idx] = o + 1e-6; a = L_(); m.W[l][idx] = o - 1e-6; b_ = L_(); m.W[l][idx] = o
            worst = max(worst, abs((a - b_) / 2e-6 - gW[l][idx]))
    return worst
print("C2 CNN GRADIENT CHECK: max abs error", f"{_gradcheck():.2e}")

def load_har_raw(seed):
    """UCI-HAR raw inertial signals (9 channels x 128 samples at 50 Hz), same subject order and split as UCI-HAR."""
    p = download("https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
                 os.path.join(DATA_DIR, "har.zip"))
    sig = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
    with zipfile.ZipFile(p) as z:
        inner = [n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip")]
        zz = zipfile.ZipFile(io.BytesIO(z.read(inner[0]))) if inner else z
        def rd(name):
            n_ = [m for m in zz.namelist() if m.endswith(name)][0]
            return pd.read_csv(io.BytesIO(zz.read(n_)), sep=r"\s+", header=None).values
        X = np.vstack([np.stack([rd(f"Inertial Signals/{s}_{part}.txt") for s in sig], 1) for part in ("train", "test")]).astype(np.float32)
        y = np.concatenate([rd("train/y_train.txt").ravel(), rd("test/y_test.txt").ravel()]) - 1
        s = np.concatenate([rd("train/subject_train.txt").ravel(), rd("test/subject_test.txt").ravel()])
    X = X.reshape(len(X), -1)                       # (n, 9*128), channel-major
    order = np.argsort(s, kind="stable"); X, y, s = X[order], y[order], s[order]
    subj = np.unique(s); pre = np.isin(s, subj[:6])
    Xs, ys, ss = X[~pre], y[~pre], s[~pre]
    drift = [i for i in range(1, len(ss)) if ss[i] != ss[i - 1]]
    return dict(name="HAR-raw-CNN", Xp=X[pre], yp=y[pre], Xs=Xs, ys=ys, drift=drift, C=6, cnn=(9, 128))

def make_synth_raw(name, seed):                        # smoke test only
    d = make_synthetic(name, seed, d=9 * 128, C=6); d["cnn"] = (9, 128); return d

LOADERS["HAR-raw-CNN"] = (lambda seed: make_synth_raw("HAR-raw-CNN", seed)) if SMOKE else load_har_raw
_orig_get = get_dataset
def get_dataset(name, seed):
    if name == "HAR-raw-CNN" and (name, seed) not in _DS_CACHE:
        _DS_CACHE[(name, seed)] = prepare(LOADERS[name](seed))
    return _orig_get(name, seed)

_orig_pretrained = pretrained_model
def pretrained_model(ds, seed):
    if "cnn" not in ds: return _orig_pretrained(ds, seed)
    key = (ds["name"], seed, "cnn")
    if key not in PRETRAINED:
        rng = np.random.default_rng(seed)
        n = len(ds["yp"]); perm = rng.permutation(n); nv = int(0.15 * n); tr, va = perm[nv:], perm[:nv]
        m = TinyCNN1D(ds["cnn"][0], ds["cnn"][1], ds["C"], rng)
        sgd_train(m, ds["Xp"][tr], ds["yp"][tr], m.L, CFG["pre_epochs"], CFG["pre_lr"], rng, batch=64)
        _, acc = loss_acc(m, ds["Xp"][va], ds["yp"][va])
        PRETRAINED[key] = (m, 1 - acc, norm_entropy(m.proba(ds["Xp"][va])))
        print(f"CNN pre-deployment validation accuracy (seed {seed}): {acc:.3f}; params {m.n_params()}, MACs/inference {m.macs_fwd()}", flush=True)
    return PRETRAINED[key]

def exp_C2(seeds):
    t0 = time.time(); rows = []
    M = ["Static", "Online-SGD", "TinyOL-Head", "RandomSkip", "Periodic-Full", "DriftTrig-Full", "CEGA-Tiny", "CEGA-Lite",
         "SB-Matched", "Entropy-Matched"]
    for seed in seeds:
        r, _ = run_methods4(get_dataset("HAR-raw-CNN", seed), seed, M); rows += r
        print(f"C2 seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT_C}/C2_cnn_har_raw.csv", index=False); return df
print("C cell ready")
