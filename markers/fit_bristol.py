"""Fit Fick and Carter-Kibler (corrected Langmuir) slab models to the Bristol flax/epoxy immersion data.

Model (slab, thickness unknown, time in the file's unit): relative uptake
    C(t) = Cinf * [1 - sum_k w_k X(t; m1*(2k-1)^2) / (1+rho)],   w_k = 8/((2k-1)^2 pi^2),
X = A^- e^{-s_- t} + A^+ e^{-s_+ t} (companion paper, Thm A), m1 = D*mu_1 (D times first Dirichlet eigenvalue),
rho = gamma/beta. Fick: X -> (1+rho) e^{-m1 (2k-1)^2 t}, i.e. C = Cinf [1 - sum w_k e^{-m1 (2k-1)^2 t}].
Desorption: by the drying remark of the companion paper the normalised loss obeys the same law.

Fitting: least squares on C (in %), Nelder-Mead on log-parameters, several starts. Pure Python 3.
Noise floor: weights are integers in mg (balance 1 mg): quantisation sd ~0.29 mg per weighing.
Run: python3 fit_bristol.py  (reads data-bristol/*.csv next to this file)."""
import math, os, random, csv

HERE = os.path.dirname(os.path.abspath(__file__))
K = 80
W = [8 / ((2 * k - 1) ** 2 * math.pi ** 2) for k in range(1, K + 1)]


def read_blocks(path, desorb=False):
    """Return {group: [(spec, times, C%)]} ; C relative to W0 (absorption: weight at t=0;
    desorption: normalised loss fraction (Wsat - Wt)/(Wsat - Wdry) * 100 using the dry column)."""
    out = {}
    times = None
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            if row[0].startswith('Specimen'):
                vals = [c for c in row[1:]]
                if desorb:
                    times = [float(x) for x in vals[1:] if x.strip()]
                else:
                    times = [float(x) for x in vals if x.strip()]
                continue
            spec = row[0].strip()
            nums = [float(x) for x in row[1:] if x.strip()]
            g = spec.split('-')[0]
            if desorb:
                wdry, ws = nums[0], nums[1:]
                n = min(len(ws), len(times))
                ws, ts = ws[:n], times[:n]
                C = [(ws[0] - w) / wdry * 100 for w in ws]   # moisture lost, % of dry weight
                sd = 0.29 * math.sqrt(2) / wdry * 100
            else:
                n = min(len(nums), len(times))
                ws, ts = nums[:n], times[:n]
                C = [(w - ws[0]) / ws[0] * 100 for w in ws]
                sd = 0.29 * math.sqrt(2) / ws[0] * 100
            out.setdefault(g, []).append((spec, ts, C, sd))
    return out


def modes(m, g, b):
    S = m + g + b
    d = math.sqrt(S * S - 4 * m * b)
    sm, sp = (S - d) / 2, (S + d) / 2
    return sm, sp, sp * (b + g - sm) / (b * (sp - sm)), sm * (sp - b - g) / (b * (sp - sm))


def ck_curve(ts, Cinf, m1, g, b):
    rho = g / b
    md = [modes(m1 * (2 * k - 1) ** 2, g, b) for k in range(1, K + 1)]
    res = []
    for t in ts:
        if t == 0:
            res.append(0.0); continue
        s = 0.0
        for w, (sm, sp, Am, Ap) in zip(W, md):
            s += w * (Am * math.exp(-sm * t) + Ap * math.exp(-sp * t))
        res.append(Cinf * (1 - s / (1 + rho)))
    return res


def fick_curve(ts, Cinf, m1):
    return [0.0 if t == 0 else Cinf * (1 - sum(w * math.exp(-m1 * (2 * k - 1) ** 2 * t) for k, w in enumerate(W, 1))) for t in ts]


def nelder_mead(f, x0, step=0.5, iters=900, tol=1e-12):
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        c = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [c[j] + (c[j] - pts[-1][j]) for j in range(n)]; fr = f(xr)
        if fr < vals[0]:
            xe = [c[j] + 2 * (c[j] - pts[-1][j]) for j in range(n)]; fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [c[j] + 0.5 * (pts[-1][j] - c[j]) for j in range(n)]; fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [[pts[0][j] + 0.5 * (p[j] - pts[0][j]) for j in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i]


def fit(data, model, seed=1):
    """data: list of (ts, C). Shared parameters across the list. Returns params, rss, npts."""
    rnd = random.Random(seed)
    npts = sum(len(ts) for ts, _ in data)
    Cmax = max(max(C) for _, C in data)
    tmax = max(max(ts) for ts, _ in data)
    def rss_of(p):
        try:
            if model == 'fick':
                Cinf, m1 = math.exp(p[0]), math.exp(p[1]); cur = lambda ts: fick_curve(ts, Cinf, m1)
            else:
                Cinf, m1, g, b = (math.exp(x) for x in p); cur = lambda ts: ck_curve(ts, Cinf, m1, g, b)
            return sum(sum((a - b_) ** 2 for a, b_ in zip(cur(ts), C)) for ts, C in data)
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e30
    best = None
    starts = 5 if model == 'fick' else 8
    for _ in range(starts):
        x0 = [math.log(Cmax * rnd.uniform(0.9, 1.5)), math.log(rnd.uniform(0.3, 30) / tmax)]
        if model == 'ck':
            x0 += [math.log(rnd.uniform(0.05, 50) / tmax), math.log(rnd.uniform(0.05, 50) / tmax)]
        x, v = nelder_mead(rss_of, x0)
        x, v = nelder_mead(rss_of, x, step=0.1)
        if best is None or v < best[1]:
            best = (x, v)
    p = [math.exp(z) for z in best[0]]
    return p, best[1], npts


def main():
    sets = [('1st absorption', 'data-bristol/1st_absorb_weight.csv', False),
            ('10th absorption', 'data-bristol/10th_absorb_weight.csv', False),
            ('10th desorption', 'data-bristol/10th_desorb_weight.csv', True)]
    for title, path, des in sets:
        blocks = read_blocks(os.path.join(HERE, path), desorb=des)
        print(f'#### {title}  ({path})')
        for g in sorted(blocks):
            specs = blocks[g]
            sd = sum(s[3] for s in specs) / len(specs)
            print(f'group {g}: {len(specs)} specimens, {len(specs[0][1])} times, quantisation sd ~{sd:.3f} %')
            per = []
            for spec, ts, C, _ in specs:
                pf, rf, n = fit([(ts, C)], 'fick')
                pc, rc, _ = fit([(ts, C)], 'ck')
                per.append((spec, pf, rf, pc, rc, n))
                Cinf, m1, gm, bt = pc
                print(f'  {spec}: Fick Cinf={pf[0]:.3f}% m1={pf[1]:.4g} rmsd={math.sqrt(rf/n):.3f} | '
                      f'CK Cinf={Cinf:.3f}% m1={m1:.4g} rho={gm/bt:.3g} beta={bt:.4g} gamma={gm:.4g} '
                      f'Deff_m1={m1/(1+gm/bt):.4g} rmsd={math.sqrt(rc/n):.3f}')
            pf, rf, n = fit([(ts, C) for _, ts, C, _ in specs], 'fick')
            pc, rc, _ = fit([(ts, C) for _, ts, C, _ in specs], 'ck')
            Cinf, m1, gm, bt = pc
            # F-test style comparison (nested? CK -> Fick as gamma->0): report AIC
            aic_f = n * math.log(rf / n) + 2 * 2
            aic_c = n * math.log(rc / n) + 2 * 4
            print(f'  pooled: Fick Cinf={pf[0]:.3f}% m1={pf[1]:.4g} rmsd={math.sqrt(rf/n):.3f} AIC={aic_f:.1f} | '
                  f'CK Cinf={Cinf:.3f}% m1={m1:.4g} rho={gm/bt:.3g} beta={bt:.4g} Deff_m1={m1/(1+gm/bt):.4g} '
                  f'a*mu1={gm/bt*(m1/(1+gm/bt))/((1+gm/bt)*bt):.3g} rmsd={math.sqrt(rc/n):.3f} AIC={aic_c:.1f}')


if __name__ == '__main__':
    main()
