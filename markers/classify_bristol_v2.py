"""Analysis of Section 7 of the paper (open flax/epoxy data):
- Fits as in fit_bristol.py: Fick (5 starts) and Carter-Kibler (8 starts), Nelder-Mead on log-parameters,
  900 iterations per run plus a 900-iteration restart with step 0.1, relative tolerance 1e-12; 80 slab modes.
- Degenerate CK fit: D mu_1 > 1 or rho > 100 (file time units), or C_inf > 2 x Fick C_inf.
- For every degenerate fit report a*mu_1 = rho D_eff mu_1/((1+rho) beta) (fast-exchange parameter of
  Theorem 4.2) and beta*T_last (slow-release parameter of Corollary 5.3).
- LOO nearest centroid with standardisation computed INSIDE the loop (training specimens only).
- Permutation test (2000 label permutations within the merged two-class problem) for LOO accuracy.
- Bootstrap (2000 resamples of specimens within each group) of group medians of the markers.
- rmsd in percentage points of relative uptake (C = 100 (W_t - W_0)/W_0).
Run: python3 classify_bristol_v2.py"""
import math, os, random, statistics as st
import fit_bristol as F

def feats(path, desorb):
    rows = []
    for g, specs in sorted(F.read_blocks(os.path.join(F.HERE, path), desorb=desorb).items()):
        for spec, ts, C, _ in specs:
            pf, rf, n = F.fit([(ts, C)], 'fick'); pc, rc, _ = F.fit([(ts, C)], 'ck')
            Cinf, m1, gm, bt = pc; rho = gm/bt; deff = m1/(1+rho)
            rows.append(dict(g=g, spec=spec, fC=pf[0], fm=pf[1], frms=math.sqrt(rf/n), C=Cinf, m=m1, rho=rho,
                             beta=bt, deff=deff, crms=math.sqrt(rc/n), amu=rho*deff/((1+rho)*bt), bT=bt*max(ts),
                             degen=(m1 > 1 or rho > 100 or Cinf > 2*pf[0])))
    return rows

def loo(rows, keys, label):
    X = [[math.log(r[k]) for k in keys] for r in rows]; ok = 0
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        mu = [st.mean(X[j][c] for j in tr) for c in range(len(keys))]
        sd = [st.pstdev([X[j][c] for j in tr]) or 1 for c in range(len(keys))]
        z = lambda v: [(a-m)/s for a, m, s in zip(v, mu, sd)]
        cents = {}
        for j in tr: cents.setdefault(label[j], []).append(z(X[j]))
        cents = {k: [st.mean(c) for c in zip(*v)] for k, v in cents.items()}
        zi = z(X[i]); pred = min(cents, key=lambda k: sum((a-b)**2 for a, b in zip(zi, cents[k])))
        ok += pred == label[i]
    return ok

def perm_p(rows, keys, label, n=2000, seed=7):
    obs = loo(rows, keys, label); rnd = random.Random(seed); cnt = 0
    for _ in range(n):
        lab = label[:]; rnd.shuffle(lab); cnt += loo(rows, keys, lab) >= obs
    return obs, (cnt+1)/(n+1)

def boot(rows, key, n=2000, seed=11):
    rnd = random.Random(seed); out = {}
    for g in sorted({r['g'] for r in rows}):
        v = [r[key] for r in rows if r['g'] == g]
        meds = sorted(st.median([rnd.choice(v) for _ in v]) for _ in range(n))
        out[g] = (st.median(v), meds[int(0.025*n)], meds[int(0.975*n)])
    return out

for title, path, des in [('1st absorption', 'data-bristol/1st_absorb_weight.csv', False),
                         ('10th absorption', 'data-bristol/10th_absorb_weight.csv', False),
                         ('10th desorption', 'data-bristol/10th_desorb_weight.csv', True)]:
    rows = feats(path, des)
    print(f'#### {title}')
    print(' rmsd (percentage points): Fick median %.3f, CK median %.3f' % (st.median(r['frms'] for r in rows), st.median(r['crms'] for r in rows)))
    for r in rows:
        if r['degen']:
            print(f"  degenerate {r['spec']}: Dmu1={r['m']:.3g} rho={r['rho']:.3g} beta={r['beta']:.3g} Cinf={r['C']:.3g} (Fick {r['fC']:.3g}) a*mu1={r['amu']:.3g} beta*T={r['bT']:.3g}")
    for key in ('fC', 'fm'):
        print(' bootstrap median [95%]', key, {g: tuple(round(x, 5) for x in v) for g, v in boot(rows, key).items()})
    good = [r for r in rows if not r['degen']]
    for key in ('rho', 'beta', 'deff'):
        print(' bootstrap median [95%] (non-degenerate CK)', key, {g: tuple(round(x, 5) for x in v) for g, v in boot(good, key).items()})
    lab4 = [r['g'] for r in rows]; lab2 = ['A' if r['g'] in ('5', '6') else 'B' for r in rows]
    lab2g = ['A' if r['g'] in ('5', '6') else 'B' for r in good]
    print(' LOO 4 groups Fick(Cinf,m1): %d/%d' % (loo(rows, ['fC', 'fm'], lab4), len(rows)))
    o, p = perm_p(rows, ['fC', 'fm'], lab2); print(' LOO {5,6} vs {7,8} Fick(Cinf,m1): %d/%d, permutation p=%.4f' % (o, len(rows), p))
    o, p = perm_p(good, ['rho', 'beta', 'deff'], lab2g, n=500); print(' LOO {5,6} vs {7,8} CK(rho,beta,Deff) non-degenerate: %d/%d, permutation p=%.4f' % (o, len(good), p))
