"""A. No inverse crime: data from finite differences (fd_radial.fd_mass, Jn=400 / fd_rect, 80x40),
Gaussian noise sd = eta*M_inf, 160 times t_i = 20 (i/160)^2, fit by the series (geom.mass_curve) with
Levenberg-Marquardt in log-parameters started from the three-step estimate.
Usage: python3 A_noinvcrime.py [geoms] [trials]   e.g.  python3 A_noinvcrime.py slab,disk,ball 10
Seeds: noise for (geometry g, point p, eta e, trial k) uses random.Random(SEED0 + 100000*g + 1000*p + 100*e + k)."""
import sys, math, random, time, os
from multiprocessing import Pool
import geom, fd_radial, fd_rect
SEED0 = 20261104
PTS = {'P1': (0.5, 0.7, 1.3), 'P2': (2.0, 0.3, 0.5), 'P3': (0.1, 1.5, 2.0)}
GIDX = {'slab': 1, 'disk': 2, 'ball': 3, 'rect': 4}
ETAS = (0.0, 1e-3, 1e-2)
TS = geom.times_grid(20.0, 160)
def fit_geom(name):           # series used for FITTING
    if name == 'rect': return geom.rect(2.0, 1.0, 4e4)
    d = {'slab': 1, 'disk': 2, 'ball': 3}[name]; return geom.radial(d, 1.0, 120 if d == 2 else 200)
def ref_geom(name):           # long series used only to measure the FD error
    if name == 'rect': return geom.rect(2.0, 1.0, 4e5)
    d = {'slab': 1, 'disk': 2, 'ball': 3}[name]; return geom.radial(d, 1.0, 150 if d == 2 else 400)
def gen(job):
    name, pn = job; D, g, b = PTS[pn]; t0 = time.time()
    if name == 'rect': fd = fd_rect.fd_mass_rect(2.0, 1.0, D, g, b, 1.0, TS)
    else: fd = fd_radial.fd_mass({'slab': 1, 'disk': 2, 'ball': 3}[name], 1.0, D, g, b, 1.0, TS, Jn=400)
    data = [fd[t] for t in TS]
    G = ref_geom(name); se = geom.mass_curve(TS, G, D, g, b); Minf = G.V*(1+g/b)
    Gf = fit_geom(name); sf = geom.mass_curve(TS, Gf, D, g, b)
    err = max(abs(x-y) for x, y in zip(data, se))/Minf
    errlate = max(abs(x-y) for x, y, t in zip(data, se, TS) if t >= 0.05)/Minf
    trunc = max(abs(x-y) for x, y in zip(sf, se))/Minf
    return (name, pn), dict(data=data, Minf=Minf, fderr=err, fderr_late=errlate, trunc=trunc, sec=time.time()-t0)
def fit(task):
    name, pn, ei, k, data, Minf = task; D, g, b = PTS[pn]; eta = ETAS[ei]
    rng = random.Random(SEED0 + 100000*GIDX[name] + 1000*int(pn[1]) + 100*ei + k)
    Ms = [m + rng.gauss(0, eta*Minf) for m in data]
    G = fit_geom(name)
    e3 = geom.three_step(TS, Ms, G.V, G.P, G.mu1)
    p0 = [min(4.0, max(-6.0, math.log(x))) if (x == x and x > 0) else 0.0 for x in e3]
    fwd = lambda p: geom.mass_curve(TS, G, *[math.exp(x) for x in p])
    p, c = geom.lm(p0, fwd, Ms)
    po, co = geom.lm([math.log(D), math.log(g), math.log(b)], fwd, Ms)   # diagnostic: oracle start at the truth
    rel = lambda q: [abs(math.exp(x)-tv)/tv for x, tv in zip(q, (D, g, b))]
    return (name, pn, ei), rel(p), math.sqrt(c/len(TS))/Minf, rel(po), math.sqrt(co/len(TS))/Minf
def run(names, trials, out, pts=tuple(PTS)):
    t0 = time.time()
    jobs = [(n, p) for n in names for p in pts]
    with Pool(min(8, len(jobs))) as P: gd = dict(P.map(gen, jobs))
    tasks = []
    for (n, p), d in gd.items():
        for ei, eta in enumerate(ETAS):
            for k in range(trials if eta > 0 else 1): tasks.append((n, p, ei, k, d['data'], d['Minf']))
    with Pool(8) as P: res = P.map(fit, tasks, chunksize=1)
    agg = {}
    for key, e, rms, eo, rmso in res: agg.setdefault(key, []).append((e, rms, eo, rmso))
    med = lambda v: sorted(v)[len(v)//2] if len(v) % 2 else 0.5*(sorted(v)[len(v)//2-1]+sorted(v)[len(v)//2])
    w = out.write
    w('# A. No inverse crime. FD data (radial Jn=400 graded CN / rect 80x40 PR-ADI), noise sd=eta*M_inf, 160 times on [0,20] quadratic,\n')
    w('# fit: series (slab,ball K=200; disk K=120; rect mu<=4e4) by LM from three-step start (pipeline) and, as a diagnostic, from the true parameters (oracle start). trials=%d (eta=0: 1 deterministic fit). seed0=%d\n' % (trials, SEED0))
    w('# fderr = max_t |M_FD - M_series|/M_inf over the 160 times (series: slab,ball K=400, disk K=150, rect mu<=4e5); fderr_late: same for t>=0.05; trunc = fit-series truncation vs reference series\n')
    for n in names:
        for p in pts:
            d = gd[(n, p)]
            w('%s %s p=%s  fderr=%.2e fderr_late=%.2e trunc=%.1e  (FD %.0fs)\n' % (n, p, PTS[p], d['fderr'], d['fderr_late'], d['trunc'], d['sec']))
            for ei, eta in enumerate(ETAS):
                L = agg[(n, p, ei)]
                w('    eta=%-6g n=%2d median rel.err  D %.2e  gamma %.2e  beta %.2e   max: D %.2e gamma %.2e beta %.2e   median rms resid/M_inf %.1e\n' % (
                    eta, len(L), med([x[0][0] for x in L]), med([x[0][1] for x in L]), med([x[0][2] for x in L]),
                    max(x[0][0] for x in L), max(x[0][1] for x in L), max(x[0][2] for x in L), med([x[1] for x in L])))
                w('       oracle-start LM (diagnostic)   D %.2e  gamma %.2e  beta %.2e   max: D %.2e gamma %.2e beta %.2e   median rms resid/M_inf %.1e\n' % (
                    med([x[2][0] for x in L]), med([x[2][1] for x in L]), med([x[2][2] for x in L]),
                    max(x[2][0] for x in L), max(x[2][1] for x in L), max(x[2][2] for x in L), med([x[3] for x in L])))
    w('# wall time %.0f s\n' % (time.time()-t0))
if __name__ == '__main__':
    names = (sys.argv[1] if len(sys.argv) > 1 else 'slab,disk,ball').split(',')
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run(names, trials, sys.stdout)
