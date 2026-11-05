"""E. Figure data for TikZ (whitespace-separated, '#' header).  Usage: python3 E_figdata.py
fig_mass.dat: t, M/M_inf for slab, disk, ball at P1, R=1, 200 points on [0,10] (series K=400/150/400).
fig_rates.dat: k, s_k^-, s_k^+ for the ball R=1 at P1, k=1..40.
fig_cond.dat: ratio = D mu1/(beta+gamma) on 12 log-spaced values in [0.1,10]; Cramer-Rao relative sd of
  (D, gamma, beta) at eta=1e-3, T=8/s1^-, 160 quadratic times, D=0.5,gamma=0.7,beta=1.3, R chosen per
  geometry so that D mu1 matches (as ../sim/sweep.py; uses sweep.crlb, 100 modes)."""
import math
import geom                      # sets dont_write_bytecode and the ../sim path
import sweep                     # read-only import from ../sim
D, g, b = 0.5, 0.7, 1.3
with open('fig_mass.dat', 'w') as f:
    f.write('# t  M/M_inf(slab)  M/M_inf(disk)  M/M_inf(ball); P1=(D,gamma,beta)=(0.5,0.7,1.3), R=1, n_inf=1, series\n')
    ts = [10*i/199 for i in range(200)]; cols = []
    for dim in (1, 2, 3):
        G = geom.radial(dim, 1.0, 150 if dim == 2 else 400); cols.append([max(0.0, m/(G.V*(1+g/b))) for m in geom.mass_curve(ts, G, D, g, b)])
    for i, t in enumerate(ts): f.write('%.6f %.8f %.8f %.8f\n' % (t, cols[0][i]+0.0, cols[1][i]+0.0, cols[2][i]+0.0))
with open('fig_rates.dat', 'w') as f:
    f.write('# k  s_k^-  s_k^+ ; ball R=1, mu_k = k^2 pi^2, P1=(0.5,0.7,1.3); s_k^- -> beta = 1.3 from below\n')
    for k in range(1, 41):
        sm, sp = geom.rates(D*k*k*math.pi**2, g, b); f.write('%d %.8f %.8f\n' % (k, sm, sp))
mu_ref = {1: math.pi**2/4, 2: 2.404825557695773**2, 3: math.pi**2}
with open('fig_cond.dat', 'w') as f:
    f.write('# ratio=Dmu1/(beta+gamma)  CRLB rel.sd of beta: slab disk ball | of D: slab disk ball | of gamma: slab disk ball ; eta=1e-3, T=8/s1^-, P1, R=sqrt(D mu_ref/m1)\n')
    for i in range(12):
        ratio = 10**(-1+2*i/11); m1 = ratio*(b+g); v = {}
        for dim in (1, 2, 3):
            R = math.sqrt(D*mu_ref[dim]/m1); v[dim] = sweep.crlb(dim, R, D, g, b)
        f.write('%.6f  %.4e %.4e %.4e  %.4e %.4e %.4e  %.4e %.4e %.4e\n' % (ratio, v[1][2], v[2][2], v[3][2], v[1][0], v[2][0], v[3][0], v[1][1], v[2][1], v[3][1]))
