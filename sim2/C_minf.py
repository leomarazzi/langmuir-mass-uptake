"""C. Decay-rate estimator with ESTIMATED M_inf (ball R=1, P1 = (0.5,0.7,1.3)), data from the series (K=400).
M~(t) = M(t) + xi_t, xi_t ~ N(0, (eta M_inf)^2) independent at t = T, T1, T2.
rho^ = M~(T)/|Omega| - 1;  s^ = log(F~(T1)/F~(T2))/(T2-T1),  F~(t) = M~(T) - M~(t).
Bound: |s^-s_*| <= (W/w1) e^{-g T1}/(T2-T1) + 4 eps e^{s_* T2}/(w1 (T2-T1)),  eps = delta + eta M_inf,
delta = M_inf (e^{-s_* T} + eta);  valid if 2 eps <= w1 e^{-s_* T2}.  20 trials per cell, worst reported.
Seed: random.Random(20261104 + 1000*window_index + 100*eta_index + trial).  Usage: python3 C_minf.py"""
import math, random
import geom
from geom import rates, amps
D, g, b = 0.5, 0.7, 1.3; G = geom.radial(3, 1.0, 400); V = G.V; rho = g/b; Minf = (1+rho)*V
allr = sorted([(r, k, s) for k, (mu, _) in enumerate(G.md, 1) for s, r in zip('-+', rates(D*mu, g, b))])
s1 = rates(D*G.md[0][0], g, b)[0]; s2 = rates(D*G.md[1][0], g, b)[0]; s1p = rates(D*G.md[0][0], g, b)[1]
gap = s2-s1; w1 = G.md[0][1]*amps(D*G.md[0][0], g, b)[0]; W = (1+rho)*V-w1
print('# ball R=1 P1: s_*=s1-=%.6f  s2-=%.6f  s1+=%.6f  gap g=%.6f  w1=%.6f  W=%.6f  M_inf=%.6f' % (s1, s2, s1p, gap, w1, W, Minf))
print('# three smallest atoms: ' + ', '.join('s%d%s=%.4f' % (k, s, r) for r, k, s in allr[:3]) + '  (smallest s+ overall = %.4f > s2-: %s)' % (min(r for r, k, s in allr if s == '+'), min(r for r, k, s in allr if s == '+') > s2))
M = lambda t: geom.mass_curve([t], G, D, g, b)[0]
WIN = [(20, 1, 3), (20, 2, 4), (20, 2, 6), (20, 5, 10), (10, 3, 6), (30, 10, 20)]
ETA = (0.0, 1e-4, 1e-3)
print('# columns: T T1 T2 eta | worst|s^-s_*| over 20 trials | bound | valid(2eps<=w1 e^{-s*T2}) | obs<=bound | worst|rho^-rho| | worst max|xi|/(eta M_inf) | #trials with F~<=0')
for wi, (T, T1, T2) in enumerate(WIN):
    MT, M1, M2 = M(T), M(T1), M(T2)
    for ei, eta in enumerate(ETA):
        delta = Minf*(math.exp(-s1*T)+eta); eps = delta+eta*Minf
        bound = (W/w1)*math.exp(-gap*T1)/(T2-T1) + 4*eps*math.exp(s1*T2)/(w1*(T2-T1))
        valid = 2*eps <= w1*math.exp(-s1*T2)
        worst = 0.0; worstr = 0.0; xr = 0.0; bad = 0
        for k in range(20 if eta > 0 else 1):
            rng = random.Random(20261104+1000*wi+100*ei+k)
            xi = [rng.gauss(0, eta*Minf) for _ in range(3)]
            mT, m1, m2 = MT+xi[0], M1+xi[1], M2+xi[2]
            worstr = max(worstr, abs(mT/V-1-rho))
            if eta > 0: xr = max(xr, max(abs(x) for x in xi)/(eta*Minf))
            F1, F2 = mT-m1, mT-m2
            if F1 <= 0 or F2 <= 0: bad += 1; continue
            worst = max(worst, abs(math.log(F1/F2)/(T2-T1)-s1))
        print('%2d %2d %2d %-6g | %.3e | %.3e | %-5s | %-5s | %.2e | %.2f | %d' % (T, T1, T2, eta, worst, bound, valid, worst <= bound, worstr, xr, bad))
