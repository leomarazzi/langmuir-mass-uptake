"""D. Local Lipschitz constants L(p) = 1/sigma_min(J(p)), J = d(M_p|_I)/dp (p = (D,gamma,beta), absolute units)
into L^2(I), I = [0,20], 400 equispaced quadrature nodes (trapezoid weights).  J by central differences,
step 1e-4*p_j.  sigma_min by one-sided Jacobi SVD of W^{1/2} J (400x3), which avoids forming J^T J.
Grid: 3x3x3 on K = [0.3,0.8]x[0.4,1.0]x[0.8,1.8] (endpoints + midpoints; exhaustive over this grid only).
Nearby pairs: q = p + 1e-3 v, v = right singular vector of sigma_min and 4 random unit vectors
(seed random.Random(20261104)).  Series: K=200 modes + tail.  Usage: python3 D_stability.py"""
import math, random, sys
import geom
NT = 400; TS = [20.0*i/(NT-1) for i in range(NT)]; H = TS[1]
WQ = [H*(0.5 if i in (0, NT-1) else 1.0) for i in range(NT)]; SW = [math.sqrt(w) for w in WQ]
def curve(G, p): return geom.mass_curve(TS, G, *p)
def l2(a, b_): return math.sqrt(sum(w*(x-y)**2 for w, x, y in zip(WQ, a, b_)))
def jac(G, p):
    cols = []
    for j in range(3):
        hj = 1e-4*p[j]; q1 = list(p); q2 = list(p); q1[j] += hj; q2[j] -= hj
        a, b_ = curve(G, q1), curve(G, q2); cols.append([s*(x-y)/(2*hj) for s, x, y in zip(SW, a, b_)])
    return cols
def jacobi_svd(cols):
    A = [c[:] for c in cols]; V = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(60):
        rot = 0
        for i in range(3):
            for j in range(i+1, 3):
                al = sum(x*x for x in A[i]); be = sum(x*x for x in A[j]); ga = sum(x*y for x, y in zip(A[i], A[j]))
                if abs(ga) <= 1e-15*math.sqrt(al*be): continue
                rot += 1; z = (be-al)/(2*ga); t = math.copysign(1.0, z)/(abs(z)+math.sqrt(1+z*z))
                c = 1/math.sqrt(1+t*t); s = c*t
                A[i], A[j] = [c*x-s*y for x, y in zip(A[i], A[j])], [s*x+c*y for x, y in zip(A[i], A[j])]
                for k in range(3):
                    vi, vj = V[k][i], V[k][j]; V[k][i] = c*vi-s*vj; V[k][j] = s*vi+c*vj
        if rot == 0: break
    sv = [math.sqrt(sum(x*x for x in A[i])) for i in range(3)]
    k = min(range(3), key=lambda i: sv[i])
    return sorted(sv), [V[r][k] for r in range(3)]
def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def ranks(x): o = sorted(range(len(x)), key=lambda i: x[i]); r = [0]*len(x); [r.__setitem__(i, k) for k, i in enumerate(o)]; return r
def run(dim, datfile):
    G = geom.radial(dim, 1.0, 200); rng = random.Random(20261104)
    grid = [(D, g, b) for D in (0.3, 0.55, 0.8) for g in (0.4, 0.7, 1.0) for b in (0.8, 1.3, 1.8)]
    rows = []; worst_vmin = 0.0; worst_rand = 0.0; worst_lin = 0.0
    for p in grid:
        sv, vmin = jacobi_svd(jac(G, p)); L = 1/sv[0]
        s1 = geom.rates(p[0]*G.md[0][0], p[1], p[2])[0]; s2 = geom.rates(p[0]*G.md[1][0], p[1], p[2])[0]
        gp = s2-s1; ratio = p[0]*G.md[0][0]/(p[1]+p[2])
        Mp = curve(G, p); qs = [vmin]
        for _ in range(4):
            v = [rng.gauss(0, 1) for _ in range(3)]; n = math.sqrt(sum(x*x for x in v)); qs.append([x/n for x in v])
        rat = []
        for v in qs:
            q = [a+1e-3*x for a, x in zip(p, v)]; rat.append(1e-3/l2(curve(G, q), Mp))
        worst_vmin = max(worst_vmin, abs(rat[0]*sv[0]-1)); worst_rand = max(worst_rand, max(r*sv[0] for r in rat[1:]))
        rows.append((p, L, gp, ratio, sv, rat))
    Ls = [r[1] for r in rows]; gs = [r[2] for r in rows]; ra = [r[3] for r in rows]
    lg = [math.log(x) for x in Ls]
    out = ['# dim=%d (%s) R=1: L(p)=1/sigma_min over 3x3x3 grid; I=[0,20], 400 nodes' % (dim, G.name),
           '# max L = %.4f at p=%s ; min L = %.4f at p=%s' % (max(Ls), rows[Ls.index(max(Ls))][0], min(Ls), rows[Ls.index(min(Ls))][0]),
           '# Pearson(L,g)=%.3f  Pearson(L,Dmu1/(b+g))=%.3f  Pearson(logL,log g)=%.3f  Pearson(logL,log ratio)=%.3f  Spearman(L,g)=%.3f  Spearman(L,ratio)=%.3f' % (
               pearson(Ls, gs), pearson(Ls, ra), pearson(lg, [math.log(x) for x in gs]), pearson(lg, [math.log(x) for x in ra]),
               pearson(ranks(Ls), ranks(gs)), pearson(ranks(Ls), ranks(ra))),
           '# nearby pairs |p-q|=1e-3: max |ratio/L - 1| along v_min = %.2e ; max ratio/L over 4 random directions per point = %.4f' % (worst_vmin, worst_rand)]
    with open(datfile, 'w') as f:
        f.write('# fig_stab data: dim=%d (%s), R=1, I=[0,20], 400 nodes. columns: D gamma beta L g=s2^- - s1^- Dmu1/(beta+gamma) sigma1 sigma2 sigma3 ratio_vmin ratio_rand_max\n' % (dim, G.name))
        for p, L, gp, r_, sv, rat in rows:
            f.write('%.3f %.3f %.3f %.6f %.6f %.6f %.6e %.6e %.6e %.6f %.6f\n' % (*p, L, gp, r_, *sv, rat[0], max(rat[1:])))
    return out
if __name__ == '__main__':
    for line in run(3, 'fig_stab.dat') + run(1, 'fig_stab_slab.dat'): print(line)
