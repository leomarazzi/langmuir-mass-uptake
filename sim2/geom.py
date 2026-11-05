"""Shared helpers for sim2 (referee experiments). Pure Python 3 standard library.
Geometries: slab (-R,R), disk, ball of radius R (via ../sim/langmuir.py), rectangle (0,a)x(0,b).
Series mass: (M_inf - M(t))/ninf = sum_k c_k^2 (A-_k e^{-s-_k t} + A+_k e^{-s+_k t}) + tail,
tail = (|Omega| - sum_k c_k^2) * (gamma/beta) e^{-beta t} (slow part of the truncated modes; same
convention as langmuir.deficit).  ninf = 1 throughout."""
import sys, os, math
sys.dont_write_bytecode = True          # never write .pyc into ../sim (read-only)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sim'))
import langmuir as LG                    # noqa: E402  (read-only import)
from langmuir import rates, amps         # noqa: E402

class Geom:
    def __init__(self, name, md, V, P):
        self.name, self.md, self.V, self.P = name, sorted(md), V, P
        self.mu1 = self.md[0][0]
        self.c2sum = sum(c for _, c in self.md)

def radial(dim, R, K):
    md = LG._modes(dim, R, K)
    return Geom({1: 'slab', 2: 'disk', 3: 'ball'}[dim], md, LG.volume(dim, R), LG.area(dim, R))

def rect_modes(a, b, mumax):
    """Odd (i,j) only; mu = pi^2 (i^2/a^2 + j^2/b^2); c^2 = 64ab/(pi^4 i^2 j^2)."""
    out = []; i = 1
    while math.pi**2 * (i*i/(a*a) + 1/(b*b)) <= mumax:
        j = 1
        while True:
            mu = math.pi**2 * (i*i/(a*a) + j*j/(b*b))
            if mu > mumax: break
            out.append((mu, 64*a*b/(math.pi**4 * i*i * j*j))); j += 2
        i += 2
    return out

def rect(a, b, mumax):
    return Geom('rect', rect_modes(a, b, mumax), a*b, 2*(a+b))

def mass_curve(ts, G, D, g, b):
    """Series mass M(t) at the times ts (ninf = 1)."""
    pre = []
    for mu, c2 in G.md:
        Am, Ap, sm, sp = amps(D*mu, g, b); pre.append((c2*Am, sm, c2*Ap, sp))
    tail = G.V - G.c2sum; Minf = G.V*(1+g/b); out = []
    ex = math.exp
    for t in ts:
        s = 0.0
        for wm, sm, wp, sp in pre:
            s += wm*ex(-sm*t)
            x = sp*t
            if x < 700: s += wp*ex(-x)
        s += tail*((g/b)*ex(-b*t) + (1.0 if t == 0 else 0.0))
        out.append(Minf - s)
    return out

def times_grid(T, n): return [T*(i/n)**2 for i in range(1, n+1)]

def solve3(A, bv):
    A = [r[:] for r in A]; bv = list(bv); n = 3
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i])); A[i], A[piv] = A[piv], A[i]; bv[i], bv[piv] = bv[piv], bv[i]
        for r in range(i+1, n):
            f = A[r][i]/A[i][i]
            for c in range(i, n): A[r][c] -= f*A[i][c]
            bv[r] -= f*bv[i]
    x = [0.0]*n
    for i in range(n-1, -1, -1): x[i] = (bv[i]-sum(A[i][c]*x[c] for c in range(i+1, n)))/A[i][i]
    return x

def lm(p0, fwd, Ms, it=40):
    """Levenberg-Marquardt in log parameters (copy of ../sim/inverse.lm with a generic forward map)."""
    p = list(p0); lam = 1e-3
    def res(p): return [f-m for f, m in zip(fwd(p), Ms)]
    r = res(p); c = sum(x*x for x in r)
    for _ in range(it):
        Jc = []
        for j in range(3):
            q = list(p); q[j] += 1e-6; rj = res(q); Jc.append([(a-bb)/1e-6 for a, bb in zip(rj, r)])
        JTJ = [[sum(x*y for x, y in zip(Jc[i], Jc[j])) for j in range(3)] for i in range(3)]
        JTr = [sum(x*y for x, y in zip(Jc[i], r)) for i in range(3)]
        dp = [0.0]*3
        while True:
            A = [[JTJ[i][j]+(lam*JTJ[i][i] if i == j else 0) for j in range(3)] for i in range(3)]
            dp = solve3(A, [-x for x in JTr]); nrm = max(abs(x) for x in dp)
            if nrm > 0.5: dp = [x*0.5/nrm for x in dp]
            q = [min(6.0, max(-12.0, p[i]+dp[i])) for i in range(3)]
            rq = res(q); cq = sum(x*x for x in rq)
            if cq < c: p, r, c = q, rq, cq; lam = max(lam/3, 1e-9); break
            lam *= 4
            if lam > 1e8: return p, c
        if max(abs(x) for x in dp) < 1e-9: break
    return p, c

def three_step(ts, Ms, V, A, mu1):
    """Copy of ../sim/inverse.three_step with |Omega|, |dOmega|, mu1 passed explicitly."""
    tail = Ms[int(0.9*len(Ms)):]; Minf = sum(tail)/len(tail); rho = Minf/V-1
    te = 0.02; E = [(t, m) for t, m in zip(ts, Ms) if t <= te]
    S11 = sum(t for t, _ in E); S12 = sum(t**1.5 for t, _ in E); S22 = sum(t*t for t, _ in E)
    y1 = sum(math.sqrt(t)*m for t, m in E); y2 = sum(t*m for t, m in E)
    det = S11*S22-S12*S12; a = (y1*S22-y2*S12)/det
    D = (a*math.sqrt(math.pi)/(2*A))**2
    L = [(t, math.log(Minf-m)) for t, m in zip(ts, Ms) if Minf-m > 0.02*Minf and t > 0.3*ts[-1]*0.2]
    L = L[len(L)//2:]
    if len(L) < 2: return D, float('nan'), float('nan')
    n = len(L); mt = sum(t for t, _ in L)/n; ml = sum(l for _, l in L)/n
    s = -sum((t-mt)*(l-ml) for t, l in L)/sum((t-mt)**2 for t, _ in L)
    m1 = D*mu1; den = m1-(1+rho)*s; b = s*(m1-s)/den if den != 0 else float('nan')
    return D, rho*b, b

def sym3_eig(F):
    """Eigenvalues/vectors of a symmetric 3x3 matrix by cyclic Jacobi rotations."""
    a = [row[:] for row in F]; V = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(100):
        off = sum(a[i][j]**2 for i in range(3) for j in range(3) if i != j)
        if off < 1e-30*sum(a[i][i]**2 for i in range(3)): break
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            if a[i][j] == 0.0: continue
            th = 0.5*math.atan2(2*a[i][j], a[j][j]-a[i][i]); c, s = math.cos(th), math.sin(th)
            for k in range(3):
                aik, ajk = a[i][k], a[j][k]; a[i][k] = c*aik-s*ajk; a[j][k] = s*aik+c*ajk
            for k in range(3):
                aki, akj = a[k][i], a[k][j]; a[k][i] = c*aki-s*akj; a[k][j] = s*aki+c*akj
            for k in range(3):
                vki, vkj = V[k][i], V[k][j]; V[k][i] = c*vki-s*vkj; V[k][j] = s*vki+c*vkj
    ev = [a[i][i] for i in range(3)]
    order = sorted(range(3), key=lambda i: ev[i])
    return [ev[i] for i in order], [[V[k][i] for k in range(3)] for i in order]
