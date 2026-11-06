"""Check of the slab lower bound for the fast-exchange gap (Lemma 4.3 and Theorem 4.4 of the paper).
Per mode, with t* = 1/(beta+gamma) and m >= Lam = 2(1+rho)(beta+gamma):
   Y(t*;m) = X(t*;m) - (1+rho) e^{-m t*/(1+rho)} >= c1 rho,  c1 = e^{-1}(1-e^{-1/2})(1-e^{-1}).
Slab (-R,R), n_inf = 1:  sup_t (Meff - M) >= c1 rho * 4R/(pi (R sqrt(Lam/D) + pi)).
Pure Python 3."""
import math
c1 = math.exp(-1)*(1-math.exp(-0.5))*(1-math.exp(-1))
def modes(m, g, b):
    S = m+g+b; d = math.sqrt(S*S-4*m*b); sm = (S-d)/2; sp = (S+d)/2
    return sm, sp, sp*(b+g-sm)/(b*(sp-sm)), sm*(sp-b-g)/(b*(sp-sm))
def Y(t, m, g, b):
    rho = g/b; sm, sp, Am, Ap = modes(m, g, b)
    return Am*math.exp(-sm*t)+Ap*math.exp(-sp*t)-(1+rho)*math.exp(-m*t/(1+rho))
print("c1 =", c1)
print("per-mode: rho, beta, min over m in [Lam, 1e4 Lam] of Y(t*)/(c1 rho)")
for rho in (0.1, 1.0, 10.0):
    for b in (1.0, 50.0):
        g = rho*b; Lam = 2*(1+rho)*(b+g); ts = 1/(b+g)
        worst = min(Y(ts, Lam*10**(i/50), g, b)/(c1*rho) for i in range(0, 201))
        print(rho, b, round(worst, 4))
R = 1.0; K = 6000
mu = [((2*k-1)*math.pi/(2*R))**2 for k in range(1, K+1)]
c2 = [16*R/((2*k-1)**2*math.pi**2) for k in range(1, K+1)]
print("slab: rho, beta, D, sup(Meff-M), lower bound, ratio sup/lower")
for rho in (0.5, 2.0):
    for beta in (10.0, 100.0, 1000.0):
        D = 1.0+rho; g = rho*beta
        ts_grid = [10**(-6+7*i/300) for i in range(1, 300)]
        sup = max(sum(cc*Y(t, D*m, g, beta) for cc, m in zip(c2, mu)) for t in ts_grid)
        Lam = 2*(1+rho)*(beta+g)
        lower = c1*rho*4*R/(math.pi*(R*math.sqrt(Lam/D)+math.pi))
        print(rho, beta, D, f"{sup:.5f}", f"{lower:.5f}", f"{sup/lower:.2f}")
