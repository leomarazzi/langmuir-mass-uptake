"""Check of the slow-release theorem (Theorem 5.2 of the paper) on the slab (-1,1), n_inf = 1:
  -(1/2) beta gamma |M| t^2 <= M_beta(t) - M_0(t) <= 0  for all t >= 0,
where M_0 is the irreversible-trap mass (beta = 0):
  M_0(t) = sum c_k^2 [ m g t/(m+g) + m^2/(m+g)^2 (1 - e^{-(m+g)t}) ],  m = D mu_k,
derived from M_hat = n (q/tau^2) sum c_k^2 m/(m+q), q = tau + g. Also checks M_0 against the beta -> 0
limit of Theorem A, and reports the ratio of the gap to the bound. Pure Python 3."""
import math
R = 1.0; K = 3000
mu = [((2*k-1)*math.pi/(2*R))**2 for k in range(1, K+1)]
c2 = [16*R/((2*k-1)**2*math.pi**2) for k in range(1, K+1)]
vol = 2*R
def M0(t, D, g):
    return sum(cc*(m*g*t/(m+g) + (m/(m+g))**2*(1-math.exp(-(m+g)*t))) for cc, m in ((cc, D*mm) for cc, mm in zip(c2, mu)))
def Mb(t, D, g, b):
    rho = g/b; s = 0
    for cc, mm in zip(c2, mu):
        m = D*mm; S = m+g+b; d = math.sqrt(S*S-4*m*b); sm = (S-d)/2; sp = (S+d)/2
        # stable form of 1+rho - X:  use x1 + x2 via residues of (q/tau)^2/(m+q) + rho/(tau+b) is ill-conditioned;
        # here b is moderate in the checks below
        Am = sp*(b+g-sm)/(b*(sp-sm)); Ap = sm*(sp-b-g)/(b*(sp-sm))
        s += cc*(1+rho - Am*math.exp(-sm*t) - Ap*math.exp(-sp*t))
    return s
D, g = 1.0, 2.0
print("limit check M_beta -> M_0 as beta -> 0 at t=1:", M0(1, D, g), [Mb(1, D, g, b) for b in (1e-2, 1e-3, 1e-4)])
print("D, gamma, beta, T, max(M_b-M_0) [<=0], min(M_b-M_0), bound -(1/2) b g |M| T^2, min/bound")
for (D, g) in ((1.0, 2.0), (0.1, 0.5), (5.0, 10.0)):
    for b in (1e-2, 1e-1, 1.0):
        for T in (0.5, 2.0, 10.0):
            vals = [Mb(t, D, g, b)-M0(t, D, g) for t in (T*i/60 for i in range(1, 61))]
            bound = -0.5*b*g*vol*T*T
            print(D, g, b, T, f"{max(vals):.3e}", f"{min(vals):.3e}", f"{bound:.3e}", f"{min(vals)/bound:.3f}")
