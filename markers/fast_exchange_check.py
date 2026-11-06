"""Check of the fast-exchange bound (Lemma 4.1 and Theorem 4.2 of the paper) on the slab (-R,R).
Per-mode: 0 <= X(t;m) - (1+rho) e^{-m t/(1+rho)} <= min(1+rho, rho m/(beta+gamma)).
Summed: 0 <= Meff - M <= e/(e-1) (1+rho) n (|M| - H(a)),  a = rho D/((1+rho)^2 beta).
Pure Python 3, no dependencies. Run: python3 fast_exchange_check.py"""
import math
def modes(m,g,b):
    S=m+g+b; d=math.sqrt(S*S-4*m*b); sm=(S-d)/2; sp=(S+d)/2
    return sm,sp,sp*(b+g-sm)/(b*(sp-sm)),sm*(sp-b-g)/(b*(sp-sm))
def per_mode(rho,e,b=1.0):
    g=rho*b; m=e*b; sm,sp,Am,Ap=modes(m,g,b); worst=0; lo=0
    T=50/min(sm,m/(1+rho))
    for i in range(1,20001):
        t=T*(i/20000)**3
        d=Am*math.exp(-sm*t)+Ap*math.exp(-sp*t)-(1+rho)*math.exp(-m*t/(1+rho))
        worst=max(worst,d); lo=min(lo,d)
    return worst,lo,min(1+rho,rho*m/(b+g))
print("per-mode: rho, m/beta, sup diff, min diff, bound")
for rho in (0.2,1,5):
    for e in (1e-3,1e-1,1,10,100):
        print(rho,e,*per_mode(rho,e))
R=1.0; K=4000
mu=[((2*k-1)*math.pi/(2*R))**2 for k in range(1,K+1)]; c2=[16*R/((2*k-1)**2*math.pi**2) for k in range(1,K+1)]
vol=2*R
def H(a): return sum(cc*math.exp(-m*a) for cc,m in zip(c2,mu))
print("summed (slab R=1, n=1, Deff=1): rho, beta, sup(Meff-M), min(Meff-M), bound, sup/sqrt(rho D/beta)")
for rho in (0.5,2.0):
    for beta in (1e1,1e2,1e3,1e4):
        g=rho*beta; D=1.0*(1+rho); md=[modes(D*m,g,beta) for m in mu]; worst=0; lo=0
        for i in range(1,400):
            t=10**(-6+7*i/400)
            d=sum(cc*(Am*math.exp(-sm*t)+Ap*math.exp(-sp*t)-(1+rho)*math.exp(-m*t)) for cc,m,(sm,sp,Am,Ap) in zip(c2,mu,md))
            worst=max(worst,d); lo=min(lo,d)
        a=rho*D/((1+rho)**2*beta)
        print(rho,beta,worst,lo,(1+rho)*math.e/(math.e-1)*(vol-H(a)),worst/math.sqrt(rho*D/beta))
