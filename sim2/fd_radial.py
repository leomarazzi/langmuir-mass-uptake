"""Radial finite differences for the Langmuir model (slab/disk/ball), adapted from ../sim/langmuir.fd_mass.
Same space discretisation (Jn cells, CN diffusion, exact exchange, Strang splitting); the only change is
GRADED time steps k = min(dtmax, max(0.2 h^2/D, 0.1 t)), which removes the Crank-Nicolson ringing caused by
the boundary/initial discontinuity at t = 0 (the original fixed dt = 5e-4 gives errors up to 3e-2 in M at
t < 0.03 when D = 2)."""
import math
def fd_mass(dim, R, D, g, b, ninf, times, Jn=200, dtmax=5e-4):
    h = R/Jn; r = [i*h for i in range(Jn+1)]
    u = [0.0]*(Jn+1); u[Jn] = ninf; N = [0.0]*(Jn+1); t = 0.0; out = {}; ti = 0; times = sorted(times)
    w = [(r[i]**(dim-1) if dim > 1 else 1.0) for i in range(Jn+1)]
    fac = {1: 2.0, 2: 2*math.pi, 3: 4*math.pi}[dim]
    tw = [fac*h*(0.5 if i in (0, Jn) else 1.0)*w[i] for i in range(Jn+1)]
    lo = [0.0]+[0.5*(1-(dim-1)*h/(2*r[i])) for i in range(1, Jn)]
    hi = [0.0]+[0.5*(1+(dim-1)*h/(2*r[i])) for i in range(1, Jn)]
    K = g+b
    def M(): return sum(a*(x+y) for a, x, y in zip(tw, u, N))
    def exch(kk):
        e = math.exp(-K*kk)
        for i in range(Jn):
            s = u[i]+N[i]; Nn = g*s/K+(N[i]-g*s/K)*e; N[i] = Nn; u[i] = s-Nn
        N[Jn] = g*ninf/b+(N[Jn]-g*ninf/b)*math.exp(-b*kk)
    kmin = 0.2*h*h/D
    while ti < len(times):
        if t >= times[ti]-1e-13: out[times[ti]] = M(); ti += 1; continue
        k = min(dtmax, max(kmin, 0.1*t), times[ti]-t); exch(k/2)
        mu = D*k/(h*h); a = [0.0]*Jn; bb = [0.0]*Jn; c = [0.0]*Jn; d = [0.0]*Jn
        bb[0] = 1+dim*mu; c[0] = -dim*mu; d[0] = (1-dim*mu)*u[0]+dim*mu*u[1]
        for i in range(1, Jn):
            l = mu*lo[i]; hh = mu*hi[i]
            a[i] = -l; bb[i] = 1+mu; c[i] = -hh; d[i] = l*u[i-1]+(1-mu)*u[i]+hh*u[i+1]
        d[Jn-1] += mu*hi[Jn-1]*ninf; c[Jn-1] = 0.0
        cp = [0.0]*Jn; dp = [0.0]*Jn; cp[0] = c[0]/bb[0]; dp[0] = d[0]/bb[0]
        for i in range(1, Jn):
            m = bb[i]-a[i]*cp[i-1]; cp[i] = c[i]/m; dp[i] = (d[i]-a[i]*dp[i-1])/m
        x = [0.0]*Jn; x[-1] = dp[-1]
        for i in range(Jn-2, -1, -1): x[i] = dp[i]-cp[i]*x[i+1]
        u = x+[ninf]; exch(k/2); t += k
    return out
