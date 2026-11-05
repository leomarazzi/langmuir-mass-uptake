"""Independent 2D finite-difference solver for the Langmuir model on the rectangle (0,a)x(0,b).
Uniform grid nx x ny cells (h = a/nx = b/ny required), u = ninf on the boundary, u = N = 0 inside at t = 0.
Diffusion: Peaceman-Rachford ADI on v = u - ninf (homogeneous Dirichlet).  Exchange: exact solution of
u' = -g u + b N, N' = g u - b N at each node (boundary nodes: u fixed, N' = g ninf - b N).  Strang
splitting exch(k/2) ADI(k) exch(k/2).  Graded steps k = min(dtmax, max(0.2 h^2/D, 0.1 t)).
Mass: 2D trapezoid rule of u + N over all nodes."""
import math
def _thomas_factor(n, a):
    # matrix tridiag(-a, 1+2a, -a) of size n
    cp = [0.0]*n; den = [0.0]*n
    m = 1+2*a; den[0] = m; cp[0] = -a/m
    for i in range(1, n):
        m = 1+2*a+a*cp[i-1]; den[i] = m; cp[i] = -a/m
    return cp, den
def _thomas_solve(cp, den, a, d):
    n = len(d); y = [0.0]*n; y[0] = d[0]/den[0]
    for i in range(1, n): y[i] = (d[i]+a*y[i-1])/den[i]
    for i in range(n-2, -1, -1): y[i] -= cp[i]*y[i+1]
    return y
def fd_mass_rect(a, b, D, g, be, ninf, times, nx=80, ny=40, dtmax=2e-3):
    h = a/nx; assert abs(b/ny-h) < 1e-12
    I, Jy = nx-1, ny-1                      # interior counts
    v = [[-ninf]*I for _ in range(Jy)]      # v = u - ninf, interior rows j (y), cols i (x)
    Nin = [[0.0]*I for _ in range(Jy)]      # N at interior nodes
    nbd = 2*(nx+ny)                         # boundary nodes (all have equal-ish weights; handled below)
    Nb = 0.0                                # N on boundary nodes (all identical: same ODE)
    K = g+be; t = 0.0; out = {}; ti = 0; times = sorted(times)
    # trapezoid weights: interior h^2; boundary edge nodes h^2/2; corners h^2/4.
    wb_total = h*h*(0.5*(2*(nx-1)+2*(ny-1)) + 0.25*4)   # = h * perimeter / 2
    def M():
        s = sum(sum(r1)+sum(r2) for r1, r2 in zip(v, Nin)) + I*Jy*ninf
        return h*h*s + wb_total*(ninf+Nb)
    def exch(kk):
        nonlocal Nb
        e = math.exp(-K*kk)
        for j in range(Jy):
            vr = v[j]; Nr = Nin[j]
            for i in range(I):
                s = vr[i]+ninf+Nr[i]; eq = g*s/K; Nn = eq+(Nr[i]-eq)*e; Nr[i] = Nn; vr[i] = s-Nn-ninf
        Nb = g*ninf/be+(Nb-g*ninf/be)*math.exp(-be*kk)
    kmin = 0.2*h*h/D
    while ti < len(times):
        if t >= times[ti]-1e-13: out[times[ti]] = M(); ti += 1; continue
        k = min(dtmax, max(kmin, 0.1*t), times[ti]-t)
        exch(k/2)
        al = 0.5*D*k/(h*h)
        cpx, dnx = _thomas_factor(I, al); cpy, dny = _thomas_factor(Jy, al)
        # half step 1: implicit in x, explicit in y
        vs = []
        for j in range(Jy):
            up = v[j+1] if j+1 < Jy else None; dn = v[j-1] if j > 0 else None; c = v[j]
            rhs = [c[i]*(1-2*al)+al*((up[i] if up else 0.0)+(dn[i] if dn else 0.0)) for i in range(I)]
            vs.append(_thomas_solve(cpx, dnx, al, rhs))
        # half step 2: implicit in y, explicit in x
        newcols = []
        for i in range(I):
            rhs = [vs[j][i]*(1-2*al)+al*((vs[j][i-1] if i > 0 else 0.0)+(vs[j][i+1] if i+1 < I else 0.0)) for j in range(Jy)]
            newcols.append(_thomas_solve(cpy, dny, al, rhs))
        v = [[newcols[i][j] for i in range(I)] for j in range(Jy)]
        exch(k/2); t += k
    return out
