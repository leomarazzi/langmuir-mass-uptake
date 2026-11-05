"""B. Non-radial geometry: rectangle (0,a)x(0,b), a=2, b=1, n_inf=1, P1=(0.5,0.7,1.3) unless stated.
(1) c_k^2 = 64ab/(pi^4 i^2 j^2) for odd (i,j), 0 otherwise: checked by midpoint quadrature (4000 pts per axis)
    of the normalized eigenfunction (2/sqrt(ab)) sin(i pi x/a) sin(j pi y/b), and sum c_k^2 -> ab.
(2) series mass (mu <= 4e5) vs independent FD (fd_rect: PR-ADI + exact exchange, Strang) on 80x40 and 160x80.
(3) experiment A for the rectangle at P1 (FD 80x40 data), via A_noinvcrime.run(['rect'], trials, ..., ['P1']).
(4) short-time slope: M(t)/sqrt t vs (2/sqrt pi)|dOmega| sqrt D, series with mu <= 4e6 (and 1e6 for a check).
Usage: python3 B_rectangle.py [trials]   (default 20; seeds as in A_noinvcrime.py)"""
import math, sys, time
import geom, fd_rect, A_noinvcrime
a, b = 2.0, 1.0; D, g, be = 0.5, 0.7, 1.3; Minf = a*b*(1+g/be)
def main(trials):
    w = sys.stdout.write
    w('# B. rectangle a=2, b=1, P1=(0.5,0.7,1.3), M_inf=%.6f, perimeter=6\n' % Minf)
    w('## (1) c^2 check: quadrature of (int phi)^2 vs 64ab/(pi^4 i^2 j^2) (0 for an even index)\n')
    n = 4000
    def I1(i, L): h = L/n; return sum(math.sin(i*math.pi*(k+0.5)*h/L) for k in range(n))*h
    worst = 0.0
    for i, j in ((1, 1), (1, 2), (2, 1), (3, 1), (1, 3), (3, 5), (7, 3), (2, 2), (9, 11)):
        c = 2/math.sqrt(a*b)*I1(i, a)*I1(j, b); f = 64*a*b/(math.pi**4*i*i*j*j) if (i % 2 and j % 2) else 0.0
        w('(i,j)=(%d,%d) quad c^2=%.10f formula=%.10f\n' % (i, j, c*c, f)); worst = max(worst, abs(c*c-f))
    w('max abs difference = %.2e\n' % worst)
    for mm in (4e4, 4e5, 4e6):
        G = geom.rect(a, b, mm); w('mu<=%.0e: %d modes, sum c^2 = %.8f (ab = %g)\n' % (mm, len(G.md), G.c2sum, a*b))
    w('## (2) series vs FD, t in {0.01,0.05,0.2,1,3}; rel.diff = (M_FD - M_series)/M_inf\n')
    ts = [0.01, 0.05, 0.2, 1.0, 3.0]
    G = geom.rect(a, b, 4e5); se = geom.mass_curve(ts, G, D, g, be)
    G2 = geom.rect(a, b, 4e6); se2 = geom.mass_curve(ts, G2, D, g, be)
    fds = {}
    for nx, ny in ((80, 40), (160, 80)):
        t0 = time.time(); fds[nx] = fd_rect.fd_mass_rect(a, b, D, g, be, 1.0, ts, nx, ny); w('# FD %dx%d: %.0f s\n' % (nx, ny, time.time()-t0))
    w('t      M_series(mu<=4e5)  [mu<=4e6 diff]   M_FD80x40   rel.diff   M_FD160x80  rel.diff   Richardson(4*fine-coarse)/3 rel.diff\n')
    for t, s, s2 in zip(ts, se, se2):
        c, f = fds[80][t], fds[160][t]; r = (4*f-c)/3
        w('%-5g  %.8f  [%.1e]  %.8f  %+.2e  %.8f  %+.2e  %+.2e\n' % (t, s, s2-s, c, (c-s)/Minf, f, (f-s)/Minf, (r-s)/Minf))
    w('## (4) short-time slope; c0 = (2/sqrt pi)*|dOmega|*sqrt(D) = %.6f\n' % (2/math.sqrt(math.pi)*6*math.sqrt(D)))
    c0 = 2/math.sqrt(math.pi)*6*math.sqrt(D)
    tt = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    m4 = geom.mass_curve(tt, G2, D, g, be); G3 = geom.rect(a, b, 1e6); m1 = geom.mass_curve(tt, G3, D, g, be)
    w('t        M(t)/sqrt(t)   ratio to c0   kappa=(M - c0 sqrt t)/(D t)   [mu<=1e6 - mu<=4e6 in M]\n')
    for t, x, y in zip(tt, m4, m1):
        w('%-7g  %.6f      %.6f      %+.4f     [%.1e]\n' % (t, x/math.sqrt(t), x/math.sqrt(t)/c0, (x-c0*math.sqrt(t))/(D*t), y-x))
    w('## (3) experiment A on the rectangle at P1\n'); sys.stdout.flush()
    A_noinvcrime.run(['rect'], trials, sys.stdout, ['P1'])
if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
