"""F. APPROXIMATE DIGITISATION of Kibler & Carter (1978), Figure 1 (library/KiblerCarter78.pdf, PDF page 13 =
journal p. 129), curve '100% RH (immersion)', filled circles, 5208 resin at 24 C.  Read BY EYE from a 400-dpi
rendering (kc78zoomA-13.png, made with: pdftoppm -f 13 -l 13 -r 400 -x 820 -y 460 -W 1440 -H 1260 -png).
Axes: x = sqrt(t) [day^{1/2}], y = m_t [weight %].  Reading uncertainty (marker centre): about +-0.15 in x and
+-0.05 % in y (markers are ~0.3 x-units wide); NOT a measurement.  Thickness 2*delta = 0.086 cm for 100% RH
(journal p. 127), so slab half-thickness R = 0.043 cm.  Model: slab, Dirichlet n_inf, dry start; data map
m_t = c * M(t)/|Omega| with unknown scale c (so m_inf = c (1 + gamma/beta)); c profiled out by linear least squares.
LM in log(D,gamma,beta) (geom.lm), series K = 200.  Linearised standard errors from sigma^2 (J^T J)^{-1} with
the profiled model (c held at its optimum; approximate).  Usage: python3 F_kc78.py"""
import math
import geom
PTS = [(1.0, 1.50), (2.0, 3.40), (2.8, 4.85), (3.6, 5.90), (4.7, 6.22), (5.6, 6.35), (6.6, 6.60), (9.9, 6.70),
       (10.8, 6.68), (11.8, 6.78), (12.8, 6.85), (13.8, 6.92), (15.0, 7.07), (15.9, 7.08), (17.0, 7.08),
       (18.9, 7.12), (21.8, 7.15), (25.3, 7.42)]
TS = [x*x for x, _ in PTS]; Y = [y for _, y in PTS]; R = 0.043; G = geom.radial(1, R, 200)
def shape(p): return [m/G.V for m in geom.mass_curve(TS, G, *[math.exp(x) for x in p])]
def scale(f): return sum(a*b for a, b in zip(f, Y))/sum(a*a for a in f)
def fwd(p): f = shape(p); c = scale(f); return [c*x for x in f]
def report(tag, p):
    f = shape(p); c = scale(f); r = [c*x-y for x, y in zip(f, Y)]; rms = math.sqrt(sum(x*x for x in r)/len(r))
    D, g, b = [math.exp(x) for x in p]
    print('%s: D=%.4g cm^2/day gamma=%.4g /day beta=%.4g /day  c=%.4f%%  m1=c=%.3f%%  m_inf=c(1+g/b)=%.3f%%  rms resid=%.3f %%  max|resid|=%.3f %%' % (
        tag, D, g, b, c, c, c*(1+g/b), rms, max(abs(x) for x in r)))
    return rms
kc = [math.log(9.090e-5), math.log(3.488e-4), math.log(2.058e-3)]
print('# %d digitised points, t from %.1f to %.0f days' % (len(TS), TS[0], TS[-1]))
report('KC78 published parameters (c profiled)', kc)
best = None
for p0 in (kc, [math.log(1e-4), math.log(1e-3), math.log(1e-2)], [math.log(5e-5), math.log(1e-4), math.log(5e-4)], [math.log(2e-4), math.log(3e-3), math.log(1e-2)]):
    p, c2 = geom.lm(p0, fwd, Y, it=200)
    if best is None or c2 < best[1]: best = (p, c2)
    report('LM from start %s' % ['%.1e' % math.exp(x) for x in p0], p)
p = best[0]; rms = report('BEST', p)
base = fwd(p); J = []
for j in range(3):
    q = list(p); q[j] += 1e-5; J.append([(a-b_)/1e-5 for a, b_ in zip(fwd(q), base)])
F = [[sum(x*y for x, y in zip(J[i], J[k])) for k in range(3)] for i in range(3)]
s2 = rms*rms*len(Y)/(len(Y)-4)
inv = [geom.solve3(F, [1.0 if i == k else 0.0 for i in range(3)]) for k in range(3)]
print('linearised rel. std. errors (log-params, approx.): D %.2f  gamma %.2f  beta %.2f' % tuple(math.sqrt(s2*inv[k][k]) for k in range(3)))
ev, _ = geom.sym3_eig(F); print('Gram eigenvalue ratio (cond) = %.3g' % (ev[2]/ev[0]))
