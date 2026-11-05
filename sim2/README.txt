sim2/ -- numerical experiments A-F for the Langmuir (Carter-Kibler) inverse problem.  Written 2026-11-04.
Python 3.14.4, standard library only (math, random, multiprocessing).  Run from this folder; `-B` keeps
bytecode out of ../sim (geom.py also sets sys.dont_write_bytecode).  ../sim/langmuir.py and ../sim/sweep.py
are imported read-only.  Model: u_t = D Lap u - g u + b N, N_t = g u - b N, u = n_inf = 1 on the boundary,
u = N = 0 at t = 0; M(t) = int (u+N).  A finite computation certifies only the cases it ran.

Shared modules
  geom.py       series mass with truncation tail, rectangle modes, LM (copy of ../sim/inverse.lm), three-step,
                3x3 Jacobi eigen-solver.
  fd_radial.py  radial FD = ../sim/langmuir.fd_mass with GRADED time steps k=min(5e-4, max(0.2h^2/D, 0.1t));
                the original fixed dt=5e-4 rings (CN) at t<0.03 and gives |dM|/M_inf up to 3e-2 at D=2.
  fd_rect.py    independent 2D FD on (0,2)x(0,1): Peaceman-Rachford ADI + exact exchange, Strang, graded steps,
                dtmax=2e-3, 2D trapezoid mass.

Script            command / wall time / seeds / exhaustiveness
A_noinvcrime.py   python3 -B A_noinvcrime.py slab,disk,ball 20 > A_noinvcrime.out      66 s (8 processes)
                  noise seed random.Random(20261104+100000*geom+1000*point+100*eta_index+trial); 20 trials per
                  eta>0, one deterministic fit at eta=0.  FD: Jn=400.  Not a search.
B_rectangle.py    python3 -B B_rectangle.py 20 > B_rectangle.out                          74 s (8 processes)
                  same seed rule (geom index 4); FD 80x40 (and 160x80 for convergence).
C_minf.py         python3 -B C_minf.py > C_minf.out                                       <1 s
                  seed random.Random(20261104+1000*window+100*eta_index+trial); 20 trials per cell;
                  6 windows (T,T1,T2) x 3 eta -- only these cells.
D_stability.py    python3 -B D_stability.py > D_stability.out (writes fig_stab.dat, fig_stab_slab.dat)   9 s
                  exhaustive over the 3x3x3 grid {0.3,0.55,0.8}x{0.4,0.7,1.0}x{0.8,1.3,1.8} only;
                  random directions seed random.Random(20261104).
E_figdata.py      python3 -B E_figdata.py (writes fig_mass.dat, fig_rates.dat, fig_cond.dat)   5 s; no randomness.
F_kc78.py         python3 -B F_kc78.py > F_kc78.out                                       <1 s; no randomness.
                  APPROXIMATE DIGITISATION by eye of Kibler-Carter 1978 Fig. 1 (100% RH), PDF page 13 (journal
                  (pdftoppm -f 13 -l 13 -r 400 -x 820 -y 460 -W 1440 -H 1260 -png).  4 LM starts only.
                  KC78's D is printed garbled ("9.090 X 10-1 cm2/day (1.409 X 10-5 in2/day)"); 9.09e-5 cm^2/day
                  is inferred from the inch value.

Data files (whitespace separated, '#' header): fig_mass.dat, fig_rates.dat, fig_cond.dat, fig_stab.dat (ball),
fig_stab_slab.dat.
