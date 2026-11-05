"""Error analysis of the three-step reconstruction and empirical Lipschitz stability (ball, R=1).
(a) rho from M(T): |rho_hat - rho| vs bound (1+rho) e^{-s_* T}.
(b) decay-rate estimator over [T1,T2]: s_hat - s_* vs bound (W/w1) e^{-g T1}/(T2-T1), g = s-(2)-s-(1).
(c) sqrt(D) from M(t)/(c sqrt t), c = (2/sqrt pi)|dOmega|: bias vs t; noise eta: optimal t and error ~ sqrt(eta).
(d) Lipschitz: max |p-p'| / ||M_p - M_p'||_{L2(0,20)} over random pairs in K = [0.3,0.8]x[0.4,1.0]x[0.8,1.8]."""
import math, random
from langmuir import modes, volume, area, mass, rates, amps, deficit
dim,R=3,1.0; D,g,b=0.5,0.7,1.3; md=modes(dim,R,400); V=volume(dim,R); A=area(dim,R); rho=g/b
mu1=md[0][0]; s1,_=rates(D*mu1,g,b); s2,_=rates(D*md[1][0],g,b)
Am1,_,_,_=amps(D*mu1,g,b); w1=md[0][1]*Am1; W=(1+rho)*V-w1; gap=s2-s1
print('# ball R=1, D=%g gamma=%g beta=%g: s_*=%.4f gap g=%.4f w1=%.4f W=%.4f'%(D,g,b,s1,gap,w1,W))
print('## (a) rho from M(T)')
for T in (2,5,10,20):
    rh=mass(T,dim,R,D,g,b,md=md)/V-1; print('T=%2d |rho_hat-rho|=%.3e  bound=%.3e'%(T,abs(rh-rho),(1+rho)*math.exp(-s1*T)))
print('## (b) decay-rate estimator (exact M_inf)')
Minf=(1+rho)*V
for (T1,T2) in ((2,6),(5,10),(10,20),(20,40)):
    F1=deficit(T1,dim,R,D,g,b,md=md); F2=deficit(T2,dim,R,D,g,b,md=md)
    sh=math.log(F1/F2)/(T2-T1); print('[%2d,%2d] s_hat-s_*=%.3e  bound=%.3e'%(T1,T2,sh-s1,(W/w1)*math.exp(-gap*T1)/(T2-T1)))
print('## (c) sqrt(D) estimator from M(t)/(c sqrt t)')
c=2/math.sqrt(math.pi)*A
for t in (1e-4,1e-3,1e-2,1e-1):
    est=mass(t,dim,R,D,g,b,md=md)/(c*math.sqrt(t)); print('t=%.0e  sqrtD_hat-sqrtD=%+.3e  (|err|/sqrt t=%.3f)'%(t,est-math.sqrt(D),abs(est-math.sqrt(D))/math.sqrt(t)))
random.seed(20261104)
C1=max(abs(mass(t,dim,R,D,g,b,md=md)-c*math.sqrt(D*t))/t for t in (1e-4,1e-3,1e-2,3e-2))
print('empirical C1 = sup |M - c sqrt(Dt)|/t on t<=0.03: %.3f'%C1)
for eta in (1e-4,1e-3,1e-2):
    topt=eta*Minf/C1; errs=[]
    for _ in range(200):
        m=mass(topt,dim,R,D,g,b,md=md)+random.gauss(0,eta*Minf); Dh=(m/(c*math.sqrt(topt)))**2; errs.append(abs(Dh-D)/D)
    errs.sort(); print('eta=%.0e t_opt=%.2e  median rel.err D=%.3e  (predicted order 4 sqrt(eta Minf C1)/c / sqrtD = %.3e)'%(eta,topt,errs[100],4*math.sqrt(eta*Minf*C1)/c/math.sqrt(D)))
print('## (d) empirical Lipschitz constant on K')
ts=[20*(i/80)**2 for i in range(1,81)]; md2=modes(dim,R,120)
def curve(p): return [mass(t,dim,R,*p,md=md2) for t in ts]
def l2(a,b_): 
    s=0.0
    for i in range(1,len(ts)): s+=0.5*((a[i]-b_[i])**2+(a[i-1]-b_[i-1])**2)*(ts[i]-ts[i-1])
    return math.sqrt(s)
K=[(0.3,0.8),(0.4,1.0),(0.8,1.8)]; best=0.0
for trial in range(150):
    p=[random.uniform(*K[j]) for j in range(3)]
    sc=10**random.uniform(-4,-0.5); q=[min(max(p[j]+sc*random.gauss(0,1),K[j][0]),K[j][1]) for j in range(3)]
    d=math.sqrt(sum((p[j]-q[j])**2 for j in range(3))); r=d/max(l2(curve(p),curve(q)),1e-300); best=max(best,r)
print('max |p-q|/||M_p-M_q||_L2 over 150 pairs = %.2f'%best)
