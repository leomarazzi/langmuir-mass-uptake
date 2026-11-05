"""Local density check (ball, dim 3): u = ninf - ninf sum_k c_k phi_k(x) x1_k(t),
N = g ninf/b - ninf sum_k c_k phi_k(x) x2_k(t), with
x1 = [(b+g-s-)e^{-s- t} - (b+g-s+)e^{-s+ t}]/(s+ - s-),  x1+x2 = G_k(t) = A- e^{-s- t} + A+ e^{-s+ t}.
Ball: phi_k = sin(k pi r/R)/(r sqrt(2 pi R)), c_k = (4 pi/sqrt(2 pi R)) R^2 (-1)^{k+1}/(k pi).
Also: drying (u=ninf, N=g ninf/b initially; u=0 on boundary) mass = M_inf - M_absorption (mirror)."""
import math
from langmuir import amps, rates
D,g,b,R,ninf=0.5,0.7,1.3,1.0,1.0
def local(r,t,K=3000):
    u=ninf; N=g*ninf/b
    for k in range(1,K+1):
        m=D*k*k*math.pi**2/R**2; Am,Ap,sm,sp=amps(m,g,b)
        x1=((b+g-sm)*math.exp(-sm*t)-(b+g-sp)*math.exp(-sp*t))/(sp-sm)
        G=Am*math.exp(-sm*t)+Ap*math.exp(-sp*t); x2=G-x1
        ck=(4*math.pi/math.sqrt(2*math.pi*R))*R*R*(-1)**(k+1)/(k*math.pi)
        ph=math.sin(k*math.pi*r/R)/(r*math.sqrt(2*math.pi*R))
        u-=ninf*ck*ph*x1; N-=ninf*ck*ph*x2
    return u,N
# finite differences (same scheme as langmuir.fd_mass) returning profiles
def fd(times,Jn=200,dt=5e-4,u0=0.0,N0=0.0,ub=1.0):
    h=R/Jn; r=[i*h for i in range(Jn+1)]; u=[u0]*(Jn+1); u[Jn]=ub; N=[N0]*(Jn+1); t=0.0; out={}; ti=0
    def exch(kk):
        K=g+b; e=math.exp(-K*kk)
        for i in range(Jn):
            s=u[i]+N[i]; Nn=g*s/K+(N[i]-g*s/K)*e; N[i]=Nn; u[i]=s-Nn
        N[Jn]=g*ub/b+(N[Jn]-g*ub/b)*math.exp(-b*kk)
    while ti<len(times):
        if t>=times[ti]-1e-12:
            M=4*math.pi*sum(h*(0.5 if i in (0,Jn) else 1)*r[i]**2*(u[i]+N[i]) for i in range(Jn+1))
            out[times[ti]]=(u[:],N[:],M); ti+=1; continue
        k=min(dt,times[ti]-t); exch(k/2); mu=D*k/(h*h)
        a=[0.0]*Jn; bb=[0.0]*Jn; c=[0.0]*Jn; d=[0.0]*Jn
        for i in range(Jn):
            if i==0: bb[0]=1+3*mu; c[0]=-3*mu; d[0]=(1-3*mu)*u[0]+3*mu*u[1]
            else:
                lo=mu/2*(1-h/r[i]); hi=mu/2*(1+h/r[i]); a[i]=-lo; bb[i]=1+mu; c[i]=-hi
                d[i]=lo*u[i-1]+(1-mu)*u[i]+hi*u[i+1]
                if i==Jn-1: d[i]+=hi*ub; c[i]=0.0
        cp=[0.0]*Jn; dp=[0.0]*Jn; cp[0]=c[0]/bb[0]; dp[0]=d[0]/bb[0]
        for i in range(1,Jn):
            m=bb[i]-a[i]*cp[i-1]; cp[i]=c[i]/m; dp[i]=(d[i]-a[i]*dp[i-1])/m
        x=[0.0]*Jn; x[-1]=dp[-1]
        for i in range(Jn-2,-1,-1): x[i]=dp[i]-cp[i]*x[i+1]
        u=x+[ub]; exch(k/2); t+=k
    return out
times=[0.05,0.3,1.5]
S=fd(times); Dr=fd(times,u0=ninf,N0=g*ninf/b,ub=0.0)
Minf=4*math.pi/3*(1+g/b)
for t in times:
    for i in (50,100,150):
        r=i/200; us,Ns=local(r,t)
        print('t=%.2f r=%.2f  u FD=%.6f series=%.6f | N FD=%.6f series=%.6f'%(t,r,S[t][0][i],us,S[t][1][i],Ns))
    print('t=%.2f  drying mass FD=%.6f   M_inf - absorption mass FD=%.6f'%(t,Dr[t][2],Minf-S[t][2]))
