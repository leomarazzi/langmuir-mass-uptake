"""Carter-Kibler (Langmuir) model, exact mass curves and finite-difference check.
u_t = D Lap u - g u + b N,  N_t = g u - b N  on Omega (slab (-R,R), disk, ball of radius R),
u = ninf on the boundary, u = N = 0 at t = 0 (dry absorption).  Pure Python.
Mass deficit F(t) = (M_inf - M(t))/ninf = sum_k c_k^2 (A-_k e^{-s-_k t} + A+_k e^{-s+_k t}),
s_pm roots of s^2 - (m+g+b) s + m b = 0 (m = D mu_k), Nn(tau) = (b+g)(tau+b+g) + g m,
A- = Nn(-s-)/(b (s+ - s-)),  A+ = Nn(-s+)/(b (s- - s+))."""
import math
def J(n,x,M=400):
    h=math.pi/M; s=0.5*(1.0+math.cos(n*math.pi))
    for i in range(1,M): t=i*h; s+=math.cos(n*t-x*math.sin(t))
    return s*h/math.pi
_JZ=None
def j0zeros(n):
    global _JZ
    if _JZ is None or len(_JZ)<n:
        z=[]; x=0.5; f=J(0,x)
        while len(z)<n:
            y=x+0.05; g=J(0,y)
            if f*g<0:
                lo,hi=x,y
                for _ in range(55):
                    m=(lo+hi)/2
                    if J(0,lo)*J(0,m)<=0: hi=m
                    else: lo=m
                z.append((lo+hi)/2)
            x,f=y,g
        _JZ=z
    return _JZ[:n]
def modes(dim,R,K):
    """(mu_k, c_k^2) for the modes with c_k != 0 (orthonormal Dirichlet eigenfunctions)."""
    out=_modes(dim,R,K); _VOL[id(out)]=volume(dim,R); return out
def _modes(dim,R,K):
    if dim==1: return [((2*k+1)**2*math.pi**2/(4*R*R),16*R/((2*k+1)**2*math.pi**2)) for k in range(K)]
    if dim==2: return [(j*j/(R*R),4*math.pi*R*R/(j*j)) for j in j0zeros(K)]
    if dim==3: return [(k*k*math.pi**2/(R*R),8*R**3/(math.pi*k*k)) for k in range(1,K+1)]
_VOL={}
def volume_md(md): return _VOL[id(md)]
def volume(dim,R): return {1:2*R,2:math.pi*R*R,3:4*math.pi*R**3/3}[dim]
def area(dim,R): return {1:2.0,2:2*math.pi*R,3:4*math.pi*R*R}[dim]
def rates(m,g,b):
    B=m+g+b; d=math.sqrt(B*B-4*m*b); return (B-d)/2,(B+d)/2
def amps(m,g,b):
    sm,sp=rates(m,g,b); Nn=lambda tau:(b+g)*(tau+b+g)+g*m
    return Nn(-sm)/(b*(sp-sm)), Nn(-sp)/(b*(sm-sp)), sm, sp
def deficit(t,dim,R,D,g,b,K=400,md=None):
    md=md or modes(dim,R,K); s=0.0
    for mu,c2 in md:
        Am,Ap,sm,sp=amps(D*mu,g,b)
        s+=c2*(Am*math.exp(-sm*t)+Ap*math.exp(-sp*t))
    # tail of truncated modes: slow part A- -> g/b, s- -> b (fast part decays like e^{-D mu t}, dropped for t>0)
    tail=volume_md(md)-sum(c2 for _,c2 in md)
    return s+tail*((g/b)*math.exp(-b*t)+(1.0 if t==0 else 0.0))
def mass(t,dim,R,D,g,b,ninf=1.0,K=400,md=None):
    return ninf*(volume(dim,R)*(1+g/b)-deficit(t,dim,R,D,g,b,K,md))
def fd_mass(dim,R,D,g,b,ninf,times,Jn=200,dt=5e-4):
    """Radial finite differences, Strang splitting (exact exchange + Crank-Nicolson diffusion)."""
    h=R/Jn; r=[i*h for i in range(Jn+1)]
    u=[0.0]*(Jn+1); u[Jn]=ninf; N=[0.0]*(Jn+1); t=0.0; out={}; ti=0; times=sorted(times)
    w=[(r[i]**(dim-1) if dim>1 else 1.0) for i in range(Jn+1)]
    def M():
        # trapezoid of (u+N) * surface measure; slab doubled (symmetric)
        fac={1:2.0,2:2*math.pi,3:4*math.pi}[dim]; s=0.0
        for i in range(Jn+1):
            wt=h*(0.5 if i in (0,Jn) else 1.0); s+=wt*w[i]*(u[i]+N[i])
        return fac*s
    def exch(kk):
        K=g+b; e=math.exp(-K*kk)
        for i in range(Jn):
            s=u[i]+N[i]; Nn=g*s/K+(N[i]-g*s/K)*e; N[i]=Nn; u[i]=s-Nn
        N[Jn]=g*ninf/b+(N[Jn]-g*ninf/b)*math.exp(-b*kk)
    while ti<len(times):
        if t>=times[ti]-1e-12: out[times[ti]]=M(); ti+=1; continue
        k=min(dt,times[ti]-t); exch(k/2)
        mu=D*k/(h*h); a=[0.0]*Jn; bb=[0.0]*Jn; c=[0.0]*Jn; d=[0.0]*Jn
        for i in range(Jn):
            if i==0:
                bb[0]=1+dim*mu; c[0]=-dim*mu; d[0]=(1-dim*mu)*u[0]+dim*mu*u[1]
            else:
                lo=mu/2*(1-(dim-1)*h/(2*r[i])); hi=mu/2*(1+(dim-1)*h/(2*r[i]))
                a[i]=-lo; bb[i]=1+mu; c[i]=-hi; d[i]=lo*u[i-1]+(1-mu)*u[i]+hi*u[i+1]
                if i==Jn-1: d[i]+=hi*ninf; c[i]=0.0
        cp=[0.0]*Jn; dp=[0.0]*Jn; cp[0]=c[0]/bb[0]; dp[0]=d[0]/bb[0]
        for i in range(1,Jn):
            m=bb[i]-a[i]*cp[i-1]; cp[i]=c[i]/m; dp[i]=(d[i]-a[i]*dp[i-1])/m
        x=[0.0]*Jn; x[-1]=dp[-1]
        for i in range(Jn-2,-1,-1): x[i]=dp[i]-cp[i]*x[i+1]
        u=x+[ninf]; exch(k/2); t+=k
    return out
if __name__=='__main__':
    D,g,b,R,ninf=0.5,0.7,1.3,1.0,1.0
    times=[0.01,0.05,0.2,1.0,3.0]
    print('# forward check: exact mass series vs finite differences; D=%g gamma=%g beta=%g R=%g'%(D,g,b,R))
    for dim in (1,2,3):
        md=modes(dim,R,400 if dim!=2 else 120)
        fd=fd_mass(dim,R,D,g,b,ninf,times)
        Minf=volume(dim,R)*(1+g/b)
        for t in times:
            print('dim=%d t=%.2f  M_series=%.6f  M_FD=%.6f  rel.diff=%.1e  (M_inf=%.4f)'%(dim,t,mass(t,dim,R,D,g,b,md=md),fd[t],abs(mass(t,dim,R,D,g,b,md=md)-fd[t])/Minf,Minf))
        print('dim=%d t=0 series M(0)=%.2e (should be 0)'%(dim,mass(0.0,dim,R,D,g,b,md=md)))
