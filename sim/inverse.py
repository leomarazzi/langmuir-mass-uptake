"""Inverse problem simulations: recover (D, gamma, beta) from a noisy absorption mass curve M(t).
Forward model: exact series of langmuir.py (verified against finite differences, forward_check.out).
Estimators: (E1) three-step (sqrt(t) law for D, saturated mass for rho=gamma/beta, late decay rate for beta);
(E2) Levenberg-Marquardt least squares on log-parameters, started from E1.
Also: Fisher-information condition number per dimension. Seed fixed (20261104)."""
import math, random, sys
from langmuir import modes, volume, area, mass, rates
def times_grid(T,n): return [T*(i/n)**2 for i in range(1,n+1)]   # denser at early times
def forward(p,dim,R,ts,md):
    D,g,b=[math.exp(x) for x in p]; return [mass(t,dim,R,D,g,b,md=md) for t in ts]
def three_step(ts,Ms,dim,R,mu1):
    V=volume(dim,R); A=area(dim,R)
    # saturated mass from the last 10% (data run long enough) -> rho
    tail=Ms[int(0.9*len(Ms)):]; Minf=sum(tail)/len(tail); rho=Minf/V-1
    # early window: M ~ a sqrt(t) + c t on t <= t_e
    te=0.02; E=[(t,m) for t,m in zip(ts,Ms) if t<=te]
    S11=sum(t for t,_ in E); S12=sum(t**1.5 for t,_ in E); S22=sum(t*t for t,_ in E)
    y1=sum(math.sqrt(t)*m for t,m in E); y2=sum(t*m for t,m in E)
    det=S11*S22-S12*S12; a=(y1*S22-y2*S12)/det
    D=(a*math.sqrt(math.pi)/(2*A))**2
    # late window: log(Minf - M) slope on t in [Tl, T2] where deficit still > 2% of Minf
    L=[(t,math.log(Minf-m)) for t,m in zip(ts,Ms) if Minf-m>0.02*Minf and t>0.3*ts[-1]*0.2]
    L=L[len(L)//2:]
    n=len(L); mt=sum(t for t,_ in L)/n; ml=sum(l for _,l in L)/n
    s=-sum((t-mt)*(l-ml) for t,l in L)/sum((t-mt)**2 for t,_ in L)
    m1=D*mu1; den=m1-(1+rho)*s; b=s*(m1-s)/den if den!=0 else float('nan')
    return D,rho*b,b
def lm(p0,dim,R,ts,Ms,md,it=40):
    p=list(p0); lam=1e-3
    def res(p): return [f-m for f,m in zip(forward(p,dim,R,ts,md),Ms)]
    r=res(p); c=sum(x*x for x in r)
    for _ in range(it):
        Jc=[]
        for j in range(3):
            q=list(p); q[j]+=1e-6; rj=res(q); Jc.append([(a-b)/1e-6 for a,b in zip(rj,r)])
        JTJ=[[sum(Jc[i][k]*Jc[j][k] for k in range(len(r))) for j in range(3)] for i in range(3)]
        JTr=[sum(Jc[i][k]*r[k] for k in range(len(r))) for i in range(3)]
        while True:
            A=[[JTJ[i][j]+(lam*JTJ[i][i] if i==j else 0) for j in range(3)] for i in range(3)]
            dp=solve3(A,[-x for x in JTr]); nrm=max(abs(x) for x in dp)
            if nrm>0.5: dp=[x*0.5/nrm for x in dp]
            q=[min(6.0,max(-12.0,p[i]+dp[i])) for i in range(3)]
            rq=res(q); cq=sum(x*x for x in rq)
            if cq<c: p,r,c=q,rq,cq; lam=max(lam/3,1e-9); break
            lam*=4
            if lam>1e8: return p
        if max(abs(x) for x in dp)<1e-9: break
    return p
def solve3(A,b):
    import copy; A=copy.deepcopy(A); b=list(b); n=3
    for i in range(n):
        piv=max(range(i,n),key=lambda r:abs(A[r][i])); A[i],A[piv]=A[piv],A[i]; b[i],b[piv]=b[piv],b[i]
        for r in range(i+1,n):
            f=A[r][i]/A[i][i]
            for c in range(i,n): A[r][c]-=f*A[i][c]
            b[r]-=f*b[i]
    x=[0.0]*n
    for i in range(n-1,-1,-1): x[i]=(b[i]-sum(A[i][c]*x[c] for c in range(i+1,n)))/A[i][i]
    return x
def fisher_cond(p,dim,R,ts,md):
    base=forward(p,dim,R,ts,md); Jc=[]
    for j in range(3):
        q=list(p); q[j]+=1e-6; Jc.append([(a-b)/1e-6 for a,b in zip(forward(q,dim,R,ts,md),base)])
    F=[[sum(Jc[i][k]*Jc[j][k] for k in range(len(ts))) for j in range(3)] for i in range(3)]
    # eigenvalues of symmetric 3x3 via power/deflation (Jacobi rotations)
    a=[row[:] for row in F]
    for _ in range(60):
        for (i,j) in ((0,1),(0,2),(1,2)):
            if abs(a[i][j])<1e-300: continue
            th=0.5*math.atan2(2*a[i][j],a[j][j]-a[i][i]); c,s=math.cos(th),math.sin(th)
            for k in range(3):
                aik,ajk=a[i][k],a[j][k]; a[i][k]=c*aik-s*ajk; a[j][k]=s*aik+c*ajk
            for k in range(3):
                aki,akj=a[k][i],a[k][j]; a[k][i]=c*aki-s*akj; a[k][j]=s*aki+c*akj
    ev=sorted(abs(a[i][i]) for i in range(3)); return ev[2]/ev[0], ev
if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 30
    D,g,b,R=0.5,0.7,1.3,1.0; T=20.0; ts=times_grid(T,160)
    random.seed(20261104)
    print('# true D=%g gamma=%g beta=%g R=%g; T=%g, %d samples (quadratic grid); noise sd = eta*M_inf; trials=%d'%(D,g,b,R,T,len(ts),trials))
    for dim in (1,2,3):
        md=modes(dim,R,100); mu1=md[0][0]; Minf=volume(dim,R)*(1+g/b)
        clean=[mass(t,dim,R,D,g,b,md=md) for t in ts]
        sm1,sp1=rates(D*mu1,g,b); sm2,_=rates(D*md[1][0],g,b)
        cond,ev=fisher_cond([math.log(D),math.log(g),math.log(b)],dim,R,ts,md)
        print('## dim=%d  mu1=%.4f  s-(1)=%.4f s+(1)=%.4f s-(2)=%.4f  gap=%.4f  Fisher cond=%.3g'%(dim,mu1,sm1,sp1,sm2,sm2-sm1,cond))
        for eta in (0.0,1e-3,1e-2):
            errs3=[];errsL=[]
            for tr in range(trials if eta>0 else 1):
                Ms=[m+random.gauss(0,eta*Minf) for m in clean]
                e=three_step(ts,Ms,dim,R,mu1); errs3.append(e)
                p0=[min(4.0,max(-6.0,math.log(x))) if (x==x and x>0) else 0.0 for x in e]
                pl=lm(p0,dim,R,ts,Ms,md); errsL.append(tuple(math.exp(x) for x in pl))
            def stat(E,j,tv):
                v=[abs(x[j]-tv)/tv for x in E if x[j]==x[j]]; v.sort()
                return (sum(v)/len(v), v[len(v)//2]) if v else (float('nan'),)*2
            for name,E in (('3-step',errs3),('LM    ',errsL)):
                print('  eta=%-6g %s  rel.err mean/median:  D %.2e/%.2e  gamma %.2e/%.2e  beta %.2e/%.2e'%(
                    eta,name,*stat(E,0,D),*stat(E,1,g),*stat(E,2,b)))
