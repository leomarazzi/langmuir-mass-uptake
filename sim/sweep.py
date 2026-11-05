"""Conditioning sweep: Fisher information of log(D,gamma,beta) from a noise-free absorption curve,
as a function of the diffusion/exchange time-scale ratio m1 = D mu1/(beta+gamma), for dim = 1,2,3,
with R chosen so that D mu1 is the same across dimensions. Reports the Cramer-Rao relative standard
deviations at noise level eta = 1e-3 (sd = eta*M_inf), samples on [0, T], T = 8/s-(1)."""
import math
from langmuir import modes, volume, mass, rates
from inverse import times_grid
def crlb(dim,R,D,g,b,eta=1e-3,n=160):
    md=modes(dim,R,100); s1,_=rates(D*md[0][0],g,b); T=8.0/s1; ts=times_grid(T,n)
    p=[math.log(D),math.log(g),math.log(b)]
    def fw(p): return [mass(t,dim,R,*[math.exp(x) for x in p],md=md) for t in ts]
    base=fw(p); J=[]
    for j in range(3):
        q=list(p); q[j]+=1e-6; J.append([(a-c)/1e-6 for a,c in zip(fw(q),base)])
    F=[[sum(J[i][k]*J[j][k] for k in range(n)) for j in range(3)] for i in range(3)]
    a,b_,c=F[0];d,e,f=F[1];g_,h,i=F[2]
    det=a*(e*i-f*h)-b_*(d*i-f*g_)+c*(d*h-e*g_)
    inv=[[(e*i-f*h)/det,(c*h-b_*i)/det,(b_*f-c*e)/det],[(f*g_-d*i)/det,(a*i-c*g_)/det,(c*d-a*f)/det],[(d*h-e*g_)/det,(b_*g_-a*h)/det,(a*e-b_*d)/det]]
    sig=eta*volume(dim,R)*(1+g/b)
    return [sig*math.sqrt(inv[k][k]) for k in range(3)]
if __name__=='__main__':
    D,g,b=0.5,0.7,1.3
    mu_ref={1:math.pi**2/4,2:2.404825557695773**2,3:math.pi**2}
    print('# CRLB relative sd (eta=1e-3) for (D, gamma, beta); D mu1 matched across dims; ratio = D mu1/(beta+gamma)')
    for ratio in (0.1,0.3,1.0,3.0,10.0,30.0):
        m1=ratio*(b+g); row=[]
        for dim in (1,2,3):
            R=math.sqrt(D*mu_ref[dim]/m1); row.append((dim,crlb(dim,R,D,g,b)))
        print('ratio=%-5g '%ratio+'  '.join('dim%d: D %.1e g %.1e b %.1e'%(dd,*v) for dd,v in row))
