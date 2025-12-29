import math

def bisection_min(fprime, a, b, tol=1e-4):
    it=0
    fa, fb=fprime(a), fprime(b)
    while b-a>tol:
        it+=1
        m=(a+b)/2
        fm=fprime(m)
        if fa*fm<=0:
            b=m
            fb=fm
        else:
            a=m
            fa=fm
    return (a+b)/2, it

def golden_section(f, a, b, tol=1e-4):
    phi=(math.sqrt(5)-1)/2
    c=b - phi*(b-a)
    d=a + phi*(b-a)
    fc, fd=f(c), f(d)
    it=0
    while b-a>tol:
        it+=1
        if fc<fd:
            b=d
            d=c
            fd=fc
            c=b - phi*(b-a)
            fc=f(c)
        else:
            a=c
            c=d
            fc=fd
            d=a + phi*(b-a)
            fd=f(d)
    return (a+b)/2, it

f=lambda x: math.exp(x)-2*x
fp=lambda x: math.exp(x)-2

bsol, bit = bisection_min(fp,0,1)
gsol, git = golden_section(f,0,1)

print(bsol, bit, gsol, git)
