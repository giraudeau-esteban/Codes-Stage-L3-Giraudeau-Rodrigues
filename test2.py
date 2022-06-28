import numpy as np
import random as rnd
import pylab as plt
from scipy.linalg import eig, solve
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs

def fst(c):
    return c[0]

plt.ion()
plt.show()
plt.figure(1)
plt.clf()

def E(alpha=1, z=1, kz=1, t=0):

    L=1
    N=9
    h=L/(N+1)

    Supp=1/h*np.ones(N+2)
    Supp[0] = 0
    Supp[-1] = 0

    A = np.diag(Supp)
    A[t+1, t+1] = alpha/h

    l0 = 10**-4
    k = 1/(1-l0/h)

    W = np.eye(N+1)
    W[t, t] = k*(1-l0/(alpha*h))

    B = np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

    BA = B.dot(A)
    Lap = (BA.transpose()).dot(W).dot(BA)

    U,V=eig(Lap)
    U=np.real(U)
    V=np.real(V)
    P = [(U[k], V[:,k]) for k in range(np.size(U))]
    P=sorted(P,key=fst)
    # print(Lap)
    # print('stop')
    # print([a for (a, b) in P])

    m = min([k for k in range(len(P)) if P[k][0] > 0])

    return([P[m + zi-1][0] for zi in z])


def minf(f):

    def pente(x, y, fx, fy):
        return((f(x)-f(y))/(x-y))

    M = [f(k*0.05) for k in range(21)]

    im = min([k for k in range(19) if M[k] - M[k+1] > 0 and M[k+2] - M[k+1] > 0])

    eps = 10**-3

    a, b = 0.05*im, 0.05*(im+2)
    fa, fb, fm = M[im], M[im+2], M[im+1]
    m = (a+b)/2
    i = 0

    while b-a > eps and i <= 50:
        i += 1
        m1 = (a+m)/2
        m2 = (b+m)/2
        fm1, fm2 = f(m1), f(m2)
        if pente(a, m1, fa, fm1)*pente(m1, m, fm1, fm) <= 0:
            m, b = m1, m
            fm, fb = fm1, fm
        elif pente(m1, m, fm1, fm)*pente(m, m2, fm, fm2) <= 0:
            a, b = m1, m2
            fa, fb = fm1, fm2
        else:
            a, m = m, m2
            fa, fm = fm, fm2

    return(a)

X = np.linspace(0.01, 1, 1000)
# Lval = np.linspace(0, 2, 11)
#
# X = np.linspace(0.05, 0.25, 100)
# Lval = np.linspace(0.80, 1.2, 50)
#
# Y = [minf(lambda x: E(x, 5, v)[1]) for v in Lval]
#
# plt.plot(Lval, Y)
#
Lval = [16*np.pi**2/0.995**2]
# # #
for v in Lval:
    Y1 = np.array([np.sqrt(E(x, [1, 2], v)[0]) for x in X])
    Y2 = np.array([np.sqrt(E(x, [1, 2], v)[1]) for x in X])
    # plt.plot(X, X**(1/3)*Y1+(1-X**(1/3))*Y2)
    # plt.plot(X, Y2 - Y1)
    plt.plot(X, Y1)
    plt.plot(X, Y2)
plt.legend(loc='upper right')

plt.plot(X, np.pi*np.ones_like(X))
plt.plot(X, np.pi/0.9*np.ones_like(X))