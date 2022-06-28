import numpy as np
import random as rnd
import pylab as plt
from scipy.linalg import eig, solve, norm
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs

def fst(c):
    return c[0]

plt.ion()
plt.show()

L=1
N=99
h=L/(N+1)

Lobj = 0.683


def L(alpha, Supp, t):

    A = np.diag(Supp)

    W = 1/h**2*np.eye(N+1)
    W[t, t] = 1/h**2 * 1/alpha

    K = np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

    BA = K.dot(A)
    Lap = (BA.transpose()).dot(W).dot(BA)

    return(Lap)

def dL(alpha, Supp, t):

    A = np.diag(Supp)

    W = np.zeros((N+1, N+1))
    W[t, t] = -2/h**2 * 1/alpha**2

    K = np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

    BA = K.dot(A)
    dLap = (BA.transpose()).dot(W).dot(BA)

    return(dLap)

def EPs(Lap, dLap, z, Supp):

    A = np.diag(Supp)

    U, V = eig(Lap)

    U = np.real(U)
    V = np.real(V)
    dU = np.diag((V.transpose()).dot(dLap).dot(V))

    P = [(U[k], A.dot(V[:,k]), dU[k]) for k in range(np.size(U))]

    P = sorted(P,key=fst)
    m = min([k for k in range(len(P)) if P[k][0] > 0])

    return([P[m+zi-1] for zi in z])

def grad(alpha, Supp, t):

    Lap = L(alpha, Supp, t)
    dLap = dL(alpha, Supp, t)
    ep = EPs(Lap, dLap, [1], Supp)

    return(2*(ep[0][0] - np.pi**2/Lobj**2)*ep[0][2])

Supp=np.ones(N+2)
Supp[0] = 0
Supp[-1] = 0

t = 99

eps = 10**-8
x = 0.5
gx = grad(x, Supp, t)
j = 0
a = 1

while norm(gx) > 10**-8 and j < 3000:

    print(j)
    print((t+x)*h)

    j += 1

    x = x - a*gx

    if x > 1:
        t += 1
        Supp[t] = 1
        x = 0.01
    elif x < 0:
        Supp[t] = 0
        t -= 1
        x=1
    gy = gx
    gx = grad(x, Supp, t)

    if gx*gy <= 0:
        a = 0.1*a

print(x)
