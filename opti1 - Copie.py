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

Lobj = 0.654


def L(X, Supp, t, s):

    A = np.diag(Supp)

    W = 1/h**2*np.eye(N+1)
    W[t, t] = 1/h**2 * 1/X[0]
    W[s, s] = 1/h**2 * 1/X[1]

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

def grad(X, Supp, t, s):

    Lap1 = L(X[0], Supp, t, s)
    dLap1 = dL(X[0], Supp, t)
    ep1 = EPs(Lap1, dLap1, [1], Supp)

    Lap2 = L(X[1], Supp, t, s)
    dLap2 = dL(X[1], Supp, s)
    ep2 = EPs(Lap2, dLap2, [1], Supp)

    return(np.array([2*(ep[0][0] - np.pi**2/Lobj**2)*ep[0][2], 2*(ep[0][0] - np.pi**2/Lobj**2)*ep[0][2]]))

Supp=np.ones(N+2)
Supp[0] = 0
Supp[-1] = 0

t, s = 99, 0

eps = 10**-8
X = np.array([0.5, 0.5])
gx = grad(X, Supp, t, s)
j = 0
a = 0.1

while norm(gx) > 10**-8 and j < 3000:

    print(j)
    print((t-s-1+X[0]+X[1])*h)

    j += 1

    X = X - a*gx

    if X[0] > 1:
        t += 1
        Supp[t] = 1
        X[0] = 0.01
    elif X[0] < 0:
        Supp[t] = 0
        t -= 1
        X[0]=1

    if X[1] > 1:
        Supp[s] = 1
        s -= 1
        X[1] = 0.01
    elif X[0] < 0:
        s += 1
        Supp[s] = 0
        X[1]=1

    # gy = gx
    gx = grad(x, Supp, t)

    # if gx*gy <= 0:
    #     a = 0.1*a

print(x)
