import ad
from ad import adnumber
import numpy as np
import random as rnd
import pylab as plt
from scipy.linalg import eig, solve, norm
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs
from torch.linalg import eigh
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def fst(c):
    return c[0]

plt.ion()
plt.show()

L=1
N=4
h=L/(N+1)

Lobj = 0.683


def L(alpha, Supp, t):

    pa = adnumber(alpha)

    A = np.diag(Supp)

    W = 1/h**2*np.eye(N+1, dtype=object)

    print(1/h**2 * 1/pa)

    W[t, t] = 1/h**2 * 1/pa

    print(W)
    print(type(W[t, t]))

    K = np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

    print('4')

    BA = K.dot(A)

    print('5')

    Lap = (BA.transpose()).dot(W).dot(BA)

    print('6')
    print(Lap)

    return(Lap)

# def dL(alpha, Supp, t):
#
#     A = np.diag(Supp)
#
#     W = np.zeros((N+1, N+1))
#     W[t, t] = -2/h**2 * 1/alpha**2
#
#     K = np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)
#
#     BA = K.dot(A)
#     dLap = (BA.transpose()).dot(W).dot(BA)
#
#     return(dLap)

def separe(M):

    f1 = np.vectorize(lambda a: a.x)
    f2 = np.vectorize(lambda a: a.d(x))

    return(f1(M), f2(M))

def EPs(LapetdLap, z, Supp):

    Lap, dLap = separe(LapetdLap)

    A = np.diag(Supp)

    U, V = eigh(Lap)

    U = np.real(U)
    V = np.real(V)
    dU = np.diag((V.transpose()).dot(dLap).dot(V))

    P = [(U[k], A.dot(V[:,k]), dU[k]) for k in range(np.size(U))]

    P = sorted(P,key=fst)
    m = min([k for k in range(len(P)) if P[k][0] > 0])

    return([P[m+zi-1] for zi in z])

def E(alpha, Supp, t):

    Lap = L(alpha, Supp, t)
    # dLap = dL(alpha, Supp, t)
    ep = EPs(Lap, [1], Supp)

    return((ep[0][0] - np.pi**2/Lobj**2)**2)

Supp=np.ones(N+2)
Supp[0] = 0
Supp[-1] = 0

t = 4

eps = 10**-8
x = 0.5
gx = E(x, Supp, t)
j = 0
a = 1

# while norm(gx) > 10**-8 and j < 3000:
#
#     print(j)
#     print((t+x)*h)
#
#     j += 1
#
#     x = x - a*gx
#
#     if x > 1:
#         t += 1
#         Supp[t] = 1
#         x = 0.01
#     elif x < 0:
#         Supp[t] = 0
#         t -= 1
#         x=1
#     gy = gx
#     gx = grad(x, Supp, t)
#
#     if gx*gy <= 0:
#         a = 0.1*a
#
# print(x)
