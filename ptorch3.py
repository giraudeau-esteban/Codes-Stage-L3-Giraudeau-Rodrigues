import numpy as np
import random as rnd
import pylab as plt
from scipy.linalg import eig, solve, norm
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs
import torch
from torch.linalg import eigh
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def fst(c):
    return c[0]

plt.ion()
plt.show()

L=1
N=199
h=L/(N+1)


def L(alpha, Supp, t):

    A = torch.diag(Supp)

    W = 1/h**2*torch.eye(N+1, dtype=float)
    W[t, t] = 1/h**2 * 1/alpha

    K = torch.tensor(np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2), dtype=float)

    BA = K.matmul(A)
    Lap = (BA.transpose(0, 1)).matmul(W).matmul(BA)

    return(Lap)

def EPs(Lap, z, Supp):

    A = torch.diag(Supp)

    U, V = eigh(Lap)

    U = torch.real(U)
    V = torch.real(V)

    P = [(U[k], A.matmul(V[:,k])) for k in range(U.size()[0])]

    P = sorted(P,key=fst)
    m = min([k for k in range(len(P)) if P[k][0] > 0])

    return([P[m+zi-1] for zi in z])

def E(alpha, Supp, t, Lobj):

    Lap = L(alpha, Supp, t)
    ep = EPs(Lap, [1], Supp)

    return((ep[0][0] - np.pi**2/Lobj**2)**2)

def opti(Lobj):

    Supp=torch.ones(N+2, dtype=float)
    Supp[0] = 0
    Supp[-1] = 0

    t = N

    eps = 10**-8
    x = torch.tensor(0.5, requires_grad=True)
    e0 = E(x, Supp, t, Lobj)
    e0.backward()
    gx = x.grad
    j = 0
    a = 1

    R = []

    while torch.norm(gx) > 10**-3 and j < 1000:

        j += 1
        x = (x - a*gx).clone().detach().requires_grad_(True)

        if x > 1:
            t += 1
            Supp[t] = 1
            x = torch.tensor(0.0001, requires_grad=True)
        elif x < 0:
            Supp[t] = 0
            t -= 1
            x = torch.tensor(1.0, requires_grad=True)

        e0 = E(x, Supp, t, Lobj)
        e0.backward()

        gx = x.grad

    return(j, )

X = np.linspace(0.05, 0.8, 20)
Y = []

for x in X:
    print(x)
    Y.append(opti(x))

plt.figure(14)
plt.clf()

plt.plot(X, Y)
