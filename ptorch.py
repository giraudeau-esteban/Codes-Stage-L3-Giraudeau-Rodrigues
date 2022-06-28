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
N=99
h=L/(N+1)

Lobj = 0.1


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

def E(alpha, Supp, t):

    Lap = L(alpha, Supp, t)
    ep = EPs(Lap, [1], Supp)

    return((ep[0][0] - np.pi**2/Lobj**2)**2)

Supp=torch.ones(N+2, dtype=float)
Supp[0] = 0
Supp[-1] = 0

t = N

eps = 10**-8
x = torch.tensor(0.5, requires_grad=True)
e0 = E(x, Supp, t)
e0.backward()
gx = x.grad
j = 0
a = 1

R = []

while torch.norm(gx) > 10**-1 and j < 1000:

    print(j)
    print(((t+x)*h).item())

    j += 1
    y = torch.tensor(x - a*gx, requires_grad=True)
    x = y.clone().detach().requires_grad_(True)


    if x > 1:
        t += 1
        Supp[t] = 1
        x = torch.tensor(0.0001, requires_grad=True)
    elif x < 0:
        Supp[t] = 0
        t -= 1
        x = torch.tensor(1.0, requires_grad=True)

    e0 = E(x, Supp, t)
    e0.backward()

    gx = x.grad
    print(gx)

    R.append([((t+x)*h).item(), e0.item(), gx.item()])
    #
    # if gx*gy <= 0:
    #     a = 0.5*a

R = np.array(R)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(R[:, 0], R[:, 1])
plt.subplot(1, 2, 2)
plt.plot(R[:, 0], R[:, 2])