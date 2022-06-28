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

Lobj = 0.4
Xabs = np.linspace(0, 1, N+2)
Uobj = np.zeros(N+2)
Uobj[10:15] = np.ones(5)/Xabs[1]
TU = torch.tensor(Uobj)

def extraction(Mat, Supp):

    j = 0

    while Supp[j] == 0.0:
        j += 1

    k = j

    while Supp[k] == 1.0:
        k += 1

    return(Mat[j:k, j:k])

def insertion1D(Mat, Supp):

    j = 0

    while Supp[j] == 0.0:
        j += 1

    k = j

    while Supp[k] == 1.0:
        k += 1

    New = torch.zeros(N+2, dtype=float)
    New[j:k] = Mat

    return(New)

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

    U, V = eigh(extraction(Lap, Supp))

    U = torch.real(U)
    V = torch.real(V)

    P = [(U[k], A.matmul(insertion1D(V[:,k], Supp))) for k in range(U.size()[0])]

    P = sorted(P,key=fst)
    m = min([k for k in range(len(P)) if P[k][0] > 0])

    return([P[m+zi-1] for zi in z])

def E(alpha, Supp, t):

    Lap = L(alpha, Supp, t)
    ep = EPs(Lap, [1], Supp)

    # print(ep[0][1])
    # print(ep[0][1]-TU)
    # print(torch.norm(ep[0][1]-TU))
    # a = ep[0][1][0]
    # a.backward()
    # print('lol', x.grad)



    return(torch.norm(ep[0][1]-TU))

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

print('hum', e0)
print(gx)

R = []

# while torch.norm(gx) > 10**-5 and j < 1000:
#
#     print(j)
#     print(((t+x)*h).item())
#
#     j += 1
#     y = torch.tensor(x - a*gx, requires_grad=True)
#     x = y.clone().detach().requires_grad_(True)
#
#
#     if x > 1:
#         t += 1
#         Supp[t] = 1
#         x = torch.tensor(0.0001, requires_grad=True)
#     elif x < 0:
#         Supp[t] = 0
#         t -= 1
#         x = torch.tensor(1.0, requires_grad=True)
#
#     e0 = E(x, Supp, t)
#     e0.backward()
#
#     gx = x.grad
#     print(gx)
#
#     R.append([((t+x)*h).item(), e0.item(), gx.item()])
