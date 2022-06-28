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
N=299
h=L/(N+1)


def L(X, Supp, t, s):

    A = torch.diag(Supp)

    W = 1/h**2*torch.eye(N+1, dtype=float)
    W[t, t] = 1/h**2 * 1/X[0]
    W[s, s] = 1/h**2 * 1/X[1]

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

def E(alpha, Supp, t, s, Lobj):

    Lap = L(alpha, Supp, t, s)
    ep = EPs(Lap, [1], Supp)

    return((ep[0][0] - np.pi**2/Lobj**2)**2)

def opti(S, M, Lobj):

    s, t = S[0], S[1]

    Supp=torch.zeros(N+2, dtype=float)
    Supp[s+1:t+1] = torch.ones(t-s, dtype=float)

    eps = 10**-8
    X = torch.tensor([M[0], M[1]], requires_grad=True)
    e0 = E(X, Supp, t, s, Lobj)
    e0.backward()
    gx = X.grad
    j = 0
    a = 0.1

    # plt.figure(1)

    H = []

    while torch.norm(gx) > 10**-5 and j < 1000:

        print(j)
        # print(((t-s-1+X[0]+X[1])*h).item())

        H.append([(s+1-X[1])*h, (t+X[0])*h])
        j += 1

        Y = X - a*gx
        X = Y.clone().detach().requires_grad_(True)

        if X[0] > 1:
            t += 1
            Supp[t] = 1
            X = torch.tensor([0.01, X[1].item()], requires_grad=True)
        elif X[0] < 0:
            Supp[t] = 0
            t -= 1
            X = torch.tensor([1.0, X[1].item()], requires_grad=True)

        if X[1] > 1:
            Supp[s] = 1
            s -= 1
            X = torch.tensor([X[0].item(), 0.01], requires_grad=True)
        elif X[1] < 0:
            s += 1
            Supp[s] = 0
            X = torch.tensor([X[0].item(), 1.0], requires_grad=True)

        e0 = E(X, Supp, t, s, Lobj)
        e0.backward()
        # gy = gx
        gx = X.grad

        print(X)
        print(s, t)

        # print(X)
        # print(s,t)
        # print(e0.item())

        # plt.clf()
        #
        # plt.plot(h*np.arange(s+1, t+1), np.zeros_like(np.arange(s+1, t+1)), color='orange', marker='+')
        # plt.plot([(s+1-X[1])*h], [0], color='blue', marker='o')
        # plt.plot([(t+X[0])*h], [0], color='red', marker='o')
        #
        # plt.axis('scaled')
        # plt.axis([-0.05, 1.05, -0.2, 0.2])
        #
        #
        # plt.pause(0.2)

    print(((t-s-1+X[0]+X[1])*h).item())

    return(H)

dif = [[[0, 200], [1.0, 1.0], 0.346], [[10, 180], [1.0, 1.0], 0.346], [[147, 153], [1.0, 1.0], 0.346], [[40, 299], [1.0, 1.0], 0.346], [[200, 230], [1.0, 1.0], 0.346]]
col = ['orange', 'blue', 'green', 'red', 'pink']

# dif = [ [[10, 80], [0.5, 0.5], 0.456]]

# [[0, 299], [1.0, 1.0], 0.056], [[0, 299], [1.0, 1.0], 0.243], [[0, 299], [1.0, 1.0], 0.357], [[0, 299], [1.0, 1.0], 0.556],

# plt.figure(2)
# plt.clf()

for i, hval in enumerate(dif):

    sol = opti(hval[0], hval[1], hval[2])

    for j, res in enumerate(sol):

        plt.plot(res, [j, j], color=col[i], linestyle='', marker='+')

    #
    # if gx*gy <= 0:
    #     a = 0.5*a