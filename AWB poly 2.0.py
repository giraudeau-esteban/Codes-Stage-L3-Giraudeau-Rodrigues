import numpy as np
import random as rnd
import pylab as plt
import torch as pt
from scipy.linalg import eig, solve
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def fst(c):
    return c[0]

def dichotomie(f,a,b,eps=10**-8):
    c=(a+b)/2

    while np.abs(f(c))>eps:
        if f(c)*f(a)>0:
            a=c
            c=(a+b)/2
        else:
            b=c
            c=(a+b)/2
    return c

def vp_Laplacien(A,W,K,n,s):
    Lap=K.dot(A)
    T=Lap.transpose()
    Lap=T.dot(W).dot(Lap)

    U,V=eigsh(Lap,k=n, sigma=s)
    U=np.real(U)
    V=np.real(A.dot(V))
    P = [(U[k], V[:,k]) for k in range(np.size(U))]
    P=sorted(P,key=fst)
    return P

def rectangle(centre, theta, A, B):

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    p1 = centre + rot.dot(np.array([A/2.0, B/2.0]))
    p2 = centre + rot.dot(np.array([-A/2.0, B/2.0]))
    p3 = centre + rot.dot(np.array([-A/2.0, -B/2.0]))
    p4 = centre + rot.dot(np.array([A/2.0, -B/2.0]))

    return(np.array([[p1, p2], [p2, p3], [p3, p4], [p4, p1]]))

def AWB_polygone(P, N, M, L, l):

    h = L/(N+1)
    k = l/(M+1)

    Supp = np.zeros([M+2,N+2])
    Horiz = 1/h**2 * np.ones([(M+2),(N+1)])
    Vert = 1/k**2 * np.ones([(M+1),(N+2)])


    for p in P:

        if np.abs(p[0][0] - p[1][0]) < 10**-8:

            p1 = np.array([int(p[0][0]/h+1/2+N/2.0), int(p[0][1]/k+1/2+M/2.0)])
            p2 = np.array([int(p[1][0]/h+1/2+N/2.0), int(p[1][1]/k+1/2+M/2.0)])

            s2 = np.sign(p[1][1]-p[0][1])

            if s2 > 0:
                p1 = p1 + np.array([0, 1])

            else:
                p2 = p2 + np.array([0, 1])

            x = p[0][0]
            xj = x/h+(N+1)/2.0
            j = int(xj)

            for i in range(min(p1[1], p2[1]), max(p1[1], p2[1])+1):

                if s2 < 0:
                    alpha = 1-(xj-j)
                else:
                    alpha = xj-j

                if alpha == 0:
                    Supp[i, j] = Supp[i, j] + 1
                    alpha = 1

                Horiz[i,j] = 1/k**2 * 1/alpha
                Supp[i,0:j+1] = Supp[i,0:j+1] + np.ones(j+1)

        elif np.abs(p[0][1] - p[1][1]) < 10**-8:

            p1 = np.array([int(p[0][0]/h+1/2+N/2.0), int(p[0][1]/k+1/2+M/2.0)])
            p2 = np.array([int(p[1][0]/h+1/2+N/2.0), int(p[1][1]/k+1/2+M/2.0)])

            s1 = np.sign(p[1][0]-p[0][0])

            if s1 > 0:
                p1 = p1 + np.array([1, 0])

            else:
                p2 = p2 + np.array([1, 0])

            y = p[0][1]
            yi = y/k+(M+1)/2.0
            i = int(yi)

            for j in range(min(p1[0], p2[0]), max(p1[0], p2[0])+1):

                if s1 > 0:
                    alpha = 1-(yi-i)
                else:
                    alpha = yi-i

                if alpha == 0:
                    Supp[i, j] = Supp[i, j] + 1
                    alpha = 1

                Vert[i,j] = 1/k**2 * 1/alpha

        else:
            a = (p[0][1]-p[1][1])/(p[0][0]-p[1][0])

            f = lambda x: a*(x-p[0][0]) + p[0][1]
            finv = lambda x: (x-p[0][1])/a + p[0][0]

            p1 = np.array([int(p[0][0]/h+1/2+N/2.0), int(p[0][1]/k+1/2+M/2.0)])
            p2 = np.array([int(p[1][0]/h+1/2+N/2.0), int(p[1][1]/k+1/2+M/2.0)])

            s2 = np.sign(p[1][1]-p[0][1])
            s1 = np.sign(p[1][0]-p[0][0])

            if s2 > 0:
                if s1 > 0:
                    p1 = p1 + np.array([1, 1])
                else:
                    p1 = p1 + np.array([0, 1])
                    p2 = p2 + np.array([1, 0])
            else:
                if s1 > 0:
                    p1 = p1 + np.array([1, 0])
                    p2 = p2 + np.array([0, 1])
                else:
                    p2 = p2 + np.array([1, 1])

            for j in range(min(p1[0], p2[0]), max(p1[0], p2[0])+1):

                x = h*(j-(N+1)/2.0)
                y = f(x)
                yi = y/k+(M+1)/2.0
                i = int(yi)

                if s1 > 0:
                    alpha = 1-(yi-i)
                else:
                    alpha = yi-i

                if alpha == 0:
                    Supp[i, j] = Supp[i, j] + 1
                    alpha = 1

                Vert[i,j] = 1/k**2 * 1/alpha

            for i in range(min(p1[1], p2[1]), max(p1[1], p2[1])+1):

                y = k*(i-(M+1)/2.0)
                x = finv(y)
                xj = x/h+(N+1)/2.0
                j = int(xj)

                if s2 < 0:
                    alpha = 1-(xj-j)
                else:
                    alpha = xj-j

                if alpha == 0:
                    Supp[i, j] = Supp[i, j] + 1
                    alpha = 1

                Horiz[i,j] = 1/k**2 * 1/alpha
                Supp[i,0:j+1] = Supp[i,0:j+1] + np.ones(j+1)

    # print(Supp)

    Supp = Supp % 2

    Horiz2 = np.zeros_like(Horiz)
    Horiz2[0:M+2,0:N+1] = Horiz[0:M+2,0:N+1]

    Vert2 = np.zeros_like(Vert)
    Vert2[0:M+1,0:N+2] = Vert[0:M+1,0:N+2]

    A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=Vert.reshape([(M+1)*(N+2)])

    R=np.concatenate((Horiz,Vert))
    W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))

    Bm=np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

    indices=np.arange(0,M+2)
    indptr=np.arange(0,M+3)
    data=np.array([Bm for i in range(M+2)])
    B1=bsr_matrix((data,indices,indptr), shape=((M+2)*(N+1),(N+2)*(M+2)))

    indices2=np.array([[i,i+1] for i in range(M+1)]).reshape(2*(M+1))
    indptr2=np.array([2*i for i in range(M+2)])
    data2=np.array([(-1)**(i+1)*np.eye(N+2) for i in range(2*(M+1))])
    B2=bsr_matrix((data2,indices2,indptr2),shape=((N+2)*(M+1),(M+2)*(N+2)))

    B=vstack((B1,B2))

    return A,W,B, Supp, Horiz2, Vert2

def simili_anneau(R1, R2, N1, N2, a=1, b=1, theta=0):

    P1 = np.cosh(R1)*np.cos(2*np.pi/N1*np.arange(0, N1+1)+2)
    P2 = np.sinh(R1)*np.sin(2*np.pi/N1*np.arange(0, N1+1)+2)
    Q1 = np.cosh(R2)*np.cos(-2*np.pi/N2*np.arange(0, N2+1)+2)
    Q2 = np.sinh(R2)*np.sin(-2*np.pi/N2*np.arange(0, N2+1)+2)

    R = np.zeros((N1+N2, 2, 2))

    R[0:N1, 0, 0] = P1[0:-1]
    R[0:N1, 0, 1] = P2[0:-1]
    R[0:N1, 1, 0] = P1[1:N1+1]
    R[0:N1, 1, 1] = P2[1:N1+1]

    R[N1:N1+N2, 0, 0] = Q1[0:-1]
    R[N1:N1+N2, 0, 1] = Q2[0:-1]
    R[N1:N1+N2, 1, 0] = Q1[1:N2+1]
    R[N1:N1+N2, 1, 1] = Q2[1:N2+1]

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for val in R:
        val[0] =  rot.dot(val[0])
        val[1] =  rot.dot(val[1])

    return(R)

plt.ion()
plt.show()
plt.figure(1)
plt.clf()
#
# # ############### pour changer la fenêtre et la discrétisation ######
#
L=4
l=4
N=299
M=299
h=L/(N+1)
k=l/(M+1)


# Ann = simili_anneau(1, 0.5, 30, 30, 1, 2, 3*np.pi/8)

Rec = rectangle(np.array([0, 0]), 0, 3.6, 2.5)

A,W,B, S, H, V=AWB_polygone(Rec, N, M, L, l)

# for seg in Rec:
#     plt.plot(seg[:, 0], seg[:, 1])

##

plt.figure(43)

plt.imshow(S[::-1, :], extent=[-L/2,L/2,-l/2,l/2],aspect='auto', cmap='Greys')
axes = plt.gca()
axes.xaxis.set_ticks(range(-2, 3))
axes.xaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
axes.yaxis.set_ticks(range(-2, 3))
axes.yaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
plt.axis('scaled')
# plt.xticks([], [])
# plt.yticks([], [])



##
plt.figure(17)
P=vp_Laplacien(A,W,B,8,8) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
# #
j=0
for p in P:
    j=j+1
    if p[0]>10**-7:  # Pour éviter d'afficher les vp correspondant à 0.
        print(p[0])
        plt.subplot(1,8,j)
        vp=np.zeros([M+2,N+2])
        for i in range(0,M+2):
            vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
        alpha=np.max(np.abs(p[1]))
        # plt.title(f'$\\omega^2={np.round(p[0],2)}$,$\\omega={np.round(np.sqrt(p[0]),2)}$, $j={j}$')
        plt.imshow(vp, extent=[-L/2,L/2,-l/2,l/2],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)
        plt.xlabel(f'$\\varphi_{j},   \\lambda_{j}$ = {p[0]:.3}', fontsize=17)

        plt.axis('scaled')


##

plt.figure(56)
plt.clf()

plt.subplot(1, 3, 1)

p = P[1]

vp=np.zeros([M+2,N+2])
for i in range(0,M+2):
    vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
alpha=np.max(np.abs(p[1]))
# plt.title('k=1')
plt.imshow(vp[::-1, :], extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)

axes = plt.gca()
axes.xaxis.set_ticks(range(5))
axes.xaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
axes.yaxis.set_ticks(range(5))
axes.yaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
plt.axis('scaled')

plt.subplot(1, 3, 2)

p = P[17]

vp=np.zeros([M+2,N+2])
for i in range(0,M+2):
    vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
alpha=np.max(np.abs(p[1]))
# plt.title('$k=2$')
plt.imshow(vp[::-1, :], extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)

axes = plt.gca()
axes.xaxis.set_ticks(range(5))
axes.xaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
axes.yaxis.set_ticks(range(5))
axes.yaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
plt.axis('scaled')

plt.subplot(1, 3, 3)

p = P[49]

vp=np.zeros([M+2,N+2])
for i in range(0,M+2):
    vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
alpha=np.max(np.abs(p[1]))
# plt.title('$k=3$')
plt.imshow(vp[::-1, :], extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)

axes = plt.gca()
axes.xaxis.set_ticks(range(5))
axes.xaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
axes.yaxis.set_ticks(range(5))
axes.yaxis.set_ticklabels(['-2', '-1', '0', '1', '2'], fontsize=17)
plt.axis('scaled')