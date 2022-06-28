import numpy as np
import random as rnd
import pylab as plt
import ad
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

def pentagone(centre, theta, A, B, h):

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p1,p2,p3,p4, p5=np.zeros([2]), np.zeros([2]),np.zeros([2]),np.zeros([2]),np.zeros([2])
    p1[0], p5[0]=A/2.0, A/2.0
    p2[0],p3[0]=-A/2.0,-A/2.0
    p1[1],p2[1]=B/2.0, B/2.0
    p3[1],p5[1]=-B/2.0, -B/2.0
    p4[0], p4[1]=0.0, h
    p1 = centre + rot.dot(p1)
    p2 = centre + rot.dot(p2)
    p3 = centre + rot.dot(p3)
    p4 = centre + rot.dot(p4)
    p5 = centre + rot.dot(p5)

    pent=np.zeros([5,2,2])
    pent[0,0]=p1
    pent[0,1]=p2
    pent[1,0]=p2
    pent[1,1]=p3
    pent[2,0]=p3
    pent[2,1]=p4
    pent[3,0]=p4
    pent[3,1]=p5
    pent[4,0]=p5
    pent[4,1]=p1
    return pent

def AWB_polygone(P, N, M, L, l):

    h = L/(N+1)
    k = l/(M+1)

    Supp = np.zeros([M+2,N+2])
    Horiz = 1/h**2 * np.ones([(M+2),(N+1)])
    Vert = 1/k**2 * np.ones([(M+1),(N+2)])


    for p in P:

        if p[0][0] == p[1][0]:
            print()
        elif p[0][1] == p[1][1]:
            print()
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

                x = h*(j-(N+1)/2)
                y = f(x)
                yi = y/k+(M+1)/2
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

                y = k*(i-(M+1)/2)
                x = finv(y)
                xj = x/h+(N+1)/2
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



#### Pentagone
plt.ion()
plt.show()
plt.figure(1)
plt.clf()

# ############### pour changer la fenêtre et la discrétisation ######

L=6
l=6
N=180
M=120
h=L/(N+1)
k=l/(M+1)

a=2
b=1
theta=0
rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

Poly=pentagone(np.array([0,0]), theta, a, b, -0.5)
A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
P=vp_Laplacien(A,W,B,2,10)

#plt.imshow(Supp)
print(P[0][0])

#### Pour voir les vp qui tendent vers le cas continu avec la discrétisation qui augmente


#
# # ############### pour changer la fenêtre et la discrétisation ######
#
L=6
l=6

NM_list=np.array([np.arange(10,200), np.arange(10,200)]).transpose()
# N=56
# M=56
vp=[]
Err=[]
for T in NM_list:
    N=T[0]
    M=T[1]
    h=L/(N+1)
    k=l/(M+1)


    Rec = rectangle(np.array([0, 0]), np.pi/4, 2, 1)
#Rec=np.array([[[ 0.9346,  1.6418],[-1.6418, -0.9346]],[[-1.6418, -0.9346],[-0.9346, -1.6418]],[[-0.9346, -1.6418],[ 1.6418,  0.9346]],[[ 1.6418,  0.9346],[ 0.9346,  1.6418]]])
    A,W,B, S, H, V=AWB_polygone(Rec, N, M, L, l)
    P=vp_Laplacien(A, W, B, 3, 12)
    i=0
    while P[i][0]<10**-7:
        i=i+1
    vp.append(P[i][0])
    print(P[i][0])
    Err.append(np.abs(12.337005501361698-P[i][0]))
##

vp=np.array(vp)

plt.ion()
plt.show()
plt.figure(1)
plt.clf()
plt.title('Convergence vers la solution continue', fontsize=16)
plt.xlabel('Discrétisation N=M', fontsize=14)
plt.ylabel('Valeur propre', fontsize=14)
plt.plot(NM_list[:,0], vp, label='solution numérique')
plt.plot(NM_list[:,0],12.337005501361698*np.ones_like(NM_list[:,0]), label='valeur propre du cas continu')
plt.legend(fontsize=12)

[a, b] = np.polyfit(np.log10(NM_list[:,0]), np.log10(Err),1)

plt.show()
plt.figure(2)
plt.clf()
plt.title('Ordre de convergence', fontsize=16)
plt.xlabel('Discrétisation N=M', fontsize=14)
plt.ylabel('Erreur par rapport à la solution continue', fontsize=14)
plt.loglog(NM_list[:,0], Err, 'o')
ti=1
tf=2.5
plt.loglog([10**ti,10**tf],[10**(a*ti+b),10**(a*tf+b)], label=f"pente : {round(a,3)}")

plt.legend(fontsize=12)


# for seg in Rec:
#     plt.plot(seg[:, 0], seg[:, 1])

#plt.imshow(1/H, extent=[0,L,0,l],aspect='auto', cmap='bwr')

#plt.axis('scaled')

# P=vp_Laplacien(A,W,B,15,35) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
# #
# j=0
# for p in P:
#     j=j+1
#     if p[0]>10**-7:  # Pour éviter d'afficher les vp correspondant à 0.
#         print(p[0])
#         plt.subplot(3,5,j)
#         vp=np.zeros([M+2,N+2])
#         for i in range(0,M+2):
#             vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
#         alpha=np.max(np.abs(p[1]))
#         plt.title(f'$\\omega^2={np.round(p[0],2)}$,$\\omega={np.round(np.sqrt(p[0]),2)}$, $j={j}$')
#         plt.imshow(vp, extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)
#
#         plt.axis('scaled')
