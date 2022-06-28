import numpy as np
import random as rnd
import pylab as plt
import time as tm
from scipy.linalg import eig, solve, norm
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs

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

plt.ion()
plt.show()

L=5
l=5
N=3
M=3
h=L/(N+1)
k=l/(M+1)

# ############ Calcul du laplacien et de ses valeurs propres : ##############
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

# ################ support rectangulaire : #########################""

Supp=np.zeros([M+2,N+2])
Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
Supp[1,2]=0
#Supp[2,1]=0

A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

# plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
# plt.axis('scaled')

# ############### création de la matrice d'incidence : #######################"

Bm=np.eye(N+1,N+2,k=1)-np.eye(N+1,N+2)

indices=np.arange(0,M+2)
indptr=np.arange(0,M+3)
data=np.array([Bm for i in range(M+2)])
K1=bsr_matrix((data,indices,indptr), shape=((M+2)*(N+1),(N+2)*(M+2)))

indices2=np.array([[i,i+1] for i in range(M+1)]).reshape(2*(M+1))
indptr2=np.array([2*i for i in range(M+2)])
data2=np.array([(-1)**(i+1)*np.eye(N+2) for i in range(2*(M+1))])
K2=bsr_matrix((data2,indices2,indptr2),shape=((N+2)*(M+1),(M+2)*(N+2)))

K=vstack((K1,K2))

# ############### Rigidité ###############################################
Alpha=np.linspace(0.001,0.999,200)
vp1=1.4067598490849047*np.ones_like(Alpha)
vp2=0.9768533465608993*np.ones_like(Alpha)

VP=[]
Vect=[]

for alpha in Alpha:

    Horiz=1/h**2*np.ones([M+2,N+1])
    Horiz[1,1]=1/h**2*1/alpha
    Horiz[1,2]=1/h**2*1/(1-alpha)
    Vert=1/k**2*np.ones([M+1,N+2])
    Vert[1,2]=1/k**2*1/alpha
    # Horiz[2,1]=1/h**2*2
    # Horiz[3,0]=1/h**2*2
    # Vert[2,1]=1/k**2*2

    # if alpha<1/4:
    #     Horiz[2,2]=1/h**2*1/(2*alpha)
    #     Horiz[3,3]=1/h**2*1/(2*alpha)
    #     Horiz[2,3]=10**12
    #     Vert[1,2]=1/k**2*1/(2*alpha)
    #     Vert[2,3]=1/k**2*1/(2*alpha)
    #     Vert[1,3]=10**12
    #
    # if alpha>=1/4 and alpha<1/2:
    #     Horiz[2,2]=1/h**2*1/(2*alpha)
    #     Horiz[3,3]=1/h**2*1/(2*alpha)
    #     Horiz[2,3]=10**12
    #     Vert[1,2]=1/k**2*2
    #     Vert[2,3]=1/k**2*1/(2*alpha)
    #     Vert[1,3]=10**12
    # if alpha>=1/2:
    #     Horiz[2,2]=1/h**2
    #     Horiz[3,3]=1/h**2
    #     Horiz[2,3]=1/h**2*1/(2*alpha-1)
    #     Vert[1,2]=1/k**2*2
    #     Vert[2,3]=1/k**2
    #     Vert[1,3]=1/h**2*1/(2*alpha-1)

    Horiz=Horiz.reshape([(M+2)*(N+1)])
    Vert=Vert.reshape([(M+1)*(N+2)])

    R=np.concatenate((Horiz,Vert))
    W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))

    P=vp_Laplacien(A,W,K,1,1)
    print(P[0][0])

    vp=P[0][0]
    VP.append(vp)
    Vect.append(P[0][1])



plt.figure(1)
plt.clf()

for i in range(1,21):
    plt.subplot(4,5,i)
    vp=np.zeros([M+2,N+2])
    for j in range(0,M+2):
        vp[j,:]=Vect[10*(i-1)][(N+2)*j:(N+2)*(j+1)]
    alpha=np.max(np.abs(Vect[10*(i-1)]))
    plt.title(f'$\\alpha={round(Alpha[10*(i-1)],3)}$')
    plt.imshow(np.abs(vp), extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)
    plt.axis('scaled')
    plt.axis('off')

plt.figure(2)
plt.clf()
plt.title('Evolution de la valeur propre $\\lambda_1$ en fonction de $\\alpha$', fontsize=16)
plt.xlabel('$\\alpha$', fontsize=14)
plt.ylabel('valeur propre', fontsize=14)

VP=np.array(VP)
plt.plot(Alpha, VP, label='$\\lambda_1(\\alpha)$')
plt.plot(Alpha, vp1, label='valeur propre domaine 1')
plt.plot(Alpha, vp2, label='valeur propre domaine 2')

plt.legend(fontsize=12)
