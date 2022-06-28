import numpy as np
import random as rnd
import pylab as plt
import ad
from scipy.linalg import eig, solve
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

def AWB_equation(bord, N,M,L,l):
    h=L/(N+1)
    k=l/(M+1)
    Supp=np.zeros([M+2,N+2])

    # ################# Masses #################################################

    for i in range(M+2):
        for j in range(N+2):
            y=k*(i-1/2-M/2.0)
            x=h*(j-1/2-N/2.0)
            if bord(x,y)<0:
                Supp[i,j]=1
    A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    # ############### Rigidité ###############################################

    Horiz=1/h**2*np.ones([(M+2),(N+1)])
    for i in range(M+2):
        for j in range(N+1):
            if Supp[i,j]==0 and Supp[i,j+1]==1:
                g=lambda alpha: bord(h*(j+1/2-N/2.0-alpha),k*(i-1/2-M/2.0))
                alpha=dichotomie(g,0,1)
                Horiz[i,j]=1/h**2*1/alpha
            if Supp[i,j]==1 and Supp[i,j+1]==0:
                g=lambda alpha: bord(h*(j-1/2-N/2.0+alpha),k*(i-1/2-M/2.0))
                alpha=dichotomie(g,0,1)
                Horiz[i,j]=1/h**2*1/alpha
            # if Supp[i,j]==0 and Supp[i,j+1]==0:
            #     Horiz[i,j]=10**6
    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=1/k**2*np.ones([(M+1),(N+2)])
    for j in range(N+2):
        for i in range(M+1):
            if Supp[i,j]==0 and Supp[i+1,j]==1:
                g=lambda alpha: bord(h*(j-1/2-N/2.0),k*(i+1/2-M/2.0-alpha))
                alpha=dichotomie(g,0,1)
                Vert[i,j]=1/k**2*1/alpha
            if Supp[i,j]==1 and Supp[i+1,j]==0:
                g=lambda alpha: bord(h*(j-1/2-N/2.0),k*(i-1/2-M/2.0+alpha))
                alpha=dichotomie(g,0,1)
                Vert[i,j]=1/k**2*1/alpha
            # if Supp[i,j]==0 and Supp[i+1,j]==0:
            #     Vert[i,j]=10**6
    Vert=Vert.reshape([(M+1)*(N+2)])

    R=np.concatenate((Horiz,Vert))
    W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))

    # ############### création de la matrice d'incidence : #######################"

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

    return A,W,B



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

# ################ différents supports : #########################

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=1
b=2
theta=2*np.pi/3
rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

# Rectangle :
rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1
# ellipse :
ellipse=lambda x,y : x**2+y**2-1
# coeur :
coeur=lambda x,y : (x**2+y**2-1)**3+x**2*y**3

#rectangles+couloir (à améliorer)
couloir=lambda x,y : rectangle((x-2),2*y)*(rectangle((x-2),2*y)<=0)+rectangle((x+3/2)*2,y)*(rectangle((x+3/2)*2,y)<=0)+rectangle(x,y*6)*(rectangle(x,y*6)<=0)




bord=lambda x,y: couloir(rot(x,y)[0]/a, rot(x,y)[1]/b)

A,W,B=AWB_equation(bord,N,M,L,l)

P=vp_Laplacien(A,W,B,15,35) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez

j=0
for p in P:
    j=j+1
    if p[0]>10**-7:  # Pour éviter d'afficher les vp correspondant à 0.
        print(p[0])
        plt.subplot(3,5,j)
        vp=np.zeros([M+2,N+2])
        for i in range(0,M+2):
            vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
        alpha=np.max(np.abs(p[1]))
        plt.title(f'$\\omega^2={np.round(p[0],2)}$,$\\omega={np.round(np.sqrt(p[0]),2)}$, $j={j}$')
        plt.imshow(vp, extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)

        plt.axis('scaled')
