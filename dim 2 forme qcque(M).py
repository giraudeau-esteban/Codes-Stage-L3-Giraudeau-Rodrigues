import numpy as np
import random as rnd
import pylab as plt
from scipy.linalg import eig, solve
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs

def fst(c):
    return c[0]

plt.ion()
plt.show()
plt.figure(1)
plt.clf()

# ############### pour changer la fenêtre et la discrétisation ######

L=6
l=6
N=26
M=26
h=L/(N+1)
k=l/(M+1)

# ################ différents supports : #########################

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=2
b=1
theta=0
rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

# Rectangle :
rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1
# ellipse :
ellipse=lambda x,y : x**2+y**2-1
# coeur :
coeur=lambda x,y : (x**2+y**2-1)**3+x**2*y**3

#rectangles+couloir (à améliorer)
couloir=lambda x,y : rectangle((x-2),2*y)*(rectangle((x-2),2*y)<=0)+rectangle((x+3/2)*2,y)*(rectangle((x+3/2)*2,y)<=0)+rectangle(x,y*6)*(rectangle(x,y*6)<=0)

f=rectangle
for i in range(M+2):
    for j in range(N+2):
        y=k*(i-1/2-M/2.0)
        x=h*(j-1/2-N/2.0)
        if f(rot(x,y)[0]/a,rot(x,y)[1]/b)<0:
            Supp[i,j]=1
#Supp[M//2-20:M//2+20,N//2-20:N//2+20]=0.5
Supp[1,2:4]=0
#Supp[2,2]=0
plt.imshow(Supp, extent=[0,L,0,l],aspect='auto', cmap='bwr')
plt.axis('scaled')
A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

# ############### Rigidité ###############################################
Horiz=1/h**2*np.ones((M+2)*(N+1))
Vert=1/k**2*np.ones((N+2)*(M+1))
R=np.concatenate((Horiz,Vert))
W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))

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

# ############ Calcul du laplacien et de ses valeurs propres : ##############

Lap=K.dot(A)
T=Lap.transpose()
Lap=T.dot(W).dot(Lap)

U,V=eigsh(Lap,k=15, sigma=7)
U=np.real(U)
V=np.real(V)
P = [(U[k], V[:,k]) for k in range(np.size(U))]
print(V)
P=sorted(P,key=fst)

# ############ Afichage des vecteurs propres ###############################
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
        plt.title(f'$\\omega={np.round(np.sqrt(p[0]),3)}$,$\\omega^2={np.round(p[0],3)}$ $j={j}$')
        plt.imshow(vp, extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)

        plt.axis('scaled')
        # plt.plot(X[1:-1],p[1])
        #plt.pause(0.75)