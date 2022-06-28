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

L=2
l=1
N=299
M=149
h=L/(N+1)
k=l/(M+1)

# ################ différents supports : #########################""

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=1/3
b=1/3
theta=0
rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

# Rectangle :
rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1
# ellipse :
ellipse=lambda x,y : 2*x**2+y**2-1
# coeur :
coeur=lambda x,y : (x**2+y**2-1)**3+x**2*y**3
# anneau
anneau=lambda x,y : (x**2+y**2-1)*(x**2+y**2-0.2)

#rectangles+couloir (à améliorer)
couloir=lambda x,y : rectangle((x-2),2*y)*(rectangle((x-2),2*y)<=0)+rectangle((x+3/2)*2,y)*(rectangle((x+3/2)*2,y)<=0)+rectangle(x,y*6)*(rectangle(x,y*6)<=0)

couloir2=lambda x,y: int(rectangle(4*(x+0.5), 4*y) >= 0)*int(rectangle(4*(x-0.5), 4*y) >= 0)*int(rectangle(2*x, 16*y) >= 0)

f=couloir2
# pts = np.array([ [h/4, k/4], [-h/4, k/4], [h/4, -k/4], [-h/4, -k/4], [0, k/4], [-h/4, 0], [h/4, 0], [0, -k/4], [0, 0] ])
for i in range(M+2):
    for j in range(N+2):
        P = rot(h*(j-1/2-N/2.0), k*(i-1/2-M/2.0))
        Supp[i,j] = int(f(P[0], P[1]) <= 0)
        # Supp[i,j]=sum([int(f((P+pi)[0], (P+pi)[1])<=0) for pi in pts])/9

plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
plt.axis('scaled')

plt.pause(3)

##

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

U,V=eigsh(Lap,k=40,sigma=20)
U=np.real(U)
V=np.real(V)
P = [(U[k], V[:,k]) for k in range(np.size(U))]
# print(V)
P=sorted(P,key=fst)

############ Afichage des vecteurs propres ###############################
j=0
for p in P:
    j=j+1
    if p[0]>10**-7:  # Pour éviter d'afficher les vp correspondant à 0.
        plt.clf()
        vp=np.zeros([M+2,N+2])
        for i in range(0,M+2):
            vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
        # plt.title(f'$\\lambda={p[0]}, \\omega={np.sqrt(p[0])}, j={j}$')
        # plt.imshow(vp+(1-Supp)*0.005, extent=[0,L,0,l],aspect='auto')

        plt.figure(102)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.linspace(0, L, N+2)
        Y = np.linspace(0, l, M+2)
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, vp, cmap=plt.cm.hsv,
                            linewidth=0, antialiased=True)

        # Customize the z axis.
        ax.set_zlim(0, 0.05)
        # ax.zaxis.set_major_locator(plt.ticker.LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # plt.plot(X[1:-1],p[1])
        # plt.axis('scaled')
        plt.pause(60)
        # plt.close()