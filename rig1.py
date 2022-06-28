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
l=2
N=200
M=200
h=L/(N+1)
k=l/(M+1)

# ################ différents supports : #########################""

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=1/2*np.sqrt(2)
b=1/2
theta=0
def rot(x, y, ang):
    return(np.array([np.cos(ang)*x-np.sin(ang)*y,np.sin(ang)*x+np.cos(ang)*y]))

# Rectangle :
rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1
# ellipse :
ellipse=lambda x,y : x**2+y**2-1
# coeur :
coeur=lambda x,y : (x**2+y**2-1)**3+x**2*y**3
cbas=lambda x,y:coeur(x, y+2.5)
chaut=lambda x,y:c1(rot(x, y, np.pi))
c1=lambda P:cbas(P[0], P[1])
trefle6 = lambda x,y: int(c1(rot(x,y,0)))*int(c1(rot(x,y,2*np.pi/6)))*int(c1(rot(x,y,4*np.pi/6)))*int(c1(rot(x,y,6*np.pi/6)))*int(c1(rot(x,y,8*np.pi/6)))*int(c1(rot(x,y,10*np.pi/6)))
# anneau
anneau=lambda x,y : (x**2+y**2-1)*(x**2+y**2-0.2)
# triangle
triangle=lambda x,y : int((y<0)) + int((np.sqrt(3)*(x+1)<y)) + int((-np.sqrt(3)*(x-1)<y))
# ve
ve =lambda x,y : int((y<0)) + int((np.sqrt(3)*(x+1)<y)) + int((-np.sqrt(3)*(x-1)<y)) + int((np.sqrt(3)*(x+1/2)>y)*(-np.sqrt(3)*(x-1/2)>y))
we = lambda x,y: ve(x-3/4, y)*ve(x+3/4, y)
tripleve = lambda x,y: ve(x-3/2, y)*ve(x+3/2, y)*ve(x,y)
triplevemiroir = lambda x,y: tripleve(x, y)*tripleve(x, -y)
# guitare
guitare = lambda x,y: int(ellipse(3*x, 3*y) >= 0)*int(ellipse(2*(x+0.75), 2.5*y) >= 0)*int(rectangle(0.8*x, 16*y) >= 0)*int(rectangle(8*(x-1.2), 12*y) >= 0)

#rectangles+couloir (à améliorer)
couloir=lambda x,y : rectangle((x-2),2*y)*(rectangle((x-2),2*y)<=0)+rectangle((x+3/2)*2,y)*(rectangle((x+3/2)*2,y)<=0)+rectangle(x,y*6)*(rectangle(x,y*6)<=0)

f=rectangle
for i in range(M+2):
    for j in range(N+2):
        y=k*(i-1/2-M/2.0)
        x=h*(j-1/2-N/2.0)
        # print(f(rot(x,y)[0]/a,rot(x,y)[1]/b))
        if f(rot(x,y, theta)[0]/a,rot(x,y, theta)[1]/b)<=0:

            Supp[i,j]=1

plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
plt.axis('scaled')

plt.pause(1)

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

U,V=eigsh(Lap,k=40,sigma=200)
U=np.real(U)
V=np.real(V)
P = [(U[k], V[:,k]) for k in range(np.size(U))]
# print(V)
P=sorted(P,key=fst)

print(np.array([u for (u, v) in P if u > 1])[:10])

############ Afichage des vecteurs propres ###############################
# j=0
# for p in P:
#     j=j+1
#     if p[0]>10**-7:  # Pour éviter d'afficher les vp correspondant à 0.
#         plt.clf()
#         vp=np.zeros([M+2,N+2])
#         for i in range(0,M+2):
#             vp[i,:]=p[1][(N+2)*i:(N+2)*(i+1)]
#         alpha = np.max(np.abs(p[1]))
#         plt.title(f'$\\lambda={p[0]}, \\omega={np.sqrt(p[0])}, j={j}$')
#         plt.imshow(vp+(1-Supp)*0.005, extent=[0,L,0,l],aspect='auto', cmap='bwr', vmin=-alpha, vmax=alpha)
#
#         # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#         #
#         # # Make data.
#         # X = np.linspace(0, L, N+2)
#         # Y = np.linspace(0, l, M+2)
#         # X, Y = np.meshgrid(X, Y)
#         #
#         # # Plot the surface.
#         # surf = ax.plot_surface(X, Y, vp, cmap=plt.cm.hsv,
#         #                     linewidth=0, antialiased=True)
#         #
#         # # Customize the z axis.
#         # ax.set_zlim(0, 0.05)
#         # # ax.zaxis.set_major_locator(plt.ticker.LinearLocator(10))
#         # # A StrMethodFormatter is used automatically
#         # # ax.zaxis.set_major_formatter('{x:.02f}')
#         #
#         # # Add a color bar which maps values to colors.
#         # fig.colorbar(surf, shrink=0.5, aspect=5)
#
#         # plt.plot(X[1:-1],p[1])
#         plt.axis('scaled')
#         plt.pause(1)
#         # plt.close()