import numpy as np
import random as rnd
import pylab as plt
import torch
from torch.optim import Adam, Adagrad, AdamW, ASGD, SGD, Adadelta
from torch.linalg import eigh, eigvalsh
import torch.autograd.forward_ad as fwAD
from torch.sparse import mm
from scipy.sparse import bsr_matrix, vstack
from scipy.sparse.linalg import eigsh, eigs
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def fst(c):
    return c[0]

def dichotomie(f,a,b,eps=10**-8):
    c=(a+b)/2

    while torch.abs(f(c))>eps:
        if f(c)*f(a)>0:
            a=c
            c=(a+b)/2
        else:
            b=c
            c=(a+b)/2
    return c

def vp_Laplacien(A,W,B,n,s):
    I=torch.eye(A.size()[0]).to_sparse_coo()

    BA=mm(B,A)
    Lap=mm(mm(BA.transpose(0,1),W),BA)
    Inv=(Lap-s*I).to_dense().inverse()

    U,V=torch.lobpcg(Inv, k=n, largest=True)
    U=torch.real(1/U+s)
    V=torch.real(A.matmul(V))
    #print(U)

    P = [(U[k], V[:,k]) for k in range(U.size()[0])]
    P=sorted(P,key=fst)
    return P

def rectangle(centre, theta, A, B):

    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p1,p2,p3,p4=torch.zeros([2], dtype=torch.double), torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double)
    p1[0], p4[0]=A/2.0, A/2.0
    p2[0],p3[0]=-A/2.0,-A/2.0
    p1[1],p2[1]=B/2.0, B/2.0
    p3[1],p4[1]=-B/2.0, -B/2.0
    p1 = centre + rot.matmul(p1)
    p2 = centre + rot.matmul(p2)
    p3 = centre + rot.matmul(p3)
    p4 = centre + rot.matmul(p4)

    rect=torch.zeros([4,2,2])
    rect[0,0]=p1
    rect[0,1]=p2
    rect[1,0]=p2
    rect[1,1]=p3
    rect[2,0]=p3
    rect[2,1]=p4
    rect[3,0]=p4
    rect[3,1]=p1
    return rect

def pentagone(centre, theta, A, B, p4):

    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p1,p2,p3, p5=torch.zeros([2], dtype=torch.double), torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double)
    p1[0], p5[0]=A/2.0, A/2.0
    p2[0],p3[0]=-A/2.0,-A/2.0
    p1[1],p2[1]=B/2.0, B/2.0
    p3[1],p5[1]=-B/2.0, -B/2.0
    p1 = centre + rot.matmul(p1)
    p2 = centre + rot.matmul(p2)
    p3 = centre + rot.matmul(p3)
    p5 = centre + rot.matmul(p5)

    pent=torch.zeros([5,2,2])
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

    Supp = torch.zeros([M+2,N+2], dtype=torch.double)
    Horiz = 1/h**2 * torch.ones([(M+2),(N+1)], dtype=torch.double)
    Vert = 1/k**2 * torch.ones([(M+1),(N+2)], dtype=torch.double)


    for p in P:

        p1=torch.zeros([2], dtype=torch.double)
        p2=torch.zeros([2], dtype=torch.double)

        p1[0]=torch.trunc(p[0][0]/h+1/2+N/2.0)
        p1[1]=torch.trunc(p[0][1]/k+1/2+M/2.0)
        p2[0]=torch.trunc(p[1][0]/h+1/2+N/2.0)
        p2[1]=torch.trunc(p[1][1]/k+1/2+M/2.0)

        if p1[0] == p2[0]:
            a = (p[0][0]-p[1][0])/(p[0][1]-p[1][1])
            finv = lambda x: (x-p[0][1])*a + p[0][0]
            s2 = torch.sign(p[1][1]-p[0][1])
            if s2 > 0:
                p1=p1+torch.tensor([0,1])
            else:
                p2=p2+torch.tensor([0,1])

            for i in range(int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))+1):

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
                Supp[i,0:j+1] = Supp[i,0:j+1] + torch.ones(j+1)
        elif p1[1] == p2[1]:
            a = (p[0][1]-p[1][1])/(p[0][0]-p[1][0])
            f = lambda x: a*(x-p[0][0]) + p[0][1]

            s1 = torch.sign(p[1][0]-p[0][0])

            if s1 > 0:
                p1=p1+torch.tensor([1,0])
            else:
                p2=p2+torch.tensor([1,0])

            for j in range(int(min(p1[0], p2[0])), int(max(p1[0], p2[0])+1)):

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
        else:
            a = (p[0][1]-p[1][1])/(p[0][0]-p[1][0])
            f = lambda x: a*(x-p[0][0]) + p[0][1]
            finv = lambda x: (x-p[0][1])/a + p[0][0]


            s2 = torch.sign(p[1][1]-p[0][1])
            s1 = torch.sign(p[1][0]-p[0][0])


            if s2 > 0:
                if s1 > 0:
                    p1 = p1 + torch.tensor([1, 1], dtype=torch.double)
                else:
                    p1 = p1 + torch.tensor([0, 1], dtype=torch.double)
                    p2 = p2 + torch.tensor([1, 0], dtype=torch.double)
            else:
                if s1 > 0:
                    p1 = p1 + torch.tensor([1, 0], dtype=torch.double)
                    p2 = p2 + torch.tensor([0, 1], dtype=torch.double)
                else:
                    p2 = p2 + torch.tensor([1, 1], dtype=torch.double)

            for j in range(int(min(p1[0], p2[0])), int(max(p1[0], p2[0])+1)):

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

            for i in range(int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))+1):

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
                Supp[i,0:j+1] = Supp[i,0:j+1] + torch.ones(j+1)
    # print(Supp)

    Supp = Supp % 2
    Horiz2 = torch.zeros_like(Horiz, dtype=torch.double)
    Horiz2[0:M+2,0:N+1] = Horiz[0:M+2,0:N+1]

    Vert2 = torch.zeros_like(Vert, dtype=torch.double)
    Vert2[0:M+1,0:N+2] = Vert[0:M+1,0:N+2]

    #A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    A=torch.sparse_coo_tensor([(torch.arange(0,(M+2)*(N+2))).tolist(),torch.arange(0,(M+2)*(N+2)).tolist()],Supp.reshape((M+2)*(N+2)), dtype=torch.double)

    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=Vert.reshape([(M+1)*(N+2)])

    R=torch.concat((Horiz,Vert))
    #W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))
    W=torch.sparse_coo_tensor([torch.arange(0,(M+2)*(N+1)+(N+2)*(M+1)).tolist(),torch.arange(0,(M+2)*(N+1)+(N+2)*(M+1)).tolist()],R, dtype=torch.double)

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
    B=torch.tensor(B.toarray()).to_sparse_coo()

    return A,W,B, Supp, Horiz2, Vert2

def AWB_equation(bord, N,M,L,l):
    h=L/(N+1)
    k=l/(M+1)
    Supp=torch.zeros([M+2,N+2], dtype=torch.double)

    # ################# Masses #################################################

    for i in range(M+2):
        for j in range(N+2):
            y=k*(i-1/2-M/2.0)
            x=h*(j-1/2-N/2.0)
            if bord(x,y)<0:
                Supp[i,j]=1
    #A=torch.diag(Supp.reshape((M+2)*(N+2)))
    A=torch.sparse_coo_tensor([np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2))],Supp.reshape((M+2)*(N+2)), dtype=torch.double)

    # ############### Rigidité ###############################################

    Horiz=1/h**2*torch.ones([(M+2),(N+1)], dtype=torch.double)
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

    Vert=1/k**2*torch.ones([(M+1),(N+2)], dtype=torch.double)
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

    R=torch.cat((Horiz,Vert))
    #W=torch.diag(R)
    W=torch.sparse_coo_tensor([np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1))],R, dtype=torch.double)

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
    #B=torch.tensor(B.toarray())
    B=torch.tensor(B.toarray()).to_sparse_coo()
    return A,W,B

def V1(P):

    i=0
    while P[i][0]<10**-7:
        i=i+1

    return P[i][0]

def E1(P, mu1):

    i=0
    while P[i][0]<10**-7:
        i=i+1
    S1=(P[i][0]-mu1)**2

    return S1

def E2(P, U1, I):
    S=0

    for i in I:
        #print(P[i])
        S=S+torch.norm(P[i]**2-U1[i]**2)**2
    return S



## Opti n°1

L=6
l=6
N=56
M=56
h=L/(N+1)
k=l/(M+1)

#mu1=3.0842513753404246 #rectangle
mu1=4

long=torch.tensor(-1.8, requires_grad=True)

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=4
b=2
theta=0
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

# Rectangle :
#rect=lambda x,y : torch.max(torch.tensor([torch.abs(x),torch.abs(y)]))-1

#bord=lambda x,y: rect(2*rot(x,y)[0]/long, 2*rot(x,y)[1])
# long_primal=torch.tensor([1.5], dtype=torch.double)
# long_tangent=torch.tensor([1.0], dtype=torch.double)

# with fwAD.dual_level():
#dual_long=fwAD.make_dual(long_primal, long_tangent)


Poly=pentagone(torch.tensor([0,0]), theta,  a, b, long)
A,W,B,Supp, Horiz, Vert=AWB_polygone(Poly, N, M, L, l)
plt.figure(1)
plt.clf()
plt.show()
plt.imshow(Supp)

E=[]
s=1
eps=2/3
for long in torch.linspace(-0.0, -2.5, 201):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,1,s-eps)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    s=float(P[i][0])
    loss=E1(P,mu1)
    E.append(s)
    print(s)
plt.figure(1)
plt.clf()
plt.show()
plt.plot(torch.linspace(-0.5, -2.0, 201), E)

# pas=0.1
#
# optimizer=Adagrad([long], lr=pas)
# s=1
# eps=2/3
# err=1
# while err>10**(-4):
#     Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
#     A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
#     P=vp_Laplacien(A,W,B,1,s-eps)
#     i=0
#     while P[i][0]<10**-6:
#         i=i+1
#     s=float(P[i][0])
#     loss=E1(P,mu1)
#     optimizer.zero_grad()
#     loss.backward()
#     glong=long.grad
#     err=torch.norm(glong)
#
#     optimizer.step()
#     #print(s)
#     print(long)
#     print(err)

Poly=Poly.detach().numpy()
plt.ion()
print(long)

plt.figure(1)
plt.clf()
plt.show()
#plt.imshow(Supp.detach().numpy())
#plt.plot([1.0,2.0,5.0],[4.5,6,1])
plt.plot([Poly[0,0,0], Poly[1,0,0],Poly[2,0,0], Poly[3,0,0], Poly[4,0,0], Poly[0,0,0]], [Poly[0,0,1], Poly[1,0,1], Poly[2,0,1], Poly[3,0,1], Poly[4,0,1], Poly[0,0,1]])
#,
#

## Opti n°2

L=6
l=6
N=25
M=25
h=L/(N+1)
k=l/(M+1)

long=torch.tensor(-1.8, requires_grad=True)

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=4
b=1.8
theta=0
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

U1=torch.zeros([(M+2),(N+2)])
for i in range(int((l-b)//(2*k))+1, int((l+b)//(2*k))+1):
    for j in range(int((L-a)/(2*h))+1, int((L+a)/(2*h))+1):
        U1[i, j]=np.sin(np.pi*2*(j-int((L-a)/(2*h)))/(int((L+a)/(2*h))-int((L-a)/(2*h))+1))*np.sin(np.pi*(i-int((l-b)/(2*k)))/(int((l+b)/(2*k))-int((l-b)/(2*k))+1))

U1=U1/torch.norm(U1)
print(U1)
plt.ion()
plt.figure(3)
plt.clf()
plt.show()
plt.imshow(U1)
#
# Supp=torch.zeros([M+2,N+2])
# Rect=rectangle(torch.tensor([0,0]), 0, 4,2)
# A,W,B,Supp,Horiz,Vert=AWB_polygone(Rect, N, M, L, l)
# plt.figure(2)
# plt.clf()
# plt.show()
# plt.imshow(Supp)

U1=U1.reshape([1,(M+2)*(N+2)])

# pas=0.1
#
# optimizer=Adagrad([long], lr=pas)
# s=3
# eps=2/3
# err=1
# while err>10**(-4):
#     Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
#     A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
#     P=vp_Laplacien(A,W,B,2,s-eps)
#     i=0
#     while P[i][0]<10**-6:
#         i=i+1
#     V=torch.zeros_like(P[i+1][1])
#     V=P[i+1][1]
#     V=V.unsqueeze(0)
#     s=float(P[i][0])
#     loss=E2(V,U1, [i])
#     optimizer.zero_grad()
#     loss.backward()
#     glong=long.grad
#     err=torch.norm(glong)
#
#     optimizer.step()
#     #print(s)
#     print(long)
#     print(err)
plt.ion()
plt.figure(2)
plt.show()
E=[]
s=1
eps=2/3
for long in torch.linspace(-1.5, 0.5, 201):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)
    #eps=eps*1/10
    i=0
    while P[i][0]<10**-6:
        i=i+1
    s=float(P[i][0])
    V=torch.zeros_like(P[i][1])
    V=P[i][1]
    V=V.unsqueeze(0)
    loss=E2(V,U1, [i])
    # V=P[i][1]
    V=V.reshape([M+2,N+2])
    plt.clf()
    plt.imshow(V)
    plt.pause(0.1)
    E.append(loss)
    print(s)
plt.figure(1)
plt.clf()
plt.show()
plt.plot(torch.linspace(-1.5, 0.5, 201), E)
##
Poly=Poly.detach().numpy()
plt.ion()
print(long)
##
plt.ion()
plt.figure(1)
plt.clf()
plt.show()
plt.axis([-3,3,-3,3])
#plt.imshow(Supp.detach().numpy())
#plt.plot([1.0,2.0,5.0],[4.5,6,1])
plt.plot([Poly[0,0,0], Poly[1,0,0],Poly[2,0,0], Poly[3,0,0], Poly[4,0,0], Poly[0,0,0]], [Poly[0,0,1], Poly[1,0,1], Poly[2,0,1], Poly[3,0,1], Poly[4,0,1], Poly[0,0,1]])

plt.axis('scaled')

## graphe double pentagone

plt.ion()
plt.show()
plt.figure(14)
plt.clf()

L=8
l=8
N=59
M=59
h=L/(N+1)
k=l/(M+1)
Ntra = 50

X, Y = torch.meshgrid(torch.linspace(-2.8, 2.8, Ntra), torch.linspace(-0.4, 3.5, Ntra))
R1 = torch.zeros_like(X)


for j in range(Ntra):
    for i in range(Ntra):

        if j == 0:
            vs = 1
        else:
            vs = R1[i, j-1]

        print(i, j)

        Poly = pentagone(torch.tensor([0,0]), 0, 6, 1, torch.tensor([X[i, j], Y[i, j]]))
        A,W,B, Supp, Horiz, Vert = AWB_polygone(Poly,N,M,L,l)
        P = vp_Laplacien(A,W,B,2,vs)

        R1[i, j] = V1(P)

## jeez

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

Cont = rectangle(torch.tensor([0,0]), np.pi, 6, 1)
sol = torch.min(R1)

ax.scatter(X, Y, R1)
ax.plot3D([Cont[3, 0, 0], Cont[0, 0, 0], Cont[1, 0, 0], Cont[2, 0, 0]], [Cont[3, 0, 1], Cont[0, 0, 1], Cont[1, 0, 1], Cont[2, 0, 1]], [sol, sol, sol, sol], color='red')
ax.plot3D([Cont[2, 0, 0], Cont[3, 0, 0]], [Cont[2, 0, 1], Cont[3, 0, 1]], [sol, sol], linestyle='--', color='red')

# plt.imshow(Supp)

## format cmap

plt.ion()
plt.show()
plt.figure(14)
plt.clf()

plt.imshow(R1)
plt.plot([Cont[3, 0, 0], Cont[0, 0, 0], Cont[1, 0, 0], Cont[2, 0, 0]], [Cont[3, 0, 1], Cont[0, 0, 1], Cont[1, 0, 1], Cont[2, 0, 1]], color='red')
plt.plot([Cont[2, 0, 0], Cont[3, 0, 0]], [Cont[2, 0, 1], Cont[3, 0, 1]], linestyle='--', color='red')


