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

def heptagone(centre, theta, A, B, p4, p6):

    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p1,p2,p3, p5, p7=torch.zeros([2], dtype=torch.double), torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double),torch.zeros([2], dtype=torch.double), torch.zeros([2], dtype=torch.double)
    p1[0], p7[0]=A/2.0, A/2.0
    p2[0],p3[0]=-A/2.0,-A/2.0
    p1[1],p2[1]=B/2.0, B/2.0
    p3[1],p7[1]=-B/2.0, -B/2.0
    p5[0], p5[1] = 0, -B/2.0
    p1 = centre + rot.matmul(p1)
    p2 = centre + rot.matmul(p2)
    p3 = centre + rot.matmul(p3)
    p5 = centre + rot.matmul(p5)
    p7 = centre + rot.matmul(p7)


    pent=torch.zeros([7,2,2])
    pent[0,0]=p1
    pent[0,1]=p2
    pent[1,0]=p2
    pent[1,1]=p3
    pent[2,0]=p3
    pent[2,1]=p4
    pent[3,0]=p4
    pent[3,1]=p5
    pent[4,0]=p5
    pent[4,1]=p6
    pent[5,0]=p6
    pent[5,1]=p7
    pent[6,0]=p7
    pent[6,1]=p1

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

    A=torch.sparse_coo_tensor([(torch.arange(0,(M+2)*(N+2))).tolist(),torch.arange(0,(M+2)*(N+2)).tolist()], Supp.reshape((M+2)*(N+2)), dtype=torch.double)

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

def M1(P):

    i=0
    while P[i][0]<10**-7:
        i=i+1

    return(P[i][1])

def E1(P, mu1):

    i=0
    while P[i][0]<10**-7:
        i=i+1
    S1=(P[i][0]-mu1)**2

    return S1

def PS2(P, U1):

    i=0
    while P[i][0]<10**-7:
        i=i+1

    U1l = U1.reshape((M+2)*(N+2))
    U1ln = U1l/torch.norm(U1l)

    S1 = torch.matmul(P[i+2][1]**2, U1ln**2)
    return(-S1)

def E2(P, U1, I):
    S=0

    for i in I:
        #print(P[i])
        S=S+torch.norm(P[i]**2-U1[i]**2)**2
    return S

def E_loc(P, k):

    i=0
    while P[i][0]<10**-7:
        i=i+1

    return (torch.norm(P[i+k-1][1], p=2)/torch.norm(P[i+k-1][1], p=4))**4

def f_espace(f, N, M, L, l):

    X, Y = torch.meshgrid(torch.linspace(-L/2.0, L/2.0, N+2), torch.linspace(-l/2, l/2, M+2))
    R = torch.zeros_like(X, dtype=torch.float64)

    for i in range(N):
        for j in range(M):
            R[i, j] = f(X[i, j], Y[i, j])

    return(R)

def affiche_espace(U, N, M, L, l):

    X, Y = torch.meshgrid(torch.linspace(-L/2.0, L/2.0, N+2), torch.linspace(-l/2, l/2, M+2))

    Xn = X.reshape((N+2)*(M+2))
    Yn = Y.reshape((N+2)*(M+2))
    Un = U.reshape((N+2)*(M+2))

    plt.plot([Xn[i] for i, t in enumerate(Un) if t > 0.5], [Yn[i] for i, t in enumerate(Un) if t > 0.5], linestyle='', marker = 'o')

    return()

def E_bord1(y1, y2):

    return(1/((1.0-y1)/0.3)**4 + 1/((1.0-y2)/0.3)**4)


def simili_anneau(R1, R2, N1, N2):

    P1 = R1*np.exp(np.complex(0, 1)*2*np.pi/N1*np.arange(0, N1+1))
    P2 = R2*np.exp(-np.complex(0, 1)*2*np.pi/N1*np.arange(0, N2+1))

    R = np.zeros((N1+N2, 2, 2))

    R[0:N1, 0, 0] = np.real(P1[0:-1])
    R[0:N1, 0, 1] = np.imag(P1[0:-1])
    R[0:N1, 1, 0] = np.real(P1[1:N1+1])
    R[0:N1, 1, 1] = np.imag(P1[1:N1+1])

    R[0:N1, 0, 0] = np.real(P2[0:-1])
    R[0:N1, 0, 1] = np.imag(P2[0:-1])
    R[0:N1, 1, 0] = np.real(P2[1:N1+1])
    R[0:N1, 1, 1] = np.imag(P2[1:N1+1])

    return(R)



## graphe double pentagone

# plt.ion()
# plt.show()
# plt.figure(14)
# plt.clf()

L=2.2
l=2.2
N=28
M=28
h=L/(N+1)
k=l/(M+1)
Ntra = 4

##

Oui = []
Non = []

Ndis1 = 50
Ndis2 = 50

X, Y = np.meshgrid(np.linspace(-0.3, 0.05, Ndis1), np.linspace(-0.8, -0.55, Ndis2))
R = np.zeros_like(X)
G = np.zeros_like(X)

s=3
eps=2/3

for k in range(Ndis1):
    for j in range(Ndis2):

        print(k, j)

        two_points = torch.tensor([X[j, k], Y[j, k]], requires_grad=True)

        p1, p2 = torch.tensor([-0.5, 0]), torch.tensor([0.5, 0])

        p1[1], p2[1] = two_points[0], two_points[1]

        Poly=heptagone(torch.tensor([0.0,0.5]), 0, 2, 1, p1, p2)
        A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
        P=vp_Laplacien(A,W,B,5,s-eps)

        loss = E_loc(P, 5) + E_bord1(two_points[0], two_points[1])
        en_val = loss.item()
        loss.backward()
        glong=two_points.grad
        err=torch.norm(glong)

        R[j, k] = err.item()
        G[j, k] = en_val

##

plt.figure(666)



Poly = heptagone(torch.tensor([0,0]), 0, 2, 1, torch.tensor([-0.5,-0.2]), torch.tensor([0.5,-1.3]))

plt.plot(Poly[:, 1, 0], Poly[:, 1, 1])

##

plt.ion()
plt.show()
plt.figure(5)
plt.clf()

plt.subplot(1, 2, 1)

x = plt.contour(X, Y, G, np.linspace(20, 200, 300), cmap='gist_rainbow_r')
# plt.clabel(x, fontsize=5, inline_spacing=1)
# cbar1 = plt.colorbar()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# cbar1.ax.tick_params(labelsize=14)
# plt.title('lignes de niveaux de l\'énergie', fontsize=14)


# plt.imshow(G, cmap='gist_rainbow')

plt.axis('scaled')

plt.subplot(1, 2, 2)

# plt.imshow(R, cmap='gist_rainbow',vmin=0, vmax=500)
# plt.colorbar()

r = plt.contour(X, Y, R, np.linspace(0, 500, 100), cmap='gist_rainbow_r')
# plt.clabel(r, fontsize=5, inline_spacing=1)
# cbar2 = plt.colorbar()
# cbar2.ax.tick_params(labelsize=14)

plt.axis('scaled')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# plt.title('lignes de niveaux de la norme du gradient', fontsize=14)

##

point = torch.tensor([0.2, 0.7], requires_grad=True)
Chemin = []

Chemin.append([point[0].item(), point[1].item()])

pas=0.01

optimizer=Adam([point], lr=pas)
s=3
eps=2/3
err=1

while err>10**(-3):

    p1, p2 = torch.tensor([-0.5, 0]), torch.tensor([0.5, 0])

    p1[1], p2[1] = point[0], point[1]

    Poly = heptagone(torch.tensor([0.0,0.5]), 0, 2, 1, p1, p2)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,5,s-eps)

    loss = E_loc(P, 5) + E_bord1(point[0], point[1])

    optimizer.zero_grad()
    loss.backward()
    glong=point.grad
    err=torch.norm(glong)

    optimizer.step()
    print(err)
    print('energie',  loss.item())
    print(point)

    Chemin.append([point[0].item(), point[1].item()])

##

Ch_tra = np.array(Chemin)

plt.figure(5)

plt.subplot(1, 2, 1)

plt.plot(Ch_tra[:, 0], Ch_tra[:, 1], color='orange')
plt.plot([Ch_tra[0][0]], [Ch_tra[0][1]], color='red', marker='o')
plt.plot([Ch_tra[-1][0]], [Ch_tra[-1][1]], color='red', marker='*')

plt.subplot(1, 2, 2)


plt.plot(Ch_tra[:, 0], Ch_tra[:, 1], color='orange')
plt.plot([Ch_tra[0][0]], [Ch_tra[0][1]], color='red', marker='o')
plt.plot([Ch_tra[-1][0]], [Ch_tra[-1][1]], color='red', marker='*')


plt.show()
##

plt.ion()
plt.show()
plt.figure(34)
plt.clf()


for j in range(4,5):

        # plt.subplot(1, 5, j+1)

        vp = np.zeros([M+2,N+2])
        for i in range(0,M+2):
            vp[i,:]=P[j][1][(N+2)*i:(N+2)*(i+1)].detach().numpy()

        plt.imshow(vp, extent=[-L/2, L/2, l/2, -l/2], cmap='bwr', vmin=-0.30, vmax=0.30)
        plt.plot(Poly.detach().numpy()[:, 1, 0], Poly.detach().numpy()[:, 1, 1], color='black')
        plt.plot(Poly.detach().numpy()[0, :, 0], Poly.detach().numpy()[0, :, 1], color='black')
        plt.scatter([-0.5, 0.5], point.detach().numpy(), color = 'green', linewidths=8)
##

plt.ion()
plt.show()
plt.figure(99)
plt.clf()


plt.plot(Poly.detach().numpy()[:, 0, 0], Poly.detach().numpy()[:, 0, 1], color='black')
plt.plot(Poly.detach().numpy()[-1, :, 0], Poly.detach().numpy()[-1, :, 1], color='black')
plt.gca().invert_yaxis()


##

plt.figure(5)
plt.clf()

Oui2 = np.array(Oui)
Non2 = np.array(Non)

plt.plot([-1.0, -1.0, 1.0, 1.0], [-0.5, 0.5, 0.5, -0.5], color='red')
plt.scatter([0], [-0.5], color='red')

affiche_espace(Uloc, N, M, L, l)

##

plt.plot(Oui2[:, 0], Oui2[:, 1], color = 'green')
plt.plot([-1.0, Oui2[-1, 0], 0], [-0.5, Oui2[-1, 1], -0.5], color = 'green', linestyle='--')
plt.plot(Non2[:, 0], Non2[:, 1], color = 'orange')
plt.plot([0, Non2[-1, 0], 1.0], [-0.5, Non2[-1, 1], -0.5], color = 'orange', linestyle='--')

## format cmap

plt.ion()
plt.show()
plt.figure(14)
plt.clf()

plt.imshow(R1)
plt.plot([Cont[3, 0, 0], Cont[0, 0, 0], Cont[1, 0, 0], Cont[2, 0, 0]], [Cont[3, 0, 1], Cont[0, 0, 1], Cont[1, 0, 1], Cont[2, 0, 1]], color='red')
plt.plot([Cont[2, 0, 0], Cont[3, 0, 0]], [Cont[2, 0, 1], Cont[3, 0, 1]], linestyle='--', color='red')

## graphe double pentagone

# plt.ion()
# plt.show()
# plt.figure(14)
# plt.clf()

L=4
l=4
N=299
M=299
h=L/(N+1)
k=l/(M+1)
Ntra = 4

##

Poly = simili_anneau(2, 2)
A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
P=vp_Laplacien(A,W,B,5,s-eps)

























