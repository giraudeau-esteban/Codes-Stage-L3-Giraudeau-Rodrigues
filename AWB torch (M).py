import numpy as np
import random as rnd
import pylab as plt
import ad
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

def vp_Laplacien(A,W,B,n,s, tau=1):
    I=torch.eye(A.size()[0]).to_sparse_coo()
    BA=mm(B,A)
    Lap=mm(mm(BA.transpose(0,1),W),BA)/tau
    Inv=(Lap-s/tau*I).to_dense().inverse()

    U,V=torch.lobpcg(Inv, k=n, largest=True)
    U=torch.real(1/U+s/tau)*tau
    V=torch.real(A.matmul(V))
    #print(U)

    P = [(U[k], V[:,k]) for k in range(U.size()[0])]
    P=sorted(P,key=fst)
    return P

def rectangle(centre, theta, A, B):

    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p1,p2,p3,p4=torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float)
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

def pentagone(centre, theta, A, B, x, y):
    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)
    p1,p2,p3,p4, p5=torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float)
    p1[0], p5[0]=A/2.0, A/2.0
    p2[0],p3[0]=-A/2.0,-A/2.0
    p1[1],p2[1]=B/2.0, B/2.0
    p3[1],p5[1]=-B/2.0, -B/2.0
    p4[0], p4[1]=x, y
    p1 = centre + rot.matmul(p1)
    p2 = centre + rot.matmul(p2)
    p3 = centre + rot.matmul(p3)
    p4 = centre + rot.matmul(p4)
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



def simili_anneau(R1, R2, N1, N2, a=1, b=1, theta=0):
    P1 = np.cosh(R1)*torch.cos(2*np.pi/N1*torch.arange(0, N1+1)+2)
    P2 = np.sinh(R1)*torch.sin(2*np.pi/N1*torch.arange(0, N1+1)+2)
    Q1 = np.cosh(R2)*torch.cos(-2*np.pi/N2*torch.arange(0, N2+1)+2)
    Q2 = np.sinh(R2)*torch.sin(-2*np.pi/N2*torch.arange(0, N2+1)+2)

    R = torch.zeros((N1+N2, 2, 2))

    R[0:N1, 0, 0] = P1[0:-1]
    R[0:N1, 0, 1] = P2[0:-1]
    R[0:N1, 1, 0] = P1[1:N1+1]
    R[0:N1, 1, 1] = P2[1:N1+1]

    R[N1:N1+N2, 0, 0] = Q1[0:-1]
    R[N1:N1+N2, 0, 1] = Q2[0:-1]
    R[N1:N1+N2, 1, 0] = Q1[1:N2+1]
    R[N1:N1+N2, 1, 1] = Q2[1:N2+1]

    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)

    for val in R:
        val[0] = rot.matmul(val[0])
        val[1] = rot.matmul(val[1])

    return(R)

# #### Maison rectangulaire avec un mur en bas : #######

def maison1(centre, theta, A,B, largeur, mur1x, mur1y):#, mur2x, mur2y, mur3x, mur3y):
    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)
    TR, TL, BL, BR, mur1bl, mur1tl, mur1tr, mur1br = torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float)
    TR[0], TR[1]=A/2.0, B/2.0
    TL[0],TL[1]=-A/2.0,B/2.0
    BL[0],BL[1]=-A/2.0, -B/2.0
    BR[0],BR[1]=A/2.0, -B/2.0
    mur1bl[0], mur1bl[1]=mur1x-largeur/2.0, -B/2.0
    mur1tl[0], mur1tl[1]=mur1x-largeur/2.0, mur1y
    mur1tr[0], mur1tr[1]=mur1x+largeur/2.0, mur1y
    mur1br[0], mur1br[1]=mur1x+largeur/2.0, -B/2.0


    #for p in [TR, TL, BL, BR, mur1bl, mur1br, mur1tr, mur1tl]:
    TR = centre + rot.matmul(TR)
    TL = centre + rot.matmul(TL)
    BL = centre + rot.matmul(BL)
    BR = centre + rot.matmul(BR)
    mur1bl = centre + rot.matmul(mur1bl)
    mur1br = centre + rot.matmul(mur1br)
    mur1tr = centre + rot.matmul(mur1tr)
    mur1tl = centre + rot.matmul(mur1tl)

    maison=torch.zeros([8,2,2])
    maison[0,0], maison[0,1] = TR, TL
    maison[1,0], maison[1,1] = TL, BL
    maison[2,0], maison[2,1] = BL, mur1bl
    maison[3,0], maison[3,1] = mur1bl, mur1tl
    maison[4,0], maison[4,1] = mur1tl, mur1tr
    maison[5,0], maison[5,1] = mur1tr, mur1br
    maison[6,0], maison[6,1] = mur1br, BR
    maison[7,0], maison[7,1] = BR, TR

    return maison

# #### Maison rectangulaire avec 3 mur (en bas, à droite et en haut) : #######

def maison2(centre, theta, A,B, largeur, mur1x, mur1y, mur2x, mur2y, mur3x, mur3y):
    rot = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)
    TR, TL, BL, BR, mur1bl, mur1tl, mur1tr, mur1br, mur2bl, mur2tl, mur2tr, mur2br, mur3bl, mur3tl, mur3tr, mur3br  = torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float),torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float), torch.zeros([2], dtype=torch.float)
    TR[0], TR[1]=A/2.0, B/2.0
    TL[0],TL[1]=-A/2.0,B/2.0
    BL[0],BL[1]=-A/2.0, -B/2.0
    BR[0],BR[1]=A/2.0, -B/2.0
    mur1bl[0], mur1bl[1]=mur1x-largeur/2.0, -B/2.0
    mur1tl[0], mur1tl[1]=mur1x-largeur/2.0, mur1y
    mur1tr[0], mur1tr[1]=mur1x+largeur/2.0, mur1y
    mur1br[0], mur1br[1]=mur1x+largeur/2.0, -B/2.0

    mur2bl[0], mur2bl[1]=mur2x, mur2y-largeur/2.0
    mur2tl[0], mur2tl[1]=mur2x, mur2y+largeur/2.0
    mur2tr[0], mur2tr[1]=A/2.0, mur2y+largeur/2.0
    mur2br[0], mur2br[1]=A/2.0, mur2y-largeur/2.0

    mur3bl[0], mur3bl[1]=mur3x-largeur/2.0, mur3y
    mur3tl[0], mur3tl[1]=mur3x-largeur/2.0, B/2.0
    mur3tr[0], mur3tr[1]=mur3x+largeur/2.0, B/2.0
    mur3br[0], mur3br[1]=mur3x+largeur/2.0, mur3y


    #for p in [TR, mur3tr, mur3br, mur3bl, mur3tl, TL, BL, mur1bl, mur1br, mur1tr, mur1tl, BR, mur2br, mur2bl, mur2tl, mur2tr]:

    TR = centre + rot.matmul(TR)
    mur3tr = centre + rot.matmul(mur3tr)
    mur3br = centre + rot.matmul(mur3br)
    mur3bl = centre + rot.matmul(mur3bl)
    mur3tl = centre + rot.matmul(mur3tl)
    TL = centre + rot.matmul(TL)
    BL = centre + rot.matmul(BL)
    mur1bl = centre + rot.matmul(mur1bl)
    mur1br = centre + rot.matmul(mur1br)
    mur1tr = centre + rot.matmul(mur1tr)
    mur1tl = centre + rot.matmul(mur1tl)
    BR = centre + rot.matmul(BR)
    mur2br = centre + rot.matmul(mur2br)
    mur2bl = centre + rot.matmul(mur2bl)
    mur2tl = centre + rot.matmul(mur2tl)
    mur2tr = centre + rot.matmul(mur2tr)

    maison=torch.zeros([16,2,2])
    maison[0,0], maison[0,1] = TR, mur3tr
    maison[1,0], maison[1,1] = mur3tr, mur3br
    maison[2,0], maison[2,1] = mur3br, mur3bl
    maison[3,0], maison[3,1] = mur3bl, mur3tl
    maison[4,0], maison[4,1] = mur3tl, TL
    maison[5,0], maison[5,1] = TL, BL
    maison[6,0], maison[6,1] = BL, mur1bl
    maison[7,0], maison[7,1] = mur1bl, mur1tl
    maison[8,0], maison[8,1] = mur1tl, mur1tr
    maison[9,0], maison[9,1] = mur1tr, mur1br
    maison[10,0], maison[10,1] = mur1br, BR
    maison[11,0], maison[11,1] = BR, mur2br
    maison[12,0], maison[12,1] = mur2br, mur2bl
    maison[13,0], maison[13,1] = mur2bl, mur2tl
    maison[14,0], maison[14,1] = mur2tl, mur2tr
    maison[15,0], maison[15,1] = mur2tr, TR

    return maison

def AWB_polygone(P, N, M, L, l):

    h = L/(N+1)
    k = l/(M+1)

    Supp = torch.zeros([M+2,N+2], dtype=torch.float)
    Horiz = 1/h**2 * torch.ones([(M+2),(N+1)], dtype=torch.float)
    Vert = 1/k**2 * torch.ones([(M+1),(N+2)], dtype=torch.float)


    for p in P:

        p1=torch.zeros([2], dtype=torch.float)
        p2=torch.zeros([2], dtype=torch.float)

        p1[0]=torch.trunc(p[0][0]/h+1/2+N/2.0)
        p1[1]=torch.trunc(p[0][1]/k+1/2+M/2.0)
        p2[0]=torch.trunc(p[1][0]/h+1/2+N/2.0)
        p2[1]=torch.trunc(p[1][1]/k+1/2+M/2.0)

        if p1[0] == p2[0]:
            if p1[1]!=p2[1]:
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
            else:
                print('ok')
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
                    p1 = p1 + torch.tensor([1, 1], dtype=torch.float)
                else:
                    p1 = p1 + torch.tensor([0, 1], dtype=torch.float)
                    p2 = p2 + torch.tensor([1, 0], dtype=torch.float)
            else:
                if s1 > 0:
                    p1 = p1 + torch.tensor([1, 0], dtype=torch.float)
                    p2 = p2 + torch.tensor([0, 1], dtype=torch.float)
                else:
                    p2 = p2 + torch.tensor([1, 1], dtype=torch.float)

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
    Horiz2 = torch.zeros_like(Horiz, dtype=torch.float)
    Horiz2[0:M+2,0:N+1] = Horiz[0:M+2,0:N+1]

    Vert2 = torch.zeros_like(Vert, dtype=torch.float)
    Vert2[0:M+1,0:N+2] = Vert[0:M+1,0:N+2]

    #A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    A=torch.sparse_coo_tensor([(torch.arange(0,(M+2)*(N+2))).tolist(),torch.arange(0,(M+2)*(N+2)).tolist()],Supp.reshape((M+2)*(N+2)), dtype=torch.float)

    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=Vert.reshape([(M+1)*(N+2)])

    R=torch.concat((Horiz,Vert))
    #W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))
    W=torch.sparse_coo_tensor([torch.arange(0,(M+2)*(N+1)+(N+2)*(M+1)).tolist(),torch.arange(0,(M+2)*(N+1)+(N+2)*(M+1)).tolist()],R, dtype=torch.float)

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
    B=torch.tensor(B.toarray(), dtype=torch.float).to_sparse_coo()

    return A,W,B, Supp, Horiz2, Vert2

def AWB_equation(bord, N,M,L,l):
    h=L/(N+1)
    k=l/(M+1)
    Supp=torch.zeros([M+2,N+2], dtype=torch.float)

    # ################# Masses #################################################

    for i in range(M+2):
        for j in range(N+2):
            y=k*(i-1/2-M/2.0)
            x=h*(j-1/2-N/2.0)
            if bord(x,y)<0:
                Supp[i,j]=1
    #A=torch.diag(Supp.reshape((M+2)*(N+2)))
    A=torch.sparse_coo_tensor([np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2))],Supp.reshape((M+2)*(N+2)), dtype=torch.float)

    # ############### Rigidité ###############################################

    Horiz=1/h**2*torch.ones([(M+2),(N+1)], dtype=torch.float)
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

    Vert=1/k**2*torch.ones([(M+1),(N+2)], dtype=torch.float)
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
    W=torch.sparse_coo_tensor([np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1))],R, dtype=torch.float)

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
    B=torch.tensor(B.toarray(), dtype=float).to_sparse_coo()
    return A,W,B

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
        S=S+(P[i]**2).matmul(U1[i]**2)
    return -S

    # for i in I:
    #     #print(P[i])
    #     S=S+torch.norm(P[i]**2-U1[i]**2)**2
    # return S
def Eloc(posx, posy, V, U1):
    chi=10**-1
    sigma=0.05
    S=(V**2).matmul(U1**2)-chi*(torch.exp(-(posx-2)**2/sigma**2)+torch.exp(-(posx+2)**2/sigma**2)+torch.exp(-(posy-0.9)**2/sigma**2))

    return -S

def Epoly(Poly, V, U1):
    R=torch.tensor([[0,1],[-1,0]], dtype=torch.float)
    chi=10**2
    sigma=0.05
    X=[]
    Y=[]
    UV=[]
    for p in Poly:
        xA=p[0][0]
        yA=p[0][1]
        xB=p[1][0]
        yB=p[1][1]
        K=int(((xB-xA)**2+(yB-yA)**2)**(1/2)/sigma)+1
        n=torch.arange(0,K, dtype=torch.float)
        X.append(xB*(n/K)+xA*(1-n/K))
        Y.append(yB*(n/K)+yA*(1-n/K))

        u=torch.tensor([xB-xA, yB-yA], dtype=torch.float)
        u=u/torch.norm(u)
        v=R.matmul(u)

        UV.append([u,v])
    # X=torch.tensor(X)
    # Y=torch.tensor(Y)
    i=0
    S=0

    for x,y in zip(X,Y):
        K=len(x)
        for k in range(K):
            for j, p in enumerate(Poly):
                #if (not(i in [1,3,7,9,12,14]) and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly)])) or (i in [1,7,12] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i+2)%len(Poly)])) or (i in [3,9,14] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i-2)%len(Poly)])):
                if not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly)]) :
                    xn=x[k]
                    yn=y[k]
                    xC=p[0][0]
                    yC=p[0][1]
                    xD=p[1][0]
                    yD=p[1][1]
                    CM=torch.tensor([xn-xC, yn-yC])
                    DM=torch.tensor([xn-xD, yn-yD])
                    d2=np.max([0,CM.matmul(-UV[j][0]), DM.matmul(UV[j][0])])**2+(CM.matmul(UV[j][1]))**2

                    #print(i, k, j, d2**0.5)
                    S=S+chi*torch.exp(-d2/sigma**2)
        i=i+1
    S=-(torch.abs(V)).matmul(U1**2)+S

    return S

def Epoly2(Poly, V):
    R=torch.tensor([[0,1],[-1,0]], dtype=torch.float)
    chi=10**2
    sigma=0.15
    X=[]
    Y=[]
    UV=[]
    for p in Poly:
        xA=p[0][0]
        yA=p[0][1]
        xB=p[1][0]
        yB=p[1][1]
        K=int(((xB-xA)**2+(yB-yA)**2)**(1/2)/sigma)+1
        n=torch.arange(0,K, dtype=torch.float)
        X.append(xB*(n/K)+xA*(1-n/K))
        Y.append(yB*(n/K)+yA*(1-n/K))

        u=torch.tensor([xB-xA, yB-yA], dtype=torch.float)
        u=u/torch.norm(u)
        v=R.matmul(u)

        UV.append([u,v])
    # X=torch.tensor(X)
    # Y=torch.tensor(Y)
    i=0
    S=0
    for x,y in zip(X,Y):
        K=len(x)
        for k in range(K):

            for j, p in enumerate(Poly):
                # if not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly)]) : ### Décommenter si ni maison 1 ni maison 2
                # if (not(i in [1,3,7,9,12,14]) and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly)])) or (i in [1,7,12] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i+2)%len(Poly)])) or (i in [3,9,14] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i-2)%len(Poly)])): ### Décommenter pour maison 2
                if (not(i in [3,5]) and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly)])) or (i in [3] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i+2)%len(Poly)])) or (i in [5] and not(j in [i,(i-1)%len(Poly),(i+1)%len(Poly), (i-2)%len(Poly)])): ### Décommenter pour maison 1
                    xn=x[k]
                    yn=y[k]
                    xC=p[0][0]
                    yC=p[0][1]
                    xD=p[1][0]
                    yD=p[1][1]
                    CM=torch.tensor([xn-xC, yn-yC])
                    DM=torch.tensor([xn-xD, yn-yD])
                    d2=np.max([0,CM.matmul(-UV[j][0]), DM.matmul(UV[j][0])])**2+(CM.matmul(UV[j][1]))**2

                    #print(i, k, j, d2**0.5)
                    S=S+chi*torch.exp(-d2/sigma**2)
        i=i+1
    S=(torch.norm(torch.abs(V), p=2)**4)/(torch.norm(torch.abs(V), p=4)**4)+S

    return S
## tests domaines
L=5
l=5
N=101
M=101
h=L/(N+1)
k=l/(M+1)

Poly=simili_anneau(1.2,0.5,25,30)

A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly, N, M, L, l)

plt.ion()
plt.show()
plt.figure(3)
plt.clf()
plt.imshow(Supp)


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
# long_primal=torch.tensor([1.5], dtype=torch.float)
# long_tangent=torch.tensor([1.0], dtype=torch.float)

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
for long in torch.linspace(0.5, -2.5, 201):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,1,s-eps)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    s=float(P[i][0])
    loss=E1(P,mu1)
    E.append(loss)
    print(s)
plt.figure(1)
plt.clf()
plt.show()
plt.plot(torch.linspace(0.5, -2.5, 201), E)

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
##
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

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=4
b=1.8
theta=0
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

U1=torch.zeros([(M+2),(N+2)], dtype=torch.float)
for i in range(int((l-b)//(2*k))+1, int((l+b)//(2*k))+1):
    for j in range(int((L-a)/(2*h))+1, int((L+a)/(2*h))+1):
        U1[i, j]=np.sin(np.pi*2*(j-int((L-a)/(2*h)))/(int((L+a)/(2*h))-int((L-a)/(2*h))+1))*np.sin(np.pi*(i-int((l-b)/(2*k)))/(int((l+b)/(2*k))-int((l-b)/(2*k))+1))

U1=U1/torch.norm(U1)
# print(U1)
# plt.ion()
# plt.figure(3)
# plt.clf()
# plt.show()
# plt.imshow(U1)
#
# Supp=torch.zeros([M+2,N+2])
# Rect=rectangle(torch.tensor([0,0]), 0, 4,2)
# A,W,B,Supp,Horiz,Vert=AWB_polygone(Rect, N, M, L, l)
# plt.figure(2)
# plt.clf()
# plt.show()
# plt.imshow(Supp)

U1=U1.reshape([1,(M+2)*(N+2)])


long=torch.tensor(-1.8, requires_grad=True)
pas=0.1

optimizer=Adam([long], lr=pas)
s=1
eps=2/3
err=1
Long=[]
while err>10**(-5):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, 0.0,long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)

    i=0
    while P[i][0]<10**-6:
        i=i+1
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    V=V.unsqueeze(0)
    s=float(P[i][0])
    loss=E2(V,U1, [i])
    optimizer.zero_grad()
    loss.backward()
    glong=long.grad
    err=torch.norm(glong)

    optimizer.step()
    #print(s)
    Long.append(long.detach().item())
    print(long.detach().item())
    print(err)
print(Long)
plt.figure(1)
plt.clf()
plt.show()
plt.plot(Long)

long=torch.tensor(-2.0, requires_grad=True)
pas=0.1

optimizer=Adagrad([long], lr=pas)
s=1
eps=2/3
err=1
Long=[]
while err>10**(-5):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, 0.0,long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    V=V.unsqueeze(0)
    s=float(P[i][0])
    loss=E2(V,U1, [i])
    optimizer.zero_grad()
    loss.backward()
    glong=long.grad
    err=torch.norm(glong)

    optimizer.step()
    #print(s)
    Long.append(long.detach().item())
    print(long.detach().item())
    print(err)
print(Long)
# plt.figure(1)
# plt.clf()
# plt.show()
plt.plot(Long)

plt.plot([0.0,100], [-0.9,-0.9])
##↨
plt.ion()
plt.figure(2)
plt.show()
E=[]
s=1
eps=2/3
for long in torch.linspace(-2.3, 0.5, 201):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, long)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)
    #eps=eps*1/10
    i=0
    while P[i][0]<10**-6:
        i=i+1
    s=float(P[i][0])
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    V=V.unsqueeze(0)
    loss=E2(V,U1, [i])
    # # V=P[i][1]
    # V=V.reshape([M+2,N+2])
    # plt.clf()
    # plt.imshow(V)
    # plt.pause(0.1)
    E.append(loss)
    print(s)

plt.figure(1)
plt.clf()
plt.show()
plt.plot(torch.linspace(-2.3, 0.5, 201), E)
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

## Opti 3

L=5
l=5
N=36
M=36
h=L/(N+1)
k=l/(M+1)

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=4.
b=1.8
theta=0
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

U1=torch.zeros([(M+2),(N+2)], dtype=torch.float)
for i in range(int((l-b)//(2*k))+1, int((l+b)//(2*k))+1):
    for j in range(int((L-a)/(2*h))+1, int((L+a)/(2*h))+1):
        U1[i, j]=np.sin(np.pi*2*(j-int((L-a)/(2*h)))/(int((L+a)/(2*h))-int((L-a)/(2*h))+1))*np.sin(np.pi*(i-int((l-b)/(2*k)))/(int((l+b)/(2*k))-int((l-b)/(2*k))+1))

U1=U1/torch.norm(U1)

#print(U1)
plt.ion()
plt.figure(3)
plt.clf()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
plt.imshow(U1**2, extent=[-2.5,2.5,-2.5,2.5], cmap='bwr', vmin=-torch.max(U1**2), vmax=torch.max(U1**2))
plt.plot([-2.0, 2.0],[-0.9, -0.9], color='red', linestyle='--')
plt.plot([-2.0, -2.0], [-0.9, 0.9], color='red', linestyle='--')
plt.plot([-2.0, 2.0],[0.9,0.9], color='red', linestyle='--')
plt.plot([2.0, 2.0], [-0.9,0.9], color='red', linestyle='--')
# Supp=torch.zeros([M+2,N+2])
# Rect=rectangle(torch.tensor([0,0]), 0, 4,2)
# A,W,B,Supp,Horiz,Vert=AWB_polygone(Rect, N, M, L, l)
# plt.figure(2)
# plt.clf()
# plt.show()
# plt.imshow(Supp)
##
U1=U1.reshape([(M+2)*(N+2)])

X=torch.linspace(-1.95,1.95,101, dtype=torch.float)
Y=torch.linspace(-1.85,0.85,101, dtype=torch.float)

mu1=3
s=1
eps=1/2
Loss=[]
t=0
for posy in Y:
    # t=t+1
    print(posy)
    for posx in X:
        Poly=pentagone(torch.tensor([0,0], dtype=torch.float), theta, a, b, posx, posy)
        A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly, N, M, L, l)
        P=vp_Laplacien(A,W,B, 2, s-eps)
        i=0
        while P[i][0]<10**-6:
            i=i+1
        V=torch.zeros_like(P[i+1][1])
        V=P[i+1][1]
        # V=V.unsqueeze(0)
        s=float(P[i][0])
        eps=s/2
        #print(s)
        #loss=E1(P,mu1)
        #loss=Eloc(posx, posy,V,U1)
        loss=Epoly(Poly, V, U1)
        #loss=Epoly2(Poly, V)
        Loss.append(loss)
        #print(s)
        # if t==17:
        #     Im=P[i+1][1].reshape([M+2,N+2])
        #     plt.imshow(Im.detach().numpy())
        #     plt.pause(0.5)

Loss=torch.tensor(Loss)
##
Loss=Loss.reshape([101,101])

[X,Y]=np.meshgrid(X,Y)
##
plt.figure(2)
plt.clf()
plt.show()
plt.imshow(Loss.detach().numpy()[:-5,5:-5],extent=[-2,2,0.5,-1.85],aspect='auto')
plt.axis('scaled')
##
pos=torch.tensor([0.5,0.0], requires_grad=True)
pas=0.1

optimizer=Adam([pos], lr=pas)
s=1
eps=2/3
err=1
Posx=[pos[0].detach().item()]
Posy=[pos[1].detach().item()]
while err>10**(-7):
    Poly=pentagone(torch.tensor([0,0]), theta, a, b, pos[0], pos[1])
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps, tau=10**-16)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    # V=V.unsqueeze(0)
    s=float(P[i][0])
    #loss=Eloc(pos[0],pos[1],V,U1)
    loss=Epoly(Poly, V, U1)
    optimizer.zero_grad()
    loss.backward()
    glong=pos.grad
    err=torch.norm(glong)
    print(loss)

    optimizer.step()
    #print(s)
    Posx.append(pos[0].detach().item())
    Posy.append(pos[1].detach().item())
    print(err)
#print(Long)
##
plt.figure(1)
plt.clf()
plt.show()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
CS=plt.contour(X,Y,Loss.detach().numpy(), [-0.0682,-0.06815,-0.068,-0.067,-0.065,-0.06, -0.05,-0.04,-0.03,-0.02,-0.01 -0.005,-0.0045,-0.004,-0.003, -0.0024, 0.0, 0.001], cmap='rainbow')


#plt.clabel(CS, inline=1,fontsize=10)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=18)
#[0.0, 10,20,30,35,40,50,60,70,80,90,100,120,140,160,180,200,250,300])
#-0.00535, -0.0053,-0.0052, -0.005,-0.0045,-0.004,-0.003, -0.0024, 0.0, 0.001,0.01,0.1,1])
#-0.42,-0.4,-0.38,-0.35,-0.325,-0.3,-0.29, -0.28,-0.27, -0.265, 0.0])

print([pos[0],pos[1]])
# plt.figure(1)
# plt.clf()
# plt.show()
plt.plot(Posx, Posy, color='brown')
plt.plot(Posx[0], Posy[0],'o', color='blue')
plt.plot(Posx[-1], Posy[-1],'*', markersize=10, color='blue')
plt.plot([-2.0, 2.0],[-0.9, -0.9], color='red', linestyle='--')
plt.plot([-2.0, -2.0], [-0.9, 0.9], color='red', linestyle='--')
plt.plot([-2.0, 2.0],[0.9,0.9], color='red', linestyle='--')
plt.plot([2.0, 2.0], [-0.9,0.9], color='red', linestyle='--')
# plt.plot([0.0,100], [-0.9,-0.9])
# plt.plot([0,100], [0.0,0.0])
plt.axis('scaled')

##
Poly=pentagone(torch.tensor([0,0]), theta, a, b, pos[0], pos[1])
A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
P=vp_Laplacien(A,W,B,2,s-eps, tau=10**-16)
i=0
while P[i][0]<10**-6:
    i=i+1
V=torch.zeros_like(P[i+1][1])
V=P[i+1][1]
V=V.reshape([M+2,N+2])


plt.figure(6515)
plt.clf()
plt.show()
plt.imshow(V.detach().numpy())

## Opti maison1

L=5
l=5
N=36
M=36
h=L/(N+1)
k=l/(M+1)

Supp=np.zeros([M+2,N+2])

# Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
a=4.
b=1.8
theta=0
largeur_murs=0.2
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

mur=torch.tensor([0.5,-0.2], requires_grad=True)
pas=10**-5

optimizer=SGD([mur], lr=pas)
s=1
eps=1/15
err=1
Posx=[]
Posy=[]
while err>10**(-4):
    Poly=maison1(torch.tensor([0,0]), theta, a, b,largeur_murs, mur[0], mur[1])
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    # V=V.unsqueeze(0)
    s=float(P[i][0])
    eps=1/15*s
    #loss=Eloc(pos[0],pos[1],V,U1)
    loss=Epoly2(Poly, V)
    optimizer.zero_grad()
    loss.backward()
    glong=mur.grad
    err=torch.norm(glong)
    print(loss)

    optimizer.step()
    print(s)
    Posx.append(mur[0].detach().item())
    Posy.append(mur[1].detach().item())
    print(glong)
    print(mur)
    print(err)
##
plt.ion()
plt.show()

plt.figure(1)
plt.clf()
plt.imshow(P[i+1][1].reshape([M+2,N+2]).detach().numpy())

plt.figure(2)
plt.clf()
plt.imshow(Supp)

##

a=4.
b=1.8
theta=0
largeur_murs=0.2
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

L=5
l=5
N=36
M=36
h=L/(N+1)
k=l/(M+1)

X=torch.linspace(-1.95,1.95,101, dtype=torch.float)
Y=torch.linspace(-0.85,0.85,101, dtype=torch.float)
#Y=torch.tensor([0.0], dtype=torch.float)

mu1=3
s=1
eps=1/10
Loss=[]
t=0
for posy in Y:
    # t=t+1
    print(posy)
    for posx in X:
        Poly=maison1(torch.tensor([0,0]), theta, a, b,largeur_murs, posx, posy)
        A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly, N, M, L, l)
        P=vp_Laplacien(A,W,B, 2, s-eps)
        i=0
        while P[i][0]<10**-6:
            i=i+1
        V=torch.zeros_like(P[i+1][1])
        V=P[i+1][1]
        # V=V.unsqueeze(0)
        s=float(P[i][0])
        eps=s/10
        #print(s)
        #loss=E1(P,mu1)
        #loss=Eloc(posx, posy,V,U1)
        # loss=Epoly(Poly, V, U1)
        loss=Epoly2(Poly, V)
        Loss.append(loss)
        #print(s)
        # if t==17:
        # Im=P[i+1][1].reshape([M+2,N+2])
        # plt.imshow(Im.detach().numpy())
        # plt.pause(0.5)

Loss=torch.tensor(Loss)
##
Loss=Loss.reshape([101,101])

[X,Y]=np.meshgrid(X,Y)
##

Im=Loss.detach().tolist()
Im.reverse()
Im=np.array(Im)

plt.figure(2)
plt.clf()
plt.show()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.imshow(Im,extent=[-1.85,1.85,-0.85,0.85],aspect='auto', vmin=110, vmax=250)
plt.axis('scaled')
cb=plt.colorbar(shrink=0.75)
cb.ax.tick_params(labelsize=20)
#[7:-7,6:-6]
##
plt.figure(1)
plt.clf()
plt.show()
CS=plt.contour(X,Y,Loss.detach().numpy(), [130,140,160,180, 190,200,220,240,260,280], cmap='hsv')

plt.clabel(CS, inline=1,fontsize=12)
#plt.colorbar()
plt.axis('scaled')

##
print([mur[0],mur[1]])
# plt.figure(1)
# plt.clf()
# plt.show()
plt.plot(Posx, Posy)
plt.plot([-2.0, 2.0],[-0.9, -0.9], color='red')
plt.plot([-2.0, -2.0], [-0.9, 0.9], color='red')
plt.plot([-2.0, 2.0],[0.9,0.9], color='red')
plt.plot([2.0, 2.0], [-0.9,0.9], color='red')
# plt.plot([0.0,100], [-0.9,-0.9])
# plt.plot([0,100], [0.0,0.0])
plt.axis('scaled')


## Opti maison2

L=5
l=5
N=36
M=36
h=L/(N+1)
k=l/(M+1)

Supp=np.zeros([M+2,N+2])

a=4.
b=1.8
theta=0
largeur_murs=0.2
rot=lambda x,y:torch.tensor([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])


# ### Conditions initiales (CI) #############################

murs=torch.tensor([0.6, 0.6, 1.,-0.64,-1.48,0.23], requires_grad=True)

# ################ Plusieurs CI à copier-coller ############
#[-0.53, -0.52, 0.51,-0.51,1.52,0.21]                    ###
#[-1.2, -0.52, 1.3,-0.3,0.0,0.21]                        ###
#[0.6, 0.6, 1.,-0.64,-1.48,0.23]                         ###
# ##########################################################


pas=0.5*10**-4

optimizer=SGD([murs], lr=pas) ## Plusieurs algorithmes possibles : SGD, Adagrad, Adam.
s=4 ## Shift pour éviter les vecteurs propres associé à 0.
eps=4/3 ## marge d'erreur pour diminuer un peu le shift (cela évite de rater  la première valeur propre).
err=20
Mur1x=[]
Mur1y=[]
Mur2x=[]
Mur2y=[]
Mur3x=[]
Mur3y=[]
while err>10**(1):
    Poly=maison2(torch.tensor([0,0]), theta, a, b,largeur_murs, murs[0], murs[1], murs[2], murs[3], murs[4], murs[5])
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps, tau=10**-18) ## tau est un paramètre à ajuster pour éviter l'erreur "input tensor is not posive definite..." lorsqu'elle apparaît.
    i=0
    while P[i][0]<10**-6:
        i=i+1
    V=torch.zeros_like(P[i+1][1])
    V=P[i+1][1]
    s=float(P[i][0])
    eps=1/3*s
    loss=Epoly2(Poly, V)
    optimizer.zero_grad()
    loss.backward()
    gmurs=murs.grad
    err=torch.norm(gmurs)
    print(loss)

    optimizer.step()
    Mur1x.append(murs[0].detach().item())
    Mur1y.append(murs[1].detach().item())
    Mur2x.append(murs[2].detach().item())
    Mur2y.append(murs[3].detach().item())
    Mur3x.append(murs[4].detach().item())
    Mur3y.append(murs[5].detach().item())
    #print(murs)
    #print(err)
##
plt.ion()
plt.show()

Polygone=Poly

Listx=[]
Listy=[]
for p in Polygone:
    Listx.append(p[0][0].detach().numpy())
    Listy.append(p[0][1].detach().numpy())
Listx.append(Polygone[0][0][0].detach().numpy())
Listy.append(Polygone[0][0][1].detach().numpy())
Listx=np.array(Listx)
Listy=np.array(Listy)

plt.figure(1)
plt.clf()
plt.plot(Listx, Listy)
plt.axis('scaled')

#plt.figure(1)
#plt.clf()
Im=P[i+1][1].reshape([M+2,N+2]).detach().tolist()
Im.reverse()
plt.imshow(np.array(Im), cmap='bwr', vmin=-np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), vmax=np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), extent=[-L/2,L/2,-l/2,l/2],aspect='auto')
plt.axis('scaled')
plt.colorbar()

plt.figure(2)
plt.clf()
Imsupp=Supp.detach().tolist()
Imsupp.reverse()
plt.imshow(np.array(Imsupp), extent=[-L/2,L/2,-l/2,l/2],aspect='auto')
plt.axis('scaled')


plt.figure(1)
#plt.clf()
#
# plt.plot([-2.0, 2.0],[-0.9, -0.9], color='red')
# plt.plot([-2.0, -2.0], [-0.9, 0.9], color='red')
# plt.plot([-2.0, 2.0],[0.9,0.9], color='red')
# plt.plot([2.0, 2.0], [-0.9,0.9], color='red')
#
# Mur1x=np.array(Mur1x)
# Mur2x=np.array(Mur2x)
# Mur3x=np.array(Mur3x)
# Mur1y=np.array(Mur1y)
# Mur2y=np.array(Mur2y)
# Mur3y=np.array(Mur3y)

plt.plot(Mur1x, Mur1y)
plt.plot(Mur2x, Mur2y)
plt.plot(Mur3x, Mur3y)

##
for mur1x, mur1y, mur2x, mur2y, mur3x, mur3y in zip(Mur1x, Mur1y, Mur2x, Mur2y, Mur3x, Mur3y):
    Poly=maison2(torch.tensor([0,0]), theta, a, b,largeur_murs, mur1x, mur1y, mur2x, mur2y, mur3x, mur3y)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    Polygone=Poly

    Listx=[]
    Listy=[]
    for p in Polygone:
        Listx.append(p[0][0].detach().numpy())
        Listy.append(p[0][1].detach().numpy())
    Listx.append(Polygone[0][0][0].detach().numpy())
    Listy.append(Polygone[0][0][1].detach().numpy())
    Listx=np.array(Listx)
    Listy=np.array(Listy)

    plt.figure(3)
    plt.clf()
    plt.plot(Listx, Listy)
    plt.axis('scaled')

    #plt.figure(1)
    #plt.clf()
    Im=P[i+1][1].reshape([M+2,N+2]).detach().tolist()
    Im.reverse()
    plt.imshow(np.array(Im), cmap='bwr', vmin=-np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), vmax=np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), extent=[-L/2,L/2,-l/2,l/2],aspect='auto')
    plt.axis('scaled')
    plt.colorbar()

    plt.pause(0.2)
##
plt.figure(65)
plt.clf()
plt.show()
P=vp_Laplacien(A,W,B, 15, s-0.1)
j=1
for p in P:
    plt.subplot(3, 5, j)
    Im=p[1].reshape([M+2,N+2]).detach().tolist()
    Im.reverse()
    plt.imshow(np.array(Im), cmap='bwr', vmin=-np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), vmax=np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), extent=[-L/2,L/2,-l/2,l/2],aspect='auto')
    plt.axis('scaled')
    j=j+1
    plt.colorbar()
    plt.plot(Listx, Listy)

##
G=len(Mur1x)
j=1
plt.figure(1)
plt.clf()
plt.show()

for mur1x, mur1y, mur2x, mur2y, mur3x, mur3y in zip([Mur1x[0], Mur1x[G//2], Mur1x[-1]], [Mur1y[0], Mur1y[G//2], Mur1y[-1]], [Mur2x[0], Mur2x[G//2], Mur2x[-1]], [Mur2y[0], Mur2y[G//2], Mur2y[-1]], [Mur3x[0], Mur3x[G//2], Mur3x[-1]], [Mur3y[0], Mur3y[G//2], Mur3y[-1]]):
    Poly=maison2(torch.tensor([0,0]), theta, a, b,largeur_murs, mur1x, mur1y, mur2x, mur2y, mur3x, mur3y)
    A,W,B, Supp, Horiz, Vert=AWB_polygone(Poly,N,M,L,l)
    P=vp_Laplacien(A,W,B,2,s-eps, tau=10**-15)
    i=0
    while P[i][0]<10**-6:
        i=i+1
    Polygone=Poly

    Listx=[]
    Listy=[]
    for p in Polygone:
        Listx.append(p[0][0].detach().numpy())
        Listy.append(p[0][1].detach().numpy())
    Listx.append(Polygone[0][0][0].detach().numpy())
    Listy.append(Polygone[0][0][1].detach().numpy())
    Listx=np.array(Listx)
    Listy=np.array(Listy)

    plt.subplot(1,3,j)
    Strings=['Initialisation', "Etat à mi-chemin de l'optimisation", "Domaine final"]
    plt.title(Strings[j-1],fontsize=16)
    plt.plot(Listx, Listy)
    plt.axis('scaled')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #plt.figure(1)
    #plt.clf()
    Im=P[i+1][1].reshape([M+2,N+2]).detach().tolist()
    Im.reverse()
    plt.imshow(np.array(Im), cmap='bwr', vmin=-np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), vmax=np.max([np.abs(np.max(P[i+1][1].detach().numpy())), np.abs(np.min(P[i+1][1].detach().numpy()))]), extent=[-L/2,L/2,-l/2,l/2],aspect='auto')
    plt.axis('scaled')
    cb=plt.colorbar(shrink=0.5)
    cb.ax.tick_params(labelsize=16)

    j=j+1