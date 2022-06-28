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

L=6
l=6
N=26
M=26
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

plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
plt.axis('scaled')

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


## #### Energie #################
#A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez

# VP=[]
# for i in range(1,N+1):
#     for j in range(1,M+1):
#         vp=(-2*(1/h**2*np.cos(i*h*np.pi/L)+1/k**2*np.cos(j*k*np.pi/l)-1/h**2-1/k**2),i,j)
#         VP.append(vp)
# VP=sorted(VP,key=fst)
#
# mu1=VP[0][0]#P[0][0]#np.pi**2*(1/(L**2)+1/(l**2)) #12.253375481614466 #
# mu2=VP[1][0]#P[1][0] #np.pi**2*(4/(L**2)+1/(l**2))#19.406861649004966 #
# mu3=VP[2][0]#P[2][0]#np.pi**2*(9/(L**2)+1/(l**2))#30.68362598160174 #
# mu4=VP[3][0]#P[3][0]#np.pi**2*(1/(L**2)+4/(l**2)) #40.86732015117716 #
#
# u1=[]
# for m in range(M+2):
#     u1=u1+[np.sin(n*VP[0][1]*np.pi*h/L)*np.sin(m*VP[0][2]*np.pi*k/l) for n in range(N+2)]
# u1=np.array(u1)
#
# u2=[]
# for m in range(M+2):
#     u2=u2+[np.sin(n*VP[1][1]*np.pi*h/L)*np.sin(m*VP[1][2]*np.pi*k/l) for n in range(N+2)]
# u2=np.array(u2)
#
# u3=[]
# for m in range(M+2):
#     u3=u3+[np.sin(n*VP[2][1]*np.pi*h/L)*np.sin(m*VP[2][2]*np.pi*k/l) for n in range(N+2)]
# u3=np.array(u3)
#
# u4=[]
# for m in range(M+2):
#     u4=u4+[np.sin(n*VP[3][1]*np.pi*h/L)*np.sin(m*VP[3][2]*np.pi*k/l) for n in range(N+2)]
# u4=np.array(u4)
#
#
#
#
# R=np.linspace(0.72,3,201)
# E1=[]
# E2=[]
# for r in R:
#     print(r)
#     Supp[M//2-5:M//2+5,N//2-5:N//2+5]=r
#     #Supp[M//2,N//2]=r
#     A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#     P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
#     i=0
#     while P[i][0]<10**-7:
#         i=i+1
#     S1=(P[i][0]-mu1)**2+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
#     print(P[i][0])
#     E1.append(S1)
#
#     v=[]
#     for i in range(4):
#         if P[i][1][N+3]>0:
#             v.append(P[i][1])
#         else:
#             v.append(-P[i][1])
#     v=np.array(v)
#
#     S2=norm(v[0]-u1)**2+norm(v[1]-u2)**2+norm(v[2]-u3)**2+norm(v[3]-u4)**2
#     E2.append(S2)
#
# plt.figure(1)
# plt.clf()
# plt.title("énergie sur les valeurs propres : $E(A)=\sum_k|\\lambda_k(L(A))-\\mu_k|^2$")
# E1=np.array(E1)
# plt.plot(R,E1)
#
# plt.figure(2)
# plt.clf()
# plt.title("énergie sur les vecteurs propres : $E(A)=\sum_k||v_k(L(A))-u_k||^2$")
# E2=np.array(E2)
# plt.plot(R,E2)

## #### Energie 2---> tests pour déterminer un alpha optimal #################
# plt.clf()
# Supp=np.zeros([M+2,N+2])
# #Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
# A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
# #P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
# #Supp=np.zeros([M+2,N+2])
#
# VP=[]
# for i in range(1,N+1):
#     for j in range(1,M+1):
#         vp=(-2*(1/h**2*np.cos(i*h*3*np.pi/(2*L))+1/k**2*np.cos(j*k*np.pi/l)-1/h**2-1/k**2),i,j)
#         VP.append(vp)
# VP=sorted(VP,key=fst)
#
# mu1=VP[0][0]#P[0][0]#np.pi**2*(1/(L**2)+1/(l**2)) #12.253375481614466 #
# mu2=VP[1][0]#P[1][0] #np.pi**2*(4/(L**2)+1/(l**2))#19.406861649004966 #
# mu3=VP[2][0]#P[2][0]#np.pi**2*(9/(L**2)+1/(l**2))#30.68362598160174 #
# mu4=VP[3][0]#P[3][0]#np.pi**2*(1/(L**2)+4/(l**2)) #40.86732015117716 #
#
#
#
#
# Long=np.linspace(1.5,2.5,201)
# E1=[]
# Alpha=[]
# for long in Long:
#     #print(long)
#     Nh=int(long/h)-1
#     alpha=long/h-int(long/h)
#     # theta= np.pi*Nh*h/(long*(Nh+1))
#     # alpha=np.sqrt(np.abs(2*(np.cos(theta)-1)*np.sin((Nh+1)*theta)/(np.sin(Nh*theta)-2*np.sin((Nh+1)*theta))))
#     # Alpha.append(alpha)
#     #Supp[1:-1,1:Nh+1]=1
#     #Supp[1:-1,Nh+1]=1/alpha
#     #plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
#     # plt.pause(0.005)
#
#     # L0=(Nh+1)*h
#     # theta= np.pi*L0/(long*(Nh+1))
#     # R=1-2*np.cos(theta)+np.sin(Nh*theta)/np.sin((Nh+1)*theta)
#
#     Horiz=1/h**2*np.ones([(M+2),(N+1)])
#     Horiz[:,Nh+1]=1/h**2*1/alpha
#     #Horiz[:,Nh+2:-1]=10**20
#     Horiz=Horiz.reshape([(M+2)*(N+1)])
#
#     Vert=1/k**2*np.ones([M+1,N+2])
#     #Vert[:,Nh+2:-1]=10**20
#     Vert=Vert.reshape([(M+1)*(N+2)])
#
#     R=np.concatenate((Horiz,Vert))
#     W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))
#
#
#     Supp[1:-1,1:Nh+2]=1
#     A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#
#     P=vp_Laplacien(A,W,K,3,mu1) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
#     i=0
#     while P[i][0]<10**-7:
#         i=i+1
#     S1=(P[i][0]-mu1)**2#+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
#     print(P[i][0])
#     E1.append(S1)
#     #
#     # # p=P[i][1]
#     # # plt.clf()
#     # # vp=np.zeros([M+2,N+2])
#     # # for j in range(0,M+2):
#     # #     vp[j,:]=p[(N+2)*j:(N+2)*(j+1)]
#     # # plt.title(f'$\\lambda={p[0]}, \\omega={np.sqrt(p[0])}$')
#     # # plt.imshow(vp, extent=[0,L,0,l],aspect='auto')
#     # #
#     # # plt.axis('scaled')
#     # # plt.pause(0.05)
#
# # Alpha=np.array(Alpha)
# # plt.plot(Long,Alpha)
#
#
# plt.figure(1)
# plt.clf()
# plt.title("énergie sur les valeurs propres : $E(A)=\sum_k|\\lambda_k(L(A))-\\mu_k|^2$")
# E1=np.array(E1)
# plt.plot(Long,E1)
# print(Long[np.argmin(E1)])

## Energie 3 (rectangle penché)
theta=np.pi/6

ellipse=lambda x,y : x**2+y**2-1

rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1

plt.clf()
Supp=np.zeros([M+2,N+2])
#Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
#Supp=np.zeros([M+2,N+2])

Supp_cible=np.zeros([M+2,N+2])
f=ellipse
for i in range(M+2):
    for j in range(N+2):
        y=k*(i-1/2-M/2.0)
        x=h*(j-1/2-N/2.0)
        if f(rot(x,y)[0]/2,rot(x,y)[1]/2)<0:
            Supp_cible[i,j]=1

A_cible=bsr_matrix((Supp_cible.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
u1=np.abs(vp_Laplacien(A_cible,W,K,1,1.4)[0][1])


mu1=np.pi**2*(1/1 + 1/4)

# Supp2=np.zeros([M+2,N+2])
# Supp2[1:-1,1:-1]=np.ones([M, N])
# A=bsr_matrix((Supp2.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))


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

                x = h*(j-(N+1)//2)
                y = f(x)
                yi = y/k+(M+1)//2
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

                y = k*(i-(M+1)//2)
                x = finv(y)
                xj = x/h+(N+1)//2
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


Long=np.linspace(1.5,2.4,201)
E1=[]
E2=[]
Alpha=[]
for long in Long:
    # print(long)
    # Nh=int(long/h)-1
    # alpha=long/h-int(long/h)

    # a=long/2
    # b=1/2
    # f=lambda x,y:rectangle(rot(x,y)[0]/a,rot(x,y)[1]/b)
    # for i in range(M+2):
    #     for j in range(N+2):
    #         y=k*(i-1/2-M/2.0)
    #         x=h*(j-1/2-N/2.0)
    #         if f(x,y)<=0:
    #             Supp[i,j]=1

    L=3
    l=3
    N=59
    M=59
    h=L/(N+1)
    k=l/(M+1)


    Rec = rectangle(np.array([0, 0]), np.pi/6, 1, long)

    A,W,B, S, H, V=AWB_polygone(Rec, N, M, L, l)


    P=vp_Laplacien(A,W,B,3,12) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
    i=0
    while P[i][0]<10**-7:
        i=i+1
    S1=(P[i][0]-mu1)**2#+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
    print(P[i][0])
    E1.append(S1)

    # S2=norm(np.abs(P[i][1])-u1)**2
    # E2.append(S2)
    #
    # # p=P[i][1]
    # # plt.clf()
    # # vp=np.zeros([M+2,N+2])
    # # for j in range(0,M+2):
    # #     vp[j,:]=p[(N+2)*j:(N+2)*(j+1)]
    # # plt.title(f'$\\lambda={p[0]}, \\omega={np.sqrt(p[0])}$')
    # # plt.imshow(vp, extent=[0,L,0,l],aspect='auto')
    # #
    # # plt.axis('scaled')
    # # plt.pause(0.05)

# Alpha=np.array(Alpha)
# plt.plot(Long,Alpha)

plt.figure(1)
plt.clf()
plt.title("énergie sur les valeurs propres : $E(A)=\sum_k|\\lambda_k(L(A))-\\mu_k|^2$")
E1=np.array(E1)
plt.plot(Long,E1)
print(Long[np.argmin(E1)])

plt.figure(2)
plt.clf()
plt.title("énergie sur les vecteurs propres : $E(A)=\sum_k||v_k(L(A))-u_k||^2$")
E2=np.array(E2)
plt.plot(Long, E2)
##
