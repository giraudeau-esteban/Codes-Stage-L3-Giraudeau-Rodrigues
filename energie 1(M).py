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
N=100
M=100
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

VP=[]
for i in range(1,N+1):
    for j in range(1,M+1):
        vp=(-2*(1/h**2*np.cos(i*h*np.pi/L)+1/k**2*np.cos(j*k*np.pi/l)-1/h**2-1/k**2),i,j)
        VP.append(vp)
VP=sorted(VP,key=fst)

mu1=VP[0][0]#P[0][0]#np.pi**2*(1/(L**2)+1/(l**2)) #12.253375481614466 #
mu2=VP[1][0]#P[1][0] #np.pi**2*(4/(L**2)+1/(l**2))#19.406861649004966 #
mu3=VP[2][0]#P[2][0]#np.pi**2*(9/(L**2)+1/(l**2))#30.68362598160174 #
mu4=VP[3][0]#P[3][0]#np.pi**2*(1/(L**2)+4/(l**2)) #40.86732015117716 #

u1=[]
for m in range(M+2):
    u1=u1+[np.sin(n*VP[0][1]*np.pi*h/L)*np.sin(m*VP[0][2]*np.pi*k/l) for n in range(N+2)]
u1=np.array(u1)

u2=[]
for m in range(M+2):
    u2=u2+[np.sin(n*VP[1][1]*np.pi*h/L)*np.sin(m*VP[1][2]*np.pi*k/l) for n in range(N+2)]
u2=np.array(u2)

u3=[]
for m in range(M+2):
    u3=u3+[np.sin(n*VP[2][1]*np.pi*h/L)*np.sin(m*VP[2][2]*np.pi*k/l) for n in range(N+2)]
u3=np.array(u3)

u4=[]
for m in range(M+2):
    u4=u4+[np.sin(n*VP[3][1]*np.pi*h/L)*np.sin(m*VP[3][2]*np.pi*k/l) for n in range(N+2)]
u4=np.array(u4)




R=np.linspace(0.72,3,201)
E1=[]
E2=[]
for r in R:
    print(r)
    Supp[M//2-5:M//2+5,N//2-5:N//2+5]=r
    #Supp[M//2,N//2]=r
    A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
    P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
    i=0
    while P[i][0]<10**-7:
        i=i+1
    S1=(P[i][0]-mu1)**2+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
    print(P[i][0])
    E1.append(S1)

    v=[]
    for i in range(4):
        if P[i][1][N+3]>0:
            v.append(P[i][1])
        else:
            v.append(-P[i][1])
    v=np.array(v)

    S2=norm(v[0]-u1)**2+norm(v[1]-u2)**2+norm(v[2]-u3)**2+norm(v[3]-u4)**2
    E2.append(S2)

plt.figure(1)
plt.clf()
plt.title("énergie sur les valeurs propres : $E(A)=\sum_k|\\lambda_k(L(A))-\\mu_k|^2$")
E1=np.array(E1)
plt.plot(R,E1)

plt.figure(2)
plt.clf()
plt.title("énergie sur les vecteurs propres : $E(A)=\sum_k||v_k(L(A))-u_k||^2$")
E2=np.array(E2)
plt.plot(R,E2)

## #### Energie 2---> tests pour déterminer un alpha optimal #################
plt.clf()
Supp=np.zeros([M+2,N+2])
#Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
#Supp=np.zeros([M+2,N+2])

VP=[]
for i in range(1,N+1):
    for j in range(1,M+1):
        vp=(-2*(1/h**2*np.cos(i*h*3*np.pi/(2*L))+1/k**2*np.cos(j*k*np.pi/l)-1/h**2-1/k**2),i,j)
        VP.append(vp)
VP=sorted(VP,key=fst)

mu1=VP[0][0]#P[0][0]#np.pi**2*(1/(L**2)+1/(l**2)) #12.253375481614466 #
mu2=VP[1][0]#P[1][0] #np.pi**2*(4/(L**2)+1/(l**2))#19.406861649004966 #
mu3=VP[2][0]#P[2][0]#np.pi**2*(9/(L**2)+1/(l**2))#30.68362598160174 #
mu4=VP[3][0]#P[3][0]#np.pi**2*(1/(L**2)+4/(l**2)) #40.86732015117716 #




Long=np.linspace(1.5,2.5,201)
E1=[]
Alpha=[]
for long in Long:
    #print(long)
    Nh=int(long/h)-1
    alpha=long/h-int(long/h)
    # theta= np.pi*Nh*h/(long*(Nh+1))
    # alpha=np.sqrt(np.abs(2*(np.cos(theta)-1)*np.sin((Nh+1)*theta)/(np.sin(Nh*theta)-2*np.sin((Nh+1)*theta))))
    # Alpha.append(alpha)
    #Supp[1:-1,1:Nh+1]=1
    #Supp[1:-1,Nh+1]=1/alpha
    #plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
    # plt.pause(0.005)

    # L0=(Nh+1)*h
    # theta= np.pi*L0/(long*(Nh+1))
    # R=1-2*np.cos(theta)+np.sin(Nh*theta)/np.sin((Nh+1)*theta)

    Horiz=1/h**2*np.ones([(M+2),(N+1)])
    Horiz[:,Nh+1]=1/h**2*1/alpha
    #Horiz[:,Nh+2:-1]=10**20
    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=1/k**2*np.ones([M+1,N+2])
    #Vert[:,Nh+2:-1]=10**20
    Vert=Vert.reshape([(M+1)*(N+2)])

    R=np.concatenate((Horiz,Vert))
    W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))


    Supp[1:-1,1:Nh+2]=1
    A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    P=vp_Laplacien(A,W,K,3,mu1) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
    i=0
    while P[i][0]<10**-7:
        i=i+1
    S1=(P[i][0]-mu1)**2#+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
    print(P[i][0])
    E1.append(S1)
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

## Energie 3 (rectangle penché)
theta=np.pi/6

ellipse=lambda x,y : x**2+y**2-1

rot=lambda x,y:np.array([np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y])

rectangle=lambda x,y : np.max([np.abs(x),np.abs(y)])-1

ModPropre=lambda x, y : np.cos(np.pi*x/2)*np.cos(np.pi*y)*(x <=1)*(x>=-1)*(y<=0.5)*(y>=-0.5)

ValPropre=lambda L,l : np.pi**2*(1/(L**2)+1/(l**2))

plt.clf()
Supp=np.zeros([M+2,N+2])
#Supp[1:-1,1:-1]=np.ones([M,N]) # rectangle
A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
#P=vp_Laplacien(A,W,K,4,30) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
#Supp=np.zeros([M+2,N+2])

# Supp_cible=np.zeros([M+2,N+2])
# f=ellipse
# for i in range(M+2):
#     for j in range(N+2):
#         y=k*(i-1/2-M/2.0)
#         x=h*(j-1/2-N/2.0)
#         if f(rot(x,y)[0]/2,rot(x,y)[1]/2)<0:
#             Supp_cible[i,j]=1
#
# A_cible=bsr_matrix((Supp_cible.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))
# u1=np.abs(vp_Laplacien(A_cible,W,K,1,1.4)[0][1])

I=np.arange(M+2)
J=np.arange(N+2)
[J,I]=np.meshgrid(J,I)

Y=k*(I-1/2-M/2.0)
X=h*(J-1/2-N/2.0)

u1=ModPropre(rot(X,Y)[0], rot(X,Y)[1])
plt.figure(1)
plt.clf()
plt.imshow(u1)

u1=u1/norm(u1)
u1=u1.reshape((M+2)*(N+2))


#mu1=1.433010753097955 # cercle
mu1=12.067048661946352

# Supp2=np.zeros([M+2,N+2])
# Supp2[1:-1,1:-1]=np.ones([M, N])
# A=bsr_matrix((Supp2.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))



Long=np.linspace(1.5,2.4,201)
E1=[]
E2=[]
Alpha=[]
for long in Long:
    # print(long)
    # Nh=int(long/h)-1
    # alpha=long/h-int(long/h)

    a=long/2
    b=1/2
    f=lambda x,y:rectangle(rot(x,y)[0]/a,rot(x,y)[1]/b)
    for i in range(M+2):
        for j in range(N+2):
            y=k*(i-1/2-M/2.0)
            x=h*(j-1/2-N/2.0)
            if f(x,y)<=0:
                Supp[i,j]=1

    # f=lambda x,y:ellipse(x/long,y/long)
    # for i in range(M+2):
    #     for j in range(N+2):
    #         y=k*(i-1/2-M/2.0)
    #         x=h*(j-1/2-N/2.0)
    #         if f(x,y)<0:
    #             Supp[i,j]=1
    A=bsr_matrix((Supp.reshape((M+2)*(N+2)),(np.arange(0,(M+2)*(N+2)),np.arange(0,(M+2)*(N+2)))))

    #plt.imshow(Supp, extent=[0,L,0,l],aspect='auto')
    # plt.axis('scaled')
    # plt.pause(0.005)

    Horiz=1/h**2*np.ones([(M+2),(N+1)])
    for i in range(M+2):
        for j in range(N+1):
            if Supp[i,j]==0 and Supp[i,j+1]==1:
                g=lambda alpha: f(h*(j+1/2-N/2.0-alpha),k*(i-1/2-M/2.0))
                alpha=dichotomie(g,0,1)
                Horiz[i,j]=1/h**2*1/alpha
            if Supp[i,j]==1 and Supp[i,j+1]==0:
                g=lambda alpha: f(h*(j-1/2-N/2.0+alpha),k*(i-1/2-M/2.0))
                alpha=dichotomie(g,0,1)
                Horiz[i,j]=1/h**2*1/alpha
            # if Supp[i,j]==0 and Supp[i,j+1]==0:
            #     Horiz[i,j]=10**6
    Horiz=Horiz.reshape([(M+2)*(N+1)])

    Vert=1/k**2*np.ones([(M+1),(N+2)])
    for j in range(N+2):
        for i in range(M+1):
            if Supp[i,j]==0 and Supp[i+1,j]==1:
                g=lambda alpha: f(h*(j-1/2-N/2.0),k*(i+1/2-M/2.0-alpha))
                alpha=dichotomie(g,0,1)
                Vert[i,j]=1/k**2*1/alpha
            if Supp[i,j]==1 and Supp[i+1,j]==0:
                g=lambda alpha: f(h*(j-1/2-N/2.0),k*(i-1/2-M/2.0+alpha))
                alpha=dichotomie(g,0,1)
                Vert[i,j]=1/k**2*1/alpha
            # if Supp[i,j]==0 and Supp[i+1,j]==0:
            #     Vert[i,j]=10**6
    Vert=Vert.reshape([(M+1)*(N+2)])

    R=np.concatenate((Horiz,Vert))
    W=bsr_matrix((R,(np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)),np.arange(0,(M+2)*(N+1)+(N+2)*(M+1)))))


    P=vp_Laplacien(A,W,K,3,mu1-0.1) # On pourrait en chercher moins, mais avec ça on est sûrs d'en avoir assez
    i=0
    while P[i][0]<10**-6:
        i=i+1
    S1=P[i][0]#(P[i][0]-mu1)**2#+(P[i+1][0]-mu2)**2+(P[i+2][0]-mu3)**2+(P[i+3][0]-mu4)**2
    print(P[i][0])
    E1.append(S1)

    S2=norm(np.abs(P[i][1])-u1)**2
    E2.append(S2)
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
##

E1=np.array(E1)
#min1=np.min(E1)
#argmin1=Long[np.argmin(E1)]
##

plt.figure(1)
plt.clf()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Longueur $l_0$ du rectangle", fontsize=16)
plt.ylabel("$\\lambda_1$", fontsize=16)
plt.title("$1^{ère}$ valeur propre en fonction de la longueur du rectangle", fontsize=18)
plt.plot(Long,E1, label='Résultats du modèle')
plt.plot(Long,ValPropre(1,Long), label='courbe théorique')
plt.legend(fontsize=16)
#plt.plot(argmin1, min1, 'o')
#plt.text(argmin1-0.1, min1+0.1, f'({round(argmin1, 3)},{round(min1, 3)})', fontsize=12, color='orange')

##
E2=np.array(E2)
min2=np.min(E2)
argmin2=Long[np.argmin(E2)]
##
plt.figure(2)
plt.clf()
plt.xlabel("Rayon $r$ du cercle", fontsize=14)
plt.ylabel("Energie $E_2$", fontsize=14)
plt.title("énergie sur les vecteurs propres : $E_2(r)=||v_1(\mathcal{L}(r))-u_1||^2$", fontsize=16)
plt.plot(Long, E2)
plt.plot(argmin2, min2, 'o')
plt.text(argmin2-0.08, min2+0.008, f'({round(argmin2, 3)},{round(min2, 3)})', fontsize=12, color='orange')
##
L=10
N=300
h=L/(N+1)
plt.clf()
alpha=np.linspace(0.1,1,201)
L0=L+alpha*h
theta= np.pi*2*L/(L0*(N+1))
R=1-2*np.cos(theta)+np.sin(N*theta)/np.sin((N+1)*theta)
#alpha=np.sqrt(-2*(np.cos(theta)-1)*np.sin((N+1)*theta)/(np.sin(N*theta)-2*np.sin((N+1)*theta)))

plt.plot(alpha,R)
plt.plot(alpha,1/alpha)


##
L=3
N=200
h=L/(N+1)


f=lambda r: np.sin(N*(np.pi*4*L/((L+r*h)*(N+1))))-2*np.sin(np.pi*4*L/(L+r*h))

r=np.linspace(0,2,201)
plt.plot(r,f(r))

# a=0
# b=2
# while err > 10**-7:
