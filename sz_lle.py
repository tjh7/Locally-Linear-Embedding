# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:36:21 2024

@author: Lenovo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 注意：这两行代码必须在文件的最开头，在加载各种包之前

import numpy as np
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import  eigs
import pandas as pd
import scipy.io
import os
from scipy.io import loadmat

def lle(X, k, d, sparse = True):
 
    D, N = X.shape
    print('LLE running on {} points in {} dimensions\n'.format(N,D))
 
    # Step1: compute pairwise distances & find neighbors
    print('-->Finding {} nearest neighbours.\n'.format(k))
 
    X2 = np.sum(X**2,axis = 0).reshape(1,-1) # 1xN
    distance = np.tile(X2,(N,1)) + np.tile(X2.T, (1, N)) - 2 * np.dot(X.T,X) # NxN
 
    index = np.argsort(distance,axis=0)
    neighborhood = index[1:1+k,:] # kxN filter itself
 
    # Step2: solve for reconstruction weights
    print('-->Solving for reconstruction weights.\n')
 
    if (k>D):
        print(' [note: k>D; regularization will be used]\n')
        tol = 1e-3 # regularlizer in case constrained fits are ill conditioned
    else:
        tol = 0
 
    w = np.zeros((k,N))
    for ii in range(N):
        xn = X[:,neighborhood[:,ii]] - np.tile(X[:,ii].reshape(-1,1),(1,k)) # shift ith pt to origin
        S = np.dot(xn.T, xn)                                 # local covariance,xn = Xi-Ni
        S = S + np.eye(k,k) * tol * np.trace(S)              # regularlization (k>D)
        Sinv = np.linalg.inv(S)                              # inv(S)
        w[:,ii] = np.dot(Sinv,np.ones((k,1))).reshape(-1,)   # solve Cw=1
        w[:,ii] = w[:,ii]/sum(w[:,ii])                       # enforce sum(wi)=1
 
    # Step 3: compute embedding from eigenvectors of cost matrix M = (I-W)'(I-W)
    print('-->Computing embedding to get eigenvectors .\n')
    
    if sparse: #  parse solution   
        M = lil_matrix(np.eye(N, N)) # use a sparse matrix lil_matrix((N,N))       
        for ii in range(N):
            wi = w[:,ii].reshape(-1,1) # kx1, i point neighborhood (wji)
            jj = neighborhood[:,ii].tolist() # k,
            M[ii,jj] = M[ii,jj] - wi.T
            M[jj,ii] = M[jj,ii] - wi
            M_temp = M[jj,:][:,jj].toarray() + np.dot(wi,wi.T)
            for ir, row in enumerate(jj): ### TO DO
                for ic, col in enumerate(jj):
                    M[row,col] = M_temp[ir,ic]
    else:  # dense solution           
        M = np.eye(N, N) # use a dense eye matrix         
        for ii in range(N):
            wi = w[:,ii].reshape(-1,1) # kx1
            jj = neighborhood[:,ii].tolist() # k,
            M[ii,jj] = M[ii,jj] - wi.reshape(-1,)
            M[jj,ii] = M[jj,ii] - wi.reshape(-1,)
            M_temp = M[jj,:][:,jj] + np.dot(wi,wi.T) # kxk
            for ir, row in enumerate(jj): ### TO DO
                for ic, col in enumerate(jj):
                    M[row,col] = M_temp[ir,ic]      
        M = lil_matrix(M)                
    # Calculation of embedding
    # note: eigenvalue(M) >=0
    eigenvals,y = eigs(M, k = d+1, sigma = 0 ) # Y-> Nx(d+1)
    y = np.real(y)[:,:d+1]  # get d+1 eigenvalue -> eigenvectors
    y = y[:,1:d+1].T * np.sqrt(N) # bottom evect is [1,1,1,1...] with eval 0
    print('Done.\n')
    
    # other possible regularizers for k>D
    #   S = S + tol*np.diag(np.diag(S))           # regularlization
    #   S = S + np.eye(k,k)*tol*np.trace(S)*k     # regularlization
    return y





        
file1 = r'D:\永兴岛\lle\no add\深圳lle\RH_1000.mat'
file2 = r'D:\永兴岛\lle\no add\深圳lle\RH_850.mat'
file3 = r'D:\永兴岛\lle\no add\深圳lle\RH_500.mat'
file4 = r'D:\永兴岛\lle\no add\深圳lle\\U_1000.mat'
file5 = r'D:\永兴岛\lle\no add\深圳lle\\U_850.mat'
file6 = r'D:\永兴岛\lle\no add\深圳lle\\U_500.mat'
file7 = r'D:\永兴岛\lle\no add\深圳lle\\V_1000.mat'
file8 = r'D:\永兴岛\lle\no add\深圳lle\\V_850.mat'
file9 = r'D:\永兴岛\lle\no add\深圳lle\V_500.mat'
file10 = r'D:\永兴岛\lle\no add\深圳lle\T_1000.mat'
file11 = r'D:\永兴岛\lle\no add\深圳lle\T_850.mat'
file12 = r'D:\永兴岛\lle\no add\深圳lle\T_500.mat'
file13 = r'D:\永兴岛\lle\no add\深圳lle\P_S.mat'
file14 = r'D:\永兴岛\lle\no add\深圳lle\LI.mat'



# mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
data1 = loadmat(file1, mat_dtype=True)
data2 = loadmat(file2, mat_dtype=True)
data3 = loadmat(file3, mat_dtype=True)
data4 = loadmat(file4, mat_dtype=True)
data5 = loadmat(file5, mat_dtype=True)
data6 = loadmat(file6, mat_dtype=True)
data7 = loadmat(file7, mat_dtype=True)
data8 = loadmat(file8, mat_dtype=True)
data9 = loadmat(file9, mat_dtype=True)
data10 = loadmat(file10, mat_dtype=True)
data11 = loadmat(file11, mat_dtype=True)
data12 = loadmat(file12, mat_dtype=True)
data13 = loadmat(file13, mat_dtype=True)
data14 = loadmat(file14, mat_dtype=True)


# 导入后的data是一个字典，取出想要的变量字段即可。

RH_1000 = data1['RH_1000']
RH_850 = data2['RH_850']
RH_500 = data3['RH_500']
U_1000 = data4['U_1000']
U_850 = data5['U_850']
U_500 = data6['U_500']
V_1000 = data7['V_1000']
V_850 = data8['V_850']
V_500 = data9['V_500']
T_1000 = data10['T_1000']
T_850 = data11['T_850']
T_500 = data12['T_500']
P_S = data13['P_S']
LI = data14['LI']
##gk = np.float32(gk)

# 设置参数
X = np.vstack((RH_1000,RH_850,RH_500,U_1000,U_850,U_500,V_1000,V_850,V_500,T_1000,T_850,T_500,P_S,LI))


k = 14
d = 150
fin_sz_174 = lle(X, k, d, sparse=True)                                          

save_path = r'C:\Users\Lenovo\Desktop\添加深圳'
os.makedirs(save_path, exist_ok=True)  # 确保目录存在
mat_file_path = os.path.join(save_path, 'fin_sz_174.mat')

# 保存结果为 .mat 文件
scipy.io.savemat(mat_file_path, {'results': fin_sz_174})