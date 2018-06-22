import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


np.random.seed(10)

input_dim = 2
num_classes = 4
X,Y = generate(320,num_classes,[[3.0,0].[3.0,3.0],[0,3.0]],True)
Y = Y%2

xr = []
xb = []
for (l,k) in zip(Y[:],X[:]):
    if l == 0:
        xr.append([k[0],k[1]])
    else :
        xb.append(k[0],k[1])
xr = np.array(xr)
xb = np.array(xb)
plt.scatter(xr[:,0],c='r',marker='+')
plt.scatter(xr[:,0],xb[:,1],c='b',marker='o')
plt.show()


