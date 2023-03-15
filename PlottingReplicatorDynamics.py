import numpy as np
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.egt import utils
from open_spiel.python.egt import dynamics


n = 15
A = np.array([[3,0],[5,1]])
B = np.array([[3,5],[0,1]])

X,Y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))

xdot = np.zeros_like(X)
ydot = np.zeros_like(Y)

for i in range(0,np.shape(xdot)[0]):
    for j in range(0,np.shape(xdot)[1]):
        x = np.array([X[i,j],1-X[i,j]])
        y = np.array([Y[i,j],1-Y[i,j]])
        xdot[i,j]   = X[i,j] * ((A[0,:] @ y) - x.T @ A @ y)
        ydot[i,j]   = Y[i,j] * ((x.T @ B[:,0]) - x.T @ B @ y)



plt.quiver(X,Y,xdot,ydot)
plt.show()


