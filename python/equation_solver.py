import numpy as np

def main():
    a = np.mat('2 9 5;7 -3 -1;1 1 14')
    b = np.mat('100;142;203')

    X = np.concatenate((b,a[:,1:3]),axis=1)
    Y = np.concatenate((a[:,0],b,a[:,2]),axis=1)
    Z = np.concatenate((a[:,0:2],b),axis=1)

    D = np.linalg.det(a)
    Dx = np.linalg.det(X)
    Dy = np.linalg.det(Y)
    Dz = np.linalg.det(Z)

    print(("x y z : %f %f %f")%(Dx/D,Dy/D,Dz/D))

main()
