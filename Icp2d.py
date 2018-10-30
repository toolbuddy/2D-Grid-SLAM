import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

def Transform(X, R, T):
    Xt = np.transpose(np.matmul(R, np.transpose(X)))
    for i in range(Xt.shape[0]):
        Xt[i] += T
    return Xt

def Align(Xc, Pc):
    # Xc = R * Pc + T
    Pave = np.mean(Pc,0)
    Xave = np.mean(Xc,0)
    Pc = Pc - Pave
    Xc = Xc - Xave

    W = np.matmul(np.transpose(Xc), Pc)
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    R = np.matmul(u,vh)
    T = Xave - np.transpose(np.matmul(R, np.transpose(Pave)))
    return R, T

def Rejection(Xc,Pc, R, T):
    error = Xc - Pc
    error = np.sum((error * error),1)
    id_sort = np.argsort(error)
    size = Xc.shape[0]
    min_id = int(size*0.1)
    max_id = int(size*0.9)
    Xc = Xc[id_sort[min_id:max_id]]
    Pc = Pc[id_sort[min_id:max_id]]

    return Xc, Pc

def Icp(iter, X, P, Rtot=np.eye(2), Ttot=np.zeros((2))):
    # X = R * P + T
    if X.shape[0] < 20 or P.shape[0] < 20:
        return np.eye(2), np.zeros((2), dtype=float)
    
    pc_match = P.copy()
    tree = KDTree(X, leaf_size=2)

    for i in range(iter):
        Pc = Transform(pc_match, Rtot, Ttot)
        Xc = X[tree.query(Pc, k=1)[1]].reshape(Pc.shape)
        Xc, Pc = Rejection(Xc,Pc, Rtot, Ttot)
        R, T = Align(Xc, Pc)

        Rtot = np.matmul(R,Rtot)
        Ttot = T + np.matmul(R,Ttot)
    
    return Rtot, Ttot

if __name__ == '__main__':
    np.random.seed(0)
    #X = np.random.random((10, 2))
    X = np.array(
        [ [0,0],[0,1],[0,2],[0,3],[0,4],
        [1,0],[2,0],[3,0],[4,0],[5,0],
        #[0,5],[1,5],[2,5],[3,5],[4,5],
        #[5,1],[5,2],[5,3],[5,4],[5,5],
        ]
    )
    plt.plot(X[:,0], X[:,1], "o")
    #tree = KDTree(X, leaf_size=2)
    #out = tree.query(np.array([[0.0,0.0]]), k=2)
    #plt.plot(X[out[1][0],0], X[out[1][0],1], "r.")

    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    T = np.array([0.3,0.7])

    P = Transform(X, R, T)
    plt.plot(P[:,0], P[:,1], "ro")

    #RR, TT = Align(P, X)
    RR, TT = Icp(2, P, X)
    PP = Transform(X, RR, TT)
    plt.plot(PP[:,0], PP[:,1], "go")

    print(R,T)
    print(RR,TT)

    plt.axis('equal')
    plt.show()

