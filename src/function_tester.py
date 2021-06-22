import numpy as np

def rotate(u, v, w, plane, theta):
    x = [u, v, w]

    if plane == 1:
        ax = (0, 1)
        rmat = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]

    elif plane == 2:
        ax = (0, 2)
        rmat = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]

    elif plane == 3:
        ax = (1, 2)
        rmat = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]

    else:
        return u, v, w
        
    x = np.matmul(rmat, x)

    return x[0], x[1], x[2]

if __name__ == '__main__':
    u = np.array([2, 2.1, 1.9, 2.2, 2.])
    v = np.array([1.5, 1.6, 1.4, 1.7, 1.2])
    w = np.array([0., 0.1, -0.1, -0.2, 0.])

    u, v, w = rotate(u, v, w, 1, np.pi*1.5)
    print(u, v, w)