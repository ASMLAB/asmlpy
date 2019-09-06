import numpy as np

def readllc90(var):
    """
    Reading data in LLC90 grid
    NEED TO GENERALIZE THE CODE!!
    """
    dim = len(np.shape(var))
    if dim == 2:
        [ny, nx] = var.shape
        v1 = var[:270,:]
        v2 = var[270:540,:]
        v3 = var[540:630,:]
        v4 = var[630:900,:]
        v5 = var[900:1170,:]
        v4 = np.flipud(np.reshape(v4.flatten(), [90,270]).T)
        v5 = np.flipud(np.reshape(v5.flatten(), [90,270]).T)
        llvar = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = v4.copy()
    if dim == 3:
        [nz, ny, nx] = var.shape
        llvar = np.zeros([nz, 270, 360])
        for k in range(nz):
            v1 = var[k,:270,:]
            v2 = var[k,270:540,:]
            v3 = var[k,540:630,:]
            v4 = var[k,630:900,:]
            v5 = var[k,900:1170,:]
            v4 = np.flipud(np.reshape(v4.flatten(), [90,270]).T)
            v5 = np.flipud(np.reshape(v5.flatten(), [90,270]).T)
            llvar[k,:,:] = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = var[:,540:630,:]

    return llvar, llcap
