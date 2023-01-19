import numpy as np

def readllc(var):
    """
    Reading data in LLC90 grid
    NEED TO GENERALIZE THE CODE!!
    """
    dim = len(np.shape(var))
    if dim == 2:
        [ny, nx] = var.shape
        N = nx*3
        v1 = var[:N,:]
        v2 = var[N:N*2,:]
        v3 = var[N*2:(N*2+nx),:]
        v4 = var[(N*2+nx):(N*3+nx),:]
        v5 = var[(N*3+nx):(N*4+nx),:]
        v4 = np.flipud(np.reshape(v4.flatten(), [nx,N]).T)
        v5 = np.flipud(np.reshape(v5.flatten(), [nx,N]).T)
        llvar = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = v3.copy()
    if dim == 3:
        [nz, ny, nx] = var.shape
        N = nx*3
        llvar = np.zeros([nz, N, nx*4])
        for k in range(nz):
            v1 = var[k,:N,:]
            v2 = var[k,N:N*2,:]
            v3 = var[k,N*2:(N*2+nx),:]
            v4 = var[k,(N*2+nx):(N*3+nx),:]
            v5 = var[k,(N*3+nx):(N*4+nx),:]
            v4 = np.flipud(np.reshape(v4.flatten(), [nx,N]).T)
            v5 = np.flipud(np.reshape(v5.flatten(), [nx,N]).T)
            llvar[k,:,:] = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = var[:,(N*2+nx):(N*3+nx),:]

    return llvar, llcap

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
        llcap = v3.copy()
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


def readllc90_vector(u, v):
    """
    Reading data in LLC90 grid
    NEED TO GENERALIZE THE CODE!!
    """
    dim = len(np.shape(u))
    if dim == 2:
        [ny, nx] = u.shape
        u_v1 = u[:270,:]
        u_v2 = u[270:540,:]
        u_v3 = u[540:630,:]
        u_v4 = u[630:900,:]
        u_v5 = u[900:1170,:]

        v_v1 = v[:270,:]
        v_v2 = v[270:540,:]
        v_v3 = v[540:630,:]
        v_v4 = v[630:900,:]
        v_v5 = v[900:1170,:]

        u_v4 = np.flipud(np.reshape(u_v4.flatten(), [90,270]).T)
        u_v5 = np.flipud(np.reshape(u_v5.flatten(), [90,270]).T)
        v_v4 = np.flipud(np.reshape(v_v4.flatten(), [90,270]).T)
        v_v5 = np.flipud(np.reshape(v_v5.flatten(), [90,270]).T)

        u_v4_shift = u_v4*0
        u_v4_shift[1:, :] = u_v4[:-1,:]
        u_v5_shift = u_v5*0
        u_v5_shift[1:, :] = u_v5[:-1,:]
   
        ll_u = np.concatenate((u_v1,u_v2,v_v4,v_v5), axis=1)
        c_u = u_v3.copy()
        ll_v = np.concatenate((v_v1,v_v2,-u_v4_shift,-u_v5_shift), axis=1)
        c_v = v_v3.copy()
    if dim == 3:
        [nz, ny, nx] = u.shape
        ll_u = np.zeros([nz, 270, 360])
        ll_v = np.zeros([nz, 270, 360])
        for k in range(nz):
            u_v1 = u[k,:270,:]
            u_v2 = u[k,270:540,:]
            u_v3 = u[k,540:630,:]
            u_v4 = u[k,630:900,:]
            u_v5 = u[k,900:1170,:]
            u_v4 = np.flipud(np.reshape(u_v4.flatten(), [90,270]).T)
            u_v5 = np.flipud(np.reshape(u_v5.flatten(), [90,270]).T)

            v_v1 = v[k,:270,:]
            v_v2 = v[k,270:540,:]
            v_v3 = v[k,540:630,:]
            v_v4 = v[k,630:900,:]
            v_v5 = v[k,900:1170,:]
            v_v4 = np.flipud(np.reshape(v_v4.flatten(), [90,270]).T)
            v_v5 = np.flipud(np.reshape(v_v5.flatten(), [90,270]).T)

            u_v4_shift = u_v4*0
            u_v4_shift[1:, :] = u_v4[:-1,:]
            u_v5_shift = u_v5*0
            u_v5_shift[1:, :] = u_v5[:-1,:]

            ll_u[k,:,:] = np.concatenate((u_v1,u_v2,v_v4,v_v5), axis=1)
            ll_v[k,:,:] = np.concatenate((v_v1,v_v2,-u_v4_shift,-u_v5_shift), axis=1)
        c_u = u[:,540:630,:]
        c_v = v[:,540:630,:]

    return ll_u, ll_v, c_u, c_v

def readllc270(var):
    """
    Reading data in LLC90 grid
    NEED TO GENERALIZE THE CODE!!
    """
    dim = len(np.shape(var))
    if dim == 2:
        [ny, nx] = var.shape
        v1 = var[:810,:]
        v2 = var[810:1620,:]
        v4 = var[1890:2700,:]
        v5 = var[2700:3510,:]
        v4 = np.flipud(np.reshape(v4.flatten(), [270,810]).T)
        v5 = np.flipud(np.reshape(v5.flatten(), [270,810]).T)
        llvar = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = var[1620:1890,:]
    if dim == 3:
        [nz, ny, nx] = var.shape
        llvar = np.zeros([nz, 810, 1080])
        for k in range(nz):
            v1 = var[k,:810,:]
            v2 = var[k,810:1620,:]
            v4 = var[k,1890:2700,:]
            v5 = var[k,2700:3510,:]
            v4 = np.flipud(np.reshape(v4.flatten(), [270,810]).T)
            v5 = np.flipud(np.reshape(v5.flatten(), [270,810]).T)
            llvar[k,:,:] = np.concatenate((v1,v2,v4,v5), axis=1)
        llcap = var[:,1620:1890,:]

    return llvar, llcap
