import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import blackman, hamming

def rotaryspectra(dt, u, v, power=True, window=None):
    """
    + objective : compute the rotary spectra using wind or flow fields
    + inputs
      - dt      : time difference (delta t) of u or v
      - u       : wind or current in zonal direction.   [m/s]
                  It can be either 1d, 2d or 3d variable
      - v       : wind or current in meridional direction.  [m/s]
                  It can be either 1d, 2d or 3d variable
      - power   : If it is True, the output represents the power
                  'XC','YC','RAC','DXC','DYC','hFacC','hFacW','hFacS',
                  'Depth','RC','RF','DRC','DRF'
      - window  : the window that is applied to the data
                  For now, 'hamming' and 'blackman' are applied here.
    + output
      - xf      : frequency [cycle per the unit of dt]
      - var     : power density function [ m2/s2 per delta(xf) ]
    """
    from scipy.signal.windows import blackman, hamming

    flow = u + v*1j
    nt = flow.shape[0]
    ndim = len(flow.shape)
    #
    #  Get frequency
    #
    xf = fftfreq(nt, dt)        # frequency
    xf = fftshift(xf)           # shift frequency domain
    #
    #  FFT
    #
    if window=='hamming':
        w = hamming(nt)
    elif window=='blackman':
        w = blackman(nt)
    else:
        w = np.ones(nt)

    if ndim == 1:
        yf = fft(flow*w)
        yf = fftshift(yf)
        var = np.abs(yf)
    else:
        if ndim == 2:
            _, nx = flow.shape
            var = np.zeros([nt, nx])
        elif ndim == 3:
            _, ny, nx = flow.shape
            var = np.zeros([nt, ny*nx])
            flow = flow.reshape(nt, ny*nx)

        for jj in range(flow.shape[-1]):
            tmpon = flow[:, jj]
            yf = fft(tmpon*w)
            yf = fftshift(yf)
            var[:, jj] = np.abs(yf)
    
        if ndim == 3:
            var = var.reshape(nt, ny, nx)

    if power is True:
        var = var**2

    return xf, var
