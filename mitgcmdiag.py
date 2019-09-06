import numpy as np
from MITgcmutils import rdmds, densjmd95

def loadgrid(dirGrid, region=None, varname=None, flag=1):
    """
    + objective : read the grid information for the MITgcm simulation
    + inputs
      - dirGrid : the location of the grid files
      - region  : if specified, read the grid information of local area
                  This is passed to "rdmds". see "rdmds" for more detail 
      - varname : the name of grid variables to be loaded.
                  if not specified, it reads 
                  'XC','YC','RAC','DXC','DYC','hFacC','hFacW','hFacS',
                  'Depth','RC','RF','DRC','DRF'
      - flag    : the default value is 1. If it is 2, then it additionally
                  reads : 'XG','YG','RAZ','DXG','DYG'
    + output
      - grd     : class variable that contains "varname"
    """
    tmp = rdmds(dirGrid + "XC")
    [Ly, Lx] = tmp.shape
    if varname is None:
        varname = ['XC','YC','RAC','DXC','DYC','hFacC','hFacW','hFacS','Depth',\
                   'RC','RF','DRC','DRF'];
        if flag==2:
            varname = np.append(varname,['XG','YG','RAZ','DXG','DYG'])

    class grd(object):
        for iv, vname in enumerate(varname):
            if region is None:
                exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                tmpvar = tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            else:
                if vname is 'RC' or vname is 'RF' or vname is 'DRC' or vname is 'DRF':
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                else:
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'",region='+str(region)+')');
                tmpvar = tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            if vname=='hFacC':
                mskC = hFacC.copy()
                mskC[mskC==0] = np.nan
                mskC[np.isfinite(mskC)] = 1.
            if vname=='hFacW':
                mskW = hFacW.copy()
                mskW[mskW==0] = np.nan
                mskW[np.isfinite(mskW)] = 1.
            if vname=='hFacS':
                mskS = hFacS.copy()
                mskS[mskS==0] = np.nan
                mskS[np.isfinite(mskS)] = 1.
            del tmpvar
    del grd.iv,grd.vname

    return grd

def calmld(RC, T, S, drho=0.03):
    """
    + object: compute the MLD using density
    + inputs 
      - RC   : vertical coordinate for T and S, 1D [m]
      - T    : vertical profile of potential temperature, 
               can be any dimension, NaN on land [degC]
      - S    : vertical profile of salinity, 
               same dimension as "T", [psu]
      - drho : density difference between the surface and MLD 
    + output
      - mld  : MLD with the same size of "T" or "S", [m] 
    """
    # compute density
    rho = densjmd95(S, T, 0) - 1000

    # check the dimension
    ndim = len(rho.shape)

    if ndim ==1:
        crirho = rho[0] + drho
        ic = np.isfinite(rho)
        # find the pressure value where rho = rho_surface + drho
        mld = np.interp(crirho, rho[ic], RC[ic])
    else:
        nz = rho.shape[0]
        rho = rho.reshape(nz, -1)
        _, nx  = rho.shape
        # Compute MLD
        mld = np.zeros(nx)
        for ix in range(nx):
            if np.isnan(rho[0,ix])==True:
                mld[ix] = np.nan
            else:
                rhocol = rho[:, ix]
                crirho = rhocol[0] + drho
                ic = np.isfinite(rhocol)
                # find the pressure value where rho = rho_surface + drho
                mld[ix] = np.interp(crirho, rhocol[ic], RC[ic])
         
        mld = mld.reshape(T.shape[1], -1) 

    return -mld

def computecurl(u, v, DXC, DYC, RAZ):
    """
    + object: Compute the curl 
    + input
      - u   : x-component of a flow or wind vector, 2D
              defined at U point
      - v   : v-component of a flow or wind vector, 2D
              defined at V point
      - DXC : distance in x direction between two grid centers, 2D, [m]
      - DYC : distance in y direction between two grid centers, 2D, [m]
      - RAZ : An area connecting 4 center points ("t" in the figure), 
              2D, [m2]
    + output
      - vcurl  : curl on the corner points at "p" in the figure
      - vcurlc : curl on the center points at "t" in the figure
      
    [ C-grid structure ]
     |       |-----------v-----------|-----------v----------|-
     |       |                       |                      |
     |       |                       |                      |
     |       |                       |                      |
     |       |                       |                      |
     |       u           t-----------u-----------t          |
     |       |          /|\          |           |          |
     |       |           |           |           |          |
     |       |           |           |           |          |
     |       |          dyC(i=1,     |           |          |
     | ---  -------------|--j=2,-----p-----------v----------|-
     | /|\   |           |  k=1)     |           |          |
     |  |    |           |           |           |          |
     |  |    |           |           |           |          |
     |dyG(   |          \|/          |           |          |
     |   i=1,u           t<---dxC(i=2,j=1,k=1)-->t          |
     |   j=1,|                       |                      |
     |   k=1)|                       |                      |
     |  |    |                       |                      |
     | \|/   |                       |                      |
     |"SB"++>|___________v___________|___________v__________|_
     |       <--dxG(i=1,j=1,k=1)----->
    """
    vcurl = np.zeros(grd.DXC.shape)
    vcurlc = np.zeros(grd.DXC.shape)

    vv1 = DXC * u
    vv2 = DYC * v
    vcurl[:, 1:] = vv2[:, 1:] - vv2[:, :-1]
    vcurl[1:, :] = vcurl[1:, :] - (vv1[1:, :] - vv1[:-1, :])
    vcurl = vcurl/RAZ

    vcurlc[1:-1, 1:-1] = 0.25 * (vcurl[1:-1, 1:-1] + vcurl[2:, 1:-1]\
                         + vcurl[1:-1, 2:] + vcurl[2:, 2:])

    return vcurl, vcurlc


