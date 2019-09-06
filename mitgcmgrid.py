from MITgcmutils import rdmds
import numpy as np

def loadgrid(gridname,region=None,varname=None,flag=1):

    if gridname=="dimes50":
        dirGrid="/Users/hajsong/Research/DIMES/Grid/"
    elif gridname=="arctic4":
        dirGrid="/net/fs09/d0/hajsong/Arctic/4km/Data/"
    elif gridname=="arctic36":
        #dirGrid="/Users/hajsong/School/Arctic/36km/Data/"
        dirGrid="/d1/jscott/ARCTIC/run_79-13_combo6_newdiags_layersT/"
    elif gridname=="dimes1deg":
        dirGrid="/Volumes/BigLacie/DIMES_OCCA/dimes_occa_dic_1deg/"
    elif gridname=="so_box":
        dirGrid="/home/hajsong/so_box_biogeo/run/"
    elif gridname=="sochannel":
        dirGrid="/net/fs09/d0/hajsong/SOchannel/Exp2d/res_exp20/"
    elif gridname=="sochannel3":
        dirGrid="/net/fs09/d0/siciak/so_chan_3d/run_1/res_t07/grid/"
    elif gridname=='soch1km':
        dirGrid="/net/fs09/d1/hajsong/SOCH/Exp1km/run/"
    elif gridname=='gyre2d':
        dirGrid="/Users/hajsong/Yonsei/Classes/ATM9107/mylectures/MITgcm_2Dgyre/run/"

    tmp=rdmds(dirGrid+"XC")
    [Ly,Lx]=tmp.shape
    if varname is None:
        varname=['XC','YC','RAC','DXC','DYC','hFacC','hFacW','hFacS','Depth',\
                 'RC','RF','DRC','DRF'];
        if flag==2:
            varname=np.append(varname,['XG','YG','RAZ','DXG','DYG'])

    class grd(object):
        for iv,vname in enumerate(varname):
            if region is None:
                exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                tmpvar=tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            else:
                if vname is 'RC' or vname is 'RF' or vname is 'DRC' or vname is 'DRF':
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                else:
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'",region='+str(region)+')');
                tmpvar=tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            if vname=='hFacC':
                mskC=hFacC.copy()
                mskC[mskC==0]=np.nan
                mskC[np.isfinite(mskC)]=1.
            if vname=='hFacW':
                mskW=hFacW.copy()
                mskW[mskW==0]=np.nan
                mskW[np.isfinite(mskW)]=1.
            if vname=='hFacS':
                mskS=hFacS.copy()
                mskS[mskS==0]=np.nan
                mskS[np.isfinite(mskS)]=1.
            del tmpvar
    del grd.iv,grd.vname

    return grd
