import numpy as np

def invert_map(fldZ, zC, zF, fldVec):
    """
    [kFld,zFld,k0Fld]=invert_map(fldZ,zC,zF,fldVec,[ktyp]);

    Description:
    Find vertical index (kFld) and depth (zFld) for each value of 
    vector fldVec according to value in vertical map field (fldZ(:,k))
    with depth zC(k) (center), zF(k) (interface) 

    Dimension: 
        inp: fldZ(nx,nr), zC(nr), zF(nr+1), fldVec(ny)
        out: kFld(nx,ny), zFld(nx,ny), k0Fld(nx,ny)
     
    - for now, assume fldZ(k) increasing with k (ktyp=1)
    """
    ny=len(fldVec)

    [nr,nx] = fldZ.shape

    kFld=np.zeros([ny,nx]); zFld=np.zeros([ny,nx]); k0Fld=np.zeros([ny,nx]);
    
    #- extended vert. res:
    zExt = np.append(zC,zF[-1]);

    #%- assume field=0 @ land-pts and field > 0 elsewhere:
    MxV=np.nanmax(fldZ); botV=np.nanmax(fldVec); botV=np.maximum(botV,MxV)+1;
    var=fldZ.copy();

    # Fill the land with botV (maximum density)
    var[np.isnan(fldZ)] = botV;

    fldV=np.ones([nr+1,nx])*botV; fldV[:-1,:] = var;
    mnV=np.min(fldV);
    var1d = fldV.flatten()

    for j,ff in enumerate(fldVec):
        kk=np.zeros(nx, dtype='int32')-1 
        found=np.zeros(nx, dtype='int32'); 
        zz=np.zeros(nx);
        var=(fldV[1:nr+1,:]-ff)*(fldV[:-1,:]-ff);
        [Ks, Is] = np.nonzero(var < 0);
        kk[Is] = Ks;
        found[Is] = 1

        # When the density is equal to the predefined density
        [Ke, Ie] = np.nonzero(fldV[:-1,:] == ff);
        nEx=len(Ie); nu=0;
        
        if nEx > 0:                
            Iu = np.unique(Ie); 
            nu = len(Iu);
            if nu == nEx:
                kk[Ie]=Ke;
            else:
                print("warning 1")
                Ku = Iu.copy();
                for l in range(nu):
                    L = np.nonzero(Ie == Iu[l]);
                    Ku[l] = Ke[L[0]];
                kk[Iu] = Ku;

        k1 = np.maximum(kk,0);    # in fact, the minimum kk is already 0.
        k2 = np.minimum(kk+1,nr); # when kk=nr, limit k2 to nr-1 (k2 is the index).
        ik1 = k1*nx; ik1 = ik1 + np.arange(nx);
        ik2 = k2*nx; ik2 = ik2 + np.arange(nx);
        k1 = k1.astype('int32'); k2 = k2.astype('int32')
        ik1 = ik1.astype('int32'); ik2 = ik2.astype('int32');

        dfld = var1d[ik2] - var1d[ik1]    # density difference between upper and lower cell with respect to ff
        dfld[np.nonzero(kk==-1)] = 1.;
        # [J]=np.nonzero(dfld <= 0 );     # J is not used

        dfld = 1./dfld;
        frac = ff - var1d[ik1]; frac = frac * dfld;
        zz = zExt[k1] + frac*(zExt[k2]-zExt[k1]);
        dk = frac * (k2 - k1);
        #print found[100], kk[100], dk[100]
        zz[np.nonzero(found==0)] = 0.;
        dk[np.nonzero(found==0)] = 0.;

        [I1]=np.nonzero(fldV[0,:] > ff);

        if len(I1) > 0:
            zz[I1]=zF[0]; kk[I1]=0;

        if (len(I1)+len(Is)+nu != nx):
            #fprintf(' error for j= %i, ff= %f : nSt= %i , nEx= %i , nL1= %i\n', ...
            #j,ff,length(Is),nu,length(I1));
            print("warning 2 "+str(ff))
        #- save in output array:
        kFld[j,:]=kk+dk;
        zFld[j,:]=zz;
        k0Fld[j,:]=kk;

    return kFld, zFld, k0Fld
