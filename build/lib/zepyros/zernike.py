import os, sys
import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
import scipy.ndimage as spd
from scipy.special import sph_harm
from matplotlib.image import imread
from matplotlib import cm


def myflip(m, axis):
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def isolate_isosurface(myprot, minV, maxV):
    #### This function groups points nearer than minD..
    
    DEB_ = 0
    
    lx, ly, lz = np.shape(myprot)
    
    prot = np.copy(myprot)
    mask = np.logical_and(prot>=minV, prot<= maxV)
    prot[:,:,:] = 0
    prot[mask] = 1.
    prot_label = np.copy(prot)
    
    # starting from label = 2
    lab = 2
        
    ## defining probe points..
    tmp = np.zeros((3,3,3))
    x, y, z = np.where(tmp==0)
    x = x-1
    y = y-1
    z = z-1
    
    # computing number of points without label..
    Nleft = np.sum(prot != 0)

    # starting iterating over different surfaces..
    while(Nleft > 0):    
        count = 1
        pos__ = np.where(prot != 0)
        if(DEB_):
            print("pos__",np.shape(pos__))
        
        # seeding: first unlabeled point takes lab label.. 
        prot_label[pos__[0][0], pos__[1][0], pos__[2][0]] = lab
        prot[pos__[0][0], pos__[1][0], pos__[2][0]] = lab
 
        if(DEB_):
            print("prot", prot)

        # iterating to find points belonging to the same surface...
        while(count > 0):
            count = 0
            pos_s = np.where(prot == lab)
            if(len(pos_s) == 0):
                break
            # creating mask for points still to be processed..
            mask = np.logical_and(prot > 0, prot != lab)
            if(DEB_):
                print("l", np.shape(pos_s))
            for i in range(np.shape(pos_s)[1]):
                if(DEB_):
                    print("pos",i,np.shape(pos_s) )                
                if(np.shape(pos_s)[1] == 1):
                    xxxx =  pos_s[0]
                    yyyy =  pos_s[1]
                    zzzz =  pos_s[2]
                else:
                    
                    xxxx = pos_s[0][i]
                    yyyy = pos_s[1][i]
                    zzzz = pos_s[2][i]
                x_ = x + xxxx
                y_ = y + yyyy
                z_ = z + zzzz
                
                mask_x = np.logical_and(x_ >= 0, x_<lx )
                mask_y = np.logical_and(y_ >= 0, y_<ly )
                mask_z = np.logical_and(z_ >= 0, z_<lz )
                
                mask_xyz = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
                
                x_ = x_[mask_xyz]
                y_ = y_[mask_xyz]
                z_ = z_[mask_xyz]
                if(DEB_):
                    print("xyz",x_, y_,z_)
                
                for j in range(len(x_)):
                    if(prot[x_[j], y_[j], z_[j]] == 1):
                        prot[x_[j], y_[j], z_[j]] = lab
                        prot_label[x_[j], y_[j], z_[j]] = lab
                        count += 1

                if(DEB_):
                    print("prot_2", prot)

                # removing processed point from system..
                prot[xxxx,yyyy,zzzz] = 0
                mm = np.logical_and(prot > 0, prot != lab)

                # creating mask for  point still to be processed..
                mmm = np.logical_and(mm,mask)
            
                mask = np.logical_and(prot > 0, prot != lab)
                if(DEB_):
                    print("prot_c", prot)
        
        # creating a new label..
        lab += 1
        # looking for how many points still to be processed...
        Nleft = np.sum(prot != 0)
        sys.stderr.write("\rleft %d"%Nleft)
        sys.stderr.flush()
    return(prot_label)


def log10_factorial(n):
    '''
    This function recursively computes  the log10 of a factorial.
    '''    
    if(n <=1):
        return(0)
    else:
        return(np.log10(n)+ log10_factorial(n-1))


class Zernike3D:
    """
    This class performs the 3D decomposition of a (voxelized) shape in the Zernike basis up to a given expansion order.
    """

    def __init__(self, Nl):
        self.zernike_dict = {}
        self.prepare_zernike(Nl)


    def prepare_zernike(self, Nl=128):
        '''
        This function initializes the quantities need to fastly compute zernike moments. It takes as input the voxel grid edge.
        '''
        self.Nl = Nl
        self.x,self.y,self.z,self.r = self.initiate_cube(self.Nl)

        self.pos = np.where(self.r<=1)
        self.Nvoxels = np.shape(self.pos)[1]

        self.x = self.x[self.pos]
        self.y = self.y[self.pos]
        self.z = self.z[self.pos]
        self.r = self.r[self.pos]

        return(1)


    def prepare_image(self, imagefile):
        '''
        This function read a dx file and return a 3 x Nl linear array containg the value of each voxel.
        It also compute the Nl value assuming that the voxelization was performed on a Nl x Nl x Nl cube.
        '''
        try:
            data = np.loadtxt(imagefile, delimiter=",")
        except:
            sys.stderr.write("Error! Can not read the input file. Check format:\n .dx file with 4 columns.\n")
            exit()

        ## finding Nl..
        Nvoxels = np.shape(data)[0]
        Nl = int(np.round(np.power(Nvoxels, 1/3.)))
        if(Nl**3 != Nvoxels):
            Ntmp_m = Nl-1
            Ntmp_p = Nl+1
            if(Ntmp_m**3 == Nvoxels):
                Nl = Ntmp_m
            elif(Ntmp_p**3 == Nvoxels):
                Nl = Ntmp_p
            else:
                sys.stderr.write("Error! Can not compute Nl (Nl x Nl x Nl = Nvoxels). Check if voxelization is cubic.\n")
                exit()

        if(self.Nl != Nl):
            self.Nl = Nl
            self.zernike_dict = {}
            self.prepare_zernike(Nl)

        self.img = data[:,3]
        self.img = self.img[self.pos]
 
        return(1)


    def epsilon(self, l, m):
        '''
        This function computes the epsilon term of the Zernike moment, Z_nl^m = epsilon(l,m)*R(n,l)
        '''        
        SUM = 0
        squared = -(self.x**2+self.y**2)/(4.*self.z**2)
        for mu in range(0, int((l-m)/2.)+1):
            tmp = sp.special.binom(l,mu)*sp.special.binom(l-mu,m+mu)*(squared)**mu
            SUM += tmp
        tmp = self.C_lm(l,m)*(0.5*(-self.x -1j*self.y))**m*self.z**(l-m)*SUM

        return(tmp)


    def c_lm(self,l,m):
        '''
        This function computes one of the terms of epsilon(l,m) 
        '''
        logc = 0.5*( np.log10(2*l+1.) + log10_factorial(l-m) + log10_factorial(l+m)) - log10_factorial(l)
        c = 10**(logc)/ (2.*np.sqrt(np.pi))
        
        return(c)


    def q_klv(self,k,l,v):
        '''
        This function computes one of the  terms of R(n,l).
        '''
        tmp1 = (-1.)**(k+v)/2.**(2.*k)*np.sqrt((2*l+4*k+3)/3.) #np.sqrt((2*l+4*k+3)/3.) #codice di zigzag... WRONG 
        tmp2 =  sp.special.binom(2*k,k)*sp.special.binom(k,v)*sp.special.binom(2*(k+l+v)+1,2*k)
        tmp3 = sp.special.binom(k+l+v,k)
        q = tmp1*tmp2/tmp3
        
        return(q)

    
    def r_nl(self,n,l):
        '''
        This function computes the R term of the Zernike moment, Z_nl^m = epsilon(l,m)*R(n,l)
        '''
        tmp = 0
        k = int(0.5*(n-l))
        for v in range(k+1):
            tmp += self.Q_klv(k,l,v)*self.r**(2*v)
        
        return(tmp)
    

    def compute_3d_moment(self, n=0, l=0, m=0, DICT_ON=True):
        '''
        This function computes the Z_nl^m Zernike moment and stores it in a dictionary if DICT_ON = True (default).
        
        Note 1) Saving the moments in the dictionary is high memory consuming.  
     
        Note 2) Each moment is one of the ortonormal basis vector of the Zernike espansion.
        Given a 3D function, f(x,y,z) defined in the unitary sphere, it can be decomposed in the 
        Zernike basis as:
        f(x,y,z) = sum_nlm c_nlm Z_nlm

        Note 3) Z_nl^m = (-1)^m conjugat(Z_nl^m) 

        '''
        M = np.abs(m)
        Z_nlm_ = self.zernike_dict.get((n,l,M))

        if Z_nlm_ is None:

            tmp = self.epsilon(l,m)*self.R_nl(n,l)
            norm =  np.sqrt(self.bracket(tmp,tmp))
            Z_nlm_ = tmp/np.absolute(norm)*np.sqrt(n+1.)
            self.zernike_dict[(n,l,M)] = (Z_nlm_).astype(np.complex64)
            if(m>=0):
                return(Z_nlm_)
            else:
                return(np.conjugate(Z_nlm_)*(-1)**m)
        else:
            if(m>=0):
                return(Z_nlm_)
            else:
                return(np.conjugate(Z_nlm_)*(-1)**m)


    def compute_3d_coefficient(self, F, n, l, m):
        '''
        This function computes the Zernike coeffient associated to the Z_nlm moment as
        
        c_nlm = int_(R<1) dxdydz F(x,y,z) * conjugate(Z_nlm) 
        
        Note that since we have voxelized the space, the integral becomes a sum over the voxels 
        divided by the number of voxels (the voxels inside the R = 1 sphere).
        '''
        Z = self.compute_3d_moment(n,l,m)
        c = self.bracket(Z,F)  #*float(n+1)

        return(c, Z)

    
    def bracket(self, Z1, Z2):
        '''
        This function computes the braket as
        c = < Z1 | Z2> = int dxdydz Z1 * conjugate(Z2) 
        '''

        c = np.sum(np.conjugate(Z1)*Z2)/float(self.Nvoxels)
        return(c)


    def initiate_cube(self, N=128):
        '''
        This function initializes the x,y,z and r meshes on the 1x1x1 cube centered in (0,0,0).
        '''
        
        v = np.linspace(0,2,N)-1.
        x = np.zeros((N,N,N))
        y = np.zeros((N,N,N))
        z = np.zeros((N,N,N))

        tmp = np.zeros((N,N))

        for i in range(N):
            tmp[i,:] = v

        tmp2 = myflip(np.transpose(tmp),axis=0)
        for i in range(N):
            x[:,:,i] = tmp
            y[:,:,i] = tmp2
            z[:,:,i] = v[N-1-i]

        r = np.sqrt(x**2+ y**2+z**2)
        
        return(x.ravel(),y.ravel(),z.ravel(),r.ravel())


    def from_unit_sphere_to_cube(self, img, Nl):
        '''
        This function takes as input the linear array of voxel values (in the unitary sphere, r<1) and 
        return a Nl x Nl x Nl voxel grid.
        Input:
        - img, an 1d array.
        - Nl, a scalar, the cube edge (Nl x Nl x Nl).
        Return:
        - data, a (Nl x Nl x Nl) matrix containg the voxelized image.
        '''
        tmp = np.zeros(Nl*Nl*Nl)
        tmp[self.pos] = img
        data = tmp.reshape((Nl, Nl, Nl))
        return(data)


    def from_cube_to_unit_sphere(self, data):
        '''
        This function takes as imput the Nl x Nl x Nl voxel grid and returns a linear array containing the values of the 
        voxels in the unitary sphere, r<1.
        Input:
        - data, a (Nl x Nl x Nl) matrix containing the voxelized image.
        Return:
        - img, an 1d array.
        '''
        tmp = data.ravel()
        img = tmp[self.pos]
        return(img)


    def compute_invariant(self, c_set, N):
        '''
        This function computes the invariant for a dictionary containing all the coefficients. 
        '''
        vet_ = []
        for n in range(0,N+1):
            for l in range(0,n+1):
                if((n-l)%2 == 0):
                    c_nl = []
                    for m in range(-l, l+1):
                        c_nl.append(c_set[(n,l,m)])
                    c_nl = np.array(c_nl)
                    c_nl_ = np.sum(c_nl*np.conjugate(c_nl))
                    vet_.append(c_nl_.real/(n+1))
        return(np.array(vet_))


    def decomposition(self, myfig, N):
        '''
        This function decomposes a 3D image in the Zernike basis up to order N.
        It returns the reconstructed image and the coefficient list (as a dictionary).
        '''
        
        come_back = np.zeros((self.Nl, self.Nl, self.Nl), dtype=complex).ravel()
        come_back = come_back[self.pos]

        c_set = {}
        for n in range(0,N+1):
            for l in range(0,n+1):
                if((n-l)%2 == 0):

                    for m in range(-l, l+1):

                        sys.stderr.write("\r Computing coefficient (n,l,m) = (%d,%d,%d)"%(n,l,m))
                        sys.stderr.flush()

                        c,Z = self.compute_3d_coefficient(myfig,n,l,m)
                        come_back += c*Z
                        c_set[(n,l,m)] = c
                    
        return(come_back, c_set)

    def plot3d(self, myobj_list, isosuface_vec, r_thres = 0.95,solo_real = True):
        '''
        This function plots the isosurfaces of the passed voxel matrixes.
        '''

        nobj = len(myobj_list)

        r_ = self.from_unit_sphere_to_cube(self.r, self.Nl)
  
        if(solo_real):
            all_obj = np.zeros((self.Nl,self.Nl,self.Nl*nobj))

            for i in range(nobj):
                tmp = self.from_unit_sphere_to_cube(myobj_list[i], self.Nl)
                tmp[r_ > r_thres] = 0
                all_obj[:,:,i*self.Nl:(i+1)*self.Nl] = np.real(tmp)
      
            
        else:
            all_obj = np.zeros((self.Nl,2*self.Nl,self.Nl*nobj))
            for i in range(nobj):
                tmp = self.from_unit_sphere_to_cube(myobj_list[i], self.Nl)
                tmp[r_ > r_thres] = 0
                all_obj[:,:self.Nl,i*self.Nl:(i+1)*self.Nl] = np.real(tmp)
                all_obj[:,self.Nl:,i*self.Nl:(i+1)*self.Nl] = np.imag(tmp)
      
        return(1)


class Zernike2d:
    '''
    This class performs the 2D decomposition of a figure in its Zernike descriptors
    '''

    def __init__(self, imagefile):

        if(type(imagefile) == str):
            self.img = self.prepare_image(imagefile)
        else:
            self.img = imagefile

        Nl  = np.shape(self.img)[0]
        self.Nl = Nl
        self.x, self.y = self.build_plane(Nl)
        self.r, self.t = self.from_cartesian_to_polar_plane(self.x,self.y)

        tmp = np.ones(np.shape(self.img))
        tmp = self.circle_image(tmp)

        self.npix = np.sum(tmp)
        self.zernike_dict = {}


    def circle_image(self,image):

        l,tmp = np.shape(image)
        new_image = image.copy()

        r  = ((l-1)/2.)
        r2 = r**2
        origin = np.array([r+1, r+1])

        for i in range(l):
            for j in range(i,l):
                d2 = (i-r)**2 + (j-r)**2
                if(d2>r2):
                    new_image[i,j] = 0
                    new_image[j,i] = 0
        return(new_image)


    def prepare_image(self, datafile):
        data = imread(datafile)
        data = data[:,:,0]
        CUT = 1
        x,y = np.shape(data)
        l = np.min([x,y])
        if(l%2 == 0):
            l -= 1
        new_image = np.zeros((l,l))

        r  = ((l-1)/2.)
        r2 = r**2
        origin = np.array([r+1, r+1])

        if(x < y):
            start = int((y-l)/2.)
            new_image[:,:] = data[:l,start:start+l]
        elif(y < x):
            start = int((x-l)/2.)
            new_image[:,:] = data[start:start+l,:l]
        else:
            new_image[:,:] = data[:l,:l]

        if(CUT):
            for i in range(l):
                for j in range(i,l):
                    d2 = (i-r)**2 + (j-r)**2
                    if(d2>r2):
                        new_image[i,j] = 0
                        new_image[j,i] = 0
        return(new_image)


    def compute_dot(self, A, B):
        c= np.sum(A*np.conjugate(B))/float(self.npix)
        return(c)


    def compute_coeff_nm(self, F, n, m):
        Nl, tmp = np.shape(F)
        dx = 1./(Nl-1)

        Z = self.compute_moment(n,m)
        c = self.compute_dot(F,Z)*float(n+1)
        return(c)


    def from_polar_to_cartesian(self, r, theta):
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return(x,y)


    def from_cartesian_to_polar_plane(self, x, y):
        l, tmp = np.shape(x)
        R_ = np.zeros((l,l))
        Theta_ = np.zeros((l,l))

        for i in range(l):
            for j in range(l):
                r, t = self.from_cartesian_to_polar(x[i,j],y[i,j])
                R_[i,j] = r
                Theta_[i,j] = t
        return(R_, Theta_)


    def from_cartesian_to_polar(self, x, y):
        r = np.sqrt(x**2 + y**2)

        if(y==0 and x >0):
            theta = 0
        elif(y==0 and x <0):
            theta = np.pi
        else:
            t = np.arctan(np.abs(y/x))
            if(x> 0 and y>0):
                theta = t
            elif(x<0 and y<0):
                theta = t + np.pi
            elif(x<0 and y>0):
                theta = np.pi - t
            elif(x>0 and y<0):
                theta = 2*np.pi - t
            elif(x==0 and y >0):
                theta = np.pi/2.
            elif(x==0 and y <0):
                theta = 3.*np.pi/2.
            else:
                theta = 0.
        return(r, theta)


    def r_nm(self, n, m, Lr):
        rr = self.r.copy()
        mask = rr == 0
        rr[mask] = 1

        log10_r = np.log10(rr)

        R_nm_ = np.zeros(Lr)
        R_nm_0 = 0
        if((n-m)%2 != 0):
            return(R_nm_)
        else:
            diff= int((n-m)/2.)
            summ = int((n+m)/2.)
            for l in np.arange(0, diff+1):
                ## using log for product..
                num = log10_factorial(n-l) + (n-2.*l)*log10_r
                den = log10_factorial(l) + log10_factorial(summ - l)  + log10_factorial(diff - l)

                if(n-2.*l == 0):
                    num0 = log10_factorial(n-l)
                    R_nm_0 += (-1)**l*10.**(num0-den)
                
                R_nm_ += (-1.)**l*10.**(num - den)
            R_nm_[mask] = R_nm_0
            return(R_nm_)


    def phi_m(self, theta, m):
        phi = np.cos(m*theta) + 1j*np.sin(m*theta)
        return(phi)


    def build_plane(self, N):
        plane_x = np.zeros((N,N))
        plane_y = np.zeros((N,N))

        Nr = int((N-1)/2.)
        dx = 1./Nr
        x = np.arange(0,N)*dx - 1.
        x_f = myflip(x,axis=0)
        for i in range(N):
            plane_x[i,:] = x
            plane_y[:,i] = x_f
        return(plane_x, plane_y)


    def count_moment(self, n):
        '''
        This function computes the number of moment that an expansion to the n order will produce.
        '''
        if(n%2 == 0):
            N_z = n/2 + 1
            for i in range(1, int(n/2.)+1):
                N_z += 2*i
        else:
            N_z = 0
            for i in range(1, int((n+1)/2.)+1):
                N_z += 2*i
        return(N_z)


    def compute_moment(self, n, m):
        Z_nm_ = self.zernike_dict.get((n,m))

        if Z_nm_ is None:
            Z_nm = np.zeros((self.Nl,self.Nl), dtype=complex)

            r_part = self.r_nm(n,np.abs(m),[self.Nl,self.Nl])

            Z_nm = r_part*self.phi_m(self.t, m)

            Z_nm[np.isnan(Z_nm)] =0
            Z_nm_ = self.circle_image(Z_nm)  #normalization factor.. #*np.sqrt((n+1))
            
            self.zernike_dict[(n,m)] = Z_nm_
            return(Z_nm_)
        else:
            return(Z_nm_)


    def zernike_reconstruction(self, order, PLOT = 1):
        ROT_A = 0, #np.pi*30/180.
        c_list = []
        come_back = np.zeros((self.Nl, self.Nl), dtype=complex)
     
        ## cycling over index n (order)..
        for n in range(order+1):
            ### cycling over moment m in (0, n)
            for m in range(0, n+1):
                if((n-m)%2 == 0):
                    c = self.compute_coeff_nm(self.img, n,m)
                    c_list.append(c)
         
                    moment_ = self.compute_moment(n,m)

                    c_abs = np.absolute(c)
                    c_phi = np.arctan2(c.imag, c.real)

                    m_abs = np.absolute(moment_)
                    m_phi = np.arctan2(moment_.imag, moment_.real)

                    #come_back += c*moment_
                    if(m != 0):
                        come_back += c_abs*m_abs*np.exp(1j*(m*(m_phi/float(m) + ROT_A) + c_phi))
                    else:
                        come_back += c_abs*m_abs*np.exp(1j*(m_phi + c_phi))
     
                    if(PLOT):
                        fig, ax = mpl.subplots(1,2, dpi = 150)
                        ax[0].imshow(moment_.real)
                        ax[1].imshow(moment_.imag)
                        mpl.show()

        if(PLOT):
            fig, ax = mpl.subplots(1,2, dpi = 150)
            ax[0].imshow(come_back.real)
            ax[1].imshow(come_back.imag)
            mpl.show()

        return(come_back, c_list)


    def zernike_decomposition(self, order):
        c_list = []
     
        ## cycling over index n (order)..
        for n in range(order+1):
            ### cycling over moment m in (0, n)
            for m in range(0, n+1):
                if((n-m)%2 == 0):
                    c = self.compute_coeff_nm(self.img, n, m)
                    c_list.append(c)
        return(c_list)

