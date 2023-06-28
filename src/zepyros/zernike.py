# import os
import sys
import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
# import scipy.ndimage as spd
from scipy.special import sph_harm
from matplotlib.image import imread
# from matplotlib import cm
from zepyros.common import log10_factorial, flip_matrix


class Zernike3D:
    """
    This class performs the 3D decomposition of a (voxelized) shape in the Zernike
    basis up to a given expansion order.
    """
    def __init__(self, n_l):
        self.zernike_dict = {}
        self.prepare_zernike(n_l)
        self.n_l = None
        self.pos = None
        self.n_voxels = None
        self.x = None
        self.y = None
        self.z = None
        self.r = None
        self.img = None

    def prepare_zernike(self, n_l=128):
        """
        This function initializes the quantities need to fastly compute zernike moments.
        It takes as input the voxel grid edge.
        """
        self.n_l = n_l
        self.x, self.y, self.z, self.r = self.initiate_cube(self.n_l)

        self.pos = np.where(self.r <= 1)
        self.n_voxels = np.shape(self.pos)[1]

        self.x = self.x[self.pos]
        self.y = self.y[self.pos]
        self.z = self.z[self.pos]
        self.r = self.r[self.pos]

        # return 1

    def prepare_image(self, image_file):
        """
        This function read a dx file and return a 3 x Nl linear array containing the value of each voxel.
        It also computes the Nl value assuming that the voxelization was performed on a Nl x Nl x Nl cube.
        """
        try:
            data = np.loadtxt(image_file, delimiter=",")
        except:
            sys.stderr.write("Error! Can not read the input file. Check format:\n .dx file with 4 columns.\n")
            exit()

        # finding Nl
        n_voxels = np.shape(data)[0]
        n_l = int(np.round(np.power(n_voxels, 1/3.)))
        if n_l**3 != n_voxels:
            n_tmp_m = n_l - 1
            n_tmp_p = n_l+1
            if n_tmp_m**3 == n_voxels:
                n_l = n_tmp_m
            elif n_tmp_p**3 == n_voxels:
                n_l = n_tmp_p
            else:
                sys.stderr.write("Error! Can not compute Nl (Nl x Nl x Nl = Nvoxels). Check if voxelization is cubic.\n")
                exit()

        if self.n_l != n_l:
            self.n_l = n_l
            self.zernike_dict = {}
            self.prepare_zernike(n_l)

        self.img = data[:, 3]
        self.img = self.img[self.pos]
 
        # return 1

    def epsilon(self, l, m):
        """
        This function computes the epsilon term of the Zernike moment, Z_nl^m = epsilon(l,m)*R(n,l)
        """
        _SUM_ = 0
        squared = -(self.x**2+self.y**2)/(4.*self.z**2)
        for mu in range(0, int((l-m)/2.)+1):
            tmp = sp.special.binom(l, mu) * sp.special.binom(l-mu, m+mu) * squared**mu
            _SUM_ += tmp
        tmp = self.c_lm(l, m) * (0.5*(-self.x - 1j*self.y))**m * self.z**(l-m) * _SUM_

        return tmp

    def c_lm(self, l, m):
        """
        This function computes one of the terms of epsilon(l,m) 
        """
        log_c = 0.5*(np.log10(2*l+1.) + log10_factorial(l-m) + log10_factorial(l+m)) - log10_factorial(l)
        c = 10**log_c / (2.*np.sqrt(np.pi))
        
        return c

    def q_klv(self, k, l, v):
        """
        This function computes one of the  terms of R(n,l).
        """
        tmp1 = (-1.)**(k+v)/2.**(2.*k) * np.sqrt((2*l+4*k+3)/3.)    # np.sqrt((2*l+4*k+3)/3.) #codice di zigzag... WRONG
        tmp2 = sp.special.binom(2*k, k) * sp.special.binom(k, v) * sp.special.binom(2*(k+l+v)+1, 2*k)
        tmp3 = sp.special.binom(k+l+v, k)
        q = tmp1*tmp2/tmp3
        
        return q

    def r_nl(self, n, l):
        """
        This function computes the R term of the Zernike moment, Z_nl^m = epsilon(l,m)*R(n,l)
        """
        tmp = 0
        k = int(0.5*(n-l))
        for v in range(k+1):
            tmp += self.q_klv(k, l, v)*self.r**(2*v)
        
        return tmp

    def compute_3d_moment(self, n=0, l=0, m=0, DICT_ON=True):
        """
        This function computes the Z_nl^m Zernike moment and stores it in a dictionary if DICT_ON = True (default).
        
        Note 1) Saving the moments in the dictionary is high memory consuming.  
     
        Note 2) Each moment is one of the orthonormal basis vector of the Zernike expansion.
        Given a 3D function, f(x,y,z) defined in the unitary sphere, it can be decomposed in the 
        Zernike basis as:
        f(x,y,z) = sum_nlm c_nlm Z_nlm

        Note 3) Z_nl^m = (-1)^m conjugate(Z_nl^m)
        """
        m_abs = np.abs(m)
        z_nlm_ = self.zernike_dict.get((n, l, m_abs))

        if z_nlm_ is None:

            tmp = self.epsilon(l, m)*self.r_nl(n, l)
            norm = np.sqrt(self.bracket(tmp, tmp))
            z_nlm_ = tmp/np.absolute(norm)*np.sqrt(n+1.)
            self.zernike_dict[(n, l, m_abs)] = z_nlm_.astype(np.complex64)
            if m >= 0:
                return z_nlm_
            else:
                return np.conjugate(z_nlm_)*(-1)**m
        else:
            if m >= 0:
                return z_nlm_
            else:
                return np.conjugate(z_nlm_)*(-1)**m

    def compute_3d_coefficient(self, f, n, l, m):
        """
        This function computes the Zernike coeffient associated to the Z_nlm moment as
        
        c_nlm = int_(R<1) dxdydz F(x,y,z) * conjugate(Z_nlm) 
        
        Note that since we have voxelized the space, the integral becomes a sum over the voxels 
        divided by the number of voxels (the voxels inside the R = 1 sphere).
        """
        z = self.compute_3d_moment(n, l, m)
        c = self.bracket(z, f)  #*float(n+1)

        return c, z

    def bracket(self, z1, z2):
        """
        This function computes the braket as
        c = < Z1 | Z2> = int dxdydz Z1 * conjugate(Z2) 
        """

        c = np.sum(np.conjugate(z1)*z2)/float(self.n_voxels)
        return c

    def initiate_cube(self, n=128):
        """
        This function initializes the x,y,z and r meshes on the 1x1x1 cube centered in (0,0,0).
        """
        v = np.linspace(0, 2, n)-1.
        x = np.zeros((n, n, n))
        y = np.zeros((n, n, n))
        z = np.zeros((n, n, n))

        tmp = np.zeros((n, n))

        for i in range(n):
            tmp[i, :] = v

        tmp2 = flip_matrix(np.transpose(tmp), axis=0)
        for i in range(n):
            x[:, :, i] = tmp
            y[:, :, i] = tmp2
            z[:, :, i] = v[n-1-i]

        r = np.sqrt(x**2 + y**2 + z**2)
        
        return x.ravel(), y.ravel(), z.ravel(), r.ravel()

    def from_unit_sphere_to_cube(self, img, n_l):
        """
        This function takes as input the linear array of voxel values (in the unitary sphere, r<1) and 
        return a Nl x Nl x Nl voxel grid.
        Input:
        - img, an 1d array.
        - Nl, a scalar, the cube edge (Nl x Nl x Nl).
        Return:
        - data, a (Nl x Nl x Nl) matrix containg the voxelized image.
        """
        tmp = np.zeros(n_l*n_l*n_l)
        tmp[self.pos] = img
        data = tmp.reshape((n_l, n_l, n_l))
        return data

    def from_cube_to_unit_sphere(self, data):
        """
        This function takes as input the Nl x Nl x Nl voxel grid and returns a linear array containing the values of the
        voxels in the unitary sphere, r<1.
        Input:
        - data, a (Nl x Nl x Nl) matrix containing the voxelized image.
        Return:
        - img, an 1d array.
        """
        tmp = data.ravel()
        img = tmp[self.pos]
        return img

    def compute_invariant(self, c_set, n_t):
        """
        This function computes the invariant for a dictionary containing all the coefficients. 
        """
        vet_ = []
        for n in range(0, n_t+1):
            for l in range(0, n+1):
                if (n-l) % 2 == 0:
                    c_nl = []
                    for m in range(-l, l+1):
                        c_nl.append(c_set[(n, l, m)])
                    c_nl = np.array(c_nl)
                    c_nl_ = np.sum(c_nl*np.conjugate(c_nl))
                    vet_.append(c_nl_.real/(n+1))
        return np.array(vet_)

    def decomposition(self, fig, n_t):
        """
        This function decomposes a 3D image in the Zernike basis up to order N.
        It returns the reconstructed image and the coefficient list (as a dictionary).
        """
        come_back = np.zeros((self.n_l, self.n_l, self.n_l), dtype=complex).ravel()
        come_back = come_back[self.pos]

        c_set = {}
        for n in range(0, n_t+1):
            for l in range(0, n+1):
                if (n-l) % 2 == 0:
                    for m in range(-l, l+1):
                        sys.stderr.write("\r Computing coefficient (n,l,m) = (%d,%d,%d)" % (n, l, m))
                        sys.stderr.flush()

                        c, z = self.compute_3d_coefficient(fig, n, l, m)
                        come_back += c*z
                        c_set[(n, l, m)] = c
                    
        return come_back, c_set

    def plot3d(self, myobj_list, isosuface_vec, r_thres=0.95, solo_real=True):
        """
        This function plots the isosurfaces of the passed voxel matrixes.
        """
        nobj = len(myobj_list)

        r_ = self.from_unit_sphere_to_cube(self.r, self.n_l)
  
        if solo_real:
            all_obj = np.zeros((self.n_l, self.n_l, self.n_l*nobj))

            for i in range(nobj):
                tmp = self.from_unit_sphere_to_cube(myobj_list[i], self.n_l)
                tmp[r_ > r_thres] = 0
                all_obj[:, :, i*self.n_l:(i+1)*self.n_l] = np.real(tmp)

        else:
            all_obj = np.zeros((self.n_l, 2*self.n_l, self.n_l*nobj))
            for i in range(nobj):
                tmp = self.from_unit_sphere_to_cube(myobj_list[i], self.n_l)
                tmp[r_ > r_thres] = 0
                all_obj[:, :self.n_l, i*self.n_l:(i+1)*self.n_l] = np.real(tmp)
                all_obj[:, self.n_l:, i*self.n_l:(i+1)*self.n_l] = np.imag(tmp)

        # return 1


class Zernike2D:
    """
    This class performs the 2D decomposition of a figure in its Zernike descriptors
    """

    def __init__(self, image_file):

        if isinstance(image_file, str):     # if type(image_file) == str:
            self.img = self.prepare_image(image_file)
        else:
            self.img = image_file

        n_l = np.shape(self.img)[0]
        self.n_l = n_l
        self.x, self.y = self.build_plane(n_l)
        self.r, self.t = self.from_cartesian_to_polar_plane(self.x, self.y)

        tmp = np.ones(np.shape(self.img))
        tmp = self.circle_image(tmp)

        self.npix = np.sum(tmp)
        self.zernike_dict = {}

    def circle_image(self, image):
        # TODO: add documentation. Maybe staticmethod?
        l, tmp = np.shape(image)
        new_image = image.copy()

        r = ((l - 1)/2.)
        r2 = r**2
        origin = np.array([r + 1, r + 1])

        for i in range(l):
            for j in range(i, l):
                d2 = (i - r)**2 + (j - r)**2
                if d2 > r2:
                    new_image[i, j] = 0
                    new_image[j, i] = 0
        return new_image

    def prepare_image(self, datafile):
        data = imread(datafile)
        data = data[:, :, 0]
        cut = 1
        x, y = np.shape(data)
        l = np.min([x, y])
        if l % 2 == 0:
            l -= 1
        new_image = np.zeros((l, l))

        r = ((l - 1)/2.)
        r2 = r**2
        origin = np.array([r+1, r+1])

        if x < y:
            start = int((y - l)/2.)
            new_image[:, :] = data[:l, start:start+l]
        elif y < x:
            start = int((x - l)/2.)
            new_image[:, :] = data[start:start+l, :l]
        else:
            new_image[:, :] = data[:l, :l]

        if cut:
            for i in range(l):
                for j in range(i, l):
                    d2 = (i - r)**2 + (j - r)**2
                    if d2 > r2:
                        new_image[i, j] = 0
                        new_image[j, i] = 0
        return new_image

    def compute_dot(self, mat_a, mat_b):
        mat_c = np.sum(mat_a*np.conjugate(mat_b))/float(self.npix)
        return mat_c

    def compute_coeff_nm(self, F, n, m):
        n_l, tmp = np.shape(F)
        dx = 1./(n_l - 1)

        z = self.compute_moment(n, m)
        c = self.compute_dot(F, z)*float(n+1)
        return c

    @staticmethod
    def _from_polar_to_cartesian(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def from_cartesian_to_polar_plane(self, x, y):
        l, tmp = np.shape(x)
        r_ = np.zeros((l, l))
        theta_ = np.zeros((l, l))

        for i in range(l):
            for j in range(l):
                r, t = self._from_cartesian_to_polar(x[i, j], y[i, j])
                r_[i, j] = r
                theta_[i, j] = t
        return r_, theta_

    @staticmethod
    def _from_cartesian_to_polar(x, y):
        r = np.sqrt(x**2 + y**2)

        if y == 0 and x > 0:
            theta = 0
        elif y == 0 and x < 0:
            theta = np.pi
        else:
            t = np.arctan(np.abs(y/x))
            if x > 0 and y > 0:
                theta = t
            elif x < 0 and y < 0:
                theta = t + np.pi
            elif x < 0 < y:   # elif x < 0 and y > 0:
                theta = np.pi - t
            elif y < 0 < x:   # elif x > 0 and y < 0:
                theta = 2*np.pi - t
            elif x == 0 and y > 0:
                theta = np.pi/2.
            elif x == 0 and y < 0:
                theta = 3.*np.pi/2.
            else:
                theta = 0.
        return r, theta

    def r_nm(self, n, m, l_r):
        rr = self.r.copy()
        mask = rr == 0
        rr[mask] = 1

        log10_r = np.log10(rr)

        r_nm_ = np.zeros(l_r)
        r_nm_0 = 0
        if (n - m) % 2 != 0:
            return r_nm_
        else:
            diff = int((n - m)/2.)
            summ = int((n + m)/2.)
            for l in np.arange(0, diff+1):
                # using log for product
                num = log10_factorial(n-l) + (n-2.*l)*log10_r
                den = log10_factorial(l) + log10_factorial(summ - l) + log10_factorial(diff - l)

                if n-2.*l == 0:
                    num0 = log10_factorial(n-l)
                    r_nm_0 += (-1)**l*10.**(num0-den)
                
                r_nm_ += (-1.)**l*10.**(num - den)
            r_nm_[mask] = r_nm_0
            return r_nm_

    def phi_m(self, theta, m):
        phi = np.cos(m*theta) + 1j*np.sin(m*theta)
        return phi

    def build_plane(self, n):
        plane_x = np.zeros((n, n))
        plane_y = np.zeros((n, n))

        n_r = int((n-1)/2.)
        dx = 1./n_r
        x = np.arange(0, n)*dx - 1.
        x_f = flip_matrix(x, axis=0)
        for i in range(n):
            plane_x[i, :] = x
            plane_y[:, i] = x_f
        return plane_x, plane_y

    def count_moment(self, n):
        """
        This function computes the number of moment that an expansion to the n order will produce.
        """
        if n % 2 == 0:
            n_z = n/2 + 1
            for i in range(1, int(n/2.)+1):
                n_z += 2*i
        else:
            n_z = 0
            for i in range(1, int((n+1)/2.)+1):
                n_z += 2*i
        return n_z

    def compute_moment(self, n, m):
        z_nm_ = self.zernike_dict.get((n, m))

        if z_nm_ is None:
            # TODO: o si assegna r_part*self.phi_m(self.t, m) tramite indici
            #  o z_nm = np.zeros(... si può rimuovere
            z_nm = np.zeros((self.n_l, self.n_l), dtype=complex)

            r_part = self.r_nm(n, np.abs(m), [self.n_l, self.n_l])

            z_nm = r_part*self.phi_m(self.t, m)

            z_nm[np.isnan(z_nm)] = 0
            z_nm_ = self.circle_image(z_nm)     # normalization factor.. #*np.sqrt((n+1))
            
            self.zernike_dict[(n, m)] = z_nm_
            return z_nm_
        else:
            return z_nm_

    def zernike_reconstruction(self, order, plot=True):
        _ROT_A = 0  # np.pi*30/180.
        c_list = []
        come_back = np.zeros((self.n_l, self.n_l), dtype=complex)
     
        # cycling over index n (order)
        for n in range(order+1):
            # cycling over moment m in (0, n)
            for m in range(0, n+1):
                if (n - m) % 2 == 0:
                    c = self.compute_coeff_nm(self.img, n, m)
                    c_list.append(c)
         
                    moment_ = self.compute_moment(n, m)

                    c_abs = np.absolute(c)
                    c_phi = np.arctan2(c.imag, c.real)

                    m_abs = np.absolute(moment_)
                    m_phi = np.arctan2(moment_.imag, moment_.real)

                    # come_back += c*moment_
                    if m != 0:
                        come_back += c_abs*m_abs*np.exp(1j*(m*(m_phi/float(m) + _ROT_A) + c_phi))
                    else:
                        come_back += c_abs*m_abs*np.exp(1j*(m_phi + c_phi))
     
                    if plot:
                        fig, ax = mpl.subplots(1, 2, dpi=150)
                        ax[0].imshow(moment_.real)
                        ax[1].imshow(moment_.imag)
                        mpl.show()

        if plot:
            fig, ax = mpl.subplots(1, 2, dpi=150)
            ax[0].imshow(come_back.real)
            ax[1].imshow(come_back.imag)
            mpl.show()

        return come_back, c_list

    def zernike_decomposition(self, order):
        c_list = []
     
        # cycling over index n (order)
        for n in range(order + 1):
            # cycling over moment m in (0, n)
            for m in range(0, n + 1):
                if (n - m) % 2 == 0:
                    c = self.compute_coeff_nm(self.img, n, m)
                    c_list.append(c)
        return c_list
