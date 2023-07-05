# import os
# import sys
import numpy as np
# import pandas as pd
import matplotlib.pyplot as mpl
# from scipy.spatial import distance_matrix
from zepyros.common import isolate_surfaces, concatenate_fig_plots, build_cone, flip_matrix, rotate_patch

DEB_FIND_ORIENT = 0


class Surface:
    """
    Surface is a class that manages a set of points that form a surface;
    it also allows analysis of parts of the surface itself, grouping the points into patches.

    Attributes
    ----------
    surface : ndarray or pandas
        coordinate array of ``x``, ``y``, ``z`` points and ``nx``, ``ny`` and ``nz`` unit vectors
    patch_num : int
    r0 : float
        the radius of the patch. The unit of measurement depends on that of the points in ``surface``
    theta_max : float
    real_br : array
    """
    # TODO: add documentation
    def __init__(self, surface, patch_num=5, r0=11, theta_max=45, real_br=None):
        if real_br is None:
            real_br = []

        # TODO: read_surface does not exist
        if isinstance(surface, str):    # if type(surface) == str:
            self.surface = self.read_surface(surface)
        else:
            self.surface = surface

        self.patch_num = patch_num      # number of patches to create
        self.r0 = r0                    # radius of the sphere to build the patch
        self.theta_max = theta_max      # maximum degree of the cone

        self.radius_of_cylinder = 2
        self.threshold_on_layers = 5

        if len(real_br) != 0:
            self.real_br = real_br      # saving the mask     #self.surface[real_br,:]
        else:
            self.real_br = []

    # TODO: maybe staticmethod and maybe remove unused THRESH parameter
    # def enlarge_pixels(self, plane, THRESH=300):
    @staticmethod
    def enlarge_pixels(plane):
        """
        Given a square matrix NxN with N < 400, repeats each cell
        in order to expand the starting matrix

        Parameters
        ----------
        `plane`: ndarray
            square matrix to expand

        Return
        ------
        `ndarray`
            expanded square matrix

        Example
        -------
        >>> surf = Surface(np.zeros(100, 6))
        >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mat
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
        >>> surf.enlarge_pixels(mat)
        array([[1., 1., 1., ..., 3., 3., 3.],
               [1., 1., 1., ..., 3., 3., 3.],
               [1., 1., 1., ..., 3., 3., 3.],
               ...,
               [7., 7., 7., ..., 9., 9., 9.],
               [7., 7., 7., ..., 9., 9., 9.],
               [7., 7., 7., ..., 9., 9., 9.]])
        >>> surf.enlarge_pixels(mat).shape
        (768, 768)
        """
        nx, ny = np.shape(plane)
        p = plane.copy()

        while nx < 400:
            tmp = np.zeros((nx*2, ny*2))
            tmp[::2, ::2] = p
            tmp[1::2, ::2] = p
            tmp[::2, 1::2] = p
            tmp[1::2, 1::2] = p

            nx, ny = np.shape(tmp)
            p = tmp.copy()
        return p

    def build_patch(self, point_ndx, d_min):
        """
        Given the index of a point belonging to a surface, define a patch
        as the set of points on that surface that are less than a given
        threshold distance from the point

        Parameters
        ----------
        `point_ndx`: int
            the index of the surface point chosen as the center of the patch
        `d_min`: float
            the distance a surface point must be from the center to belong to the patch

        Return
        ------
        `tuple`
            - the points on the surface belonging to the patch (`ndarray`)
            - the indices of the surface points belonging to the patch (`ndarray`)
        """
        d2 = (self.surface[:, 0] - self.surface[point_ndx, 0])**2 + (self.surface[:, 1] - self.surface[point_ndx, 1])**2 + (self.surface[:, 2] - self.surface[point_ndx, 2])**2

        # mask = self.distance_matrix[point_pos, :] <= self.r0
        mask = d2 <= self.r0**2
        patch_points = self.surface[mask, :]

        # processing patch to remove islands
        index_ = isolate_surfaces(patch_points, d_min)

        val, counts = np.unique(index_, return_counts=True)
        pos_ = np.where(counts == np.max(counts))[0][0]
        lab_ = val[pos_]

        mmm = np.ones(len(mask))*-1000.
        mmm[mask] = index_
        new_mask = mmm == lab_

        patch_points__ = self.surface[new_mask, :]

        return patch_points__, new_mask

    def build_patch_no_is(self, point_pos, d_min):
        # TODO: add documentation. Remove unused d_min param. Probably deprecated
        d2 = (self.surface[:, 0] - self.surface[point_pos, 0])**2 + (self.surface[:, 1] - self.surface[point_pos, 1])**2 + (self.surface[:, 2] - self.surface[point_pos, 2])**2

        # mask = self.distance_matrix[point_pos, :] <= self.r0
        mask = d2 <= self.r0**2
        patch_points = self.surface[mask, :]

        return patch_points, mask

    def find_patch_orientation(self, rot_protein, patch_mask):
        # TODO: add documentation
        rot_a = rot_protein.copy()
        rot_p = rot_a[patch_mask, :]

        # finding center of rotated patch
        cm = np.mean(rot_p[:, :3], axis=0)

        r = self.radius_of_cylinder         # AA the cylinder radius
        thresh = self.threshold_on_layers    # AA the threshold for different layers

        # translating surface to patch center
        rot_a[:, :3] -= cm

        # finding points inside a cylinder of radius R having center in the origin and height along the z axis
        d = np.sqrt(rot_a[:, 0]**2 + rot_a[:, 1]**2)
        # finding points of the surface inside the cylinder but not belonging to the patch
        mask = np.logical_and(d <= r, np.logical_not(patch_mask))

        z_ = rot_a[mask, 2]
        count, zz = np.histogram(z_, bins=20)
        zz = zz[:-1]

        if DEB_FIND_ORIENT:
            # showing complex
            mpl.figure()
            mpl.plot(zz, count)
            mpl.axvline(0)
            mpl.show()
            res, c = concatenate_fig_plots(list_=[rot_a[:, :3], rot_a[patch_mask, :3]])  # , rot_a[mask,:]])

        zz = zz[count > 0]
        check = zz > 0

        if np.sum(check) == 0:
            orientation = 1.
        elif np.sum(np.logical_not(check)) == 0:
            orientation = -1.
        else:
            positive = 0
            # if points are a mix of positive and negative the first must be negative
            negative = 1

            start = zz[0]
            for i in range(1, len(zz)):
                if start < 0 < zz[i]:  # if start < 0 and zz[i] > 0:
                    positive += 1.
                    start = zz[i]
                if start < 0:
                    if zz[i] - start > thresh:
                        negative += 1
                    start = zz[i]
                elif start > 0:
                    if zz[i] - start > thresh:
                        positive += 1
                    start = zz[i]

            # checking counting
            if positive % 2 == 0 and negative % 2 != 0:
                orientation = 1
            elif positive % 2 != 0 and negative % 2 == 0:
                orientation = -1.
            else:
                print("Attention! The code does not account for this case!\n Returned up orientation..")
                orientation = 2.

        return orientation, zz

    def create_plane(self, patch, z_c, n_p=20):
        # TODO: add documentation
        _, lc = np.shape(patch)
        
        rot_p = patch.copy()

        # computing geometrical center
        cm = np.mean(rot_p[:, :3], axis=0)  # TODO: unused. Remove?

        # shifting patch to have the cone origin in [0,0,0]
        rot_p[:, 2] -= z_c

        # computing distances between points and the origin
        weights = np.sqrt(rot_p[:, 0]**2 + rot_p[:, 1]**2 + rot_p[:, 2]**2)

        # computing angles
        thetas = np.arctan2(rot_p[:, 1], rot_p[:, 0])

        # computing distances in plane
        dist_plane = np.sqrt(rot_p[:, 0]**2 + rot_p[:, 1]**2)

        # computing the circle radius as the maximum distant point..
        r = np.max(dist_plane)*1.01

        # creating plane matrix
        if lc == 3:
            plane = np.zeros((n_p, n_p))
        else:
            plane = np.zeros((n_p, n_p), dtype=np.complex)
        # adapting points to pixels
        rot_p[:, 0] += r
        rot_p[:, 1] -= r

        pos_plane = rot_p[:, :2]    # np.abs(rot_p[:,:2])

        dr = 2.*r/n_p
        rr_x = 0
        rr_y = 0
        for i in range(n_p):
            rr_y = 0
            for j in range(n_p):
                mask_x = np.logical_and(pos_plane[:, 0] > rr_x, pos_plane[:, 0] <= rr_x + dr)
                mask_y = np.logical_and(pos_plane[:, 1] < -rr_y, pos_plane[:, 1] >= -(rr_y + dr))
                mask = np.logical_and(mask_x, mask_y)
                if len(weights[mask]) > 0:
                    w = np.mean(weights[mask])
                    if lc == 3:
                        plane[j, i] = w
                    else:
                        w_el = np.mean(patch[mask, 3])
                        plane[j, i] = w + 1j*w_el
                rr_y += dr
            rr_x += dr

        return plane, weights, dist_plane, thetas

    def find_origin(self, rotated_patch, check=0):
        """
        This function finds the origin of a cone that incorporates the patch with a maximum angle of 45 degrees.
        Input:
        - patch points (matrix)
        - CHECK, if 1 the cone is plotted.

        Output:
        - the origin (z-axis) of the cone

        TODO: generalize to a chosen degree...
        """
        # TODO: Update and improve documentation. Maybe staticmethod?
        # copying patch matrix ndO
        rot = rotated_patch.copy()

        # computing distances of points from the origin (geometrical center) in the xy plane
        dist_in_plane = np.sqrt(rot[:, 0]**2 + rot[:, 1]**2)

        # finding point with maximum distance
        max_dist = np.max(dist_in_plane)

        pos_max = np.where(dist_in_plane == max_dist)[0]
        if len(pos_max) > 1:
            pos_max = pos_max[0]

        # translating patch to put the centre of the cone in the origin
        d = -max_dist + rot[pos_max, 2]
        rot[:, 2] -= d
        # looking for points outside the cone: their plane distance must be bigger than their z component
        mask = dist_in_plane > np.abs(rot[:, 2])

        # shifting the cone origin until all points are inside the cone
        while np.sum(mask) != 0:

            # finding the maximum distance only among points outside the cone
            new_d = np.max(dist_in_plane[mask])
            pos_max_new = np.where(dist_in_plane == new_d)[0]

            if len(pos_max_new) > 1:
                pos_max_new = pos_max_new[0]

            # shifting the patch
            d = -new_d + rot[pos_max_new, 2]
            rot[:, 2] -= d

            # looking for outer point outside
            mask = dist_in_plane > np.abs(rot[:, 2])

        # plotting cone + patch
        if check:
            cone = build_cone(30, 50)
            all_ = np.row_stack([cone, rot])
            col = np.concatenate([np.ones(len(cone[:, 0]))*-10, np.ones(len(rot[:, 0]))*10])

        # find new center of the patch, corresponding to find the overall z shift
        new_cm = np.mean(rot[:, :3], axis=0)

        return -new_cm[2]

    def fill_inner_gap(self, plane_):
        """
        This function fills the inner gaps (pixel with zero value) in the unit circle of a NxN plane.
        It replaces the zero pixel with the mean of the nearby pixels, only for those pixels that have non-zero near pixels.
        Input:
        - plane (square matrix)

        Output:
        - Filled plane
        """

        # copying plane..
        plane = plane_.copy()

        plane_bin = np.copy(plane)
        plane_bin[plane_bin != 0] = 1.

        # finding dimensions..
        xl, yl = np.shape(plane)

        # defining radius
        r = int((xl-1)/2.)

        # defining radial
        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        # r_mat = x**2 + np.flip(y, axis=0)**2
        r_mat = x**2 + flip_matrix(y, axis=0)**2
        r2 = r**2

        # defining mask
        tmp = np.zeros((3, 3))
        tmp[1, 1] = 1
        x_, y_ = np.where(tmp == 0)
        x_ -= 1
        y_ -= 1

        list_x, list_y = np.where(plane == 0)
        l_x = len(list_x)
        l_x_old = 2*l_x
        count = 0

        while l_x < l_x_old:
            for i in range(l_x):
                x = list_x[i]
                y = list_y[i]

                if r_mat[x, y] < r2:
                    xx = x_ + x
                    yy = y_ + y
                    mask_x = np.logical_and(xx >= 0, xx < xl)
                    mask_y = np.logical_and(yy >= 0, yy < yl)
                    mask = np.logical_and(mask_x, mask_y)
                    xx = xx[mask]
                    yy = yy[mask]
                    tmp = plane_bin[xx, yy]
                    count = np.sum(tmp)
                    if count >= 6:
                        tmp = plane[xx, yy]
                        l__ = np.sum(tmp > 0)
                        tmp = np.sum(tmp)/l__
                        plane[x, y] = tmp

            list_x, list_y = np.where(plane == 0)
            l_x_old = l_x
            l_x = len(list_x)
            plane_bin = np.copy(plane)
            plane_bin[plane_bin != 0] = 1.

        return plane

    def fill_gap_everywhere(self, plane_):
        """
        This function fills the gaps (pixel with zero value) in the unit circle of a NxN plane.
        It replaces the zero pixel with the mean of the nearby pixels.
        Input:
        - plane (square matrix)
        
        Output:
        - Filled plane
        """
        # TODO: improve documentation. Maybe staticmethod?
        plane = plane_.copy()

        xl, yl = np.shape(plane)

        r = int((xl-1)/2.)

        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        r_mat = x**2 + flip_matrix(y, axis=0)**2
        r2 = r**2

        tmp = np.zeros((3, 3))
        x_, y_ = np.where(tmp == 0)
        x_ -= 1
        y_ -= 1

        list_x, list_y = np.where(plane == 0)

        count = 0

        while len(list_x) != 0 and count < 50:

            list_x, list_y = np.where(plane == 0)

            for i in range(len(list_x)):
                x = list_x[i]
                y = list_y[i]

                if r_mat[x, y] < r2:
                    if (
                        (plane[x+1, y] != 0 and plane[x-1, y] != 0) or
                        (plane[x, y+1] != 0 and plane[x, y-1] != 0)
                    ):
                        tmp = plane[x_ + x, y_ + y]

                        l__ = np.sum(tmp != 0)
                        tmp = np.sum(tmp)/l__
                        plane[x, y] = tmp

            list_x, list_y = np.where(plane == 0)

            for i in range(len(list_x)):
                x = list_x[i]
                y = list_y[i]

                if r_mat[x, y] < r2:
                    if(
                        (plane[x+1, y+1] !=0 and plane[x-1, y-1] != 0) or
                        (plane[x-1, y+1] !=0 and plane[x+1,y-1] != 0)
                    ):
                        tmp = plane[x_ + x, y_ + y]

                        l__ = np.sum(tmp != 0)
                        tmp = np.sum(tmp)/l__
                        plane[x, y] = tmp
            count += 1

        # final cycle
        list_x, list_y = np.where(np.logical_and(plane == 0, r_mat < r2))

        count = 0

        while len(list_x) != 0 and count < 50:

            list_x, list_y = np.where(plane == 0)

            for i in range(len(list_x)):
                x = list_x[i]
                y = list_y[i]

                if r_mat[x, y] < r2:

                    tmp = plane[x_ + x, y_ + y]

                    l__ = np.sum(tmp != 0)
                    if l__ == 0:
                        l__ = 1
                    tmp = np.sum(tmp)/l__
                    plane[x, y] = tmp
            count += 1

        plane[r_mat > r2] = 0

        return plane

    def patch_reorient(self, patch_points, verso):
        # TODO: add documentation. Maybe staticmethod?
        ll = np.shape(patch_points)[0]
    
        mean_v = np.mean(patch_points[:, 3:6], axis=0)
        pin = np.mean(patch_points[:, :3], axis=0)
    
        res, c11 = concatenate_fig_plots(list_=[patch_points[:, :3], patch_points[:, :3] + patch_points[:, 3:6]])

        phi, rot_patch_all = rotate_patch(res[:, :3], mean_v, verso, pin)
    
        rot_patch = rot_patch_all[:ll, :3]
        rot_normal_vec = rot_patch_all[ll:, :3] - rot_patch
    
        return rot_patch, rot_normal_vec
