import numpy as np
import sys
import matplotlib.pyplot as mpl


def rotate_matrix(cos, sin, axis):
    """
    Given sine and cosine of the angle and the axis of rotation returns
    the rotation matrix

    Parameters
    ----------
    `cos`: real
        the cosine of the angle
    `sin`: real
        the sine of the angle
    `axis`: integer
        the axis of rotation: 0 for x, 1 for y and 2 for z

    Returns
    -------
    matrix
        the rotation matrix
    """
    rotated_matrix = np.zeros((3, 3))
    if axis == 0:
        rotated_matrix[0, 0] = 1
        rotated_matrix[1, 1] = cos
        rotated_matrix[1, 2] = -sin
        rotated_matrix[2, 1] = sin
        rotated_matrix[2, 2] = cos
    if axis == 1:
        rotated_matrix[1, 1] = 1
        rotated_matrix[0, 0] = cos
        rotated_matrix[0, 2] = sin
        rotated_matrix[2, 0] = -sin
        rotated_matrix[2, 2] = cos
    if axis == 2:
        rotated_matrix[2, 2] = 1
        rotated_matrix[0, 0] = cos
        rotated_matrix[1, 0] = sin
        rotated_matrix[0, 1] = -sin
        rotated_matrix[1, 1] = cos
    return rotated_matrix


def rotate_patch(patch, mean_normal_v, axis_z_or, pin):
    """
    Doc me!
    """
    # TODO: add documentation
    xy = True
    yz = True
    # xz = True   # unused

    _ESP_ = 1e-10

    patch_trans = patch[:, :3] - pin

    n_points = np.shape(patch_trans)[0]

    # defining rotating vectors...
    r_z = np.array([0, 0, axis_z_or])
    r_vn = mean_normal_v.copy()  # np.mean(normal_v, axis=0) #nter_atom_pos[1, :]

    p = np.abs(1 - r_z.dot(r_vn) / np.sqrt((r_z.dot(r_z)) * (r_vn.dot(r_vn))))

    r_vn1 = r_vn.copy()

    while p > _ESP_:

        r1 = np.array([r_vn1[0], r_vn1[1]])
        # r2 = np.array([r_z[0], r_z[1]])   # unused
        r1 /= np.sqrt(r1.dot(r1))

        cos_theta = r1[1] / np.sqrt(r1.dot(r1))  # (r1[0]*r2[0] + r1[1]*r2[1])/r1.dot(r1)
        sin_theta = r1[0] / np.sqrt(r1.dot(r1))  # (r1[0]*r2[1] - r1[1]*r2[0])/r1.dot(r1)

        r = rotate_matrix(cos_theta, sin_theta, 2)
        if xy:
            for i in range(n_points):
                patch_trans[i, :] = np.dot(r, patch_trans[i, :])
            r_vn1 = np.dot(r, r_vn1)

        # y-z plane
        r1 = np.array([r_vn1[1], r_vn1[2]])
        # r2 = np.array([r_z[1], r_z[2]])   # unused
        r1 /= np.sqrt(r1.dot(r1))

        if axis_z_or > 0:
            cos_theta = r1[1]
            sin_theta = r1[0]
        else:
            cos_theta = -r1[1]
            sin_theta = -r1[0]

        r = rotate_matrix(cos_theta, sin_theta, 0)
        if yz:
            for i in range(n_points):
                patch_trans[i, :] = np.dot(r, patch_trans[i, :])
            r_vn1 = np.dot(r, r_vn1)

        p = np.abs(1 - r_z.dot(r_vn1) / np.sqrt((r_z.dot(r_z)) * (r_vn1.dot(r_vn1))))
    return r_vn1, patch_trans


def flip_matrix(mat, axis):
    """
    Flips a matrix based on its axis

    Parameters
    ----------
    `mat`: ndarray
        the matrix to flip
    `axis`: int
        the axis about which to flip the ndarray (0 for x, 1 for y)

    Return
    ------
    matrix
        the flipped matrix with respect the axis

    Examples
    --------
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> mat
    array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
    >>> flip_matrix(mat, 0)
    array([[7, 8, 9],
       [4, 5, 6],
       [1, 2, 3]])
    >>> flip_matrix(mat, 1)
    array([[3, 2, 1],
       [6, 5, 4],
       [9, 8, 7]])
    """
    if not hasattr(mat, 'ndim'):
        m = np.asarray(mat)
    indexer = [slice(None)] * mat.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, mat.ndim))
    return mat[tuple(indexer)]


def isolate_surfaces(surface, min_d=1.):
    """
    Given a surface of x, y, z points, form groups of points that are
    at a distance less than a given threshold

    Parameters
    ----------
    `surface`: ndarray
        array of points where each row is a point in space.
        If the number of columns is greater than three,
        only the first three columns will be used
    `min_d`: float
        the minimum distance at which two points must be located
        to be grouped in the same cluster.
        The unit of measurement depends on that of the points in `surface`

    Return
    ------
    array
        list of integer (starts from 2), where each integer represents
        the label of a specific group. The i-th element indicates the group
        to which the i-th point of the initial surface belongs

    Examples
    --------
    >>> mat = np.array([[4.5, 4.5, 4.5], [1.5, 1.5, 1.5], [2., 1., 1.5], [5., 5., 5.], [1., 1., 1.]])
    >>> mat
    array([[4.5, 4.5, 4.5],
           [1.5, 1.5, 1.5],
           [2. , 1. , 1.5],
           [5. , 5. , 5. ],
           [1. , 1. , 1. ]])
    >>> isolate_surfaces(mat, 1)
    array([2, 3, 3, 2, 3], dtype=int8)
    """
    min_d2 = min_d ** 2  # squaring distance to avoid sqrt
    l, tmp = np.shape(surface)  # computing number of surface points

    # initializing label vectors
    surf_label = np.ones(l, dtype=np.int8)
    surf_tmp = np.ones(l, dtype=np.int8)

    lab = 2  # starting from label = 2
    n_left = np.sum(surf_tmp != 0)  # computing number of points without label

    # starting iterating over different surfaces
    while n_left > 0:
        count = 1
        pos__ = np.where(surf_tmp != 0)

        # seeding: first unlabeled point takes lab label
        surf_label[pos__[0][0]] = lab
        surf_tmp[pos__[0][0]] = lab

        # iterating to find points belonging to the same surface
        while count > 0:
            count = 0
            pos_s = np.where(surf_tmp == lab)
            # creating mask for points still to be processed
            mask = np.logical_and(surf_tmp > 0, surf_tmp != lab)

            for i in pos_s[0]:
                # computing distances between points and the i-th point
                d = (surface[:, 0] - surface[i, 0]) ** 2 + (surface[:, 1] - surface[i, 1]) ** 2 + (
                            surface[:, 2] - surface[i, 2]) ** 2

                m_np = d < min_d2  # m_np: mask near points
                surf_tmp[i] = 0  # removing processed point from system
                m_proc = np.logical_and(m_np, mask)  # m_proc: mask for point still to be processed

                tot_to_proc = np.sum(m_proc)

                if tot_to_proc > 0:
                    surf_label[m_proc] = lab
                    surf_tmp[m_proc] = lab
                    count += 1
                mask = np.logical_and(surf_tmp > 0, surf_tmp != lab)

        lab += 1  # creating a new label
        n_left = np.sum(surf_tmp != 0)  # looking for how many points still to be processed

    return surf_label


def find_border(new_plane_ab):
    """
    This function finds the border of a figure in the plane...
    """
    # TODO: add documentation
    a, b = np.shape(new_plane_ab)
    p_h = np.ones((a, b))
    p_v = np.ones((a, b))

    index = np.arange(0, a)

    for i in range(a):

        # horizontal
        tmp = new_plane_ab[i, :]
        lr = tmp != 0
        if len(index[lr]) > 0:
            pos_1 = index[lr][0]
            pos_2 = index[lr][-1]

            p_h[i, :pos_1] = 0
            p_h[i, (pos_2 + 1):] = 0
        else:
            p_h[i, :] = 0

        # vertical
        tmp = new_plane_ab[:, i]
        lr = tmp != 0
        if len(index[lr]) > 0:
            pos_1 = index[lr][0]
            pos_2 = index[lr][-1]

            p_v[:pos_1, i] = 0
            p_v[(pos_2 + 1):, i] = 0
        else:
            p_v[:, i] = 0

    return p_v * p_h


def contact_points(list_1, list_2, thresh):
    """
    This function finds the groups of point of list1 and list2 that have a distance lesser that thresh
    from at least one point of the other list.
    """
    thresh2 = thresh ** 2
    contact_1 = [0, 0, 0, 0]
    contact_2 = [0, 0, 0]
    l1 = np.shape(list_1)[0]
    l2 = np.shape(list_2)[0]

    list_index_1 = []
    list_index_2 = []

    indexes_l1 = np.arange(l1)
    indexes_l2 = np.arange(l2)

    mmm = np.zeros(4)

    for i in range(l1):
        if i % 1000 == 0:
            sys.stderr.write("\rProgress %d / %d" % (i, l1))
        d2 = (list_1[i, 0] - list_2[:, 0]) ** 2 + (list_1[i, 1] - list_2[:, 1]) ** 2 + (
                    list_1[i, 2] - list_2[:, 2]) ** 2
        mask = d2 < thresh2

        if np.sum(mask) > 0:
            mmm[:3] = list_1[i, :3]
            mmm[3] = np.min(d2[mask])

            list_index_1 = np.concatenate([list_index_1, [i]])
            list_index_2 = np.concatenate([list_index_2, indexes_l2[mask]])

            contact_1 = np.row_stack([contact_1, mmm])
            contact_2 = np.row_stack([contact_2, list_2[mask, :3]])

    list_index_2 = np.unique(list_index_2)

    try:
        contact_2 = np.unique(contact_2, axis=0)
    except:
        aaa = contact_2.tolist()
        output = [0, 0, 0]
        for x in aaa:
            if x not in output:
                output = np.row_stack([output, x])
        contact_2 = output.copy()

    l1 = np.shape(contact_2)[0]
    mmm = []
    for i in range(l1):
        if i % 1000 == 0:
            sys.stderr.write("\rProgress %d / %d" % (i, l1))
        d2 = (list_1[:, 0] - contact_2[i, 0]) ** 2 + (list_1[:, 1] - contact_2[i, 1]) ** 2 + (
                    list_1[:, 2] - contact_2[i, 2]) ** 2
        mmm.append(np.min(d2))
    contact_2 = np.column_stack([contact_2, np.array(mmm)])

    return contact_1[1:, :], contact_2[1:, :], list_index_1, list_index_2


def eig_rotation(points, eig_vec):
    rot = np.transpose(np.dot(np.transpose(eig_vec), np.transpose(points)))
    return rot


# TODO: what is this?
def plot_3d_points_and_vectors(x, y, z, u, v, w, color=None):
    if color is None:
        color = []
    ll = len(x)
    mask = np.random.choice(np.arange(ll), 100, replace=False)


def build_cone(z_max, n_disk):
    dz = z_max / float(n_disk)
    z = 0

    n = 100
    res = [0, 0, 0]
    rad = np.linspace(0, 2 * np.pi, n)
    for i in range(n_disk):
        z += dz
        x = z * np.cos(rad)
        y = z * np.sin(rad)

        res = np.row_stack([res, np.column_stack([x, y, np.ones(n) * z])])
    return res


def concatenate_fig_plots(list_):
    # TODO: add documentation
    l = len(list_)
    res = list_[0]

    n, tmp = np.shape(list_[0])

    col_list = np.linspace(-100, 100, l)
    col = np.ones(n) * col_list[0]

    if l > 1:
        for i in range(1, l):
            res = np.row_stack([res, list_[i]])
            n, tmp = np.shape(list_[i])
            col = np.concatenate([col, np.ones(n) * col_list[i]])
    return res, col


def fix_bridge_real_bs(patch_ab, patch_ag, d_pp):
    """
    This function isolates the different groups of points in two given sets (patches)
    according to a cutoff distance Dpp.
    It associates each group to the closest group of the other set and returns
    a list of matched patches.
    """
    # TODO: add documentation
    # processing  patches to remove islands
    # ab
    index_ab_bd_ = isolate_surfaces(patch_ab, d_pp)

    val_ab, counts_ab = np.unique(index_ab_bd_, return_counts=True)
    print("val", val_ab, "counts", counts_ab)

    # ag
    index_ag_bd_ = isolate_surfaces(patch_ag, d_pp)

    val_ag, counts_ag = np.unique(index_ag_bd_, return_counts=True)
    print("val", val_ag, "counts", counts_ag)

    # creating matrix of number of points in each group and label returned by IsolateSurface func.
    # Columns are ordered from the biggest to the smallest patch.
    tmp = np.row_stack([counts_ab, val_ab])

    # s_c_ab = np.flip(tmp[:, np.argsort(tmp[0, :])], axis=1)
    s_c_ab = flip_matrix(tmp[:, np.argsort(tmp[0, :])], axis=1)

    tmp = np.row_stack([counts_ag, val_ag])

    s_c_ag = flip_matrix(tmp[:, np.argsort(tmp[0, :])], axis=1)

    # finding center of mass of each group...
    cm_ab = []
    cm_ag = []
    for l in s_c_ab[1, :]:
        tmp = patch_ab[index_ab_bd_ == l]
        cm_ab.append(np.mean(tmp[:, :3], axis=0))
    for l in s_c_ag[1, :]:
        tmp = patch_ag[index_ag_bd_ == l]
        cm_ag.append(np.mean(tmp[:, :3], axis=0))

    l_ag = np.shape(s_c_ag)[1]
    l_ab = np.shape(s_c_ab)[1]

    # computing distance matrix between the centers of the groups intra sets...
    d = np.zeros((l_ab, l_ag))
    for i in range(l_ab):
        for j in range(l_ag):
            d[i, j] = np.sum((np.array(cm_ab[i]) - np.array(cm_ag[j])) ** 2)

    # associating groups according to the minimal distance...
    index_ab = []
    index_ag = []
    l_min = np.min([l_ag, l_ab])
    for i in range(l_min):
        if l_min == l_ab:
            x = np.where(d[i, :] == np.min(d[i, :]))[0][0]
            index_ab.append(i)
            index_ag.append(x)
        else:
            x = np.where(d[:, i] == np.min(d[:, i]))[0][0]
            index_ab.append(x)
            index_ag.append(i)

    patch_ab_list = []
    patch_ag_list = []

    lab_ab = s_c_ab[1, index_ab]
    lab_ag = s_c_ag[1, index_ag]

    # defining list of matched patches
    for i in range(l_min):
        patch_ab_list.append(patch_ab[index_ab_bd_ == lab_ab[i]])
        patch_ag_list.append(patch_ag[index_ag_bd_ == lab_ag[i]])

    return patch_ab_list, patch_ag_list


def isolate_isosurface(my_prot, min_v, max_v):
    # This function groups points nearer than minD

    _DEB_ = 0

    lx, ly, lz = np.shape(my_prot)

    prot = np.copy(my_prot)
    mask = np.logical_and(prot >= min_v, prot <= max_v)
    prot[:, :, :] = 0
    prot[mask] = 1.
    prot_label = np.copy(prot)

    # starting from label = 2
    lab = 2

    # defining probe points
    tmp = np.zeros((3, 3, 3))
    x, y, z = np.where(tmp == 0)
    x = x - 1
    y = y - 1
    z = z - 1

    # computing number of points without label
    n_left = np.sum(prot != 0)

    # starting iterating over different surfaces
    while n_left > 0:
        count = 1
        pos__ = np.where(prot != 0)
        if _DEB_:
            print("pos__", np.shape(pos__))

        # seeding: first unlabeled point takes lab label.
        prot_label[pos__[0][0], pos__[1][0], pos__[2][0]] = lab
        prot[pos__[0][0], pos__[1][0], pos__[2][0]] = lab

        if _DEB_:
            print("prot", prot)

        # iterating to find points belonging to the same surface...
        while count > 0:
            count = 0
            pos_s = np.where(prot == lab)
            if len(pos_s) == 0:
                break
            # creating mask for points still to be processed
            mask = np.logical_and(prot > 0, prot != lab)
            if _DEB_:
                print("l", np.shape(pos_s))
            for i in range(np.shape(pos_s)[1]):
                if _DEB_:
                    print("pos", i, np.shape(pos_s))
                if np.shape(pos_s)[1] == 1:
                    xxxx = pos_s[0]
                    yyyy = pos_s[1]
                    zzzz = pos_s[2]
                else:
                    xxxx = pos_s[0][i]
                    yyyy = pos_s[1][i]
                    zzzz = pos_s[2][i]
                x_ = x + xxxx
                y_ = y + yyyy
                z_ = z + zzzz

                mask_x = np.logical_and(x_ >= 0, x_ < lx)
                mask_y = np.logical_and(y_ >= 0, y_ < ly)
                mask_z = np.logical_and(z_ >= 0, z_ < lz)

                mask_xyz = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

                x_ = x_[mask_xyz]
                y_ = y_[mask_xyz]
                z_ = z_[mask_xyz]
                if _DEB_:
                    print("xyz", x_, y_, z_)

                for j in range(len(x_)):
                    if prot[x_[j], y_[j], z_[j]] == 1:
                        prot[x_[j], y_[j], z_[j]] = lab
                        prot_label[x_[j], y_[j], z_[j]] = lab
                        count += 1

                if _DEB_:
                    print("prot_2", prot)

                # removing processed point from system
                prot[xxxx, yyyy, zzzz] = 0
                mm = np.logical_and(prot > 0, prot != lab)

                # creating mask for  point still to be processed
                mmm = np.logical_and(mm, mask)

                mask = np.logical_and(prot > 0, prot != lab)
                if _DEB_:
                    print("prot_c", prot)

        # creating a new label
        lab += 1
        # looking for how many points still to be processed
        n_left = np.sum(prot != 0)
        sys.stderr.write("\rleft %d" % n_left)
        sys.stderr.flush()
    return prot_label


def log10_factorial(n):
    """
    Compute log10(n) factorial

    Parameters
    ----------
    `n`: int
        Input values. If n < 0, the return value is 0

    Return
    ------
    real
        Factorial of log10(n): ``log(n)! = log(n) + log(n-1) + ... + log(1)``

    Examples
    --------
    >>> log10_factorial(10)
    6.559763032876794
    """
    if n <= 1:
        return 0
    else:
        return np.log10(n) + log10_factorial(n - 1)
