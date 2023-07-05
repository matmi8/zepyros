import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as m_ticker
import seaborn as sns
import zepyros.surface as sf
import zepyros.zernike as zf
mpl.use('Agg')


def get_zernike(surf, radius, ndx, order, verso=1, n_pixel=25):
    """
    Identifies a patch and calculates Zernike polynomials that characterize it

    Parameters
    ----------
    `surf`: ndarray
        surface of the whole system with 6 columns: coordinate ``x``, ``y``, ``z`` points
        and ``nx``, ``ny`` and ``nz`` unit vectors
    `radius`: float
        the radius of the patch
    `ndx`: int
        index of the point selected as center of the patch
    `order`: int
        the expansion order of polynomials
    `verso`: int
        the direction of the patch with respect to its axis:
        1 for the positive direction, -1 for the negative direction (default: ``1``)
    `n_pixel`: int
        number of pixels per side with which the patch will be reconstructed (default: ``25``)

    Return
    ------
    `tuple`
        - polynomial coefficients (`ndarray`)
        - Zernike 2d disk data (`ndarray`)
        - the indices of the surface points belonging to the patch (`array`)
    """
    # TODO: passi già la superficie come deve essere
    # lag = len(surf["x"])
    # surf_z = np.zeros((lag, 6))
    # surf_z[:, :] = surf[["x", "y", "z", "nx", "ny", "nz"]]

    if isinstance(surf, pd.DataFrame):
        surf = surf.to_numpy()
    elif not isinstance(surf, np.ndarray):
        raise TypeError('`surf` must be pandas dataframe or numpy ndarray')

    surf_obj = sf.Surface(surf[:, :], patch_num=0, r0=float(radius), theta_max=45)

    patch, mask = surf_obj.build_patch(point_ndx=ndx, d_min=.5)
    surf_obj.real_br = mask

    rot_patch, rot_ag_patch_nv = surf_obj.patch_reorient(patch, verso)
    z = surf_obj.find_origin(rot_patch)
    plane, weights, dist_plane, thetas = surf_obj.create_plane(patch=rot_patch, z_c=z, n_p=n_pixel)

    if np.shape(rot_patch)[1] == 4:
        new_plane_re = surf_obj.fill_gap_everywhere(plane_=np.real(plane))
        new_plane_im = surf_obj.fill_gap_everywhere(plane_=np.imag(plane))
        new_plane_re_ = surf_obj.enlarge_pixels(new_plane_re)
        new_plane_im_ = surf_obj.enlarge_pixels(new_plane_im)
        new_plane_ = new_plane_re_ + 1j * new_plane_im_ / np.max(np.abs(new_plane_im_))
    else:
        new_plane = surf_obj.fill_gap_everywhere(plane_=plane)
        new_plane_ = surf_obj.enlarge_pixels(new_plane)     # enlarging plane

    # try:
    #     zernike_env.img = new_plane_
    # except:
    #     zernike_env = zf.Zernike2D(new_plane_)

    zernike_env = zf.Zernike2D(new_plane_)
    br_coeff = zernike_env.zernike_decomposition(order=int(order))
    disk_data = zernike_env.img
    reduced_disk = disk_data[np.ix_(range(0, 400, 16), range(0, 400, 16))]

    df_disk = pd.DataFrame(reduced_disk).reset_index().melt('index')
    df_disk.columns = ['row', 'column', 'value']

    return np.absolute(br_coeff), df_disk, mask


def plot_disk(df_disk, save_path=None):
    """
    Plots 2d Zernike disk

    Parameters
    ----------
    `disk_data`: ndarray
        the disk data obtained from ``get_zernike`` function
    `save_path`: string
        optional; the path where to save the disk image

    Return
    ------
    `matplotlib.figure.Figure`
        the disk image as object
    """
    # reduced_disk = disk_data[np.ix_(range(0, 400, 16), range(0, 400, 16))]
    #
    # df_disk = pd.DataFrame(reduced_disk).reset_index().melt('index')
    # df_disk.columns = ['row', 'column', 'value']
    #
    x = df_disk["column"]
    y = df_disk["row"]
    z = df_disk["value"]

    xi = np.linspace(0, 24, 501)  # [i for i in range(0, 25)]
    yi = np.linspace(0, 24, 501)  # [i for i in range(0, 25)]
    zi = sp.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

    xy_center = [12, 12]
    radius = 11

    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(len(xi)):
        for j in range(len(yi)):
            r = np.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
            if (r - dr / 2) > radius:
                zi[j, i] = "nan"

    # make figure
    fig = plt.figure(figsize=(10, 10))

    # set aspect = 1 to make it a circle
    ax = fig.add_subplot(111, aspect=1)

    # cs = ax.contourf(xi, yi, zi, 1000, cmap='viridis', zorder=1)
    ax.contourf(xi, yi, zi, 1000, cmap='viridis', zorder=1)

    circle = mpl.patches.Circle(
        xy=xy_center,
        radius=radius,
        edgecolor="steelblue",
        facecolor="none",
        linewidth=4)

    ax.add_patch(circle)

    # make the axis invisible
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    pl_disk = ax.get_figure()

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')
        plt.close()

    return pl_disk


def plot_coeff(coeff, save_path=None):
    """
    Plots the zernike invariants, i.e. the absolute value of the real part
    of the coefficients of the polynomials

    Parameters
    ----------
    ``coeff``: ndarray
        the zernike invariants obtained for example by ``get_zernike`` function
    ``save_path``: string
        optional; the path where to save the disk image

    Return
    ------
    `matplotlib.figure.Figure`
        the zernike invariants image as object
    """
    x = [i for i in range(1, (len(coeff) + 1))]

    df_coeff = pd.DataFrame(
        {
            'x': x,
            'y': coeff
        }
    )

    plt.figure(figsize=(14, 6))

    ax = sns.lineplot(data=df_coeff, x="x", y="y", linewidth=2, color="#36648B")

    x_label = ax.get_xticks().tolist()
    y_label = ax.get_yticks().tolist()

    yrest = [i % 1 for i in y_label]

    ax.xaxis.set_major_locator(m_ticker.FixedLocator(x_label))
    ax.yaxis.set_major_locator(m_ticker.FixedLocator(y_label))
    ax.set_xticklabels([str(int(i)) for i in x_label], fontsize=16)

    if sum([i != 0 for i in yrest]) > 0:
        ax.set_yticklabels([str(float(i)) for i in y_label], fontsize=16)
    else:
        ax.set_yticklabels([int(float(i)) for i in y_label], fontsize=16)

    plt.xlabel('Index', size=18)
    plt.ylabel('Zernike invariants', size=18)

    pl_coeff = ax.get_figure()

    if save_path is not None:
        plt.savefig(save_path, facecolor='white', transparent=False, bbox_inches='tight')
        plt.close()

    return pl_coeff
