import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, voronoi_plot_2d

#Makes voronoi finite; from stackoverflow
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

#calculates centroid of polygon
def centroid(vertices):
    #print(vertices)
    xs = np.append(vertices[:,0],vertices[0,0])
    ys = np.append(vertices[:,1],vertices[0,1])
    xs = np.where((xs < 0.0), 0.0, xs)
    ys = np.where((ys < 0.0), 0.0, ys)
    xs = np.where((xs > 1.0), 1.0, xs)
    ys = np.where((ys > 1.0), 1.0, ys)
    #print(xs)
    area = 0.5 * np.sum(xs[:-1]*ys[1:] - xs[1:]*ys[:-1])
    if area == 0.0:
        c_x = np.mean(xs[:-1])
        c_y = np.mean(ys[:-1])
    else:
        c_x = 1/(6.0*area)*np.sum((xs[:-1] + xs[1:])*(xs[:-1]*ys[1:] - xs[1:]*ys[:-1]))
        c_y = 1/(6.0*area)*np.sum((ys[:-1] + ys[1:])*(xs[:-1]*ys[1:] - xs[1:]*ys[:-1]))
    #if c_x < 0.0:
        #print(c_x)
        #print(xs)
        #print(np.where(xs < 0.0, 0, xs))
    return c_x,c_y

#plot voronoi diagrams
def plot_voronoi(newpoints,regions,vertices):
    newpoints = np.swapaxes(newpoints,0,1)
    fig,ax = plt.subplots()
    patches = []
    for region in regions:
        reg = np.array(region)
        if reg.size:
            inds = reg[reg != -1]
            polygon = vertices[inds]
            patches.append(Polygon(polygon))
    plt.plot(newpoints[0],newpoints[1], lw = 0, ms = 10, marker = '.', c = 'b')
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()



#voronoi relaxation alg
def relax(points, N):
    regions,vertices = voronoi_finite_polygons_2d(Voronoi(points))
    for n in np.arange(N):
        cxs,cys = [],[]
        for region in regions:
            reg = np.array(region)
            if reg[reg != -1].size:
                polygon = vertices[reg[reg != -1]]
                cx,cy = centroid(polygon)
                cxs.append(cx)
                cys.append(cy)
        points = np.swapaxes(np.array([cxs,cys]),0,1)
        regions,vertices = voronoi_finite_polygons_2d(Voronoi(points))
    return points,regions,vertices

#points = np.random.rand(500,2)

#newpoints,regions,vertices = relax(points,4)
#plot_voronoi(newpoints,regions,vertices)
