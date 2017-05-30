import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from noise import pnoise2
from voronoi import *
import random
import datetime


#Recommended settings: N = 1000 with 50 evolutions
N = 1000

randpoints = np.random.rand(N,2)

newpoints,regions,vertices = relax(randpoints,4)
#plot_voronoi(newpoints,regions,vertices)

def island_elev(elevations, points,dist):
    elevs = np.copy(elevations)
    elevs = np.where(((points[:,1] < dist) | (points[:,1] > 1 - dist) | (points[:,0] < dist) | (points[:,0] > 1 - dist)), -1.0, elevs)
    return elevs

def plot_heat(newpoints,regions,vertices, vals):
    noise = 0.005
    blur = True
    newpoints = np.swapaxes(newpoints,0,1)
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    patches = []
    for region in regions:
        reg = np.array(region)
        if reg.size:
            inds = reg[reg != -1]
            polygon = vertices[inds]
            patches.append(Polygon(polygon))
    #plt.plot(newpoints[0],newpoints[1], lw = 0, ms = 5, marker = '.', c = 'b')

    cdict = {'red':[(0.0,  0.0, 0.0),
                   (0.49,  0.5, 0.9),
                   (0.98,  0.0, 0.2),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.3),
                   (0.49, 0.7, 0.85),
                   (0.98, 0.5, 0.5),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0,  0.5),
                   (0.49,  0.6, 0.6),
                   (0.98, 0.2, 0.2),
                   (1.0,  1.0, 1.0)]}

    zachcm = LinearSegmentedColormap('zachcm', cdict)
    p = PatchCollection(patches, cmap=zachcm, alpha=1, lw = 0)
    p.set_array(vals)
    p.set_clim([-1,1.01])
    ax.add_collection(p)
    plt.xlim(0,1)
    plt.ylim(0,1)
    #fig.colorbar(p, ax=ax)
    water_regs = np.array(regions)[(vals < 0.0)]
    land_regs = np.array(regions)[(vals >= 0.0)]
    water_inds = np.concatenate(np.array(regions)[(vals < 0.0)])
    land_inds = np.concatenate(np.array(regions)[(vals >= 0.0)])
    boundary = vertices[np.intersect1d(water_inds, land_inds)]
    water_boundary = []
    land_boundary = []
    for w in water_regs:
        if np.any(np.in1d(np.array(w), np.intersect1d(water_inds, land_inds))):
            water_boundary.append(w)
    for l in land_regs:
        if np.any(np.in1d(np.array(l), np.intersect1d(water_inds, land_inds))):
            land_boundary.append(l)
    for w in water_boundary:
        for l in land_boundary:
            aw,al = np.array(w),np.array(l)
            shared = np.intersect1d(aw,al)
            if shared.size:
                #print(shared)
                shared_v = vertices[shared]
                plt.plot(np.linspace(shared_v[0,0],shared_v[1,0],10) + noise*(np.random.rand(10) - 0.5),np.linspace(shared_v[0,1],shared_v[1,1], 10)+ noise*(np.random.rand(10) - 0.5), lw = 2, c = '#004d80', ms = 0)
    #print(boundary)
    #plt.plot(boundary[:,0],boundary[:,1], lw = 0, ms = 5, marker = '.', c = 'r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    #plt.savefig('maps/' + datetime.datetime.now().strftime("%B_%d_%Y_%I_%M%p") + '.png', dpi = 500)
    #plt.show()

def local_avg(elevations,points):
    elevs = np.copy(elevations)
    dist = 0.01
    for n in np.arange(elevs.size):
        localElevs = elevs[np.sum(np.square(points - points[n]), axis = 1) <= dist]
        newElev = (np.sum(localElevs) + elevs[n]*2)/(2 + localElevs.size)
        elevs[n] = newElev
    return elevs

def repeated_avg(elevations,points,num):
    elevs = np.copy(elevations)
    for n in np.arange(num):
        elevs = local_avg(elevs,points)
    return elevs

def evolviter(elevations,points):
    elevs = np.copy(elevations)
    dist = 0.005
    for n in np.arange(elevs.size):
        localElevs = elevs[np.sum(np.square(points - points[n]), axis = 1) <= dist]
        avg = np.mean(localElevs)
        if avg < -0.6:
            test = random.random()
            if test < 0.01:
                elevs[n] = 1
            else:
                elevs[n] = -1
        if avg >= -0.6 and avg < -0.3:
            test = random.random()
            if test < 0.2:
                elevs[n] = 1
            else:
                elevs[n] = -1
        if avg >= -0.3 and avg <= 0.3:
            test = random.random()
            if test < 0.8:
                if elevs[n] >= 0:
                    elevs[n] = 1
                else:
                    elevs[n] = -1
        if avg > 0.3 and avg <= 0.6:
            test = random.random()
            if test < 0.2:
                elevs[n] = -1
            else:
                elevs[n] = 1
        if avg > 0.6:
            test = random.random()
            if test < 0.01:
                elevs[n] = 0.01
            else:
                elevs[n] = 1
    return elevs

def random_evolve(elevations,points,num):
    elevs = np.copy(elevations)
    for n in np.arange(num):
        elevs = island_elev(elevs,points,0.1)
        elevs = evolviter(elevs,points)
    elevs = local_avg(elevs,points)
    return elevs


elevation = island_elev(np.full(N,1.0), newpoints, 0.2)
#print(newpoints.shape)
#plot_heat(newpoints,regions,vertices, elevation)
elevation = random_evolve(elevation,newpoints,50)
elevation = np.where((elevation < 0.0), np.min(elevation), elevation)
plot_heat(newpoints,regions,vertices, elevation)
plt.show()
