'''
File specifying various operations on mesh and vertice-level inputs.
'''

import pymesh
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.spatial.transform as scipy_tf


def plot_mesh_verts(mesh, ax, va = 0.4, fa = 0.8, cmap=None):
    '''Plot mesh alongside vertices through mpl'''
    if not hasattr(mesh, 'vertices'): 
        plot_pointfield(mesh, ax, a = va, cmap=cmap)
    vertices = [[v[0], v[1], v[2]] for v in mesh.vertices]
    polyset = [[vertices[vi] for vi in face] for face in mesh.faces]
    if fa: ax.add_collection3d(Poly3DCollection(polyset, fc='orange', linewidths=1, alpha=fa))
    if va: ax.scatter3D(*zip(*vertices), s=0.1, alpha=va, cmap=cmap)


def plot_pointfield(verts, ax, a = 1, cmap='Greens'):
    '''Plot pointfield in mpl'''
    if hasattr(verts, 'vertices'): verts = verts.vertices
    x = np.array([p[0] for p in verts])
    y = np.array([p[1] for p in verts])
    z = np.array([p[2] for p in verts])
    ax.scatter3D(x, y, z, alpha=a, c=z, cmap=cmap)


def voxelize(mesh, cells):
    '''Voxelize a mesh into a grid with `cells`^2 blocks'''
    if not hasattr(mesh, 'vertices'): 
        raise TypeError("voxelize(...) requires mesh, not vertices") 
    cell_size = max(b[1]-b[0] for b in zip(*mesh.bbox))/cells
    grid = pymesh.VoxelGrid(cell_size, mesh.dim)
    grid.insert_mesh(mesh)
    grid.create_grid()
    return grid.mesh


def to_pointcloud(mesh):
    '''Convert a mesh into an explicit point cloud format. Usually not necessary'''
    mesh.add_attribute("vertex_normal")
    v_normals = mesh.get_vertex_attribute("vertex_normal")
    out = pymesh.form_mesh(mesh.vertices, np.zeros((0, 3), dtype=int))
    [out.add_attribute(d) for d in ('nx', 'ny', 'nz')]
    [out.set_attribute(d, v_normals[:,i].ravel()) for i,d in enumerate(('nx', 'ny', 'nz'))]
    return out


def sample_pointcloud(verts, n):
    '''Get a sample of n points from a point cloud (or set of vertices)'''
    return_mesh = hasattr(verts, 'vertices') 
    if return_mesh: verts = verts.vertices
    s_idx = np.random.choice(verts.shape[0], n, replace=False)
    if return_mesh: return pymesh.form_mesh(verts[s_idx], np.zeros((0, 3), dtype=int))
    return verts[s_idx]


def augment_verts(verts, n_samps, n_augs = 50, rotate=True, min_scale=0.3):  
    '''Sample, apply random rotation, scale down to unit or smaller, and move to random spot'''
    if hasattr(verts, 'vertices'): 
        verts = verts.vertices
    if n_samps is None: 
        out = np.array([verts for _ in range(n_augs)])
    else:
        out = np.array([sample_pointcloud(verts, n_samps) for _ in range(n_augs)])
    for i in range(n_augs):
        if rotate: 
            R = scipy_tf.Rotation.random().as_matrix()
            out[i] = np.matmul(R, out[i].transpose()).transpose()
        D_min = np.amin(out[i], axis=0)
        D_max = np.amax(out[i], axis=0)
        aug_scale = min_scale + np.random.random() * (1 - min_scale)
        out[i] = aug_scale * (out[i] - D_min) / (D_max - D_min)
        out[i] += (1 - aug_scale) * np.random.random(3)
    return out
