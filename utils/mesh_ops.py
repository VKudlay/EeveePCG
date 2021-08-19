'''
File specifying various operations on mesh and vertice-level inputs.
'''

import pymesh
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.spatial.transform as scipy_tf

#######################################################################################
## Plotting Utilities

def plot_mesh_verts(mesh, ax, va = 0.4, fa = 0.8, cmap=None):
    '''Plot mesh alongside vertices through mpl'''
    if not hasattr(mesh, 'vertices'): 
        plot_pointfield(mesh, ax, a = va, cmap=cmap)
    vertices = [[v[0], v[1], v[2]] for v in mesh.vertices]
    polyset = [[vertices[vi] for vi in face] for face in mesh.faces]
    if fa: ax.add_collection3d(Poly3DCollection(polyset, fc='orange', linewidths=1, alpha=fa))
    if va: ax.scatter3D(*zip(*vertices), s=0.1, alpha=va, cmap=cmap)


def plot_pointfield(verts, ax, a=1, cmap='Greens', c=None):
    '''Plot pointfield in mpl'''
    if hasattr(verts, 'vertices'): verts = verts.vertices
    x = np.array([p[0] for p in verts])
    y = np.array([p[1] for p in verts])
    z = np.array([p[2] for p in verts])
    if c is None: c = z
    ax.scatter3D(x, y, z, alpha=a, c=c, cmap=cmap)

#######################################################################################
## Converting From Mesh-Like

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

#######################################################################################
## Point Cloud Samplers

def sample_pointcloud(mesh_like, n_samples=2**10):
    '''
    Samples point cloud using barycentric coordinates if mesh-type (i.e. w/ faces).
    Otherwise, just go ahead and try to sample vertices directly (default behavior).
    '''
    if hasattr(mesh_like, 'vertices'):
        return sample_pointcloud_mesh(mesh_like, n_samples)
    else: 
        print("Depricated: sample_pointcloud defaulting to vertex sampling")
        return sample_pointcloud_verts(mesh_like, n_samples)

def sample_pointcloud_verts(verts, n_samples=2**10):
    '''Get a sample of n points from a point cloud (or set of vertices)'''
    return_mesh = hasattr(verts, 'vertices') 
    if return_mesh: verts = verts.vertices
    s_idx = np.random.choice(verts.shape[0], n_samples, replace=False)
    if return_mesh: return pymesh.form_mesh(verts[s_idx], np.zeros((0, 3), dtype=int))
    return verts[s_idx]

def sample_pointcloud_mesh(mesh, n_samples=2**10):
    """
    Samples point cloud on the surface of the model defined as vectices and faces. 
    This function uses vectorized operations so fast at the cost of some memory.
    
    Source: https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    """
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise ValueError("sample_faces(...) requires mesh, not vertices") 
    vertices = mesh.vertices
    faces = mesh.faces
    
    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :]
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples_new = np.sum(n_samples_per_face) # What the heck?

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples_new, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples_new, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = A * (1 - np.sqrt(r[:,0:1])) \
      + B * np.sqrt(r[:,0:1]) * (1 - r[:,1:]) \
      + C * np.sqrt(r[:,0:1]) * r[:,1:]
    P = P[np.random.choice(P.shape[0], n_samples, replace=False), :]
    return P

#######################################################################################
## Data Augmentation Function

def augment_generic(sampler, mesh_like, n_samps, n_augs = 50, rotate=True, min_scale=0.3):  
    '''Sample, apply random rotation, scale down to unit or smaller, and move to random spot'''
    if hasattr(mesh_like, 'vertices'): 
        verts = mesh_like.vertices
    # Sample n_samp points. Will return all vertices if n_samp unspecified.
    if n_samps is None: 
        out = np.array([verts for _ in range(n_augs)])
    else:
        out = np.array([sampler(mesh_like, n_samps) for _ in range(n_augs)])
    for i in range(n_augs):
        if rotate: 
            # Apply uniform random rotation
            R = scipy_tf.Rotation.random().as_matrix()
            out[i] = np.matmul(R, out[i].transpose()).transpose()
        # Scale Down To (Sub)-Unit (Hyper)-Cube
        D_min = np.amin(out[i], axis=0)
        D_max = np.amax(out[i], axis=0)
        aug_scale = min_scale + np.random.random() * (1 - min_scale)
        out[i] = aug_scale * (out[i] - D_min) / (D_max - D_min)
        out[i] += (1 - aug_scale) * np.random.random(len(D_min))
    return out

def augment_verts(verts, n_samps, n_augs = 50, rotate=True, min_scale=0.3):  
    '''Augmenter that uses vertex sampling (i.e. w/ voxels)'''
    return augment_generic(sample_pointcloud_verts, verts, n_samps, n_augs, rotate, min_scale)

def augment_mesh(mesh, n_samps, n_augs = 50, rotate=True, min_scale=0.3):  
    '''Augmenter that uses mesh sampling (i.e. w/ barycentric sampling)'''
    return augment_generic(sample_pointcloud_mesh, mesh, n_samps, n_augs, rotate, min_scale)