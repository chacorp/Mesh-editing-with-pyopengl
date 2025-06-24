from __future__ import print_function

import sys
import os
from os.path import exists, join
# from glob import glob
import numpy as np

import scipy
from scipy.sparse import diags, coo_matrix, csr_matrix

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from PIL import Image
import glm

# from testwindow import *
from imgui.integrations.glfw import GlfwRenderer
import imgui

import torch
import igl

# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Uniform Laplacian matrix
# 2) Cotangent Laplacian matrix
# -------------------------------------------------------------------- #

def laplacian_and_adjacency(verts_N: torch.Tensor, edges: torch.Tensor):
    """
    Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts_N: number of vertices (N) of the mesh (N, 3)
        edges:   tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
        A: Adjacency matrix (V, V)
    """
    V = verts_N
    E = edges.numpy()
    
    idx01 = np.stack([E[:,0], E[:,1]], axis=1)  # (E, 2)
    idx10 = np.stack([E[:,1], E[:,0]], axis=1)  # (E, 2)
    idx = np.r_[idx01, idx10].T  # (2, 2*E)
    
    ones = np.ones(idx.shape[1])
    A  = scipy.sparse.csr_matrix((ones, idx), shape=(V, V))
    
    degree = np.asarray(A.sum(axis=1)).squeeze()
    I_ = scipy.sparse.identity(V)
    D_ = diags(1/degree) @ A
    L = I_ - D_
    
    return L, A

def laplacian_cotangent(verts, faces, eps=1e-12):
    """
    Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =         -1          , if i == j
    L[i, j] = cot a_ij + cot b_ij , iff (i, j) is an edge in meshes.
    L[i, j] =          0          , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.
    
    Args:
        verts: (V, 3) containing the vertices of the graph
        faces: (F, 3) containing the vertex indices of each face
    Returns:
        2-element tuple containing
        - L:  Laplacian matrix.
        - inv_areas: (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    
    V = verts.shape[0]
    F = faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = np.linalg.norm(v1 - v2, axis=1)
    B = np.linalg.norm(v0 - v2, axis=1)
    C = np.linalg.norm(v0 - v1, axis=1)

    s = 0.5 * (A + B + C)
    area = np.sqrt(np.clip(s * (s - A) * (s - B) * (s - C), eps, np.inf))

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = np.stack([cota, cotb, cotc], axis=1) / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    
    # import pdb;pdb.set_trace()
    idx = np.stack([ii, jj], axis=0).reshape(2, F * 3)
    L = scipy.sparse.csr_matrix((cot.reshape(-1), idx), shape=(V, V))
    # L = coo_matrix((Vals, (I, J)), shape=(V, V))
    
    
    # Construct sparse matrix L
    # L = (L + L.T).tocsr()  # Make symmetric
    # i_is_j = np.asarray(L.sum(axis=1)).squeeze()
    # L = scipy.sparse.diags(i_is_j) - L
    
    # Compute inverse area per vertex
    idx = faces.reshape(-1)
    val = np.tile(area, (3,)).reshape(-1)
    inv_areas = np.zeros(V, dtype=np.float64)
    np.add.at(inv_areas, idx, val)
    nonzero = inv_areas > 0
    inv_areas[nonzero] = 1.0 / inv_areas[nonzero]
    inv_areas = inv_areas[:, None]

    return L, inv_areas

def laplacian_matrix_ring(mesh):
    """
    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    
    N = mesh.v.shape[0]
    
    ## rotations
    RS = np.zeros([N, 3, 3])
    for i in range(N):
        
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        
        wij = mesh.L[i].data[1:]
        D_ring = np.diag(wij)
        
        # import pdb;pdb.set_trace()
        E_ring       = (mesh.v[i] - mesh.v[ring]).T # (3, N)
        E_prime_ring = (mesh.v_prime[i] - mesh.v_prime[ring]).T # (3, N)
        
        # (3, N) x (N, N) x (N, 3)
        S_i = E_ring @ D_ring @ E_prime_ring.T
        # S_i = E_ring @ E_prime_ring.T
        
        u_i, s_i, vt_i = np.linalg.svd(S_i)
        
        R_i = vt_i.T @ u_i.T
        
        if np.linalg.det(R_i) < 0:
            vt_i[-1, :] *= -1
            R_i = vt_i.T @ u_i.T
            
        RS[i] = R_i
        # RS[i] = np.eye(3)
    
        
    # laplacian matrix for rhs
    # LS = mesh.LS.copy()
    LS = np.zeros([3*N, 3*N])
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        n_ring = len(ring)
        
        
        disk_idxs = np.hstack([disk, disk+N, disk+2*N])
        # ring_idxs = np.hstack([ring, ring+N, ring+2*N])
        # i_idxs = np.hstack([i, i+N, i+2*N])
        
        # wij = 0.5 * mesh.LS[[i, i+N, i+2*N]][:, ring_idxs].reshape(3, 3, n_ring).transpose(2,0,1)
        wij = 0.5 * mesh.L[i, ring].data[:,None,None]
        
        # wij/2 * (Ri + Rj)(pi - pj)
        wij_Ri_Rj = wij * (RS[i][None] + RS[ring])
        
        wij_Ri = wij_Ri_Rj.sum(0)
        wij_Ri_Rj_all = np.vstack([-wij_Ri[None], wij_Ri_Rj])
        
        wij_Ri_Rj_all_t = wij_Ri_Rj_all.transpose(1,2,0) # (disk, 3, 3) | disk:{i, j0, j1, j2,...jn}
        
        LS[i,     disk_idxs] += wij_Ri_Rj_all_t[0].reshape(-1)
        LS[i+N,   disk_idxs] += wij_Ri_Rj_all_t[1].reshape(-1)
        LS[i+2*N, disk_idxs] += wij_Ri_Rj_all_t[2].reshape(-1)
        
    return LS

def _laplacian_matrix_ring(mesh):
    """
    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    
    N = mesh.v.shape[0]
    
    ## rotations
    RS = np.zeros([N, 3, 3])
    for i in range(N):
        
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        
        wij = mesh.L[i].data[1:]
        D_ring = np.diag(wij)
        
        import pdb;pdb.set_trace()
        E_ring       = (mesh.v[i] - mesh.v[ring]).T # (3, N)
        E_prime_ring = (mesh.v_prime[i] - mesh.v_prime[ring]).T # (3, N)
        
        # (3, N) x (N, N) x (N, 3)
        # S_i = E_ring @ D_ring @ E_prime_ring.T
        S_i = E_ring @ E_prime_ring.T
        
        u_i, s_i, vt_i = np.linalg.svd(S_i)
        
        R_i = vt_i.T @ u_i.T
        
        if np.linalg.det(R_i) < 0:
            vt_i[-1, :] *= -1
            R_i = vt_i.T @ u_i.T
            
        RS[i] = R_i
        # RS[i] = np.eye(3)
    
        
    # laplacian matrix for rhs
    # LS = mesh.LS.copy()
    LS = np.zeros([3*N, 3*N])
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        n_ring = len(ring)
        # Ri(pi - pj)
        # LS[[i, i+N, i+2*N]][:, rind_idxs] += R_ring.repeat(n_ring, 1)
        
        # wij/2 * (Ri + Rj)(pi - pj)
        disk_idxs = np.hstack([disk, disk+N, disk+2*N])
        ring_idxs = np.hstack([ring, ring+N, ring+2*N])
        i_idxs = np.hstack([i, i+N, i+2*N])
        
        # LS[[i, i+N, i+2*N]][:, i_idxs] += RS[i] * mesh.LS[[i, i+N, i+2*N]][:, i_idxs] * 0.5
        
        # wii = -0.5 * mesh.LS[[i, i+N, i+2*N]][:, i_idxs] # multiply -1 because it is already negative! (-wii == wij)
        # wij = 0.5 *mesh.LS[[i, i+N, i+2*N]][:, ring_idxs].reshape(3, 3, n_ring).transpose(2,0,1)
        # wii = -0.5 * mesh.L[i,i]
        wij = 0.5 * mesh.L[i, ring].data[:,None,None]
        
        # wij_Rj = (mesh.L[i,ring].data[:,None,None] * RS[ring]).sum(0)
        
        # wii_Ri = wii * RS[i]
        # wij_Rj = (wij * RS[ring]).sum(0)
        
        # wij_Ri = 0.5 * (wii_Ri + wij_Rj)
        # wij_Ri = 0.5 * (wii * RS[i] + (wij * RS[ring]).sum(0))
        
        # wij_Ri_Rj = 0.5 * mesh.L[i, ring].data[:,None,None] * (RS[i][None] + RS[ring])
        wij_Ri_Rj = wij * (RS[i][None] + RS[ring])
        wij_Ri = wij_Ri_Rj.sum(0)
        wij_Ri_Rj_all = np.vstack([-wij_Ri[None], wij_Ri_Rj])
        wij_Ri_Rj_all_t = wij_Ri_Rj_all.transpose(1,2,0) # (disk, 3, 3) | disk:{i, j0, j1, j2,...jn}
        
        LS[i,     disk_idxs] += wij_Ri_Rj_all_t[0].reshape(-1)
        LS[i+N,   disk_idxs] += wij_Ri_Rj_all_t[1].reshape(-1)
        LS[i+2*N, disk_idxs] += wij_Ri_Rj_all_t[2].reshape(-1)
        
        # LS[i,     i_idxs] += wij_Ri[0]
        # LS[i+N,   i_idxs] += wij_Ri[1]
        # LS[i+2*N, i_idxs] += wij_Ri[2]
        
        # # for j in range(n_ring):
        # #     jdx = mesh.ring_indices[i][j]
        # #     j_idxs = np.hstack([jdx, jdx+N, jdx+2*N])
        # #     LS[i,     j_idxs] += wij_Ri_Rj[j,0]
        # #     LS[i+N,   j_idxs] += wij_Ri_Rj[j,1]
        # #     LS[i+2*N, j_idxs] += wij_Ri_Rj[j,2]
        # import pdb;pdb.set_trace()
        # wij_Ri_Rj_ = wij_Ri_Rj.transpose(1,2,0)
        # LS[i,     ring_idxs] -= wij_Ri_Rj_[0].reshape(-1)
        # LS[i+N,   ring_idxs] -= wij_Ri_Rj_[1].reshape(-1)
        # LS[i+2*N, ring_idxs] -= wij_Ri_Rj_[2].reshape(-1)
        
        
        
        # LS[i,     i_idxs] += (RS[i, 0]*mesh.LS[i,     i_idxs] + (RS[ring, 0] * mesh.LS[i,     ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        # LS[i+N,   i_idxs] += (RS[i, 1]*mesh.LS[i+N,   i_idxs] + (RS[ring, 1] * mesh.LS[i+N,   ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        # LS[i+2*N, i_idxs] += (RS[i, 2]*mesh.LS[i+2*N, i_idxs] + (RS[ring, 2] * mesh.LS[i+2*N, ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        
        # # RS[ring, 0]
        # # RS[ring].transpose(1,2,0).reshape(3, -1)
        # # LS[[i, i+N, i+2*N]][:, ring_idxs] += RS[ring].transpose(1,2,0).reshape(3, -1) * mesh.LS[[i, i+N, i+2*N]][:, ring_idxs] * 0.5
        
        # LS[i,     ring_idxs] -= (RS[i, 0] + RS[ring, 0]).transpose(1,0).reshape(-1) * mesh.LS[i,     ring_idxs] * 0.5
        # LS[i+N,   ring_idxs] -= (RS[i, 1] + RS[ring, 1]).transpose(1,0).reshape(-1) * mesh.LS[i+N,   ring_idxs] * 0.5
        # LS[i+2*N, ring_idxs] -= (RS[i, 2] + RS[ring, 2]).transpose(1,0).reshape(-1) * mesh.LS[i+2*N, ring_idxs] * 0.5
        
        # LS[i,     ring_idxs] += R_ring[0]
        # LS[i+N,   ring_idxs] += R_ring[1]
        # LS[i+2*N, ring_idxs] += R_ring[2]
    
    # LS = LS * mesh.LS * 0.5
    # LS[np.arange(N),np.arange(N)]=0
    # import pdb;pdb.set_trace()
    # L = mesh.L.todense()
    
    return LS


def get_constraints(mesh, mask, handle_idx, handle_pos, Wb=1.0):
    
    N  = mesh.v.shape[0]
    
    len_H = len(handle_idx)
    constraint_coef = np.zeros([N + len_H, N])
    constraint_b    = np.zeros([N + len_H, 3])
    
    # Boundary constraints
    i = 0
    h = 0
    
    # Boundary constraints
    for vidx, val in enumerate(mask): # num: N
        # i: index of Boundary constraint
        # idx: index of v in V
                
        if val == 0:
            constraint_coef[i, vidx] = 1.0 * Wb # one-hot
            constraint_b[i]         = mesh.v[vidx] * Wb # fixed position
        i = i + 1
        
    # Handle constraints
    for vidx in handle_idx: # num: len_H
        # i: index of Handle constraint
        # idx: index of v in V
        
        constraint_coef[i, vidx] = 1.0 * Wb # one-hot
        constraint_b[i]         = handle_pos[h] * Wb
        
        i = i + 1
        h = h + 1
        
    return constraint_coef, constraint_b

def get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb=1.0):
    N  = mesh.v.shape[0]
    
    len_H = len(handle_idx)
    constraint_coef = np.zeros([3*N + 3*len_H, 3*N])
    constraint_b    = np.zeros([3*N + 3*len_H])
    
    i = 0
    h = 0
    
    # Boundary constraints + outside ROI
    for vidx, val in enumerate(mask): # num: N
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Boundary constraint
        idx_v = [vidx, vidx+N, vidx+2*N] # index of v in V
        
        if val == 0:
            constraint_coef[idx_c, idx_v] = 1.0 * Wb # one-hot
            constraint_b[idx_c]           = mesh.v[vidx] * Wb # fixed position
        i = i + 1
        
    # Handle constraints
    for vidx in handle_idx: # num: len_H
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Handle constraint
        idx_v = [vidx, vidx+N, vidx+2*N] # index of v in V
        
        constraint_coef[idx_c, idx_v] = 1.0 * Wb # one-hot
        constraint_b[idx_c]           = handle_pos[h] * Wb
        
        i = i + 1
        h = h + 1
    return constraint_coef, constraint_b

def laplacian_surface_editing(mesh, mask, boundary_idx, handle_idx, handle_pos, Wb=1.0):
    """
    Args:
        mesh (class):        mesh properties
        boundary_idx (list):    vertex indices of boundary
        handle_idx (list):      vertex indices of handle
        handle_pos (np.array):  displacement of handle
        mask (np.array):  vertices outside boundary (V, 1)

    Returns:
        new_verts (np.array):   new vertices after laplacian surface editing
    """
    
    N  = mesh.v.shape[0]
    LS = mesh.LS
    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef, constraint_b = get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb)
    # -------------------------- Solve the Linear System ------------------------- #
    A        = np.vstack([LS,         constraint_coef])
    b        = np.hstack([np.zeros(3*N), constraint_b])
    # A        = LS
    # b        = np.zeros(3*N)
    spA      = scipy.sparse.coo_matrix(A)
        
    V_prime  = scipy.sparse.linalg.lsqr(spA, b)[0]
    
    new_verts = V_prime.reshape(3, -1).T
    return new_verts

def as_rigid_as_possible_surface_modeling(mesh, mask, boundary_idx, handle_idx, handle_pos, Wb=1.0, iteration=2):
    """
    Args:
        mesh (class):        mesh properties
        boundary_idx (list):    vertex indices of boundary
        handle_idx (list):      vertex indices of handle
        handle_pos (np.array):  displacement of handle
        mask (np.array):  vertices outside boundary (V, 1)

    Returns:
        new_verts (np.array):   new vertices after laplacian surface editing
    """
    mesh.v = mesh.orig_v
    N  = mesh.v.shape[0]
    
    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef, constraint_b = get_constraints(mesh, mask, handle_idx, handle_pos, Wb)
    # constraint_coef_3N, constraint_b_3N = get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb)
    
    # -------------------------- Solve the Linear System ------------------------- #
    # A = np.vstack([mesh.L.todense()])
    # b = np.vstack([mesh.delta])
    A = np.vstack([mesh.L.todense(), constraint_coef])
    b = np.vstack([mesh.delta,          constraint_b])
    # b = np.vstack([np.zeros((N,3)), constraint_b])
    
    ATA = scipy.sparse.coo_matrix(A.T @ A)
    lu = scipy.sparse.linalg.splu(ATA.tocsc())
    
    ATb = A.T @ b
    v_prime = lu.solve(ATb)
    v_prime = np.asarray(v_prime)
    
    mesh.v_prime = v_prime
    
    for it in range(iteration):
        # rhs = mesh.L @ mesh.v
        # import pdb;pdb.set_trace()
        
        # rhs = mesh.LS @ mesh.v.transpose(1,0).reshape(-1)
        rhs = laplacian_matrix_ring(mesh) @ mesh.v.transpose(1,0).reshape(-1)
        rhs = rhs.reshape(3,-1).transpose(1,0)
        b = np.vstack([rhs, constraint_b])
        
        ATb = A.T @ b
        v_prime = lu.solve(ATb)
                
        v_prime = np.asarray(v_prime)
        mesh.v_prime = v_prime.copy()
                
    return mesh.v_prime

def load_texture(path):
    """
    Args:
        path (str): image path

    Returns:
        texture (int): texture id for the image
    """
    texture = glGenTextures(1)
    print("texture buffer: ",texture)
    glBindTexture(GL_TEXTURE_2D, texture)
    image = Image.open(path)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB') # 'RGBA
    image_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    return texture

def rotation(M, angle, x, y, z):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    n = np.sqrt(x*x + y*y + z*z)
    x,y,z = x/n, y/n, z/n
    cx,cy,cz = (1-c)*x, (1-c)*y, (1-c)*z
    R = np.array([[cx*x + c,   cy*x - z*s, cz*x + y*s, 0.0],
                  [cx*y + z*s, cy*y + c,   cz*y - x*s, 0.0],
                  [cx*z - y*s, cy*z + x*s, cz*z + c,   0.0],
                  [0.0,        0.0,        0.0,        1.0]], dtype=M.dtype)

    return np.dot(M, R.T)

def y_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[   c,       0.0,          s,        0.0],
                  [ 0.0,       1.0,        0.0,        0.0],
                  [  -s,       0.0,          c,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def z_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[   c,        -s,        0.0,        0.0],
                  [   s,         c,        0.0,        0.0],
                  [ 0.0,       0.0,        1.0,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def x_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[ 1.0,       0.0,        0.0,        0.0],
                  [ 0.0,         c,         -s,        0.0],
                  [ 0.0,         s,          c,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def normalize_np(V):
    ### numpy
    V = np.array(V)
    V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    return V

def normalize_torch(V):
    ### torch
    V = (V-(V.max(0).values + V.min(0).values) * 0.5)/max(V.max(0).values - V.min(0).values)
    return V

def recurr_adj(mask_v, adj_mat, idx, boundary_idx):
    if mask_v[idx] == 1:
        return
    
    mask_v[idx] = 1
    for jdx in adj_mat[idx].indices:
        if not jdx in boundary_idx:
            recurr_adj(mask_v, adj_mat, jdx, boundary_idx)
    return

class Mesh_container():
    def __init__(self, mesh_path, boundary_idx, handle_idx):
        self.load_obj_mesh(mesh_path)
        self.compute_edges()
        
        self.L, self.Adj            = laplacian_and_adjacency(self.v.shape[0], self.e)
        
        ## cotangent laplacian        
        # something gone wrong...!
        # self.L, self.inv_A  = laplacian_cotangent(self.v, self.f, eps=1e-12)
        # using igl library
        igl_cot_l = igl.cotmatrix(self.v, self.f)
        self.L = igl_cot_l
        
        self.orig_v = self.v.copy()
        
        self.delta          = self.L @ self.v
        
        self.ring_indices   = [self.Adj[i].indices.tolist() for i in range(self.v.shape[0])]
        
        self.mask_v = np.zeros([self.v.shape[0],1]).astype(int)
        self.mask_v[boundary_idx] = 1
        for h_idx in handle_idx:
            recurr_adj(self.mask_v, self.Adj, h_idx, boundary_idx)
        self.mask_v[boundary_idx] = 0
        # print(np.where(self.mask_v>0)[0])
        
        N = self.v.shape[0]
        self.LS = np.zeros([3*N, 3*N])
        self.LS[0*N:1*N, 0*N:1*N] = self.L.todense()
        self.LS[1*N:2*N, 1*N:2*N] = self.L.todense()
        self.LS[2*N:3*N, 2*N:3*N] = self.L.todense()
        
    
    def compute_edges(self):
        """
        Computes edges in packed form from the packed version of faces and verts.
        reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html#Meshes.edges_packed
        """
        
        faces = torch.tensor(self.f)
        v0, v1, v2 = faces.chunk(3, dim=1)
        e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
        e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
        e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

        # All edges including duplicates.
        edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
        
        # rows in edges after sorting will be of the form (v0, v1) where v1 > v0.
        edges, _ = edges.sort(dim=1)

        # Remove duplicate edges: convert each edge (v0, v1) into an
        # integer hash = V * v0 + v1; this is much faster than edges.unique(dim=1)
        # After finding the unique elements reconstruct the vertex indices as:
        # (v0, v1) = (hash / V, hash % V)
        V = self.v.shape[0]
        edges_hash = V * edges[:, 0] + edges[:, 1]
        uqe, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
        
        uqe_V = uqe.div(V, rounding_mode='floor')
        self.e = torch.stack([uqe_V, uqe % V], dim=1)

    def load_obj_mesh(self, mesh_path):
        vertex_data = []
        vertex_normal = []
        vertex_texture = []
        face_data = []
        face_texture = []
        face_normal = []
        for line in open(mesh_path, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:]))
                vertex_data.append(v)
            if values[0] == 'vn':
                vn = list(map(float, values[1:]))
                vertex_normal.append(vn)
            if values[0] == 'vt':
                vt = list(map(float, values[1:]))
                vertex_texture.append(vt)
            if values[0] == 'f':
                f = list(map(lambda x: int(x.split('/')[0]),  values[1:]))
                face_data.append(f)
                if len(values[1].split('/')) >=2:
                    ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
                    face_texture.append(ft)
                if len(values[1].split('/')) >=3:
                    ft = list(map(lambda x: int(x.split('/')[2]),  values[1:]))
                    face_normal.append(ft)
        
        self.v  = normalize_np(
            np.array(vertex_data)
        )
        self.vn = np.array(vertex_normal)
        self.vt = np.array(vertex_texture)
        self.f  = np.array(face_data)
        self.ft = np.array(face_texture)
        self.fn = np.array(face_normal)
        if self.f.min() > 0:
            self.f  = self.f  - 1
            self.ft = self.ft - 1
            self.fn = self.fn - 1
        
    
def main(resolution=512,
        boundary_idx = [3, 13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 223, 238, 253, 268, 283, 298, 313, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464],
        # boundary_idx = [8, 21, 36, 51, 66, 81, 96, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 261, 276, 291, 306, 321, 337, 352, 367, 382, 397, 412, 427, 442, 457, 472],
        # handle_idx   = [0, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 326, 341, 356, 371, 386, 401, 416, 431, 446, 461],
        handle_idx   = [325],
         ):
    ### sphere mesh
    # path                = "data/sphere_trimesh.obj"
    path                = "data/sphere.obj"
    
    ## for loading face and other stuff: vt, vn, ft, fn ...
    mesh = Mesh_container(path, boundary_idx=boundary_idx, handle_idx=handle_idx)
    mask_v = mesh.mask_v
    image_path  = "checkerboard.png"

    rendered    = render(mesh, resolution, image_path, boundary_idx=boundary_idx, handle_idx=handle_idx, mask_v=mask_v)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('LSE')
    if not exists(savefolder):
        os.makedirs(savefolder)
    
    savefile    = join(savefolder, 'experiemtn.png')

    Image.fromarray(rendered).save(savefile)
    return
 
def render(mesh, 
         resolution, 
         image_path, 
         disp_path=None, 
         json_object=None, 
         boundary_mask_path=None, 
         timer=False, 
         boundary_idx=[],
         handle_idx=[],
         mask_v=None,
         mean=None, 
         coef=None, 
         basis=None
         ):
    if timer == False:
        import time
        start = time.time()
    
    v, f, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    
    if not glfw.init():
        return

    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    
    # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(resolution, resolution, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    new_vt = vt[ft].reshape(-1,2)
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    new_v  = v[f].reshape(-1, 3)
    
    # if f.max() == vn.shape[0]:
    if True:
        new_vn = vn[mesh.fn].reshape(-1, 3)
    else:
        new_vn = vn[f].reshape(-1, 3)
    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    quad = np.array(quad, dtype=np.float32)

    ############################################## shader ################
    vertex_shader_source   = open('shader.vs', 'r').read()
    fragment_shader_source = open('shader.fs', 'r').read()
    
    vertex_shader   = shaders.compileShader(vertex_shader_source,   GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader          = shaders.compileProgram(vertex_shader, fragment_shader)
    ############################################## buffer ################

    # VAO = glGenBuffers(1)
    # glBindVertexArray(VAO)


    # EBO = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*indices.shape[0]*indices.shape[1], indices, GL_STATIC_DRAW)
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
    """
        GL_STREAM_DRAW:  the data is set only once and used by the GPU at most a few times.
        GL_STATIC_DRAW:  the data is set only once and used many times.
        GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    """
    
    # 4*3*3 : size of float * len(X,Y,Z) * len(pos, tex, nor)
    vertex_stride = 4 * quad.shape[1]
    
    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    
    texcoord = glGetAttribLocation(shader, "texcoord")
    glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(texcoord)

    normal = glGetAttribLocation(shader, "normal")
    glVertexAttribPointer(normal,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(normal)

    ############################################## texture map ###########
    
    glUseProgram(shader)
                
    texture0 = load_texture(image_path)
    glUniform1i(glGetUniformLocation(shader, "texture0"), 0)
    if disp_path:
        texture1 = load_texture(disp_path)
        glUniform1i(glGetUniformLocation(shader, "texture1"), 1)
    if boundary_mask_path:
        textureB = load_texture(boundary_mask_path)
    ############################################## uniform ###############
    i = 0
    rotation_ = 0
    
    H_rot_x = 0
    H_rot_y = 0
    H_rot_z = 0
        
    scaleX = 1.0
    scaleY = 1.0
    scaleZ = 1.0
    
    transX = 0.0
    transY = 0.0
    transZ = -1.0
    
    Reset_button = False
    
    iteration_ = 1
    
    tex_alpha = 1.0
    H_r_mat = np.eye(4)
    handle_pos_old = np.array([0.0, 0.0, 0.0])
    handle_pos_new = np.array([0.0, 0.0, 0.0])
    # handle_pos_new = new_v[handle_idx]
    handle_pos = mesh.v[handle_idx]
    handle_change = False
    use_LSE = True
    normalize = False
    _ratio = 1.0
    
    if json_object:
        d_range = json_object['d_range']
        d_range_0 = np.zeros_like(d_range)
        gl_d_range = glGetUniformLocation(shader, "d_range")
        use_disp = False
        # use_disp = True
    
    onoff_tex = True
    
    scale     = glGetUniformLocation(shader, "scale") # scale rotate translate
    transform = glGetUniformLocation(shader, "transform")
    translate = glGetUniformLocation(shader, "trans")
        
    glView    = glGetUniformLocation(shader, "proj")
    gl_alpha  = glGetUniformLocation(shader, "_alpha")
    
    # view = glm.ortho(-1.0, 2.0, -1.0, 1.0, 0.0001, 1000.0) 
    zoom = 1.0
    view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.000001, 100.0) 
    glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
    ############################################## uniform ###############

    ############################################## gl setting ############
    # glBindVertexArray(VAO)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND);
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    # glEnable(GL_CULL_FACE);
    # glFrontFace(GL_CCW); 
    ############################################## gl setting ############

    ############################################## imgui #################
    use_imgui = True
    if use_imgui:
        imgui.create_context()
        # window = impl_glfw_init() # already initialized!
        impl = GlfwRenderer(window)
    ############################################## imgui init ############    
        
    while not glfw.window_should_close(window):
        
        if onoff_tex:
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture0)
        else:
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureB)
        
        if json_object:
            if use_disp:
                glUniform3fv(gl_d_range, 1, d_range*100)
            else:
                glUniform3fv(gl_d_range, 1, d_range_0)
            
            if disp_path:
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, texture1)
            
        ### trans * rotate * scale * 3D model
        rotation_mat = y_rotation(rotation_)
        affine_mat = np.eye(4)
        
        affine_mat[:3,:3] = affine_mat[:3,:3] * np.array([scaleX, scaleY, scaleZ])
        trans = np.array([transX, transY, transZ])
        rotation_mat = rotation_mat @  affine_mat.T
        
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        glUniform3fv(translate, 1, trans)
        
        view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
        glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
        
        ### Laplacian Surface Editing
        if handle_change:
            handle_pos_mean = handle_pos.mean(0)
            handle_pos_new_ = (handle_pos-handle_pos_mean) @ H_r_mat[:3,:3]
            handle_pos_new_ = handle_pos_new_ + handle_pos_mean + handle_pos_new.copy()
            
            if use_LSE:
                new_v = as_rigid_as_possible_surface_modeling(
                    mesh, mask_v, boundary_idx=boundary_idx, handle_idx=handle_idx, handle_pos=handle_pos_new_,
                    iteration=iteration_
                    )
            else:
                new_v = mesh.orig_v
                new_v[handle_idx] = handle_pos_new_
            
            new_v = new_v[f].reshape(-1, 3)
            handle_change = False
        
        upd_v = new_v
        quad[:, :3] = upd_v
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.poll_events()
        if use_imgui:
            impl.process_inputs()
            imgui.new_frame()
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):
                    clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)
                    if clicked_quit:
                        sys.exit(0)
                    imgui.end_menu()
                imgui.end_main_menu_bar()
            
            imgui.set_next_window_position(resolution, 20)
            # imgui.text("mean vert: {}".format(upd_v.mean(0)))
            if json_object:
                imgui.text("d_range: {}".format(d_range))
                
            if iteration_ < 1:
                imgui.text("initial")
            else:
                imgui.text(f"iteration: {iteration_}")
                
            imgui.text("handle_pos[0]: {}".format(handle_pos[0]))
            imgui.text("handle_idx: {}".format(handle_idx))
            imgui.text("renew[0]: {}".format(mesh.v[0]+handle_pos_new))
            # imgui.text("handle_pos_old: {}".format(handle_pos_old))
            
            clicked, tex_alpha  = imgui.slider_float(label="_alpha",    value=tex_alpha, min_value=0.0, max_value=1.0)
            clicked, _ratio     = imgui.slider_float(label="_ratio",    value=_ratio, min_value=0.0, max_value=1.0)
            clicked, rotation_  = imgui.slider_float(label="Rotate", value=rotation_, min_value=0.0, max_value=360.0,)
            
            clicked, scaleX     = imgui.slider_float(label="Scale x",   value=scaleX, min_value= 0.0,  max_value= 10.0,)
            changed, scaleX     = imgui.input_float(label="set Scale x",value=scaleX, step=0.001)
            clicked, scaleY     = imgui.slider_float(label="Scale y",   value=scaleY, min_value= 0.0,  max_value= 10.0,)
            changed, scaleY     = imgui.input_float(label="set Scale y",value=scaleY, step=0.001)
            clicked, scaleZ     = imgui.slider_float(label="Scale z",   value=scaleZ, min_value= 0.0,  max_value= 10.0,)
            
            clicked, transX     = imgui.slider_float(label="Trans x",   value=transX, min_value=-5.0,  max_value= 5.0,)
            changed, transX     = imgui.input_float(label="set Trans x",value=transX, step=0.001)
            clicked, transY     = imgui.slider_float(label="Trans y",   value=transY, min_value=-5.0,  max_value= 5.0,)
            changed, transY     = imgui.input_float(label="set Trans y",value=transY, step=0.001)
            clicked, transZ     = imgui.slider_float(label="Trans z",   value=transZ, min_value=-100,  max_value= 10,)
            changed, transZ     = imgui.input_float(label="set Trans z",value=transZ, step=0.001)
            
            iter_changed, iteration_ = imgui.input_int(label="ARAP iteration", value=iteration_, step=1)
            if iteration_ < 0:
                iteration_ = 0
                iter_changed = False
            # changed, iteration_     = imgui.input_int(str label, int value, int step=1, int step_fast=100, ImGuiInputTextFlags flags=0)
            
            h_c_0, handle_pos_new[0] = imgui.slider_float(label="Handle x",   value=handle_pos_new[0], min_value=-2,  max_value= 2,)
            h_c_1, handle_pos_new[1] = imgui.slider_float(label="Handle y",   value=handle_pos_new[1], min_value=-2,  max_value= 2,)
            h_c_2, handle_pos_new[2] = imgui.slider_float(label="Handle z",   value=handle_pos_new[2], min_value=-2,  max_value= 2,)
            
            h_r_0, H_rot_x  = imgui.slider_float(label="Handle Rotate x", value=H_rot_x, min_value=0.0, max_value=360.0,)
            h_r_1, H_rot_y  = imgui.slider_float(label="Handle Rotate y", value=H_rot_y, min_value=0.0, max_value=360.0,)
            h_r_2, H_rot_z  = imgui.slider_float(label="Handle Rotate z", value=H_rot_z, min_value=0.0, max_value=360.0,)
            
            
            
            
            clicked, zoom       = imgui.slider_float(label="Zoom", value=zoom, min_value=0.1,  max_value= 2.0,)
            changed, Reset_button = imgui.menu_item("Reset", None, Reset_button)
            
            LSE_changed, use_LSE = imgui.menu_item("Use Laplacian Surface Editing", None, use_LSE)
            
            
            if json_object:
                clicked_use_disp, use_disp   = imgui.menu_item("Use_disp", None, use_disp)
                clicked_use_tex, onoff_tex   = imgui.menu_item("OnOff Tex", None, onoff_tex)

                if clicked_use_disp:
                    use_disp != use_disp
                if clicked_use_tex:
                    onoff_tex != onoff_tex
                
            if h_c_0 or h_c_1 or h_c_2:
                handle_change = True
                handle_pos_old = handle_pos_new
                
            if h_r_0 or h_r_1 or h_r_2:
                handle_change = True
                H_r_mat = y_rotation(H_rot_y) @ x_rotation(H_rot_x) @ z_rotation(H_rot_z)
            
            if LSE_changed:
                handle_change = True
                
            if iter_changed:
                handle_change = True
            
            # print(use_disp)
            
            if Reset_button:
                zoom        = 1.0
                scaleX      = 1.0
                scaleY      = 1.0
                scaleZ      = 1.0
                transX      = 0.0
                transY      = 0.0
                H_rot_x     = 0.0
                H_rot_y     = 0.0
                H_rot_z     = 0.0
                H_r_mat     = np.eye(4)
                transZ      = -1.0
                rotation_   = 0
                iteration_  = 0
                normalize   = False
                Reset_button= False
                handle_change = True
                use_LSE = True
                handle_pos_new = np.array([0.0, 0.0, 0.0])
                                
            rotation_ = rotation_ % 360
            H_rot_x   = H_rot_x % 360
            H_rot_y   = H_rot_y % 360
            H_rot_z   = H_rot_z % 360
                        
            imgui.render()
            impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        glReadBuffer(GL_FRONT)
        # glReadBuffer(GL_BACK)

        pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
        a = np.frombuffer(pixels, dtype=np.uint8)
        a = a.reshape((resolution, resolution, 4))
        # # mask  = a[::-1, :, :3] / 255 
        # vis_part = (np.array(Image.open(image_path).resize((1024,1024))) * mask).astype(np.uint8)
        
        # Image.fromarray(a[::-1, :, :3]).save('test2/test_vis{:04}.png'.format(i))
        # i = i + 10
        
        # break
    ################################################## imgui end ############
    if use_imgui:
        impl.shutdown()
    ################################################## imgui end ############
    glfw.terminate()
    return a

if __name__ == '__main__':
    main(resolution=1024)