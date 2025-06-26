import numpy as np

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from PIL import Image
import glm

import igl

from utils.laplacian import adjacency_matrix, laplacian_cotangent, laplacian_and_adjacency
from utils.util import normalize_np
from utils.processing import laplacian_matrix_ring_LSE

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
        
        # self.Adj = adjacency_matrix(self.v.shape[0], self.e, return_idx=False)

        # uniform laplacian
        self.UL, self.Adj = laplacian_and_adjacency(self.v.shape[0], self.e)

        ## cotangent laplacian        
        # something gone wrong...!
        # self.L  = laplacian_cotangent(self.v, self.f, eps=1e-12)
        # using igl library
        self.L = igl.cotmatrix(self.v, self.f)
        
        self.orig_v = self.v.copy()
        
        self.delta = self.L @ self.v
        
        self.ring_indices   = [self.Adj[i].indices.tolist() for i in range(self.v.shape[0])]
        
        self.mask_v = np.zeros([self.v.shape[0],1]).astype(int)
        self.mask_v[boundary_idx] = 1
        for h_idx in handle_idx:
            recurr_adj(self.mask_v, self.Adj, h_idx, boundary_idx)
        self.mask_v[boundary_idx] = 0
        # print(np.where(self.mask_v>0)[0])
        
        self.L_LSE = laplacian_matrix_ring_LSE(self)


        N = self.v.shape[0]
        self.LS = np.zeros([3*N, 3*N])
        self.LS[0*N:1*N, 0*N:1*N] = self.L.todense()
        self.LS[1*N:2*N, 1*N:2*N] = self.L.todense()
        self.LS[2*N:3*N, 2*N:3*N] = self.L.todense()
    
    def compute_edges(self):
        faces = self.f  # shape: (F, 3)
        v0, v1, v2 = faces[:, 0:1], faces[:, 1:2], faces[:, 2:3]  # shape: (F, 1)

        e01 = np.hstack([v0, v1])  # (F, 2)
        e12 = np.hstack([v1, v2])  # (F, 2)
        e20 = np.hstack([v2, v0])  # (F, 2)

        edges = np.vstack([e12, e20, e01])  # shape: (3F, 2)
        edges = np.sort(edges, axis=1)      # make edges in (min, max) form

        V = self.v.shape[0]
        edges_hash = V * edges[:, 0] + edges[:, 1]

        uqe, idx = np.unique(edges_hash, return_index=True)
        edges_unique = edges[idx]  # extract unique edges

        self.e = edges_unique.astype(int)  # store as integer array

    def load_obj_mesh(self, mesh_path, normalize=True):
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
        ) if normalize else np.array(vertex_data)
        
        self.vn = np.array(vertex_normal)
        self.vt = np.array(vertex_texture)
        self.f  = np.array(face_data)
        self.ft = np.array(face_texture)
        self.fn = np.array(face_normal)

        if self.f.min() > 0:
            self.f  = self.f  - 1
            self.ft = self.ft - 1
            self.fn = self.fn - 1