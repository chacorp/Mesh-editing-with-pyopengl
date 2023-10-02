from __future__ import print_function

import os
from os.path import exists, join, split
from glob import glob
import numpy as np

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
from PIL import Image
import glm
from easydict import EasyDict

# from testwindow import *
from imgui.integrations.glfw import GlfwRenderer
import imgui
import sys
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
import json
import torch
import pickle as pkl
from tqdm import tqdm

from pytorch3d.structures import Meshes
# from pytorch3d.ops import laplacian_matrix
from scipy.sparse.linalg import cg
import scipy.sparse
import numpy as np

# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Standard Laplacian matrix
# 2) Cotangent Laplacian matrix
# 3) Norm Laplacian matrix
# -------------------------------------------------------------------- #

def adjacency_matrix(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    return A

def laplacian_matrix(verts: torch.Tensor, edges: torch.Tensor):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def compute_face_norm(vn, f):
    v1 = vn[f:, 0]
    v2 = vn[f:, 1]
    v3 = vn[f:, 2]
    e1 = v1 - v2
    e2 = v2 - v3

    return np.cross(e1, e2)

def LoadTexture(filename):
    pBitmap = Image.open(filename)
    pBitmap = pBitmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    glformat = GL_RGB # if pBitmap.mode == "RGB" else GL_RGBA
    pBitmap = pBitmap.convert('RGB') # 'RGBA
    pBitmapData = np.array(pBitmap, np.uint8)
    
    # pBitmapData = np.array(list(pBitmap.getdata()), np.int8)
    texName = glGenTextures(1)
    # import pdb; pdb.set_trace()    
    glBindTexture(GL_TEXTURE_2D, texName)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
    )

    # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    ### Texture Wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
    ### Texture Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    """        
        GL_NEAREST_MIPMAP_NEAREST: nearest mipmap to match the pixel size and uses nearest neighbor interpolation for texture sampling.
        GL_LINEAR_MIPMAP_NEAREST:  nearest mipmap level and samples that level using linear interpolation.
        GL_NEAREST_MIPMAP_LINEAR:  linearly interpolates between the two closest mipmaps & samples via nearest neighbor interpolation.
        GL_LINEAR_MIPMAP_LINEAR:   linearly interpolates between the two closest mipmaps & samples via linear interpolation.
    """
    
    # glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texName

def load_textures(filenames):
    texture = glGenTextures(len(filenames))
    for i, filename in enumerate(filenames):
        pBitmap = Image.open(filename)
        pBitmap = pBitmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM) 
        # glformat = GL_RGB # if pBitmap.mode == "RGB" else GL_RGBA
        pBitmap = pBitmap.convert('RGB') # 'RGBA
        pBitmapData = np.array(pBitmap, np.uint8)
            
    
        glBindTexture(GL_TEXTURE_2D, texture[i])
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
        )
    return texture

def load_texture(path):
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

def normalize_v(V):
    V = np.array(V)
    V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    # V = (V-(V.max(0).values + V.min(0).values) * 0.5)/max(V.max(0).values - V.min(0).values)
    
    # FLAME
    # V = V - V.mean(0).values
    # V = V - V.min(0).values[1]
    # V = V / V.max(0).values[1]
    # V = (V * 2.0) - 1.0
    return V

def load_obj_mesh(mesh_path):
    mesh = EasyDict()
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
            # import pdb; pdb.set_trace()
            f = list(map(lambda x: int(x.split('/')[0]),  values[1:]))
            face_data.append(f)
            if len(values[1].split('/')) >=2:
                ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
                face_texture.append(ft)
            if len(values[1].split('/')) >=3:
                ft = list(map(lambda x: int(x.split('/')[2]),  values[1:]))
                face_normal.append(ft)
    # mesh.v  = np.array(vertex_data)
    mesh.v  = normalize_v(np.array(vertex_data))
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    mesh.ft = np.array(face_texture) -1
    mesh.fn = np.array(face_normal) -1
    return mesh

def vertex_normal(v1, v2, v3):
    v1c = np.cross(v2 - v1, v3 - v1)
    v1n = v1c/np.linalg.norm(v1c)
    return v1n

def computeTangentBasis(vertex, uv):
    tangents = []
    tangents = np.zeros_like(vertex)
    # bitangents = []
    for idx in range(0, len(vertex)//3):
        
        # import pdb;pdb.set_trace()
        offset = idx*3
        v0 = vertex[offset]
        v1 = vertex[offset+1]
        v2 = vertex[offset+2]

        offset = idx*3
        uv0 =    uv[offset]
        uv1 =    uv[offset+1]
        uv2 =    uv[offset+2]
        #print v0,v1,v2
        deltaPos1 = np.array([v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]])
        deltaPos2 = np.array([v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]])

        deltaUV1 = np.array([uv1[0]-uv0[0], uv1[1]-uv0[1]])
        deltaUV2 = np.array([uv2[0]-uv0[0], uv2[1]-uv0[1]])

        f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0])
        tangent = (deltaPos1 * deltaUV2[1]   - deltaPos2 * deltaUV1[1]) * f
        # bitangent = (deltaPos2 * deltaUV1[0]   - deltaPos1 * deltaUV2[0]) * f

        tangents[offset]   = tangent
        tangents[offset+1] = tangent
        tangents[offset+2] = tangent
    # import pdb;pdb.set_trace()
    return tangents

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

class SolveLaplacian:
    def __init__(self, mesh):
        self.mesh = mesh
        self.A = None
        self.ATA = None
        self.solver = None
        self.ATb = None
        self.x = [None, None, None]
        self.b = [None, None, None]
        self.build_system_matrix()

        m, n = self.A.shape
        self.ATb = np.zeros(n)
        for i in range(3):
            self.x[i] = np.zeros(n)
            self.b[i] = np.zeros(m)
            for j, v in enumerate(mesh.vList):
                self.x[i][j] = v.Position()[i]
            self.b[i] = self.A @ self.x[i]

        self.build_system_matrix()
        self.build_matrix_ATA()
        self.solver = self.factorization()
        print("ok" if self.solver else "fail")

    def build_system_matrix(self):
        raise NotImplementedError

    def build_matrix_ATA(self):
        AT = self.A.transpose()
        self.ATA = AT @ self.A

    def factorization(self):
        print("Factorization")
        ATA_csr = csr_matrix(self.ATA)
        solver = factorized(ATA_csr)
        return solver

    def deform(self):
        vList = self.mesh.Vertices()
        n = len(vList)
        for i in range(3):
            for j, v in enumerate(vList):
                if v.Flag():
                    # add constraints with weight = 1000 
                    self.b[i][j+n] = v.Position()[i] * 1000
            self.ATb = self.A.T @ self.b[i]
            self.x[i] = self.solver(self.ATb)
        for j, v in enumerate(vList):
            if v.Flag() == 0:
                v.SetPosition(np.array([self.x[0][j], self.x[1][j], self.x[2][j]]))

def render(resolution=512, mesh=None):
    if mesh is None:
        # mesh = load_obj_mesh("R:\eNgine_visual_wave\engine_obj\M_012.obj")
        
        # ### high res smpl mesh
        # mesh = load_obj_mesh("experiment/smpl_hres/smpl_hres_mesh_2.obj")
        # # meshes.num_verts_per_mesh()
        # meshes = load_objs_as_meshes(["experiment/smpl_hres/smpl_hres_mesh_2.obj"])
        
        ### smpl mesh
        # mesh = load_obj_mesh("D:/Dataset/smpl_mesh_1.obj")
        # meshes = load_objs_as_meshes(["D:/Dataset/smpl_mesh_1.obj"])
        
        path   = "D:/test/Mingle/experiment/sphere/sphere.obj"
        mesh   = load_obj_mesh(path)
        meshes = load_objs_as_meshes([path])
        
        ### mignle mesh
        # path = "N:/01-Projects/2023_KOCCA_AvatarFashion/10_data sample/Avatar_00_no_hair-male.obj"
        # mesh = load_obj_mesh(path)
        # meshes = load_objs_as_meshes([path])
        # mesh_ = load_obj_mesh("D:/Dataset/smpl_mesh_1.obj")
        
        ### mignle hres mesh
        # mesh = load_obj_mesh("D:/test/RaBit/experiment/mingle/hres_to_Avatar_00.obj")
        # mesh_ = load_obj_mesh("experiment/smpl_hres/smpl_hres_mesh_2.obj")
        # mesh.vn = mesh_.vn
        # mesh.fn = mesh_.fn
        
        # mesh = load_obj_mesh("mean.obj")
        # tmp_obj = load_obj("experiment/smpl_hres/smpl_hres_mesh_2.obj")
        # mesh        = EasyDict()
        # mesh.v      = tmp_obj[0]
        # mesh.f      = tmp_obj[1].verts_idx
        # mesh.ft     = tmp_obj[1].textures_idx
        # mesh.vt     = tmp_obj[2].verts_uvs
        # mesh.vn     = tmp_obj[2].normals
        
        
        ### laplacian matrix
        L = meshes.laplacian_packed()
        
        ## deform mesh
        edges_packed = meshes.edges_packed()
        verts_packed = meshes.verts_packed()
        Adj = adjacency_matrix(verts_packed, edges_packed)
        
        mesh.e = edges_packed
        mesh.L = L
        mesh.Adj = Adj
        
        
        # adjacency_matrix = Adj.to_dense()
        # # import pdb;pdb.set_trace()
        # # AA = A @ verts_packed
        # # from pytorch3d.structures import packed_to_list
        # valences = torch.Tensor([adjacents.sum() for adjacents in adjacency_matrix])
        # Diag = torch.diag(valences)
        # LAP = I - torch.linalg.inv(Diag) @ adjacency_matrix


        # # Given a Meshes object `meshes`
        # verts = meshes.verts_packed() # vertices of your mesh
        # faces = meshes.faces_packed() # faces of your mesh

        # 1. Calculate Laplacian Matrix
        # lap = laplacian(verts_packed, edges_packed)
        # laplacian = laplacian_matrix(verts_packed, edges_packed)
        # L = meshes.laplacian_packed()

        # # 2. Compute Laplacian coordinates
        # delta = torch.matmul(L, verts)

        # # 3. Constraints
        # # Define your constraints
        # # You are going to constrain vertices 0 to 10 and move them along the x-axis by a scalar of 2.
        # constraints = np.zeros((len(verts), 3))
        # constraints_idx =11
        # constraints[0:11] *= 2

        # # Create a one-hot matrix for the constraint
        # C = np.zeros((len(verts), len(verts)))
        # C[0:11, 0:11] = np.eye(11)  # Only the first 11 vertices are constrained

        # # Extract the submatrix of L corresponding to the unconstrained vertices
        # laplacian = L.to_dense()
        # Luu = laplacian[11:, 11:]
        # Luf = laplacian[11:, :11]

        # # Calculate the right-hand side of the system
        # bu = delta[11:] - Luf @ constraints[:11]
        
        # delta = torch.matmul(L, verts)

        # # Iterative update
        # MAX_ITER = 1000
        # EPSILON = 1e-6
        # old_verts = verts[11:].cpu().numpy()
        # for i in tqdm(range(MAX_ITER)):
        #     new_verts = scipy.sparse.linalg.spsolve(Luu, bu)
        #     if np.max(np.abs(old_verts - new_verts)) < EPSILON:
        #         break
        #     old_verts = new_verts

        # # Combine constrained vertices and updated vertices
        # new_verts_all = np.vstack([constraints[:11], new_verts])
        # print(new_verts_all.shape)

        # # Convert the new vertices to a tensor
        # new_verts = torch.from_numpy(new_verts_all).float().to(verts.device)
        # 'Lx = d', where L is the Laplacian matrix, x is the vertex positions, and d is the Laplacian coordinates (computed from the original surface)
        # 'Cx = c', where C is a binary matrix (a one-hot matrix indicating the fixed vertices), x is the vertex positions, and c is the fixed vertex positions
        # | L   C^T | | x | = | d |
        # | C   0   | | y |   | c |
        # Here, y are the Lagrange multipliers. They are introduced to enforce the constraints but we are not interested in their values.
        # constraint_indices = np.arange(800)+3000  # vertex indices 10-20
        
        # modify_idx = np.arange(5)
        # modify_verts = verts.clone()
        # modify_verts[modify_idx] = modify_verts[modify_idx] * 1.2
                
        # C = scipy.sparse.coo_matrix((np.ones(len(constraint_indices)), (constraint_indices, np.arange(len(constraint_indices)))), shape=(len(verts), len(constraint_indices)))
        
        # # Create the fixed vertex positions c
        # cc = np.zeros((len(constraint_indices), 3))
        # cc[:, 0]  = 2  # move along the x-axis by 2

        
        # # import pdb;pdb.set_trace()
        # L = L.to_dense()
        # # Construct the square linear system
        # L_upper   = scipy.sparse.hstack([L, C])
        # C_lower   = scipy.sparse.hstack([C.transpose(), scipy.sparse.coo_matrix((C.shape[1], C.shape[1]))])
        # L_square  = scipy.sparse.vstack([L_upper, C_lower]).tocsc()
        
        # d_upper = np.vstack([delta, cc])
        
        # # Solve the square linear system
        # print(verts.shape)
        # new_verts = scipy.sparse.linalg.spsolve(L_square, d_upper)
        # print(new_verts.shape)
        
        """_summary_
        # # Extract the vertex positions from the solution
        # new_verts = x[:3*len(verts)].reshape((-1, 3))

        # # Convert the new vertices to a tensor
        # new_verts = torch.from_numpy(new_verts).to(verts.device)

        # new_delta = L.mm(modify_verts)
        # new_delta = new_delta.numpy()
        
        # one_hot   = torch.zeros([L.shape[0]]).scatter_(0, torch.tensor(constraints_idx), 1)
        # # L_Tilda = torch.cat([L.to_dense(), one_hot[None]])
        # # L_Tilda = L_Tilda.numpy()
        # # new_verts,_ = np.linalg.solve(L_Tilda, new_delta)[0]
        
        
        # # Convert PyTorch tensor to SciPy sparse matrix
        # L_np = L.to_dense().cpu().numpy()
        # L_sp = scipy.sparse.csc_matrix(L_np)

        # # Assuming one_hot_vector and laplacian_np have been defined
        # one_hot_2d = scipy.sparse.csc_matrix(one_hot.reshape(-1, 1))  # Convert the one-hot vector to a 2D sparse matrix

        # # Concatenate the one-hot vector to the Laplacian
        # L_sp_extended = scipy.sparse.hstack([L_sp, one_hot_2d])
        # new_delta = np.concatenate([new_delta, np.atleast_2d(modify_verts[modify_idx])], axis=0)
        
        # new_verts = torch.linalg.solve(L_sp_extended, new_delta)
        # # # Compute the right-hand side of the system
        # # B = delta - torch.matmul(L, constraints)
        # # B = B.cpu().numpy()
        

        # # # Solve using Conjugate Gradient method
        # # new_verts = np.empty_like(verts)
        # # # Solve separately for x, y, z
        # new_verts = scipy.sparse.linalg.lsqr(L_sp_extended, new_delta)
        # new_verts[:, 0], _ = cg(L_sp_extended, new_delta[:, 0])
        # new_verts[:, 1], _ = cg(L_sp_extended, new_delta[:, 1])
        # new_verts[:, 2], _ = cg(L_sp_extended, new_delta[:, 2])      

        # # Convert result back to PyTorch tensor
        # new_verts = torch.from_numpy(new_verts).to(verts.device)
        # import pdb;pdb.set_trace()
        """
        # mesh.v = new_verts[:mesh.v.shape[0]]
        # Update the mesh with the new vertices
        # new_meshes = Meshes(verts=new_verts, faces=faces)
        
        if True:
            # -------------------------- Build the Linear System ------------------------- #
            V = verts_packed
            L2D = L.to_dense().numpy()
            Delta = L.mm(V).numpy()
            N = L.shape[0]
            V = V.numpy()

            LS = np.zeros([3*N, 3*N])
            LS[0*N:1*N, 0*N:1*N] = (-1) * L2D
            LS[1*N:2*N, 1*N:2*N] = (-1) * L2D
            LS[2*N:3*N, 2*N:3*N] = (-1) * L2D

            for i in range(N):
                # nb_idx = get_local_neighbor(subgraph, i, l2g, g2l)
                nb_idx = Adj[i].coalesce().indices()[0].tolist()
                ring = np.array([i] + nb_idx)
                n_ring = len(ring)
                V_ring = V[ring]
                # n_ring = V_ring.shape[0]
                
                A = np.zeros([n_ring * 3, 7])
                zer0_ring = np.zeros(n_ring)
                
                A[:n_ring,         0:4] = np.stack([V_ring[:,0],    zer0_ring,  V_ring[:,2], -V_ring[:,1]], axis=1)
                A[n_ring:2*n_ring, 0:4] = np.stack([V_ring[:,1], -V_ring[:,2],    zer0_ring,  V_ring[:,0]], axis=1)
                A[2*n_ring:,       0:4] = np.stack([V_ring[:,2],  V_ring[:,1], -V_ring[:,0],    zer0_ring], axis=1)

                A[:n_ring,           4] = 1
                A[n_ring:2*n_ring,   5] = 1
                A[2*n_ring:,         6] = 1
                
                # Moore-Penrose Inversion
                A_pinv = np.linalg.pinv(A)
                s = A_pinv[0]
                h = A_pinv[1:4]
                t = A_pinv[4:7]            

                T_delta = np.vstack([
                     Delta[i,0]*s    - Delta[i,1]*h[2] + Delta[i,2]*h[1], #+ t[0],
                     Delta[i,0]*h[2] + Delta[i,1]*s    - Delta[i,2]*h[0], #+ t[1],
                    -Delta[i,0]*h[1] + Delta[i,1]*h[0] + Delta[i,2]*s,    #+ t[2],
                ])
                
                LS[i,     np.hstack([ring, ring+N, ring+2*N])] += T_delta[0]
                LS[i+N,   np.hstack([ring, ring+N, ring+2*N])] += T_delta[1]
                LS[i+2*N, np.hstack([ring, ring+N, ring+2*N])] += T_delta[2]
            # ------------------- Add Constraints to the Linear System ------------------- #
            constraint_coef = []
            constraint_b = []

            # Boundary constraints
            boundary_idx = [3, 13] + [28 + i*15 for i in range(20)] + [329 + i*15 for i in range(10)]
            for idx in boundary_idx:
                constraint_coef.append(np.arange(3*N) == idx)
                constraint_coef.append(np.arange(3*N) == idx + N)
                constraint_coef.append(np.arange(3*N) == idx + 2*N)
                constraint_b.append(V[idx, 0])
                constraint_b.append(V[idx, 1])
                constraint_b.append(V[idx, 2])
                
            # Handle constraints
            handle_idx = [325]
            for idx in handle_idx:
                # idx = g2l[gid]
                constraint_coef.append(np.arange(3*N) == idx)
                constraint_coef.append(np.arange(3*N) == idx + N)
                constraint_coef.append(np.arange(3*N) == idx + 2*N)
                constraint_b.append(V[idx, 0] * 1.2)
                constraint_b.append(V[idx, 1] * 1.2)
                constraint_b.append(V[idx, 2] * 1.2)
                
            constraint_coef = np.matrix(constraint_coef)
            constraint_b = np.array(constraint_b)

            # -------------------------- Solve the Linear System ------------------------- #
            A         = np.vstack([LS, constraint_coef])
            b         = np.hstack([np.zeros(3*N), constraint_b])
            spA       = scipy.sparse.coo_matrix(A)

            V_prime  = scipy.sparse.linalg.lsqr(spA, b)
            
            new_pnts = []
            for i in range(N):
                new_pnts.append([V_prime[0][i], V_prime[0][i+N], V_prime[0][i+2*N]])

            # -------------------------- Output the Edited Mesh -------------------------- #
            new_verts = mesh.v.copy()
            ## all vertices
            all_idx = [i for i in range(N) if i != handle_idx[0]]
            for idx, pnt in enumerate(new_pnts):
                # if idx in all_idx:
                new_verts[idx] = np.array(pnt)
            # import pdb;pdb.set_trace()
            mesh.v = new_verts
        
        # ## new vert position
        # new_pos_0 = verts_packed[0] * 1.2
        # ## transformation or vertex movement
        # transform = new_pos_0 - verts_packed[0]

        # verts_packed[0] = new_pos_0
        
        # ## add transformation to all vertices
        # verts_packed = verts_packed + transform
        
        # ### laplacian coordinate
        # delta = L.mm(verts_packed)
        
        # new_vers_packed = verts_packed.clone()
        # ### deformed mesh : move first vertex
        # new_vers_packed[0] = new_vers_packed[0] * 3
        
        # # 3. Define your constraints
        # # This depends heavily on what you want to achieve.
        # # Here is a dummy example where we constrain the first vertex to stay at its initial position
        # constraints = torch.zeros_like(new_vers_packed)
        # constraints[0] = new_vers_packed[0]

        # For simplicity, lets assume that the Laplacian matrix is full rank, 
        # and the system is over-constrained (more vertices than constraints)
        # We can use the pseudo-inverse to solve for the vertices
        # In a more general case, you might have to use quadratic programming here
        # new_verts = torch.matmul(laplacian_pseudo_inv, lap_coord - torch.matmul(L_dense, constraints)) + constraints
        ## ||delta - lap(v')|| + ||v' - u||
        # new_verts = torch.matmul(laplacian_pseudo_inv, delta - L.mm(constraints)) + constraints
        
        # laplacian_pseudo_invp = torch.linalg.pinv(L_dense)
        # new_vert = torch.linalg.solve(L_dense, lap_coord)
        
        # >>> A = torch.randn(3, 3) square matrix
        # >>> b = torch.randn(3)
        # >>> x = torch.linalg.solve(A, b)
        # >>> torch.allclose(A @ x, b)
        
        # new_vert = laplacian_pseudo_inv @ lap_coord
        # new_vert = L.t() @ lap_coord
        
        # import pdb;pdb.set_trace()
        
    
    # mesh.v      = normalize_v(mesh.v)
    
    # image_path  = "white.png"
    boundary_mask_path  = "experiment/smpl_hres/SMPL_boundary_mask.png"
    
    num = None
    num = 12 # long pants
    # num = 315 # long pants
    # num = 38243 # short pants
    # num = 39417 # short pants
    
    base_path = "M:\\SihunCha\\Publication\\[EG_2023]\\9_fast_forward"
    if num:
        # image_path= base_path + "\\re_try\\infer\\Refiner-Sampler4-new-2\\final texture\\image_{:06}_output.png".format(num)
        # disp_path = base_path + "\\re_try\\displacement\\image_{:06}_refine_smpld.json_crop_displacement_map.png".format(num)
        # json_file = base_path + "\\re_try\\displacement\\json\\image_{:06}_refine_smpld.json".format(num)
        image_path= "D:/test/Mingle/experiment/rp_aaron_posed_003/rp_aaron_posed_003.png"
        # disp_path = "D:/test/Mingle/experiment/rp_aaron_posed_003/rp_aaron_posed_003_smpld_disp.json_crop_displacement_map.png"
        # json_file = "D:/test/Mingle/experiment/rp_aaron_posed_003/rp_aaron_posed_003_smpld_disp.json"
        disp_path = "D:/test/Mingle/experiment/rp_aaron/rp_aaron_posed_003_smpld_disp.json_crop_displacement_map.png"
        json_file = "D:/test/Mingle/experiment/rp_aaron/rp_aaron_posed_003_smpld_disp.json"
        
        with open(json_file) as f:
            json_object = json.load(f)
    else:
        disp_path = None
        json_object = None
    
        
    # rendered    = main(mesh, resolution, image_path, timer=True)
    rendered    = main(mesh, resolution, image_path, disp_path, json_object, boundary_mask_path, timer=True)
    # rendered    = main(mesh, resolution, image_path, disp_path=None, json_object=None, timer=True)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('output_mingle')
    if not exists(savefolder):
        os.makedirs(savefolder)
    # savefile    = join(savefolder, 'rendered_{:06}_tex.png'.format(num))
    # savefile    = join(savefolder, 'rendered_test_{:06}.png'.format(num))
    savefile    = join(savefolder, 'exp_{:06}-.png'.format(num))
    # savefile    = join(savefolder, 'rendered_{:06}_tex.png'.format(0))

    Image.fromarray(rendered).save(savefile)
    return

def data_render(resolution=512):
    meshlist = sorted(glob("R:\\3DBiCar\\data\\*\\tpose\\m.obj"))
    
    from pytorch3d.io import load_obj    
    # objs = "./data/*/tpose/m.obj"
    # obj_list = sorted(glob(objs))    
    # tmp_obj = load_obj(obj_list[0])
    # verts = tmp_obj[0]
    # faces = tmp_obj[1].verts_idx
    # ft    = tmp_obj[1].textures_idx
    # uvs   = tmp_obj[2].verts_uvs
    # vn    = tmp_obj[2].normals
    
    for meshdir in meshlist:
        idx = meshdir.split('\\')[-3]
        # mesh = load_obj_mesh(meshdir)
        tmp_obj = load_obj(meshdir)
        mesh = EasyDict()
        mesh.v  = tmp_obj[0]
        mesh.f  = tmp_obj[1].verts_idx
        mesh.ft = tmp_obj[1].textures_idx
        mesh.vt = tmp_obj[2].verts_uvs
        mesh.vn = tmp_obj[2].normals
        mesh.v = normalize_v(mesh.v)
        mesh.v[:,2] = mesh.v[:,2] * -1
                
        # image_path  = "white.png"
        image_path  = "experiment/tex.png"
        rendered    = main(mesh, resolution, image_path, timer=True)
        rendered    = rendered[::-1, :, :]
        
        # make directory
        savefolder  = join('output')
        if not exists(savefolder):
            os.makedirs(savefolder)
        print(idx)
        savefile    = join(savefolder, 'rendered_{}.png'.format(idx))

        Image.fromarray(rendered).save(savefile)
        # return

def pca_render(mean, coef, basis, uvs, vn, faces, ft, resolution=512):    
    pca_mesh    = EasyDict()
    pca_mesh.v  = None # dummy -> calculated in main()
    pca_mesh.vt = uvs
    pca_mesh.vn = vn
    pca_mesh.f  = faces
    pca_mesh.ft = ft
    
    # image_path  = "white.png"
    image_path  = "experiment/test/tex.png"
    rendered    = main(pca_mesh, resolution, image_path, timer=True, 
                       pca_v=True, mean=mean, coef=coef, basis=basis)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('experiment/output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile    = join(savefolder, 'rendered.png')

    Image.fromarray(rendered).save(savefile)
    return

def main(mesh, resolution, image_path, disp_path=None, json_object=None, boundary_mask_path=None, timer=False, 
         pca_v=False, mean=None, coef=None, basis=None):
    if timer == True:
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
    
    if pca_v == True:
        blendshape = np.zeros_like(coef)
        new_v = np.zeros_like(new_vt) # dummy -> will be updated in while loop
    else:
        new_v  = v[f].reshape(-1, 3)
    
    # import pdb;pdb.set_trace()
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
        
    scaleX = 1.0
    scaleY = 1.0
    scaleZ = 1.0
    
    transX = 0.0
    transY = 0.0
    transZ = -1.0
    
    Reset_button = False
    
    tex_alpha = 1.0
    
    normalize = False
    _ratio = 1.0
    
    if json_object:
        d_range = json_object['d_range']
        d_range_0 = np.zeros_like(d_range)
        gl_d_range = glGetUniformLocation(shader, "d_range")
        use_disp = False
        # use_disp = True
    
    onoff_tex = True
    
    scale   = glGetUniformLocation(shader, "scale") # scale rotate translate
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
            
        ## trans * rotate * scale * 3D model
        rotation_mat = y_rotation(rotation_)
        rotation_mat_x = x_rotation(-30)
        affine_mat = np.eye(4)
        affine_mat[:3,:3] = affine_mat[:3,:3] * np.array([scaleX, scaleY, scaleZ])
        trans = np.array([transX, transY, transZ])
        # affine_mat[:3,-1] = trans
        rotation_mat = rotation_mat_x @ rotation_mat @  affine_mat.T
        
        # glUniformMatrix4fv(scale, 1, GL_FALSE, affine_mat)
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        glUniform3fv(translate, 1, trans)
        
        view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
        glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
        
        if pca_v == True:
            upd_v = mean + np.dot(coef * blendshape, basis)
            upd_v = upd_v.reshape(-1,3)
            # upd_v = normalize_v(upd_v)
            # upd_v[:, 2] = upd_v[:, 2] * -1
            upd_v = upd_v[f].reshape(-1, 3)
            quad[:, :3] = upd_v
        else:
            upd_v = new_v
            # upd_v = new_v * np.array([scaleX, scaleY, scaleZ])
            # upd_v = upd_v + np.array([transX, transY, transZ])
            quad[:, :3] = upd_v
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.poll_events()
        if use_imgui or pca_v:
            impl.process_inputs()
            imgui.new_frame()
            imgui.text("mean vert: {}".format(upd_v.mean(0)))
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):

                    clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", "Cmd+Q", False, True
                    )

                    if clicked_quit:
                        sys.exit(0)

                    imgui.end_menu()
                imgui.end_main_menu_bar()
            
            imgui.text("d_range: {}".format(d_range))
            
            clicked, tex_alpha = imgui.slider_float(label="_alpha",    value=tex_alpha, min_value=0.0, max_value=1.0)
            clicked, _ratio    = imgui.slider_float(label="_ratio",    value=_ratio, min_value=0.0, max_value=1.0)
            clicked, rotation_ = imgui.slider_float(label="Rotate", value=rotation_, min_value=0.0, max_value=360.0,)
            
            clicked, scaleX = imgui.slider_float(label="Scale x",   value=scaleX, min_value= 0.1,  max_value= 10.0,)
            changed, scaleX = imgui.input_float(label="set Scale x",value=scaleX, step=0.001)
            clicked, scaleY = imgui.slider_float(label="Scale y",   value=scaleY, min_value= 0.1,  max_value= 10.0,)
            changed, scaleY = imgui.input_float(label="set Scale y",value=scaleY, step=0.001)
            clicked, scaleZ = imgui.slider_float(label="Scale z",   value=scaleZ, min_value= 0.1,  max_value= 10.0,)
            
            clicked, transX = imgui.slider_float(label="Trans x",   value=transX, min_value=-5.0,  max_value= 5.0,)
            changed, transX = imgui.input_float(label="set Trans x",value=transX, step=0.001)
            clicked, transY = imgui.slider_float(label="Trans y",   value=transY, min_value=-5.0,  max_value= 5.0,)
            changed, transY = imgui.input_float(label="set Trans y",value=transY, step=0.001)
            clicked, transZ = imgui.slider_float(label="Trans z",   value=transZ, min_value=-100,  max_value= 10,)
            changed, transZ = imgui.input_float(label="set Trans z",value=transZ, step=0.001)
            
            clicked, zoom   = imgui.slider_float(label="Zoom", value=zoom, min_value=0.1,  max_value= 2.0,)
            changed, Reset_button = imgui.menu_item("Reset", None, Reset_button)
            clicked_use_disp, use_disp   = imgui.menu_item("Use_disp", None, use_disp)
            clicked_use_tex, onoff_tex   = imgui.menu_item("OnOff Tex", None, onoff_tex)

            if clicked_use_disp:
                use_disp != use_disp
            if clicked_use_tex:
                onoff_tex != onoff_tex
            # print(use_disp)
            
            if Reset_button:
                if pca_v == True:
                    blendshape  = blendshape * 0
                zoom        = 1.0
                scaleX      = 1.0
                scaleY      = 1.0
                scaleZ      = 1.0
                transX      = 0.0
                transY      = 0.0
                transZ      = -1.0
                rotation_   = 0
                normalize   = False
                Reset_button= False
                                
            rotation_ = rotation_ % 360
            
            if pca_v == True:
                for i in range(len(blendshape)):    
                    clicked, blendshape[i] = imgui.slider_float(
                        label="Value"+ str(i),
                        value=blendshape[i],
                        min_value = -1.0,
                        max_value =  1.0,
                    )
            
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
    if use_imgui or pca_v:
        impl.shutdown()
    ################################################## imgui end ############
    glfw.terminate()
    return a

if __name__ == '__main__':
    render(resolution=1024)
    # data_render(resolution=512)

