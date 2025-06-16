from __future__ import print_function

import sys
import os
from os.path import exists, join
# from glob import glob
import numpy as np

import scipy
from scipy.sparse import diags

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from PIL import Image
import glm

# from testwindow import *
from imgui.integrations.glfw import GlfwRenderer
import imgui

import torch


# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Standard Laplacian matrix
# 2) Cotangent Laplacian matrix
# 3) Norm Laplacian matrix
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

def laplacian_matrix_ring(mesh):
    """
    Reference: 
        - Laplacian Surface Editing, Olga Sorkine, Daniel Cohen-Or, 2004
        - https://github.com/luost26/laplacian-surface-editing/blob/master/main.py

    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    V = mesh.v
    N = mesh.v.shape[0]
    
    # laplacian matrix
    L = mesh.L.todense()
    
    # laplacian coordinates :: Delta = L @ V
    # delta = mesh.delta
    delta = np.concatenate([mesh.delta, np.ones([N, 1])], axis=-1)
    
    # print(L.shape)
    # print(3*N)
    LS = np.zeros([3*N, 3*N])
    # LS[0::3, 0::3] = (-1) * L
    # LS[1::3, 1::3] = (-1) * L
    # LS[2::3, 2::3] = (-1) * L
    LS[0*N:1*N, 0*N:1*N] = -L
    LS[1*N:2*N, 1*N:2*N] = -L
    LS[2*N:3*N, 2*N:3*N] = -L
    
    # compute the l
    for i in range(N):
        nb_idx      = mesh.ring_indices[i]
        ring        = np.array([i] + nb_idx)
        n_ring      = len(ring)
        
        Ai          = np.zeros([n_ring * 3, 7])
        zer0_ring   = np.zeros(n_ring)
        ones_ring   = np.ones(n_ring)
        
        V_ring      = V[ring].numpy()
        Ai[:n_ring,        ] = np.stack([V_ring[:,0],    zer0_ring,  V_ring[:,2], -V_ring[:,1], ones_ring, zer0_ring, zer0_ring], axis=1)
        Ai[n_ring:2*n_ring,] = np.stack([V_ring[:,1], -V_ring[:,2],    zer0_ring,  V_ring[:,0], zer0_ring, ones_ring, zer0_ring], axis=1)
        Ai[2*n_ring:,      ] = np.stack([V_ring[:,2],  V_ring[:,1], -V_ring[:,0],    zer0_ring, zer0_ring, zer0_ring, ones_ring], axis=1)
        # Ai[:n_ring,           4] = 1
        # Ai[n_ring:2*n_ring,   5] = 1
        # Ai[2*n_ring:,         6] = 1
        
        # import pdb; pdb.set_trace()
        
        # Moore-Penrose Inversion
        AiTAi_pinv = np.linalg.pinv(Ai.T @ Ai)
        Ai_pinv = AiTAi_pinv @ Ai.T
                
        s = Ai_pinv[0]
        h = Ai_pinv[1:4]
        t = Ai_pinv[4:7]
        
        ones_3 = np.ones(n_ring * 3)
        zer0_3 = np.zeros(n_ring * 3)
        
        Ti = np.array([
            # [    s ,  -h[2],  h[1],],
            # [  h[2],    s ,  -h[0],],
            # [ -h[1],  h[0],     s ,],
            
            [    s , -h[2],  h[1],  t[0],],
            [  h[2],    s , -h[0],  t[1],],
            [ -h[1],  h[0],    s ,  t[2],],
            [zer0_3,zer0_3,zer0_3,ones_3,],
            
            # [    s ,  h[2], -h[1],],
            # [ -h[2],    s ,  h[0],],
            # [  h[1], -h[0],    s ,],
            
            # [    s,   h[2], -h[1], zer0_3,],
            # [ -h[2],     s,  h[0], zer0_3,],
            # [  h[1], -h[0],     s, zer0_3,],
            # [  t[0],  t[1],  t[2], ones_3,],
        ])
        
        # T_delta = (Ti @ delta[i])
        T_delta =(Ti.transpose(2,0,1) @ delta[i].T).transpose(1,0)
        # T_delta = (delta[i].T @ Ti )
        
        # import pdb; pdb.set_trace()
        
        rind_idxs = np.hstack([ring, ring+N, ring+2*N])
        LS[[i, i+N, i+2*N]][:, rind_idxs] += T_delta[:3]
        # LS[i,     rind_idxs] += T_delta[0]
        # LS[i+N,   rind_idxs] += T_delta[1]
        # LS[i+2*N, rind_idxs] += T_delta[2]
        
        # import pdb;pdb.set_trace()
        # rind_idxs = np.hstack([0+ring*3, 1+ring*3, 2+ring*3])
        # LS[i*3+0:i*3+3*n_ring:3, ring] += T_delta[0]
        # LS[i*3+1, rind_idxs] += T_delta[1]
        # LS[i*3+2, rind_idxs] += T_delta[2]
        
    return LS

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
    V  = mesh.v
    N  = mesh.v.shape[0]
    LS = mesh.LS
    # ------------------- Add Constraints to the Linear System ------------------- #
    len_H = len(handle_idx)
    constraint_coef = np.zeros([3*N + 3*len_H, 3*N])
    constraint_b    = np.zeros([3*N + 3*len_H])
    
    # Boundary constraints
    i = 0
    h = 0
    
    # for idx in boundary_idx:
    for idx, val in enumerate(mask):
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Boundary constraint
        idx_v = [idx, idx+N, idx+2*N] # index of v in V
        
        if val == 0:
            constraint_coef[idx_c, idx_v] = 1.0 * Wb # one-hot
            constraint_b[idx_c]           = V[idx] * Wb # fixed position
        else:
            constraint_coef[idx_c, idx_v] = 0
            constraint_b[idx_c]           = 0
        i = i + 1
        
    # Handle constraints
    for idx in handle_idx:
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Handle constraint
        idx_v = [idx, idx+N, idx+2*N] # index of v in V
        
        constraint_coef[idx_c, idx_v] = 1.0 # one-hot
        constraint_b[idx_c]           = handle_pos[h]
        
        i = i + 1
        h = h + 1
    # -------------------------- Solve the Linear System ------------------------- #
    A        = np.vstack([LS,         constraint_coef])
    b        = np.hstack([np.zeros(3*N), constraint_b])
    # A        = LS
    # b        = np.zeros(3*N)
    spA      = scipy.sparse.coo_matrix(A)
        
    V_prime  = scipy.sparse.linalg.lsqr(spA, b)[0]
    
    new_verts = V_prime.reshape(3, -1).T
    return new_verts

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
        
        self.L, self.Adj    = laplacian_and_adjacency(self.v.shape[0], self.e)
        self.delta          = self.L @ self.v
        
        self.ring_indices   = [self.Adj[i].indices.tolist() for i in range(self.v.shape[0])]
        
        self.mask_v = np.zeros([self.v.shape[0],1]).astype(int)
        self.mask_v[boundary_idx] = 1
        for h_idx in handle_idx:
            recurr_adj(self.mask_v, self.Adj, h_idx, boundary_idx)
        self.mask_v[boundary_idx] = 0
        
        ### laplacian surface editing
        self.LS = laplacian_matrix_ring(self)
    
    def compute_edges(self):
        """
        Computes edges in packed form from the packed version of faces and verts.
        reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html#Meshes.edges_packed
        """
        
        faces = self.f
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
        
        self.v  = normalize_torch(
            torch.tensor(vertex_data)
        )
        self.vn = torch.tensor(vertex_normal)
        self.vt = torch.tensor(vertex_texture)
        self.f  = torch.tensor(face_data)
        self.ft = torch.tensor(face_texture)
        self.fn = torch.tensor(face_normal)
        if self.f.min() > 0:
            self.f  = self.f  - 1
            self.ft = self.ft - 1
            self.fn = self.fn - 1
        
    
def main(resolution=512,
        boundary_idx = [3, 13, 481, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 223, 238, 253, 268, 283, 298, 313, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464],
        handle_idx   = [0, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 326, 341, 356, 371, 386, 401, 416, 431, 446, 461],
         ):
    ### sphere mesh
    path                = "data/sphere.obj"
    # path               = r"experiment\faces\FLAME_with_normal.obj"
    
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
         boundary_idx=[3, 13, 481] + [28 + i*15 for i in range(20)] + [329 + i*15 for i in range(10)],
         handle_idx=[325],
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

    new_vt = vt[ft].reshape(-1,2).numpy()
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    new_v  = v[f].reshape(-1, 3).numpy()
    
    # if f.max() == vn.shape[0]:
    if True:
        new_vn = vn[mesh.fn].reshape(-1, 3).numpy()
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
    
    tex_alpha = 1.0
    H_r_mat = np.eye(4)
    handle_pos_old = np.array([0.0, 0.0, 0.0])
    handle_pos_new = np.array([0.0, 0.0, 0.0])
    # handle_pos_new = new_v[handle_idx]
    handle_pos = mesh.v[handle_idx]
    handle_change = False
    
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
            # print('1')
            # handle_pos_new_ = handle_pos + handle_pos_new.copy()
            # handle_pos_new_ = handle_pos.copy()
            
            # handle_pos_new_ = handle_pos_new.copy()
            # if h_r_0 or h_r_1 or h_r_2:
            handle_pos_mean = handle_pos.mean(0)
            handle_pos_new_ = (handle_pos-handle_pos_mean) @ H_r_mat[:3,:3]
            handle_pos_new_ = handle_pos_new_ + handle_pos_mean + handle_pos_new.copy()
                
            new_v = laplacian_surface_editing(mesh, mask_v, boundary_idx=boundary_idx, handle_idx=handle_idx, handle_pos=handle_pos_new_)
            new_v = new_v[f].reshape(-1, 3)
            
            handle_change = False
        
        upd_v = new_v
        # if mask_v is not None:
        #     mmm = mask_v.numpy().repeat(3, axis=1)[f].reshape(-1, 3)
        #     quad[:, :3] = mmm * upd_v + quad[:, :3] * (1-mmm)
        # else:
        #     quad[:, :3] = upd_v
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
            imgui.text("handle_pos_new: {}".format(handle_pos_new))
            imgui.text("handle_pos[0]: {}".format(handle_pos[0]))
            imgui.text("renew[0]: {}".format(mesh.v[0].numpy()+handle_pos_new))
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
            
            h_c_0, handle_pos_new[0] = imgui.slider_float(label="Handle x",   value=handle_pos_new[0], min_value=-2,  max_value= 2,)
            h_c_1, handle_pos_new[1] = imgui.slider_float(label="Handle y",   value=handle_pos_new[1], min_value=-2,  max_value= 2,)
            h_c_2, handle_pos_new[2] = imgui.slider_float(label="Handle z",   value=handle_pos_new[2], min_value=-2,  max_value= 2,)
            
            h_r_0, H_rot_x  = imgui.slider_float(label="Handle Rotate x", value=H_rot_x, min_value=0.0, max_value=360.0,)
            h_r_1, H_rot_y  = imgui.slider_float(label="Handle Rotate y", value=H_rot_y, min_value=0.0, max_value=360.0,)
            h_r_2, H_rot_z  = imgui.slider_float(label="Handle Rotate z", value=H_rot_z, min_value=0.0, max_value=360.0,)
            
            
            
            
            clicked, zoom       = imgui.slider_float(label="Zoom", value=zoom, min_value=0.1,  max_value= 2.0,)
            changed, Reset_button = imgui.menu_item("Reset", None, Reset_button)
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
                normalize   = False
                Reset_button= False
                handle_change = True
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