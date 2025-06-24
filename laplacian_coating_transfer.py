from __future__ import print_function

import os
from os.path import exists, join
# from glob import glob
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
from pytorch3d.io import load_objs_as_meshes
import torch

import scipy.sparse
import numpy as np

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

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32)
    A  = torch.sparse.FloatTensor(idx, ones, (V, V))

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
    idx = torch.arange(V)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L, A

def laplacian_matrix_ring(mesh):
    """
    Reference: 
        - Laplacian Surface Editing, Olga Sorkine, Daniel Cohen-Or, 2004
        - https://github.com/luost26/laplacian-surface-editing/blob/master/main.py

    Args:
        mesh (EasyDict): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    V = mesh.v
    N = mesh.v.shape[0]
    
    # laplacian matrix
    L = mesh.L.to_dense().numpy()
    
    # laplacian coordinates :: Delta = L @ V
    Delta = np.array(mesh.Delta)

    LS = np.zeros([3*N, 3*N])
    LS[0*N:1*N, 0*N:1*N] = (-1) * L
    LS[1*N:2*N, 1*N:2*N] = (-1) * L
    LS[2*N:3*N, 2*N:3*N] = (-1) * L
    
    # compute the l
    for i in range(N):
        nb_idx      = mesh.ring_indices[i]
        ring        = np.array([i] + nb_idx)
        n_ring      = len(ring)
        V_ring      = V[ring]
        
        Ai          = np.zeros([n_ring * 3, 7])
        zer0_ring   = np.zeros(n_ring)
        
        Ai[:n_ring,         0:4] = np.stack([V_ring[:,0],    zer0_ring,  V_ring[:,2], -V_ring[:,1]], axis=1)
        Ai[n_ring:2*n_ring, 0:4] = np.stack([V_ring[:,1], -V_ring[:,2],    zer0_ring,  V_ring[:,0]], axis=1)
        Ai[2*n_ring:,       0:4] = np.stack([V_ring[:,2],  V_ring[:,1], -V_ring[:,0],    zer0_ring], axis=1)

        Ai[:n_ring,           4] = 1
        Ai[n_ring:2*n_ring,   5] = 1
        Ai[2*n_ring:,         6] = 1
        
        # Moore-Penrose Inversion
        Ai_pinv = np.linalg.pinv(Ai)
        s = Ai_pinv[0]
        h = Ai_pinv[1:4]
        # t = Ai_pinv[4:7]
        # import pdb; pdb.set_trace()
        
        Ti = np.array([
            [    s , -h[2],  h[1], ], #  t[0],
            [  h[2],    s , -h[0], ], #  t[1],
            [ -h[1],  h[0],    s , ], #  t[2],
            # [0,  0,  0,  1]
        ])
        T_delta = (Delta[i].T @ Ti )
        
        rind_idxs = np.hstack([ring, ring+N, ring+2*N])
        # LS[[i, i+N, i+2*N]][:, rind_idxs] += T_delta
        LS[i,     rind_idxs] += T_delta[0]
        LS[i+N,   rind_idxs] += T_delta[1]
        LS[i+2*N, rind_idxs] += T_delta[2]
        
    return LS

def laplacian_surface_editing(
        mesh, 
        boundary_idx = [3, 13, 481] + [28 + i*15 for i in range(20)] + [329 + i*15 for i in range(10)],
        handle_idx   = [325],
        handle_pos   = np.array([0.5, 0.5, 0.5]),
    ):
    """
    Args:
        mesh (EasyDict):        mesh properties
        boundary_idx (list):    vertex indices of boundary
        handle_idx (list):      vertex indices of handle
        handle_pos (np.array):  displacement of handle

    Returns:
        new_verts (np.array):   new vertices after laplacian surface editing
    """
    V  = mesh.v
    N  = mesh.v.shape[0]
    LS = mesh.LS
    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef = np.zeros([len(boundary_idx)*3+ len(handle_idx)*3, 3*N])
    constraint_b    = np.zeros([len(boundary_idx)*3+ len(handle_idx)*3])

    # Boundary constraints
    i = 0
    for idx in boundary_idx:
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Boundary constraint
        idx_v = [idx, idx+N, idx+2*N] # index of v in V
        
        constraint_coef[idx_c, idx_v] = 1.0 # one-hot
        constraint_b[idx_c]           = V[idx] # fixed position
        
        # constraint_coef[i*3+0] = (_array_ == idx)       * 1.0
        # constraint_coef[i*3+1] = (_array_ == idx + N)   * 1.0
        # constraint_coef[i*3+2] = (_array_ == idx + 2*N) * 1.0
        # constraint_b[i*3+0]    = V[idx, 0]
        # constraint_b[i*3+1]    = V[idx, 1]
        # constraint_b[i*3+2]    = V[idx, 2]
        i = i + 1
        
    # Handle constraints
    for idx in handle_idx:
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Handle constraint
        idx_v = [idx, idx+N, idx+2*N] # index of v in V
        
        constraint_coef[idx_c, idx_v] = 1.0 # one-hot
        constraint_b[idx_c]           = V[idx] + handle_pos
        
        # constraint_coef[i*3+0] = (_array_ == idx)       * 1.0
        # constraint_coef[i*3+1] = (_array_ == idx + N)   * 1.0
        # constraint_coef[i*3+2] = (_array_ == idx + 2*N) * 1.0
        # constraint_b[i*3+0]    = V[idx, 0] + handle_pos[0]
        # constraint_b[i*3+1]    = V[idx, 1] + handle_pos[1]
        # constraint_b[i*3+2]    = V[idx, 2] + handle_pos[2]
        i = i + 1

    # -------------------------- Solve the Linear System ------------------------- #
    # import pdb;pdb.set_trace()
    # L = mesh.L.to_dense().numpy()
    A        = np.vstack([LS,         constraint_coef])
    b        = np.hstack([np.zeros(3*N), constraint_b])
    spA      = scipy.sparse.coo_matrix(A)

    V_prime  = scipy.sparse.linalg.lsqr(spA, b)[0]
    
    new_verts = V_prime.reshape(3, -1).T
    # new_verts[handle_idx] = mesh.v[handle_idx] + handle_pos
    
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
            
            if len(values[1].split('/')) >=2 and values[1].split('/')[1] != '':
                ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
                face_texture.append(ft)
            if len(values[1].split('/')) >=3 and values[1].split('/')[2] != '':
                fn = list(map(lambda x: int(x.split('/')[2]),  values[1:]))
                face_normal.append(fn)
    # mesh.v  = np.array(vertex_data)
    mesh.v  = normalize_np(vertex_data)
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    mesh.ft = np.array(face_texture) -1
    mesh.fn = np.array(face_normal) -1
    return mesh

def compute_normals(v, f):
    v_ = v[f]
    fn = np.cross(v_[:, 1] - v_[:, 0], v_[:, 2] - v_[:,0])

    fs = np.sqrt(np.sum((v_[:, [1, 2, 0], :] - v_) ** 2, axis = 2))
    fs_ = np.sum(fs, axis = 1) / 2 # heron's formula
    fa = np.sqrt(fs_ * (fs_ - fs[:, 0]) * (fs_ - fs[:, 1]) * (fs_ - fs[:, 2]))[:, None]

    vn = np.zeros_like(v, dtype = np.float32)
    vn[f[:, 0]] += fn * fa # weight by area
    vn[f[:, 1]] += fn * fa
    vn[f[:, 2]] += fn * fa

    vlen = np.sqrt(np.sum(vn ** 2, axis = 1))[np.any(vn != 0, axis = 1), None]
    vn[np.any(vn != 0, axis = 1), :] /= vlen
    return vn 

def main(resolution=512):
    ### Laplacian from mesh 1
    path                = "experiment/faces/carell.obj"
    mesh1                = load_obj_mesh(path)
    meshes1              = load_objs_as_meshes([path])
    mesh1.v              = normalize_torch(meshes1.verts_packed())
    mesh1.e              = meshes1.edges_packed()
    mesh1.L              = meshes1.laplacian_packed()
    
    mesh1.Delta          = mesh1.L.mm(mesh1.v)
    
    ### Laplacian from mesh 2
    path2               = "experiment/faces/carell_style.obj"
    mesh2               = load_obj_mesh(path2)
    meshes2             = load_objs_as_meshes([path2])
    mesh2.v             = normalize_torch(meshes2.verts_packed())
    mesh2.e             = meshes2.edges_packed()
    mesh2.L             = meshes2.laplacian_packed()
    # import pdb;pdb.set_trace()
    
    mesh2.Delta         = mesh2.L.mm(mesh2.v)

    ### Coating Transfer    
    mesh1.Xi =  mesh2.Delta - mesh1.Delta
    mesh1.Linv = torch.inverse(mesh1.L.to_dense())
    
    ### target mesh
    path                = "experiment/faces/jayz.obj"
    # path                = "experiment/faces/carell.obj"
    mesh                = load_obj_mesh(path)
    mesh.v              = mesh.v + np.array(mesh1.Linv.mm(mesh1.Xi))
    # mesh.v              = mesh.v + np.array(mesh1.Linv.mm(mesh1.Xi*0.5))
    
    mesh.vn             = compute_normals(mesh.v, mesh.f)
        
    image_path  = "data/white.png"

    rendered    = render(mesh, resolution, image_path)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('LSE')
    if not exists(savefolder):
        os.makedirs(savefolder)
    
    savefile    = join(savefolder, 'coating_transfer-exp.png')

    Image.fromarray(rendered).save(savefile)
    return
 
def render(mesh, 
         resolution, 
         image_path, 
         disp_path=None, 
         json_object=None, 
         boundary_mask_path=None, 
         timer=False, 
         mean=None, 
         coef=None, 
         basis=None
         ):
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

    # new_vt = vt[ft].reshape(-1,2)
    # new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    new_v  = v[f].reshape(-1, 3)
    new_vn = vn[f].reshape(-1, 3)
    # new_vn = np.zeros_like(new_v)
    # new_vn[:,0] = vertex_normal(new_v[:,0], new_v[:,1], new_v[:,2])
    # new_vn[:,1] = vertex_normal(new_v[:,1], new_v[:,2], new_v[:,0])
    # new_vn[:,2] = vertex_normal(new_v[:,2], new_v[:,0], new_v[:,1])
    # new_vn = new_vn.reshape(-1, 3)
    # new_v  = new_v.reshape(-1, 3)
    
    # import pdb;pdb.set_trace()
    # if f.max() == vn.shape[0]:
    # if True:
    #     new_vn = vn[mesh.fn].reshape(-1, 3)
    # else:
    #     new_vn = vn[f].reshape(-1, 3)
    new_vt = np.ones_like(new_v)
    # new_vn = np.zeros_like(new_v)
    
    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    quad = np.array(quad, dtype=np.float32)

    ############################################## shader ################
    vertex_shader_source   = open('shader/shader.vs', 'r').read()
    fragment_shader_source = open('shader/shader.fs', 'r').read()
    
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
    transZ = 0.0
    
    Reset_button = False
    
    tex_alpha = 1.0
    
    handle_pos_old = np.array([0.0, 0.0, 0.0])
    handle_pos_new = np.array([0.0, 0.0, 0.0])
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
        # if handle_change:
        #     # print('1')
        #     new_v = laplacian_surface_editing(mesh, handle_pos=handle_pos_new)
        #     new_v = new_v[f].reshape(-1, 3)
        #     handle_pos_old = handle_pos_new
        #     handle_change = False
        
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
            # imgui.text("handle_pos_new: {}".format(handle_pos_new))
            # imgui.text("handle_pos_old: {}".format(handle_pos_old))
            
            clicked, tex_alpha  = imgui.slider_float(label="_alpha",    value=tex_alpha, min_value=0.0, max_value=1.0)
            clicked, _ratio     = imgui.slider_float(label="_ratio",    value=_ratio, min_value=0.0, max_value=1.0)
            clicked, rotation_  = imgui.slider_float(label="Rotate", value=rotation_, min_value=0.0, max_value=360.0,)
            
            clicked, scaleX     = imgui.slider_float(label="Scale x",   value=scaleX, min_value= 0.1,  max_value= 10.0,)
            changed, scaleX     = imgui.input_float(label="set Scale x",value=scaleX, step=0.001)
            clicked, scaleY     = imgui.slider_float(label="Scale y",   value=scaleY, min_value= 0.1,  max_value= 10.0,)
            changed, scaleY     = imgui.input_float(label="set Scale y",value=scaleY, step=0.001)
            clicked, scaleZ     = imgui.slider_float(label="Scale z",   value=scaleZ, min_value= 0.1,  max_value= 10.0,)
            
            clicked, transX     = imgui.slider_float(label="Trans x",   value=transX, min_value=-5.0,  max_value= 5.0,)
            changed, transX     = imgui.input_float(label="set Trans x",value=transX, step=0.001)
            clicked, transY     = imgui.slider_float(label="Trans y",   value=transY, min_value=-5.0,  max_value= 5.0,)
            changed, transY     = imgui.input_float(label="set Trans y",value=transY, step=0.001)
            clicked, transZ     = imgui.slider_float(label="Trans z",   value=transZ, min_value=-100,  max_value= 10,)
            changed, transZ     = imgui.input_float(label="set Trans z",value=transZ, step=0.001)
            
            # h_c_0, handle_pos_new[0] = imgui.slider_float(label="Handle x",   value=handle_pos_new[0], min_value=-2,  max_value= 2,)
            # h_c_1, handle_pos_new[1] = imgui.slider_float(label="Handle y",   value=handle_pos_new[1], min_value=-2,  max_value= 2,)
            # h_c_2, handle_pos_new[2] = imgui.slider_float(label="Handle z",   value=handle_pos_new[2], min_value=-2,  max_value= 2,)
            # if h_c_0 or h_c_1 or h_c_2:
            #     handle_change = True
            
            clicked, zoom       = imgui.slider_float(label="Zoom", value=zoom, min_value=0.1,  max_value= 2.0,)
            changed, Reset_button = imgui.menu_item("Reset", None, Reset_button)
            if json_object:
                clicked_use_disp, use_disp   = imgui.menu_item("Use_disp", None, use_disp)
                clicked_use_tex, onoff_tex   = imgui.menu_item("OnOff Tex", None, onoff_tex)

                if clicked_use_disp:
                    use_disp != use_disp
                if clicked_use_tex:
                    onoff_tex != onoff_tex
                            
            if Reset_button:
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
                handle_change = True
                handle_pos_new = np.array([0.0, 0.0, 0.0])
                                
            rotation_ = rotation_ % 360
                        
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