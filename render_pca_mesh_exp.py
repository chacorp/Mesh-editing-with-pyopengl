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
import cv2
from easydict import EasyDict

# from testwindow import *
from imgui.integrations.glfw import GlfwRenderer
import imgui
import sys

import pickle as pkl
import glm
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex,
)

def load_texture(path):
    texture = glGenTextures(1)
    print("texture buffer: ",texture)
    glBindTexture(GL_TEXTURE_2D, texture)
    image = Image.open(path)
    image = image.convert('RGB') # 'RGBA
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
            if len(values[1].split('/')) > 1:
                ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
                face_texture.append(ft)
    # mesh.v  = np.array(vertex_data)
    mesh.v  = normalize_v(np.array(vertex_data))
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    mesh.ft  = np.array(face_texture) -1
    return mesh

def convert_to_verts_colors(textures_uv, mesh):
    verts = mesh[0]
    faces = mesh[1].verts_idx
    verts_colors_packed = torch.zeros_like(verts)
    verts_colors_packed[faces] = textures_uv.faces_verts_textures_packed()
    return verts_colors_packed

def render(resolution=512, mesh=None):
    rendered    = main(resolution)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('experiment/pg2023_pca')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile    = join(savefolder, 'rendered.png')

    Image.fromarray(rendered).save(savefile)
    return

def main(resolution):
    # bg_path    = "experiment/real/white.png"
    # bg_path    = "white.png"
    # image_path = "Head_Diff_old.jpg"
    # image_path  = "experiment/mesh/girl/0_girl/Head_Diff.png"
    
    ### mask
    # tex_map     = np.array(Image.open(image_path))[..., :3]     # (H, W, 3)
    # tex_map     = torch.tensor(tex_map).unsqueeze(0) * (1/255)
    # texUV       = TexturesUV(verts_uvs=[mesh.vt], faces_uvs=[mesh.ft], maps=tex_map)
    # mesh.upper  = np.array(convert_to_verts_colors(texUV, tmp_obj))
    # print(mesh.upper.shape)
    # import pdb;pdb.set_trace()
    
    character_name = "girl"
    # character_name = "piers"
    # part = "eye_brow_r"
    part = "figure"
    # part = "eye_brow_r"
    # character_name = "piers"
    # character_name = "metahuman"
    
    root_dir = 'experiment/mesh/{0}/0_{0}/'.format(character_name)
    # animation
    # anim_path = 'experiment/mesh/{0}/0_{0}/weight.pth'.format(character_name)
    anim_path = root_dir + 'weight.pth'
    if os.path.exists(anim_path):
        pca_weight = np.array(torch.load(anim_path))
    else:
        anim_path = 'experiment/mesh/{0}/0_{0}/animation.pth'.format(character_name)
        pca_weight = np.array(torch.load(anim_path))
    # pca_weight = np.array(torch.load("D:/test/RaBit/experiment/girl.pth").cpu())   
    # pca_weight = np.array(torch.load("D:/test/RaBit/experiment/piers.pth").cpu())   
    
    pca_value  = np.array(torch.load('experiment/mesh/{0}/0_{0}/eigenvalue.pth'.format(character_name)))
    pca_vector = np.array(torch.load('experiment/mesh/{0}/0_{0}/eigenvector.pth'.format(character_name)))
    pca_mean   = np.array(torch.load('experiment/mesh/{0}/0_{0}/mean.pth'.format(character_name)))
        
    # tmp_obj     = load_obj("data/100/tpose/m.obj")
    tmp_obj     = load_obj("experiment/mesh/{0}/0_{0}/neutral.obj".format(character_name))
    mesh        = EasyDict()
    mesh.v      = tmp_obj[0]
    mesh.f      = tmp_obj[1].verts_idx
    mesh.ft     = tmp_obj[1].textures_idx
    mesh.vt     = tmp_obj[2].verts_uvs
    mesh.vn     = tmp_obj[2].normals
    
    extensions = ['png', 'jpg', 'tif']
    tex_path = ""
    for ext in extensions:
        temp_path = os.path.join(root_dir, f"Head_Diff.{ext}")
        if character_name in ["girl", "piers"]:
            temp_path = os.path.join(root_dir, f"white.{ext}")
            
        if os.path.isfile(temp_path):
            tex_path = temp_path
            print(f"testure image exists: {tex_path}")
            break
        else:
            print(f"testure image doesn't exist: {temp_path}")
        
    # tex_path = "white.png"
    # tex_path = "white3.png"
    bg_path1    = "experiment/real/patch_eyel.png"
    bg_path2    = "experiment/real/patch_eyer.png"
    # bg_path3    = "experiment/real/patch_lips.png"
    bg_path3    = "experiment/real/patch_lips_.png"
    bg_path4    = "experiment/real/patch_eye_brow_l.png"
    bg_path5    = "experiment/real/patch_eye_brow_r.png"
    
    save_path = 'experiment/{}/{}'.format(character_name, part)
    # save_path = 'experiment/{}/{}'.format(character_name, part)
    image_path= save_path + '/image'
    mask_path = save_path + '/mask'
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path,  exist_ok=True)
    
        
    v, f, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    # mean_ = pkl.load(open("./pca_npy/pca_30_v2.pkl",'rb')).mean_.reshape(-1, 3)
    
    if not glfw.init():
        return

    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    
    # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(resolution, resolution, "My OpenGL window", None, None)
    # window = glfw.create_window(int(resolution*1.5), resolution, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    new_vt = vt[ft].reshape(-1, 2)
    new_vt = np.concatenate((
        new_vt, 
        np.ones((new_vt.shape[0], 1))
    ), axis=1)
               
    blendshape = np.zeros_like(pca_value)
    # new_v  = np.zeros_like(new_vt)
    new_v  = v[f].reshape(-1,3)
    new_vn = np.zeros_like(new_v)
    # vn = np.array(vn)[:, ::-1]
    # new_vn = vn[f].reshape(-1,3)
    
    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    quad = np.array(quad, dtype=np.float32)
    print("quad: ", quad.shape)
    
    ### background
    # import pdb;pdb.set_trace()
    bg_quad = np.array([
        #    positions       texture coords       normals
        [-1.0, -1.0, -1.0,   0.0, 0.0, -1.0,   0.0, 0.0, 0.0],   # bottom left
        [ 1.0,  1.0, -1.0,   1.0, 1.0, -1.0,   0.0, 0.0, 0.0],   # top right
        [-1.0,  1.0, -1.0,   0.0, 1.0, -1.0,   0.0, 0.0, 0.0],   # top left 
        
        [-1.0, -1.0, -1.0,   0.0, 0.0, -1.0,   0.0, 0.0, 0.0],   # bottom left
        [ 1.0, -1.0, -1.0,   1.0, 0.0, -1.0,   0.0, 0.0, 0.0],   # bottom right
        [ 1.0,  1.0, -1.0,   1.0, 1.0, -1.0,   0.0, 0.0, 0.0],   # top right
    ], dtype=np.float32)
    print("bg_quad: ", bg_quad.shape)
    
    quad = np.concatenate((bg_quad, quad), axis=0)
    
    ############################################## shader ################
    vertex_shader_source   = open('shader2.vs', 'r').read()
    fragment_shader_source = open('shader2.fs', 'r').read()
    
    vertex_shader   = shaders.compileShader(vertex_shader_source,   GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader          = shaders.compileProgram(vertex_shader, fragment_shader)

    ############################################## buffer ################

    # VAO = glGenBuffers(1)
    # glBindVertexArray(VAO)

    # EBO = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*indices.shape[0]*indices.shape[1], indices, GL_STATIC_DRAW)
    
    """
        GL_STREAM_DRAW:  the data is set only once and used by the GPU at most a few times.
        GL_STATIC_DRAW:  the data is set only once and used many times.
        GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    """
    
    # BG_VBO = glGenBuffers(1)
    # glBindBuffer(GL_ARRAY_BUFFER, BG_VBO)
    # glBufferData(GL_ARRAY_BUFFER, 4*bg_quad.shape[0]*bg_quad.shape[1], bg_quad, GL_STREAM_DRAW)
    
    # bg_vertex_stride = 4 * bg_quad.shape[1]
    
    # position = glGetAttribLocation(shader, "position")
    # glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, bg_vertex_stride, ctypes.c_void_p(0))
    # glEnableVertexAttribArray(position)
    # texcoord = glGetAttribLocation(shader, "texcoord")
    # glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, bg_vertex_stride, ctypes.c_void_p(3*4))
    # glEnableVertexAttribArray(texcoord)
    # normal = glGetAttribLocation(shader, "normal")
    # glVertexAttribPointer(normal,   3, GL_FLOAT, GL_FALSE, bg_vertex_stride, ctypes.c_void_p(2*3*4))
    # glEnableVertexAttribArray(normal)
    
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
    
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
    
    # tangent = glGetAttribLocation(shader, "tangent")
    # glVertexAttribPointer(tangent,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(36))
    # glEnableVertexAttribArray(tangent)
    
    
    ############################################## texture map ###########
    # glEnable(GL_TEXTURE_2D)
    texture0 = load_texture(tex_path)
    texture1 = load_texture(bg_path1)
    texture2 = load_texture(bg_path2)
    texture3 = load_texture(bg_path3)
    texture4 = load_texture(bg_path4)
    texture5 = load_texture(bg_path5)
    # texture1 = load_texture(bg_path)
    # texture2 = load_texture(image_path)
    
    glUseProgram(shader)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture0)
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture1)
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture2)
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture3)
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, texture4)
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, texture5)
    
    glUniform1i(glGetUniformLocation(shader, "texture0"), 0)
    glUniform1i(glGetUniformLocation(shader, "texture1"), 1)
    glUniform1i(glGetUniformLocation(shader, "texture2"), 2)
    glUniform1i(glGetUniformLocation(shader, "texture3"), 3)
    glUniform1i(glGetUniformLocation(shader, "texture4"), 4)
    glUniform1i(glGetUniformLocation(shader, "texture5"), 5)
    ############################################## texture map ###########
    
    ############################################## uniform ###############
    i = 0
    rotation_ = 0
    
    frame = 0
    max_frame = pca_weight.shape[0]-1
    min_frame = 0
    
    scaleX = 1.0
    scaleY = 1.0
    scaleZ = 1.0
    
    transX = 0.0
    transY = 0.0
    transZ = -10.0
    
    Reset_button = False
    show_bg = 1.0
    use_bg = True
    
    tex_alpha = 1.0
    show_m = 1.0
    show_m = True
    
    Save_frame = False
    Save_all = False
    Save_stop = False
    
    region_x = -100
    region_y = -100
    
    select_ldm = False
    normalize = False
    _ratio = 1.0
    
    ldm_shift  = 0
    show_ldm   = False
    m_ldm1     = np.array([-10.0, -10.0, -10.0])
    m_ldm2     = np.array([-10.0, -10.0, -10.0])
    m_ldm3     = np.array([-10.0, -10.0, -10.0])
    m_ldm4     = np.array([-10.0, -10.0, -10.0])
    m_ldm1_idx = 0
    m_ldm2_idx = 0
    m_ldm3_idx = 0
    m_ldm4_idx = 0
    
    transform = glGetUniformLocation(shader, "transform")    
    
    gltrans3fv= glGetUniformLocation(shader, "trans")
    glmouse2fv= glGetUniformLocation(shader, "mouse")
    
    gl_ldm1   = glGetUniformLocation(shader, "m_ldm1")
    gl_ldm2   = glGetUniformLocation(shader, "m_ldm2")
    gl_ldm3   = glGetUniformLocation(shader, "m_ldm3")
    gl_ldm4   = glGetUniformLocation(shader, "m_ldm4")
    glshow_ldm= glGetUniformLocation(shader, "show_ldm")
    
    glView    = glGetUniformLocation(shader, "proj")
    glshow_bg = glGetUniformLocation(shader, "show_bg")
    glshow_m  = glGetUniformLocation(shader, "show_m")
    gl_alpha  = glGetUniformLocation(shader, "_alpha")
    
    # view = glm.ortho(-1.0, 2.0, -1.0, 1.0, 0.0001, 1000.0) 
    zoom = 1.0
    view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
    glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
    ############################################## uniform ###############
    
    ############################################## gl setting ############
    # glBindVertexArray(VAO)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_DEPTH)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
    
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW); 
    # glCullFace(GL_FRONT);
    # glFrontFace(GL_CW); 
    # glCullFace(GL_BACK);
    ############################################## gl setting ############
    
    ############################################### model ################
    # rotation_mat = rotation(np.eye(4, dtype=np.float32), angle*-18.0, 0.0, 1.0, 0.0)
    # rotation_mat = y_rotation(angle*-18.0)
    # rotation_mat = y_rotation(0)
    ############################################### model ################
    
    ############################################## imgui init ############
    use_imgui = True
    if use_imgui:
        imgui.create_context()
        # window = impl_glfw_init() # already initialized!
        # import pdb; pdb.set_trace()
        impl = GlfwRenderer(window)
    ############################################## imgui init ############
        
    ############################################## rescale ###############    
    if character_name == "victor":
        pca_center_diff = np.array([0.0, 0.0, 0.0])
        pca_scale = 0.37
    elif character_name == "girl":
        pca_center_diff = np.array([0.0, 0.0, 0.0])
        pca_scale = 0.45
    elif character_name == "piers":
        pca_center_diff = np.array([0.0, 0.6, 0.0])
        pca_scale = 0.58
    elif character_name == "metahuman":
        pca_center_diff = np.array([0.0, 0.0, 0.0])
        pca_scale = 0.35
    elif character_name == "malcolm": 
        pca_center_diff = np.array([0.0, 0.5, 0.0])
        pca_scale = 0.65
    elif character_name == "mery":
        pca_center_diff = np.array([0.0, -0.5, 0.0])
        pca_scale = 0.55
    elif character_name == "child": 
        pca_center_diff = np.array([0.0, 0.5, 0.0])
        pca_scale = 0.65
    
    pca_mean   = pca_mean.reshape(-1, 3)
    pca_center = pca_mean.mean(0) + pca_center_diff
    pca_mean   = pca_mean - pca_center
    pca_scale  = max(abs(pca_mean).max(0)) * pca_scale
    pca_mean   = pca_mean / pca_scale
    pca_vector = pca_vector / pca_scale
    pca_mean   = pca_mean.reshape(-1)
    ############################################## rescale ###############
        
    # import pdb;pdb.set_trace()  
    while not glfw.window_should_close(window):        
        mouse = np.array([region_x, region_y])
        glUniform2fv(glmouse2fv, 1, mouse)
        
        trans = np.array([transX, transY, transZ])
        glUniform3fv(gltrans3fv, 1, trans)
                        
        glUniform3fv(gl_ldm1, 1, m_ldm1)
        glUniform3fv(gl_ldm2, 1, m_ldm2)
        glUniform3fv(gl_ldm3, 1, m_ldm3)
        glUniform3fv(gl_ldm4, 1, m_ldm4)
        
        glUniform1f(glshow_ldm, show_ldm)
        glUniform1f(glshow_bg,  show_bg)
        glUniform1f(glshow_m,   show_m)
        glUniform1f(gl_alpha,   tex_alpha)
        
        rotation_mat = y_rotation(rotation_)
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        
        view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
        glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
            
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture0)
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture1)
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, texture2)
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, texture3)
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, texture4)
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, texture5)
        
        ## PCA reconstruction
        # frame = 2668
        pca_v = pca_mean + np.dot(pca_vector, pca_weight[frame])
        # import pdb;pdb.set_trace()
        
        ## translate based on upper min
        pca_v = pca_v.reshape(-1,3) # 1402 3
        
        # save_obj(f"{character_name}_{frame}.obj", torch.tensor(pca_v),torch.tensor(f))
        # break
        full_v = pca_v
        if m_ldm1.mean() != -10.0 or m_ldm2.mean() != -10.0:
            if normalize:
                min_x  = min(full_v[m_ldm1_idx][0], full_v[m_ldm2_idx][0])
                max_x  = max(full_v[m_ldm1_idx][0], full_v[m_ldm2_idx][0])
                mean_y = (full_v[m_ldm1_idx][1] + full_v[m_ldm2_idx][1]) * 0.5
                
                # normalize based on x
                full_v[:, 0] = full_v[:,0] - min_x
                full_v[:, 0] = full_v[:, 0] / (max_x - min_x) * 2 - 1
                full_v[:, 0] = full_v[:, 0] * _ratio
                full_v[:, 1] = full_v[:, 1] - mean_y
                
                
                # min_xx = np.array([min_x, min_x])
                # max_xx = np.array([max_x, max_x])
                # # normalize based on x
                # # full_v[:, :2] = full_v[:, :2]
                # mean_y = mean_y - min_x / (max_x - min_x) * 2 - 1
                # full_v[:, :2] = full_v[:, :2] - min_xx / (max_xx - min_xx) * 2 - 1
                # full_v[:, :2] = full_v[:, :2] * _ratio
                # full_v[:, 1] = full_v[:, 1] - mean_y 
                
                # # normalize x y
                # min_x = full_v[[m_ldm1_idx,m_ldm2_idx,m_ldm3_idx,m_ldm4_idx]][:,0].min()
                # max_x = full_v[[m_ldm1_idx,m_ldm2_idx,m_ldm3_idx,m_ldm4_idx]][:,0].max()
                # min_y = full_v[[m_ldm1_idx,m_ldm2_idx,m_ldm3_idx,m_ldm4_idx]][:,1].min()
                # max_y = full_v[[m_ldm1_idx,m_ldm2_idx,m_ldm3_idx,m_ldm4_idx]][:,1].max()
                
                # full_v[:, 0] = full_v[:, 0] / (max_x - min_x) * 2 - 1
                # full_v[:, 1] = full_v[:, 1] / (max_y - min_y) * 2 - 1
                # # full_v[:, :2] = full_v[:, :2] - min_xy / (max_xy - min_xy) # * 2 - 1
                # full_v[:, :2] = full_v[:, :2] * _ratio
                # mean_xy = full_v[[m_ldm1_idx,m_ldm2_idx,m_ldm3_idx,m_ldm4_idx]].mean(0)
                # full_v[:, :2] = full_v[:, :2] - mean_xy[:2]
                                
                # normalize = False
        full_v = full_v * np.array([scaleX, scaleY, scaleZ])
        full_v = full_v + trans
        
        quad[bg_quad.shape[0]:, 0:3] = full_v[f].reshape(-1, 3)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])
        

        glfw.poll_events()
        ############################################## imgui ############
        if use_imgui:
            impl.process_inputs()
            imgui.new_frame()
            # imgui.menu_item(label="(dummy menu)", shortcut=None, selected=False, enabled=False)
            # if imgui.begin_main_menu_bar():
            #     if imgui.begin_menu("File", True):
            #         clicked_quit, selected_quit = imgui.menu_item(
            #             "Quit", "Cmd+Q", False, True
            #         )
            #         if clicked_quit:
            #             sys.exit(0)
            #         imgui.end_menu()
            #     imgui.end_main_menu_bar()
            # rotation_, blendshape = show_example_slider(rotation_, blendshape)
            mean_v = pca_v.mean(0) + trans
            imgui.text("mean vert: {}".format(mean_v))
            
            # pos = imgui.get_cursor_screen_pos()
            region_x = (imgui.get_mouse_position().x / resolution) * 2 - 1
            region_y = (imgui.get_mouse_position().y / resolution) * -2 + 1
            
            if imgui.is_mouse_double_clicked(0) and select_ldm:
                region_xy = np.array([region_x, region_y])
                # print(region_xy.shape)
                condition = np.linalg.norm(full_v[:, :2] - region_xy, axis=1)
                # print(condition.shape)
                show_ldm = True
                if ldm_shift == 0:
                    m_ldm1_idx = condition.argmin()                
                    ldm_shift = ldm_shift +1
                elif ldm_shift == 1:
                    m_ldm2_idx = condition.argmin()                
                    # ldm_shift = ldm_shift +1
                    ldm_shift = 0
                # elif ldm_shift == 2:
                #     m_ldm3_idx = condition.argmin()                
                #     ldm_shift = ldm_shift +1
                # else:
                #     m_ldm4_idx = condition.argmin()                
                #     ldm_shift = 0
                    
            if show_ldm:
                m_ldm1 = full_v[m_ldm1_idx]
                m_ldm2 = full_v[m_ldm2_idx]
                m_ldm3 = full_v[m_ldm3_idx]
                m_ldm4 = full_v[m_ldm4_idx]
            
            imgui.text("ldm_1: {} {}".format(m_ldm1_idx, m_ldm1))
            imgui.text("ldm_2: {} {}".format(m_ldm2_idx, m_ldm2))
            # imgui.text("ldm_3: {} {}".format(m_ldm3_idx, m_ldm3))
            # imgui.text("ldm_4: {} {}".format(m_ldm4_idx, m_ldm4))
            # imgui.text("region_x: {}".format(region_x))
            # imgui.text("region_y: {}".format(region_y))

            clicked_ldm_, select_ldm= imgui.menu_item("Select landmark", None, select_ldm)
            clicked_ldm_, show_ldm  = imgui.menu_item("Show landmark", None, show_ldm)
            clicked_bg, show_m      = imgui.menu_item("Show Model", None, show_m)
            if show_m:
                show_m = 1
            else:
                show_m = 0
             
            clicked_bg, use_bg      = imgui.menu_item("Show reference", None, use_bg)
            changed, show_bg        = imgui.input_int(label="reference", value=show_bg, step=1)
            if show_bg > 5:
                show_bg = 0 
            if use_bg:
                if show_bg == 0:
                    show_bg = 1
            else:
                show_bg = 0
            
            clicked, tex_alpha = imgui.slider_float(label="_alpha",    value=tex_alpha, min_value=0.0, max_value=1.0)
            clicked, _ratio    = imgui.slider_float(label="_ratio",    value=_ratio, min_value=0.0, max_value=1.0)
            clicked, frame     = imgui.slider_int(label="frame",       value=frame, min_value=min_frame, max_value=max_frame)
            changed, frame     = imgui.input_int(label="select frame", value=frame, step=1)
            # clicked, rotation_ = imgui.slider_float(label="Rotate", value=rotation_, min_value=0.0, max_value=360.0,)
            
            if frame < 0:
                frame = 0
            if frame > max_frame:
                frame = max_frame
            
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
            
            clicked, zoom   = imgui.slider_float(label="Zoom", value=zoom, min_value=0.1,  max_value= 2.0,)
            
            changed, Reset_button     = imgui.menu_item("Reset", None, Reset_button)
            clicked_save1, Save_frame = imgui.menu_item("Save frame", None, Save_frame)
            clicked_save2, Save_all   = imgui.menu_item("Save all frames", None, Save_all)
            clicked_save3, Save_stop  = imgui.menu_item("Stop save", None, Save_stop)
            clicked_norm, normalize   = imgui.menu_item("Normalize", None, normalize)

                                         
            if Reset_button:
                blendshape  = blendshape * 0
                zoom        = 1.0
                scaleX      = 1.0
                scaleY      = 1.0
                scaleZ      = 1.0
                transX      = 0.0
                transY      = 0.0
                transZ      = -10.0
                frame       = 0
                rotation_   = 0
                normalize   = False
                show_ldm    = False
                select_ldm  = False
                Reset_button= False
            
            if Save_stop:
                Save_frame = False
                Save_all   = False
                Save_stop  = False
                
            if Save_frame:
                # glfw.swap_buffers(window)
                glReadBuffer(GL_FRONT)
                pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
                a = np.frombuffer(pixels, dtype=np.uint8)
                a = a.reshape((resolution, resolution, 4))
                
                save_path = 'experiment/{}/{}'.format(character_name,part)
                image_path= save_path + '/image'
                mask_path = save_path + '/mask'
                os.makedirs(image_path, exist_ok=True)
                os.makedirs(mask_path,  exist_ok=True)
                Image.fromarray(a[::-1, :, :3]).save('{}/local_patch{:06}.png'.format(image_path, frame))
                Image.fromarray(a[::-1, :,  3]).save('{}/local_patch_mask{:06}.png'.format(mask_path, frame))
                
                Save_frame = False
                print("frame: {}\nscale x: {}\nscale y: {}\ntrans x: {}\ntrans y: {}\nratio: {}\nLDM 1: {}\nLDM 2: {}\n".format(
                    frame, scaleX, scaleY, transX, transY, _ratio, m_ldm1_idx, m_ldm2_idx))
                save_obj(f"{character_name}_{frame}.obj", 
                         torch.tensor(pca_v),
                         mesh.f.clone().detach(),
                         verts_uvs=mesh.vt.clone().detach(),
                         faces_uvs=mesh.ft.clone().detach())
                # frame = frame + 1
            
            if Save_all:
                glReadBuffer(GL_FRONT)
                pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
                a = np.frombuffer(pixels, dtype=np.uint8)
                a = a.reshape((resolution, resolution, 4))
                
                Image.fromarray(a[::-1, :, :3]).save('{}/local_patch{:06}.png'.format(image_path, frame))
                Image.fromarray(a[::-1, :,  3]).save('{}/local_patch_mask{:06}.png'.format(mask_path, frame))
                
                frame = frame + 1
                if frame >= max_frame:
                    Save_all = False
                    
            rotation_ = rotation_ % 360
                        
            ######################################## blendshape model #######
            blendshape = pca_weight[frame]
            for bs in range(len(blendshape)):    
                clicked, blendshape[bs] = imgui.slider_float(
                    label = "blendshape"+ str(bs),
                    value = blendshape[bs],
                    min_value = -30.0,
                    max_value =  30.0,
                )
            ######################################## blendshape model #######
                
            ############################################## imgui ############
            # show_test_window()
            imgui.render()
            impl.render(imgui.get_draw_data())
            ############################################## imgui ############
        
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
    # print("frame: {}\nscale x: {}\nscale y: {}\ntrans x: {}\ntrans x: {}\n".format(frame, scaleX, scaleY, transX, transY))
    return a

if __name__ == '__main__':
    render(resolution=256*2)
    # data_render(resolution=512)

