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

import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex,
)

def load_texture(path):
    texture = glGenTextures(1)
    print("texture buffer:", texture)
    glBindTexture(GL_TEXTURE_2D, texture)
    image = Image.open(path)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB') # 'RGBA
    image_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
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
    savefolder  = join('experiment/output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile    = join(savefolder, 'rendered.png')

    Image.fromarray(rendered).save(savefile)
    return

def main(resolution):
    tmp_obj = load_obj("data/100/tpose/m.obj")

    mesh    = EasyDict()
    mesh.v  = tmp_obj[0]
    mesh.f  = tmp_obj[1].verts_idx
    mesh.ft = tmp_obj[1].textures_idx
    mesh.vt = tmp_obj[2].verts_uvs
    mesh.vn = tmp_obj[2].normals
        
    # image_path  = "white.png"
    image_path  = "experiment/test/tex.png"
    
    tex_map   = np.array(Image.open("experiment/test/upper_mask_.png"))[..., :3]     # (H, W, 3)
    tex_map   = torch.tensor(tex_map).unsqueeze(0) * (1/255)
    texUV     = TexturesUV(verts_uvs=[mesh.vt], faces_uvs=[mesh.ft], maps=tex_map)
    mesh.upper= np.array(convert_to_verts_colors(texUV, tmp_obj))
    # print(mesh.upper.shape)
    # import pdb;pdb.set_trace()
    
    upper_pca = pkl.load(open("./pca_npy/pca_30_upper_v1.pkl",'rb'))
    upper_mean  = upper_pca.mean_
    upper_coef  = upper_pca.explained_variance_
    upper_basis = upper_pca.components_
    
    lower_pca = pkl.load(open("./pca_npy/pca_30_lower_v1.pkl",'rb'))
    lower_mean  = lower_pca.mean_
    lower_coef  = lower_pca.explained_variance_
    lower_basis = lower_pca.components_
        
    v, f, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    mean_ = pkl.load(open("./pca_npy/pca_30_v2.pkl",'rb')).mean_.reshape(-1, 3)
    
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
               
    blendshape = np.zeros_like(upper_coef)
    lower_blendshape = np.zeros_like(lower_coef)
    new_v  = np.zeros_like(new_vt) # dummy
    
    # new_vn = vn[f].reshape(-1, 3)
    new_vn = np.zeros_like(new_v)
    
    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    # quad = np.concatenate( (new_v, new_vt, new_vn, new_vtan), axis=1)
    quad = np.array(quad, dtype=np.float32)
    # print(quad.shape)

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
    
    # tangent = glGetAttribLocation(shader, "tangent")
    # glVertexAttribPointer(tangent,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(36))
    # glEnableVertexAttribArray(tangent)
    
    ############################################## texture map ###########
    # glEnable(GL_TEXTURE_2D)
    texture1 = load_texture(image_path)
    # texture2 = load_texture(normal_path)
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1)
    
    # glActiveTexture(GL_TEXTURE1);
    # glBindTexture(GL_TEXTURE_2D, texture2)
        
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "texture1"), 0)
    # glUniform1i(glGetUniformLocation(shader, "texture2"), 1)
    
    ############################################## render ################
    # glBindVertexArray(VAO)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)

    ############################################## camera ################
    # rotation_mat = rotation(np.eye(4, dtype=np.float32), angle*-18.0, 0.0, 1.0, 0.0)
    # rotation_mat = y_rotation(angle*-18.0)
    # rotation_mat = y_rotation(0)

    transform = glGetUniformLocation(shader, "transform")    
    # gltimey = glGetUniformLocation(shader, "timer_y")
    # gltimex = glGetUniformLocation(shader, "timer_x")

    ############################################## imgui init ############
    imgui.create_context()
    # window = impl_glfw_init() # already initialized!
    impl = GlfwRenderer(window)
    ############################################## imgui init ############
    
    i = 0
    rotation_angle = 180
    scale = 1.0
    Reset_button = False
    
    ## boolean mask for upper body
    upper_bool_mask = (mesh.upper>0).all(1)
    upper_v = mean_[upper_bool_mask]
    upper_v_y_argmin = np.argmin(upper_v[:, 1])
    upper_v_y_min = upper_v[upper_v_y_argmin, 1]
    upper_trans_limit = upper_v_y_min.copy()
    
    ## boolean mask for lower body
    lower_bool_mask = (mesh.upper==0).all(1)
    lower_v = mean_[lower_bool_mask]
    lower_v_y_argmax = np.argmax(lower_v[:, 1])
    lower_v_y_max = lower_v[lower_v_y_argmax, 1]
    lower_trans_limit = lower_v_y_max.copy()
    
    upper_disp_v = upper_mean.reshape(-1,3)
    upper_v_y_argmin = np.argmin(upper_disp_v[:, 1])
    
    lower_disp_v = lower_mean.reshape(-1,3)
    lower_v_y_argmax = np.argmax(lower_disp_v[:, 1])
    
    while not glfw.window_should_close(window):
        
        upper_disp_v = upper_mean + np.dot(upper_coef * blendshape, upper_basis)
        lower_disp_v = lower_mean + np.dot(lower_coef * lower_blendshape, lower_basis)
        
        ## translate based on upper min
        upper_disp_v = upper_disp_v.reshape(-1,3)
        lower_disp_v = lower_disp_v.reshape(-1,3)
                
        
        upper_disp_min = (upper_disp_v[upper_v_y_argmin,1]).min()
        upper_disp_v[:,1] = upper_disp_v[:,1] - upper_disp_min + upper_v_y_min
        
        # disp_min = (disp_v[upper_bool_mask][:,1]).min()
        # lower_disp_max = (lower_disp_v[lower_v_y_argmax,1]).min()
        # lower_disp_v[:,1] = lower_disp_v[:,1] - lower_disp_max + lower_v_y_max

        full_v = lower_disp_v * (1-mesh.upper) + upper_disp_v * mesh.upper
        full_v = normalize_v(full_v) * scale
        quad[:, 0:3] = full_v[f].reshape(-1, 3)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
            
        rotation_mat = y_rotation(rotation_angle)
        
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.poll_events()
        ############################################## imgui ############
        impl.process_inputs()
        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", "Cmd+Q", False, True
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()
            
        # rotation_angle, blendshape = show_example_slider(rotation_angle, blendshape)
        # imgui.menu_item(label="(dummy menu)", shortcut=None, selected=False, enabled=False)
        imgui.text("disp_min: {}".format(upper_disp_min))
        imgui.text("upper_v_min: {}".format(upper_v_y_min))
        
        clicked, upper_v_y_min = imgui.slider_float(
                label="upper_body",
                value=upper_v_y_min,
                min_value = upper_trans_limit,
                max_value = 1.0,
            )
        
        clicked, rotation_angle = imgui.slider_float(
                label="Rotate",
                value=rotation_angle,
                min_value = 0.0,
                max_value = 360.0,
            )
        clicked, scale = imgui.slider_float(
                label="Scale",
                value=scale,
                min_value = 0.1,
                max_value = 2.0,
            )
        clicked_quit, Reset_button = imgui.menu_item(
            "Reset", None, Reset_button
            )
        if Reset_button:
            blendshape = blendshape * 0
            lower_blendshape = lower_blendshape * 0
            upper_v_y_min = upper_trans_limit
            scale = 1.0
            Reset_button = False
            
        for i in range(len(blendshape)):    
            clicked, blendshape[i] = imgui.slider_float(
                label="Upper Value"+ str(i),
                value=blendshape[i],
                min_value = -1.0,
                max_value =  1.0,
            )
        for i in range(len(lower_blendshape)):    
            clicked, lower_blendshape[i] = imgui.slider_float(
                label="Lower Value"+ str(i),
                value=lower_blendshape[i],
                min_value = -1.0,
                max_value =  1.0,
            )
            
        rotation_angle = rotation_angle % 360
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
    impl.shutdown()
    glfw.terminate()
    return a

if __name__ == '__main__':
    render(resolution=512)
    # data_render(resolution=512)

