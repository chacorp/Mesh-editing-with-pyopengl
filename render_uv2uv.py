from __future__ import print_function

import os
from os.path import exists, join

import sys
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import glm

from imgui.integrations.glfw import GlfwRenderer
import imgui

from utils.util import *


def load_texture(path):
    texture = glGenTextures(1)
    print("texture buffer: ",texture)
    glBindTexture(GL_TEXTURE_2D, texture)
    image = Image.open(path)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB') # 'RGBA
    #print(image.shape)
    image_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    return texture

def load_obj_mesh(mesh_path, norm=True):
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
    if norm:
        mesh.v  = normalize_v(np.array(vertex_data))
    else:
        mesh.v  = np.array(vertex_data)
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    mesh.ft = np.array(face_texture) -1
    mesh.fn = np.array(face_normal) -1
    return mesh

def main(from_mesh, to_mesh, image_path, resolution=1024):
    
    ############################################## window init ###########
    if not glfw.init():
        return

    window = glfw.create_window(resolution, resolution, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    ######################################################################
        
    
    mesh = load_obj_mesh(from_mesh)
    mesh2 = load_obj_mesh(to_mesh)
    
    v, f, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    v2, f2, vt2, ft2, vn2 = mesh2.v, mesh2.f, mesh2.vt, mesh2.ft, mesh2.vn
    
    # v[:,1]=v[:,1]*-1
    # v[:,2]=v[:,2]*-1
    #new_v  = v[f].reshape(-1, 3) # 3D vertex position (from_mesh)
    new_vt = vt[ft].reshape(-1,2) # UV position (from_mesh)
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    
    #new_v2  = v2[f2].reshape(-1, 3) # 3D vertex position (to_mesh)
    new_vt2 = vt2[ft2].reshape(-1,2) # UV position (to_mesh)
    new_vt2 = np.concatenate((new_vt2, np.zeros((new_vt2.shape[0],1)) ), axis=1)
    
    
    if f.max() == vn.shape[0]:
        new_vn = vn[mesh.fn].reshape(-1, 3)
    else:
        fn = compute_face_norm(mesh.v, mesh.f)
        vn = compute_vert_norm(torch.from_numpy(mesh.f).type(torch.int64), torch.from_numpy(fn)[None]).squeeze(0)
        vn = np.array(vn)
        new_vn = vn[mesh.f].reshape(-1, 3)
    
    
    ############# needs to be arranged via correspondance ################
    quad = np.concatenate( (new_vt2, new_vt, new_vn), axis=1)
    quad = np.array(quad, dtype=np.float32)
    ######################################################################



    ############################################## shader ################
    vertex_shader_source   = open('shader_uv.vs', 'r').read()
    fragment_shader_source = open('shader_uv.fs', 'r').read()
    
    vertex_shader   = shaders.compileShader(vertex_shader_source,   GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader          = shaders.compileProgram(vertex_shader, fragment_shader)
    ######################################################################
    

    ############################################## buffer ################
    """
        GL_STREAM_DRAW:  the data is set only once and used by the GPU at most a few times.
        GL_STATIC_DRAW:  the data is set only once and used many times.
        GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    """
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
    
    # 4*3*3 : size of float * len(X,Y,Z) * len(pos, tex, nor)
    vertex_stride = 4 * quad.shape[1]
    
    to_coord = glGetAttribLocation(shader, "to_coord")
    glVertexAttribPointer(to_coord, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(to_coord)
    
    from_coord = glGetAttribLocation(shader, "from_coord")
    glVertexAttribPointer(from_coord, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(from_coord)

    normal = glGetAttribLocation(shader, "normal")
    glVertexAttribPointer(normal,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(normal)
    ######################################################################

    ############################################## texture map ###########
    glUseProgram(shader)
    texture0 = load_texture(image_path)
    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture0)
    
    
    glUniform1i(glGetUniformLocation(shader, "texture0"), 0)    
    glUniform1i(glGetUniformLocation(shader, "textureB"), 1)
    ######################################################################
    
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
    
    show_m = True
    
    region_x = -100
    region_y = -100
    
    select_ldm = False
    normalize = False
    _ratio = 1.0
    
    ldm_shift  = 0
    show_ldm   = False
    m_ldm1_idx = 0
    m_ldm2_idx = 0
    m_ldm3_idx = 0
    m_ldm4_idx = 0
    m_ldm1 = np.array([-10.0, -10.0, -10.0])
    m_ldm2 = np.array([-10.0, -10.0, -10.0])
    m_ldm3 = np.array([-10.0, -10.0, -10.0])
    m_ldm4 = np.array([-10.0, -10.0, -10.0])
    
    
    scale = glGetUniformLocation(shader, "scale") # scale rotate translate
    transform = glGetUniformLocation(shader, "transform")
    translate = glGetUniformLocation(shader, "trans")
    
    glmouse2fv = glGetUniformLocation(shader, "mouse")
        
    glView = glGetUniformLocation(shader, "proj")
    gl_alpha = glGetUniformLocation(shader, "_alpha")
    
    glshow_m = glGetUniformLocation(shader, "show_m")
    
    # view = glm.ortho(-1.0, 2.0, -1.0, 1.0, 0.0001, 1000.0) 
    zoom = 1.0
    view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.000001, 100.0) 
    glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
    ######################################################################

    ############################################## gl setting ############
    # glBindVertexArray(VAO)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND);
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    # glEnable(GL_CULL_FACE);
    # glFrontFace(GL_CCW); 
    ######################################################################

    ############################################## imgui #################
    use_imgui = False
    if use_imgui:
        imgui.create_context()
        # window = impl_glfw_init() # already initialized!
        impl = GlfwRenderer(window)
    ############################################## imgui init ############    
        
    while not glfw.window_should_close(window):
        mouse = np.array([region_x, region_y])
        glUniform2fv(glmouse2fv, 1, mouse)
        
        glUniform1f(glshow_m,   show_m)
        
        # ---------------------------------- Update 3D mesh ----------------------------------
        
        ## trans * rotate * scale * 3D model 
        rotation_mat = y_rotation(rotation_)
        affine_mat = np.eye(4)
        affine_mat[:3,:3] = affine_mat[:3,:3] * np.array([scaleX, scaleY, scaleZ])
        trans = np.array([transX, transY, transZ])
        # affine_mat[:3,-1] = trans
        rotation_mat = rotation_mat @ affine_mat.T
        
        # glUniformMatrix4fv(scale, 1, GL_FALSE, affine_mat)
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        glUniform3fv(translate, 1, trans)
        
        view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
        glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
        
        upd_v = new_vt2
        upd_v = np.hstack((upd_v, np.ones((upd_v.shape[0], 1))))
        if m_ldm1.mean() != -10.0 or m_ldm2.mean() != -10.0:
            if normalize:
                min_x  = min(upd_v[m_ldm1_idx][0], upd_v[m_ldm2_idx][0])
                max_x  = max(upd_v[m_ldm1_idx][0], upd_v[m_ldm2_idx][0])
                mean_y = (upd_v[m_ldm1_idx][1] + upd_v[m_ldm2_idx][1]) * 0.5
                
                # normalize based on x
                upd_v[:, 0] = upd_v[:, 0] - min_x
                upd_v[:, 0] = upd_v[:, 0] / (max_x - min_x) * 2 - 1
                upd_v[:, 1] = upd_v[:, 1] - mean_y
                        
        ## apply transform
        upd_v = np.dot(upd_v, rotation_mat.T)[:,:3] + trans
        quad[:, :3] = upd_v
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])
        # ------------------------------------------------------------------------------------

        glfw.poll_events()
        if use_imgui:
            impl.process_inputs()
            imgui.new_frame()
            imgui.text("mean vert: {}".format(upd_v.mean(0)))
            imgui.text("min vert: {}".format(upd_v.min()))
            imgui.text("max vert: {}".format(upd_v.max()))
            
            region_x = (imgui.get_mouse_position().x / resolution) * 2 - 1
            region_y = (imgui.get_mouse_position().y / resolution) * -2 + 1
            
            if imgui.is_mouse_double_clicked(0) and select_ldm:
                region_xy = np.array([region_x, region_y])
                condition = np.linalg.norm(upd_v[:, :2] - region_xy, axis=1)
                show_ldm = True
                if ldm_shift == 0:
                    m_ldm1_idx = condition.argmin()                
                    ldm_shift = ldm_shift +1
                elif ldm_shift == 1:
                    m_ldm2_idx = condition.argmin()                
                    ldm_shift = 0
            if show_ldm:
                m_ldm1 = upd_v[m_ldm1_idx]
                m_ldm2 = upd_v[m_ldm2_idx]
                m_ldm3 = upd_v[m_ldm3_idx]
                m_ldm4 = upd_v[m_ldm4_idx]
                    
            imgui.text("ldm_1: {} {}".format(m_ldm1_idx, m_ldm1))
            imgui.text("ldm_2: {} {}".format(m_ldm2_idx, m_ldm2))
                
            clicked_ldm_, select_ldm= imgui.menu_item("Select landmark", None, select_ldm)
            clicked_ldm_, show_ldm  = imgui.menu_item("Show landmark", None, show_ldm)
            clicked_bg, show_m      = imgui.menu_item("Show Model", None, show_m)
            if show_m:
                show_m = 1
            else:
                show_m = 0
                
                
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):

                    clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", "Cmd+Q", False, True
                    )

                    if clicked_quit:
                        sys.exit(0)

                    imgui.end_menu()
                imgui.end_main_menu_bar()
            
            #imgui.text("d_range: {}".format(d_range))
            
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
            clicked_norm, normalize   = imgui.menu_item("Normalize", None, normalize)
            
            changed, Reset_button = imgui.menu_item("Reset", None, Reset_button)

            
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
                show_ldm    = False
                select_ldm  = False
                Reset_button= False
                                
            rotation_ = rotation_ % 360
            imgui.render()
            impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        glReadBuffer(GL_FRONT)
        # glReadBuffer(GL_BACK)

        pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
        a = np.frombuffer(pixels, dtype=np.uint8)
        a = a.reshape((resolution, resolution, 4))
        
    ################################################## imgui end #########
    if use_imgui:
        impl.shutdown()
    ######################################################################
    glfw.terminate()
    
    rendered = a[::-1, :, :] # BGR to RGB
    
    # make directory
    savefolder = join('output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile = join(savefolder, 'exp_{:06}.png'.format(0))

    Image.fromarray(rendered).save(savefile)
    

if __name__ == '__main__':
    
    main(from_mesh = r"D:\test\Mesh-editing-with-pyopengl\_tmp\test.obj", # from
         to_mesh = r"D:\test\Mesh-editing-with-pyopengl\_tmp\016039.obj", # to
         image_path= r"D:\test\Mesh-editing-with-pyopengl\_tmp\material_0.png")
        
    # main(
    #     from_mesh = r"D:\test\Mesh-editing-with-pyopengl\_tmp\016039.obj", 
    #     to_mesh = r"D:\test\Mesh-editing-with-pyopengl\_tmp\test.obj", 
    #     image_path= r"D:\test\Mesh-editing-with-pyopengl\_tmp\exp_000000.png")
    
    # data_render(resolution=512)

