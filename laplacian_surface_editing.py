from __future__ import print_function

import sys
import os
from os.path import exists, join
import numpy as np


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from PIL import Image
import glm

# from testwindow import *
from imgui.integrations.glfw import GlfwRenderer
import imgui

from utils.processing import laplacian_surface_editing
from utils.loader import load_texture, Mesh_container
from utils.util import (
    x_rotation, 
    y_rotation, 
    z_rotation,
)

# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Uniform Laplacian matrix
# 2) Cotangent Laplacian matrix
# -------------------------------------------------------------------- #

    
 
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
                new_v = laplacian_surface_editing(mesh, mask_v, boundary_idx=boundary_idx, handle_idx=handle_idx, handle_pos=handle_pos_new_)
            else:
                new_v = mesh.v
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
            imgui.text("handle_pos_new: {}".format(handle_pos_new))
            imgui.text("handle_pos[0]: {}".format(handle_pos[0]))
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

def main(resolution=512,
        boundary_idx = [3, 13, 481, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 223, 238, 253, 268, 283, 298, 313, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464],
        handle_idx   = [0, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 326, 341, 356, 371, 386, 401, 416, 431, 446, 461],
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
    
    savefile = join(savefolder, 'experiemtn.png')

    Image.fromarray(rendered).save(savefile)
    return

if __name__ == '__main__':
    main(resolution=1024)