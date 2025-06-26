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

from utils.laplacian import adjacency_matrix, laplacian_cotangent
from utils.processing import as_rigid_as_possible_surface_modeling, laplacian_surface_editing
from utils.loader import load_texture, Mesh_container
from utils.util import (
    rotation, 
    x_rotation, 
    y_rotation, 
    z_rotation,
    normalize_np,
    normalize_torch,
    compute_vertex_normals
)




        
        

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
    fn = mesh.fn
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

    # import pdb;pdb.set_trace()
    new_v  = v[f].reshape(-1, 3)
    if len(vt) != 0:
        new_vt = vt[ft].reshape(-1,2)
        new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
        if len(fn) != 0:
            new_vn = vn[fn].reshape(-1, 3)
        else:
            new_vn = vn[f].reshape(-1, 3)
    else:
        new_vt = np.zeros_like(new_v)
        vn = compute_vertex_normals(v, f)
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
    rotation_X = 0
    rotation_Y = 0
    rotation_Z = 0
    
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
    use_LSE = False
    use_ARAP = True
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
        rotation_Xmat = x_rotation(rotation_X)
        rotation_Ymat = y_rotation(rotation_Y)
        rotation_Zmat = z_rotation(rotation_Z)
        affine_mat = np.eye(4)
        
        affine_mat[:3,:3] = affine_mat[:3,:3] * np.array([scaleX, scaleY, scaleZ])
        trans = np.array([transX, transY, transZ])
        # rotation_Ymat = rotation_Ymat @  affine_mat.T
        
        rotation_mat_all = rotation_Zmat @ rotation_Ymat @ rotation_Xmat @ affine_mat.T
        
        # glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_Ymat)
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat_all)
        glUniform3fv(translate, 1, trans)
        
        view = glm.ortho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, 0.0001, 1000.0) 
        glUniformMatrix4fv(glView, 1, GL_FALSE, glm.value_ptr(view))
        
        ### Surface Editing
        if handle_change:
            handle_pos_mean = handle_pos.mean(0)
            handle_pos_new_ = (handle_pos-handle_pos_mean) @ H_r_mat[:3,:3]
            handle_pos_new_ = handle_pos_new_ + handle_pos_mean + handle_pos_new.copy()
            
            if use_LSE:
                new_v = laplacian_surface_editing(
                    mesh, mask_v, boundary_idx=boundary_idx, handle_idx=handle_idx, handle_pos=handle_pos_new_
                )
            elif use_ARAP:
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
            
            # clicked, tex_alpha  = imgui.slider_float(label="_alpha",    value=tex_alpha, min_value=0.0, max_value=1.0)
            # clicked, _ratio     = imgui.slider_float(label="_ratio",    value=_ratio, min_value=0.0, max_value=1.0)
            
            clicked, rotation_X  = imgui.slider_float(label="Rotate X", value=rotation_X, min_value=-180.0, max_value=180.0,)
            clicked, rotation_Y  = imgui.slider_float(label="Rotate Y", value=rotation_Y, min_value=-180.0, max_value=180.0,)
            clicked, rotation_Z  = imgui.slider_float(label="Rotate Z", value=rotation_Z, min_value=-180.0, max_value=180.0,)
            
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
            ARAP_changed, use_ARAP = imgui.menu_item("Use ARAP", None, use_ARAP)
            
            
            if json_object:
                clicked_use_disp, use_disp = imgui.menu_item("Use_disp", None, use_disp)
                clicked_use_tex, onoff_tex = imgui.menu_item("OnOff Tex", None, onoff_tex)

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
            
            if ARAP_changed:
                use_LSE = False
                handle_change = True

            if LSE_changed:
                use_ARAP = False
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
                rotation_X   = 0
                rotation_Y   = 0
                rotation_Z   = 0
                mesh.v = mesh.orig_v
                iteration_  = 0
                normalize   = False
                Reset_button= False
                handle_change = True
                use_LSE = False
                use_ARAP = True
                handle_pos_new = np.array([0.0, 0.0, 0.0])
                                
            rotation_X = 180 if rotation_X <= -180 else rotation_X
            rotation_Y = 180 if rotation_Y <= -180 else rotation_Y
            rotation_Z = 180 if rotation_Z <= -180 else rotation_Z
            # rotation_X = np.clip(rotation_X, -180, 180)
            # rotation_Y = np.clip(rotation_Y, -180, 180)
            # rotation_Z = np.clip(rotation_Z, -180, 180)
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
  
def main(
    # mesh_path = "data/sphere_trimesh.obj"
    mesh_path = "data/sphere.obj",
    resolution=512,
    boundary_idx = [3, 13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 223, 238, 253, 268, 283, 298, 313, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464],
    # boundary_idx = [8, 21, 36, 51, 66, 81, 96, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 261, 276, 291, 306, 321, 337, 352, 367, 382, 397, 412, 427, 442, 457, 472],
    handle_idx   = [0, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 326, 341, 356, 371, 386, 401, 416, 431, 446, 461],
    # handle_idx   = [325],
    ):
    ### sphere mesh
    # path                = "data/sphere_trimesh.obj"
    # mesh_path                = "data/sphere.obj"
    
    ## for loading face and other stuff: vt, vn, ft, fn ...
    mesh = Mesh_container(mesh_path, boundary_idx=boundary_idx, handle_idx=handle_idx)
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
 
 
if __name__ == '__main__':
    main(
        resolution=1024,
        # mesh_path = "data/sphere.obj",
        
        # mesh_path = "data/decimated_knight.obj",
        # boundary_idx=[26, 56, 61, 79, 82, 90, 104, 124, 130, 179, 181, 190, 197, 198, 209, 247, 277, 354, 390, 391, 407, 417, 473],
        # handle_idx=[144],
        
        mesh_path = "data/decimated_knight.obj",
        boundary_idx=[0, 43, 166, 194, 202, 256, 271, 301, 388, 404],
        handle_idx=[324],
        
        # mesh_path = "data/square_21_spikes.obj",
        # boundary_idx=[399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440],
        # handle_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    )