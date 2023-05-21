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
    print(texture)
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
            ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
            face_data.append(f)
            face_texture.append(ft)
    # mesh.v  = np.array(vertex_data)
    mesh.v  = normalize_v(np.array(vertex_data))
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    mesh.ft  = np.array(face_texture) -1
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

def render(resolution=512, mesh=None):
    if mesh is None:
        mesh = load_obj_mesh("R:\eNgine_visual_wave\engine_obj\M_012.obj")
    
    mesh.v      = normalize_v(mesh.v)
    
    # image_path  = "D:\\test\\RaBit\\white.png"
    image_path  = "D:\\test\\RaBit\\output\\tex.png"
    rendered    = main(mesh, resolution, image_path, timer=True)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile    = join(savefolder, 'rendered.png')

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
                
        # image_path  = "D:\\test\\RaBit\\white.png"
        image_path  = "D:\\test\\RaBit\\test\\tex.png"
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
    
    # image_path  = "D:\\test\\RaBit\\white.png"
    image_path  = "D:\\test\\RaBit\\test\\tex.png"
    rendered    = main(pca_mesh, resolution, image_path, timer=True, 
                       pca_v=True, mean=mean, coef=coef, basis=basis)
    rendered    = rendered[::-1, :, :]
    
    # make directory
    savefolder  = join('output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile    = join(savefolder, 'rendered.png')

    Image.fromarray(rendered).save(savefile)
    return

def main(mesh, resolution, image_path, timer=False, 
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
    
    if pca_v == True:
        blendshape = np.ones_like(coef)
        new_v = mean + np.dot(coef * blendshape, basis)
        new_v = new_v.reshape(-1,3)
        new_v = normalize_v(new_v)
        new_v[:, 2] = new_v[:, 2] * -1
        new_v = new_v[f].reshape(-1, 3)
    else:
        new_v  = v[f].reshape(-1, 3)
        
    new_vt = vt[ft].reshape(-1,2)
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    
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

    ############################################## imgui ################
    imgui.create_context()
    # window = impl_glfw_init() # already initialized!
    impl = GlfwRenderer(window)    
    
    i = 0
    rotation_angle = 20
    while not glfw.window_should_close(window):
        # curr_time = (time.time()-start)
        if pca_v == True:
            new_v = mean + np.dot(coef * blendshape, basis)
            new_v = new_v.reshape(-1,3)
            new_v = normalize_v(new_v)
            new_v[:, 2] = new_v[:, 2] * -1
            new_v = new_v[f].reshape(-1, 3)
            quad[:, :3] = new_v
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
            
        rotation_mat = y_rotation(rotation_angle)
        
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.poll_events()
        if pca_v == True:
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
            imgui.menu_item(label="(dummy menu)", shortcut=None, selected=False, enabled=False)
            clicked, rotation_angle = imgui.slider_float(
                    label="Rotate",
                    value=rotation_angle,
                    min_value=0.0,
                    max_value=360.0,
                )
            for i in range(100):    
                clicked, blendshape[i] = imgui.slider_float(
                    label="Value"+ str(i),
                    value=blendshape[i],
                    min_value=-1.0 if i >=2 else 0.0,
                    max_value=1.0,
                )
                
            rotation_angle = rotation_angle % 360
            # show_test_window()
            
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
    if pca_v == True:
        impl.shutdown()
    glfw.terminate()
    return a

if __name__ == '__main__':
    # render(resolution=512)
    data_render(resolution=512)

