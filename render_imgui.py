
import sys

import pygame
import OpenGL.GL as gl
from PIL import Image
import numpy as np

from imgui.integrations.pygame import PygameRenderer
import imgui
from easydict import EasyDict

def loadImage(image):
    """
    Args:
        image (PIL.Image):  Image to bind as texture

    Returns:
        texture (int):      texture id
        image.width (int):  image width
        image.height (int): image height
    """
    
    ## Resize image to 512x512
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    ## Flip image vertically (OpenGL's origin is bottom-left)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGBA') # 'RGBA
    
    ## Convert image to numpy array
    textureData = np.array(list(image.getdata()), np.uint8)

    ## Create texture
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.width, image.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textureData)

    return texture, image.width, image.height

def show_imgui_windows(preset):
    """
    Args:
        preset (EasyDict): includes all the parameters for imgui and the blendshape
    """
    imgui.new_frame()
    ###-------- make a top bar ---------------------------------
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):

            clicked_quit, selected_quit = imgui.menu_item(
                "Quit", 'Cmd+Q', False, True
            )

            if clicked_quit:
                exit(1)

            imgui.end_menu()
        imgui.end_main_menu_bar()

    
    ###-------- make a window to draw the texture -------------
    imgui.set_next_window_size(preset.img_window_size, preset.img_window_size)
    imgui.set_next_window_position(preset.prm_window_width, preset.topbar_height)
    imgui.begin("3DMM render", True)
    imgui.image(preset.texture0, preset.width, preset.height, border_color=(1, 0, 0, 1))
    imgui.end()
    
    imgui.set_next_window_size(preset.img_window_size, preset.img_window_size)
    imgui.set_next_window_position(preset.prm_window_width+preset.img_window_size, preset.topbar_height)
    imgui.begin("StyleGAN render", True)
    imgui.image(preset.texture1, preset.width, preset.height, border_color=(1, 0, 0, 1))
    imgui.end()
    
    # imgui.show_test_window()
    
    ###-------- make a bar window to draw the sliders ---------
    imgui.set_next_window_size(preset.prm_window_width, preset.img_window_size)
    imgui.set_next_window_position(0, preset.topbar_height)
    imgui.begin("parameter bar", True)
    
    clicked_quit, preset.reset_dummy = imgui.menu_item(
        "Reset", None, preset.reset_dummy
    )
        
    for idx in range(preset.blendshape.shape[0]):
        ### change the width of the slider
        # imgui.set_next_item_width(100)
        
        ### name the axis
        axis = ''
        start = 0
        if idx < 80:
            axis = preset.blendshape_list[0] # 80
        if 80<= idx and idx < 144:
            axis = preset.blendshape_list[1] # 64
            start = 80
        if 144<= idx and idx < 224:
            axis = preset.blendshape_list[2] # 80
            start = 144
        if 224<= idx and idx < 227:
            axis = preset.blendshape_list[3] # 3
            start = 224
        if 227<= idx and idx < 254:
            axis = preset.blendshape_list[4] # 27
            start = 227
        if 254<= idx:
            axis = preset.blendshape_list[5] # 2 ???
            start = 254
        
        ### update the blendshape
        clicked, preset.blendshape[idx] = imgui.slider_float(
            label     = "{} {:03d}".format(axis, idx-start),
            value     = preset.blendshape[idx], 
            min_value = -1.0, 
            max_value = 1.0
        )
        if clicked:
            # actually not recommended 
            # drawing every frame could lead to memory issues...
            image1 = Image.open("white3.png")
            image2 = Image.open("white.png")            
            
            preset.texture0, preset.width, preset.height = loadImage(image1) 
            preset.texture1, preset.width, preset.height = loadImage(image2) 
    imgui.end()
        
        
    if clicked_quit:
        preset.blendshape  = preset.blendshape * 0.0
        preset.reset_dummy = False
        # preset.blendshape_change = False
        
    # imgui.push_id("1")
    # if imgui.image_button(texture, width, height):
    #     print("Yay")
    # imgui.pop_id()
    
    # imgui.begin("Custom window2", True)
    # imgui.text("Bar")
    # imgui.text_colored("Eggs", 0.2, 1., 0.)
    # imgui.end()

def main():    
    #--------------------------parameters--------------------------
    preset = EasyDict()
    
    ### number of images
    preset.image_len        = 2  # two images for now
    preset.image_size       = 512
    
    ### top menu bar height
    preset.topbar_height    = 20
    
    ### width of the parameter bar
    preset.prm_window_width = 384
    
    ### size of the image window
    preset.img_window_size  = preset.image_size+35
    
    ### offset for the display window
    preset.offset_x         = 15
    preset.offset_y         = 65
    
    preset.display_size     = preset.prm_window_width + (preset.image_len*preset.img_window_size), (preset.image_size+preset.offset_y)
    
    preset.reset_dummy = False
    
    ### blendshape parameters
    preset.blendshape = np.zeros([256])
    preset.blendshape_list = [
        'id_coeffs',
        'exp_coeffs',
        'tex_coeffs',
        'angles',
        'gammas',
        'translations',
    ]
    preset.blendshape_list_len = len(preset.blendshape_list)
    # preset.blendshape_change = False
    #--------------------------------------------------------------
    
    pygame.init()
    pygame.display.set_mode(preset.display_size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = preset.display_size
    
    #------------------------load PIL image------------------------
    image1 = Image.open("white3.png")
    image2 = Image.open("white.png")
    
    preset.texture0, preset.width, preset.height = loadImage(image1) 
    preset.texture1, preset.width, preset.height = loadImage(image2)
    
    event_holder = None;
    
    while 1:
        for event in pygame.event.get():
            #impl.process_event(event) usually here but this will cause the image to not render
            if event.type == pygame.QUIT:
                sys.exit()
            event_holder = event;

        impl.process_event(event_holder)# outside of the for loop and now pass in the global event variable 
        
        ###---------call the function to draw the windows----------
        show_imgui_windows(preset)
        ###--------------------------------------------------------
        
        # note: cannot use screen.fill((1, 1, 1)) because pygame's screen
        #       does not support fill() on OpenGL sufraces
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())
        pygame.display.flip()

if __name__ == "__main__":
    main()