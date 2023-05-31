# import sys
# import matplotlib.pyplot as plt
import os
from os.path import join

# import time
import json
import argparse

from glob import glob
from PIL import Image

# import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch3d.io import save_obj, load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch, packed_to_list
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams,
    TexturesUV,
    TexturesAtlas,
    TexturesVertex,
)

# for custom rasterizer, shader
from pytorch3d.renderer.mesh.rasterizer import Fragments, RasterizationSettings
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.transforms.rotation_conversions import _axis_angle_rotation
"""
Reference: 
    https://github.com/facebookresearch/pytorch3d/issues/854
    https://github.com/facebookresearch/pytorch3d/issues/889
"""
def convert_to_verts_colors(textures_uv: TexturesUV, meshes: Meshes):
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()
    return verts_colors_packed

def convert_to_TexturesVertex(textures_uv: TexturesUV, meshes: Meshes):
    verts_colors_packed = convert_to_verts_colors(textures_uv, meshes)
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))


def load_objs_with_tex(
        files: list,
        device=None,
        load_textures: bool = True,
        tex_files: list = None,
        disp_files: list = None,
        betas_files: list = None,
        create_texture_atlas: bool = False,
        texture_atlas_size: int = 4,
        texture_wrap = "repeat",
        path_manager = None,
    ):
    """
    when load meshes we need to put on texture

    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj, f_tex in zip(files, tex_files):
        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )

        tex = None
        if create_texture_atlas:
            # TexturesAtlas
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV
            verts_uvs = aux.verts_uvs.to(device)      # (V, 2)
            faces_uvs = faces.textures_idx.to(device) # (F, 3)

            divide    = 1/255
            tex_img   = Image.open(f_tex)
            tex_img   = tex_img.resize((256,256))
            tex_map   = np.array(tex_img)[..., :3]     # (H, W, 3)
            tex_map   = torch.tensor(tex_map).unsqueeze(0) * divide
            tex_map   = tex_map.to(device)
            tex       = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=tex_map)

        z_ = torch.ones(verts_uvs.size(0), 1).to(device)
        mesh = Meshes(
            verts    = [verts.to(device)], 
            faces    = [faces.verts_idx.to(device)],
            textures = tex
        )

        """ Verts offsets must have dimension (V, 3) """ ################################## displacement
        # vert_displacement = convert_to_verts_colors(disp, mesh) # (V, 3)
        # mesh = mesh.offset_verts_(vert_displacement)
        
        mesh_list.append(mesh)
        
    return mesh_list

class CustomMeshRasterizer(nn.Module):
    def __init__(self, cameras=None, raster_settings=None, in_UV=False) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings
        self.in_UV = in_UV

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.
        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space
        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        if self.in_UV:
            # replace mesh to UV
            verts_uvs = meshes_world.textures._verts_uvs_list[0] #(V, 2)
            faces_uvs = meshes_world.textures._faces_uvs_list[0] #(F, 3)

            z_ = torch.zeros(verts_uvs.size(0), 1).to(verts_uvs)
            verts_uvs_z_ = torch.cat(((verts_uvs*2)-1, z_),dim=1) # (V, 3)
            # verts_uvs_z_ = torch.cat(((verts_uvs-0.5)*2, z_),dim=1) # (V, 3)
            # verts_uvs_z_ = torch.cat(((verts_uvs), z_),dim=1) # (V, 3)

            meshes_world = Meshes(
                verts = [verts_uvs_z_], 
                faces = [faces_uvs]
            )

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        # view to NDC transform
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = cameras.get_projection_transform(**kwargs).compose(
            to_ndc_transform
        )
        verts_ndc = projection_transform.transform_points(verts_view, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward(self, meshes_world, **kwargs):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )

class CustomSoftPhongShader(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """
    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = self._get_cameras(**kwargs)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        # get pixel coordinate
        # verts = meshes.verts_packed()  # (V, 3)
        # faces_verts = verts[faces]
        # pixel_coords = interpolate_face_attributes(
        #     fragments.pix_to_face, fragments.bary_coords, faces_verts
        # )

        # get world normal
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]        
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        colors = phong_shading(
        # colors = _phong_shading_with_pixels(
            meshes=meshes,
            fragments=fragments,
            # texels=texels,
            texels=pixel_normals * 0.5 + 0.5,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kwargs.get("zfar",  getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images

class ToyRenderer(nn.Module):
    def __init__(self,
            image_size   = 512, 
            device       = None,
            elev         = None, 
            azim         = None, 
            at_change    = False,
            opt          = None,
        ):
        super().__init__()
        self.opt     = opt
        self.azim    = azim
        self.elev    = elev
        self.rot_mat = None
        self.device  = device

        # the number of different viewpoints from which we want to render the mesh.
        # Initialize a camera.  World coordinates +X left, +Y up and +Z in.
        R, T = look_at_view_transform(
            dist = 1, 
            elev = 0.0, 
            azim = 0.0, 
            at = ((0.0, 0.0, 0.0),), ################################################################## for normal map rendering
            # at     =  ((0.0, 0.9, 0.0),),
            up = ((0.0, 1.0, 0.0),) 
        )  # 0.3

        # We arbitrarily choose one particular view that will be used to visualize results
        select_camera = FoVOrthographicCameras(
            R=R, 
            T=T,
            device=device
        )

        # back ground color
        blend_params = BlendParams(
            background_color = [0., 0., 0.]
            # background_color = [1., 1., 1.]
        )

        raster_settings = RasterizationSettings(
            image_size      = image_size, 
            blur_radius     = 0.0,
            faces_per_pixel = 1,
            # bin_size        = 64,
            # max_faces_per_bin = 64,
        )

        lights = DirectionalLights(
            ambient_color  = [[1., 1., 1.]], 
            diffuse_color  = [[0., 0., 0.]],  
            specular_color = [[0., 0., 0.]], 
            direction      = T, 
            device         = device
        )

        # custom_rasterizer = MeshRasterizer(
        custom_rasterizer = CustomMeshRasterizer( ##################################################### for normal map rendering
            cameras         = select_camera, 
            raster_settings = raster_settings,
            in_UV           = False
        )

        # custom_shader = SoftPhongShader(
        custom_shader = CustomSoftPhongShader( ######################################################## for normal map rendering
            device       = device,
            cameras      = select_camera,
            lights       = lights,
            blend_params = blend_params
        )

        renderer = MeshRenderer(
            rasterizer = custom_rasterizer,
            shader     = custom_shader
        )
        self.renderer = renderer
        self.camera = select_camera
        self.lights = lights
        # self.out_dir = out_dir
        # if not self.out_dir:
        #     print('No output directory specified: self.out_dir == None, rendered image will not be saved')

    def degree2radian(self, degree):
        return torch.deg2rad(torch.tensor([degree]))

    def random_rotation_Y(self):
        # rand_ang = torch.randint(-9, 10, (1,)) * 10
        rand_rad = self.degree2radian(0)
        # rand_rad = torch.rand(1) * torch.pi #* 2
        rot_mat  = _axis_angle_rotation("Y", rand_rad).squeeze(0)
        return rot_mat.to(self.device)

    def apply_random_rotation_Y(self, meshes):
        self.rot_mat = self.random_rotation_Y()
        for B in range(len(meshes._verts_list)):
            meshes._verts_list[B] = meshes._verts_list[B] @ self.rot_mat
        return meshes

    def forward(self, mesh, R, T):
        # if type(self.azim) == list:
        #     mesh = mesh.extend(len(self.azim))
        
        # if True:
        #     mesh = self.apply_random_rotation_Y(mesh)

        rendered = self.renderer(mesh, cameras=self.camera, lights=self.lights, R=R, T=T)
        return rendered

def set_seed(seed):
    from torch.backends import cudnn
    import random
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    from tqdm import tqdm
    set_seed(1234)
    # CUDA
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## get files texture / objs
    tex_files  = ["tex.png"]
    
    obj_files  = sorted(glob("data/*/tpose/m.obj"))    
    
    len_objs   = len(obj_files)
    tex_files  = tex_files * len_objs
    len_texs   = len(tex_files)

    azim_range = [0]
    len_azim   = len(azim_range)
    len_data   = len_texs // len_azim
    
    # print(f'path: {args.tex}')
    print(f"device: {device}\t obj:{len_objs}\t tex: {len_texs}\t azim: {len_azim}\t data: {len_data}")

    ## padding
    padd = True

    # save directory
    out_dir = 'output'
    os.makedirs(out_dir,  exist_ok=True)
    
    with torch.no_grad():
        for idx, azim in enumerate(azim_range):            
            renderer = ToyRenderer(image_size = 512, device= device)
            
            meshes = load_objs_with_tex(
                    files        = obj_files,  # obj_files, # [obj_files] # it should be a list
                    tex_files    = tex_files,  # tex_files, 
                    device       = device,
                )
            
            # import pdb;pdb.set_trace()
            # pbar = tqdm(zip(tex_files, obj_files))
            pbar = tqdm(enumerate(meshes))
            for idx, mesh in pbar:
                name     = obj_files[idx].split('/')[1]
                pbar.set_description(f"Processing [{azim}]: {name}")

                R, T = look_at_view_transform(
                    dist   =  1, 
                    elev   =  0.0, 
                    azim   =  azim, 
                    at     =  ((0.0, 0.0, 0.0),), 
                    up     =  ((0.0, 1.0, 0.0),),
                    device = device
                )
                # import pdb;pdb.set_trace()
                img = renderer(mesh, R=R, T=T)
                
                img  = img.permute(0,3,1,2)[0]
                img[3][img[3] > 0] = 1.0
                
                mask = (img[3:] > 0).all(0) * 1.0
                # save_img = img[:3] * mask
                torchvision.utils.save_image(img[:3],  join(out_dir,  f"render{name}.png"))
        
    # Print the amount of time
    # print('Done: {:.2f}'.format(time.time() - begin_time))


if __name__ == "__main__":
    main()

