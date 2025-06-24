import random
import torch
import numpy as np
import pickle
import trimesh

import os
import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f"{abs_path}/utils"]

def mesh_standardization(
        mesh, 
        mesh_data="ict", 
        return_idx=False
    ):
    """Apply mesh standardization with pre-calculated vertex indices and face indices

    Args
    -------
        mesh (trimesh.Trimesh)
        mesh_data (str)
        util_dir (str)
        return_idx (bool)

    Return
    -------
        new_mesh (trimesh.Trimesh)
    """
    # load info
    basename = os.path.join(f"{abs_path}", mesh_data, "standardization.npy")
    info_dict= np.load(basename, allow_pickle=True).item()

    # set new mesh
    new_v = mesh.vertices[info_dict['v_idx']]
    new_f = info_dict['new_f']
    new_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False, maintain_order=True)
    
    if return_idx:
        return new_mesh, info_dict['v_idx']
    return new_mesh

class ICT_face_model():
    """
    Original code: https://github.com/USC-ICT/ICT-FaceKit
    This is a quick loader version for personal interest    
    """
    def __init__(self, 
                 face_only=False, 
                 narrow_only=False, 
                 scale=0.1, 
                 base_dir=None, 
                 use_decimate=False, 
                 device='cpu'):
        """quick load ict face model
        Args:
            face_only (bool): if True, use face region only 
            scale (float): re-scale mesh 
            device (str): device 
        """
        self.device = device
        self.scale  = scale
        base_dir = abs_path if base_dir==None else base_dir
        self.base_dir = base_dir
        
        # get vertices and faces
        ## default: full = face + head + neck
        self.region = {
            0: [11248, 11144], # v_idx, quad_f_idx
            1: [9409, 9230], # v_idx, quad_f_idx # face_only
            2: [6706, 6560], # v_idx, quad_f_idx # narrow_only
        }
        v_idx, quad_f_idx = self.region[0]
        if face_only:
            v_idx, quad_f_idx = self.region[1]
        if narrow_only: # (not used)
            v_idx, quad_f_idx = self.region[2]
        
        self.use_decimate = use_decimate
        self.ict_deci = np.load(f'{base_dir}/utils/ict/ICT_decimate.npz')
        
        ## mesh faces
        quad_Faces = torch.load(f'{base_dir}/utils/ict/quad_faces.pt')[:quad_f_idx] #, map_location='cuda:0')
        self.faces = quad_Faces[:, [[0, 1, 2],[0, 2, 3]] ].permute(1, 0, 2).reshape(-1, 3).numpy()
        self.f_num = self.faces.shape[0]
        self.v_num = v_idx

        ## mesh verticies (alignment)
        neutral_verts = (torch.load(f'{base_dir}/utils/ict/neutral_verts.pt') * scale) - torch.tensor([0.0, 0.0, 0.5])
        self.neutral_verts = neutral_verts[:v_idx].numpy()

        ## blendshape basis
        self.exp_basis= torch.load(f'{base_dir}/utils/ict/exp_basis.pt') * scale
        self.id_basis = torch.load(f'{base_dir}/utils/ict/id_basis.pt') * scale
                
        ## send to device
        #self.neutral_verts = self.neutral_verts.to(self.device)
        self.exp_basis= self.exp_basis.numpy()
        self.id_basis = self.id_basis.numpy()
    
    def get_region_num(self, vertices):
        """
        Args:
            vertices: batched vertices or vertices [B, V, 3] or [V, 3]
        """
        N = vertices.shape[-2]
        
        if N == 11248:
            return 0
        elif N == 9409:
            return 1
        else:
            return 2
        
    def get_random_v_and_f(self, select=None, mode='np'):
        """
        Returns:
            tuple(int, np.ndarray)
        """
        if select is None:
            select = random.randint(0, 2)
        
        # 0: face to shoulder | 1: face to neck | 2. narrow face
        v_idx, quad_f_idx = self.region[select]
        
        qf_pth = f'{self.base_dir}/utils/ict/quad_faces.pt'
        quad_Faces = torch.load(qf_pth)[:quad_f_idx]
        tri_faces = quad_Faces[:, [[0, 1, 2],[0, 2, 3]] ].permute(1, 0, 2).reshape(-1, 3)
        
        tri_faces = tri_faces.numpy() if mode == 'np' else tri_faces
            
        return v_idx, tri_faces

    def get_mesh(self, mesh_std=False, return_idx=False):
        """
        Args:
            mesh_std (bool): if True, apply `mesh standardization`
        """
        mesh = trimesh.Trimesh(
            vertices=self.neutral_verts, 
            faces=self.faces, 
            process=False, maintain_order=True
        )
        mesh_v_idx = np.arange(self.neutral_verts.shape[0])
        
        ## mesh_standardization
        if mesh_std:
            mesh, mesh_v_idx = mesh_standardization(mesh, mesh_data='ict', return_idx=True)
        
        if return_idx:
            return mesh, mesh_v_idx
        return mesh
    
    def get_random_mesh(self, mesh_std=False, return_idx=False):
        """
        Args:
            mesh_std (bool): if True, apply `mesh standardization`
        """
        id_disps = self.get_id_disp(np.random.rand(100))[0]
        mesh = trimesh.Trimesh(
            vertices=self.neutral_verts + id_disps, 
            faces=self.faces, 
            process=False, maintain_order=True
        )
        mesh_v_idx = np.arange(self.neutral_verts.shape[0])
        
        ## mesh_standardization
        if mesh_std:
            mesh, mesh_v_idx = mesh_standardization(mesh, mesh_data='ict', return_idx=True)
        
        if return_idx:
            return mesh, mesh_v_idx
        return mesh

    def apply_coeffs(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False, region=0):
        """
        Args:
            id_coeff (np.ndarray): [100] ICT-facekit identity coeff
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
        """
        # id vertices
        id_disps = self.get_id_disp(id_coeff)
        
        # exp vertices
        exp_disp = self.get_exp_disp(exp_coeffs)
        
        id_verts = self.neutral_verts + id_disps
        id_exp_verts = id_verts + exp_disp
        
        # apply std
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[mesh_v_idx]
        
        id_verts = id_verts[:self.region[region][0]]
        id_exp_verts = id_exp_verts[:self.region[region][0]]
        
        if return_all:
            return id_exp_verts, id_verts, exp_disp
        return id_exp_verts
    
    def apply_coeffs_batch(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False, region=0):
        """
        Args:
            id_coeff (np.ndarray): [N,100] ICT-facekit identity coeff
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
        """
        device = id_coeff.device
        
        # id vertices
        id_disps = self.get_id_disp(id_coeff)
        
        # exp vertices
        exp_disps = self.get_exp_disp(exp_coeffs)
            
        id_verts = self.neutral_verts[:, :self.region[region][0]] + id_disps
        id_exp_verts = id_verts + exp_disps
        
        # apply std
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[:, mesh_v_idx]
            
        id_verts = id_verts[:, :self.region[region][0]]
        id_exp_verts = id_exp_verts[:, :self.region[region][0]]
        
        if return_all:
            return id_exp_verts, id_verts, exp_disps
        return id_exp_verts
    
    def apply_coeffs_batch_torch(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False,region=0):
        """
        Args:
            id_coeff (torch.Tensor): [N,100] ICT-facekit identity coeff
            exp_coeffs (torch.Tensor): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (torch.Tensor): <int> array of std mesh vertex indices
        """
        device = id_coeff.device
        
        # id vertices
        #id_disps = self.get_id_disp(id_coeff)
        neutral_verts = self.neutral_verts[:self.region[region][0]]
        
        B = id_coeff.shape[0]
        id_basis_reshaped = torch.from_numpy(self.id_basis.reshape(100, -1)).float().to(device)
        id_disps = torch.mm(id_coeff, id_basis_reshaped).reshape(B, -1, 3)
        id_disps = id_disps[:, :self.region[region][0]]
        
        # exp vertices
        #exp_disps =self.get_exp_disp(exp_coeffs)
        T = exp_coeffs.shape[0]
        exp_basis_reshaped = torch.from_numpy(self.exp_basis.reshape(53, -1)).float().to(device)
        exp_disps = torch.mm(exp_coeffs, exp_basis_reshaped).reshape(T, -1, 3)
        exp_disps = exp_disps[:, :self.region[region][0]]
            
        id_verts = torch.from_numpy(neutral_verts).float().to(device) + id_disps
        id_exp_verts = id_verts + exp_disps
        
        if self.use_decimate:
            mesh_v_idx = self.ict_deci["v_idx"]
        
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[:, mesh_v_idx]
            id_verts = id_verts[:, mesh_v_idx]
            exp_disps = exp_disps[:, mesh_v_idx]
            
        if return_all:
            return id_exp_verts, id_verts, exp_disps
                
        return id_exp_verts
    
    def get_id_disp(self, id_coeff=None, mesh_v_idx=None, region=0):
        """
        Args:
            id_coeff (np.ndarray): [100] / [B, 100] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
            
        Returns:
            id_disps (np.ndarray): [V, 3] / [B, V, 3]mesh vertices
        """
        # id vertices
        if id_coeff is not None:
            if len(id_coeff.shape) < 2:
                id_coeff = id_coeff[None]
            B = id_coeff.shape[0]
            id_basis_reshaped = self.id_basis.reshape(100, -1)
            id_disps = np.matmul(id_coeff, id_basis_reshaped).reshape(B, -1, 3)
            id_disps = id_disps[:, :self.region[region][0]]
        else:
            id_disps = 0.0

        # apply std
        if mesh_v_idx is not None:
            id_disps = id_disps[:, mesh_v_idx]
            
        return id_disps
    
    def get_exp_disp(self, exp_coeffs=None, mesh_v_idx=None, region=0):
        """
        Args:
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices

        Returns:
            exp_disps (np.ndarray): [T, V, 3] mesh vertices
        """
        # exp vertices
        if exp_coeffs is not None:
            if len(exp_coeffs.shape) < 2:
                exp_coeffs = exp_coeffs[None]
                
            T = exp_coeffs.shape[0]
            #exp_disps = np.einsum('jk,kls->jls', exp_coeffs.float(), self.exp_basis.to(device))[:, :self.region[region][0]]
            
            exp_basis_reshaped = self.exp_basis.reshape(53, -1)
            exp_disps = np.matmul(exp_coeffs, exp_basis_reshaped).reshape(T, -1, 3)
            exp_disps = exp_disps[:, :self.region[region][0]]
        else:
            exp_disps = 0.0
        
        # apply std
        if mesh_v_idx is not None:
            exp_disps = exp_disps[:, mesh_v_idx]
            
        return exp_disps