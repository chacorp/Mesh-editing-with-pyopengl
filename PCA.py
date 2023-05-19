import scipy.io
from glob import glob
from easydict import EasyDict as edict
import numpy as np

# run PCA
from sklearn.decomposition import PCA

def load_obj_mesh(mesh_path):
    mesh = edict()
    vertex_data = []
    vertex_normal = []
    verts_uvs  = []
    face_data = []
    faces_uvs  = []
    for line in open(mesh_path, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        if values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            vertex_normal.append(vn)
        if values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            verts_uvs .append(vt)
        if values[0] == 'f':
            f = list(map(lambda x: int(x.split('/')[0]),  values[1:4]))
            face_data.append(f)
            f = list(map(lambda x: int(x.split('/')[1]),  values[1:4]))
            faces_uvs.append(f)
    mesh.v  = np.array(vertex_data)
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(verts_uvs)
    mesh.f  = np.array(face_data)
    mesh.ft = np.array(faces_uvs)
    return mesh

if __name__ == "__main__":
    import os.path
    from pytorch3d.io import save_obj, load_obj, load_objs_as_meshes
    import torch
    
    objs = "./data/*/tpose/m.obj"
    obj_list = sorted(glob(objs))

    print(len(obj_list)) # length: 442
    
    tmp_obj = load_obj(obj_list[0])
    verts = tmp_obj[0]
    faces = tmp_obj[1].verts_idx
    uvs   = tmp_obj[2].verts_uvs
    
    # tmp_obj = load_obj_mesh(obj_list[0])
    # faces   = tmp_obj.f
    # uvs     = tmp_obj.vt
    
    if os.path.exists("./pca_npy/pca_basis.npy"):
        pca_3d = edict()
        with open('./pca_npy/pca_basis.npy', 'rb') as f:
            pca_3d.components_ = np.load(f)
        with open('./pca_npy/pca_coeff.npy', 'rb') as f:
            pca_3d.explained_variance_ = np.load(f)
        with open('./pca_npy/pca_mean.npy', 'rb') as f:
            pca_3d.mean_ = np.load(f)
    else:
        # load all meshes from data
        # meshes  = [ load_obj_mesh(obj) for obj in obj_list]
        meshes  = load_objs_as_meshes(obj_list)
        meshes_verts = [mesh.v for mesh in meshes]
        
        DATA_3D = np.c_[meshes_verts] # (442, 38726, 3)
        DATA_3D_ = DATA_3D.reshape(442, -1)

        pca_3d = PCA(n_components=100)
        pca_3d.fit(DATA_3D_)
        ### save PCA basis and coefficiants
        np.save('./pca_npy/pca_basis', pca_3d.components_)
        np.save('./pca_npy/pca_coeff', pca_3d.explained_variance_)
        np.save('./pca_npy/pca_mean', pca_3d.mean_)

    print("eigenvectors: \n", len(pca_3d.components_))
    print("eigenvalues: \n", len(pca_3d.explained_variance_))
    import pdb;pdb.set_trace()
    
    new_pc_val = np.ones_like(pca_3d.explained_variance_)

    new_pc_val[0] = pca_3d.explained_variance_[0] * 1.0
    new_pc_val[1] = pca_3d.explained_variance_[1] * 2.0
    new_pc_val[2] = pca_3d.explained_variance_[2] * 1.0

    print(f"original: \n{pca_3d.explained_variance_}\n")
    print(f"edited: \n{new_pc_val}\n")

    new_v = torch.tensor((pca_3d.mean_ + np.dot(new_pc_val, pca_3d.components_)).reshape(-1,3))
    
    save_obj("new_obj.obj", new_v, faces, verts_uvs=uvs)