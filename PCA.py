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

def reconstruct(mean, coef, basis):
    new_v = mean + np.dot(coef, basis)
    new_v = torch.tensor(new_v.reshape(-1,3))
    return new_v

def recon_and_save(mean, coef, basis, faces, uvs):
    new_v = reconstruct(mean, coef, basis)
    save_obj("new_obj.obj", new_v, faces, verts_uvs=uvs)

if __name__ == "__main__":
    import os.path
    from pytorch3d.io import save_obj, load_obj, load_objs_as_meshes
    import torch
    # from render_obj import render, pca_render
    import pickle as pkl
    
    objs = "./data/*/tpose/m.obj"
    obj_list = sorted(glob(objs))

    
    tmp_obj = load_obj(obj_list[0])
    verts = tmp_obj[0]
    faces = tmp_obj[1].verts_idx
    ft    = tmp_obj[1].textures_idx
    uvs   = tmp_obj[2].verts_uvs
    vn    = tmp_obj[2].normals
    
    # tmp_obj = load_obj_mesh(obj_list[0])
    # faces   = tmp_obj.f
    # uvs     = tmp_obj.vt
    

    # if os.path.exists("./pca_npy/pca_basis.npy"):
    # if False:
    if os.path.exists("./pca_npy/pca_100.pkl"):
        pca_3d = edict()
        # with open('./pca_npy/pca_basis.npy', 'rb') as f:
        #     pca_3d.components_ = np.load(f)
        # with open('./pca_npy/pca_coeff.npy', 'rb') as f:
        #     pca_3d.explained_variance_ = np.load(f)
        # with open('./pca_npy/pca_mean.npy', 'rb') as f:
        #     pca_3d.mean_ = np.load(f)
            
        pca_3d = pkl.load(open("./pca_npy/pca_50.pkl",'rb'))
    else:
        # load all meshes from data
        # meshes  = load_objs_as_meshes(obj_list)
        # meshes_verts = meshes._verts_list
        # meshes  = [ load_obj_mesh(obj) for obj in obj_list]
        # meshes_verts = [mesh.v for mesh in meshes]
        meshes       = [load_obj(obj) for obj in obj_list]
        meshes_verts = [np.array(mesh[0]) for mesh in meshes]
        # import pdb;pdb.set_trace()
        
        DATA_3D  = np.c_[meshes_verts] # (442, 38726, 3)
        DATA_3D_ = DATA_3D.reshape(442, -1)

        pca_3d = PCA(n_components=50)
        pca_3d.fit(DATA_3D_)
        pkl.dump(pca_3d, open("./pca_npy/pca_50.pkl","wb"))
        ### save PCA basis and coefficiants
        # np.save('./pca_npy/pca_basis', pca_3d.components_)
        # np.save('./pca_npy/pca_coeff', pca_3d.explained_variance_)
        # np.save('./pca_npy/pca_mean', pca_3d.mean_)

    print("eigenvectors: ", len(pca_3d.components_))
    print("eigenvalues: ",  len(pca_3d.explained_variance_))
    # import pdb ; pdb.set_trace()
    
    # new_pc_val = np.ones_like(pca_3d.explained_variance_)
    new_pc_val = pca_3d.explained_variance_.copy()

    idx = 0;     new_pc_val[idx] = pca_3d.explained_variance_[idx] * -10

    # print(f"original: \n{pca_3d.explained_variance_}\n")
    # print(f"edited: \n{new_pc_val}\n")

    recon_and_save(pca_3d.mean_, new_pc_val, pca_3d.components_, faces, uvs)
    
    # pca_mesh    = edict()
    # pca_mesh.v  = reconstruct(pca_3d.mean_, new_pc_val, pca_3d.components_)
    # pca_mesh.v[:,2] = pca_mesh.v[:,2] * -1
    # pca_mesh.vt = uvs
    # pca_mesh.vn = vn
    # pca_mesh.f  = faces
    # pca_mesh.ft = ft
    
    # render(resolution=1024, mesh=pca_mesh)
    # pca_render(pca_3d.mean_, new_pc_val, pca_3d.components_, uvs, vn, faces, ft, resolution=1024)
