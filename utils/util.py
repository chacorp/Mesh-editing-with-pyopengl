import numpy as np
import scipy
from collections import defaultdict
from itertools import combinations

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import lsqr, splu, spsolve

import scipy.sparse.csgraph as csgraph
from sklearn.neighbors import NearestNeighbors

from easydict import EasyDict
import torch

import igl
import trimesh

"""
Stacked all functions in one file... 
NOTE! these are unsorted!!
"""

# rotations
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

def x_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[ 1.0,       0.0,        0.0,        0.0],
                  [ 0.0,         c,         -s,        0.0],
                  [ 0.0,         s,          c,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

# scaling
def rescale(V1 ,V2):
    """rescale V1 to V2"""
    V1mean = (V1.max(0)+V1.min(0))*0.5
    V2mean = (V2.max(0)+V2.min(0))*0.5
    V = (V1-V1mean)/max(V1.max(0)-V1.min(0)) * max(V2.max(0)-V2.min(0)) + V2mean
    return V

def normalize_v(V, mode='np'):
    if mode == 'np':
        V = np.array(V)
        V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    else: # torch
        V = (V-(V.max(0).values + V.min(0).values) * 0.5)/max(V.max(0).values - V.min(0).values)
    return V

def normalize_np(V):
    ### numpy
    V = np.array(V)
    V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    return V

def normalize_torch(V):
    ### torch
    V = (V-(V.max(0).values + V.min(0).values) * 0.5)/max(V.max(0).values - V.min(0).values)
    return V

def mesh_area(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    AB = v1 - v0
    AC = v2 - v0
    area = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)
    area_diag = np.repeat(area[:, np.newaxis], 3, axis=-1).reshape(-1)
    area = scipy.sparse.diags(area_diag)
    return area

def calc_norm(mesh):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def calc_norm_trimesh(vertices, faces):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def compute_triangle_centers(V, F):
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

def compute_triangle_normals(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals

def compute_vertex_normals(V, F, eps=1e-8):
    N = V.shape[0]
    Vn = np.zeros((N, 3))
    n = compute_triangle_normals(V, F)
    
    for i in range(3):
        np.add.at(Vn, F[:,i], n)
    Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True) + eps
    return Vn

def get_vertex_normals_from_tri_normals(V, F, Fn, eps=1e-8):
    N = V.shape[0]
    Vn = np.zeros((N, 3))
    n = compute_triangle_normals(V, F)
    
    for i in range(3):
        np.add.at(Vn, F[:,i], n)
    Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True) + eps
    return Vn

def get_triangle_basis(V, F):
    VF = V[F]
    E1 = VF[:,1] - VF[:,0]
    E2 = VF[:,2] - VF[:,0]

    Vn = compute_triangle_normals(V, F)
    
    # V4 = Vn + VF[:,0]
    # returns [V2-V1, V3-V1, V4-V1]
    return np.stack([E1, E2, Vn], axis=1).transpose(0,2,1)

def fit_local_basis(points):
    """
    Fit a local coordinate frame using PCA (best-fit plane normal = last principal axis)
    Returns: (3,3) basis matrix [e1; e2; n]
    """
    center = np.mean(points, axis=0)
    pts_centered = points - center
    U, S, VT = np.linalg.svd(pts_centered, full_matrices=False)
    basis = VT.T  # columns = principal axes
    return basis  # (3,3)

def get_cell_bbox_scales(V, F, eps=1e-8):
    """
    For each vertex, compute the bounding box scale (x,y,z) over its 1-disk region.
    Ensures all scales are strictly positive.

    Returns:
        scales: (N, 3) array of bounding box scales in object space
    """
    N = V.shape[0]
    scales = np.zeros((N, 3))
    vert2tris = [[] for _ in range(N)]

    for ti, tri in enumerate(F):
        for v in tri:
            vert2tris[v].append(ti)

    for vi in range(N):
        tris = vert2tris[vi]
        if not tris:
            continue

        v_set = set()
        for ti in tris:
            v_set.update(F[ti])
        v_set = list(v_set)
        local_pts = V[v_set]

        # fit local coordinate frame
        basis = fit_local_basis(local_pts)
        centered = local_pts - np.mean(local_pts, axis=0)
        obj_coords = centered @ basis  # project to local space

        # bounding box in local space
        bbox_min = np.min(obj_coords, axis=0)
        bbox_max = np.max(obj_coords, axis=0)
        scale = bbox_max - bbox_min

        # enforce positivity and stability
        scale = np.maximum(scale, eps)
        scales[vi] = scale

    return scales

# def build_vertex_adjacency(F):
#     adj = defaultdict(set)
#     for tri in F:
#         for i in range(3):
#             for j in range(i+1,3):
#                 u, v = tri[i], tri[j]
#                 adj[u].add(v); adj[v].add(u)
#     return {i: list(neigh) for i, neigh in adj.items()}

# def build_triangle_adjacency(F):
#     T = len(F)
#     edge_dict = defaultdict(list)
#     for t, tri in enumerate(F):
#         for i in range(3):
#             a, b = sorted((tri[i], tri[(i+1)%3]))
#             edge_dict[(a,b)].append(t)
#     adj = [[] for _ in range(T)]
#     for tris in edge_dict.values():
#         if len(tris) > 1:
#             for i in tris:
#                 for j in tris:
#                     if i != j and j not in adj[i]:
#                         adj[i].append(j)
#     return adj


# def look_triangle_has_v(vtx_idx, tri_idx, F):
#     holds = []
#     for idx, triangle in enumerate(F[:tri_idx]):
#         for vidx in triangle:
#             if vtx_idx == vidx:
#                 holds.append(idx)
                
#     for idx, triangle in enumerate(F[tri_idx+1:]):
#         for vidx in triangle:
#             if vtx_idx == vidx:
#                 holds.append(idx+tri_idx+1)
#     return holds

# def get_adj_triangle_mat(F):
#     mat = np.zeros([len(F), len(F)])

#     for i in range(len(F)):
#         js=[]
#         for vidx in F[i]:
#             j_list = look_triangle_has_v(vidx, i, F)
#             js += j_list
#         js = list(set(js))
#         mat[i,i] = len(js)
#         mat[i, js] = -1
#     return mat
    
# def get_adj_triangle(F):
#     mat = []
#     for i in range(len(F)):
#         js=[]
#         for vidx in F[i]:
#             j_list = look_triangle_has_v(vidx, i, F)
#             js += j_list
#         js = list(set(js))
#         mat.append(js)
#     return mat

def get_adj_triangle_list(F):
    """
    Return a list of adjacent triangle indices per triangle.
    F: (T,3) array of triangle indices.
    """
    edge_dict = defaultdict(list)
    for t_idx, tri in enumerate(F):
        edges = [(min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3])) for i in range(3)]
        for edge in edges:
            edge_dict[edge].append(t_idx)

    adj_list = [[] for _ in range(len(F))]
    for t_indices in edge_dict.values():
        if len(t_indices) < 2:
            continue
        for i in t_indices:
            for j in t_indices:
                if i != j and j not in adj_list[i]:
                    adj_list[i].append(j)
    return adj_list

def lu_factor_ata(A):
    """
    A sparse matrix
    """
    AtA = A.T @ A 
    lu = splu(AtA.tocsc())
    # x= lu.solve(ATb)
    return lu

def creat_A_csr_matrix(V, F, INV_MAT):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    # T (num triangle) x 3 (v indicies) x 3 (xyz coordinates)
    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
            -INV_MAT[:, :, 0:2].sum(-1, keepdims=True), 
            INV_MAT[:, :, 0:2]
        ), axis = -1)
                    
    R_=3
    C_=3
    for i, H in enumerate(COEFF): # foreach T
        # H : coeff mat 3x3            
        for r in range(R_):
            for c in range(C_): # foreach xyz
                for k, v_j in enumerate(F[i]): 
                    # v_j: vertex indicies in triangle
                    # k: num indicies (0, 1, 2)
                    cols.append(C_*v_j+c)
                    rows.append(9*i+R_*r+c)                
                    data.append(H[r,k])
                    
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(T * 3 * R_, N * C_))

def creat_A_csr_matrix(V, F, INV_MAT):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    # T (num triangle) x 3 (v indicies) x 3 (xyz coordinates)
    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
            -INV_MAT[:, :, 0:2].sum(-1, keepdims=True), 
            INV_MAT[:, :, 0:2]
        ), axis = -1)
                    
    R_=3
    C_=3
    for i, H in enumerate(COEFF): # foreach T
        # H : coeff mat 3x3            
        for r in range(R_):
            for c in range(C_): # foreach xyz
                for k, v_j in enumerate(F[i]): 
                    # v_j: vertex indicies in triangle
                    # k: num indicies (0, 1, 2)
                    cols.append(C_*v_j+c)
                    rows.append(9*i+R_*r+c)                
                    data.append(H[r,k])
                    
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(T * 3 * R_, N * C_))

def creat_A_csr_matrix_34_small(V, F, INV_MAT, W=1.0):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
        -INV_MAT.sum(-1, keepdims=True),
         INV_MAT
    ), axis=-1)

    R_, C_ = 3, 3
    for i, H in enumerate(COEFF):  # H: (3,4)
        for r in range(R_):
            row_idx = 3*i + r
            # vertex terms (v1~v3 + v4)
            cols.extend([ F[i][k]    for k in range(3)]+[N + i])
            rows.extend([ row_idx    for _ in range(3)]+[row_idx])
            data.extend([ W*H[r, k]  for k in range(3)]+[W*H[r, 3]])

    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3 * T, N + T))
    return MAT

def creat_A_csr_matrix_34(V, F, INV_MAT, W=1.0):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
        -INV_MAT.sum(-1, keepdims=True),
         INV_MAT
    ), axis=-1)

    R_, C_ = 3, 3
    for i, H in enumerate(COEFF):  # H: (3,4)
        for r in range(R_):
            for c in range(C_):
                row_idx = 9*i + R_*r + c
                # vertex terms (v1~v3 + v4)
                cols.extend([C_*F[i][k] + c for k in range(3)]+[C_*N + c + 3*i])
                rows.extend([row_idx]*3+[row_idx])
                data.extend([W*H[r, k] for k in range(3)]+[H[r, 3]])

    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3 * R_ * T, (C_ * N) + (C_ * T)))
    return MAT

def creat_A_csr_matrix_34_marker(V, F, src_m, tgt_m, V_tgt, W=1.0, return_RHS=False):
    """
    V, F: source vertices, triangles
    src_m: source marker
    tgt_m: source marker
    V_tgt: target vertices
    """
    N_m = len(src_m)
    
    assert len(src_m) == len(tgt_m)
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    I_ = 3
    for i, (s_m, t_m) in enumerate(zip(src_m, tgt_m)):
        cols.extend([I_*s_m + r for r in range(3)])
        rows.extend([I_*i + r for r in range(3)])
        data.extend([W*1.0 for r in range(3)])
    
    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(I_ * N_m, (I_ * N) + (I_ * T)))
    
    if return_RHS:
        RHS = V_tgt[np.array(tgt_m)].reshape(-1) * W
        return MAT, RHS
    return MAT

def creat_A_csr_matrix_34_marker_small(V, F, src_m, tgt_m, V_tgt, W=1.0, return_RHS=False):
    """
    V, F: source vertices, triangles
    src_m: source marker
    tgt_m: source marker
    V_tgt: target vertices
    """
    N_m = len(src_m)
    
    assert len(src_m) == len(tgt_m)
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    for i, (s_m, t_m) in enumerate(zip(src_m, tgt_m)):
                
        cols.extend([s_m])
        rows.extend([i])
        data.extend([W*1.0])
    
    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N_m, N + T))
    
    if return_RHS:
        RHS = V_tgt[np.array(tgt_m)] * W
        return MAT, RHS
    return MAT

    
def creat_A_csr_matrix_34_adj_triangle_small(V, F, INV_MAT, W=1.0, return_RHS=False):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    adj_list = get_adj_triangle_list(F)

    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
        -INV_MAT.sum(-1, keepdims=True),
         INV_MAT
    ), axis=-1)
    
    
    R_ = 3
    for i, adj_tri in enumerate(adj_list):
        deg = 1 / len(adj_tri)
        coeff_i = COEFF[i]
        F_i = F[i]

        for r in range(R_):
            row_idx = R_*i + r
            cols.extend([     F[i][k]     for k in range(3)]+[N + i])
            rows.extend([     row_idx     for _ in range(3)]+[row_idx])
            data.extend([ -W*coeff_i[r, k] for k in range(3)]+[-W*coeff_i[r, 3]])

            # # i triangle: v1~v3 + ghost
            # cols.extend([C_*v + c for v in F_i] + [C_*N + c + 3*i])
            # rows.extend([row_idx]*4)
            # data.extend([W * coeff_i[r, k] for k in range(3)] + [W * coeff_i[r, 3]])

            for j in adj_tri:
                coeff_j = COEFF[j]
                F_j = F[j]
                W_j = W * deg
                cols.extend([     F_j[k]        for k in range(3)]+[N + j])
                rows.extend([     row_idx       for _ in range(3)]+[row_idx])
                data.extend([ W_j*coeff_j[r, k] for k in range(3)]+[W_j*coeff_j[r, 3]])
            
                # cols.extend([C_*v + c for v in F_j] + [C_*N + c + 3*j])
                # rows.extend([row_idx]*4)
                # data.extend([W_j* coeff_j[r, k] for k in range(3)] + [W_j* coeff_j[r, 3]])

    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(R_ * T, N + T))
    if return_RHS:
        RHS = np.zeros((R_ * T, 3)) * W
        return MAT, RHS
    return MAT
    
def creat_A_csr_matrix_34_adj_triangle(V, F, INV_MAT, W=1.0, return_RHS=False):
    rows = []
    cols = []
    data = []

    T = F.shape[0]
    N = V.shape[0]

    adj_list = get_adj_triangle_list(F)

    INV_MAT = INV_MAT.transpose(0,2,1)
    COEFF = np.concatenate((
        -INV_MAT.sum(-1, keepdims=True),
         INV_MAT
    ), axis=-1)
    
    
    R_, C_ = 3, 3
    for i, adj_tri in enumerate(adj_list):
        deg = 1 / len(adj_tri)
        coeff_i = COEFF[i]
        F_i = F[i]

        for r in range(R_):
            for c in range(C_):
                row_idx = 9*i + R_*r + c

                # i triangle: v1~v3 + ghost
                cols.extend([C_*v + c for v in F_i] + [C_*N + c + 3*i])
                rows.extend([row_idx]*4)
                data.extend([W * coeff_i[r, k] for k in range(3)] + [W * coeff_i[r, 3]])

                for j in adj_tri:
                    coeff_j = COEFF[j]
                    F_j = F[j]
                    cols.extend([C_*v + c for v in F_j] + [C_*N + c + 3*j])
                    rows.extend([row_idx]*4)
                    data.extend([-W  * deg* coeff_j[r, k] for k in range(3)] + [-W  * deg* coeff_j[r, 3]])

    MAT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3 * R_ * T, (C_ * N) + (C_ * T)))
    if return_RHS:
        RHS = np.zeros((3 * R_ * T,)) * W
        return MAT, RHS
    return MAT



def find_closest_valid_points(V_src, F_src, V_tar, F_tar, normal_weight=0.5, k=10):
    """
    For each source vertex, find the closest valid point on the target mesh,
    considering both distance and normal similarity. Triangles >90° normal mismatch are skipped.

    Args:
        V_src: (N,3) source vertices
        F_src: (T1,3) source triangles
        V_tar: (M,3) target vertices
        F_tar: (T2,3) target triangles
        normal_weight: (0~1) balance between distance and normal alignment
        k: number of nearest triangle candidates to consider

    Returns:
        closest_points: (N,3) closest valid points on target mesh
    """
    N = V_src.shape[0]
    src_normals = compute_vertex_normals(V_src, F_src)      # (N,3)
    tri_centers = compute_triangle_centers(V_tar, F_tar)    # (T,3)
    tri_normals = compute_triangle_normals(V_tar, F_tar)    # (T,3)

    tree = cKDTree(tri_centers)
    dists, idxs = tree.query(V_src, k=k)

    closest_points = np.zeros((N, 3))
    for i in range(N):
        best_score = np.inf
        best_point = V_src[i] # self
        v = V_src[i]
        n = src_normals[i]

        for j in idxs[i]:
            tri = V_tar[F_tar[j]] # (3,3)
            t_n = tri_normals[j]
            dot = np.dot(n, t_n)

            if dot < 0:
                continue

            proj = project_point_to_triangle(v, tri)
            dist = np.linalg.norm(v - proj)
            normal_sim = 1.0 - dot       # lower = better
            score = (1 - normal_weight) * dist + normal_weight * normal_sim

            if score < best_score:
                best_score = score
                best_point = proj

        closest_points[i] = best_point

    return closest_points

def find_closest_valid_points_normal_first(V_src, F_src, V_tar, F_tar, normal_weight=0.5, k=10):
    """
    For each source vertex, find the closest valid point on the target mesh,
    prioritizing normal similarity first, then distance.
    
    Args:
        V_src: (N,3) source vertices
        F_src: (T1,3) source triangles
        V_tar: (M,3) target vertices
        F_tar: (T2,3) target triangles
        normal_weight: (0~1) balance between distance and normal alignment (used in final ranking)
        k: number of top normal-matching triangles to consider

    Returns:
        closest_points: (N,3) closest valid points on target mesh
    """
    N = V_src.shape[0]
    src_normals = compute_vertex_normals(V_src, F_src)      # (N,3)
    tri_centers = compute_triangle_centers(V_tar, F_tar)    # (T,3)
    tri_normals = compute_triangle_normals(V_tar, F_tar)    # (T,3)

    closest_points = np.zeros((N, 3))
    for i in range(N):
        v = V_src[i]
        n = src_normals[i]

        # similar normal direction
        dots = tri_normals @ n
        valid = np.where(dots >= 0)[0]
        # if len(valid) == 0:
        #     closest_points[i] = v
        #     continue

        topk_idx = valid[np.argsort(-dots[valid])[:k]]
        best_score = np.inf
        best_point = v  # fallback

        for j in topk_idx:
            tri = V_tar[F_tar[j]]
            t_n = tri_normals[j]
            proj = project_point_to_triangle(v, tri)
            dist = np.linalg.norm(v - proj)
            normal_sim = 1.0 - np.dot(n, t_n)
            score = (1 - normal_weight) * dist + normal_weight * normal_sim

            if score < best_score:
                best_score = score
                best_point = proj

        closest_points[i] = best_point

    return closest_points


def find_closest_valid_feature(V_src, F_src, V_tar, F_tar, val_tar=None, normal_weight=0.5, k=5, is_mat=False, use_pnt=False):
    """
    Return closest valid points on the target mesh for each source vertex.
    
    Args:
        V_src: (N,3) source vertices
        F_src: (N,3) source triangles
        V_tar: (M,3) target vertices
        F_tar: (T,3) target triangles
        val_tar: (M, C) target per vertex feature
        normal_weight: how much weight to give to normal similarity (0~1)
    """
    if val_tar is None:
        val_tar = V_src
        
    N_src = compute_vertex_normals(V_src, F_src)
    #print()
    N = V_src.shape[0]
    tri_centers = compute_triangle_centers(V_tar, F_tar)    # (T,3)
    tri_normals = compute_triangle_normals(V_tar, F_tar)    # (T,3)

    tree = cKDTree(tri_centers)  # Efficient nearest triangle lookup

    if is_mat:
        closest_points = np.zeros((V_src.shape[0], 3, 3))
    else:
        closest_points = np.zeros_like(V_src)

    for i in range(N):
        v_src = V_src[i]
        n_src = N_src[i]

        # 1. Query k nearest triangle centers
        dists, idxs = tree.query(v_src, k=k)  # k-nearest triangle centers
        best_score = float('inf')
        # best_point = None
        best_tri = None

        for j, dist in zip(idxs, dists):
            # tri = V_tar[F_tar[j]]
            tri = F_tar[j]
            tri_normal = tri_normals[j]

            # Normal similarity (1 - cosine)
            normal_sim = 1 - np.dot(n_src, tri_normal)

            # Combined score (smaller the better)
            score = (1 - normal_weight) * dist + normal_weight * normal_sim

            if score < best_score:
                best_score = score
                # best_point = proj
                best_tri = tri

        # closest_points[i] = best_point
        u,v,w = project_point_to_triangle(v_src, V_tar[best_tri], return_weight=True)
        val_tri = val_tar[best_tri]
        # print(val_tri.shape)
        if is_mat: # val_tar : Mx3x3
            closest_value = blend_rotation_quaternions_dq(val_tri, (u,v,w))
        else: # val_tar Mx3
            closest_value = u*val_tri[0] + v*val_tri[1] + w*val_tri[2]
            
        if use_pnt:
            if u < v:
                if v < w:
                    # u,v,w = 0,0,1
                    closest_value = val_tri[2]
                else:
                    # u,v,w = 0,1,0
                    closest_value = val_tri[1]
            else:
                if w < u:
                    # u,v,w = 1,0,0
                    closest_value = val_tri[0]
                else:
                    # u,v,w = 0,0,1
                    closest_value = val_tri[2]
        
        closest_points[i] = closest_value

    return closest_points

def find_closest_valid_feature_normal_first(V_src, F_src, V_tar, F_tar,
                                            val_tar=None, normal_weight=0.5, k=5, is_mat=False):
    if val_tar is None:
        val_tar = V_src.copy().astype(np.float64)

    src_normals = compute_vertex_normals(V_src, F_src)
    tri_centers = compute_triangle_centers(V_tar, F_tar)
    tri_normals = compute_triangle_normals(V_tar, F_tar)

    nearest_idxs = []
    bary_weights = []

    # No need for KD-tree on centers since we filter by normals first
    for i, v in enumerate(V_src.astype(np.float64)):
        n_src = src_normals[i]
        # dot with all target normals
        dots = tri_normals @ n_src
        valid = np.where(dots > 0)[0]
        
        if valid.size == 0:
            nearest_idxs.append(-1)
            bary_weights.append(np.zeros(3, dtype=np.float64))
            continue
            
        # find closest among valid centers
        centers_valid = tri_centers[valid]
        dists = np.linalg.norm(centers_valid - v, axis=1)
        best = valid[np.argmin(dists)]
        _, bc = project_point_to_triangle(v, V_tar[F_tar[best]], return_weight=True)
        nearest_idxs.append(best)
        bary_weights.append(bc)

    return np.array(nearest_idxs, dtype=int), np.vstack(bary_weights)

def compute_geodesic_distance_matrix(V, F):
    N = V.shape[0]
    I, J = F[:, [0,1,2]], F[:, [1,2,0]]
    edges = np.concatenate([I.reshape(-1,1), J.reshape(-1,1)], axis=1)
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    vi, vj = edges[:,0], edges[:,1]
    edge_len = np.linalg.norm(V[vi] - V[vj], axis=1)
    W = scipy.sparse.coo_matrix((edge_len, (vi, vj)), shape=(N,N))
    W = W + W.T
    dist_matrix = csgraph.dijkstra(W, directed=False)
    return dist_matrix


def corr_system(src_mesh, tgt_mesh, src_m, tgt_m, num_iter=1, Ws=1, Wi=0.001, Wc=0, Wm=1.0, show=False, mode=1):
    N = len(src_mesh.v)
    T = len(src_mesh.f)
    
    src_def_mesh = EasyDict()
    src_def_mesh.v = src_mesh.v.copy()
    src_def_mesh.f = src_mesh.f
    
    for i in range(num_iter):
        # source restpose span ####################
        V_S = get_triangle_basis(src_def_mesh.v, src_def_mesh.f)
        ###########################################
        
        ## a.k.a source transformation `S = V^~ @ V^-1`
        V_S_INV = np.linalg.inv(V_S)
        ###########################################
        
        ## Marker constraint ######################
        # A_m, RHS_m = creat_A_csr_matrix_34_marker(src_def_mesh.v, src_def_mesh.f, src_m, tgt_m, tgt_mesh.v, W=Wm, return_RHS=True)
        # little bit faster
        A_m, RHS_m = creat_A_csr_matrix_34_marker_small(src_def_mesh.v, src_def_mesh.f, src_m, tgt_m, tgt_mesh.v, W=Wm, return_RHS=True)
        ###########################################
        
        
        A_list = [A_m]
        RHS_list = [RHS_m]
        
        ## Deformation smoothness #################
        if Ws > 0:
            #A_ds, RHS_ds = creat_A_csr_matrix_34_adj_triangle(src_def_mesh.v, src_def_mesh.f, V_S_INV, W=Ws, return_RHS=True)
            A_ds, RHS_ds = creat_A_csr_matrix_34_adj_triangle_small(src_def_mesh.v, src_def_mesh.f, V_S_INV, W=Ws, return_RHS=True)
            A_list.append(A_ds)
            RHS_list.append(RHS_ds)
        ###########################################
        
        ## Deformation identity ###################
        if Wi > 0:
            # A_i = creat_A_csr_matrix_34(src_def_mesh.v, src_def_mesh.f, V_S_INV, W=Wi)
            # RHS_i = np.eye(3)[None].repeat(src_def_mesh.f.shape[0], 0).reshape(-1) * Wi
            A_i = creat_A_csr_matrix_34_small(src_def_mesh.v, src_def_mesh.f, V_S_INV, W=Wi)
            RHS_i = np.eye(3)[None].repeat(src_def_mesh.f.shape[0], 0).reshape(-1, 3) * Wi
            A_list.append(A_i)
            RHS_list.append(RHS_i)
        ###########################################

        # cloest valid point ######################
        if Wc > 0:
            if mode==1:
                closest_pts0 = find_closest_valid_points(V_src=src_def_mesh.v, F_src=src_def_mesh.f, V_tar=tgt_mesh.v, F_tar=tgt_mesh.f, normal_weight=0.1, k=5)
            elif mode == 2:
                closest_pts0 = find_closest_valid_points_normal_first(V_src=src_def_mesh.v, F_src=src_def_mesh.f, V_tar=tgt_mesh.v, F_tar=tgt_mesh.f, normal_weight=0.1, k=5)
            # A_c, RHS_c = creat_A_csr_matrix_34_marker_small(src_mesh.v, src_mesh.f, src_m, tgt_m, closest_pts0, W=Wc, return_RHS=True)
            A_c, RHS_c = creat_A_csr_matrix_34_marker_small(src_def_mesh.v, src_def_mesh.f, torch.arange(N), torch.arange(N), closest_pts0, W=Wc, return_RHS=True)
            A_list.append(A_c)
            RHS_list.append(RHS_c)
        ###########################################
        
        ## stack A ################################
        # building ATA and solver
        A_ = scipy.sparse.vstack(A_list, format='csr')
        lu_ = lu_factor_ata(A_)
        ###########################################
            
        ## stack RHS ##############################
        # building ATb
        # RHS = np.hstack(RHS_list)
        RHS_ = np.vstack(RHS_list)
        Atb_ = A_.T @ RHS_
        ###########################################
        
        
        # solve ###################################
        solved_v = lu_.solve(Atb_)
        # solved_v = solved_v.reshape(-1,3)[:N]
        ###########################################
        
        src_def_mesh.v = solved_v[:N].copy()
        # recenter
        #src_def_mesh.v = (solved_v - solved_v.mean(0) + src_mesh.v.mean(0)).copy()
        
        if show:
            import meshplot as mp
            d = mp.subplot(tgt_mesh.v, tgt_mesh.f, s=[2, 2, 0])
            dd = mp.subplot(src_def_mesh.v, src_def_mesh.f, data=d, s=[2, 2, 1])
            mp.subplot(src_mesh.v, src_mesh.f, data=dd, s=[2, 2, 2])
            # v_list1=[src_mesh.v, src_def_mesh.v, tgt_mesh.v,]
            # f_list1=[src_mesh.f, src_def_mesh.f, tgt_mesh.f,]
            # rot_list=[ [0,0,0] ]*len(v_list)
            # plot_mesh_gouraud(v_list1, f_list1, mesh_scale=.65, rot_list=rot_list, size=3, mode='shade')
    
    return src_def_mesh




def project_point_to_triangle(p, tri, return_weight=False):
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = ab @ ap
    d2 = ac @ ap
    d00 = ab @ ab
    d01 = ab @ ac
    d11 = ac @ ac

    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d1 - d01 * d2) / denom
    w = (d00 * d2 - d01 * d1) / denom
    u = 1.0 - v - w

    # clamp to triangle
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    w = 1.0 - u - v
    if return_weight:
        return np.array([u, v, w])
    else:
        return u*a + v*b + w*c

def get_triangle_barycentric(p, tri):
    # Project point p onto triangle (3,3) using barycentric coordinates
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    d00 = np.dot(ab, ab)
    d01 = np.dot(ab, ac)
    d11 = np.dot(ac, ac)
    denom = d00 * d11 - d01 * d01 + 1e-8

    v = (d11 * d1 - d01 * d2) / denom
    w = (d00 * d2 - d01 * d1) / denom
    u = 1.0 - v - w

    # Clamp to triangle if outside (optional)
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    w = 1 - u - v

    return u, v, w

def get_vertex_to_face_map(V, F):
    """
    For each vertex, returns a list of face indices it belongs to.

    Args:
        V (np.ndarray): Vertices array of shape (n, 3)
        F (np.ndarray): Faces array of shape (m, 3)

    Returns:
        vertex_face_map (list of lists): vertex_face_map[i] contains the list of face indices that vertex i belongs to
    """
    num_vertices = V.shape[0]
    vertex_face_map = [[] for _ in range(num_vertices)]

    for face_idx, face in enumerate(F):
        for vertex_idx in face:
            vertex_face_map[vertex_idx].append(face_idx)

    return vertex_face_map

def get_TBN_foreach_vertex(V, F):
    """
    Get TBN matrix for each vertex

    Args:
        V (np.ndarray): vertices (N, 3)
        F (np.ndarray): faces / trianlges (M, 3)

    Returns:
        np.ndarray: matrix (N, 3, 3)
        [
            [Tx, Ty, Tz],
            [Bx, By, Bz],
            [Nx, Ny, Nz],
        ]
    """
    # normal
    Vn = compute_vertex_normals(V, F)
    
    al = igl.adjacency_list(F) # all neighbors
    nb = [nbrs[0] for nbrs in al] # select one
    
    Vt_nb = V[nb] - V
    Vt_nb = Vt_nb/(np.linalg.norm(Vt_nb, axis=1)[:,None]+1e-8)

    # project
    _Vt = np.sum(Vt_nb * Vn, axis=1, keepdims=True)
    
    # tangent
    Vt = Vt_nb - (_Vt * Vn)
    Vt = Vt/(np.linalg.norm(Vt, axis=1, keepdims=True)+1e-8)
    
    # bitangent
    Vb = np.cross(Vn, Vt)
    Vb = Vb/(np.linalg.norm(Vb, axis=1, keepdims=True)+1e-8)
    
    return np.concatenate([
        Vt[:,None], 
        Vb[:,None],
        Vn[:,None], 
    ], axis=1)

def get_TriangleArea_foreach_vertex(V,F):
    """
    Returns (V,)
    """
    DArea = igl.doublearea(V, F) * 0.5
    print(DArea.shape)

    V2F = get_vertex_to_face_map(V,F)
    VArea = []
    for v2f in V2F:
        VArea.append(np.sum(DArea[v2f])/len(v2f))
    VArea = np.array(VArea)
    return VArea

def get_scale_foreach_vertex(V,F):
    """
    Returns (V, 3)
    """
    # Vxyz = []
    # for vf in V[F]: # F, 3, 3
    #     Vxyz.append(vf.max(0)-vf.min(0))
    # Vxyz = np.array(Vxyz)
    
    V2F = get_vertex_to_face_map(V,F)
    Vscale = []
    for v2f in V2F:
        Vscale.append(np.std(V[F[v2f]].reshape(-1,3), axis=0))
    Vscale = np.array(Vscale)
    return Vscale

def get_scale_foreach_triangle(Vsrc, Fsrc):
    """
    get scale ratio of DeformedSourceModelLocalBoudingBox / SourceModelLocalBoudingBox
    Returns (V, 3, 3)
    """
    S_matrices = np.zeros((Vsrc.shape[0], 3))

    all_nbh = igl.adjacency_list(Fsrc) # all neighbors for each vertex    
    for v_idx, nbh in enumerate(all_nbh):

        # get positions in both source and deformed meshes
        local_src = np.r_[Vsrc[nbh], Vsrc[v_idx]]

        # bounding box: max - min
        bbox_src = np.max(local_src, axis=0) - np.min(local_src, axis=0)

        # prevent divide by zero
        bbox_src[bbox_src == 0] = 1e-8

        S_matrices[v_idx] = bbox_src

    return S_matrices

## mesh smoothing function
def smooth(vert, lap, loop=1, t=0.01):
    D_Inv = scipy.sparse.diags(1 / lap.diagonal())
    _lap = D_Inv @ lap
    
    # a = 1-t
    I_L = scipy.sparse.identity(lap.shape[0]) - _lap*t
    # I_L = (I_L*t).power(loop)
    for i in range(loop):
        I_L = I_L.T @ I_L
    
    vert = I_L @ vert
    return vert

def taubin_smooth(vert, lap, values, loop=2, m=0.01, l=0.01):
    D_Inv = scipy.sparse.diags(1 / lap.diagonal())
    _lap = D_Inv @ lap
    
    I_L = scipy.sparse.identity(lap.shape[0]) - _lap*l
    I_M = scipy.sparse.identity(lap.shape[0]) - _lap*m
    
    I_L = I_L @ I_M
    for i in range(loop-1):
        I_L = I_L @ I_L
    
    values = I_L @ values
    return values

def mesh_smooth(V, F, values, tau=0.001):

    # Mesh smoothing with libigl
    l = igl.cotmatrix(V, F)  # laplace-beltrami operator in libigl
    m = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # mass matrix in libigl
    s = m - tau * l
    return spsolve(s, m @ values)


# playing with rotation representation
def random_rotation_matrix(randgen=None):
    """
    Borrowed from https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py
    
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
    
def random_rotate_points(pts, randgen=None, return_rot=False):
    R = random_rotation_matrix(randgen) 
    if return_rot:
        return np.matmul(pts, R), R
    return np.matmul(pts, R)

def rodrigues_rotation_matrix(rotvec):
    """
    Args:
        rotvec (np.ndarray): (3,) rotvec = axis * angle
    Returns:
        R (np.ndarray) : (3, 3) rotation matrix
    """
    
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.eye(3)

    axis = rotvec / theta
    x, y, z = axis

    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotation_matrix_from_vectors(v1, v2):
    """v1 → v2 (Rodrigues' formula)

    Args:
        v1, v2 (np.ndarray): (3,)

    Returns:
        R (np.ndarray): (3,3)
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        # find other axis
        ortho = np.array([1, 0, 0]) if not np.isclose(v1[0], 1.0) else np.array([0, 1, 0])
        axis = np.cross(v1, ortho)
        axis /= np.linalg.norm(axis)
        return rodrigues_rotation_matrix(axis * np.pi)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    R = np.eye(3) + skew + (skew @ skew) * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R

def rotation_matrix_from_vectors_batch(v1s, v2s):
    """
    Args:
        v1s (np.ndarray): (B, 3)
        v2s (np.ndarray): (B, 3)

    Returns:
        R (np.ndarray): (B, 3, 3)
    """
    v1s = v1s / np.linalg.norm(v1s, axis=1, keepdims=True)
    v2s = v2s / np.linalg.norm(v2s, axis=1, keepdims=True)

    B = v1s.shape[0]
    cross = np.cross(v1s, v2s)
    dot = np.einsum('bi,bi->b', v1s, v2s)

    R_all = np.zeros((B, 3, 3))

    for i in range(B):
        if np.isclose(dot[i], 1.0):
            R_all[i] = np.eye(3)
        elif np.isclose(dot[i], -1.0):
            v1 = v1s[i]
            ortho = np.array([1, 0, 0]) if not np.isclose(v1[0], 1.0) else np.array([0, 1, 0])
            axis = np.cross(v1, ortho)
            axis = axis / np.linalg.norm(axis)
            R_all[i] = rodrigues_rotation_matrix(axis * np.pi)
        else:
            c = cross[i]
            c_norm_sq = np.dot(c, c)
            skew = np.array([
                [0, -c[2], c[1]],
                [c[2], 0, -c[0]],
                [-c[1], c[0], 0]
            ])
            R = np.eye(3) + skew + skew @ skew * ((1 - dot[i]) / c_norm_sq)
            R_all[i] = R

    return R_all

def from_6D_to_rotation_matrix_torch(in_6d, eps=1e-12):
    """
    6D representation (B, 6) → rotation matrix (B, 3, 3) *following Zhou et al. (CVPR 2019)
    Args:
        in_6d (torch.Tensor): (B, 6)
    Returns:
        R (torch.Tensor): (B, 3, 3)
    """
    a1, a2 = in_6d[..., :3], in_6d[..., 3:]
    
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack((b1, b2, b3), dim=-2)


# using torch
@torch.no_grad()
def rodrigues_rotation_matrix_torch(rotvec):
    """
    rotvec: (B, 3) rotation vectors
    returns: (B, 3, 3) rotation matrices
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # (B, 1)
    axis = rotvec / (theta + 1e-8)                    # (B, 3)
    x, y, z = axis.unbind(-1)

    K = torch.stack([
        torch.stack([torch.zeros_like(x), -z, y], dim=-1),
        torch.stack([z, torch.zeros_like(x), -x], dim=-1),
        torch.stack([-y, x, torch.zeros_like(x)], dim=-1),
    ], dim=-2)  # (B, 3, 3)

    I = torch.eye(3, device=rotvec.device).unsqueeze(0)
    theta = theta.unsqueeze(-1)  # (B, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

@torch.no_grad()
def rotation_matrix_from_vectors_batch_fast(v1s, v2s, eps=1e-12):
    """
    Args:
        v1s, v2s: (B, N, 3)
    Returns:
        R: (B, N, 3, 3)
    """
    if v1s.ndim == 2:
        v1s = v1s.unsqueeze(0)
        v2s = v2s.unsqueeze(0)

    B, N = v1s.shape[:2]
    device = v1s.device

    v1 = v1s / (v1s.norm(dim=-1, keepdim=True) + eps)
    v2 = v2s / (v2s.norm(dim=-1, keepdim=True) + eps)

    cross = torch.cross(v1, v2, dim=-1)              # (B, N, 3)
    dot = (v1 * v2).sum(dim=-1)                      # (B, N)
    v_norm_sq = (cross ** 2).sum(dim=-1) + eps       # (B, N)

    # skew-symmetric matrix K
    K = torch.zeros(B, N, 3, 3, device=device)
    K[:, :, 0, 1] = -cross[:, :, 2]
    K[:, :, 0, 2] =  cross[:, :, 1]
    K[:, :, 1, 0] =  cross[:, :, 2]
    K[:, :, 1, 2] = -cross[:, :, 0]
    K[:, :, 2, 0] = -cross[:, :, 1]
    K[:, :, 2, 1] =  cross[:, :, 0]

    I = torch.eye(3, device=device).view(1, 1, 3, 3)
    K2 = K @ K
    coef = ((1 - dot) / v_norm_sq).view(B, N, 1, 1)
    R = I + K + coef * K2  # (B, N, 3, 3)

    # dot ≈ 1
    mask_eq1 = (dot > 1.0 - 1e-4)
    R[mask_eq1] = torch.eye(3, device=device)

    # dot ≈ -1
    mask_eq_neg1 = (dot < -1.0 + 1e-4)
    if mask_eq_neg1.any():
        idx = mask_eq_neg1.nonzero(as_tuple=True)
        v1_neg = v1[idx]  # (M, 3)
        ortho = torch.where(
            (v1_neg[:, 0].abs() < 0.9).unsqueeze(-1),
            torch.tensor([1., 0., 0.], device=device),
            torch.tensor([0., 1., 0.], device=device)
        )
        axis = torch.cross(v1_neg, ortho)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
        R180 = rodrigues_rotation_matrix_torch(axis * torch.pi)  # (M, 3, 3)
        R[idx[0], idx[1]] = R180

    return R


# some experiments...
def blend_three_quaternions_dq(R0, R1, R2, weights):
    """
    Blend 3 unit quaternions Q, R_, S using barycentric weights via dual quaternion-style.

    Args:
        R0, R1, R2: (4,) quaternions in [x, y, z, w] format (as in scipy)
        weights: (3,) barycentric weights (u, v, w)

    Returns:
        (3,3) rotation matrix representing blended rotation
    """

    # Weighted sum of quaternions
    quat_blend = weights[0] * R0 + weights[1] * R1 + weights[2] * R2

    # Normalize back to unit quaternion
    quat_blend /= np.linalg.norm(quat_blend)

    # Convert to rotation matrix
    return R.from_quat(quat_blend).as_matrix()

def blend_rotation_quaternions_dq(tri_mat, weights):
    """
    Blend 3 unit quaternions Q, R_, S using barycentric weights via dual quaternion-style.

    Args:
        tri_mat: (Nx3x3) 
        weights: (3,) barycentric weights (u, v, w)

    Returns:
        (3,3) rotation matrix representing blended rotation
    """

    _R0 = R.from_matrix(tri_mat[0]).as_quat()
    _R1 = R.from_matrix(tri_mat[1]).as_quat()
    _R2 = R.from_matrix(tri_mat[2]).as_quat()
    
    # Weighted sum of quaternions
    quat_blend = weights[0] * _R0 + weights[1] * _R1 + weights[2] * _R2

    # Normalize back to unit quaternion
    quat_blend /= np.linalg.norm(quat_blend)

    # Convert to rotation matrix
    return R.from_quat(quat_blend).as_matrix()


# using torch
def compute_face_norm(v, f):
    vf = v[f]
    e1 = vf[..., 0] - vf[..., 1]
    e2 = vf[..., 1] - vf[..., 2]
    norm = (torch.cross(e1, e2))
    norm = norm / (torch.linalg.norm(norm, dim=-1, keepdims=True) + 1e-12)
    return norm

def compute_vert_norm(faces, face_normals, num_vertices=None):
    """Computes normals for every vertex by averaging face normals
    assigned to that vertex for every face that has this vertex.

    Args:
       faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
            shape (F, 3).
       face_normals (torch.FloatTensor): pre-normalized xyz normal values
            for every vertex of every face with shape (B, F, 3)
       num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

    Return:
        (torch.FloatTensor): of shape (B, V, 3)
    """
    
    if num_vertices is None:
        num_vertices = int(faces.max()) + 1


    B = face_normals.shape[0]
    V = num_vertices
    F = faces.shape[0]
    FSz = faces.shape[1]
    # print(B, V, F, FSz)

    vertex_normals = torch.zeros((B, V, 3), dtype=face_normals.dtype, device=face_normals.device)
    counts = torch.zeros((B, V), dtype=face_normals.dtype, device=face_normals.device)

    faces = faces.unsqueeze(0)
    fake_counts = torch.ones((B, F), dtype=face_normals.dtype, device=face_normals.device)
    #              B x F          B x F x 3
    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    
    for i in range(FSz):
        # print(faces[..., i:i + 1].repeat(1, 1, 3).shape) ## torch.Size([1, F, 3])
        # print(face_normals.shape)                        ## torch.Size([1, F])
        vertex_normals.scatter_add_(1, faces[..., i:i + 1].repeat(1, 1, 3), face_normals)
        counts.scatter_add_(1, faces[..., i], fake_counts)

    counts = counts.clip(min=1).unsqueeze(-1)
    vertex_normals = vertex_normals / counts
    return vertex_normals

# something wrong
def _computeTangentBasis(vertex, uv):
    tangents = []
    tangents = np.zeros_like(vertex)
    # bitangents = []
    for idx in range(0, len(vertex)//3):
        
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
    return tangents


def average_rotation_matrix_batch(rot_mats):
    """
    Args:
        rot_mats: (B, N, 3, 3) torch.Tensor

    Returns:
        R_mean: (B, 3, 3)
    """
    assert rot_mats.ndim == 4 and rot_mats.shape[2:] == (3, 3)
    B, N = rot_mats.shape[:2]
    
    M = rot_mats.mean(dim=1)  # (B, 3, 3)
    
    U, _, Vh = torch.linalg.svd(M)  # 각: (B, 3, 3)
    
    R = U @ Vh
    
    det = torch.linalg.det(R)
    mask = det < 0
    if mask.any():
        U[mask, :, -1] *= -1
        R[mask] = U[mask] @ Vh[mask]

    return R

def quaternion_to_rotation_matrix(q):
    """
    Quaternion (w, x, y, z) → Rotation matrix
    Args:
        q: (B, 4) or (4,) torch tensor, where q = [w, x, y, z]

    Returns:
        R: (B, 3, 3) or (3, 3) rotation matrix
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)  # (1, 4)

    q = q / q.norm(dim=1, keepdim=True)  # normalize
    w, x, y, z = q.unbind(dim=1)

    B = q.shape[0]

    R = torch.empty((B, 3, 3), dtype=q.dtype, device=q.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    if R.shape[0] == 1:
        return R[0]  # remove batch dim if input was (4,)
    return R

# remeshings
def decimate_mesh(mesh, target_faces):
    """decimate mesh
    Args:
        mesh (trimesh.Trimesh)
    Return
        decimated_mesh (trimesh.Trimesh)
    """
    decimated_mesh = mesh.simplify_quadric_decimation(target_faces)
    return decimated_mesh

def map_vertices(original_mesh, decimated_mesh):
    """
    Get nearest neighbor vertex on original mesh
    Args:
        original_mesh (trimesh.Trimesh)
        decimated_mesh (trimesh.Trimesh)
    Return
        distances (np.ndarray): distance with the nearest neighbor vertex on original mesh
        vertex_map (np.ndarray): index of the nearest neighbor vertex on original mesh
    """
    tree = cKDTree(original_mesh.vertices)
    distances, vertex_map = tree.query(decimated_mesh.vertices, k=1)
    return distances, vertex_map

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """calculate standardized mesh
    Args:
        vertices (torch.tensor): [V, 3] array of vertices 
        faces (torch.tensor): [F, 3] array of face indices 
        v_idx (torch.tensor): [N] list of vertex index to remove from mesh
    Return:
        updated_verts (torch.tensor): [V', 3] new array of vertices 
        updated_faces (torch.tensor): [F', 3] new array of face indices  
        updated_verts_idx (torch.tensor): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = torch.arange(max_index)

    if invert:
        mask = torch.zeros(max_index, dtype=torch.bool)
        mask[v_idx] = 1
    else:
        mask = torch.ones(max_index, dtype=torch.bool)
        mask[v_idx] = 0

    updated_verts     = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = torch.tensor([
                    [index_mapping.get(idx.item(), -1) for idx in face]
                    for face in faces
                ])
    valid_faces = ~torch.any(updated_faces == -1, dim=1)
    updated_faces = updated_faces[valid_faces]
    return updated_verts, updated_faces, updated_verts_idx

def decimate_mesh_vertex(mesh, num_vertex, tolerance=2, verbose=False):
    """
    Decimate the mesh to have approximately the target number of vertices.
    Args:
        mesh (trimesh.Trimesh): Mesh to decimate.
        num_vertex (int): Target vertex number.
    Return:
        mesh (trimesh.Trimesh): Decimated mesh.
    """
    
    #NOTE Euler Characteristic: V - E + F = 2
    num_faces = 100 + 2 * num_vertex
    prev_num_faces = mesh.faces.shape[0]
    
    while abs(mesh.vertices.shape[0] - num_vertex) > tolerance:
        if num_faces == prev_num_faces:
            num_faces = num_faces -1
        mesh = mesh.simplify_quadric_decimation(num_faces)
        if verbose:
            print("Decimated to", num_faces, "faces, mesh has", mesh.vertices.shape[0], "vertices")
        num_faces -= (mesh.vertices.shape[0] - num_vertex) // 2
        prev_num_faces = num_faces
    if verbose:
        print('Output mesh has', mesh.vertices.shape[0], 'vertices and', mesh.faces.shape[0], 'faces')
    return mesh

# for visualization
def compute_face_gradient(V, F, f):
    grads = np.zeros((F.shape[0], 3))
    for i, face in enumerate(F):
        v0, v1, v2 = V[face]
        f0, f1, f2 = f[face]

        e1 = v1 - v0
        e2 = v2 - v0

        normal = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(normal)
        normal /= (2 * area + 1e-8)

        g = ((f1 - f0) * np.cross(normal, v2 - v0) +
             (f2 - f0) * np.cross(v0 - v1, normal)) / (2 * area + 1e-8)

        grads[i] = g

    return grads


def compute_MVC_vertexwise_torch(src_v, cage_v, cage_f, eps=1e-8):
    """
    Compute Mean Value Coordinates per cage vertex for each mesh vertex

    Args:
        src_v (torch.tensor): (V, 3) source points
        cage_v (torch.tensor): (Nc, 3) cage/control points
        cage_f (torch.tensor): (F, 3) triangle indices of cage faces
        eps (float): numerical stability epsilon

    Returns:
        weights per cage vertex (V, Nc)
    """
    V = src_v.shape[0]
    Nc = cage_v.shape[0]
    F = cage_f.shape[0]

    # Get cage face vertices
    i0, i1, i2 = cage_f[:, 0], cage_f[:, 1], cage_f[:, 2]
    P0 = cage_v[i0]  # (F, 3)
    P1 = cage_v[i1]
    P2 = cage_v[i2]

    X = src_v[:, None, :]  # (V, 1, 3)

    D0 = P0[None, :, :] - X  # (V, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    # clamp min by eps to prevent division by 0
    d0 = D0.norm(dim=-1).clamp(min=eps)
    d1 = D1.norm(dim=-1).clamp(min=eps)
    d2 = D2.norm(dim=-1).clamp(min=eps)

    U0 = D0 / d0[..., None]
    U1 = D1 / d1[..., None]
    U2 = D2 / d2[..., None]

    # compute angles between normalized directions
    L0 = (U1 - U2).norm(dim=-1)
    L1 = (U2 - U0).norm(dim=-1)
    L2 = (U0 - U1).norm(dim=-1)

    theta0 = 2 * torch.arcsin(torch.clamp(L0 * 0.5, -0.999999, 0.999999))
    theta1 = 2 * torch.arcsin(torch.clamp(L1 * 0.5, -0.999999, 0.999999))
    theta2 = 2 * torch.arcsin(torch.clamp(L2 * 0.5, -0.999999, 0.999999))
    h = 0.5 * (theta0 + theta1 + theta2)

    near_pi = torch.abs(torch.pi - h) < eps
    not_near_pi = ~near_pi

    s_theta0 = torch.sin(theta0)
    s_theta1 = torch.sin(theta1)
    s_theta2 = torch.sin(theta2)

    # added eps to prevent division by 0
    c0 = (2 * torch.sin(h) * torch.sin(h - theta0)) / (s_theta1 * s_theta2 + eps) - 1
    c1 = (2 * torch.sin(h) * torch.sin(h - theta1)) / (s_theta2 * s_theta0 + eps) - 1
    c2 = (2 * torch.sin(h) * torch.sin(h - theta2)) / (s_theta0 * s_theta1 + eps) - 1

    # added eps to prevent sqrt(0)
    s0 = torch.sqrt(torch.clamp(1 - c0**2, 0, 1)+eps)
    s1 = torch.sqrt(torch.clamp(1 - c1**2, 0, 1)+eps)
    s2 = torch.sqrt(torch.clamp(1 - c2**2, 0, 1)+eps)

    # triangle-wise weights (V, F)
    W0 = torch.zeros((V, F), dtype=src_v.dtype, device=src_v.device)
    W1 = torch.zeros_like(W0)
    W2 = torch.zeros_like(W0)

    deg_weight = s_theta0 * d1 * d2
    W0[near_pi] = deg_weight[near_pi]
    W1[near_pi] = deg_weight[near_pi]
    W2[near_pi] = deg_weight[near_pi]

    num0 = (theta0 - c1 * theta2 - c2 * theta1)
    num1 = (theta1 - c2 * theta0 - c0 * theta2)
    num2 = (theta2 - c0 * theta1 - c1 * theta0)

    denom0 = 2 * s1 * s_theta2 * d0
    denom1 = 2 * s2 * s_theta0 * d1
    denom2 = 2 * s0 * s_theta1 * d2

    # added eps to prevent division by 0
    W0[not_near_pi] = num0[not_near_pi] / (denom0[not_near_pi] + eps)
    W1[not_near_pi] = num1[not_near_pi] / (denom1[not_near_pi] + eps)
    W2[not_near_pi] = num2[not_near_pi] / (denom2[not_near_pi] + eps)

    # Final (V, Nc) weight matrix
    W_vert = torch.zeros((V, Nc), dtype=src_v.dtype, device=src_v.device)
    W_vert.index_add_(1, i0, W0)
    W_vert.index_add_(1, i1, W1)
    W_vert.index_add_(1, i2, W2)

    # Normalize to ensure partition of unity
    W_vert = W_vert / (W_vert.sum(dim=1, keepdim=True) + eps)

    return W_vert  # (V, Nc)
