import numpy as np
import torch


# ------------------------ Laplacian Matrices ------------------------ #
# This file contains implementations of differentiable laplacian matrices.
# These include
# 1) Standard Laplacian matrix
# 2) Cotangent Laplacian matrix
# 3) Norm Laplacian matrix
# -------------------------------------------------------------------- #

def adjacency_matrix(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    return A

def laplacian_matrix(verts: torch.Tensor, edges: torch.Tensor):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

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


def normalize_v(V, mode='np'):
    if mode == 'np':
        V = np.array(V)
        V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    else: # torch
        V = (V-(V.max(0).values + V.min(0).values) * 0.5)/max(V.max(0).values - V.min(0).values)
    return V

def compute_face_norm(v, f):
    vf = v[f]
    e1 = vf[..., 0] - vf[..., 1]
    e2 = vf[..., 1] - vf[..., 2]
    norm = (np.cross(e1, e2))
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-12)
    return norm

def compute_vert_norm(faces, face_normals, num_vertices=None):
    r"""Computes normals for every vertex by averaging face normals
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

def computeTangentBasis(vertex, uv):
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
    # import pdb;pdb.set_trace()
    return tangents