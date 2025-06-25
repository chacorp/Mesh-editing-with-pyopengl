import numpy as np
import scipy
import scipy.sparse

def adjacency_matrix(verts_N, edges, return_idx=False):
    """
    Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts_N: number of vertices (N) of the mesh (N, 3)
        edges:   tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        A: Adjacency matrix (V, V)
    """
    V = verts_N
    E = edges
    
    idx01 = np.stack([E[:,0], E[:,1]], axis=1)  # (E, 2)
    idx10 = np.stack([E[:,1], E[:,0]], axis=1)  # (E, 2)
    idx = np.r_[idx01, idx10].T  # (2, 2*E)
    
    ones = np.ones(idx.shape[1])
    A  = scipy.sparse.csr_matrix((ones, idx), shape=(V, V))

    if return_idx:
        return A, idx
    return A

def laplacian_and_adjacency(verts_N, edges):
    """
    Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts_N: number of vertices (N) of the mesh (N, 3)
        edges:   tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Uniform Laplacian matrix (V, V)
        A: Adjacency matrix (V, V)
    """
    
    A = adjacency_matrix(verts_N, edges, return_idx=False)
    
    degree = np.asarray(A.sum(axis=1)).squeeze()
    I_ = scipy.sparse.identity(verts_N)
    D_ = scipy.sparse.diags(1/degree) @ A
    L = I_ - D_
    
    return L, A

def laplacian_cotangent(verts, faces, use_normed=True, eps=1e-12):
    """
    Returns
    -------
    L : scipy.sparse.csr_matrix
        (V,V) cotangent Laplacian (normalized or unnormalized).
    inv_areas : np.ndarray, shape (V,)
        1 / (sum of adjacent face areas) for each vertex.
    """
    V = verts.shape[0]
    F = faces.shape[0]
    
    fv = verts[faces]  # (F,3,3)
    v0, v1, v2 = fv[:,0], fv[:,1], fv[:,2]
    A = np.linalg.norm(v1 - v2, axis=1)
    B = np.linalg.norm(v0 - v2, axis=1)
    C = np.linalg.norm(v0 - v1, axis=1)
    
    s = 0.5 * (A + B + C)
    area = np.sqrt(np.clip(s*(s-A)*(s-B)*(s-C), eps, np.inf))  # (F,)
    
    A2, B2, C2 = A*A, B*B, C*C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = np.stack([cota, cotb, cotc], axis=1) / 4.0  # (F,3)
    
    ii = faces[:, [1,2,0]].reshape(-1)
    jj = faces[:, [2,0,1]].reshape(-1)
    
    W = scipy.sparse.csr_matrix((cot.reshape(-1), (ii, jj)), shape=(V, V))
    W = (W + W.T).tocsr()
    
    deg = np.array(W.sum(axis=1)).flatten()  # (V,)
    
    if use_normed:
        d_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        D_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
        L = scipy.sparse.eye(V, format='csr') - (D_inv_sqrt @ W @ D_inv_sqrt)
    else:
        L = scipy.sparse.diags(deg) - W
        
    # inv_areas = np.zeros(V, dtype=np.float64)
    # idx = faces.ravel()
    # np.add.at(inv_areas, idx, np.repeat(area, 3))
    # mask = inv_areas > 0
    # inv_areas[mask] = 1.0 / inv_areas[mask]

    # return L.tocsr(), inv_areas
    return L.tocsr()