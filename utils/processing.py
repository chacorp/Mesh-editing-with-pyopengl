# import torch
import numpy as np
import scipy


def laplacian_matrix_ring_LSE(mesh):
    """
    Reference: 
        - Laplacian Surface Editing, Olga Sorkine, Daniel Cohen-Or, 2004
        - https://github.com/luost26/laplacian-surface-editing/blob/master/main.py

    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    V = mesh.v
    N = V.shape[0]
    
    # laplacian matrix
    L = mesh.UL.todense()
    
    # laplacian coordinates :: Delta = L @ V
    delta = mesh.UL @ V
    # delta = np.concatenate([mesh.delta, np.ones([N, 1])], axis=-1)
    # delta = mesh.L @ np.concatenate([mesh.v, np.ones([N, 1])], axis=-1)
    
    # print(L.shape)
    # print(3*N)
    LS = np.zeros([3*N, 3*N])
    LS[0*N:1*N, 0*N:1*N] = -1*L
    LS[1*N:2*N, 1*N:2*N] = -1*L
    LS[2*N:3*N, 2*N:3*N] = -1*L
    
    # compute the l
    for i in range(N):
        ring        = np.array([i] + mesh.ring_indices[i])
        n_ring      = len(ring)
        V_ring      = V[ring]
        
        Ai          = np.zeros([n_ring * 3, 7])
        zer0_ring   = np.zeros(n_ring)
        ones_ring   = np.ones(n_ring)
                
        Ai[:n_ring,        ] = np.stack([V_ring[:,0],    zer0_ring,  V_ring[:,2], -V_ring[:,1], ones_ring, zer0_ring, zer0_ring], axis=1)
        Ai[n_ring:2*n_ring,] = np.stack([V_ring[:,1], -V_ring[:,2],    zer0_ring,  V_ring[:,0], zer0_ring, ones_ring, zer0_ring], axis=1)
        Ai[2*n_ring:,      ] = np.stack([V_ring[:,2],  V_ring[:,1], -V_ring[:,0],    zer0_ring, zer0_ring, zer0_ring, ones_ring], axis=1)
                
        # Moore-Penrose Inversion
        AiTAi_pinv = np.linalg.pinv(Ai.T @ Ai)
        Ai_pinv = AiTAi_pinv @ Ai.T
        
        si = Ai_pinv[0]
        hi = Ai_pinv[1:4]
        # ti = Ai_pinv[4:7]
        
        Ti = np.array([
            # [     si,  -hi[2],  hi[1], zer0_3],
            # [  hi[2],      si, -hi[0], zer0_3],
            # [ -hi[1],   hi[0],     si, zer0_3],
            
            # [    si,  -hi[2],  hi[1], ones_3],
            # [  hi[2],     si,  -hi[0], ones_3],
            # [ -hi[1],  hi[0],      si, ones_3],
            
            [     si, -hi[2],  hi[1],],
            [  hi[2],     si, -hi[0],],
            [ -hi[1],  hi[0],     si,],
            
            # [     si, -hi[2],  hi[1],  ti[0],],
            # [  hi[2],     si, -hi[0],  ti[1],],
            # [ -hi[1],  hi[0],     si,  ti[2],],
            
            # [     si,  hi[2], -hi[1],],
            # [ -hi[2],     si,  hi[0],],
            # [  hi[1], -hi[0],     si,],
            # [ones_3,ones_3,ones_3,],
            
            # [     si,  hi[2], -hi[1],],
            # [ -hi[2],     si,  hi[0],],
            # [  hi[1], -hi[0],     si,],
            # [  ti[0],  ti[1],  ti[2],],
        ])
        
        # T_delta = (Ti @ delta[i])
        T_delta =(Ti.transpose(2,0,1) @ delta[i].T).transpose(1,0)
        # T_delta =(Ti.transpose(2,1,0) @ delta[i].T).transpose(1,0)
        # T_delta = (delta[i].T @ Ti ).squeeze()
        
        disk_idxs = np.hstack([ring, ring+N, ring+2*N])
        LS[i,     disk_idxs] += T_delta[0]
        LS[i+N,   disk_idxs] += T_delta[1]
        LS[i+2*N, disk_idxs] += T_delta[2]
    
    return LS

def get_rotation(mesh):
    N = mesh.v.shape[0]
    RS = np.zeros([N, 3, 3])
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        wij = np.array([mesh.L[i, j] for j in ring])
        D_ring = np.diag(wij)

        E_ring       = (mesh.v[i] - mesh.v[ring]).T # (3, Nj)
        E_prime_ring = (mesh.v_prime[i] - mesh.v_prime[ring]).T  # (3, Nj)

        # (3, N) x (N, N) x (N, 3)
        S_i = E_ring @ D_ring @ E_prime_ring.T
        S_i /= abs(S_i).max()
        u_i, _, vt_i = np.linalg.svd(S_i)
        
        R_i = vt_i.T @ u_i.T

        if np.linalg.det(R_i) < 0:
            vt_i[-1, :] *= -1
            R_i = vt_i.T @ u_i.T

        RS[i] = R_i
        # RS[i] = np.eye(3)
    return RS

def laplacian_matrix_ring_ARAP(mesh):
    """
    Args:
        mesh (class): contains mesh properties
            vertices, vertices prime, L (cotangent Laplacian) and mesh.ring_indices ...

    Returns:
        LS (np.array): 3N x 3N Laplacian matrix with ARAP rotation terms
    """
    N = mesh.v.shape[0]
    RS = get_rotation(mesh)

    LS = np.zeros((3*N, 3*N))
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        n_ring = len(ring)

        disk_idxs = np.hstack([disk, disk + N, disk + 2*N])
        wij = np.array([mesh.L[i, j] for j in ring])[:, None, None]  # (Nj, 1, 1)

        Ri = RS[i][None]
        Rj = RS[ring]  # (Nj, 3, 3)
        wij_Ri_Rj = 0.5 * wij * (Ri + Rj)  # (Nj, 3, 3)

        wij_Ri = wij_Ri_Rj.sum(axis=0)  # (3, 3)
        all_blocks = np.vstack([-wij_Ri[None], wij_Ri_Rj])  # (Nj+1, 3, 3)
        
        # (disk, 3, 3) --> (3, 3, disk) | disk:{i, j0, j1, j2,...jn}
        all_blocks = all_blocks.transpose(1, 2, 0).reshape(3, 3*(n_ring+1)) # (3, 3*(Nj+1))

        LS[i,     disk_idxs] += all_blocks[0]
        LS[i+N,   disk_idxs] += all_blocks[1]
        LS[i+2*N, disk_idxs] += all_blocks[2]

    return LS

def laplacian_surface_editing(mesh, mask, boundary_idx, handle_idx, handle_pos, Wb=1.0):
    """
    Args:
        mesh (class):        mesh properties
        boundary_idx (list):    vertex indices of boundary
        handle_idx (list):      vertex indices of handle
        handle_pos (np.array):  displacement of handle
        mask (np.array):  vertices outside boundary (V, 1)

    Returns:
        new_verts (np.array):   new vertices after laplacian surface editing
    """
    V = mesh.orig_v
    N  = V.shape[0]

    # uniform laplacian
    LS = mesh.L_LSE

    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef, constraint_b = get_constraints_3N(V, mask, handle_idx, handle_pos, Wb)
    # -------------------------- Solve the Linear System ------------------------- #
    A        = np.vstack([LS,         constraint_coef])
    b        = np.hstack([np.zeros(3*N), constraint_b])
    # A        = LS
    # b        = np.zeros(3*N)
    spA      = scipy.sparse.coo_matrix(A)
        
    V_prime  = scipy.sparse.linalg.lsqr(spA, b)[0]
    
    new_verts = V_prime.reshape(3, -1).T
    return new_verts

def as_rigid_as_possible_surface_modeling(mesh, mask, boundary_idx, handle_idx, handle_pos, Wb=1.0, iteration=2):
    """
    Args:
        mesh (class):        mesh properties
        boundary_idx (list):    vertex indices of boundary
        handle_idx (list):      vertex indices of handle
        handle_pos (np.array):  displacement of handle
        mask (np.array):  vertices outside boundary (V, 1)

    Returns:
        new_verts (np.array):   new vertices after laplacian surface editing
    """
    mesh.v = mesh.orig_v
    N  = mesh.v.shape[0]
    
    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef, constraint_b = get_constraints(mesh.v, mask, handle_idx, handle_pos, Wb)
    # constraint_coef_3N, constraint_b_3N = get_constraints_3N(mesh.v, mask, handle_idx, handle_pos, Wb)
    
    # -------------------------- Solve the Linear System ------------------------- #
    # A = np.vstack([mesh.L.todense()])
    # b = np.vstack([mesh.delta])
    
    A = np.vstack([mesh.L.todense(), constraint_coef])
    b = np.vstack([mesh.delta,          constraint_b])
    # b = np.vstack([np.zeros((N,3)), constraint_b])
    
    ATA = scipy.sparse.coo_matrix(A.T @ A)
    lu = scipy.sparse.linalg.splu(ATA.tocsc())
    
    # ATb = A.T @ b
    # v_prime = lu.solve(ATb)
    # v_prime = np.asarray(v_prime)
    
    # mesh.v_prime = v_prime
    mesh.v_prime = mesh.v
    
    for it in range(iteration):
        # rhs = mesh.L @ mesh.v
        # import pdb;pdb.set_trace()
        
        # rhs = mesh.LS @ mesh.v.transpose(1,0).reshape(-1)
        rhs = laplacian_matrix_ring_ARAP(mesh) @ mesh.v.transpose(1, 0).reshape(-1)
        rhs = rhs.reshape(3,-1).transpose(1, 0)
        # b = np.vstack([rhs])
        b = np.vstack([rhs, constraint_b])
        
        ATb = A.T @ b
        v_prime = lu.solve(ATb)
                
        v_prime = np.asarray(v_prime)
        mesh.v_prime = v_prime
        # mesh.v = mesh.v_prime
    
    return mesh.v_prime
    # return v_prime

def get_constraints(vertices, mask, handle_idx, handle_pos, Wb=1.0):
    
    N  = vertices.shape[0]
    
    len_H = len(handle_idx)
    constraint_coef = np.zeros([N + len_H, N])
    constraint_b    = np.zeros([N + len_H, 3])
    
    # Boundary constraints
    i = 0
    h = 0
    
    # Boundary constraints
    for vidx, val in enumerate(mask): # num: N
        # i: index of Boundary constraint
        # idx: index of v in V
                
        if val == 0:
            constraint_coef[i, vidx] = 1.0 * Wb # one-hot
            constraint_b[i]         = vertices[vidx] * Wb # fixed position
        i = i + 1
        
    # Handle constraints
    for vidx in handle_idx: # num: len_H
        # i: index of Handle constraint
        # idx: index of v in V
        
        constraint_coef[i, vidx] = 1.0 * Wb # one-hot
        constraint_b[i]         = handle_pos[h] * Wb
        
        i = i + 1
        h = h + 1
        
    return constraint_coef, constraint_b

def get_constraints_3N(vertices, mask, handle_idx, handle_pos, Wb=1.0):
    N  = vertices.shape[0]
    
    len_H = len(handle_idx)
    constraint_coef = np.zeros([3*N + 3*len_H, 3*N])
    constraint_b    = np.zeros([3*N + 3*len_H])
    
    i = 0
    h = 0
    
    # Boundary constraints + outside ROI
    for vidx, val in enumerate(mask): # num: N
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Boundary constraint
        idx_v = [vidx, vidx+N, vidx+2*N] # index of v in V
        
        if val == 0:
            constraint_coef[idx_c, idx_v] = 1.0 * Wb # one-hot
            constraint_b[idx_c]           = vertices[vidx] * Wb # fixed position
        i = i + 1
        
    # Handle constraints
    for vidx in handle_idx: # num: len_H
        idx_c = [i*3+0, i*3+1, i*3+2] # index of Handle constraint
        idx_v = [vidx, vidx+N, vidx+2*N] # index of v in V
        
        constraint_coef[idx_c, idx_v] = 1.0 * Wb # one-hot
        constraint_b[idx_c]           = handle_pos[h] * Wb
        
        i = i + 1
        h = h + 1
    return constraint_coef, constraint_b

def compute_MVC(src_v, cage_v, cage_f, eps=1e-8):
    """
    MVC computation
    Reference from original paper

    Args:
        src_v (np.ndarray): (V, 3) numpy array of mesh vertices
        cage_v (np.ndarray): (Nc, 3) numpy array of cage vertices
        cage_f (np.ndarray): (F, 3) numpy array of triangle indices into cage_v

    Returns:
        W_vert (np.ndarray): (V, F, 3) numpy array of per triangle cage weights for each vertex
    """
    V = src_v.shape[0]
    F = cage_f.shape[0]
    
    # cage vertices per face
    i0 = cage_f[:, 0]
    i1 = cage_f[:, 1]
    i2 = cage_f[:, 2]

    P0 = cage_v[i0]  # (F, 3)
    P1 = cage_v[i1]
    P2 = cage_v[i2]

    # Expand to (V, F, 3)
    X = src_v[:, None, :]  # (V, 1, 3)
    D0 = P0[None, :, :] - X  # (V, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    d0 = np.linalg.norm(D0, axis=-1) + eps
    d1 = np.linalg.norm(D1, axis=-1) + eps
    d2 = np.linalg.norm(D2, axis=-1) + eps

    U0 = D0 / d0[..., None]
    U1 = D1 / d1[..., None]
    U2 = D2 / d2[..., None]

    # angles
    L0 = np.linalg.norm(U1 - U2, axis=-1)
    L1 = np.linalg.norm(U2 - U0, axis=-1)
    L2 = np.linalg.norm(U0 - U1, axis=-1)

    theta0 = 2 * np.arcsin(np.clip(L0 * 0.5, -0.999999, 0.999999))
    theta1 = 2 * np.arcsin(np.clip(L1 * 0.5, -0.999999, 0.999999))
    theta2 = 2 * np.arcsin(np.clip(L2 * 0.5, -0.999999, 0.999999))

    h = 0.5 * (theta0 + theta1 + theta2)

    near_pi = np.abs(np.pi - h) < eps
    not_near_pi = ~near_pi

    s_theta0 = np.sin(theta0)
    s_theta1 = np.sin(theta1)
    s_theta2 = np.sin(theta2)

    c0 = (2 * np.sin(h) * np.sin(h - theta0)) / (s_theta1 * s_theta2 + eps) - 1
    c1 = (2 * np.sin(h) * np.sin(h - theta1)) / (s_theta2 * s_theta0 + eps) - 1
    c2 = (2 * np.sin(h) * np.sin(h - theta2)) / (s_theta0 * s_theta1 + eps) - 1

    s0 = np.abs(np.sqrt(np.clip(1 - c0**2, 0, 1)+ eps))
    s1 = np.abs(np.sqrt(np.clip(1 - c1**2, 0, 1)+ eps))
    s2 = np.abs(np.sqrt(np.clip(1 - c2**2, 0, 1)+ eps))

    W0 = np.zeros((V, F))
    W1 = np.zeros((V, F))
    W2 = np.zeros((V, F))

    # near-pi: degenerate case
    sin_theta0 = np.sin(theta0)
    deg_weight = sin_theta0 * d1 * d2

    W0[near_pi] = deg_weight[near_pi]
    W1[near_pi] = deg_weight[near_pi]
    W2[near_pi] = deg_weight[near_pi]

    # general case
    num0 = (theta0 - c1 * theta2 - c2 * theta1)
    num1 = (theta1 - c2 * theta0 - c0 * theta2)
    num2 = (theta2 - c0 * theta1 - c1 * theta0)

    denom0 = 2 * s1 * s_theta2 * d0
    denom1 = 2 * s2 * s_theta0 * d1
    denom2 = 2 * s0 * s_theta1 * d2

    W0[not_near_pi] = num0[not_near_pi] / (denom0[not_near_pi] + eps)
    W1[not_near_pi] = num1[not_near_pi] / (denom1[not_near_pi] + eps)
    W2[not_near_pi] = num2[not_near_pi] / (denom2[not_near_pi] + eps)

    # stack to (V, F, 3)
    W_vert = np.stack([W0, W1, W2], axis=-1)
    W_vert = W_vert / (W_vert.sum(dim=1, keepdim=True) + eps)
    return W_vert

def apply_MVC_weight(W, cage_f, cage_function, eps=1e-8):
    """
    Args:
        W (np.ndarray): (V, F, 3) array of MVC weights
        cage_f (np.ndarray): (F, 3) array of face indices
        cage_function (np.ndarray): (Nc, 3) array of cage vertex attributes (e.g., position, displacement)
    
    Returns:
        new_V: (V, 3) array of interpolated vertex values
    """
    V, F, _ = W.shape

    # Get cage function values per face vertex (F, 3, 3)
    tri_values = cage_function[cage_f]  # (F, 3, 3)

    # Broadcast to (V, F, 3, 3)
    tri_values = np.broadcast_to(tri_values[None, :, :, :], (V, F, 3, 3))  # (V, F, 3, 3)
    W_exp = W[..., None]  # (V, F, 3, 1)

    weighted_sum = (W_exp * tri_values).sum(axis=2)  # (V, F, 3)
    total = weighted_sum.sum(axis=1)  # (V, 3)

    weight_sum = W.sum(axis=(1, 2), keepdims=True)  # (V, 1, 1)
    weight_sum = np.clip(weight_sum, eps, np.inf)

    result = total / weight_sum.squeeze(-1)  # (V, 3)
    return result

def compute_MVC_vertexwise(src_v, cage_v, cage_f, eps=1e-8):
    """
    MVC computation vertex-wise for simple linear reproduction
    
    new_vertex_function = MVC @ cage_function

    Args:
        src_v (np.ndarray): (V, 3) numpy array of mesh vertices
        cage_v (np.ndarray): (Nc, 3) numpy array of cage vertices
        cage_f (np.ndarray): (F, 3) numpy array of triangle indices into cage_v

    Returns:
        W_vert (np.ndarray): (V, Nc) numpy array of cage weights per vertex
    """
    V = src_v.shape[0]
    Nc = cage_v.shape[0]
    F = cage_f.shape[0]

    i0, i1, i2 = cage_f[:, 0], cage_f[:, 1], cage_f[:, 2]
    P0 = cage_v[i0]  # (F, 3)
    P1 = cage_v[i1]
    P2 = cage_v[i2]

    X = src_v[:, None, :]  # (V, 1, 3)

    D0 = P0[None, :, :] - X  # (V, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    d0 = np.linalg.norm(D0, axis=-1).clip(min=eps)
    d1 = np.linalg.norm(D1, axis=-1).clip(min=eps)
    d2 = np.linalg.norm(D2, axis=-1).clip(min=eps)

    U0 = D0 / d0[..., None]
    U1 = D1 / d1[..., None]
    U2 = D2 / d2[..., None]

    L0 = np.linalg.norm((U1 - U2), axis=-1)
    L1 = np.linalg.norm((U2 - U0), axis=-1)
    L2 = np.linalg.norm((U0 - U1), axis=-1)

    clamp_arcsin = lambda x: np.clip(x, -0.999999, 0.999999)
    
    theta0 = 2 * np.arcsin(clamp_arcsin(L0 * 0.5))
    theta1 = 2 * np.arcsin(clamp_arcsin(L1 * 0.5))
    theta2 = 2 * np.arcsin(clamp_arcsin(L2 * 0.5))
    h = 0.5 * (theta0 + theta1 + theta2)

    near_pi = np.abs(np.pi - h) < eps
    not_near_pi = ~near_pi

    s_theta0 = np.sin(theta0)
    s_theta1 = np.sin(theta1)
    s_theta2 = np.sin(theta2)

    c0 = (2 * np.sin(h) * np.sin(h - theta0)) / (s_theta1 * s_theta2 + eps) - 1
    c1 = (2 * np.sin(h) * np.sin(h - theta1)) / (s_theta2 * s_theta0 + eps) - 1
    c2 = (2 * np.sin(h) * np.sin(h - theta2)) / (s_theta0 * s_theta1 + eps) - 1

    def safe_sqrt(x):
        return np.sqrt(np.clip(1 - x**2, 0, 1.0)+eps)
    
    s0 = safe_sqrt(c0)
    s1 = safe_sqrt(c1)
    s2 = safe_sqrt(c2)

    W0 = np.zeros((V, F), dtype=np.float64)
    W1 = np.zeros_like(W0)
    W2 = np.zeros_like(W0)

    deg_weight = s_theta0 * d1 * d2
    W0[near_pi] = deg_weight[near_pi]
    W1[near_pi] = deg_weight[near_pi]
    W2[near_pi] = deg_weight[near_pi]

    num0 = (theta0 - c1 * theta2 - c2 * theta1)
    num1 = (theta1 - c2 * theta0 - c0 * theta2)
    num2 = (theta2 - c0 * theta1 - c1 * theta0)

    denom0 = 2 * s1 * s_theta2 * d0 + eps
    denom1 = 2 * s2 * s_theta0 * d1 + eps
    denom2 = 2 * s0 * s_theta1 * d2 + eps

    W0[not_near_pi] = num0[not_near_pi] / denom0[not_near_pi]
    W1[not_near_pi] = num1[not_near_pi] / denom1[not_near_pi]
    W2[not_near_pi] = num2[not_near_pi] / denom2[not_near_pi]

    # Convert to (V, Nc)
    W_vert = np.zeros((V, Nc), dtype=np.float64)
    for j, W in zip([i0, i1, i2], [W0, W1, W2]):
        np.add.at(W_vert, (slice(None), j), W)

    # Normalize to ensure partition of unity
    W_vert = W_vert / (np.sum(W_vert, axis=1, keepdims=True) + eps)

    return W_vert  # (V, Nc)
