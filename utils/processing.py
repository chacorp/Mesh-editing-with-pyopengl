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
    N = mesh.v.shape[0]
    
    # laplacian matrix
    L = mesh.UL.todense()
    
    # laplacian coordinates :: Delta = L @ V
    delta = mesh.UL @ mesh.v
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

def _laplacian_matrix_ring2(mesh):
    """
    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): 3N x 3N Laplacian matrix with rotation matrix in ring coordinates
    """
    
    N = mesh.v.shape[0]
    
    ## rotations
    RS = np.zeros([N, 3, 3])
    for i in range(N):
        
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        
        wij = mesh.L[i, ring].data[1:]
        # wij = mesh.L[i].data[1:]
        D_ring = np.diag(wij)
        
        # import pdb;pdb.set_trace()
        E_ring       = (mesh.v[i] - mesh.v[ring]).T # (3, N)
        E_prime_ring = (mesh.v_prime[i] - mesh.v_prime[ring]).T # (3, N)
        
        # (3, N) x (N, N) x (N, 3)
        S_i = E_ring @ D_ring @ E_prime_ring.T
        # S_i = E_ring @ E_prime_ring.T
        
        u_i, s_i, vt_i = np.linalg.svd(S_i)
        
        R_i = vt_i.T @ u_i.T
        
        if np.linalg.det(R_i) < 0:
            vt_i[-1, :] *= -1
            R_i = vt_i.T @ u_i.T
            
        RS[i] = R_i
        # RS[i] = np.eye(3)
    
        
    # laplacian matrix for rhs
    # LS = mesh.LS.copy()
    LS = np.zeros([3*N, 3*N])
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        n_ring = len(ring)
        
        
        disk_idxs = np.hstack([disk, disk+N, disk+2*N])
        ring_idxs = np.hstack([ring, ring+N, ring+2*N])
        # i_idxs = np.hstack([i, i+N, i+2*N])
        
        # wij = 0.5 * mesh.LS[[i, i+N, i+2*N]][:, ring_idxs].reshape(3, 3, n_ring).transpose(2,0,1) # (Nj, 3,3)
        wij = 0.5 * mesh.L[i, ring].data[:,None,None] # (Nj, 1,1)
        # print(wij.shape)
        
        # wij/2 * (Ri + Rj) (pi - pj)
        wij_Ri_Rj = wij * (RS[i][None] + RS[ring])
        
        wij_Ri = wij_Ri_Rj.sum(0)
        wij_Ri_Rj_all = np.vstack([-wij_Ri[None], wij_Ri_Rj])
        
        wij_Ri_Rj_all_t = wij_Ri_Rj_all.transpose(1,2,0) # (disk, 3, 3) | disk:{i, j0, j1, j2,...jn}
        
        LS[i,     disk_idxs] += wij_Ri_Rj_all_t[0].reshape(-1)
        LS[i+N,   disk_idxs] += wij_Ri_Rj_all_t[1].reshape(-1)
        LS[i+2*N, disk_idxs] += wij_Ri_Rj_all_t[2].reshape(-1)
        
    return LS

def _laplacian_matrix_ring(mesh):
    """
    Args:
        mesh (class): contains the mesh properties

    Returns:
        LS (np.array): Laplacian matrix with ring coordinates
    """
    
    N = mesh.v.shape[0]
    
    ## rotations
    RS = np.zeros([N, 3, 3])
    for i in range(N):
        
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        
        wij = mesh.L[i].data[1:]
        D_ring = np.diag(wij)
        
        import pdb;pdb.set_trace()
        E_ring       = (mesh.v[i] - mesh.v[ring]).T # (3, N)
        E_prime_ring = (mesh.v_prime[i] - mesh.v_prime[ring]).T # (3, N)
        
        # (3, N) x (N, N) x (N, 3)
        # S_i = E_ring @ D_ring @ E_prime_ring.T
        S_i = E_ring @ E_prime_ring.T
        
        u_i, s_i, vt_i = np.linalg.svd(S_i)
        
        R_i = vt_i.T @ u_i.T
        
        if np.linalg.det(R_i) < 0:
            vt_i[-1, :] *= -1
            R_i = vt_i.T @ u_i.T
            
        RS[i] = R_i
        # RS[i] = np.eye(3)
    
        
    # laplacian matrix for rhs
    # LS = mesh.LS.copy()
    LS = np.zeros([3*N, 3*N])
    for i in range(N):
        ring = np.array(mesh.ring_indices[i])
        disk = np.array([i] + mesh.ring_indices[i])
        n_ring = len(ring)
        # Ri(pi - pj)
        # LS[[i, i+N, i+2*N]][:, rind_idxs] += R_ring.repeat(n_ring, 1)
        
        # wij/2 * (Ri + Rj)(pi - pj)
        disk_idxs = np.hstack([disk, disk+N, disk+2*N])
        ring_idxs = np.hstack([ring, ring+N, ring+2*N])
        i_idxs = np.hstack([i, i+N, i+2*N])
        
        # LS[[i, i+N, i+2*N]][:, i_idxs] += RS[i] * mesh.LS[[i, i+N, i+2*N]][:, i_idxs] * 0.5
        
        # wii = -0.5 * mesh.LS[[i, i+N, i+2*N]][:, i_idxs] # multiply -1 because it is already negative! (-wii == wij)
        # wij = 0.5 *mesh.LS[[i, i+N, i+2*N]][:, ring_idxs].reshape(3, 3, n_ring).transpose(2,0,1)
        # wii = -0.5 * mesh.L[i,i]
        wij = 0.5 * mesh.L[i, ring].data[:,None,None]
        
        # wij_Rj = (mesh.L[i,ring].data[:,None,None] * RS[ring]).sum(0)
        
        # wii_Ri = wii * RS[i]
        # wij_Rj = (wij * RS[ring]).sum(0)
        
        # wij_Ri = 0.5 * (wii_Ri + wij_Rj)
        # wij_Ri = 0.5 * (wii * RS[i] + (wij * RS[ring]).sum(0))
        
        # wij_Ri_Rj = 0.5 * mesh.L[i, ring].data[:,None,None] * (RS[i][None] + RS[ring])
        wij_Ri_Rj = wij * (RS[i][None] + RS[ring])
        wij_Ri = wij_Ri_Rj.sum(0)
        wij_Ri_Rj_all = np.vstack([-wij_Ri[None], wij_Ri_Rj])
        wij_Ri_Rj_all_t = wij_Ri_Rj_all.transpose(1,2,0) # (disk, 3, 3) | disk:{i, j0, j1, j2,...jn}
        
        LS[i,     disk_idxs] += wij_Ri_Rj_all_t[0].reshape(-1)
        LS[i+N,   disk_idxs] += wij_Ri_Rj_all_t[1].reshape(-1)
        LS[i+2*N, disk_idxs] += wij_Ri_Rj_all_t[2].reshape(-1)
        
        # LS[i,     i_idxs] += wij_Ri[0]
        # LS[i+N,   i_idxs] += wij_Ri[1]
        # LS[i+2*N, i_idxs] += wij_Ri[2]
        
        # # for j in range(n_ring):
        # #     jdx = mesh.ring_indices[i][j]
        # #     j_idxs = np.hstack([jdx, jdx+N, jdx+2*N])
        # #     LS[i,     j_idxs] += wij_Ri_Rj[j,0]
        # #     LS[i+N,   j_idxs] += wij_Ri_Rj[j,1]
        # #     LS[i+2*N, j_idxs] += wij_Ri_Rj[j,2]
        # import pdb;pdb.set_trace()
        # wij_Ri_Rj_ = wij_Ri_Rj.transpose(1,2,0)
        # LS[i,     ring_idxs] -= wij_Ri_Rj_[0].reshape(-1)
        # LS[i+N,   ring_idxs] -= wij_Ri_Rj_[1].reshape(-1)
        # LS[i+2*N, ring_idxs] -= wij_Ri_Rj_[2].reshape(-1)
        
        
        
        # LS[i,     i_idxs] += (RS[i, 0]*mesh.LS[i,     i_idxs] + (RS[ring, 0] * mesh.LS[i,     ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        # LS[i+N,   i_idxs] += (RS[i, 1]*mesh.LS[i+N,   i_idxs] + (RS[ring, 1] * mesh.LS[i+N,   ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        # LS[i+2*N, i_idxs] += (RS[i, 2]*mesh.LS[i+2*N, i_idxs] + (RS[ring, 2] * mesh.LS[i+2*N, ring_idxs].reshape(n_ring,-1)).sum(0)) * 0.5
        
        # # RS[ring, 0]
        # # RS[ring].transpose(1,2,0).reshape(3, -1)
        # # LS[[i, i+N, i+2*N]][:, ring_idxs] += RS[ring].transpose(1,2,0).reshape(3, -1) * mesh.LS[[i, i+N, i+2*N]][:, ring_idxs] * 0.5
        
        # LS[i,     ring_idxs] -= (RS[i, 0] + RS[ring, 0]).transpose(1,0).reshape(-1) * mesh.LS[i,     ring_idxs] * 0.5
        # LS[i+N,   ring_idxs] -= (RS[i, 1] + RS[ring, 1]).transpose(1,0).reshape(-1) * mesh.LS[i+N,   ring_idxs] * 0.5
        # LS[i+2*N, ring_idxs] -= (RS[i, 2] + RS[ring, 2]).transpose(1,0).reshape(-1) * mesh.LS[i+2*N, ring_idxs] * 0.5
        
        # LS[i,     ring_idxs] += R_ring[0]
        # LS[i+N,   ring_idxs] += R_ring[1]
        # LS[i+2*N, ring_idxs] += R_ring[2]
    
    # LS = LS * mesh.LS * 0.5
    # LS[np.arange(N),np.arange(N)]=0
    # import pdb;pdb.set_trace()
    # L = mesh.L.todense()
    
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
    
    N  = mesh.v.shape[0]

    # uniform laplacian
    LS = mesh.L_LSE

    # ------------------- Add Constraints to the Linear System ------------------- #
    constraint_coef, constraint_b = get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb)
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
    constraint_coef, constraint_b = get_constraints(mesh, mask, handle_idx, handle_pos, Wb)
    # constraint_coef_3N, constraint_b_3N = get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb)
    
    # -------------------------- Solve the Linear System ------------------------- #
    # A = np.vstack([mesh.L.todense()])
    # b = np.vstack([mesh.delta])
    A = np.vstack([mesh.L.todense(), constraint_coef])
    b = np.vstack([mesh.delta,          constraint_b])
    # b = np.vstack([np.zeros((N,3)), constraint_b])
    
    ATA = scipy.sparse.coo_matrix(A.T @ A)
    lu = scipy.sparse.linalg.splu(ATA.tocsc())
    
    ATb = A.T @ b
    v_prime = lu.solve(ATb)
    v_prime = np.asarray(v_prime)
    
    mesh.v_prime = v_prime
    
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
        mesh.v_prime = v_prime.copy()
                
    return mesh.v_prime

def get_constraints(mesh, mask, handle_idx, handle_pos, Wb=1.0):
    
    N  = mesh.v.shape[0]
    
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
            constraint_b[i]         = mesh.v[vidx] * Wb # fixed position
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

def get_constraints_3N(mesh, mask, handle_idx, handle_pos, Wb=1.0):
    N  = mesh.v.shape[0]
    
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
            constraint_b[idx_c]           = mesh.v[vidx] * Wb # fixed position
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