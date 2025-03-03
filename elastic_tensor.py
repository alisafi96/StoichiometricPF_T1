import cupy as cp
import numpy as np
import time
from cupyx.scipy.sparse import csr_matrix

"2D functions"
def full_2x2_to_voigt_3_index(i, j):
    if i == j:
        # i == j => 0->0, 1->1
        return i
    else:
        # off-diagonal (0,1) or (1,0) => 2
        return 2

def voigt_3x3_to_full_2x2x2x2(C_voigt):
    # Initialize the 2x2x2x2 array (zeros)
    C_out = np.zeros((2, 2, 2, 2))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    vi = full_2x2_to_voigt_3_index(i, j)
                    vj = full_2x2_to_voigt_3_index(k, l)
                    C_out[i][j][k][l] = C_voigt[vi][vj]    
    return C_out

def symmetrize_C_2d(c11_hcp, c12_hcp, c44_hcp,
                    c11_fcc, c12_fcc, c44_fcc,
                    flag):
    C_hcp_2d =  np.array([
        [c11_hcp, c12_hcp,   0.0    ],  
        [c12_hcp, c11_hcp,   0.0    ],  
        [   0.0  ,   0.0  , c44_hcp ]   
    ])

    C_fcc_2d =  np.array([
        [c11_fcc, c12_fcc,   0.0    ],
        [c12_fcc, c11_fcc,   0.0    ],
        [   0.0  ,   0.0  , c44_fcc ]
    ])
    
    C_eff_voigt = np.zeros((3, 3))
    dC_voigt    = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            if flag == 'hetero':    # Heterogeneous
                C_eff_voigt[i][j] = 0.5*(C_hcp_2d[i][j] + C_fcc_2d[i][j])
                dC_voigt[i][j]    =      C_hcp_2d[i][j] - C_fcc_2d[i][j]
            elif flag == 'homo_T1': # Homogeneous T1
                C_eff_voigt[i][j] = C_hcp_2d[i][j]
                dC_voigt[i][j]    = 0.0
            elif flag == 'homo_al': # Homogeneous Al
                C_eff_voigt[i][j] = C_fcc_2d[i][j]
                dC_voigt[i][j]    = 0.0

    C_eff = voigt_3x3_to_full_2x2x2x2(C_eff_voigt)
    dC    = voigt_3x3_to_full_2x2x2x2(dC_voigt)
    return C_eff, dC

def full_3x3_to_voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j


def voigt_6x6_to_full_3x3x3x3(C_voigt):
    # Initialize 3x3x3x3 with zeros
    C_out = np.zeros((3, 3, 3, 3))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    vi = full_3x3_to_voigt_6_index(i, j)
                    vj = full_3x3_to_voigt_6_index(k, l)
                    C_out[i][j][k][l] = C_voigt[vi][vj]
    return C_out

def symmetrize_C_3d(cp11, cp12, cp13, cp33, cp44, cm11, cm12, cm44, flag):
    # Construct C_hcp (6x6)
    C_hcp = np.array([
        [cp11, cp12, cp13,   0.0,   0.0,               0.0             ],
        [cp12, cp11, cp13,   0.0,   0.0,               0.0             ],
        [cp13, cp13, cp33,   0.0,   0.0,               0.0             ],
        [ 0.0,  0.0,  0.0,  cp44,   0.0,               0.0             ],
        [ 0.0,  0.0,  0.0,   0.0,  cp44,               0.0             ],
        [ 0.0,  0.0,  0.0,   0.0,   0.0, (cp11 - cp12)/2.0 ]
    ])

    # Construct C_fcc (6x6)
    C_fcc = np.array([
        [cm11, cm12, cm12,  0.0,  0.0,  0.0],
        [cm12, cm11, cm12,  0.0,  0.0,  0.0],
        [cm12, cm12, cm11,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, cm44,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0, cm44,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0, cm44]
    ])

    # Create empty 6x6 for C_eff_V and dC_V
    C_eff_V = np.zeros((6, 6))
    dC_V    =np.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            if flag == 'hetero':    # -- Heterogeneous --
                C_eff_V[i][j] = (C_hcp[i][j] + C_fcc[i][j]) / 2.0
                dC_V[i][j]    =  C_hcp[i][j] - C_fcc[i][j]
            if flag == 'homo_T1':   # -- Homogeneous T1  --
                C_eff_V[i][j] = C_hcp[i][j]
                dC_V[i][j]    = 0.0
            if flag == 'homo_al':   # -- Homogeneous Al --
                C_eff_V[i][j] = C_fcc[i][j]
                dC_V[i][j]    = 0.0

    # Convert 6x6 to 3x3x3x3
    C_eff = voigt_6x6_to_full_3x3x3x3(C_eff_V)
    dC    = voigt_6x6_to_full_3x3x3x3(dC_V)

    return C_eff, dC

def compute_green_2d(C, k):
    K = cp.zeros((2, 2) + k[0].shape)
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                for ll in range(2):
                    K[ii][kk] += C[ii][jj][kk][ll] * k[jj] * k[ll]

    K_inv = cp.empty_like(K)
    dets = K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0]

    dets = cp.where(cp.abs(dets) == 0, 1.0, dets)
    print(cp.max(dets))
    print(cp.min(dets))
    
    K_inv[0, 0] = (1/dets) * K[1, 1]
    K_inv[0, 1] = -(1/dets) * K[0, 1]
    K_inv[1, 0] = -(1/dets) * K[1, 0]
    K_inv[1, 1] = (1/dets) * K[0, 0]
    Omega = cp.zeros((2, 2, 2, 2) + k[0].shape)                       
    for kk in range(2):
        for ll in range(2):
            for ii in range(2):
                for jj in range(2):
                    Omega[kk, ll, ii, jj] = 0.25*(K_inv[ll, ii] * k[jj] * k[kk]
                                                    + K_inv[kk, ii] * k[jj] * k[ll]
                                                    + K_inv[ll, jj] * k[ii] * k[kk]
                                                    + K_inv[kk, jj] * k[ii] * k[ll])
    return Omega

def compute_green_3d(C, k):
    k_= k.copy()
    K = cp.zeros((3, 3) + k_[0].shape)
    
    for i in range(3):
        for j in range(3):
            for p in range(3):
                for q in range(3):
                    K[i][j] += C[i][p][q][j] * k_[p] * k_[q]

    _, _, Nx, Ny, Nz,= K.shape  # (3,3, Ny, Nz, Nz)

    eps = 1e-14
    K_inv = cp.empty_like(K)
    dets =  (K[0, 0, :] * K[1, 1, :] * K[2, 2, :] 
            + K[0, 1, :] * K[1, 2, :] * K[2, 0, :] 
            + K[0, 2, :] * K[1, 0, :] * K[2, 1, :] 
            - K[0, 2, :] * K[1, 1, :] * K[2, 0, :] 
            - K[0, 1, :] * K[1, 0, :] * K[2, 2, :] 
            - K[0, 0, :] * K[1, 2, :] * K[2, 1, :])

    dets = cp.where(cp.abs(dets) < eps, 1.0, dets)
    print(dets.shape)
    print(cp.max(dets))
    print(cp.min(dets))
    K_inv[0, 0, :] = (1/dets) * (K[1, 1, :]*K[2, 2, :] - K[1, 2, :]*K[2, 1, :])
    K_inv[0, 1, :] = (1/dets) * (K[0, 2, :]*K[2, 1, :] - K[0, 1, :]*K[2, 2, :])
    K_inv[0, 2, :] = (1/dets) * (K[0, 1, :]*K[1, 2, :] - K[0, 2, :]*K[1, 1, :])
    
    K_inv[1, 0, :] = (1/dets) * (K[1, 2, :]*K[2, 0, :] - K[1, 0, :]*K[2, 2, :])
    K_inv[1, 1, :] = (1/dets) * (K[0, 0, :]*K[2, 2, :] - K[0, 2, :]*K[2, 0, :])
    K_inv[1, 2, :] = (1/dets) * (K[0, 2, :]*K[1, 0, :] - K[0, 0, :]*K[1, 2, :])
    
    K_inv[2, 0, :] = (1/dets) * (K[1, 0, :]*K[2, 1, :] - K[1, 1, :]*K[2, 0, :])
    K_inv[2, 1, :] = (1/dets) * (K[0, 1, :]*K[2, 0, :] - K[0, 0, :]*K[2, 1, :])
    K_inv[2, 2, :] = (1/dets) * (K[0, 0, :]*K[1, 1, :] - K[0, 1, :]*K[1, 0, :])

    Omega = 0.25 * (cp.einsum('hixyz,jxyz,kxyz->khijxyz', K_inv, k_, k_)
                    + cp.einsum('kixyz,jxyz,hxyz->khijxyz', K_inv, k_, k_)
                    + cp.einsum('hjxyz,ixyz,kxyz->khijxyz', K_inv, k_, k_)
                    + cp.einsum('kjxyz,ixyz,hxyz->khijxyz', K_inv, k_, k_))
    return Omega

def solve_elasticity(omega, C_eff, dC, sfts, strain, stress, H, shape, alpha, tol, max_iter):
    norm = 0.0
    norm_old = 0.0
    dim = len(shape)
    if dim == 2:
        conv = 'ijklxy,klxy->ijxy'
        axis = (2, 3)
    elif dim == 3:
        conv = 'ijklxyz,klxyz->ijxyz'
        axis = (2, 3, 4)
    else:
        raise ValueError('Only 2D and 3D elasticity is supported')
    
    e_strain = cp.zeros_like(strain)
    ea = cp.zeros_like(strain)
    
    # Parameters for adapting alpha
    alpha_min = 1e-6
    alpha_max = 1.0
    alpha_increase_factor = 1.1
    alpha_decrease_factor = 0.7
    max_norm_diff = 1e3
    conver = 0.0
    no_convergence_counter = 0  # Counter to track consecutive non-convergence iterations

    for it in range(max_iter):
        strain = cp.fft.fftn(strain, axes=axis)
        stress = cp.fft.fftn(stress, axes=axis)
        
        strain -= alpha * cp.einsum(conv, omega, stress)
        
        strain = cp.real(cp.fft.ifftn(strain, axes=axis))
        stress = cp.real(cp.fft.ifftn(stress, axes=axis))
        
        e_strain = strain - sfts + ea

        stress = cp.zeros_like(stress)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        stress[i][j] += (C_eff[i][j][k][l] + dC[i][j][k][l] * (H - 0.5)) * e_strain[k][l]
        
        norm_old = norm
        norm = cp.sum(stress**2)
        
        # Check convergence
        if it > 0:
            conver = cp.abs(norm - norm_old) / (norm_old + 1e-12)
            if conver < tol:
                print(f'Converged in {it} iterations')
                break
            
            # Increase alpha if steady convergence
            if conver < 0.1 and alpha != alpha_max :  # Steady convergence
                alpha = min(alpha * alpha_increase_factor, alpha_max)
                print(f'Increasing alpha to {alpha} for faster convergence')

            if conver > 0.5 and it >= 10:  
                alpha = max(alpha * alpha_decrease_factor, alpha_min)
                print(f'Reducing alpha to {alpha} due to lack of convergence in 10 iterations')

        print(f'Iteration {it}, Current norm: {conver:.2e}, Alpha: {alpha:.2e}')
        
    else:
        print('Solver did not converge within the maximum iterations')
    
    return strain, stress, alpha


