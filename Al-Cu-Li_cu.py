import numpy as np
import math
import cupy as cp
import matplotlib.pyplot as plt
from time import time
import gpustat
import os
from IC import *
from CALPHAD import *
from visualization import *
from prepare_fft import *
from elastic_tensor import *
from sklearn.decomposition import PCA
import datetime
    
if cp.cuda.is_available():
    print('Using GPU')
else:
    print('Using CPU')

# Modify the problem dimensions for 1D, 2D, and 3D
dims = 3
nx = ny = nz = 64
dx = dy = dz = 1.0e-9
dV = dx*dy*dz
#dx = dy = dz = 1.0
keyword = "3D-Multiparticle-Nucl"
timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
folder_name = f"{keyword}_{timestamp}"
os.mkdir(folder_name)
total_step = 30000
nprint = 1000

dt = 1.42

ic = 'ellipsoid'        # Initial condition: 'ellipsoid', 'sphere' 

elastic_flag = False  # Calculate elasticity
heterogenity = 'hetero'         # Possible values: 'hetero', 'homo_T1', 'homo_al'

radius = (6*dx, 6*dx, 6*dx)
transition_width = 2*dx

var_mobility = True
L_eta = 1.0e-11  # Linear reaction rate

nucl_flag = True
# Initialize concentration array and other parameters based on the dimensionality
if dims == 3:
    shape = (nx, ny, nz)
    sum_conv = 'ij,ixyz,jxyz->xyz'
elif dims == 2:
    shape = (nx, ny)
    sum_conv = 'ij,ixy,jxy->xy'
elif dims == 1:
    shape = (nx,)
    sum_conv = 'ij,ix,jx->x'

v_al = 0.5
v_cu = 0.25
v_li = 0.25

# cu0 = 0.02
# li0 = 0.035
# cueq = 0.02
# lieq = 0.035

#3D sims performed with this
cu0 = 0.01
li0 = 0.03
cueq = 0.01
lieq = 0.03

# 1D/2D-Eq sims performed with this
# cu0 = 0.02
# li0 = 0.02
# cueq = 0.01
# lieq = 0.03

T = 273 + 155  # Temperature [K]
R = 8.31446  # Gas constant [Jmol-1K-1]
Vm = 1.06e-5  # Molar volume [m3mol-1]
D0_cu = 6.54e-5  # Pre-exponential factor for Copper [m2s-1]
D0_li = 3.5e-5  # Pre-exponential factor for Lithium [m2s-1]
Q_cu = 136.0e3  # Activation energy for Copper [Jmol-1]
Q_li = 126.1e3  # Activation energy for Lithium [Jmol-1]
damping_factor = 1000
# Calculate diffusion constants
D_cu = D0_cu * math.exp(-(Q_cu / (R * T)))  # Diffusion constant for Copper [m2s-1]
D_li = D0_li * math.exp(-(Q_li / (R * T)))  # Diffusion constant for Lithium [m2s-1]

M_cu = D_cu*Vm/(R*T)  # Interdiffusion mobility for Copper [m5J-1s-1]
M_li = D_li*Vm/(R*T)  # Interdiffusion mobility for Lithium [m5J-1s-1]

alpha = 1.0
tol = 1e-6
max_iter = 100

cp11, cp12, cp13, cp33, cp44 = 174.2e9, 73.8e9, 32.9e9, 156.7e9, 21.6e9
cm11, cm12, cm44 = 107.9e9, 62.9e9, 33.8e9
sfts = np.array([
    [-0.001, 0, 0],
    [0, -0.0509, 0],
    [0, 0, -0.001]
])

int_width = 1*dx
# Interfacial energy coefficients (https://doi.org/10.1016/j.jallcom.2024.174495)
# gamma_bas = 0.110  # Interfacial energy of the basal plane [Jm-2]
# gamma_periph = 0.694  # Interfacial energy of the periphery planes [Jm-2]

gamma_bas = 0.110  # Interfacial energy of the basal plane [Jm-2]
gamma_periph = 0.694  # Interfacial energy of the periphery planes [Jm-2]
print(f"gradient energy basal: {1.5 * int_width * gamma_bas}")
print(f"gradient energy periphery: {1.5 * int_width * gamma_periph}")

w = 12*gamma_bas/(int_width)
print(f"double-well height: {w:.2e}")

kappa = np.array([
    [1.5 * int_width * gamma_periph, 0, 0],
    [0, 1.5 * int_width * gamma_bas, 0],
    [0, 0, 1.5 * int_width * gamma_periph]
])

Rot_matrix = np.array([
    [
        [-0.707, 0.577, 0.408],
        [0.707, 0.577, 0.408],
        [0.0, 0.577, -0.816]
    ],
    [
        [-0.707, -0.577, -0.408],
        [-0.707, 0.577, 0.408],
        [0.0, 0.577, -0.816]
    ],
    [
        [0.707, 0.577, 0.408],
        [0.707, -0.577, -0.408],
        [0.0, 0.577, -0.816]
    ],
    [
        [0.707, 0.577, 0.408],
        [-0.707, 0.577, 0.408],
        [0.0, -0.577, 0.816]
    ]
])
#Rot_matrix = np.array([np.identity(3),np.identity(3),np.identity(3),np.identity(3)])

# Create kappa_rot as a (4, 3, 3) array to store the rotated kappa matrices
kappa_rot = np.zeros((4, dims, dims))
sfts_rot = np.zeros((4, dims, dims))

# Apply each rotation matrix to kappa
for i in range(4):
    # Rotated kappa: R * kappa * R^T
    kappa_rot[i] = Rot_matrix[i][:dims, :dims] @ kappa[:dims, :dims] @ Rot_matrix[i][:dims, :dims].T
    #kappa_rot[i] = kappa[:dims, :dims]
    sfts_rot[i] = Rot_matrix[i][:dims, :dims] @ sfts[:dims, :dims] @ Rot_matrix[i][:dims, :dims].T
print(kappa_rot)
print(sfts_rot)
mu_t1 = calculate_mu_T1(T, v_al, v_cu, v_li)
L_AlCu_0, L_AlCu_1, L_AlCu_2 = calculate_L_AlCu(T)
L_AlLi_0, L_AlLi_1, L_AlLi_2 = calculate_L_AlLi(T)
L_CuLi_0, L_CuLi_1, L_CuLi_2 = calculate_L_CuLi(T)


h_cu, h_li, h_Ycu, h_Yli, h_eta_t1 = init_micro(shape, dx, radius, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic)
k, k2 = calc_wave_vector(shape, dx)

cu = cp.asarray(h_cu)
li = cp.asarray(h_li)
Ycu = cp.asarray(h_Ycu)
Yli = cp.asarray(h_Yli)
eta_t1 = [cp.asarray(h_eta_t1[i]) for i in range(4)]

kappa_rot = cp.asarray(kappa_rot)
sfts_rot = cp.asarray(sfts_rot)
Rot_matrix = cp.asarray(Rot_matrix)
eta_t1_old = [cp.zeros(shape) for _ in range(4)]
strain = cp.zeros((dims, dims) + shape)
stress = cp.zeros_like(strain)
f_eta_t1 = [cp.zeros(shape) for _ in range(4)]
Ycu_old = Ycu.copy()
Yli_old = Yli.copy()

f_Ycu = cp.zeros(shape)
f_Yli = cp.zeros(shape)

if dims == 2:
    C_eff, dC = symmetrize_C_2d(cp11, cp12, cp44, cm11, cm12, cm44, heterogenity)
    print(C_eff)
    print(dC)
    C_eff = cp.asarray(C_eff)
    dC = cp.asarray(dC)
    omega = compute_green_2d(C_eff,k)
    print(cp.max(omega))
    print(cp.min(omega))
elif dims == 3:
    C_eff, dC = symmetrize_C_3d(cp11, cp12, cp13, cp33, cp44, cm11, cm12, cm44, heterogenity)
    C_eff = cp.asarray(C_eff)
    dC = cp.asarray(dC)
    omega = compute_green_3d(C_eff,k)
    print(cp.max(omega))
    print(cp.min(omega))
    
#write_vtk('output_000000.vtk', h_cu, h_li, h_Ycu, h_Yli, cp.asnumpy(f_Ycu), cp.asnumpy(f_Yli), h_eta_t1, [cp.asnumpy(f_eta_t1[i]) for i in range(4)])

# Time-stepping loop
start = time.time()
for istep in range(total_step + 1):
    total_time = istep * dt
    deta_t1_dt = [(eta_t1[i] - eta_t1_old[i]) / dt for i in range(4)]
    h = [eta_t1[i]**3 *(10.0 - 15.0*eta_t1[i] + 6.0*eta_t1[i]**2)  for i in range(4)]
    dh_deta_t1 = [30.0*eta_t1[i]**2*(1.0 - eta_t1[i])**2 for i in range(4)]
    H = sum(h)
    
    sfts = cp.zeros_like(strain)
    if elastic_flag:
        for i in range(dims):
            for j in range(dims): 
                for v in range(len(h)):
                    sfts[i][j] += sfts_rot[v][i][j] * h[v]
    
    mu_al = calculate_mu_al(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    mu_cu = calculate_mu_cu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    mu_li = calculate_mu_li(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    
    dmu_al_dcu = calculate_dmu_al_dcu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    dmu_cu_dcu = calculate_dmu_cu_dcu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    dmu_al_dli = calculate_dmu_al_dli(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    dmu_li_dli = calculate_dmu_li_dli(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2)
    
    #M_cu = D_cu * Vm /(dmu_cu_dcu - dmu_al_dcu)
    #M_li = D_li* Vm /(dmu_li_dli - dmu_al_dli)

    mu_el_t1 = cp.zeros(shape)
    if elastic_flag:
        strain, stress, alpha = solve_elasticity(omega, C_eff, dC, sfts, strain, stress, H, shape, alpha, tol, max_iter)
        for ii in range(dims):
            for jj in range(dims):
                for kk in range(dims):
                    for ll in range(dims):
                        mu_el_t1 += (C_eff[ii][jj][kk][ll] + dC[ii][jj][kk][ll]  * (H-0.5)) * (strain[ii][jj]-sfts[ii][jj]) * (sfts[kk][ll])
        
    f_eta_t1[0] = (1/Vm)*dh_deta_t1[0]*(mu_t1 - v_al*mu_al - v_cu*mu_cu - v_li*mu_li) + w*(4*eta_t1[0]**3 - 6*eta_t1[0]**2 + 2*eta_t1[0] + 10*(eta_t1[0]*(eta_t1[1]**2 + eta_t1[2]**2 + eta_t1[3]**2))) -dh_deta_t1[0]*mu_el_t1
    f_eta_t1[1] = (1/Vm)*dh_deta_t1[1]*(mu_t1 - v_al*mu_al - v_cu*mu_cu - v_li*mu_li) + w*(4*eta_t1[1]**3 - 6*eta_t1[1]**2 + 2*eta_t1[1] + 10*(eta_t1[1]*(eta_t1[0]**2 + eta_t1[2]**2 + eta_t1[3]**2))) -dh_deta_t1[1]*mu_el_t1
    f_eta_t1[2] = (1/Vm)*dh_deta_t1[2]*(mu_t1 - v_al*mu_al - v_cu*mu_cu - v_li*mu_li) + w*(4*eta_t1[2]**3 - 6*eta_t1[2]**2 + 2*eta_t1[2] + 10*(eta_t1[2]*(eta_t1[0]**2 + eta_t1[1]**2 + eta_t1[3]**2))) -dh_deta_t1[2]*mu_el_t1
    f_eta_t1[3] = (1/Vm)*dh_deta_t1[3]*(mu_t1 - v_al*mu_al - v_cu*mu_cu - v_li*mu_li) + w*(4*eta_t1[3]**3 - 6*eta_t1[3]**2 + 2*eta_t1[3] + 10*(eta_t1[3]*(eta_t1[0]**2 + eta_t1[1]**2 + eta_t1[2]**2))) -dh_deta_t1[3]*mu_el_t1
        
    f_Ycu = calc_df_CH(Vm, M_cu, D_cu, v_cu, cu, Ycu, Ycu_old, mu_al, mu_cu, dh_deta_t1, deta_t1_dt, k, k2, dt, H, damping_factor)
    f_Yli = calc_df_CH(Vm, M_li, D_li, v_li, li, Yli, Yli_old, mu_al, mu_li, dh_deta_t1, deta_t1_dt, k, k2, dt, H, damping_factor)

    eta_t1_old = [eta_t1[i].copy() for i in range(4)]
    Ycu_old = Ycu.copy()
    Yli_old = Yli.copy()

    Ycu = cp.real(cp.fft.ifftn((cp.fft.fftn(Ycu) + dt * cp.fft.fftn(f_Ycu))/(1 + k2 * dt * damping_factor*D_cu)))
    Yli = cp.real(cp.fft.ifftn((cp.fft.fftn(Yli) + dt * cp.fft.fftn(f_Yli))/(1 + k2 * dt * damping_factor*D_li)))
    if dims == 3 and var_mobility:
        L0 = calcMobility(eta_t1[0], L_eta, k, Rot_matrix[0])
        L1 = calcMobility(eta_t1[1], L_eta, k, Rot_matrix[1])
        L2 = calcMobility(eta_t1[2], L_eta, k, Rot_matrix[2])
        L3 = calcMobility(eta_t1[3], L_eta, k, Rot_matrix[3])
    else:
        L0 = L_eta
        L1 = L_eta
        L2 = L_eta
        L3 = L_eta
    
    if dims == 3 or dims == 2:
        eta_t1[0] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[0]) - cp.fft.fftn(dt * L0 *f_eta_t1[0]))/(1 + cp.fft.fftn(dt * L0 * cp.fft.ifftn(cp.einsum(sum_conv, kappa_rot[0], k, k))))))
        eta_t1[1] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[1]) - cp.fft.fftn(dt * L1 *f_eta_t1[1]))/(1 + cp.fft.fftn(dt * L1 * cp.fft.ifftn(cp.einsum(sum_conv, kappa_rot[1], k, k))))))
        eta_t1[2] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[2]) - cp.fft.fftn(dt * L2 *f_eta_t1[2]))/(1 + cp.fft.fftn(dt * L2 * cp.fft.ifftn(cp.einsum(sum_conv, kappa_rot[2], k, k))))))
        eta_t1[3] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[3]) - cp.fft.fftn(dt * L3 *f_eta_t1[3]))/(1 + cp.fft.fftn(dt * L3 * cp.fft.ifftn(cp.einsum(sum_conv, kappa_rot[3], k, k))))))
    else:
        #explicit
        eta_t1[0] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[0]) - cp.fft.fftn(dt * L1 *f_eta_t1[0]))/(1 + cp.fft.fftn(dt * L1 * cp.fft.ifftn(kappa_rot[0]*k2)))))
        eta_t1[1] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[1]) - cp.fft.fftn(dt * L1 *f_eta_t1[1]))/(1 + cp.fft.fftn(dt * L1 * cp.fft.ifftn(kappa_rot[1]*k2)))))
        eta_t1[2] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[2]) - cp.fft.fftn(dt * L2 *f_eta_t1[2]))/(1 + cp.fft.fftn(dt * L2 * cp.fft.ifftn(kappa_rot[2]*k2)))))
        eta_t1[3] = cp.real(cp.fft.ifftn((cp.fft.fftn(eta_t1[3]) - cp.fft.fftn(dt * L3 *f_eta_t1[3]))/(1 + cp.fft.fftn(dt * L3 * cp.fft.ifftn(kappa_rot[3]*k2)))))
    
    cu = cp.exp(Ycu)/(1 + cp.exp(Ycu))
    li = cp.exp(Yli)/(1 + cp.exp(Yli))
    if istep % nprint == 0:
        h_cu = cp.asnumpy(cu)
        h_li = cp.asnumpy(li)
        h_Ycu = cp.asnumpy(Ycu)
        h_Yli = cp.asnumpy(Yli)
        h_eta_t1 = [cp.asnumpy(eta_t1[i]) for i in range(4)]
        print(f"Volume fraction eta_t1: {cp.mean(eta_t1[0])}")
        print(f"Average concentration cu: {cp.mean(cu + v_cu*eta_t1[0])}")
        print(f"Average concentration li: {cp.mean(li + v_li*eta_t1[0])}")
        if dims == 3:
            for i in range(1):
                coords = np.column_stack(np.nonzero(np.where(h_eta_t1[0] > 0.5, 1, 0)))
                if coords.size == 0:
                    continue
                pca = PCA(n_components=3)
                pca.fit(coords)
                lengths = 4 * np.sqrt(pca.explained_variance_) * dx
            #     print(f"Particle {i+1} axis lengths (in meters): {lengths}")
            # with open(folder_name + "/lengths.txt", "a") as file:
            #     file.write(f"{total_time} {lengths[0]} {lengths[1]} {lengths[2]}\n")
        if dims == 1 or dims == 2:
            with open(folder_name + "/lengths.txt", "a") as file:
                file.write(f"{total_time} {np.sum(h_eta_t1[0] > 0.5)*dx} {np.mean(h_eta_t1[0])} \n")
        h_f_eta_t1 = [cp.asnumpy(f_eta_t1[i]) for i in range(4)]
        
        h_mu_al = cp.asnumpy(calculate_mu_al(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2))
        h_mu_cu = cp.asnumpy(calculate_mu_cu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2))
        h_mu_li = cp.asnumpy(calculate_mu_li(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2))
        #h_mu_t1 = calculate_mu_T1(T, v_al, v_cu, v_li)*np.ones(shape)
        h_mu_t1 = cp.asnumpy(mu_t1 - v_al*mu_al - v_cu*mu_cu - v_li*mu_li)
        h_strain = cp.asnumpy(strain)
        write_vtk(f'output_{istep+1:06}.vtk', folder_name, h_cu, h_li, h_Ycu, h_Yli, cp.asnumpy(f_Ycu.real), cp.asnumpy(f_Yli.real), h_eta_t1, h_f_eta_t1, h_mu_t1, h_mu_al, h_mu_cu, h_mu_li, h_strain)

        print(f"Step {istep+1}/{total_step}")
end = time.time()

gpu_stats = gpustat.GPUStatCollection.new_query()
for gpu in gpu_stats.gpus:
    print(f"GPU {gpu.index}: {gpu.name}, Utilization: {gpu.utilization}%")

print("It takes ", (end-start)*1000.0, "ms")