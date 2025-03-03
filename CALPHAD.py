import math
import cupy as cp
import numpy as np
def gibbs_energy_Al(T):
    """
    Calculate Gibbs free energy for Aluminum (Al) at temperature T.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: Gibbs free energy at temperature T.
    """
    # Constants
    A = -7976.15
    B = 137.093038
    C = -24.3671976
    D = -1.884662e-3
    E = -0.877664e-6
    F = 74092

    # Calculate Gibbs free energy
    G = A + B * T + C * T * math.log(T) + D * T**2 + E * T**3 + F * T**-1
    return G
    
def gibbs_energy_Cu(T):
    """
    Calculate Gibbs free energy for Copper (Cu) at temperature T.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: Gibbs free energy at temperature T.
    """
    # Constants
    A = -7770.458
    B = 130.485235
    C = -24.112392
    D = -2.65684E-3
    E = 0.129223E-6
    F = 52478

    # Calculate Gibbs free energy
    G = A + B*T + C*T*math.log(T) + D*pow(T, 2) + E*pow(T, 3) + F*pow(T, -1)
    return G

def gibbs_energy_Li(T):
    """
    Calculate Gibbs free energy for Lithium (Li) at temperature T.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: Gibbs free energy at temperature T.
    """
    # Constants
    A = -10583.817
    B = 217.637482
    C = -38.940488
    D = 35.466931E-3
    E = -19.869816E-6
    F = 159994

    # Calculate Gibbs free energy
    G = A + B*T + C*T*math.log(T) + D*pow(T, 2) + E*pow(T, 3) + F*pow(T, -1)
    return G

def calculate_mu_T1(T, v_al, v_cu, v_li):
    """
    Calculate the chemical potential of T1.

    Parameters:
    T (float): Temperature in Kelvin.
    v_al, v_cu, v_li (float): Coefficients for Aluminum, Copper, and Lithium.

    Returns:
    float: The chemical potential of T1.
    """
    mu_T1 = -24560 + 6.0 * T
    return mu_T1

def calculate_L_AlCu(T):
    """
    Calculate the binary parameters for Al-Cu.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    tuple: (L_AlCu_0, L_AlCu_1, L_AlCu_2)
    """
    # L_AlCu_0 = -53520 + 2 * T
    # L_AlCu_1 = 38590 - 2 * T
    # L_AlCu_2 = 1170
    # Kroupa et al. (https://doi.org/10.1007/s10853-020-05423-7)
    L_AlCu_0 = -54220.45 + 2.0034 * T
    L_AlCu_1 = 39015  - 2.368 * T
    L_AlCu_2 = 3218.23
    return L_AlCu_0, L_AlCu_1, L_AlCu_2

def calculate_L_AlLi(T):
    """
    Calculate the binary parameters for Al-Li.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    tuple: (L_AlLi_0, L_AlLi_1, L_AlLi_2)
    """
    # Cost507
    # L_AlLi_0 = -27000.0 + 8.0 * T
    # L_AlLi_1 = 1e-6
    # L_AlLi_2 = 3000.0 + 0.1 * T
    
    # Azza et al. (https://www.jmaterenvironsci.com/Document/vol6/vol6_N12/402-Azza.pdf)
    L_AlLi_0 = 0.0
    L_AlLi_1 = -55000+12*T
    L_AlLi_2 = -4*T
    return L_AlLi_0, L_AlLi_1, L_AlLi_2

def calculate_L_CuLi(T):
    """
    Calculate the binary parameters for Cu-Li.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    tuple: (L_CuLi_0, L_CuLi_1, L_CuLi_2)
    """
    # Cost507
    # L_CuLi_0 = 2750 + 13.0 * T
    # L_CuLi_1 = -1000
    # L_CuLi_2 = 0.0
    
    # Li et al. (https://doi.org/10.1016/j.calphad.2016.04.003)
    L_CuLi_0 = 34383.0459 - 70.0863278 * T + 11.37545667 * cp.log(T)
    L_CuLi_1 = -204984.122 + 140.574711 * T
    L_CuLi_2 = 0.0
    return L_CuLi_0, L_CuLi_1, L_CuLi_2

def calculate_mu_al(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    """
    Calculate the chemical potential mu_Al.

    Parameters:
    R (float): Gas constant.
    T (float): Temperature.
    al, cu, li (float): Concentrations of Al, Cu, and Li.
    L_AlCu_0, L_AlCu_1, L_AlCu_2 (float): Interaction parameters for Al-Cu.
    L_AlLi_0, L_AlLi_1, L_AlLi_2 (float): Interaction parameters for Al-Li.
    L_CuLi_0, L_CuLi_1, L_CuLi_2 (float): Interaction parameters for Cu-Li.

    Returns:
    float: Chemical potential of Al.
    """
    al = 1 - cu - li
    mu_al =(
        R * T * np.log(al)
        + cu*(L_AlCu_0 + L_AlCu_1 * (al - cu) + L_AlCu_2 *(al - cu)*(al - cu))
        + li*(L_AlLi_0 + L_AlLi_1 * (al - li) + L_AlLi_2 *(al - li)*(al - li))
        )
    return mu_al

def calculate_mu_cu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    """
    Calculate the chemical potential mu_Cu.

    Parameters:
    R (float): Gas constant.
    T (float): Temperature.
    al, cu, li (float): Concentrations of Al, Cu, and Li.
    L_AlCu_0, L_AlCu_1, L_AlCu_2 (float): Interaction parameters for Al-Cu.
    L_AlLi_0, L_AlLi_1, L_AlLi_2 (float): Interaction parameters for Al-Li.
    L_CuLi_0, L_CuLi_1, L_CuLi_2 (float): Interaction parameters for Cu-Li.

    Returns:
    float: Chemical potential of Cu.
    """
    al = 1 - cu - li
    mu_cu =(
        R * T * np.log(cu)
        + al*(L_AlCu_0 + L_AlCu_1 * (cu - al) + L_AlCu_2 *(cu - al)*(cu - al))
        + li*(L_CuLi_0 + L_CuLi_1 * (cu - li) + L_CuLi_2 *(cu - li)*(cu - li))
        )
    return mu_cu

def calculate_mu_li(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    """
    Calculate the chemical potential mu_Li.

    Parameters:
    R (float): Gas constant.
    T (float): Temperature.
    al, cu, li (float): Concentrations of Al, Cu, and Li.
    L_AlCu_0, L_AlCu_1, L_AlCu_2 (float): Interaction parameters for Al-Cu.
    L_AlLi_0, L_AlLi_1, L_AlLi_2 (float): Interaction parameters for Al-Li.
    L_CuLi_0, L_CuLi_1, L_CuLi_2 (float): Interaction parameters for Cu-Li.

    Returns:
    float: Chemical potential of Li.
    """
    al = 1 - cu - li
    mu_li =(    
        R * T * np.log(li)
        + al*(L_AlLi_0 + L_AlLi_1 * (li - al) + L_AlLi_2 *(li - al)*(li - al))
        + cu*(L_CuLi_0 + L_CuLi_1 * (li - cu) + L_CuLi_2 *(li - cu)*(li - cu))
        )
    return mu_li

# Derivatives
def calculate_dmu_al_dcu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    al = 1 - cu - li
    dmu_al_dcu =(
        -R * T / (al)
        + (L_AlCu_0 + L_AlCu_1 * (al - cu) + L_AlCu_2 *(al - cu)**2)
        + cu*(- 2*L_AlCu_1 -4*L_AlCu_2 *(al - cu)) + li*(-L_CuLi_1 - 2*L_CuLi_2 *(al - li))
        )
    return dmu_al_dcu

def calculate_dmu_cu_dcu(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    al = 1 - cu - li
    dmu_cu_dcu =(
        R * T / (cu)
        - (L_AlCu_0 + L_AlCu_1 * (cu - al) + L_AlCu_2 *(cu - al)**2)
        + al*(2*L_AlCu_1 + 4*L_AlCu_2 *(cu - al)) + li*(L_CuLi_1 + 2*L_CuLi_2 *(cu - li))
        )
    return dmu_cu_dcu

def calculate_dmu_al_dli(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    al = 1 - cu - li
    dmu_al_dli =(
        -R * T / (al)
        + (L_AlLi_0 + L_AlLi_1 * (al - li) + L_AlLi_2 *(al - li)**2)
        + li*(- 2*L_AlLi_1 -4*L_AlLi_2 *(al - li)) + cu*(-L_CuLi_1 - 2*L_CuLi_2 *(al - cu))
        )
    return dmu_al_dli

def calculate_dmu_li_dli(R, T, cu, li, L_AlCu_0, L_AlCu_1, L_AlCu_2, L_AlLi_0, L_AlLi_1, L_AlLi_2, L_CuLi_0, L_CuLi_1, L_CuLi_2):
    al = 1 - cu - li
    dmu_li_dli =(
        R * T / (li)
        - (L_AlLi_0 + L_AlLi_1 * (li - al) + L_AlLi_2 *(li - al)**2)
        + al*(2*L_AlLi_1 + 4*L_AlLi_2 *(li - al)) + cu*(L_CuLi_1 + 2*L_CuLi_2 *(li - cu))
        )
    return dmu_li_dli

def calc_df_CH(Vm, M, D, v, comp, Ycomp, Ycomp_old, mu_al, mu_comp, dh_deta_t1, deta_t1_dt, k, k2, dt, H, damping_factor):
    if len(mu_al.shape) == 3:
        conv1 = 'ixyz,xyz->ixyz'
        conv2 = 'ixyz,ixyz->xyz'
    elif len(mu_al.shape) == 2:
        conv1 = 'ixy,xy->ixy'
        conv2 = 'ixy,ixy->xy'
    else:
        conv1 = 'ijx,x->x'
        conv2 = 'ijx,x->x'
    
    term1 = cp.fft.ifftn(- cp.einsum(conv2, k, cp.fft.fftn((1/Vm)*M* cp.fft.ifftn(cp.einsum(conv1, k, cp.fft.fftn(mu_comp - mu_al)))))).real
    term2 = sum(dh * ddt for dh, ddt in zip(dh_deta_t1, deta_t1_dt)) * (v - comp)
    term3 = cp.fft.ifftn(- k2 *cp.fft.fftn(Ycomp)).real
    term4 = ((1-H) * cp.exp(Ycomp)/((1 + cp.exp(Ycomp))**2 )- 1) * (Ycomp - Ycomp_old)/dt

    f_Y = term1 - term2  - damping_factor*D*term3 - term4
    return f_Y



def calcMobility(eta, L_eta, k, Rot_matrix):
    # Determine input dimensionality
    is_3d = len(eta.shape) == 3
    pad_width = ((1, 1), (1, 1), (1, 1)) if is_3d else ((1, 1), (1, 1))
    
    # Extend the array to handle boundary conditions
    extended_array = cp.pad(eta, pad_width=pad_width, mode='wrap')
    grad_x_extended = cp.gradient(extended_array, axis=0)
    grad_y_extended = cp.gradient(extended_array, axis=1)
    grad_z_extended = cp.gradient(extended_array, axis=2)
    grad_x = grad_x_extended[1:-1, 1:-1, 1:-1]
    grad_y = grad_y_extended[1:-1, 1:-1, 1:-1]
    grad_z = grad_z_extended[1:-1, 1:-1, 1:-1]
    
    grad = (grad_x, grad_y, grad_z)
    
    # Compute norm (avoid division by zero)
    epsilon = 1e-10
    norm = cp.sqrt(sum(cp.square(g) for g in grad))
    norm = cp.maximum(norm, epsilon)
    
    # Compute theta
    theta = cp.arccos((grad[0] * Rot_matrix[0][1] + grad[1] * Rot_matrix[1][1] + grad[2] * Rot_matrix[2][1]) / norm) - 0.5 * cp.pi
    
    # Handle NaNs in theta
    if cp.isnan(theta).any():
        print("NaN detected in theta")
    
    # Constants
    PHI = cp.pi / 10000
    beta = 10000
    factor_L = L_eta / (1 + beta)
    
    # Initialize L array
    L = cp.zeros_like(eta)
    
    # Conditions for setting L
    condition_1 = (theta >= -0.5 * cp.pi) & (theta < -0.5 * cp.pi + PHI)
    condition_2 = (theta >= -0.5 * cp.pi + PHI) & (theta <= 0.5 * cp.pi - PHI)
    condition_3 = (theta >= 0.5 * cp.pi - PHI) & (theta <= 0.5 * cp.pi)
    
    L[condition_1] = factor_L * (1 + beta / cp.sin(PHI) + beta * cp.cos(PHI) * cp.sin(theta[condition_1]) / cp.sin(PHI))
    L[condition_2] = factor_L * (1 + beta * cp.cos(theta[condition_2]))
    L[condition_3] = factor_L * (1 + beta / cp.sin(PHI) - beta * cp.cos(PHI) * cp.sin(theta[condition_3]) / cp.sin(PHI))
    
    #L = factor_L * (1 + beta * cp.cos(theta)**4)
    # Apply h_prime condition
    h_prime = 30 * eta**2 - 60 * eta**3 + 30 * eta**4
    #condition = h_prime < 0.001
    condition = eta < 0.001
    L[condition] = L_eta
    
    return L
