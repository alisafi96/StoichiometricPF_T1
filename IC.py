import numpy as np
from scipy.ndimage import affine_transform

def init_micro_1d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic):
    """
    Initialize the microstructure with a sharp sphere.
    """
    # Extract dimensions from shape
    Nx = shape[0]

    # Initialize arrays based on shape
    h_cu = np.zeros(shape)
    h_li = np.zeros(shape)
    h_Ycu = np.zeros(shape)
    h_Yli = np.zeros(shape)
    h_eta_t1 = [np.zeros(shape) for _ in range(4)]

    # Calculate center coordinates
    centreX = (Nx * dx) / 2.0

    x = np.arange(Nx) * dx
    X = np.meshgrid(x, indexing='ij')[0]
    dist = np.abs((X - centreX))
    radius_tanh = 0.5 * (1.0 - np.tanh((dist - radii[0]) / transition_width))

    h_cu = cu0*np.ones(shape) - np.mean(radius_tanh*0.25)
    h_li = li0*np.ones(shape) - np.mean(radius_tanh*0.25)
    if np.any(h_cu < 0) or np.any(h_li < 0):
        raise ValueError("comps are negative, aborting.")
    h_Ycu = np.log(h_cu / (1 - h_cu))
    h_Yli = np.log(h_li / (1 - h_li))
    h_eta_t1[0] = radius_tanh
    
    return h_cu, h_li, h_Ycu, h_Yli, h_eta_t1

def init_micro_2d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic):
    """
    Initialize the microstructure with a sharp sphere.
    """
    # Extract dimensions from shape
    Nx = shape[0]
    Ny = shape[1]

    # Initialize arrays based on shape
    h_cu = np.zeros(shape)
    h_li = np.zeros(shape)
    h_Ycu = np.zeros(shape)
    h_Yli = np.zeros(shape)
    h_eta_t1 = [np.zeros(shape) for _ in range(4)]

    # Calculate center coordinates
    centreX = (Nx * dx) / 2.0
    centreY = (Ny * dx) / 2.0

    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dx
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt(((X - centreX)) ** 2 + ((Y - centreY)) ** 2)

    radius_tanh = 0.5 * (1.0 - np.tanh((dist - radii[0]) / transition_width))
    h_cu = cu0 + radius_tanh * (cueq - cu0)
    h_li = li0 + radius_tanh * (lieq - li0)
    h_eta_t1[0] = radius_tanh
    
    h_cu = cu0*np.ones(shape) - np.mean(radius_tanh*0.25)
    h_li = li0*np.ones(shape) - np.mean(radius_tanh*0.25)
    if np.any(h_cu < 0) or np.any(h_li < 0):
        raise ValueError("comps are negative, aborting.")
    
    h_Ycu = np.log(h_cu / (1 - h_cu))
    h_Yli = np.log(h_li / (1 - h_li))
    
    return h_cu, h_li, h_Ycu, h_Yli, h_eta_t1

def init_micro_3d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic):
    Nx = shape[0]
    Ny = shape[1] if len(shape) > 1 else 1
    Nz = shape[2] if len(shape) > 2 else 1

    # Make arrays for composition
    h_cu = np.zeros(shape)
    h_li = np.zeros(shape)
    h_Ycu = np.zeros(shape)
    h_Yli = np.zeros(shape)
    h_eta_t1 = [np.zeros(shape) for _ in range(4)]

    # Global coordinate mesh
    x = np.arange(Nx)*dx
    y = np.arange(Ny)*dx
    z = np.arange(Nz)*dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Controlls initial diffuse width
    smoothness = 3
    if ic == 'ellipsoid':
        rad = np.array([radii[0], radii[1], radii[2]])
        
        Xc = X- dx * Nx/2.0
        Yc = Y- dx * Ny/2.0
        Zc = Z- dx * Nz/2.0
        
        X_rot = Rot_matrix[0][0][0] * Xc + Rot_matrix[0][1][0] * Yc + Rot_matrix[0][2][0] * Zc 
        Y_rot = Rot_matrix[0][0][1] * Xc + Rot_matrix[0][1][1] * Yc + Rot_matrix[0][2][1] * Zc
        Z_rot = Rot_matrix[0][0][2] * Xc + Rot_matrix[0][1][2] * Yc + Rot_matrix[0][2][2] * Zc

        radius_tanh = 0.5 * (1 - np.tanh(smoothness * (((X_rot**2 / rad[0]**2) + (Y_rot**2 / rad[1]**2) + (Z_rot**2 / rad[2]**2)) - 1)))
    elif ic == 'sphere':
        centre = np.array([Nx*dx/2.0, Ny*dx/2.0, Nz*dx/2.0], dtype=float)
        if len(shape) == 1:
            centre[1] = 0.0
            centre[2] = 0.0
        elif len(shape) == 2:
            centre[2] = 0.0
            
        dist = np.sqrt((X - centre[0])**2 + (Y - centre[1])**2 + (Z - centre[2])**2)
        radius_tanh = 0.5 * (1.0 - np.tanh((dist - radii[0]) / transition_width))
    else:
        # No disc shape if `ic != 'sphere'`
        radius_tanh = np.zeros(shape)
    
    h_cu = cu0 + radius_tanh * (cueq - cu0)
    h_li = li0 + radius_tanh * (lieq - li0)
    

    # and the logit transforms
    eps = 1e-16
    h_Ycu = np.log((h_cu + eps)/(1.0 - h_cu + eps))
    h_Yli = np.log((h_li + eps)/(1.0 - h_li + eps))

    # store our disc shape in h_eta_t1[0] for convenience
    h_eta_t1[0] = radius_tanh

    return h_cu, h_li, h_Ycu, h_Yli, h_eta_t1

# Note: use/uncomment for single particle
# def init_micro(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic):
#     """
#     Initialize the microstructure with given parameters, adaptable for 1D, 2D, or 3D grids.

#     Here we show how to define a 'disc' in a local reference frame and then
#     rotate it into the global frame using the provided rotation matrix.
#     """
#     dims = len(shape)
    
#     if dims == 1:
#         h_cu, h_li, h_Ycu, h_Yli, h_eta_t1 = init_micro_1d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic)
#     elif dims == 2:
#         h_cu, h_li, h_Ycu, h_Yli, h_eta_t1 = init_micro_2d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic)
#     elif dims == 3:
#         h_cu, h_li, h_Ycu, h_Yli, h_eta_t1 = init_micro_3d(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic)
#     else:
#         raise ValueError("Invalid number of dimensions in shape")

#     return h_cu, h_li, h_Ycu, h_Yli, h_eta_t1

# Note: use/uncomment for Multiparticle
def init_micro(shape, dx, radii, transition_width, cu0, li0, cueq, lieq, Rot_matrix, ic):
    """
    Initialize the microstructure with particles at specified centers.

    Parameters:
    shape (tuple): Grid dimensions (Nx, Ny, Nz). Ny and Nz can be 1 for 1D or 2D cases.
    dx (float): Uniform grid spacing for all dimensions.
    radii (tuple or float): Radii of the ellipsoid in each dimension (ax, ay, az).
    transition_width (float): Transition width for the microstructure.
    cu0, li0, cueq, lieq (float): Initial and equilibrium concentrations for Cu and Li.
    particle_centers (list): List of fractional coordinates for particles (relative to Nx, Ny, Nz).

    Returns:
    tuple: Initialized arrays for h_cu, h_li, h_Ycu, h_Yli, h_eta_t1.
    """
    particle_centers = [
        [0.53736822, 0.24788356, 0.8756677],
        [0.72010626, 0.85159915, 0.81586188],
        [0.57831998, 0.83749939, 0.170794],
        [0.25678629, 0.13618183, 0.36026426],
        [0.71094183, 0.31707923, 0.26299001],
        [0.38540266, 0.32474761, 0.53415687],
        [0.21273938, 0.74175758, 0.15964051],
        [0.88950955, 0.71779582, 0.25897255],
        [0.10441769, 0.75236914, 0.66548588],
        [0.68320573, 0.71701628, 0.15923572],
        [0.88677258, 0.19269525, 0.79048274],
        [0.5986385, 0.36471842, 0.15084668],
    ]
    #Twoparticle cases
    # particle_centers = [
    #     [0.25, 0.5, 0.5],
    #     [0.75, 0.5, 0.5],
    # ]
    # particle_centers = [
    #     [0.3, 0.6, 0.3],
    #     [0.6, 0.3, 0.6],
    # ]
    # Extract dimensions from shape
    Nx = shape[0]
    Ny = shape[1] if len(shape) > 1 else 1
    Nz = shape[2] if len(shape) > 2 else 1

    # Initialize arrays based on shape
    h_cu = np.zeros(shape)
    h_li = np.zeros(shape)
    h_Ycu = np.zeros(shape)
    h_Yli = np.zeros(shape)
    h_eta_t1 = [np.zeros(shape) for _ in range(4)]

    # Create the grid
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dx if Ny > 1 else [0]
    z = np.arange(Nz) * dx if Nz > 1 else [0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    i = 0
    smoothness = 3
    # Process each particle
    for center in particle_centers:
        center_x = center[0] * Nx * dx
        center_y = center[1] * Ny * dx
        center_z = center[2] * Nz * dx

        Xc = X - center_x
        Yc = Y - center_y
        Zc = Z - center_z
        e = i % 4 # Which particle to update
        X_rot = Rot_matrix[e][0][0] * Xc + Rot_matrix[e][1][0] * Yc + Rot_matrix[e][2][0] * Zc 
        Y_rot = Rot_matrix[e][0][1] * Xc + Rot_matrix[e][1][1] * Yc + Rot_matrix[e][2][1] * Zc
        Z_rot = Rot_matrix[e][0][2] * Xc + Rot_matrix[e][1][2] * Yc + Rot_matrix[e][2][2] * Zc

        h_eta_t1[e] += 0.5 * (1 - np.tanh(smoothness * (((X_rot**2 / radii[0]**2) + (Y_rot**2 / radii[1]**2) + (Z_rot**2 / radii[2]**2)) - 1)))
        i+=1
    
    h_cu = cu0 + np.sum(h_eta_t1, axis=0) * (cueq - cu0)
    h_li = li0 + np.sum(h_eta_t1, axis=0) * (lieq - li0)        
    h_Ycu = np.log(h_cu / (1 - h_cu))
    h_Yli = np.log(h_li / (1 - h_li))

    return h_cu, h_li, h_Ycu, h_Yli, h_eta_t1