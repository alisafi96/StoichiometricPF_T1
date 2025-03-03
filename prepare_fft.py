import cupy as cp
def compute_k_component(k_array, size, delta, half_size):
        for i in range(size):
            if i < half_size:
                k_array[i] = i * delta
            else:
                k_array[i] = (i - size) * delta
        return k_array
        
def calc_wave_vector(shape, dx):
    """
    Calculate the wave vector components for a grid of given shape and spacing.

    Parameters:
    shape (tuple): Shape of the grid (nx, ny, nz).
    dx (float): Grid spacing.

    Returns:
    cp.ndarray: Wave vector components with shape (3, nx, ny, nz).
    """
    # Calculate midpoints and deltas for wave vectors
    Nx21 = shape[0] / 2
    Ny21 = shape[1] / 2 if len(shape) > 1 else 1
    Nz21 = shape[2] / 2 if len(shape) > 2 else 1

    delkx = 2.0 * cp.pi / (shape[0] * dx)
    delky = 2.0 * cp.pi / (shape[1] * dx) if len(shape) > 1 else 0
    delkz = 2.0 * cp.pi / (shape[2] * dx) if len(shape) > 2 else 0

    # Initialize arrays for wave vectors
    kx = cp.zeros(shape[0])
    ky = cp.zeros(shape[1]) if len(shape) > 1 else cp.array([0])
    kz = cp.zeros(shape[2]) if len(shape) > 2 else cp.array([0])

    # Use a dictionary to simulate switch-case based on the number of dimensions
    switch_case = {
        1: lambda: compute_k_component(kx, shape[0], delkx, Nx21),
        2: lambda: (compute_k_component(kx, shape[0], delkx, Nx21),
                    compute_k_component(ky, shape[1], delky, Ny21)),
        3: lambda: (compute_k_component(kx, shape[0], delkx, Nx21),
                    compute_k_component(ky, shape[1], delky, Ny21),
                    compute_k_component(kz, shape[2], delkz, Nz21)),
    }

    # Call the switch case based on the number of dimensions in the shape
    switch_case[len(shape)]()

    # Create the meshgrid for wave vectors
    if len(shape) == 1:
        Kx = cp.meshgrid(kx)
        k = cp.array([Kx])
    elif len(shape) == 2:
        Kx, Ky = cp.meshgrid(kx, ky)
        k = cp.array([Kx, Ky])
    elif len(shape) == 3:
        Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
        k = cp.array([Kx, Ky, Kz])
    k2 = cp.sum(k**2, axis=0)
    return k, k2