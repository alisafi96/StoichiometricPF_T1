import vtk
import numpy as np
import os

def write_vtk(filename,folder_name, h_cu, h_li, h_Ycu, h_Yli, h_fYcu, h_fYli, h_eta_t1, h_f_eta_t1, h_mu_t1, h_mu_al, h_mu_cu, h_mu_li, h_strain):
    """
    Write the given h_ variables to a VTK file. 
    Adaptable for 1D, 2D, or 3D data.

    Parameters:
    filename (str): The name of the output VTK file.
    h_cu, h_li, h_Ycu, h_Yli (np.ndarray): 1D, 2D, or 3D arrays to write.
    h_eta_t1 (list of np.ndarray): List of 4 arrays to write (can be 1D, 2D, or 3D).
    h_strain (np.ndarray): Strain tensor of shape (dim, dim, shape).
    """
    # Determine the dimensions of the input data
    shape = h_cu.shape
    if len(shape) == 1:
        Nx, Ny, Nz = shape[0], 1, 1  # 1D
    elif len(shape) == 2:
        Nx, Ny, Nz = shape[0], shape[1], 1  # 2D
    else:
        Nx, Ny, Nz = shape  # 3D

    # Create a vtkStructuredPoints object
    structured_points = vtk.vtkStructuredPoints()
    structured_points.SetDimensions(Nx, Ny, Nz)

    # Create a vtkDoubleArray for each scalar field
    def create_vtk_array(name, data):
        vtk_array = vtk.vtkDoubleArray()
        vtk_array.SetName(name)
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(Nx * Ny * Nz)
        vtk_array.SetVoidArray(data.ravel(), Nx * Ny * Nz, 1)
        return vtk_array

    # Create vtkDoubleArray objects for each h_ variable
    cu_array = create_vtk_array("h_cu", h_cu)
    li_array = create_vtk_array("h_li", h_li)
    Ycu_array = create_vtk_array("h_Ycu", h_Ycu)
    Yli_array = create_vtk_array("h_Yli", h_Yli)
    f_Ycu_array = create_vtk_array("h_fYcu", h_fYcu)
    f_Yli_array = create_vtk_array("h_fYli", h_fYli)
    eta1_array = create_vtk_array("h_eta_t1_1", h_eta_t1[0])
    eta2_array = create_vtk_array("h_eta_t1_2", h_eta_t1[1])
    eta3_array = create_vtk_array("h_eta_t1_3", h_eta_t1[2])
    eta4_array = create_vtk_array("h_eta_t1_4", h_eta_t1[3])
    f_eta1_array = create_vtk_array("h_f_eta_t1_1", h_f_eta_t1[0])
    f_eta2_array = create_vtk_array("h_f_eta_t1_2", h_f_eta_t1[1])
    f_eta3_array = create_vtk_array("h_f_eta_t1_3", h_f_eta_t1[2])
    f_eta4_array = create_vtk_array("h_f_eta_t1_4", h_f_eta_t1[3])
    
    mu_t1_array = create_vtk_array("h_mu_t1", h_mu_t1)
    mu_al_array = create_vtk_array("h_mu_al", h_mu_al)
    mu_cu_array = create_vtk_array("h_mu_cu", h_mu_cu)
    mu_li_array = create_vtk_array("h_mu_li", h_mu_li)

    # Create vtkDoubleArray objects for each component of the strain tensor
    strain_arrays = []
    dim = h_strain.shape[0]
    for i in range(dim):
        for j in range(dim):
            strain_name = f"h_strain_{i}_{j}"
            strain_array = create_vtk_array(strain_name, h_strain[i, j])
            strain_arrays.append(strain_array)

    # Add these arrays to the structured points object
    structured_points.GetPointData().AddArray(cu_array)
    structured_points.GetPointData().AddArray(li_array)
    structured_points.GetPointData().AddArray(Ycu_array)
    structured_points.GetPointData().AddArray(Yli_array)
    structured_points.GetPointData().AddArray(f_Ycu_array)
    structured_points.GetPointData().AddArray(f_Yli_array)
    structured_points.GetPointData().AddArray(eta1_array)
    structured_points.GetPointData().AddArray(eta2_array)
    structured_points.GetPointData().AddArray(eta3_array)
    structured_points.GetPointData().AddArray(eta4_array)
    structured_points.GetPointData().AddArray(f_eta1_array)
    structured_points.GetPointData().AddArray(f_eta2_array)
    structured_points.GetPointData().AddArray(f_eta3_array)
    structured_points.GetPointData().AddArray(f_eta4_array)
    structured_points.GetPointData().AddArray(mu_t1_array)
    structured_points.GetPointData().AddArray(mu_al_array)
    structured_points.GetPointData().AddArray(mu_cu_array)
    structured_points.GetPointData().AddArray(mu_li_array)

    for strain_array in strain_arrays:
        structured_points.GetPointData().AddArray(strain_array)

    # Write the data to a VTK file
    writer = vtk.vtkStructuredPointsWriter()
    full_path = os.path.join(folder_name, filename)
    writer.SetFileName(full_path)
    writer.SetInputData(structured_points)
    writer.Write()