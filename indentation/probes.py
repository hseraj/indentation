import numpy as np

def get_probe_geometry(probe_type: str, params: dict) -> tuple[float, float]:
    """
    Returns the contact coefficient (C) and the displacement exponent (h_power_exp)
    for a given probe type and its physical parameters.
    
    Args:
        probe_type: Name of the probe (e.g., "Spherical", "Conical").
        params: Dictionary of physical parameters (e.g., {"nu": 0.5, "R": 2e-3}).
        
    Returns:
        (C, h_power_exp)
    """
    nu = params["nu"]
    if not (0 <= nu <= 0.5):
        raise ValueError("Poisson's ratio (nu) must be between 0 and 0.5.")
        
    if probe_type == "Spherical":
        R = params["R"]
        if R <= 0: raise ValueError("Radius (R) must be greater than 0.")
        C = (4.0 * np.sqrt(R)) / (3.0 * (1.0 - nu**2))
        return C, 1.5
        
    elif probe_type == "Cylindrical":
        R = params["R"]
        if R <= 0: raise ValueError("Radius (R) must be greater than 0.")
        C = (2.0 * R) / (1.0 - nu**2)
        return C, 1.0
        
    elif probe_type == "Conical":
        theta = params["theta_deg"]
        if not (0 < theta < 90): raise ValueError("Theta must be between 0 and 90 degrees.")
        C = (2.0 * np.tan(np.radians(theta))) / (np.pi * (1.0 - nu**2))
        return C, 2.0
        
    elif probe_type == "Four-sided Pyramidal":
        theta = params["theta_deg"]
        if not (0 < theta < 90): raise ValueError("Theta must be between 0 and 90 degrees.")
        C = (3.0 * np.tan(np.radians(theta))) / (4.0 * (1.0 - nu**2))
        return C, 2.0
        
    elif probe_type == "Blunted Four-sided Pyramidal":
        theta = params["theta_deg"]
        if not (0 < theta < 90): raise ValueError("Theta must be between 0 and 90 degrees.")
        C = np.tan(np.radians(theta)) / (np.sqrt(2) * (1.0 - nu**2))
        return C, 2.0
        
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")