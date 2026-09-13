import numpy as np

# =============================================================================
# VISCOELASTIC MODELS REGISTRY
# =============================================================================

MODELS = {
    "Elastic_Spring": {
        "display": "Elastic Spring",
        "param_names": ["E"],
        "bounds": [(1e-6, 1e12)],
        "E_star_func": lambda s, E: np.full_like(s, E),
        "E_relax_func": lambda t, E: np.full_like(t, E),
    },
    "Dashpot": {
        "display": "Dashpot",
        "param_names": ["eta"],
        "bounds": [(1e-6, 1e12)],
        "E_star_func": lambda s, eta: eta * s,
        "E_relax_func": lambda t, eta: np.zeros_like(t),  # Impulsive, approximated as 0 for general t
    },
    "Kelvin_Voigt": {
        "display": "Kelvin-Voigt",
        "param_names": ["E", "eta"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e12)],
        "E_star_func": lambda s, E, eta: E + eta * s,
        "E_relax_func": lambda t, E, eta: np.full_like(t, E),  # Singular delta term omitted in time-domain build
    },
    "Maxwell": {
        "display": "Maxwell",
        "param_names": ["E", "tau"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e6)],
        "E_star_func": lambda s, E, tau: (E * tau * s) / (1.0 + tau * s),
        "E_relax_func": lambda t, E, tau: E * np.exp(-t / tau),
    },
    "Standard_Linear_Solid": {
        "display": "Standard Linear Solid",
        "param_names": ["E1", "E2", "tau"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e12), (1e-6, 1e6)],
        "E_star_func": lambda s, E1, E2, tau: E1 + (E2 * tau * s) / (1.0 + tau * s),
        "E_relax_func": lambda t, E1, E2, tau: E1 + E2 * np.exp(-t / tau),
    },
    "Prony_N1": {
        "display": "Prony N=1",
        "param_names": ["E_inf", "E1", "tau1"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e12), (1e-6, 1e6)],
        "E_star_func": lambda s, E_inf, E1, t1: E_inf + (E1 * t1 * s) / (1.0 + t1 * s),
        "E_relax_func": lambda t, E_inf, E1, t1: E_inf + E1 * np.exp(-t / t1),
    },
    "Prony_N2": {
        "display": "Prony N=2",
        "param_names": ["E_inf", "E1", "tau1", "E2", "tau2"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e12), (1e-6, 1e6), (1e-6, 1e12), (1e-6, 1e6)],
        "E_star_func": lambda s, E_inf, E1, t1, E2, t2: E_inf + (E1 * t1 * s) / (1.0 + t1 * s) + (E2 * t2 * s) / (1.0 + t2 * s),
        "E_relax_func": lambda t, E_inf, E1, t1, E2, t2: E_inf + E1 * np.exp(-t / t1) + E2 * np.exp(-t / t2),
    },
    "Prony_N3": {
        "display": "Prony N=3",
        "param_names": ["E_inf", "E1", "tau1", "E2", "tau2", "E3", "tau3"],
        "bounds": [(1e-6, 1e12), (1e-6, 1e12), (1e-6, 1e6), (1e-6, 1e12), (1e-6, 1e6), (1e-6, 1e12), (1e-6, 1e6)],
        "E_star_func": lambda s, E_inf, E1, t1, E2, t2, E3, t3: E_inf + (E1 * t1 * s) / (1.0 + t1 * s) + (E2 * t2 * s) / (1.0 + t2 * s) + (E3 * t3 * s) / (1.0 + t3 * s),
        "E_relax_func": lambda t, E_inf, E1, t1, E2, t2, E3, t3: E_inf + E1 * np.exp(-t / t1) + E2 * np.exp(-t / t2) + E3 * np.exp(-t / t3),
    },
}