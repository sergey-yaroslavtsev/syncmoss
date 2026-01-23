"""
Mathematical support functions for SYNCMoss.

This module provides utility functions for mathematical operations including:
- Partial derivatives calculation
- Error propagation
- Intensity calculations with proper uncertainty estimation
"""

import numpy as np


def calculate_partial_derivative_numerical(func, x, param_index, h=1e-8):
    """
    Calculate partial derivative numerically using finite differences.
    
    Args:
        func: Function to differentiate (callable)
        x: Point at which to evaluate derivative (array-like)
        param_index: Index of parameter to differentiate with respect to
        h: Step size for numerical differentiation
    
    Returns:
        float: Partial derivative df/dx[param_index]
    """
    x = np.array(x, dtype=float)
    x_plus = x.copy()
    x_plus[param_index] += h
    
    return (func(x_plus) - func(x)) / h


def calculate_intensity_percentage_error(t_values, t_errors, covariance_matrix, t_indices, fixed_params=None):
    """
    Calculate intensity percentage and its error for each model component.
    Uses calculate_expression_error with generated expression strings.
    
    For model i: Intensity_i = 100 * T_i / Sum(T_j)
    
    Args:
        t_values: Array of all fitted parameters
        t_errors: Array of parameter errors (standard deviations)
        covariance_matrix: Covariance matrix from fitting (only variable parameters)
        t_indices: List of indices where T parameters are located in the full parameter array
        fixed_params: Array of indices of fixed parameters (where er is nan), or None
    
    Returns:
        tuple: (intensities, intensity_errors)
            intensities: numpy array of intensity percentages
            intensity_errors: numpy array of intensity errors
    """
    if not t_indices or len(t_indices) == 0:
        return np.array([]), np.array([])
    
    # Extract T values
    T = np.array([t_values[i] for i in t_indices])
    T_sum = np.sum(T)
    
    if T_sum == 0:
        return np.zeros(len(T)), np.zeros(len(T))
    
    # Calculate intensities
    intensities = 100 * T / T_sum
    
    # Calculate errors using expression-based approach
    intensity_errors = np.zeros(len(T))
    
    # Build sum expression for denominator
    sum_expr = '+'.join([f'p[{idx}]' for idx in t_indices])
    
    for i, t_idx in enumerate(t_indices):
        # Create expression: 100 * p[t_idx] / (p[t1] + p[t2] + ...)
        expr_str = f'100*p[{t_idx}]/({sum_expr})'
        
        # Use unified error calculation
        intensity_errors[i] = calculate_expression_error(
            expr_str, t_values, t_errors, covariance_matrix, fixed_params
        )
    
    return intensities, intensity_errors


def calculate_expression_error(expr_str, parameters, errors, covariance_matrix, fixed_params=None):
    """
    Calculate error for an Expression model using covariance matrix.
    Handles the case where covariance_matrix only includes variable parameters.
    
    Args:
        expr_str: Expression string (e.g., "p[5]*2 + p[10]" or "100*p[5]/(p[5]+p[10])")
        parameters: Array of all fitted parameters
        errors: Array of parameter errors (nan for fixed parameters)
        covariance_matrix: Covariance matrix from fitting (only variable parameters)
        fixed_params: Array of indices of fixed parameters, or None (will be inferred from errors)
    
    Returns:
        float: Error (standard deviation) of the expression result
    """
    # Extract parameter indices used in expression
    import re
    param_indices = []
    for match in re.finditer(r'p\[(\d+)\]', expr_str):
        idx = int(match.group(1))
        param_indices.append(idx)
    
    param_indices = sorted(set(param_indices))
    
    if not param_indices:
        return 0.0
    
    # Determine fixed parameters (where error is nan)
    if fixed_params is None:
        fixed_params = np.array([i for i in range(len(errors)) if np.isnan(errors[i])], dtype=int)
    
    # Create mapping from full parameter index to covariance matrix index
    # Covariance matrix only includes variable (non-fixed) parameters
    variable_params = [i for i in range(len(parameters)) if i not in fixed_params]
    param_to_cov_idx = {param: cov_idx for cov_idx, param in enumerate(variable_params)}
    
    # Filter out fixed parameters from expression calculation
    variable_param_indices = [idx for idx in param_indices if idx not in fixed_params]
    
    if not variable_param_indices:
        return 0.0  # All parameters in expression are fixed
    
    # Define function for expression
    def expr_func(p):
        return eval(expr_str)
    
    # Calculate variance using covariance matrix
    variance = 0.0
    
    for i in variable_param_indices:
        for j in variable_param_indices:
            # Get covariance matrix indices
            cov_i = param_to_cov_idx.get(i)
            cov_j = param_to_cov_idx.get(j)
            
            if cov_i is None or cov_j is None:
                continue
            
            # Check bounds
            if cov_i >= covariance_matrix.shape[0] or cov_j >= covariance_matrix.shape[1]:
                continue
            
            # Numerical partial derivatives
            df_di = calculate_partial_derivative_numerical(expr_func, parameters, i)
            df_dj = calculate_partial_derivative_numerical(expr_func, parameters, j)
            
            # Add contribution from covariance
            variance += df_di * df_dj * covariance_matrix[cov_i, cov_j]
    
    return np.sqrt(abs(variance))
