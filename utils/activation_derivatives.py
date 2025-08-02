# File: utils/activation_derivatives.py
import torch
import math

# --- GELU Derivatives ---
# Based on structure from Taylor-Unswift official code & Appendix A
# h_n(x) = (d/dx)^n (Gaussian PDF phi(x))
def _gelu_phi(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x**2)

def _gelu_h_recursive(x, n, h_cache):
    if n in h_cache:
        return h_cache[n]
    if n < 0: # Base cases should handle this, but for safety
        return torch.zeros_like(x)
    if n == 0:
        h_cache[0] = _gelu_phi(x)
        return h_cache[0]
    if n == 1:
        h_cache[1] = -x * _gelu_h_recursive(x, 0, h_cache)
        return h_cache[1]
    
    # h_n(x) = -x * h_{n-1}(x) - (n-1) * h_{n-2}(x)
    res = -x * _gelu_h_recursive(x, n - 1, h_cache) - (n - 1) * _gelu_h_recursive(x, n - 2, h_cache)
    h_cache[n] = res
    return res

def gelu_taylor_derivative_calculator(x_values, order):
    """
    Calculates the n-th order derivative of GELU.
    gelu_n(x) = x * Phi^(n)(x) + n * Phi^(n-1)(x)
    where Phi^(k)(x) is the k-th derivative of the Gaussian CDF.
    And Phi^(k)(x) = h_{k-1}(x) from the paper's notation.
    So, gelu_n(x) = x * h_{n-1}(x) + n * h_{n-2}(x)
    """
    if not isinstance(x_values, torch.Tensor):
        x_values = torch.tensor(x_values, dtype=torch.float32)

    if order == 0: # GELU(x) itself
        return x_values * 0.5 * (1.0 + torch.erf(x_values / math.sqrt(2.0)))
    
    h_cache = {} # For memoizing h_k(x) values for current x_values batch

    # Calculate h_{order-1}(x)
    # _gelu_h_recursive computes h_n(x) = (d/dx)^n phi(x)
    h_order_minus_1 = _gelu_h_recursive(x_values, order - 1, h_cache)
    
    # Calculate h_{order-2}(x)
    if order - 2 >= 0:
        h_order_minus_2 = _gelu_h_recursive(x_values, order - 2, h_cache)
    else: # Corresponds to Phi(x) when order=1, handled by Phi_x + x*phi_x for gelu'(x)
          # For gelu_n(x) = x*h_{n-1} + n*h_{n-2}, if n=1, term is 1*h_{-1}.
          # h_{-1} is conceptually Phi(x)
        if order == 1: # gelu'(x) = x*h_0(x) + Phi(x)
            Phi_x = 0.5 * (1.0 + torch.erf(x_values / math.sqrt(2.0)))
            return x_values * h_order_minus_1 + Phi_x # h_0 = phi(x)
        else: # Should not be reached if order > 1 and order-2 < 0
            h_order_minus_2 = torch.zeros_like(x_values) 

    return x_values * h_order_minus_1 + order * h_order_minus_2

# --- SiLU Derivatives ---
# silu(x) = x * sigmoid(x)
# silu_n(x) = x * sigmoid^(n)(x) + n * sigmoid^(n-1)(x)
def _sigmoid(x):
    return torch.sigmoid(x)

def _sigmoid_derivative_recursive(x, n, sig_cache):
    if n in sig_cache:
        return sig_cache[n]
    if n == 0:
        sig_cache[0] = _sigmoid(x)
        return sig_cache[0]
    
    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    # sigmoid^(n)(x) = d/dx (sigmoid^(n-1)(x))
    # Using general formula for (f*g)^(m) is complex.
    # Alternative from official Taylor-Unswift: h_n = sigmoid^(n)
    # h_n = - sum_{k=0 to n-1} (C(n-1, k) * h_k * h_{n-k-1}) for n >= 1, with h_1 = h_0(1-h_0)
    # This seems to be for d^n/dx^n (sigmoid(x)*(1-sigmoid(x))), not sigmoid^(n) directly.
    # Let's use a simpler direct approach for a few orders or use autograd for higher ones if pure impl. is too complex.
    # For now, let's ensure the structure allows it.
    # The Taylor-Unswift code uses a specific recursive def for h_n(x) = sigma^(n)(x):
    # h_n(x) = - sum_{k=0}^{n-1} C(n-1, k) * h_k(x) * h_{n-k-1}(x) * (-1)^{n-k-1}, where h_0 = sigma. This is error prone.
    # Let's refer to their actual implementation:
    # `h_n(x) = sigma_fn(x) * (1-sigma_fn(x))` for `n=1`
    # `h_n(x) = h_1(x) * (1-2*sigma_fn(x))` for `n=2`
    # Higher orders become complicated to hardcode.
    # A common way is to express derivatives of sigmoid in terms of sigmoid itself.
    # sigma' = sigma * (1-sigma)
    # sigma'' = sigma' * (1-sigma) - sigma * sigma' = sigma'(1-2sigma)
    # sigma''' = sigma''(1-2sigma) - 2sigma*sigma'
    # This recursive structure is what's typically used.

    if n == 1: # sigma'
        s = _sigmoid_derivative_recursive(x, 0, sig_cache)
        sig_cache[1] = s * (1 - s)
        return sig_cache[1]
    
    # For n > 1, compute d/dx of sigma^(n-1)
    # This requires autograd or more complex symbolic expansion.
    # For practical N_taylor (e.g., up to 8-10), explicit forms can be derived or autograd used.
    # The official repo has explicit forms up to n=5 for SiLU derivatives (which involves sigmoid derivatives).
    # Given the complexity, we might limit N_taylor for SiLU or use simpler pre-derived forms.
    # For this example, we'll implement up to n=2 for sigmoid derivative for simplicity.
    if n == 2: # sigma''
        s = _sigmoid_derivative_recursive(x, 0, sig_cache)
        s_prime = _sigmoid_derivative_recursive(x, 1, sig_cache)
        sig_cache[2] = s_prime * (1 - 2 * s)
        return sig_cache[2]

    # Placeholder for higher orders - consider using torch.autograd.functional.jacobian for exactness if needed
    # or pre-calculating symbolic forms.
    # For now, returning zero for higher orders not explicitly handled.
    print(f"Warning: SiLU sigmoid derivative order {n} not fully implemented, returning 0.")
    return torch.zeros_like(x)


def silu_taylor_derivative_calculator(x_values, order):
    if not isinstance(x_values, torch.Tensor):
        x_values = torch.tensor(x_values, dtype=torch.float32)

    if order == 0: # SiLU(x) itself
        return x_values * torch.sigmoid(x_values)

    sig_cache = {}
    # silu_n(x) = x * sigmoid^(n)(x) + n * sigmoid^(n-1)(x)
    sigmoid_n = _sigmoid_derivative_recursive(x_values, order, sig_cache)
    
    if order - 1 >= 0:
        sigmoid_n_minus_1 = _sigmoid_derivative_recursive(x_values, order - 1, sig_cache)
    else: # Should not happen for order=0 (handled) or order=1 (order-1=0)
        sigmoid_n_minus_1 = torch.zeros_like(x_values) # Should be _sigmoid(x) if order=1
        if order == 1:
            sigmoid_n_minus_1 = _sigmoid_derivative_recursive(x_values, 0, sig_cache)


    return x_values * sigmoid_n + order * sigmoid_n_minus_1


ACTIVATION_DERIVATIVES_CALCULATORS = {
    'gelu': gelu_taylor_derivative_calculator,
    'silu': silu_taylor_derivative_calculator, # Note: SiLU derivatives are simplified here
}