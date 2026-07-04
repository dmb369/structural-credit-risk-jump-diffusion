"""
Merton (1974) structural credit risk model — GBM baseline.

Core idea: equity is a call option on firm assets, struck at the debt (default
point). Given observed equity value E and equity volatility sigma_E, we solve
for the UNOBSERVED asset value A and asset volatility sigma_A that are
consistent with Black-Scholes option pricing:

    E = A * N(d1) - D * exp(-rT) * N(d2)          ... (1) equity as call option
    sigma_E * E = N(d1) * sigma_A * A               ... (2) Ito's lemma / delta relation

where:
    d1 = [ln(A/D) + (r + 0.5*sigma_A^2)*T] / (sigma_A*sqrt(T))
    d2 = d1 - sigma_A*sqrt(T)

Once (A, sigma_A) are solved, Distance-to-Default and PD follow directly.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


def _merton_equations(x, E, sigma_E, D, r, T):
    """System of 2 equations in 2 unknowns: asset value A, asset vol sigma_A."""
    A, sigma_A = x
    if A <= 0 or sigma_A <= 0:
        return [1e10, 1e10]  # penalize invalid guesses so fsolve avoids them

    d1 = (np.log(A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)

    eq1 = A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_A * A - sigma_E * E

    return [eq1, eq2]


def calibrate_merton(E, sigma_E, D, r, T, A_guess=None, sigma_A_guess=None):
    """
    Solve for implied asset value (A) and asset volatility (sigma_A).

    Parameters
    ----------
    E : float           market value of equity
    sigma_E : float     annualized equity volatility (from historical returns)
    D : float           default point (debt threshold)
    r : float           risk-free rate (annualized)
    T : float           horizon in years (e.g. 1.0 for 1-year PD)

    Returns
    -------
    (A, sigma_A) : tuple of floats
    """
    if A_guess is None:
        A_guess = E + D  # naive starting point: assets ~ equity + debt
    if sigma_A_guess is None:
        sigma_A_guess = sigma_E * E / (E + D)  # naive de-levered vol guess

    sol, info, ier, msg = fsolve(
        _merton_equations, x0=[A_guess, sigma_A_guess],
        args=(E, sigma_E, D, r, T), full_output=True
    )

    A, sigma_A = sol
    converged = (ier == 1) and A > 0 and sigma_A > 0
    return A, sigma_A, converged


def distance_to_default(A, sigma_A, D, r, T):
    """
    DD = number of standard deviations the (log) asset value is above the
    default point at horizon T, under the risk-neutral GBM drift.
    """
    dd = (np.log(A / D) + (r - 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    return dd


def probability_of_default(A, sigma_A, D, r, T):
    """PD = N(-DD). Under GBM, tail risk is thin -> PD collapses fast as DD grows."""
    dd = distance_to_default(A, sigma_A, D, r, T)
    return norm.cdf(-dd), dd
