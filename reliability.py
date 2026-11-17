"""Weibull reliability with two unambiguous input parameter pairs."""
import math
import numpy as np

def weibull_params(mode, first, second):
    first, second = float(first), float(second)
    if not (math.isfinite(first) and math.isfinite(second) and first > 0 and second > 0):
        raise ValueError('Weibull inputs must be finite and positive')
    if mode == 'MTTF + β':
        mttf, beta = first, second
        eta = mttf / math.gamma(1 + 1 / beta)
    elif mode == 'η + β':
        eta, beta = first, second
        mttf = eta * math.gamma(1 + 1 / beta)
    else:
        raise ValueError('Choose MTTF + β or η + β')
    return {'mttf_s':mttf,'eta_s':eta,'beta':beta}

def sample_lifetime(rng, eta_s, beta):
    return float(eta_s * rng.weibull(beta))

def sample_repair(rng, mean_s, cv):
    if mean_s < 0 or cv < 0: raise ValueError('Repair mean and CV must be nonnegative')
    if mean_s == 0: return 0.0
    if cv == 0: return float(mean_s)
    sdlog = math.sqrt(math.log1p(cv*cv))
    mulog = math.log(mean_s) - sdlog*sdlog/2
    return float(rng.lognormal(mulog,sdlog))
