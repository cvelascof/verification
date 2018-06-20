"""Evaluation and skill scores for probabilistic forecasts."""

import numpy as np

def CRPS(X_f, X_o):
  """Compute the average continuous ranked probability score (CRPS) for a set 
  of forecast ensembles and the corresponding observations. This implementation 
  is adapted from 
  
  H. Hersbach:  Decomposition of the Continuous Ranked Probability Score for 
  Ensemble Prediction Systems, Weather and Forecasting, 15(5), 559-570, 2000.
  
  Parameters
  ----------
  X_f : array_like
    Array of shape (n,m) containing n ensembles of forecast values with each 
    ensemble having m members.
  X_o : array_like
    Array of n observed values.
  
  Returns
  -------
  out : float
    The continuous ranked probability score.
  """
  mask = np.logical_and(np.all(np.isfinite(X_f), axis=1), np.isfinite(X_o))
  
  X_f = X_f[mask, :].copy()
  X_f.sort(axis=1)
  X_o = X_o[mask]
  
  n = X_f.shape[0]
  m = X_f.shape[1]
  
  alpha = np.zeros((n, m+1))
  beta  = np.zeros((n, m+1))
  
  for i in xrange(1, m):
    mask = X_o > X_f[:, i]
    alpha[mask, i] = X_f[mask, i] - X_f[mask, i-1]
    beta[mask, i]  = 0.0
    
    mask = np.logical_and(X_f[:, i] > X_o, X_o > X_f[:, i-1])
    alpha[mask, i] = X_o[mask] - X_f[mask, i-1]
    beta[mask, i]  = X_f[mask, i] - X_o[mask]
    
    mask = X_o < X_f[:, i-1]
    alpha[mask, i] = 0.0
    beta[mask, i]  = X_f[mask, i] - X_f[mask, i-1]
  
  mask = X_o < X_f[:, 0]
  alpha[mask, 0] = 0.0
  beta[mask, 0]  = X_f[mask, 0] - X_o[mask]
  
  mask = X_f[:, -1] < X_o
  alpha[mask, -1] = X_o[mask] - X_f[mask, -1]
  beta[mask, -1]  = 0.0
  
  p = 1.0*np.arange(m+1) / m
  res = np.sum(alpha*p**2.0 + beta*(1.0-p)**2.0, axis=1)
  
  return np.mean(res)
