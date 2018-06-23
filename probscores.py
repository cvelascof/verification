"""Evaluation and skill scores for probabilistic forecasts."""

import numpy as np

def CRPS(X_f, X_o):
    """Compute the average continuous ranked probability score (CRPS) for a set 
    of forecast ensembles and the corresponding observations. This implementation 
    is adapted from 
    
    H. Hersbach. Decomposition of the Continuous Ranked Probability Score for 
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

def ROC_curve_init(X_min, n_prob_thrs=10):
  """Initialize a ROC curve object.
  
  Parameters
  ----------
  X_min : float
    Precipitation intensity value for yes/no prediction.
  n_prob_thrs : int
    The number of probability thresholds to use. The interval [0,1] is divided 
    into n_prob_thrs evenly spaced values.
  
  Returns
  -------
  out : dict
    The ROC curve object.
  """
  ROC = {}
  
  ROC["X_min"]        = X_min
  ROC["hits"]         = np.zeros(n_prob_thrs, dtype=int)
  ROC["misses"]       = np.zeros(n_prob_thrs, dtype=int)
  ROC["false_alarms"] = np.zeros(n_prob_thrs, dtype=int)
  ROC["corr_neg"]     = np.zeros(n_prob_thrs, dtype=int)
  ROC["prob_thrs"]    = np.linspace(0.0, 1.0, n_prob_thrs)
  
  return ROC

def ROC_curve_accum(ROC, P_f, X_o):
    """Accumulate the given probability-observation pairs into the given ROC 
    object.
    
    Parameters
    ----------
    ROC : dict
      A ROC curve object created with ROC_curve_init.
    P_f : array_like
      Forecasted probabilities for exceeding the threshold specified in the ROC 
      object. Non-finite values are ignored.
    X_o : array_like
      Observed values. Non-finite values are ignored.
    """
    mask = np.logical_and(np.isfinite(P_f), np.isfinite(X_o))
    
    P_f = P_f[mask]
    X_o = X_o[mask]
    
    for i,p in enumerate(ROC["prob_thrs"]):
        ROC["hits"][i]         += np.sum(np.logical_and(P_f >= p, X_o >= ROC["X_min"]))
        ROC["misses"][i]       += np.sum(np.logical_and(P_f <  p, X_o >= ROC["X_min"]))
        ROC["false_alarms"][i] += np.sum(np.logical_and(P_f >= p, X_o <  ROC["X_min"]))
        ROC["corr_neg"][i]     += np.sum(np.logical_and(P_f <  p, X_o <  ROC["X_min"]))

def ROC_curve_compute(ROC, compute_area=False):
    """Compute the ROC curve and its area from the given ROC object.
    
    Parameters
    ----------
    ROC : dict
      A ROC curve object created with ROC_curve_init.
    compute_area : bool
      If True, compute the area under the ROC curve (between 0.5 and 1).
    
    Returns
    -------
    out : tuple
      A two-element tuple containing the probability of detection (POD) and 
      probability of false detection (POFD) for the probability thresholds 
      specified in the ROC curve object. If compute_area is True, return the 
      area under the ROC curve as the third element of the tuple.
    """
    POD_vals  = []
    POFD_vals = []
    
    for i in xrange(len(ROC["prob_thrs"])):
        POD_vals.append(1.0*ROC["hits"][i] / (ROC["hits"][i] + ROC["misses"][i]))
        POFD_vals.append(1.0*ROC["false_alarms"][i] / \
                         (ROC["corr_neg"][i] + ROC["false_alarms"][i]))
    
    if compute_area:
        # Compute the total area of parallelepipeds under the ROC curve.
        area = (1.0 - POFD_vals[0]) * (1.0 + POD_vals[0]) / 2.0
        for i in range(len(ROC["prob_thrs"])-1):
          area += (POFD_vals[i] - POFD_vals[i+1]) * (POD_vals[i+1] + POD_vals[i]) / 2.0
        area += POFD_vals[-1] * POD_vals[-1] / 2.0
        
        return POFD_vals,POD_vals,area
    else:
        return POFD_vals,POD_vals
