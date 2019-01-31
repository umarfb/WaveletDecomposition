import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import os
from astropy.io import ascii

# Method to get GPR lightcurve fit data
def get_gprlc(path, filename):
    lc_fit = ascii.read(path + filename)
    epoch_fit = lc_fit['MJD']
    mag_fit = lc_fit['mag_pred']
    mag_fit_sigma = lc_fit['mag_pred_sigma']
    
    sn_name = filename.strip('_gpr_fit.csv')
    
    return epoch_fit, mag_fit, mag_fit_sigma, sn_name

# Method to get GPR fit parameters
def get_gprparams(path, filename):
    params_tab = ascii.read(path + filename)
    parameters = params_tab['value']
    
    theta = parameters[0:-3]
    sn_type = parameters[-2]
    sn_filter = parameters[-1]
    
    return theta, sn_type, sn_filter

# Method to do 2-level wavelet decomposition, returns wavelet coefficients
def get_wavedec(x_in, y_in):
    coeffs = pywt.wavedec2([x_in, y_in], 'sym2', level=2)
    
    cA2 = coeffs[0]
    cH2, cV2, cD2 = coeffs[1][0], coeffs[1][1], coeffs[1][2]
    cH1, cV1, cD1 = coeffs[2][0], coeffs[2][1], coeffs[2][2]
    
    coeffs_vector = [cA2, cH2, cV2, cD2, cH1, cV1, cD1]
    
    return coeffs_vector

# Method to flatten vector of coefficients
def flatten_coeffs(coeff_vector):
    flattened_arr = []
    
    for coeff in coeff_vector:
        flat_coeffs = coeff.flatten()
        flattened_arr.extend(flat_coeffs)
    
    return flattened_arr

# Method to create a dictionary of coeff labels and values
def mk_coeff_dict(coeff_vector):
    len_cv = len(coeff_vector)
    
    coeff_labels = []
    for i in range(len_cv):
        coeff_labels.append('coeff{0}'.format(i))
    
    coeff_dict = {}
    
    for j, label in enumerate(coeff_labels):
        coeff_dict[label] = coeff_vector[j]
    
    return coeff_dict

# path to GPR fit data
gprfit_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/lightcurve_GPR/GPR_fits/'
# path to GPR parameters
gprparams_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/lightcurve_GPR/GPR_params/'

# Get list of GPR fit data files
gpr_lc_list = os.listdir(gprfit_path)

# Get list of GPR parameter data files
gpr_params_list = os.listdir(gprparams_path)

# Create dataframe containing the wavelet coefficients of all SNe
sn_features = []

for sn_lc in gpr_lc_list:
    x_fit, y_fit, y_fit_err, sn_name = get_gprlc(gprfit_path, sn_lc)
    #print('Wavelet decomposition for {0}'.format(sn_name))
    sn_coeffs = get_wavedec(x_fit, y_fit)
    sn_coeff_vector = flatten_coeffs(sn_coeffs)
    sn_coeff_dict = mk_coeff_dict(sn_coeff_vector)
    
    sn_features.append(sn_coeff_dict)

sn_features = pd.DataFrame(sn_features)
sn_features

# Write SN wavelet coefficients to .csv file
sn_features.to_csv('sn_wave_coeffs.csv', header=True, index=False)