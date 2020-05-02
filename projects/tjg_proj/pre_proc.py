import pandas as pd
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt, sosfreqz

def convert_to_optical_density(fnirs_data_col):
    """
    Converts a column from it's raw wavelength to optical density value.
    """

    # get mean of col.
    data_mean = fnirs_data_col.mean()

    fnirs_data_col = fnirs_data_col / data_mean
    fnirs_data_col = np.log(fnirs_data_col)
    fnirs_data_col *= -1

    return fnirs_data_col


def beer_lambert_law(fnirs_data_col, ppf=0.1):
    """
    Convert NIRS optical density data to haemoglobin concentration.
    """

    absorb_coeffs = {
                     "760": 0.1495594, # for Oxy.
                     "840": 0.1789109, # for DeOxy.
                    }

    if "wl1" in fnirs_data_col.name:
        abs_coef = absorb_coeffs["760"]
    else:
        abs_coef = absorb_coeffs["840"]

    distance = [[3.0 for elm in fnirs_data_col]]
    distance = np.array(distance)

    EL = abs_coef * distance * ppf
    iEL = linalg.pinv(EL)
    fnirs_data_col = (fnirs_data_col * iEL[0]) * 1e-3

    return fnirs_data_col


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y
