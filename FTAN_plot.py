#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Module Description:
# Functions to perform basic FTAN analysis on a signal. Will output a frequency/time plot rather than a group velocity/period plot (which is the default display for FTAN analysis).
# Method:
# 1. Get real + complex signal (by taking Hilbert transform of waveform signal) -> W(t) = w(t) + w_bar(t).i (w_bar is Hilbert transform of w, the real waveform observed)
# 2. Takes FFT of W(t) -> K(w) (in frequency domain)
# 3. Specify Gaussian filter function, in frequency domain -> G(w) (based upon bandwitdth and centre frequency)
# 4. Obtain the FTAN function for a particular centre frequency: Do inverse FFT of G(w).K(w) -> S(w_H,t)
# 5. Loop over range of central frequencies to obtain 2D FTAN array S(w_H_range,t)
# To get standard output, calculate time period from frequency (T = 1/f) and group velocity from time (U = interstation distance/time).
# Calculation based on method outlined in "Seismic surface waves in a laterally inhomogeneuos medium", p133-155.

# Dependencies:
# scipy, numpy, matplotlib

# Usage:
# See end of script or jupyter-notebook for example.

# Created by Tom Hudson, 13th July 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

# Main script functions:
def ftan_plot(data, samp_rate, centre_freq_range=[], centre_freq_range_step=1.0, band_width_gau_filter=[1.25], axes=None, log=False, cmap='viridis', return_ftan_data=False):
    """
    Computes the frequency-time-analysis space of input data.
    Based on method in "Seismic surface waves in a laterally inhomogeneous 
    meduim" (Keilis-Borok 1989) (P133-155).
    
    Inputs:
    data - Seismogram (1D array)
    samp_rate - Sampling rate of the data (float)
    centre_freq_range - Lower and upper bounds of frequency range ([float, float])
    centre_freq_range_step - Size of freq. step (float)
    band_width_gau_filter - Width of Gaussian band-pass filter ([float] or 1D array of floats for all centre freq values)
    axes - Axes to plot to (matplotlib axis)
    log - Plots with log scale on y axis if true (bool)
    cmap - Color map to use
    return_ftan_data - If True, will return ftan space data (bool)
    
    Outputs:
    axes - If axes given as input, will return axes
    if return_ftan_data is True:
        S_t_domain_array - FTAN space (of shape(len(time_array), len(centre_freqs_array)))
        centre_freqs_array - Array of centre freq. values
        time_array - Array of time values
    
    """
    
    # Confirm that inputs are correct format:
    samp_rate = float(samp_rate)
    real_waveform = data
    
    # Define optional parameters if not user-defined:
    if len(centre_freq_range) ==0:
        centre_freq_range = [1.0, samp_rate/2.]
    
    # Get Hilbert transform of data:
    # (outputs analytical solution with real and imaginary parts)
    W_t_domain = hilbert(real_waveform)

    # Take FFT of W(t) to find K(w):
    K_f_domain = np.fft.fft(W_t_domain)/(len(W_t_domain)**0.5)
    # And get frequency array associated with signal:
    freqs = np.fft.fftfreq(len(W_t_domain), d=(1/samp_rate))
    freqs_rad_per_s = freqs*2*np.pi
    
    # Get FTAN output:
    centre_freqs_array = np.arange(centre_freq_range[0], centre_freq_range[1], centre_freq_range_step)
    time_array = np.linspace(0,len(real_waveform)/samp_rate,num=len(real_waveform))
    # Specify array to store output FTAN data:
    S_t_domain_array = np.zeros((len(real_waveform), len(centre_freqs_array))).astype(complex) # Array containing [time along trace x centre frequencies]
    # Loop over central frequencies:
    for i in np.arange(len(centre_freqs_array)):
        w_H = centre_freqs_array[i]*2*np.pi # Current centre frequency to work on
        # Get bandwidth for current central frequency:
        try:
            band_width_w_H_tmp = band_width_gau_filter[i]*2*np.pi # bandwidth in rad
        except:
            band_width_w_H_tmp = band_width_gau_filter[0]*2*np.pi  # bandwidth in rad
        # Get gaussian filter function in frequency domain, for specific centre frequency:
        Gauss_filt_freq_domain = np.zeros(len(freqs_rad_per_s)).astype(complex)
        for j in np.arange(len(freqs_rad_per_s)):
            Gauss_filt_freq_domain[j] = (1/(((2*np.pi)**0.5)*band_width_w_H_tmp))*np.exp(-1*((freqs_rad_per_s[j] - w_H)**2)/(2*(band_width_w_H_tmp**2)))
        # Then find FTAN values for current centre freq:
        S_f_domain = Gauss_filt_freq_domain*K_f_domain
        S_t_domain_array[:, i] = np.fft.ifft(S_f_domain)*(len(W_t_domain)**0.5) # Normalised by root(n)
            
    # Plot results:
    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes
    if log:
        ax.set_yscale('log')
    y_grid, x_grid = np.meshgrid(centre_freqs_array,time_array)
    col_mesh = ax.pcolormesh(x_grid, y_grid, np.absolute(S_t_domain_array), cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Centre frequency (Hz)")
    plt.show()
    
    # Return outputs:
    if return_ftan_data:
        if axes:
            return axes, S_t_domain_array, centre_freqs_array, time_array
        else:
            return S_t_domain_array, centre_freqs_array, time_array
    else:
        if axes:
            return axes
    
    
if __name__ == "__main__":
    
    # Example of how to use function:
    # Import data:
    import obspy
    data = obspy.read("S_waveform_E_2014_180_2037_event.m")[0].data # Data associated with trace of real waveform observed
    # And run function:
    ftan_plot(data, samp_rate=500.0, centre_freq_range=[4.0,50.0], centre_freq_range_step=0.5, band_width_gau_filter=[1.25], axes=None)