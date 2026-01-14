import numpy as np
import plotly.express as px
import pandas as pd
from scipy.signal import lombscargle
import re
import matplotlib.pyplot as plt

def get_time_series(path, point, quantity):
    """
    Grab data column from csv file. Requires specific format.
    Args:
        path (str): path to csv file including .../file.csv
        point (str): point from which data should be read one of [p0, p1, p2, p3, p4, ...]. If the point does not exist the smallest index point that does exist will be used.
        quantity (str): One of [ux, uy, uz]

    Returns:
        np.ndarray: sampling time in seconds
        np.ndarray: requested quantity
    """
    df = pd.read_csv(path)
    t = df['time'].values   
    y = None
    old_point = point
    for i in range(10):
        if y is not None:
            break
        
        try:
            y = df[f'{point}_{quantity}'].values
        except KeyError:
            number = int(re.findall(r'-?\d*\.?\d+', point)[0]) - 1
            point = f"p{number}"
    if point != old_point:
        print(f"[INFO] Point {old_point} did not exist. Returning values for point {point}")
    return t, y


def get_frequency_lombscargle(t, y, threshold, freqs=np.linspace(0.01, 5, 20000), warmup_period=True):
    """
    Extract dominant frequency from y using Least-squares spectral analysis (Lomb-Scargle periodogram)
    Args:
        t (np.ndarray): sampling time in seconds
        y (np.ndarray): Signal
        threshold (float): 
        freqs (np.ndarray): The frequency space to search through
        warmup_period (Boolean): If true only the second half of the signal is used to extract frequencies

    Returns:
        float: Dominant frequency
    """
    if warmup_period:
        y = y[len(y)//2:]
        t = t[len(t)//2:]
    y_detrended = y - np.mean(y)

    w = 2 * np.pi * freqs

    window = np.hanning(len(y))
    y_windowed = y_detrended * window

    pgram = lombscargle(t, y_windowed, w, precenter=True)
    #add threshold, else return 0
    pgram = np.where(pgram>threshold, pgram, 0)
    if np.all(pgram <= 0): return 0

    pgram = pgram / np.max(pgram)

    return freqs[np.where(pgram == 1)][0]


def get_frequency_fourier(t, y, threshold,warmup_period=True):
    """
    Extract dominant frequency from y using interpolation and FFT
    Args:
        t (np.ndarray): sampling time in seconds
        y (np.ndarray): Signal
        warmup_period (Boolean): If true only the second half of the signal is used to extract frequencies

    Returns:
        float: Dominant frequency
    """
    if warmup_period:
        y = y[len(y)//2:]
        t = t[len(t)//2:]

    dt_avg = np.mean(np.diff(t))
    fs = 1.0 / dt_avg
    t_uniform = np.arange(t[0], t[-1], dt_avg)

    y_uniform = np.interp(t_uniform, t, y)
        
    y_detrended = y_uniform - np.mean(y_uniform)
    window = np.hanning(len(y_detrended))
    y_windowed = y_detrended * window

    n = len(y_windowed)
    yf = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(n, d=dt_avg)

    power = np.abs(yf)**2
    #add threshold, else return 0
    power = np.where(power>threshold, power, 0)
    if np.all(power <= 0): return 0

    power = power / np.max(power)
    return freqs[np.where(power == 1)][0]


def strouhal_number(shedding_freq, flow_velocity, characteristic_length):
    """
    Calculates the strouhal number
    Args:
        shedding_freq (float): Observed vortex shedding frequency
        flow_velocity (float): Domainant flow velocity (in our case inlet velocity)
        characteristic_length (float): The characteristic length of the flow phenomenon (in our case the front face height of the bar)

    Returns:
        float: Dominant frequency
    """
    return shedding_freq*characteristic_length/flow_velocity

def print_run_stats():
    """
    Prints Reynolds number, velocity, time of simulation for each run.
    """
    characteristic_length = 0.05
    kinematic_viscosity = 1.5e-5
    t_star = 600

    reynolds_numbers = [20, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 150]

    velocities = [Re * kinematic_viscosity / characteristic_length
                for Re in reynolds_numbers]

    simulation_times = [
        t_star * characteristic_length / U if U > 0 else float("inf")
        for U in velocities
    ]

    print(f"{'Re':>10} {'Velocity [m/s]':>18} {'Sim time [s] (t*=250)':>25}")
    print("-" * 55)

    for Re, U, t in zip(reynolds_numbers, velocities, simulation_times):
        print(f"{Re:10.0f} {U:18.6f} {t:25.2f}")