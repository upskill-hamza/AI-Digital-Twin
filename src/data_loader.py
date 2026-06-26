import numpy as np
import pandas as pd
import xarray as xr
import os

def read_imd_binary(file_path, header_size=6, dtype=np.float32):
    """Read entire binary file, skip header, return flat array."""
    data = np.fromfile(file_path, dtype=dtype)
    if data.size < header_size:
        raise ValueError(f"File {file_path} too small, header_size may be wrong.")
    return data[header_size:]

def detect_grid(file_path, expected_ndays, header_size=6):
    """Figure out (nlat, nlon) from file size and expected number of days."""
    data = read_imd_binary(file_path, header_size)
    total_points = data.size
    points_per_day = total_points / expected_ndays
    if points_per_day != int(points_per_day):
        # Try with header_size 0 in case there is no header
        data_full = np.fromfile(file_path, dtype=np.float32)
        points_per_day = data_full.size / expected_ndays
        if points_per_day != int(points_per_day):
            raise ValueError(f"Cannot divide evenly: total points {total_points}, ndays {expected_ndays}")
        points_per_day = int(points_per_day)
        print(f"Warning: used header_size=0 for {file_path}")
    points_per_day = int(points_per_day)
    # Common grid shapes
    common = {
        'rainfall': (129, 135),   # lat, lon for 0.25°
        'temperature': (32, 32)   # for 1° (can be 32x32 or 33x35)
    }
    # Try automatic factorisation
    for i in range(int(np.sqrt(points_per_day)), 0, -1):
        if points_per_day % i == 0:
            return i, points_per_day // i
    raise RuntimeError(f"Cannot determine grid shape for {points_per_day} points")

def load_year(file_path, year, var_type='rainfall', header_size=6):
    """Load one year of daily grids into an xarray DataArray."""
    # Number of days
    ndays = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365

    nlat, nlon = detect_grid(file_path, ndays, header_size)

    data = read_imd_binary(file_path, header_size)
    grids = data.reshape(ndays, nlat, nlon)
    # Replace common fill values
    grids = np.where(grids < -900, np.nan, grids)

    dates = pd.date_range(start=f'{year}-01-01', periods=ndays, freq='D')

    # Approximate coordinates
    if 'rain' in var_type.lower():
        lat = np.linspace(6.5, 38.5, nlat)
        lon = np.linspace(66.5, 100.5, nlon)
    else:  # temperature
        lat = np.linspace(6.5, 38.5, nlat)
        lon = np.linspace(66.5, 100.5, nlon)

    da = xr.DataArray(
        grids,
        dims=['time', 'lat', 'lon'],
        coords={'time': dates, 'lat': lat, 'lon': lon},
        name=var_type
    )
    return da