"""
Generate synthetic seismic arrivals from a known hypocenter.

Travel times are computed using the Python TRVDRV so that the
forward problem is identical for both the Python and Fortran locators.
The Fortran should then reproduce the input hypocenter to within
numerical precision.
"""

import numpy as np
from obspy import UTCDateTime

from hypo71py.model.station_phase import Station, PhasePick
from hypo71py.model.velocity_model import CrustalVelocityModel
from hypo71py.core.single import CALC_EPICENTRAL_XY_OFFSET
from hypo71py.core.trvdrv import TRVDRV


def make_stations_ring(center_lon, center_lat, n=6, radius_km=40.0):
    """
    Place n stations evenly spaced around a circle of radius_km km
    centred on (center_lon, center_lat).

    Returns a list of Station objects with codes STA1..STAn, elevation 0.
    """
    # Approximate degrees per km at this latitude
    deg_per_km_lat = 1.0 / 111.2
    deg_per_km_lon = 1.0 / (111.2 * np.cos(np.radians(center_lat)))

    stations = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        dlat = radius_km * np.cos(angle) * deg_per_km_lat
        dlon = radius_km * np.sin(angle) * deg_per_km_lon
        code = f'ST{i+1:02d}'
        stations.append(Station(code,
                                lon=center_lon + dlon,
                                lat=center_lat + dlat,
                                elevation=0.0,
                                depth=0.0))
    return stations


def make_synthetic_picks(true_lon, true_lat, true_depth_km,
                         true_origin, stations, velocity_model,
                         include_s=False, pos=1.73):
    """
    Compute exact synthetic P (and optionally S) arrival times from a
    known hypocenter using TRVDRV.

    Parameters
    ----------
    true_lon, true_lat : float
        Epicentre in decimal degrees.
    true_depth_km : float
        Focal depth (km).
    true_origin : UTCDateTime
        Origin time.
    stations : list of Station
    velocity_model : CrustalVelocityModel
    include_s : bool
        Whether to also compute S arrivals.
    pos : float
        Vp/Vs ratio used for S travel times.

    Returns
    -------
    picks : dict  {station_code: {'P': UTCDateTime, 'S': UTCDateTime|None}}
    """
    lons = np.array([s.lon for s in stations], dtype='f')
    lats = np.array([s.lat for s in stations], dtype='f')
    elevs = np.array([-s.elevation for s in stations], dtype='f')   # TRVDRV sign

    DX, DY = CALC_EPICENTRAL_XY_OFFSET(true_lon, true_lat, lons, lats)
    delta = np.sqrt(DX**2 + DY**2) + 1e-6

    n = len(stations)
    # VIDXS=0 for P, =1 for S
    vidxs_p = np.zeros(n, dtype=int)
    vidxs_s = np.ones(n, dtype=int)

    T_p, _, _ = TRVDRV(velocity_model, true_depth_km, delta, elevs, vidxs_p)

    T_s = None
    if include_s:
        T_s, _, _ = TRVDRV(velocity_model, true_depth_km, delta, elevs, vidxs_s)

    picks = {}
    for i, sta in enumerate(stations):
        p_arr = true_origin + float(T_p[i])
        s_arr = (true_origin + float(T_s[i])) if T_s is not None else None
        picks[sta.code] = {
            'P': PhasePick('P', p_arr, station_code=sta.code),
            'S': (PhasePick('S', s_arr, station_code=sta.code)
                  if s_arr is not None else None),
        }

    return picks


def uniform_halfspace(vp_km_s=6.0, vp_vs_ratio=1.73):
    """
    Return a uniform half-space CrustalVelocityModel.

    TRVDRV requires at least two layers (it needs thicknesses), so we add a
    deep dummy layer with identical velocity.  The second layer is never
    reached for the shallow events used in tests.
    """
    vs = vp_km_s / vp_vs_ratio
    return CrustalVelocityModel(
        depths=np.array([0.0, 100.0]),
        VP=np.array([vp_km_s, vp_km_s]),
        VS=np.array([vs, vs]),
        name='uniform_halfspace',
    )


def two_layer_model(vp1=6.0, vp2=7.5, interface_km=15.0, vp_vs_ratio=1.73):
    """Return a two-layer CrustalVelocityModel."""
    return CrustalVelocityModel(
        depths=np.array([0.0, interface_km]),
        VP=np.array([vp1, vp2]),
        VS=np.array([vp1 / vp_vs_ratio, vp2 / vp_vs_ratio]),
        name='two_layer',
    )
