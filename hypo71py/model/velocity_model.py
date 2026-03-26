"""
Velocity model for use with hypo71.

Contains most of the functionality in the TRVDRV subroutine of hypo71.f
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

__all__ = [
    "CrustalVelocityModel",
]




class CrustalVelocityModel:
    """
    1D layered velocity model

    :param depths:
        depth of layer tops (in km)
    :param VP:
        P-wave veolicities (in km/s)
    :param VS:
        S-wave velocities (in km/s)
    :param name:
        str, model name
    """
    def __init__(self, depths, VP, VS, name=''):
        assert len(depths) == len(VP) == len(VS)
        self.depths = np.asarray(depths, dtype='f')
        self.VP = np.asarray(VP, dtype='f')
        self.VS = np.asarray(VS, dtype='f')
        self.name = name

        ## Layer thicknesses (excluding halfspace)
        self.thicknesses = np.diff(depths)

        ## Refraction matrices
        self.RXTT = None
        self.RXHD = None

    def __len__(self):
        return len(self.depths)

    def __repr__(self):
        txt = '<CrustalVelocityModel "%s" (n=%d)>'
        txt %= (self.name, self.num_layers)
        return txt

    @property
    def num_layers(self):
        """
        Number of layers, excluding halfspace
        """
        return len(self) - 1

    @property
    def max_depth(self):
        """
        Maximum depth
        """
        return self.depths[-1]

    def recalc_vs_from_vp(self, vp_vs_ratio):
        """
        Compute and set VS based on VP and VP/VS ratio

        :param vp_vs_ratio:
            float, VP/VS ratio
        """
        self.VS = self.VP / vp_vs_ratio

    def estimate_densities_from_vs(self):
        """
        Compute densities based on shear-wave velocities
        Adapted from SITE_AMP by David Boore, but using effective VP values
        instead of VP estimated from VS.

        :return:
            1-D float array, densities in kg/m**3
        """
        VS = self.VS

        Rho = np.zeros_like(VS)

        Rho[VS < 0.3] = 1.93

        VS2 = VS[VS >= 0.3]
        Rho2 = Rho[VS >= 0.3]
        #VP2 = 0.9409 + 2.0947*VS2 -0.8206*VS2**2 + 0.2683*VS2**3 - 0.0251*VS2**4
        VP2 = self.VP[VS >= 0.3]
        Rho2[VS2 < 3.55] = 1.74 * VP2[VS2 < 3.55]**0.25
        VP3 = VP2[VS2 >= 3.55]
        Rho2[VS2 >= 3.55] = (1.6612*VP3 - 0.4721*VP3**2 + 0.0671*VP3**3
                            - 0.0043*VP3**4 + 0.000106*VP3**5)
        Rho[VS >= 0.3] = Rho2

        return Rho * 1000

    def get_velocities(self, wave='P'):
        """
        Get velocities corresponding to given wave type

        :param wave:
            char ('P' or 'S') or int (0 = P, 1 = S)
            (default: 'P')

        :return:
            1D array, velocities in km/s [NL + 1]
        """
        if wave == 0:
            V = self.VP
        elif wave == 1:
            V = self.VS
        elif wave.upper() == 'P':
            V = self.VP
        elif wave.upper() == 'S':
            V = self.VS

        return V

    def get_velocity_ratios(self, wave='P'):
        """
        Compute ratio between layer velocity and velocity in layer below

        :param wave:
            see :meth:`get_velocities`

        :return:
            1D array [NL]
        """
        V = self.get_velocities(wave=wave)
        Vratios = V[:-1] / V[1:]

        return Vratios

    def get_velocity_at_depth(self, depth, wave='P'):
        """
        Get velocity at particular depth

        :param depth:
            float, depth (in km)
        :param wave:
            see :meth:`get_velocities`

        :return:
            float, velocity (in km/s)
        """
        idx = self.get_layer_index(depth)
        return self.get_velocities(wave=wave)[idx]

    def calc_critical_angles(self, wave='P', deg=False):
        """
        Compute critical angles in each layer for given wave type

        :param wave:
            see :meth:`get_velocities`
        :param deg:
            bool, whether or not to return angles as degrees
            (default: False, will return as radians)

        :return:
            1D array, critical angles [NL]
        """
        angles = np.arcsin(self.get_velocity_ratios(wave=wave))
        if deg:
            angles = np.degrees(angles)

        return angles

    def calc_travel_times(self, layer_angles, wave='P'):
        """
        Compute travel times in each layer corresponding to given
        layer angles and wave type

        :param layer_angles:
            float or 1D array, angles in radians [NL or less]
            If array and length is smaller than number of layers in model,
            bottom layers are supposed to be missing, and the result
            will be truncated to the same length
        :param wave:
            see :meth:`get_velocities`

        :return:
            1D array, travel times in seconds, with length:
            - NL if :param:`layer_angles` is scalar
            - length of layer_angles array.
        """
        if np.isscalar(layer_angles):
            NL = self.num_layers
        else:
            NL = len(layer_angles)
        V = self.get_velocities(wave=wave)[:NL]
        THK = self.thicknesses[:NL]

        DIS = THK / np.cos(layer_angles)
        TIM = DIS / V
        return TIM

    def calc_horizontal_distances(self, layer_angles):
        """
        Compute horizontal distances in each layer corresponding to
        given layer angles

        :param layer_angles:
            see :meth:`calc_travel_times`

        :return:
            1D array, horizontal distances in km, with length:
            - NL if :param:`layer_angles` is scalar
            - length of layer_angles array.
        """
        NL = len(layer_angles)
        THK = self.thicknesses[:NL]

        X = THK * np.tan(layer_angles)
        return X

    def calc_layer_angles_above(self, bottom_layer_idx, bottom_angle, wave='P'):
        """
        Compute layer angles in higher layers for a ray with a given
        angle in the layer below

        :param bottom_layer_idx:
            int, index of bottom layer
        :param bottom_angle:
            float, angle in bottom layer (in radians)
        :param wave:
            see :meth:`get_velocities`

        :return:
            1D array, with length equal to :param:`bottom_layer_idx` + 1,
            i.e., bottom layer is included
        """
        if bottom_layer_idx < 0:
            bottom_layer_idx = self.num_layers + bottom_layer_idx
        assert bottom_layer_idx <= self.num_layers

        # TODO: should we check if angles remain below critical angle
        # in each layer?
        Vratios = self.get_velocity_ratios(wave=wave)[:bottom_layer_idx]
        layer_angles = [bottom_angle]
        for L in range(bottom_layer_idx - 1, -1, -1):
            angle = np.arcsin(Vratios[L] * np.sin(layer_angles[-1]))
            layer_angles.append(angle)

        return np.array(layer_angles[::-1])

    def get_layer_index(self, Z):
        """
        Find index of layer containing a particular depth

        Note: if Z is smaller than top depth in model, returned index
        is zero.

        :param Z:
            float, depth (in km)

        :return:
            int, layer index
        """
        try:
            return np.where(self.depths <= Z)[0][-1]
        except:
            ## Z is above top depth (e.g., station at altitude > 0), return 0
            return 0

    def calc_refraction_matrices(self, recalc=False):
        """
        Compute cumulative travel times and horizontal distances
        for downgoing rays from any top layer critically refracting
        in any bottom layer

        :param recalc:
            bool, whether or not to force recalculating the matrices
            (default: False)

        :return:
            (RXTT, RXHD) tuples of 3D arrays [2, NL, NL], with dimensions
            corresponding to [wave types, top layers, bottom layers]
            - RXTT: cumulative travel times
            - RXHD: cumulative horizontal distances
            Note: these matrices are calculated only once, and stored in
            :prop:`RXTT` and :prop:`RXHD`
        """
        if self.RXTT is None or self.RXHD is None or recalc:
            NL = self.num_layers
            self.RXTT = np.zeros((2, NL, NL))
            self.RXHD = np.zeros((2, NL, NL))
            for W, wave in enumerate(('P', 'S')):
                theta_crit = self.calc_critical_angles(wave=wave)
                for BL in range(NL):
                    bottom_angle = theta_crit[BL]
                    layer_angles = self.calc_layer_angles_above(BL, bottom_angle,
                                                                wave=wave)
                    travel_times = self.calc_travel_times(layer_angles, wave=wave)
                    cumul_travel_times = np.cumsum(travel_times[::-1])[::-1]
                    hdistances = self.calc_horizontal_distances(layer_angles)
                    cumul_hdistances = np.cumsum(hdistances[::-1])[::-1]
                    self.RXTT[W, :BL+1, BL] = cumul_travel_times
                    self.RXHD[W, :BL+1, BL] = cumul_hdistances

        return (self.RXTT, self.RXHD)

    def calc_downgoing_ray_tt_and_hd(self, Z, wave='P'):
        """
        Compute cumulative travel times / horizontal distances for
        a downgoing ray originating at depth Z, critically refracting
        at different bottom layers.

        :param Z:
            float, depth at which ray originates (in km)
        :param wave:
            see :meth:`get_velocities`

        :return:
            (rxtt, rxhd) tuple of 1D arrays [NL]
            - rxtt: cumulative travel times (in seconds)
            - rxhd: cumulative horizontal distances (in km)
            Note: values in layers above Z are zero
        """
        RXTT, RXHD = self.calc_refraction_matrices()
        if wave in (0, 1):
            W = wave
        elif wave.upper() == 'P':
            W = 0
        elif wave.upper() == 'S':
            W = 1
        RXTT, RXHD = RXTT[W], RXHD[W]

        TL = self.get_layer_index(Z)
        NL = self.num_layers
        if TL >= NL:
            rxtt = np.zeros(NL) * np.nan
            rxhd = np.zeros(NL) * np.nan
        else:
            ## Correct for depth inside layer
            ## (works also for negative depths inside layer, which are treated
            ## as lying above the layer, but with the same velocities)
            vdl = Z - self.depths[TL]
            thkl = self.thicknesses[TL]
            ## Note: take copy, otherwise original RXTT and RXHD arrays are modified!
            rxtt, rxhd = RXTT[TL].copy(), RXHD[TL].copy()
            if TL < NL-1:
                ## Time / distance in layer in which Z is situated
                ## (for different bottom layers)
                ttl = RXTT[TL] - RXTT[TL+1]
                hdl = RXHD[TL] - RXHD[TL+1]
            elif TL == NL - 1:
                ttl, hdl = RXTT[TL], RXHD[TL]
            idxs = (rxtt > 0)
            rxtt[idxs] -= (ttl[idxs] * vdl/thkl)
            rxhd[idxs] -= (hdl[idxs] * vdl/thkl)

        return (rxtt, rxhd)

    def calc_refwav_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Compute travel times for refracted waves in different bottom
        layers between focus at depth Zf and station at depth Zs,
        separated by epicentral distance Repi.

        :param Zf:
            float, depth of focus (in km)
        :param Zs:
            float, depth of station (in km)
        :param Repi:
            float, epicentral distance (= horizontal distance between
            earthquake focus and receiver station
        :param wave:
            see :meth:`get_velocities`

        :return:
            1D array, travel times corresponding to different bottom
            layers (in seconds) [NL]
            Note: travel times in layers above the maximum of Zf and Zs
            are set to NaN. If both Zf and Zs are situated in the half-
            space, only NaN values will be returned.
        """
        rxttf, rxhdf = self.calc_downgoing_ray_tt_and_hd(Zf, wave=wave)
        rxtts, rxhds = self.calc_downgoing_ray_tt_and_hd(Zs, wave=wave)
        V = self.get_velocities(wave)

        NL = self.num_layers
        Lf = self.get_layer_index(Zf)
        Ls = self.get_layer_index(Zs)
        ## highest possible bottom layer
        BLmin = max(Lf, Ls)

        ## Limit BLmin to NL - 1
        #if np.isclose(max(Zf, Zs), self.max_depth):
        #	BLmin = NL - 1
        ## Capture case where lower of Zf or Zs corresponds to layer interface
        if np.isclose(max(Zf, Zs), self.depths).any():
            BLmin -= 1

        refwav_tt = np.zeros(NL) * np.nan
        if BLmin < NL:
            ## If Zf and/or Zs are in halfspace, there is no refracted wave
            #if not (Lf == NL and Ls == NL):
                for BL in range(BLmin, NL):
                    VTT = rxttf[BL] + rxtts[BL]
                    ## Distance traveled horizontally along top of layer below
                    HD = Repi - (rxhdf[BL] + rxhds[BL])
                    if HD >= 0:
                        HTT = HD / V[BL+1]
                        refwav_tt[BL] = VTT + HTT

        return refwav_tt

    def calc_min_refwav_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Compute minimum travel time for a refracted wave between focus
        at depth Zf and station at depth Zs, separated by epicentral
        distance Repi.

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`calc_refwav_tt`

        :return:
            (BL, tmin) tuple:
            - BL: int, bottom layer index
            - tmin: float, minimum travel time in seconds
            May be (None, np.inf) if refraction is not possible
        """
        refwav_tt = self.calc_refwav_tt(Zf, Zs, Repi, wave=wave)
        if not np.isnan(refwav_tt).all():
            BL = np.nanargmin(refwav_tt)
            tmin = refwav_tt[BL]
        else:
            BL, tmin = None, np.inf
        return (BL, tmin)

    def constrain_depth_range(self, Zh, Zl):
        """
        Constrain model between upper and lower depth

        Notes:
        - if Zh is smaller than top depth of first layer,
        thickness of top layer will be increased
        - if Zl is larger than top of halfspace, an additional
        layer will be added with the same velocities as the halfspace

        :param Zh:
            float, upper depth (km)
        :param Zl:
            float, lower depth (km)

        :return:
            instance of :class:`VelocityModel`
        """
        Lh = self.get_layer_index(Zh)
        Ll = self.get_layer_index(Zl)
        ## Avoid bottom layer with zero thickness!
        if self.depths[Ll] == Zl:
            Ll -= 1
        ## Add layer in halfspace if Zl > max_depth
        if Zl > self.max_depth:
            depths = np.hstack([self.depths, [Zl]])
            VP = np.hstack([self.VP, [self.VP[-1]]])
            VS = np.hstack([self.VS, [self.VS[-1]]])
        else:
            depths, VP, VS = self.depths, self.VP, self.VS

        depths2 = depths[Lh:Ll+2] - depths[Lh]
        depths2[1:] -= (Zh - depths[Lh])
        depths2[-1] = Zl - Zh
        VP2 = VP[Lh:Ll+2]
        VS2 = VS[Lh:Ll+2]

        return self.__class__(depths2, VP2, VS2)

    def find_reflection_ray_angles_and_tts(self, Zf, Zs, Repi, wave='P'):
        """
        Determine reflection angles and travel times of reflected rays
        in different bottom layers between focus at depth Zf and station
        at depth Zs, separated by epicentral distance Repi.

        :param Zf:
            float, depth of focus (in km)
        :param Zs:
            float, depth of station (in km)
        :param Repi:
            float, epicentral distance (= horizontal distance between
            earthquake focus and receiver station
        :param wave:
            see :meth:`get_velocities`

        Note: there is no constraint on the relative depth of Zf and Zs
        (they may be interchanged)

        :return:
            (refl_thetas, refl_tts) tuple:
            - refl_thetas: 1D array, reflection angles in radians [NL]
            - refl_tts: 1D array, travel times in seconds [NL]
        """
        ## Determine higher/lower depth and layer index
        Zh = min(Zf, Zs)
        Zl = max(Zf, Zs)
        Lh = self.get_layer_index(Zh)
        Ll = self.get_layer_index(Zl)

        ## Construct secondary velocity models from Zh and Zl downwards
        vmodel1 = self.constrain_depth_range(Zh, self.depths[-1])
        vmodel2 = self.constrain_depth_range(Zl, self.depths[-1])

        #BL = Ll - self.num_layers
        NL = self.num_layers
        refl_tts = np.zeros(NL) * np.nan
        refl_thetas = np.zeros(NL) * np.nan

        def minimize_func(theta):
            layer_angles1 = vmodel1.calc_layer_angles_above(BL, theta, wave=wave)
            hdistances1 = vmodel1.calc_horizontal_distances(layer_angles1)
            tot_hdistance1 = np.sum(hdistances1)
            layer_angles2 = vmodel2.calc_layer_angles_above(BL, theta, wave=wave)
            hdistances2 = vmodel2.calc_horizontal_distances(layer_angles2)
            tot_hdistance2 = np.sum(hdistances2)
            tot_hdistance = tot_hdistance1 + tot_hdistance2

            return np.abs(Repi - tot_hdistance)

        for BL in range(-1, Ll - NL - 1, -1):
            result = minimize_scalar(minimize_func, bounds=(0, np.pi/2.),
                                    method='bounded')

            if result.success:
                theta = result.x
            else:
                print(result.message)

            layer_angles1 = vmodel1.calc_layer_angles_above(BL, theta, wave=wave)
            travel_times1 = vmodel1.calc_travel_times(layer_angles1, wave=wave)
            layer_angles2 = vmodel2.calc_layer_angles_above(BL, theta, wave=wave)
            travel_times2 = vmodel2.calc_travel_times(layer_angles2, wave=wave)
            tt = np.sum(travel_times1) + np.sum(travel_times2)

            refl_thetas[BL] = theta
            refl_tts[BL] = tt

        return (refl_thetas, refl_tts)

    def calc_min_reflection_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Compute minimum travel time for a reflected wave between focus
        at depth Zf and station at depth Zs, separated by epicentral
        distance Repi.

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`find_reflection_ray_angles_and_tts`

        :return:
            (BL, tmin) tuple:
            - BL: int, bottom layer index
            - tmin: float, minimum travel time in seconds
        """
        if max(Zf, Zs) < self.max_depth:
            _, refl_tts = self.find_reflection_ray_angles_and_tts(Zf, Zs, Repi,
                                                                wave=wave)
            BL = np.nanargmin(refl_tts)
            tmin = refl_tts[BL]
        else:
            BL, tmin = None, np.nan
        return (BL, tmin)

    def find_emerging_ray_angle_and_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Determine angle and travel time of emerging ray (direct wave!) between
        focus at depth Zf and station at depth Zs, separated by epicentral
        distance Repi.

        :param Zf:
            float, depth of focus (in km)
        :param Zs:
            float, depth of station (in km)
        :param Repi:
            float, epicentral distance (= horizontal distance between
            earthquake focus and receiver station
        :param wave:
            see :meth:`get_velocities`

        Note: there is no constraint on the relative depth of Zf and Zs
        (they may be interchanged)

        :return:
            (theta, tt) tuple:
            - theta: float, angle of upgoing ray in radians
            - tt, float, travel time in seconds
        """
        ## Determine higher/lower depth and layer index
        Zh = min(Zf, Zs)
        Zl = max(Zf, Zs)
        Lh = self.get_layer_index(Zh)
        Ll = self.get_layer_index(Zl)

        # TODO: if Zl coincides with layer boundary, consider
        # refracted wave traveling along this boundary
        # (not necessary, I think)

        if Lh == Ll:
            ## Focus and station in same layer
            DZ = Zl - Zh
            theta = np.pi / 2 - np.arctan2(DZ, Repi)
            D = np.sqrt(DZ**2 + Repi**2)
            V = self.get_velocities(wave)
            tt = D / V[Lh]

        else:
            ## Construct secondary velocity model going from Zh to Zl
            vmodel2 = self.constrain_depth_range(Zh, Zl)
            #print(vmodel2.depths)

            def minimize_func(theta):
                layer_angles = vmodel2.calc_layer_angles_above(vmodel2.num_layers-1,
                                                                theta, wave=wave)
                hdistances = vmodel2.calc_horizontal_distances(layer_angles)
                tot_hdistance = np.sum(hdistances)
                #print(tot_hdistance)
                return np.abs(Repi - tot_hdistance)

            result = minimize_scalar(minimize_func, bounds=(0, np.pi/2.),
                                    method='bounded')
            if not result.success:
                print(result.message)
            else:
                theta = result.x
                layer_angles = vmodel2.calc_layer_angles_above(vmodel2.num_layers-1,
                                                                theta, wave=wave)
                travel_times = vmodel2.calc_travel_times(layer_angles, wave=wave)
                tt = np.sum(travel_times)

        return (theta, tt)

    def calc_dirwav_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Compute travel time of direct wave between focus at depth Zf
        and station at depth Zs, separated by epicentral distance Repi.

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`find_emerging_ray_angle_and_tt`

        Note: there is no constraint on the relative depth of Zf and Zs
        (they may be interchanged)

        :return:
            float, travel time in seconds
        """
        return self.find_emerging_ray_angle_and_tt(Zf, Zs, Repi, wave=wave)[1]

    def calc_all_tt(self, Zf, Zs, Repi):
        """
        Compute travel times for all simple phases:
        - Pg: direct P wave
        - Pn: refracted P wave
        - Pm: reflected P wave
        - P: min(Pg, Pn, Pm)
        - Sg: direct S wave
        - Sn: refracted S wave
        - Sm: reflected S wave
        - S: min(Sg, Sn, Sm)

        :param Zf:
        :param Zs:
        :param Repi:
            see :meth:`calc_dirwav_tt` or :meth:`calc_min_refwav_tt`

        :return:
            dict, mapping phase names (str) to travel times in seconds (float)
        """
        phase_tt = {}
        phase_tt['Pg'] = self.calc_dirwav_tt(Zf, Zs, Repi, wave='P')
        phase_tt['Pn'] = self.calc_min_refwav_tt(Zf, Zs, Repi, wave='P')[1]
        phase_tt['Pm'] = self.calc_min_reflection_tt(Zf, Zs, Repi, wave='P')[1]
        phase_tt['P'] = min(phase_tt['Pg'], phase_tt['Pn'], phase_tt['Pm'])
        phase_tt['Sg'] = self.calc_dirwav_tt(Zf, Zs, Repi, wave='S')
        phase_tt['Sn'] = self.calc_min_refwav_tt(Zf, Zs, Repi, wave='S')[1]
        phase_tt['Sm'] = self.calc_min_reflection_tt(Zf, Zs, Repi, wave='S')[1]
        phase_tt['S'] = min(phase_tt['Sg'], phase_tt['Sn'], phase_tt['Sm'])

        return phase_tt

    def calc_min_tt(self, Zf, Zs, Repi, wave='P'):
        """
        Compute minimum travel time (refracted or direct wave) between
        focus at depth Zf and station at depth Zs, separated by
        epicentral distance Repi.

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`calc_dirwav_tt` or :meth:`calc_min_refwav_tt`

        :return:
            (tmin, wave_type) tuple:
            - tmin: float, minimum travel time in seconds
            - wave_type: str, type of wave ('REF' or 'DIR')
        """
        ref_tmin = self.calc_min_refwav_tt(Zf, Zs, Repi, wave=wave)[1]
        dir_tmin = self.calc_dirwav_tt(Zf, Zs, Repi, wave=wave)
        if ref_tmin < dir_tmin:
            return (ref_tmin, 'REF')
        else:
            return (dir_tmin, 'DIR')

    def calc_takeoff_and_incidence_angles(self, Zf, Zs, Repi, wave='P',
                                                    wave_type=None):
        """
        Compute takeoff and incidence angles between
        focus at depth Zf and station at depth Zs, separated by
        epicentral distance Repi.

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`calc_dirwav_tt` or :meth:`calc_min_refwav_tt`
        :param wave_type:
            str, type of wave ('REF', 'DIR', 'REFL' or 'N', 'G', 'M')
            Alternatively, wave type may also be included in :param:`wave`,
            e.g. as 'Pg', 'Pn', 'Pm'
            (default: None, will calculate for fastest arrival)

        :return:
            (takeoff_angle, incidence_angle) tuple of floats:
            - takeoff_angle: angle at which ray leaves the source,
                measured from downwards pointing vertical (in degrees)
            - incidence_angle, angle at which ray arrives at the station,
                measured from downards pointing vertical (in degrees)
        """
        ## Not the most efficient way, some calculations are probably duplicated
        if len(wave) > 1 and wave_type is None:
            wave, wave_type = wave[:1], wave[1:].upper()

        if not wave_type:
            wave_type = self.calc_min_tt(Zf, Zs, Repi, wave=wave)[1]
        wave_type = wave_type.upper()

        if wave_type in ('REF', 'N'):
            BL, tmin = self.calc_min_refwav_tt(Zf, Zs, Repi, wave=wave)
            if BL is None:
                return (None, None)
            takeoff_angle = self.calc_critical_angles(wave)[BL]
        elif wave_type in ('DIR', 'G'):
            BL = self.get_layer_index(max(Zf, Zs))
            takeoff_angle = self.find_emerging_ray_angle_and_tt(Zf, Zs, Repi,
                                                                            wave=wave)[0]
        elif wave_type in ('REFL', 'M'):
            BL, Tmin = self.calc_min_reflection_tt(Zf, Zs, Repi, wave=wave)
            takeoff_angle = self.find_reflection_ray_angles_and_tts(Zf, Zs, Repi,
                                                                            wave=wave)[0][BL]

        incidence_angle = self.calc_layer_angles_above(BL, takeoff_angle,
                                                                    wave=wave)[0]
        ## Next line converts to angle from horizontal downwards,
        ## but this is not the convention
        ## (see https://service.iris.edu/irisws/rotation/docs/1/help/)
        #incidence_angle = np.pi/2. - incidence_angle

        ## Direct wave leaves upwards from source
        if wave_type in ('DIR', 'G'):
            takeoff_angle = np.pi - takeoff_angle

        return np.degrees(takeoff_angle), np.degrees(incidence_angle)

    def calc_tt_and_angles(self, Zf, Zs, Repi, wave='P', wave_type=None):
        """
        Simultaneously compute travel time and takeoff and incidence angles
        between focus at depth Zf and station at depth Zs, separated by
        epicentral distance Repi.

        This is more efficient than calling :math:`calc_min_tt` and
        :meth:`calc_takeoff_and_incidence_angles` separately

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`calc_dirwav_tt` or :meth:`calc_min_refwav_tt`
        :param wave_type:
            str, type of wave ('REF', 'DIR', 'REFL' or 'N', 'G', 'M')
            Alternatively, wave type may also be included in :param:`wave`,
            e.g. as 'Pg', 'Pn', 'Pm'
            (default: None, will calculate for fastest arrival)

        :return:
            (travel_time takeoff_angle, incidence_angle, wave_type) tuple:
            - travel_time: float, travel time (in seconds)
            - takeoff_angle: float, angle at which ray leaves the source,
                measured from downwards pointing vertical (in degrees)
            - incidence_angle, float, angle at which ray arrives at the station,
                measured from downards pointing vertical (in degrees)
            - wave_type: str, corresponding wave type
        """
        if len(wave) > 1 and wave_type is None:
            wave, wave_type = wave[:1], wave[1:].upper()

        if not wave_type:
            ## Find fastest wave type
            wave_types = ['REF', 'DIR']
        else:
            wave_types = [wave_type]

        travel_times = []
        takeoff_angles = []
        incidence_angles = []
        for wave_type in wave_types:
            if wave_type in ('DIR', 'G'):
                to_angle, tt = self.find_emerging_ray_angle_and_tt(Zf, Zs, Repi,
                                                                                wave=wave)
                ## Direct wave leaves upwards from source
                to_angle = np.pi - to_angle
                BL = self.get_layer_index(max(Zf, Zs))
            elif wave_type in ('REF', 'N'):
                BL, tt = self.calc_min_refwav_tt(Zf, Zs, Repi, wave=wave)
                if BL is None:
                    to_angle, tt = np.nan, np.nan
                else:
                    to_angle = self.calc_critical_angles(wave)[BL]
            elif wave_type in ('REFL', 'M'):
                if max(Zf, Zs) < self.max_depth:
                    _to_angles, _tts = self.find_reflection_ray_angles_and_tts(Zf, Zs,
                                                                                Repi, wave=wave)
                    BL = np.nanargmin(_tts)
                    tt = _tts[BL]
                    to_angle = _to_angles[BL]
                else:
                    tt, to_angle = np.nan, np.nan
                    BL = None
            travel_times.append(tt)
            takeoff_angles.append(to_angle)

            ## Calculate incidence angle from takeoff angle and bottom layer idx
            if BL is not None:
                incidence_angle = self.calc_layer_angles_above(BL, to_angle,
                                                                        wave=wave)[0]
            else:
                incidence_angle = np.nan
            incidence_angles.append(incidence_angle)

        takeoff_angles = np.degrees(takeoff_angles)
        incidence_angles = np.degrees(incidence_angles)

        if len(travel_times) > 1:
            ## Find index corresponding to fastest arrival
            i = np.nanargmin(travel_times)
        else:
            i = 0

        return (travel_times[i], takeoff_angles[i], incidence_angles[i],
                wave_types[i])

    def calc_sp_interval(self, Zf, Zs, Repi):
        """
        Compute time difference between S and P-wave arrivals

        :param Zf:
        :param Zs:
        :param Repi:
            see :meth:`calc_dirwav_tt` or :meth:`calc_min_refwav_tt`

        :return:
            float, S-P time interval
        """
        Pmin_tt = self.calc_min_tt(Zf, Zs, Repi, wave='P')[0]
        Smin_tt = self.calc_min_tt(Zf, Zs, Repi, wave='S')[0]
        SminusP = Smin_tt - Pmin_tt
        return SminusP

    def calc_epicentral_distance(self, Zf, Zs, sp_interval):
        """
        Compute epicentral distance based on difference between
        S-wave and P-wave arrival times

        :param Zf:
            float, depth of focus (in km)
        :param Zs:
            float, depth of station (in km)
        :param sp_interval:
            float, time between S-wave and P-wave arrivals (in seconds)

        :return:
            float, epicentral distance (in km)
        """
        def minimize_func(repi):
            SminusP = self.calc_sp_interval(Zf, Zs, repi)
            return np.abs(SminusP - sp_interval)

        max_dist = sp_interval * 10
        result = minimize_scalar(minimize_func, bounds=(0, max_dist),
                                method='bounded')
        if result.success:
            Repi = result.x
        else:
            print(result.message)

        return Repi

    def calc_tt_residuals(self, Zf, Tf, Repi, Zs, Ts, Vidx):
        """
        Compute travel-time residuals.
        Residuals are positive when observed travel time is larger
        than calculated travel time

        :param Zf:
            float, depth of focus (in km)
        :param Tf:
            float, origin time (in seconds)
        :param Repi:
            1D array, epicentral distances (in km) [num_phases]
        :param Zs:
            1D array, station depths (in km) [num_phases]
        :param Ts:
            1D array, phase arrival times (in seconds) [num_phases]
        :param Vidx:
            1D array, velocity indexes (0=P or 1=S) [num_phases]

        :return:
            (tt, tt_residuals) tuple
            - tt: 1D array, calculated travel times (in seconds)
            - tt_residuals: 1D array, travel time residuals (in seconds)
        """
        assert len(Repi) == len(Zs) == len(Ts) == len(Vidx)

        num_phases = len(Repi)
        tt = np.zeros(num_phases)
        tt_residuals = np.zeros(num_phases)
        for k in range(num_phases):
            repi, zs, ts, vidx = Repi[k], Zs[k], Ts[k], Vidx[k]
            wave = 'PS'[vidx]
            tt_obs = ts - Tf
            tt_calc = self.calc_min_tt(Zf, zs, repi, wave=wave)[0]
            tt[k] = tt_calc
            tt_residuals[k] = tt_obs - tt_calc

        return tt, tt_residuals

    def to_taup_model(self, densities=None, QP=None, QS=None, output_file=''):
        """
        Create obspy TauP model
        Note: this takes very long, and the result needs to be written to a file
        (taup_model.serialize(output_filename)) before it can be used in obspy...

        :param densities:
            1D array, layer densities (in kg/m**3)
            (default: None, will extimate from VS)
        :param QP:
            float, P-wave quality factor
            (default: None)
        :param QS:
            float, S-wave quality factor
            (default: None)
        :param output_file:
            str, full path to file where TauP model should be saved
            (default: '', will not save)

        :return:
            instance of :class:`obspy.taup.TauPyModel
        """
        from obspy.taup.velocity_layer import VelocityLayer
        from obspy.taup.velocity_model import VelocityModel
        from obspy.taup.taup_create import TauPCreate
        from obspy.taup import _DEFAULT_VALUES

        if densities is None:
            densities = self.estimate_densities_from_vs()
        densities /= 1000

        ## Note: varying attenuation factors not yet supported in obspy.taup
        if QP is None:
            QP = _DEFAULT_VALUES['qp']
        QP = np.ones(self.num_layers+1) * QP
        if QS is None:
            QS = _DEFAULT_VALUES['qs']
        QS = np.ones(self.num_layers+1) * QS

        moho_depth = self.max_depth
        cmb_depth = _DEFAULT_VALUES['default_cmb']
        iocb_depth = _DEFAULT_VALUES['default_iocb']
        planet_radius = 6371

        layers = np.empty(self.num_layers+1, dtype=VelocityLayer)
        for i in range(self.num_layers + 1):
            layers[i]['top_depth'] = self.depths[i]
            if i < self.num_layers:
                layers[i]['bot_depth'] = self.depths[i+1]
            else:
                layers[i]['bot_depth'] = planet_radius
            layers[i]['top_p_velocity'] = self.VP[i]
            layers[i]['bot_p_velocity'] = self.VP[i]
            layers[i]['top_s_velocity'] = self.VS[i]
            layers[i]['bot_s_velocity'] = self.VS[i]
            layers[i]['top_density'] = densities[i]
            layers[i]['bot_density'] = densities[i]
            layers[i]['top_qp'] = QP[i]
            layers[i]['bot_qp'] = QP[i]
            layers[i]['top_qs'] = QS[i]
            layers[i]['bot_qs'] = QS[i]

        # Remove zero thickness layers
        mask = layers['top_depth'] == layers['bot_depth']
        layers = layers[~mask]

        velocity_model = VelocityModel(model_name=self.name,
                                radius_of_planet=planet_radius,
                                min_radius=0,
                                max_radius=planet_radius,
                                moho_depth=moho_depth,
                                cmb_depth=cmb_depth,
                                iocb_depth=iocb_depth,
                                is_spherical=True,
                                layers=layers)
        #return velocity_model

        input_filename = ''
        taup_create = TauPCreate(input_filename, output_file,
                                    allow_inner_core_s=False)
        taup_create.debug = True
        tau_model = taup_create.create_tau_model(velocity_model)

        if output_file:
            tau_model.serialize(output_file)

        return tau_model

    def plot_tt_diagram(self, Zf, Zs, Repi, wave="P", **kwargs):
        """
        Plot travel time vs. epicentral distance (no external plotting deps).

        Parameters
        ----------
        Zf : float
            Focus depth (km)
        Zs : float
            Station depth (km)
        Repi : array-like
            Epicentral distances (km)
        wave : {'P','S'}, default 'P'
            Wave type
        **kwargs :
            Optional matplotlib keywords:
            - ax : existing Axes instance
            - figsize : tuple
            - xmin, ymin : axis limits
            - xlabel, ylabel, title : labels and title
            - xgrid, ygrid : bools
            - colors : list of color strings

        Returns
        -------
        matplotlib.axes.Axes
        """
        def finite_max(*arrays):
            vals = [np.nanmax(a[np.isfinite(a)]) for a in arrays if np.any(np.isfinite(a))]
            return np.nanmax(vals) if len(vals) else 1.0  # fallback if all invalid
        import matplotlib.pyplot as plt
        # ensure numpy array
        Repi = np.asarray(Repi, dtype=float)
        num = len(Repi)

        # --- compute travel times ---
        refx_tt = np.zeros(num)
        refl_tt = np.zeros(num)
        dir_tt  = np.zeros(num)
        for k, repi in enumerate(Repi):
            refx_tt[k] = self.calc_min_refwav_tt(Zf, Zs, repi, wave=wave)[1]
            refl_tt[k] = self.calc_min_reflection_tt(Zf, Zs, repi, wave=wave)[1]
            dir_tt[k]  = self.calc_dirwav_tt(Zf, Zs, repi, wave=wave)

        # --- prepare axes ---
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

        # --- plot curves ---
        colors = kwargs.pop("colors", ["g", "m", "b"])
        labels = ["Direct wave (Pg)", "Refracted wave (Pn)", "Reflected wave (Pm)"]
        datasets = [(Repi, dir_tt), (Repi, refx_tt), (Repi, refl_tt)]

        for (x, y), color, label in zip(datasets, colors, labels):
            ax.plot(x, y, color=color, lw=1.6, label=label)

        # --- axis limits and labels ---
        xmin = kwargs.pop("xmin", 0)
        ymin = kwargs.pop("ymin", 0)
        # safe finite max
        ymax = finite_max(dir_tt, refx_tt, refl_tt) * 1.05
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, Repi.max() * 1.05)
        #ax.set_ylim(ymin, max(dir_tt.max(), refx_tt.max(), refl_tt.max()) * 1.05)

        ax.set_xlabel(kwargs.pop("xlabel", "Epicentral distance (km)"))
        ax.set_ylabel(kwargs.pop("ylabel", "Travel time (s)"))
        ax.set_title(kwargs.pop("title", f"{self.name} ({wave.upper()} waves)"))

        # --- grid and legend ---
        if kwargs.pop("xgrid", True) or kwargs.pop("ygrid", True):
            ax.grid(True, ls=":", lw=0.5)
        ax.legend(loc="best", fontsize=8)

        return ax

    def calc_path_elements(self, Zf, Zs, Repi, wave_type, wave='P'):
        """
        Compute elements (X and Y coordinates) of travel path

        :param Zf:
        :param Zs:
        :param Repi:
            see :meth:`calc_min_tt`
        :param wave_type:
            str, wave type: 'DIR', 'REF' or 'REFL' for direct, refracted
            and reflected wave
        :param wave:
            see :meth:`calc_min_tt`

        :return:
            X, Y tuple of arrays: X and Y coordinates (in km) of travel path,
            starting from the hypocenter (0, Zf)
        """
        X, Y = [], []

        if wave_type in ('DIR', 'g'):
            ## Direct wave
            X, Y = [Repi], [Zs]
            if not np.isclose(Zs, Zf):
                vmodel2 = self.constrain_depth_range(Zs, Zf)
                bottom_angle, tt = vmodel2.find_emerging_ray_angle_and_tt(vmodel2.max_depth,
                                                                        0, Repi, wave=wave)
                layer_angles = vmodel2.calc_layer_angles_above(vmodel2.num_layers - 1,
                                                                bottom_angle, wave=wave)
                hdistances = vmodel2.calc_horizontal_distances(layer_angles)
                for l in range(len(hdistances)):
                    X.append(X[-1] - hdistances[l])
                    Y.append(vmodel2.depths[l+1] + Zs)
            else:
                X.append(0)
                Y.append(Zf)
            X = X[::-1]
            Y = Y[::-1]

        elif wave_type in ('REF', 'REFR', 'n'):
            ## Refracted wave
            BL, tmin = self.calc_min_refwav_tt(Zf, Zs, Repi, wave=wave)
            if BL != None:
                bottom_angle = self.calc_critical_angles(wave=wave)[BL]
                vmodel1 = self.constrain_depth_range(Zf, self.depths[BL+1])
                layer_angles = vmodel1.calc_layer_angles_above(vmodel1.num_layers - 1,
                                                                bottom_angle, wave=wave)
                hdistances = vmodel1.calc_horizontal_distances(layer_angles)
                X.append(0)
                Y.append(Zf)
                for l in range(len(hdistances)):
                    X.append(X[-1] + hdistances[l])
                    Y.append(vmodel1.depths[l+1] + Zf)
                vmodel2 = self.constrain_depth_range(Zs, self.depths[BL+1])
                layer_angles = vmodel2.calc_layer_angles_above(vmodel2.num_layers - 1,
                                                                bottom_angle, wave=wave)
                hdistances = vmodel2.calc_horizontal_distances(layer_angles)
                X.append(Repi - np.sum(hdistances))
                Y.append(Y[-1])
                for l in range(len(hdistances)):
                    X.append(X[-1] + hdistances[-(l+1)])
                    Y.append(vmodel2.depths[vmodel2.num_layers - l - 1] + Zs)

        elif wave_type in ('REFL', 'm'):
            ## Reflected wave
            BL, tmin = self.calc_min_reflection_tt(Zf, Zs, Repi, wave=wave)
            if BL != None:
                refl_angles, _ = self.find_reflection_ray_angles_and_tts(Zf, Zs, Repi,
                                                                        wave=wave)
                bottom_angle = refl_angles[BL]
                vmodel1 = self.constrain_depth_range(Zf, self.depths[BL+1])
                layer_angles = vmodel1.calc_layer_angles_above(vmodel1.num_layers - 1,
                                                                bottom_angle, wave=wave)
                hdistances = vmodel1.calc_horizontal_distances(layer_angles)
                X.append(0)
                Y.append(Zf)
                for l in range(len(hdistances)):
                    X.append(X[-1] + hdistances[l])
                    Y.append(vmodel1.depths[l+1] + Zf)
                vmodel2 = self.constrain_depth_range(Zs, self.depths[BL+1])
                layer_angles = vmodel2.calc_layer_angles_above(vmodel2.num_layers - 1,
                                                                bottom_angle, wave=wave)
                hdistances = vmodel2.calc_horizontal_distances(layer_angles)
                for l in range(len(hdistances)):
                    X.append(X[-1] + hdistances[-(l+1)])
                    Y.append(vmodel2.depths[vmodel2.num_layers - l - 1] + Zs)

        return (np.array(X), np.array(Y))

    def calc_travel_distance(self, Zf, Zs, Repi, wave_type, wave='P'):
        """
        Compute travel distance

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave_type:
        :param wave:
            see :meth:`calc_path_elements`

        :return:
            float, travel distance (in km)
        """
        X, Y = self.calc_path_elements(Zf, Zs, Repi, wave_type, wave=wave)
        d = np.sum(np.hypot(np.diff(X), np.diff(Y)))

        return d

    def calc_average_velocity(self, Zf, Zs, Repi, wave='P'):
        """
        Compute average velocity over travel path

        :param Zf:
        :param Zs:
        :param Repi:
        :param wave:
            see :meth:`calc_min_tt`

        :return:
            float, average velocity (in km/s)
        """
        tt, wave_type = self.calc_min_tt(Zf, Zs, Repi, wave=wave)
        d = self.calc_travel_distance(Zf, Zs, Repi, wave_type, wave=wave)
        Vavg = d / tt

        return Vavg



                        
    


    def plot_rays(self, Zf, Zs, Repi, wave="P", **kwargs):
        """
        Plot ray paths through a layered crustal velocity model.
        Pure matplotlib version – no dependency on generic_mpl.

        Parameters
        ----------
        Zf : float
            Focus (source) depth in km
        Zs : float
            Receiver (station) depth in km
        Repi : float
            Epicentral distance in km
        wave : str
            Wave type: 'P' or 'S' (default 'P')
        **kwargs : dict
            Optional keyword args for matplotlib (e.g., figsize, ax, title, etc.)

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        # allow external axes
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

        xmin = kwargs.pop("xmin", -10)
        xmax = kwargs.pop("xmax", Repi + 10)

        # --- Plot layer boundaries ---
        for i, z in enumerate(self.depths):
            ax.plot([xmin, xmax], [z, z], "k-", lw=0.8,
                    label="Layers" if i == 0 else "_nolegend_")

        # --- Direct wave ---
        X, Y = self.calc_path_elements(Zf, Zs, Repi, "DIR", wave=wave)
        if len(X):
            ax.plot(X, Y, "g-", lw=1.5, label="Direct wave (Pg)")

        # --- Refracted wave ---
        X, Y = self.calc_path_elements(Zf, Zs, Repi, "REF", wave=wave)
        if len(X):
            ax.plot(X, Y, "m-", lw=1.5, label="Refracted wave (Pn)")

        # --- Reflected wave ---
        X, Y = self.calc_path_elements(Zf, Zs, Repi, "REFL", wave=wave)
        if len(X):
            ax.plot(X, Y, "b-", lw=1.5, label="Reflected wave (Pm)")

        # --- Focus and station markers ---
        ax.plot(0, Zf, "ro", ms=8, label="Focus")
        ax.plot(Repi, Zs, "^", color="gold", ms=8, label="Station")

        # --- Aesthetics ---
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(max(self.depths) + 5, -5)  # depth downwards
        ax.set_xlabel(kwargs.pop("xlabel", "Epicentral distance (km)"))
        ax.set_ylabel(kwargs.pop("ylabel", "Depth (km)"))
        ax.set_title(kwargs.pop("title", f"{self.name} ({wave.upper()} waves)"))

        ax.grid(True, ls=":", lw=0.5)
        ax.legend(loc="best", fontsize=8)

        return ax

    def perturb(self, depth_factor=0.1, vel_factor=0.05, vpvs_scale=1.75,
                name_suffix="_pert", seed=None):
                """
                Randomly perturb layer depths and velocities in this CrustalVelocityModel.

                Parameters
                ----------
                depth_factor : float
                    Fractional perturbation for layer depths (except top & bottom).
                vel_factor : float
                    Fractional perturbation for Vp values.
                vpvs_scale : float
                    Fixed Vp/Vs ratio to compute Vs from perturbed Vp.
                name_suffix : str
                    Optional suffix to append to model name for identification.
                seed : int or None
                    Seed for the random number generator.  Pass an integer for
                    reproducible perturbations; None (default) gives a different
                    result each call.

                Returns
                -------
                perturbed : CrustalVelocityModel
                    New velocity model instance with perturbed parameters.
                """
                rng = np.random.default_rng(seed)
                perturbed = deepcopy(self)

                # --- Perturb depths but keep top and bottom fixed ---
                depths = np.array(self.depths, dtype=float)
                top, bottom = depths[0], depths[-1]
                inner = depths[1:-1]
                thickness = np.diff(depths)
                inner += rng.uniform(-depth_factor, depth_factor, size=len(inner)) * thickness[:-1]
                new_depths = np.concatenate(([top], np.clip(inner, top, bottom), [bottom]))

                # --- Perturb velocities ---
                vp = np.array(self.VP, dtype=float)
                pert_vp = vp * (1 + rng.uniform(-vel_factor, vel_factor, size=len(vp)))
                pert_vs = pert_vp / vpvs_scale

                # --- Assign new fields ---
                perturbed.depths = new_depths.tolist()
                perturbed.VP = pert_vp.tolist()
                perturbed.VS = pert_vs.tolist()
                perturbed.name = f"{self.name}{name_suffix}"
                return perturbed


##################
#Additional functions
#################

def plot_velocity_models(
    models,
    ax=None,
    colors=None,
    labels=None,
    lw=2,
    invert_y=True,
    show_vs=False,
    vp_color="tab:red",
    vs_color="tab:blue",
    max_depth=None,
    xlim=None
):
    """
    Plot one or more CrustalVelocityModel objects as stepwise (piecewise constant) profiles.
    Accepts:
      - a single CrustalVelocityModel
      - a list/tuple of CrustalVelocityModels
      - a dict of name -> CrustalVelocityModel

    Parameters
    ----------
    models : CrustalVelocityModel, list, or dict
        Model(s) to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
    colors : list of str, optional
        Colors for the models.
    labels : list of str, optional
        Labels for the models (ignored if dict input).
    lw : float
        Line width.
    invert_y : bool
        Whether to invert the depth axis (default True).
    show_vs : bool
        If True, plot Vs alongside Vp (dashed lines).
    vp_color, vs_color : str
        Default colors when plotting a single model.
    max_depth : float, optional
        Maximum depth to display (km).
    xlim : tuple, optional
        Optional (xmin, xmax) limits for velocity axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the plot.
    """
    import matplotlib.pyplot as plt
    
    # --- Normalize input ---
    if isinstance(models, dict):
        labels = list(models.keys())
        models = list(models.values())
    elif not isinstance(models, (list, tuple)):
        models = [models]
        labels = labels or [getattr(models[0], "name", "Model 1")]

    n = len(models)

    # --- Setup plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = ax.figure

    if colors is None:
        colors = plt.cm.tab10.colors[:n]

    # --- Plot each model ---
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        depths = np.asarray(model.depths, dtype=float)
        vp = np.asarray(model.VP, dtype=float)
        vs = np.asarray(getattr(model, "VS", None), dtype=float) if hasattr(model, "VS") else None

        # Create stepwise representation
        depth_steps = np.repeat(depths, 2)[1:-1]
        vp_steps = np.repeat(vp, 2)[:len(depth_steps)]
        if vs is not None:
            vs_steps = np.repeat(vs, 2)[:len(depth_steps)]

        # Plot Vp
        ax.step(vp_steps, depth_steps, where="post",
                color=color if n > 1 else vp_color,
                lw=lw, label=f"{label} (Vp)" if show_vs else label)

        # Optionally plot Vs
        if show_vs and vs is not None:
            ax.step(vs_steps, depth_steps, where="post",
                    color=color if n > 1 else vs_color,
                    lw=lw * 0.9, ls="--", alpha=0.8,
                    label=f"{label} (Vs)")

    # --- Formatting ---
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")
    if invert_y:
        ax.invert_yaxis()
    if max_depth is not None:
        ax.set_ylim(max_depth, 0 if invert_y else max_depth)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.tight_layout()

    return ax



def save_velocity_model_to_csv(model, filename):
    """
    Save a CrustalVelocityModel-like object to CSV.
    
    Parameters
    ----------
    model : object
        Must have attributes `depths`, `VP`, `VS`, and optionally `name`.
    filename : str
        Output CSV path.
    """
    df = pd.DataFrame({
        "Depth_km": np.asarray(model.depths, dtype=float),
        "Vp_km_per_s": np.asarray(model.VP, dtype=float),
        "Vs_km_per_s": np.asarray(model.VS, dtype=float)
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Saved velocity model '{getattr(model, 'name', 'Unnamed')}' to {filename}")


    return model

def load_velocity_model_from_csv(filename, name=None):
    """
    Load a crustal velocity model from CSV and return a CrustalVelocityModel instance.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    name : str, optional
        Name to assign to the loaded model.

    Returns
    -------
    CrustalVelocityModel
        Instance containing depths, VP, VS, and name.
    """
    df = pd.read_csv(filename)

    # Basic validation
    required_cols = {"Depth_km", "Vp_km_per_s", "Vs_km_per_s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filename}: {', '.join(missing)}")

    depths = df["Depth_km"].to_numpy(dtype=float)
    vp = df["Vp_km_per_s"].to_numpy(dtype=float)
    vs = df["Vs_km_per_s"].to_numpy(dtype=float)
    model_name = name or os.path.splitext(os.path.basename(filename))[0]

    model = CrustalVelocityModel(depths, vp, vs, name=model_name)
    print(f"✅ Loaded velocity model '{model.name}' from {filename}")
    return model


    

