# -*- coding: iso-8859-1 -*-
"""
Helper classes for hypo-location programs
"""

import os
import numpy as np
import pandas as pd

from obspy import UTCDateTime

__all__ = [
    "PhasePick",
    "Station",
    "StationPhases",
    "bare_sta",
]


def bare_sta(code: str) -> str:
    """
    Return just the station part of any SEED-style code.

    Works for all common formats::

        'ABM1Y'          → 'ABM1Y'   # bare code — unchanged
        'AW.ABM1Y'       → 'ABM1Y'   # NET.STA
        'AW.ABM1Y.00'    → 'ABM1Y'   # NET.STA.LOC  (uses STA part)
        'AW.ABM1Y.00.HHZ'→ 'ABM1Y'   # full SEED    (uses STA part)

    The second dot-separated field is always the SEED station code;
    when there is no dot the whole string is returned as-is.
    """
    if '.' not in code:
        return code
    parts = code.split('.')
    return parts[1] if len(parts) >= 2 else parts[-1]

class PhasePick:
    """
    Simplified replacement for robspy.PhasePick, compatible with hypo_minimize().
    """
    def __init__(self, phase_name, datetime, id_earth=None, station_code=None,
                 component='', first_motion='', db_id=None, include_in_loc=True,
                 onset=None, amplitude=None, period=None, magnitude=None,
                 mag_type=None, Repi=None, Rhypo=None, azimuth=None,
                 back_azimuth=None, residual=None, takeoff_angle=None,
                 incidence_angle=None, manual=True, station_lon=None,
                 station_lat=None, station_altitude=0., station_depth=0.,
                 station_network='VX', station_location='', agency='UOM'):
        # ---- core fields ----
        self.phase_name = phase_name
        self.datetime = datetime if isinstance(datetime, UTCDateTime) else UTCDateTime(datetime)
        self.id_earth = id_earth
        self.station_code = station_code
        self.component = component
        self.first_motion = first_motion
        self.db_id = db_id
        self.include_in_loc = bool(include_in_loc)

        # ---- optional metadata ----
        self.onset = onset
        self.amplitude = amplitude
        self.period = period
        self.magnitude = magnitude
        self.mag_type = mag_type

        # ---- geometry fields ----
        self.Repi = Repi
        self.Rhypo = Rhypo
        self.azimuth = azimuth
        self.back_azimuth = back_azimuth
        self.residual = residual
        self.takeoff_angle = takeoff_angle
        self.incidence_angle = incidence_angle

        # ---- station meta ----
        self.manual = manual
        self.station_lon = station_lon
        self.station_lat = station_lat
        self.station_altitude = station_altitude
        self.station_depth = station_depth
        self.station_network = station_network
        self.station_location = station_location
        self.agency = agency

    def __repr__(self):
        return f"<PhasePick {self.phase_name} {self.datetime.isoformat()} @ {self.station_code}>"

    def __sub__(self, other):
        """Subtract to get time difference in seconds."""
        if isinstance(other, PhasePick):
            other = other.datetime
        if isinstance(other, UTCDateTime):
            return self.datetime - other
        raise TypeError("Subtraction only supported against PhasePick or UTCDateTime")

    def get_time_shifted_copy(self, time_shift=0):
        """Return a new PhasePick with shifted time."""
        import copy
        new_pick = copy.copy(self)
        new_pick.datetime = self.datetime + time_shift
        return new_pick



class Station:
	"""
	Basic station used for locating an earthquake

	:param code:
		str, station code (<network>.<station_code>)
	:param lon:
		float, station longitude (in degrees)
	:param lat:
		float, station latitude (in degrees)
	:param elevation:
		float, absolute station altitude (in km)
	:param depth:
		float, station depth relative to surface (in km)
		(default: 0)
	"""
	def __init__(self, code, lon, lat, elevation, depth=0):
		self.code = code
		self.lon = lon
		self.lat = lat
		self.elevation = elevation
		self.depth = depth

	@property
	def sta(self) -> str:
		"""
		Bare station code, regardless of how :attr:`code` was constructed.

		Safe to use wherever a 4-char SEED station name is needed (e.g. HYPO71
		fixed-format columns).  Examples::

		    Station('ABM1Y', ...).sta        → 'ABM1Y'
		    Station('AW.ABM1Y', ...).sta     → 'ABM1Y'
		    Station('AW.ABM1Y.00', ...).sta  → 'ABM1Y'
		"""
		return bare_sta(self.code)

	@property
	def network(self) -> str | None:
		"""
		Network code if present in :attr:`code`, otherwise ``None``.

		Examples::

		    Station('ABM1Y', ...).network     → None
		    Station('AW.ABM1Y', ...).network  → 'AW'
		"""
		if '.' not in self.code:
			return None
		return self.code.split('.')[0]


class StationPhases(object):
	"""
	Combination of stations and phase picks

	:param stations:
		list with instances of :class:`Station`
	:param phase_picks:
		dict, mapping station codes to dicts, mapping phase names
		('P' or 'S') to instances of :class:`robspy.PhasePick`
		Note: station codes should match exactly with 'code' attribute
		of instances in :param:`stations`
	:param use_s_picks:
		bool, indicating whether or not to use S phases
		(default: True)
	"""
	def __init__(self, stations, phase_picks, use_s_picks=True):
		self.stations = stations
		self.phase_picks = phase_picks
		self.use_s_picks = use_s_picks
		self._construct_array()
		self._define_indexes()

	def _define_indexes(self):
		## Define various indexes
		self.PIDXS = ~self.data['Ptime'].isna()
		self.SIDXS = ~self.data['Stime'].isna()
		self.SPIDXS = self.PIDXS & self.SIDXS
		## Station index for each phase
		NS = self.num_stations
		self.IDXS = np.hstack([np.arange(NS)[self.PIDXS],
								np.arange(NS)[self.SIDXS]])
		## Velocity index for each phase
		self.VIDXS = np.hstack([[0] * np.sum(self.PIDXS),
								[1] * np.sum(self.SIDXS)])
		## Ensure VIDXS is int for TRVDRV function
		self.VIDXS = self.VIDXS.astype('int')

	@classmethod
	def from_phase_pick_list(cls, phase_picks, include_in_loc_only=False):
		"""
		Construct from list of phase picks

		:param phase_picks:
			list with instances of :class:`robspy.PhasePick`
		:param include_in_loc_only:
			bool, whether or not to include only picks that have been used
			for locating the earthquake
			(default: False, will import all phase picks)

		:return:
			instance of :class:`StationPhases`
		"""
		from ..rob import get_stations

		## Convert phase pick list to dictionary
		phase_pick_dict = {}
		for pick in phase_picks:
			if not include_in_loc_only:
				pick.include_in_loc = True
			station_code = pick.station_code
			network = pick.station_network
			key = "%s.%s" % (network, station_code)
			if not key in phase_pick_dict:
				phase_pick_dict[key] = {}
			phase_pick_dict[key][pick.phase_name] = pick

		## Fetch station coordinates
		stations = []
		for key, pick_dict in phase_pick_dict.items():
			network, station_code = key.split('.')
			stats = get_stations(station_code, network=network, strict_code=False)
			try:
				stat = stats[0]
			except:
				print('Skipping station %s.%s' % (network, station_code))
			else:
				stat_coords = stat.get_coordinates()
				code = '%s.%s' % (network, station_code)
				stations.append(Station(code, stat_coords['longitude'],
										stat_coords['latitude'],
										stat_coords['elevation'] / 1000.,
										stat_coords.get('depth', 0) / 1000.))

		return cls(stations, phase_pick_dict)

	@property
	def stat_codes(self):
		#return np.array([str(code.astype('|U10')) for code in self.data['code']])
		return self.data.code.to_numpy()

	@property
	def stat_lons(self):
		return self.data['lon'].to_numpy()

	@stat_lons.setter
	def stat_lons(self, values):
		self.data['lon'] = values

	@property
	def stat_lats(self):
		return self.data['lat'].to_numpy()

	@stat_lats.setter
	def stat_lats(self, values):
		self.data['lat'] = values

	@property
	def stat_elevs(self):
		return self.data['elev'].to_numpy()

	@stat_elevs.setter
	def stat_elevs(self, values):
		self.data['elev'] = values

	@property
	def stat_depths(self):
		return self.data['depth'].to_numpy()

	@stat_depths.setter
	def stat_depths(self, values):
		self.data['depth'] = values

	@property
	def stat_weights(self):
		return self.data['stat_weight'].to_numpy()

	@stat_weights.setter
	def stat_weights(self, values):
		self.data['stat_weight'] = values

	@property
	def num_stations(self):
		return len(self.stations)

	@property
	def num_phases(self):
		return self.num_p_phases + self.num_s_phases

	@property
	def num_p_phases(self):
		return np.sum(self.PIDXS)

	@property
	def num_s_phases(self):
		return np.sum(self.SIDXS)

	def _construct_array(self):
		"""
		Create structured array with station and phase data
		"""
		from pandas import DataFrame

		NS = self.num_stations

		codes = [stat.code for stat in self.stations]
		station_phases = {'code': codes}

		for key, dtype in [('lon', np.float32),
								('lat', np.float32),
								('elev', np.float32),
								('depth', np.float32),
								('stat_weight', np.float32),
								('delay1', np.float32),
								('delay2', np.float32),
								('mno', np.int8),
								('Ptime', 'object'),
								('Ponset', np.byte),
								('Pfm', np.byte),
								('Pweight', np.float32),
								('Stime', 'object'),
								('Sonset', np.byte),
								('Sfm', np.byte),
								('Sweight', np.float32),
								('takeoff_angle', np.float32),
								('azimuth', np.int16),
								('az_weight', np.float32),
								('Repi', np.float32),
								('Rhypo', np.float32),
								('dist_weight', np.float32),
								('Pres', np.float32),
								('Sres', np.float32),
								('Ptot_weight', np.float32),
								('Stot_weight', np.float32)]:
			station_phases[key] = np.zeros(NS, dtype=dtype)

		station_phases = DataFrame(station_phases)
		station_phases.loc[:, 'Ptime'] = np.nan
		station_phases.loc[:, 'Stime'] = np.nan
		station_phases.loc[:, 'Pres'] = np.nan
		station_phases.loc[:, 'Sres'] = np.nan

		for s, station in enumerate(self.stations):
			station_phases.loc[s, 'code'] = str(station.code)
			station_phases.loc[s, 'lon'] = np.float32(station.lon)
			station_phases.loc[s, 'lat'] = np.float32(station.lat)
			station_phases.loc[s, 'elev'] = np.float32(station.elevation)
			station_phases.loc[s, 'depth'] = np.float32(station.depth)
			station_phases.loc[s, 'stat_weight'] = 1

			# Match the station against the phase_picks dict.  pick_dict keys may
			# be bare codes ('ABM1Y') or NET.STA ('AW.ABM1Y').  Try the full
			# station.code first; fall back to the bare station name so that
			# e.g. Station('AW.ABM1Y') matches a pick_dict keyed by 'ABM1Y'.
			station_code = station.code
			if station_code not in self.phase_picks:
				station_code = station.sta
			picks = self.phase_picks.get(station_code, {})
			Ppick = picks.get('P')
			if Ppick and Ppick.include_in_loc:
				station_phases.loc[s, 'Ptime'] = Ppick.datetime
				station_phases.loc[s, 'Pweight'] = 1
				if Ppick.Rhypo is not None:
					station_phases.loc[s, 'Rhypo'] = Ppick.Rhypo
				if Ppick.Repi is not None:
					station_phases.loc[s, 'Repi'] = Ppick.Repi
				if Ppick.azimuth is not None:
					station_phases.loc[s, 'azimuth'] = Ppick.azimuth
			else:
				station_phases.loc[s, 'Ptime'] = np.nan
				station_phases.loc[s, 'Pweight'] = 0
			if self.use_s_picks:
				Spick = picks.get('S')
				if Spick and Spick.include_in_loc:
					station_phases.loc[s, 'Stime'] = Spick.datetime
					station_phases.loc[s, 'Sweight'] = 1
					if Spick.Rhypo is not None:
						station_phases.loc[s, 'Rhypo'] = Spick.Rhypo
					if Spick.Repi is not None:
						station_phases.loc[s, 'Repi'] = Spick.Repi
					if Spick.azimuth is not None:
						station_phases.loc[s, 'azimuth'] = Spick.azimuth
				else:
					station_phases.loc[s, 'Stime'] = np.nan
					station_phases.loc[s, 'Sweight'] = 0

		self.data = station_phases

	@property
	def phase_lons(self):
		return np.asarray(self.stat_lons)[self.IDXS]

	@property
	def phase_lats(self):
		return np.asarray(self.stat_lats)[self.IDXS]

	@property
	def phase_elevs(self):
		return np.asarray(self.stat_elevs)[self.IDXS]

	@property
	def phase_depths(self):
		return np.asarray(self.stat_depths)[self.IDXS]

	@property
	def Ptimes(self):
		"""P-wave arrival time for each station (including NaN values)"""
		return np.asarray(self.data["Ptime"])

	@property
	def Stimes(self):
		"""S-wave arrival time for each station (including NaN values)"""
		return np.asarray(self.data["Stime"])

	@property
	def arrival_times(self):
		"""All valid P-wave then S-wave arrival times"""
		return np.hstack([self.Ptimes[self.PIDXS], self.Stimes[self.SIDXS]])

	@property
	def Presiduals(self):
		"""P-wave residual for each station (including NaN values)"""
		return np.asarray(self.data["Pres"])

	@property
	def Sresiduals(self):
		"""S-wave residual for each station (including NaN values)"""
		return np.asarray(self.data["Sres"])

	@property
	def residuals(self):
		"""All valid P-wave then S-wave residuals"""
		return np.hstack([self.Presiduals[self.PIDXS], self.Sresiduals[self.SIDXS]])

	def get_index_of_nearest_station(self):
		"""
		Determine index of nearest station (i.e., with fastest P-arrival)

		:return:
			int
		"""
		try:
			return np.nanargmin(self.Ptimes[self.PIDXS])
		except ValueError:
			return np.nanargmin(self.Stimes[self.SIDXS])

	def get_index_of_station_with_shortest_sp_interval(self):
		"""
		Determine index of station with shortest S-P interval

		:return:
			int
		"""
		station_idxs = np.arange(self.num_stations)[self.SPIDXS]
		sp_intervals = self.get_sp_intervals()
		sp_idx = np.argmin(sp_intervals)
		return station_idxs[sp_idx]

	def get_sp_intervals(self):
		"""
		Determine S-P intervals for stations having both P- and
		S-arrivals

		:return:
			1D array
		"""
		sp_intervals = (self.data['Stime'][self.SPIDXS]
						- self.data['Ptime'][self.SPIDXS])

		return sp_intervals

	def get_shortest_sp_interval(self):
		"""
		Determine shortest S-P interval

		:return:
			float
		"""
		sp_intervals = self.get_sp_intervals()
		return sp_intervals.min()

	def get_phase_weights(self):
		"""
		Compute phase weights as product of station weight and
		P- or S-arrival weight

		:return:
			1D array
		"""
		PWT = self.data['Pweight'][self.PIDXS]
		SWT = self.data['Sweight'][self.SIDXS]
		W = np.hstack([PWT, SWT])
		W *= self.stat_weights[self.IDXS]
		return W

	def get_azimuthal_gap(self):
		"""
		Determine azimuthal gap

		:return:
			int, azimuthal gap (in degrees)
		"""
		az_sorted = np.sort(self.data['azimuth'])
		gap = az_sorted[0] + 360 - az_sorted[-1]
		daz = np.hstack([[gap], np.diff(az_sorted)])
		gap = daz.max()

		return gap

	def calc_azimuthal_weights(self, which='phases', normalize=True, verbose=False):
		"""
		Compute azimuthal weights similar to hypo71

		:param which:
			str, what to use for calculating weights, one of 'phases', 'P', 'S'
			or 'stations'
			(default: 'phases', similar to hypo71)
		:param normalize:
			bool, whether or not to normalize weights to max. of 1
			(default: True)
		:param verbose:
			bool, whether or not to print additional information

		:return:
			1D array, azimuthal weights for each station
		"""
		## Determine azimuthal gap
		if which == 'stations':
			AZ = self.data['azimuth']
		elif which == 'phases':
			AZ = self.data['azimuth'][self.IDXS]
		elif which == 'P':
			AZ = self.data['azimuth'][self.PIDXS]
		elif which == 'S':
			AZ = self.data['azimuth'][self.SIDXS]
		J = len(AZ)
		AZ_SORTED = np.sort(AZ)
		GAP = AZ_SORTED[0] + 360 - AZ_SORTED[-1]
		DAZ = np.hstack([[GAP], np.diff(AZ_SORTED)])
		IG = np.argmax(DAZ)
		GAP = DAZ[IG]

		## Define azimuth quadrants
		TX = np.array([0., 90., 180., 270.])
		TX += AZ_SORTED[IG] - 0.5 * GAP
		TX[TX < 0] += 360
		TX[TX > 360] -= 360
		TX = np.sort(TX)
		if verbose:
			print('Azimuthal quadrants: ', TX)

		## Count number of stations in each quadrant
		KEMP = np.digitize(AZ, TX)
		KEMP[KEMP==4] = 0
		TXN = np.bincount(KEMP, minlength=4)
		if verbose:
			print('No. stations/quadrant: ', TXN)

		XN = np.sum(TXN > 0)
		FJ = float(J) / XN
		AZWT = (FJ / TXN[KEMP])

		if normalize:
			AZWT /= AZWT.max()

		return AZWT

	def to_phase_pick_dict(self):
		#Note: distances set by hypo71 are epicentral
		pass

	def get_residuals_and_weights(self, phases='PS'):
		"""
		Fetch residuals and weights

		:param phases:
			str or list of str, phases (P and/or S) for which to get residuals
			(default: 'PS')

		:return:
			(residuals, weights) tuple of arrays
		"""
		residuals, weights = [], []
		for _, phase_data in self.data.iterrows():
			station_weight = float(phase_data['stat_weight'])
			if 'P' in phases and not pd.isna(phase_data['Ptime']):
				residual = float(phase_data['Pres'])
				residuals.append(residual)
				#phase_weight = float(phase_data['Pweight']) * station_weight
				tot_weight = float(phase_data['Ptot_weight'])
				weights.append(tot_weight)
			if 'S' in phases and not pd.isna(phase_data['Stime']):
				residual = float(phase_data['Sres'])
				residuals.append(residual)
				#phase_weight = float(phase_data['Sweight']) * station_weight
				tot_weight = float(phase_data['Stot_weight'])
				weights.append(tot_weight)

		return np.array(residuals), np.array(weights)

	def calc_rmse(self, phases='PS', weighted=True):
		"""
		Compute RMS error

		:param phases:
			see :meth:`get_residuals_and_weights`
		:param weighted:
			bool, whether or not to take into account phase weights
			(default: True)

		:return:
			float, RMSE
		"""
		residuals, weights = self.get_residuals_and_weights(phases=phases)
		if len(residuals):
			squared_residuals = np.power(residuals, 2)
			if weighted:
				try:
					rmse = np.sqrt(np.average(squared_residuals, weights=weights))
				except:
					rmse = np.nan
			else:
				rmse = np.sqrt(np.mean(squared_residuals))

		return rmse

	def get_phase_residuals_table(self, order_by='code'):
		"""
		Generate formatted table of phase residuals

		:param order_by:
			str, column name to order results by, one of 'station',
			'distance', 'azimuth', 'phase', 'residual' or 'weight'
			(default: 'code')

		:return:
			instance of :class:`PrettyTable`
		"""
		from prettytable import PrettyTable

		columns = ['Station', 'Phase', 'Residual (s)', 'Distance (km)',
					'Azimuth', 'Takeoff', 'Pha. Wt', 'Dist. Wt',
					'Az. Wt', 'Tot. Wt']
		tab = PrettyTable(columns)
		residuals, weights = [], []
		if order_by == 'station':
			order_by = 'code'
		if order_by == 'distance':
			order_by = 'Rhypo'
		if order_by in ('code', 'Rhypo', 'Repi', 'azimuth'):
			_order_by = order_by
		else:
			_order_by = None
		for _, phase_data in self.data.sort_values(order_by).iterrows():
			#station = (phase_data['code'].astype('|U10'))
			station = phase_data['code']
			takeoff_angle = float(phase_data['takeoff_angle'])
			azimuth = int(phase_data['azimuth'])
			distance = float(phase_data['Rhypo'])
			az_weight = float(phase_data['az_weight'])
			dist_weight = float(phase_data['dist_weight'])
			station_weight = float(phase_data['stat_weight'])
			if not pd.isna(phase_data['Ptime']):
				phase = 'P'
				residual = float(phase_data['Pres'])
				residuals.append(residual)
				phase_weight = float(phase_data['Pweight']) * station_weight
				tot_weight = float(phase_data['Ptot_weight'])
				weights.append(tot_weight)
				row = [station, phase, residual, distance, azimuth, takeoff_angle,
					phase_weight, dist_weight, az_weight, tot_weight]
				tab.add_row(row)
			if not pd.isna(phase_data['Stime']):
				phase = 'S'
				residual = float(phase_data['Sres'])
				residuals.append(residual)
				phase_weight = float(phase_data['Sweight']) * station_weight
				tot_weight = float(phase_data['Stot_weight'])
				weights.append(tot_weight)
				row = [station, phase, residual, distance, azimuth, takeoff_angle,
					phase_weight, dist_weight, az_weight, tot_weight]
				tab.add_row(row)

		if len(residuals):
			squared_residuals = np.power(residuals, 2)
			rms = np.sqrt(np.mean(squared_residuals))
			wrms = np.sqrt(np.average(squared_residuals, weights=weights))
			row = ['RMS', '', rms, '', '', '', '', '', '', '']
			tab.add_row(row)
			row = ['WRMS', '', wrms, '', '', '', '', '', '', '']
			tab.add_row(row)

		tab.float_format['Residual (s)'] = '5.2'
		tab.float_format['Distance (km)'] = '5.1'
		tab.float_format['Pha. Wt'] = '.2'
		tab.float_format['Az. Wt'] = '.2'
		tab.float_format['Dist. Wt'] = '.2'
		tab.float_format['Tot. Wt'] = '.2'
		tab.float_format['Takeoff'] = '.1'

		tab.align = 'r'
		tab.align['Station'] = 'c'
		tab.align['Phase'] = 'c'

		if order_by == 'phase':
			tab.sortby = 'Phase'
		elif order_by == 'residual':
			tab.sortby = 'Residual (s)'
		elif order_by == 'weight':
			tab.sortby = 'Tot. Wt'

		return tab

	def print_phase_residuals(self, order_by='code', out_file=None, as_html=False):
		"""
		Print table with phase residuals

		:param order_by:
			see :meth:`get_phase_residuals_table`

		:param out_file:
			str, full path to output file (ascii or html)
			(default: None, will print to screen)
		:param as_html:
			bool, whether or not to print as HTML
			(default: False)
		"""
		tab = self.get_phase_residuals_table(order_by=order_by)
		if out_file:
			of = open(out_file, 'w')
			if as_html or os.path.splitext(out_file)[-1].lower()[:4] == '.htm':
				of.write(tab.get_html_string())
			else:
				of.write(tab.get_string())
			of.close()
		else:
			if as_html:
				print(tab.get_html_string())
			else:
				print(tab)

	def remove_outliers(self, residual_threshold=5, verbose=True):
		"""
		Remove picks with residuals larger than given threshold

		:param residual_threshold:
			float, max. allowed residual (in s)
			(default: 5)
		:param verbose:
			bool, whether or not to print some useful information
			(default: True)

		:return:
			(num_invalid_p, num_invalid_s, num_empty_stations) tuple of ints
		"""
		invalid_p = np.abs(self.Presiduals) > residual_threshold
		invalid_p_idxs = np.where(invalid_p)[0]
		num_invalid_p = np.sum(invalid_p)
		self.data['Ptime'][invalid_p] = np.nan
		self.data['Ponset'][invalid_p] = 0
		self.data['Pweight'][invalid_p] = 0
		self.data['Pres'][invalid_p] = 0
		self.data['Ptot_weight'][invalid_p] = 0
		self.PIDXS[invalid_p] = False
		for idx in invalid_p_idxs:
			station = self.stations[idx]
			self.phase_picks[station.code].pop('P')

		invalid_s = np.abs(self.Sresiduals) > residual_threshold
		invalid_s_idxs = np.where(invalid_s)[0]
		num_invalid_s = np.sum(invalid_s)
		self.data['Stime'][invalid_s] = np.nan
		self.data['Sonset'][invalid_s] = 0
		self.data['Sweight'][invalid_s] = 0
		self.data['Sres'][invalid_s] = 0
		self.data['Stot_weight'][invalid_s] = 0
		self.SIDXS[invalid_s] = False
		for idx in sorted(invalid_s_idxs, reverse=True):
			station = self.stations[idx]
			self.phase_picks[station.code].pop('S')

		is_empty_station = (self.PIDXS == False) & (self.SIDXS == False)
		empty_station_idxs = np.where(is_empty_station)[0]
		num_empty_stations = np.sum(is_empty_station)
		if len(empty_station_idxs):
			self.data = self.data[~is_empty_station]
		for idx in sorted(empty_station_idxs, reverse=True):
			station = self.stations[idx]
			self.phase_picks.pop(station.code)
			self.stations.pop(idx)

		if verbose and num_invalid_p + num_invalid_s > 0:
			print('Removed %d invalid P, %d invalid S, %d empty stations'
					% (num_invalid_p, num_invalid_s, num_empty_stations))

		self._define_indexes()

		return (num_invalid_p, num_invalid_s, num_empty_stations)

	def calc_phase_residuals(self, vmodel, focal_depth, origin_time):
		"""
		Compute phase residuals from epicentral distances and arrival times

		:param vmodel:
			instance of :class:`robspy.location.CrustalVelocityModel`
		:param focal_depth:
			float, focal depth (in km)
		:param origin_time:
			datetime spec understood by robspy, origin time
			If None, it will be computed from weighted average travel times
			and returned

		:return:
			None, :prop:`data` is modified in-place
		"""
		from ..time import parse_datetime

		Zf = focal_depth
		Repi = self.data.Repi.to_numpy()
		assert not np.isclose(Repi, 0).all()
		Zs = self.stat_depths

		Pidxs = self.PIDXS
		Ptimes = self.Ptimes[Pidxs]
		Sidxs = self.SIDXS
		Stimes = self.Stimes[Sidxs]

		if origin_time is not None:
			Tf = parse_datetime(origin_time)
		else:
			Tf = parse_datetime(Ptimes.min())

		Vidx = np.zeros(len(Ptimes), dtype=int)
		Ptt, Ptt_res = vmodel.calc_tt_residuals(Zf, Tf, Repi[Pidxs], Zs[Pidxs],
													Ptimes, Vidx)

		Vidx = np.ones(len(Stimes), dtype=int)
		Stt, Stt_res = vmodel.calc_tt_residuals(Zf, Tf, Repi[Sidxs], Zs[Sidxs],
													Stimes, Vidx)

		if origin_time is not None:
			self.data.loc[Pidxs, 'Pres'] = Ptt_res
			self.data.loc[Sidxs, 'Sres'] = Stt_res

		else:
			Pt0 = np.array([t.timestamp for t in Ptimes]) - Ptt
			St0 = np.array([t.timestamp for t in Stimes]) - Stt
			Pweights = self.data.Ptot_weight[Pidxs]
			Sweights = self.data.Stot_weight[Sidxs]
			t0 = np.concatenate([Pt0, St0])
			weights = np.concatenate([Pweights, Sweights])
			origin_time = parse_datetime(np.average(t0, weights=weights))
			Ptt_obs = Ptimes - origin_time
			Stt_obs = Stimes - origin_time
			self.data.loc[Pidxs, 'Pres'] = Ptt_obs - Ptt
			self.data.loc[Sidxs, 'Sres'] = Stt_obs - Stt

			return origin_time

	def plot_phase_residuals(self, color_by_azimuth=False, **kwargs):
		"""
		Plot phase residuals versus distance

		:param color_by_azimuth:
			bool, whether or not to apply thematic coloring by phase azimuth
			(default: False)
		:kwargs:
			additonal keyword arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		import matplotlib
		from plotting.generic_mpl import plot_xy

		distances = self.data['Rhypo'].to_numpy()
		azimuths = self.data['azimuth'].to_numpy()
		Pidxs = ~self.data['Ptime'].isna()
		Sidxs = ~self.data['Stime'].isna()
		Presiduals = self.data['Pres'][Pidxs].to_numpy()
		Sresiduals = self.data['Sres'][Sidxs].to_numpy()

		datasets = [(distances[Pidxs], Presiduals), (distances[Sidxs], Sresiduals)]
		labels = kwargs.pop('labels', ['P', 'S'])
		markers = kwargs.pop('markers', ['o', 's'])
		linestyles = kwargs.pop('linestyles', ['', ''])
		if color_by_azimuth:
			## Cyclic colormap
			az_colors = matplotlib.cm.hsv(azimuths / 360.)
			colors = [az_colors[Pidxs], az_colors[Sidxs]]
		else:
			colors = []

		xlabel = kwargs.pop('xlabel', 'Distance (km)')
		ylabel = kwargs.pop('ylabel', 'Residual (s)')
		xgrid = kwargs.pop('xgrid', 1)
		ygrid = kwargs.pop('ygrid', 1)
		xmin = kwargs.pop('xmin', 0)

		return plot_xy(datasets, labels=labels, markers=markers,
						linestyles=linestyles, colors=colors,
						xlabel=xlabel, ylabel=ylabel, xgrid=xgrid, ygrid=ygrid,
						xmin=xmin, **kwargs)

	def plot_phase_arrival_times(self, origin_time, **kwargs):
		"""
		Plot phase arrival times versus distance

		:param origin_time:
			time spec understood by :func:`robspy.parse_datetime`,
			earthquake origin time
		:kwargs:
			additonal keyword arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy
		from ..time import parse_datetime

		distances = self.data['Rhypo'].to_numpy()
		origin_time = parse_datetime(origin_time)
		Pidxs = ~self.data['Ptime'].isna()
		Sidxs = ~self.data['Stime'].isna()
		Parrival_times = [parse_datetime(val) for val in self.data['Ptime'][Pidxs]]
		Sarrival_times = [parse_datetime(val) for val in self.data['Stime'][Sidxs]]
		Ptraveltimes = np.array([Pt - origin_time for Pt in Parrival_times])
		Straveltimes = np.array([St - origin_time for St in Sarrival_times])

		datasets = [(distances[Pidxs], Ptraveltimes),
						(distances[Sidxs], Straveltimes)]
		labels = kwargs.pop('labels', ['P', 'S'])
		markers = kwargs.pop('markers', ['o', 's'])
		linestyles = kwargs.pop('linestyles', ['', ''])

		xlabel = kwargs.pop('xlabel', 'Distance (km)')
		ylabel = kwargs.pop('ylabel', 'Travel time (s)')
		xgrid = kwargs.pop('xgrid', 1)
		ygrid = kwargs.pop('ygrid', 1)
		xmin = kwargs.pop('xmin', 0)
		ymin = kwargs.pop('ymin', 0)

		return plot_xy(datasets, labels=labels, markers=markers,
						linestyles=linestyles, xlabel=xlabel, ylabel=ylabel,
						xgrid=xgrid, ygrid=ygrid, xmin=xmin, ymin=ymin, **kwargs)

#################
#Additional helpers as part of redbelly
#################

def build_pick_dict_from_event(ev):
    """
    Extract pick information from an ObsPy Event object into a dictionary
    keyed by station code, with nested phase picks.

    Parameters
    ----------
    ev : obspy.core.event.Event
        Event containing picks and origins.

    Returns
    -------
    origin_info : dict
        Dictionary with 'latitude', 'longitude', 'depth_km', and 'time'.
    pick_dict : dict
        Nested dict: {station_code: {phase: PhasePick(...)}}
    picked_stations : set
        Set of station codes that have picks.
    """
    # --- Extract origin ---
    origin = ev.preferred_origin() or ev.origins[0]
    origin_info = dict(
        latitude=origin.latitude,
        longitude=origin.longitude,
        depth_km=origin.depth / 1000.0 if origin.depth else 5.0,
        time=origin.time,
    )

    # --- Build pick_dict ---
    pick_dict = {}
    for pick in ev.picks:
        sta = pick.waveform_id.station_code
        phase = pick.phase_hint or "P"
        if sta not in pick_dict:
            pick_dict[sta] = {}
        pick_dict[sta][phase] = PhasePick(
            phase_name=phase,
            datetime=pick.time,
            station_code=sta,
            station_network=pick.waveform_id.network_code,
            station_location=pick.waveform_id.location_code,
        )

    picked_stations = set(pick_dict.keys())
    return origin_info, pick_dict, picked_stations