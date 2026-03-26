"""
single.py Python implementation of hypo71
"""

import numpy as np
from obspy import UTCDateTime

from hypo71py.model.station_phase import (
    Station,
    StationPhases,
    PhasePick,
)

__all__ = [
    "SINGLE",
    "CALC_EPICENTRAL_XY_OFFSET",
    "SHIFT_LONLAT_BY_XY",
]


## This python version of SINGLE now produces exactly the same results as
## the Fortran version of HYPO71 from seiscomp, if all parameters are the same.
## TRVDRV and SWMREG give identical results (within precision) as Fortran
## subroutines compiled with f2py and called from python
## CALC_EPICENTRAL_XY_OFFSET and SHIFT_LONLAT_BY_XY take geographic coordinates
## in degrees rather than in minutes as in the original version, but have been
## verified to produce essentially the same results
## Residuals, azimuths and takeoff angles calculated for the same fixed origin
## are the same as those reported in HYPO71 output file; total weights are also
## very similar, within precision.



"""
SINGLE: Solution for a single earthquake
	TEST: hypo71 options
	KNO: index of crustal structure model to use for station delay
		in variable first-layer model

	IW: station weights [num_stations]
	NSTA: station names [num_stations]
	INS: North or South for each station [num_stations]
	IEW: East or West for each stations [num_stations]
	DLY: station delays for 2 crustal structure models [2, num_stations]
	FMGC: station corrections for FMAG (can be ignored)
	XMGC: station corrections for XMAG (can be ignored)
	KLAS: index of predefined calibration curve for each station
		(can be ignored)
	PRR: standard period for XMAG (can be ignored)
	CALR: standard calibration for XMAG (can be ignored)
	ICAL: calibration indicator (can be ignored)
	LAT: station latitudes [num_stations]
	LON: station longitudes [num_stations]

	V: P-wave velocities in km/s [num_layers]
	D: depth to top of layers in km [num_layers]
	DEPTH: = D [num_layers]
	VSQ: squared P-wave velocities [num_layers]
	THK: layer thicknesses?? [num_layers]
	H: = THK [num_layers]
	G: G-term, computed from V and VSQ in input1 subroutine [4, num_layers]
	F: F-term (1 or 2), computed in input1 [num_layers, num_layers]
	TID: time ...? computed in input1 [num_layers, num_layers]
	DID: distance ...? computed in input1 [num_layers, num_layers]

	FLT: ? computed in input1 [2, num_stations] (used for variable first layer model)
	QSPA: ? read in input1 from HYPO71PC.INP [9, 40]
		(not used, can be ignored)

	MSTA: station names for phases [num_phases]
	PRMK: P remark, string part only [num_phases]
	W: weights of P arrivals [num_phases]
	JMIN: common time base for P and S arrivals in minutes [num_phases]
	P: seconds of P arrivals [num_phases]
	S: seconds of S arrivals [num_phases]
	SRMK: S remark, string part only [num_phases]
	WS: weights of S arrivals [num_phases]
	AMX: max. peak-to-peak amplitude in mm (can be ignored)
	PRX: period of max. peak-to-peak amplitude in s (can be ignored)
	CALX: peak-to-peak ampl. of 10 uV calibration signal in s
		(can be ignored)
	RMK: remark for each phase card (can be ignored)
	DT: time correction in seconds [num_phases]
	FMP: F-P time (=eq duration) in seconds (can be ignored)
	AZRES: ?, 4-char string [num_phases] (can probably be ignored)
	SYM: ?, char [num_phases] (can probably be ignored)
	QRMK: ?, char [num_phases] (can probably be ignored)

	KDX: station index [num_phases]
	LDX: whether or not phase data contains S, 0 or 1 [num_phases]
	JDX: ?, 0 or 1, set in input2 subroutine [num_stations]
	TP: P arrival times in seconds [num_phases]
	WRK: ?, string [num_phases]
	KSMP: array holding 1 (for S-P interval data) or 0 (P or S) [num_stations]
		contains only ones in the beginning, but zeros are appended
		if there are S data
	TS: S arrival times in seconds [num_phases]
	TIME1: ?, used to check if earthquakes are in chronological order [1]
		(can be ignored)
	TIME2: common time base (date + hour) for all stations [1],
		only used for output

	AVXM: average of XMAG of available stations [1] (can be ignored)
	AVFM: average of FMAG of available stations [1] (can be ignored)
	XMAG: XMAG [num_phases] (can be ignored)
	FMAG: FMAG [num_phases] (can be ignored)
	NRES: ?, integer [2, num_stations] (1st dim: P or S)
	SR: travel time residuals [2, num_stations] (1st dim: P or S)
	SRSQ: squared travel time residuals [2, num_stations] (1st dim: P or S)
	SRWT: residual weights? [2, num_stations] (1st dim: P or S)
	QNO: array with number of earthquakes in each quality class [4] (can be ignored)
	MNO: preferred crustal structure model [num_stations]
"""


def CALC_EPICENTRAL_XY_OFFSET(LONEP, LATEP, LON, LAT):
	"""
	Compute X/Y components of epicentral distance using Richter's method

	:param LONEP:
		float, epicenter longitude (in degrees)
	:param LATEP:
		float, epicenter latitude (in degrees)
	:param LON:
		1D array, phase longitudes (in degrees)
	:param LAT:
		1D array, phase latitudes (in degrees)

	:return:
		(DX, DY) tuple of 1D arrays, X/Y offsets in km
	"""
	## ------- CALCULATE EPICENTRAL DISTANCE BY RICHTER'S METHOD -------------

	#LONEP, LATEP = LONEP * 60, LATEP * 60
	#LON, LAT = LON * 60, LAT * 60

	## Note: division by 2 instead of 120 because we have degrees instead of minutes
	PHI = 0.0174532 * ((LAT + LATEP) / 2.)
	#PHI = 0.0174532 * ((LAT + LATEP) / 120.)
	SINPHI = np.sin(PHI)
	SINP2 = SINPHI * SINPHI
	SINP4 = SINP2 * SINP2
	CA = 1.8553654 + 0.0062792*SINP2 + 0.0000319*SINP4
	CB = 1.8428071 + 0.0187098*SINP2 + 0.0001583*SINP4
	## Note: multiplication by 60 because we have degrees instead of minutes
	DX = (LON - LONEP) * 60 * CA * np.cos(PHI)
	DY = (LAT - LATEP) * 60 * CB
	#DX = (LON - LONEP) * CA * np.cos(PHI)
	#DY = (LAT - LATEP) * CB

	return (DX, DY)

def promote_to_sp_pairs(pick_dict):
    """
    Convert {station: {'P': PhasePick, 'S': PhasePick}} →
            {station: {'P': PhasePick}} where the .datetime
    now stores the S–P interval (in seconds).

    Any station missing either P or S is dropped.
    """
    new_dict = {}

    for sta, phases in pick_dict.items():
        p = phases.get("P")
        s = phases.get("S")

        # Require both
        if p is None or s is None:
            continue

        # Use .datetime from PhasePick
        ptime = getattr(p, "datetime", None)
        stime = getattr(s, "datetime", None)
        if ptime is None or stime is None:
            continue

        sp = stime - ptime  # should yield seconds (float)
        if not (np.isfinite(sp) and sp > 0):
            continue

        # Create a new PhasePick representing S–P interval
        new_pick = PhasePick(
            phase_name="P",  # still treated as P for TRVDRV
            datetime=UTCDateTime(sp),  # store interval as time
            station_code=p.station_code,
            id_earth=p.id_earth if hasattr(p, "id_earth") else None
        )

        new_dict[sta] = {"P": new_pick}

    return new_dict

def SHIFT_LONLAT_BY_XY(LONEP, LATEP, DX, DY):
	"""
	Shift position in geographic coordinates by given X/Y offset

	:param LONEP:
		float, epicenter longitude (in degrees)
	:param LATEP:
		float, epicenter latitude (in degrees)
	:param DX:
		float, X offset (in km)
	:param DY:
		float, Y offset (in km)

	:return:
		(LONEP, LATEP) tuple of floats:
			shifted lon/lat of epicenter
	"""
	#LONEP, LATEP = LONEP * 60, LATEP * 60

	PHI = 0.0174532 * LATEP
	#PHI = 0.0174532 * (LATEP / 60.)
	SINPHI = np.sin(PHI)
	SINP2 = SINPHI * SINPHI
	SINP4 = SINP2 * SINP2
	CA = 1.8553654 + 0.0062792 * SINP2 + 0.0000319 * SINP4
	CB = 1.8428071 + 0.0187098 * SINP2 + 0.0001583 * SINP4
	LATEP += (DY / (CB * 60.))
	LONEP += (DX / (CA * 60. * np.cos(PHI)))
	#LATEP += (DY / CB)
	#LONEP += (DX / (CA * np.cos(PHI)))

	#LONEP, LATEP = LONEP / 60., LATEP / 60.

	return (LONEP, LATEP)


def SINGLE(stations, pick_dict, velocity_model,
			ZTR, origin=(),
			ISW='',
			max_num_iterations=100, min_hypocentral_adjustment=0.05,
			use_s_picks=True,
			use_s_minus_p=False,
			fix_depth=False, fix_origin=False, max_altitude=2.,
			azimuthal_weighting=True, XNEAR=50., XFAR=200.,
			min_rms_jeffrey=0.1, max_horizontal_adjustment=100.,
			max_horizontal_adjustment_for_depth_adjustment=10.,
			max_depth_adjustment=5., depth_above_max_altitude_adjustment=0.5,
			f_crit=2., f_crit_divisor=4.,
			compute_auxilliary_rms=False,
			apply_al_lindh_mod=True, force_hypo71_times=False,
			min_num_phases=4, use_fortran_speedups=True,
			station_elevation='absolute',
			verbose=0):
	"""
	Locate a single earthquake

	:param stations:
		list with instances of :class:`Station`
	:param pick_dict:
		dict, mapping station codes to dicts, in turn mapping
		phase names ('P' or 'S') to phase picks
	:param velocity_model:
		instance of :class:`CrustalVelocityModel`
	:param ZTR:
		float, trial focal depth (in km)
	:param origin:
		(lon, lat, time) tuple with initial or fixed hypocentral location
		and origin time,
		(default: ())
	:param ISW:
		str, selection card:
		- blank: use station delay model
		- '1': use variable first layer model
	:param max_num_iterations:
		int, maximum number of iterations in the hypocentral adjustment.
		Corresponds to TEST[11] in original program
		(default: 100)
	:param min_hypocentral_adjustment:
		float, minimum hypocentral adjustment (in km). If hypocentral
		adjustment is less than this value, Geiger's iteration is
		terminated.
		Corresponds to TEST[04] in original program
		(default: 0.05)
	:param use_s_picks:
		bool, whether or not to use S phase picks for the location
		Corresponds to KNST option in original program
		(default: True)
	:param fix_depth:
		bool, whether or not to fix hypocentral depth to :param:`ZTR`
		Corresponds to INST=1 option in original program
		(default: False)
	:param fix_origin:
		bool, whether or not to fix hypocentral location and origin time
		in :param:`origin`, and just compute travel-time residuals.
		Implies :param:`fix_depth`= True.
		Corresponds to INST=9 in original program
		(default: False)
	:param max_altitude:
		float, maximum altitude of earthquake in km
		(default: 2.)
	:param azimuthal_weighting:
		bool, whether or not azimuthal weighting should be applied
		Corresponds to KAZ option in original program
		(default: True)
	:param XNEAR:
		float, epicentral distance up to which distance weighting is 1
		(in km)
		(default: 50)
	:param XFAR:
		float, epicentral distance beyond which distance weighting is 0
		(in km)
		(default: 200)
	:param min_rms_jeffrey:
		float, cutoff value for RMS below which Jeffreys' weighting
		of residuals is not used. It should be set to a value
		approximately equal to the overall timing accuracy of
		P arrival times (in seconds)
		Corresponds to TEST[01] in original program
		(default: 0.1)
	:param max_horizontal_adjustment:
		float, if the latitude or longitude adjustment (DX or DY)
		is greater than this value, then DX is reset to DX / (J+1)
		and DY is reset to DY / (J+1), where J = D / max_horizontal_adjustment,
		D being the larger of DX or DY.
		Corresponds to TEST[10] in original program
		(default: 100.)
	:param max_horizontal_adjustment_for_depth_adjustment:
		float, for each iteration step, if horizontal adjustment is
		larger than or equal to this value, this step is recalculated
		without focal_depth adjustment. This value should be set to
		a value approximately equal to station spacing in km.
		Corresponds to TEST[02] in original program
		(default: 10)
	:param max_depth_adjustment:
		float, if the focal-depth adjustment (DZ) is greater than this
		value, DZ = reset to DZ / (K + 1),
		where K = DZ / max_depth_adjustment
		Should be set to a value approximately equal to half the range
		of focal depth expected.
		Corresponds to TEST[05] in original program
		(default: 5.)
	:param depth_above_max_altitude_adjustment:
		float, if the focal-depth adjustment (DZ) would place the
		hypocenter in the air (or: above :param:`max_altitude`),
		then DZ is reset to DZ = -Z * depth_above_max_altitude_adjustment,
		where Z is the focal depth.
		Corresponds to TEST[12] in original program
		(default: 0.5)
	:param f_crit:
		float, critical F-value for the stepwise multiple regression.
		Should be set according to the number and quality of P- and S-
		arrivals. A value between 0.5 and 2 is recommended.
		If set to zero, a simple multiple regression is performed
		regardless of whether the matrix is ill-conditioned. This is
		not desirable because the hypocenter solution may be meaningless.
		On the other hand, if it is set to 2 or greater, then Geiger's
		iteration may be terminated prematurely, before a good hypocenter
		is found.
		Corresponds to TEST[03] option in original program
		(default: 2.)
	:param f_crit_divisor:
		float, division factor for :param:`f_crit` if no significant
		variable is found in the stepwise multiple regression. In that
		case, f_crit is reduced to f_crit / f_crit_divisor, and the
		regression is repeated. If it is <= 1, then the regression is
		repeated to find one variable, and the adjustment is made only
		if it is greater than f_crit_divisor * standard error.
		If :param:`f_crit` is set to less than 2, then f_crit_divisor
		should be set to 1.
		Corresponds to TEST[06] option in original program
		(default: 4.)
	:param compute_auxilliary_rms:
		bool, whether or not to calculate auxiliary RMS values at
		ten points on a sphere centered at the hypocenter.
		This helps to determine if the solution is at the RMS minimum.
		Corresponds to KTEST option in original program
		(default: False)
	:param apply_al_lindh_mod:
		bool, whether or not to apply Al Lindh's modification
		(default: True)
	:param force_hypo71_times:
		bool, whether or not hypo71 limitations on arrival times
		(S without P not supported, resort to S-P interval if S and P
		don't have a common time base) should be applied
		(default: False)
	:param min_num_phases:
		int, minimum number of phases required to compute a location
		(4 in seiscomp version, 3 in ROB version)
		(default: 4)
	:param use_fortran_speedups:
		bool, whether or not to use Fortran speedups for routines
		TRVDRV and SWMREG, if available
		(default: True)
	:param station_elevation:
		str, how to handle station elevation/depth, one of:
		- 'absolute': use negative absolute elevation
		- 'relative': use negative elevation with respect to mean
			station elevation
		- 'depth': use depth with respect to the surface
		- 'zero': set all station depths to zero
		(default: 'absolute')
	:param verbose:
		int, indicator for printed output:
		0 for no output
		1 for final solution and station residuals
		2 for above plus one line per iteration
		3 for above plus station residuals per iteration
		4 for above plus details from stepwise multiple regression
		Corresponds to IPRN option in original program
		(default: 0)

	:return:
		(LONEP, LATEP, Z, ORG, SE, station_phases, QSD, NI, RMS) tuple
		- LONEP: float, longitude of epicenter
		- LATEP: float, latitude of epicenter
		- Z: float, hypocentral depth
		- ORG: instance of :class:`obspy.UTCDateTime`, origin time
		  (None if non-finite)
		- SE: 1-D array [4], standard deviations on X, Y, Z position
		  and time
		- station_phases: instance of :class:`StationPhases`,
		  containing residuals, distances, azimuths, weights
		  and angles of incidence
		- QSD: str (char QS + char QD), quality class of hypocentral
		  solution; 'DD' if insufficient data
		- NI: int, number of iterations performed
		- RMS: float, root-mean-square travel-time residual (seconds);
		  0.0 if insufficient data
	"""
	from hypo71py.model.time import parse_datetime
	from .azwtos import AZWTOS

	if use_fortran_speedups:
		try:
			from .fortran.trvdrv_wrapper import TRVDRV
		except ImportError:
			print('Fortran version of TRVDRV not available!')
		else:
			if verbose:
				print('Using Fortran version of TRVDRV')
	try:
		TRVDRV
	except NameError:
		from .trvdrv import TRVDRV

	if use_fortran_speedups:
		try:
			from .fortran.swmreg_wrapper import SWMREG
		except ImportError:
			print('Fortran version of SWMREG not available!')
		else:
			if verbose:
				print('Using Fortran version of SWMREG')
	try:
		SWMREG
	except NameError:
		from .swmreg import SWMREG

	IPRN = verbose - 1

	if verbose:
		## ROB version
		#print(' ***** PROGRAM: HYPO2000 (Version 1: 01/04/00) *****')
		#print(' ***** PROGRAM: HYPO71PC (Version 1: 11/29/85) *****')
		## Seiscomp version
		#print(' ***** PROGRAM: HYPO71PC (Version 2: ********) *****')
		## Python version
		print(' ***** PROGRAM: PYHYPO71 (Version 1: ********) *****')

	if fix_origin:
		assert origin
		fix_depth = True

	# TODO: allow fix_origin = True and fix_depth = False...
	#fix_origin_and_depth = fix_origin and fix_depth

	if fix_origin:
		INST = 9
	elif fix_depth:
		INST = 1
	else:
		INST = 0

	if use_s_picks:
		KNST = 1
	else:
		KNST = 0

	#treats S-P as if we only had P obervations
	#if use_s_minus_p:
	#pick_dict = promote_to_sp_pairs(pick_dict)
	if use_s_minus_p:
		pick_dict = promote_to_sp_pairs(pick_dict)
	## Set up array with "TEST" variables
	## Default HYPO71 values
	TEST_DEF = np.zeros(20, dtype='f')
	TEST_DEF[0] = 0.1
	TEST_DEF[1] = 10
	TEST_DEF[2] = 2
	TEST_DEF[3] = 0.05
	TEST_DEF[4] = 5
	TEST_DEF[5] = 4
	#TEST_DEF[6] = -0.87
	#TEST_DEF[7] = 2.
	#TEST_DEF[8] = 0.0035
	TEST_DEF[9] = 100
	TEST_DEF[10] = 8
	TEST_DEF[11] = 0.5
	TEST_DEF[12] = 1
	TEST_DEF[14] = 0
	TEST_DEF[19] = 1

	TEST = np.zeros(20, dtype='f')
	TEST[0] = min_rms_jeffrey
	TEST[1] = max_horizontal_adjustment_for_depth_adjustment
	TEST[2] = f_crit
	TEST[3] = min_hypocentral_adjustment
	TEST[4] = max_depth_adjustment
	TEST[5] = f_crit_divisor
	## TEST[6,7,8] are related to magnitude calculation, not used here
	TEST[9] = max_horizontal_adjustment
	TEST[10] = max_num_iterations
	TEST[11] = depth_above_max_altitude_adjustment
	TEST[12] = 1.  ## radius for calculation of auxilliary RMS
	TEST[14] = -max_altitude  ## seiscomp only
	TEST[19] = 1  ## seiscomp option for scaling station elevations, not used here

	if verbose:
		for i in range(20):
			if not np.isclose(TEST[i], TEST_DEF[i]):
				print('RESET TEST(%02d): %G -> %G' % (i+1, TEST_DEF[i], TEST[i]))

	## ------- SQUARE SOME TEST-VARIABLES FOR LATER USE ----------------------
	## From input1.f
	#    8 TEST(1)=TEST(1)**2
	#      TEST(2)=TEST(2)**2
	#      TEST(4)=TEST(4)**2

	TEST[0] **= 2
	TEST[1] **= 2
	TEST[3] **= 2

	## Create structured array with station and phase data
	#station_phases = StationPhases(stations, pick_dict, use_s_picks=use_s_picks)
	station_phases = StationPhases(stations, pick_dict, use_s_picks=not use_s_minus_p)
	NS = station_phases.num_stations
	## P and S indexes
	PIDXS = station_phases.PIDXS
	SIDXS = station_phases.SIDXS
	#print('PIDXS', PIDXS)
	#print('SIDXS', SIDXS)
	## Station index for each phase
	KIDXS = station_phases.IDXS
	#print('KIDXS', KIDXS)
	## Velocity index for each phase
	VIDXS = station_phases.VIDXS
	#print('VIDXS', VIDXS)
	## Index indicating S - P data
	#KSMP = np.zeros_like(KIDXS)s
	# ---------------------------------------------------------------------
	# If S–P mode, preprocess pick_dict before anything else
	# ---------------------------------------------------------------------
	#if use_s_minus_p:
	#	pick_dict = promote_to_sp_pairs(pick_dict)
	#	KSMP = np.ones(len(pick_dict), dtype=bool)  # every station is S–P
	#else:
#		KSMP = np.zeros(len(pick_dict), dtype=bool)  # or np.zeros(len(pick_dict), dtype=bool)

	#print(station_phases)
	#print(station_phases.stat_lons[KIDXS])

	# ------- TREAT S DATA BY AUGMENTING P DATA -----------------------------

	## What happens in the following lines:
	## if there are S data (IDXS != 0):
	## - append S times and weights to P times and weights
	## - set S weight to zero if KNST flag is set to not use S data
	## - zeros are appended to KSMP
	## - KDX indexes for S phases are set to those for the P phases
	## - LDX indexes are incremented
	## - WRK ??
	## Normally, we shouldn't need any of this

	#   30 IF (IDXS .EQ. 0) GO TO 80
	#      NOS=0
	#      DO 65 I=1,NRP
	#      IF (LDX(I) .EQ. 0) GO TO 65
	#      NOS=NOS+1
	#      NRS=NRP+NOS
	#      TP(NRS)=TS(I)
	#      W(NRS)=WS(I)
	#      KSMP(NRS)=0
	#      IF ((KNST.NE.1).AND.(KNST.NE.6)) W(NRS)=0.
	#      KDX(NRS)=KDX(I)
	#      LDX(I)=NRS
	#      WRK(NRS)='    '
	#   65 CONTINUE
	#      NR=NRP+NOS

	TPS = np.array([parse_datetime(t).timestamp
						for t in station_phases.arrival_times])
	W = station_phases.get_phase_weights().astype('f')

	# Always define KSMP as a phase-length int array (0 = not S–P, 1 = S–P)
	#NR = len(KIDXS)
	#KSMP = np.zeros(NR, dtype=int)   # <--- THIS fixes the AttributeError
	#if use_s_minus_p:
	#	pick_dict = promote_to_sp_pairs(pick_dict)
	#	KSMP = np.ones(len(pick_dict), dtype=bool)  # every station is S–P
	NR = len(station_phases.arrival_times)
	KSMP  = np.ones(NR, int) if use_s_minus_p else np.zeros(NR, int)
	
	#print("\n--- SHAPE CHECK ---")
	#print(f"NR={NR}")
	#print(f"KIDXS={KIDXS.shape}, VIDXS={VIDXS.shape}, W={W.shape}, TPS={TPS.shape}, KSMP={KSMP.shape}")
	#print(f"VIDXS unique={np.unique(VIDXS)}  (0=P, 1=S)")
	#print(f"Num S–P={KSMP.sum()}")

	assert len(KIDXS) == len(VIDXS) == len(W) == len(TPS) == len(KSMP)

	if force_hypo71_times:
		idxs_to_remove = []
		TP = station_phases.Ptimes[PIDXS]
		TS = station_phases.Stimes[SIDXS]
		for p, (Ptime, Stime) in enumerate(zip(TP, TS)):
			s = p + NS
			if not np.isnan(Stime):
				St = UTCDateTime(Stime)
				if np.isnan(Ptime):
					## Skip, S without P not supported by hypo71
					SIDXS[s] = False
					idxs_to_remove.append(s)
				else:
					Pt = UTCDateTime(Ptime)
					St_min, St_sec = St.minute, St.second
					if St.hour != Pt.hour:
						St_min += 60
					if St_min != Pt.minute:
						St_sec += 60 * (St_min - Pt.minute)
						if St_sec >= 100:
							## No common time base, replace with S-P interval
							TPS[s] = Stime - Ptime
							SIDXS[s] = False
							VIDXS[s] = 0
							KSMP[s] = 1

		TPS = np.delete(TPS, idxs_to_remove)
		KIDXS = np.delete(KIDXS, idxs_to_remove)
		KSMP = np.delete(KSMP, idxs_to_remove)
		VIDXS = np.delete(VIDXS, idxs_to_remove)
		W = np.delete(W, idxs_to_remove)

	## NR: number of phase records
	## NRP: number of P phases
	## NOS: number of S phases

	NRP = np.sum(PIDXS)
	NOS = np.sum(SIDXS)
	NR = NRP + NOS
	if verbose:
		print('Using %d phases: %d P + %d S + %d S-P'
				% (NR, NRP, NOS, np.sum(KSMP)))

	#      DATA WF/.95,0.95,0.95,0.95,0.95,0.95,0.94,0.94,0.94,0.93,
	#     1       0.92,0.92,0.91,0.90,0.88,0.87,0.85,0.83,0.80,0.77,
	#     2       0.73,0.69,0.64,0.59,0.53,0.47,0.41,0.34,0.28,0.23,
	#     3       0.18,0.14,0.11,0.08,0.06,0.04,0.03,0.02,0.01,0.01,0./
	#      DATA LA/1,1,1,1,0,0,-1,-1,-1,-1/,
	#     1     LO/+1,-1,+1,-1,0,0,+1,-1,+1,-1/,
	#     2     ALZ/-1.0,-1.0,+1.0,+1.0,-1.732,+1.732,-1.0,-1.0,+1.0,+1.0/

	WF = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0.94,0.94,0.94,0.93,
					0.92,0.92,0.91,0.90,0.88,0.87,0.85,0.83,0.80,0.77,
					0.73,0.69,0.64,0.59,0.53,0.47,0.41,0.34,0.28,0.23,
					0.18,0.14,0.11,0.08,0.06,0.04,0.03,0.02,0.01,0.01,0.],
					dtype='f')
	LA = np.array([1,1,1,1,0,0,-1,-1,-1,-1], dtype='int')
	LO = np.array([+1,-1,+1,-1,0,0,+1,-1,+1,-1], dtype='int')
	ALZ = np.array([-1.0,-1.0,+1.0,+1.0,-1.732,+1.732,-1.0,-1.0,+1.0,+1.0],
					dtype='f')

	#      LATRT=0.
	#      LONRT=0.
	#      LATSV=0.
	#      LONSV=0.
	#  778 AVRPS = 0.0

	LATRT, LONRT = 0., 0.
	LATSV, LONSV = 0., 0.
	ZSV = 0.
	## Average P and S residual
	AVRPS = 0.
	## Exit value used in main routine to determine if magnitude residuals
	## should be computed, not used here
	#IEXIT = 0


	## Read KNST, INST and ZRES from "instruction card" at end of phase list
	## (which is usually blank)
	## Not used here
	## ZRES should be trial focal depth, overriding overall ZTR parameter
	## KNST indicates if S picks should be used (1 or 6) or not (0 or 5)
	## INST indicates if depth should be fixed (1) or not (0) or if
	## hypocentral position should be fixed completely (9)

	#      ZRES=P(NR+1)
	#      KNST=JMIN(NR+1)/10
	#      INST=JMIN(NR+1)-KNST*10
	#      NRP=NR


	## -------- INITIALIZE SUMMARY OF RESIDUALS ------------------------------

	## The following is done in main.f, but belongs here
	#   44 DO 48 L=1,NS
	#      NRES(1,L)=0
	#      NRES(2,L)=0
	#      NXM(L)=0
	#      NFM(L)=0
	#      SR(1,L)=0.
	#      SR(2,L)=0.
	#      SRSQ(1,L)=0.
	#      SRSQ(2,L)=0.
	#      SRWT(1,L)=0.
	#      SRWT(2,L)=0.
	#      SXM(L)=0.
	#      SXMSQ(L)=0.
	#      SFM(L)=0.
	#      SFMSQ(L)=0.
	#   48 CONTINUE
	#      DO 49 I=1,4
	#   49 QNO(I)=0.

	NRES = np.zeros((2, NS), dtype='int')
	#NXM = np.zeros(NS, dtype='int')
	#NFM = np.zeros(NS, dtype='int')
	SR = np.zeros((2, NS), dtype='f')
	SRSQ = np.zeros_like(SR)
	SRWT = np.zeros((2, NS), dtype='f')
	#SXM = np.zeros(NS, dtype='float64')
	#SXMSQ = np.zeros_like(SXM)
	#SFM = np.zeros(NS, dtype='float64')
	#SFMSQ = np.zeros_like(SFM)
	#QNO = np.zeros(4, dtype='int')


	## Not explicitly initialized in original program
	XMEAN = np.zeros(4, dtype='f')
	Y = np.zeros(4, dtype='f')

	#      XFN=XFAR-XNEAR+0.000001
	XFN = float(XFAR - XNEAR)

	LAT = station_phases.stat_lats[KIDXS]
	LON = station_phases.stat_lons[KIDXS]

	## Different ways to handle station elevation/depth
	if station_elevation == 'absolute':
		ELEV = -station_phases.stat_elevs[KIDXS]
	elif station_elevation == 'relative':
		mean_elev = np.mean(station_phases.stat_elevs)
		ELEV = mean_elev - station_phases.stat_elevs[KIDXS]
	elif station_elevation == 'depth':
		ELEV = station_phases.stat_depths[KIDXS]
	elif station_elevation == 'zero':
		ELEV = np.zeros(NR)
	#print(ELEV)


	## From INPUT2 subroutine
	## 0 or blank = full weight
	## 1 = 3/4 weight
	## 2 = 1/2 weight
	## 3 = 1/4 weight
	## 4 = no weight
	#      W(L)=(4.-W(L))/4.

	## We don't use this, as we consider weights are floats already
	#W = (4. - W) / 4.

	DLY = np.zeros((2, NR), dtype='f')
	DLY[0] = station_phases.data['delay1'][KIDXS]
	DLY[1] = station_phases.data['delay2'][KIDXS]

	GAP = 0

	## ------- INITIALIZE TRIAL HYPOCENTER -----------------------------------
	#   80 K=KDX(NEAR)
	#      SVY1 = 0.0
	#      SVY2 = 0.0
	#      SVY3 = 0.0
	#      ERLMT = 0.
	#      DO 25 I = 1,3
	#      ISKP(I)=0
	#   25 CONTINUE

	NEAR = station_phases.get_index_of_nearest_station()
	PMIN = parse_datetime(station_phases.Ptimes[PIDXS][NEAR]).timestamp
	SVY1, SVY2, SVY3 = 0., 0., 0.
	ERLMT = 0.
	ISKP = np.zeros(4, 'int')

	#      IF (INST .NE. 9) GO TO 90
	#      READ(12,85) ORG1,ORG2,LAT1,LAT2,LON1,LON2,Z
	#   85 FORMAT(F5.0,F5.2,I5,F5.2,I5,2F5.2)
	#      ORG=60.*ORG1+ORG2
	#      LATEP=60.*LAT1+LAT2
	#      LONEP=60.*LON1+LON2
	#      GO TO 105

	#  100 Z=ZTR
	#	zsv=ztr
	#      IF (AZRES(NRP+1).NE. '    ') Z=ZRES
	#      ORG=PMIN-Z/5.-1.
	#      IF(LATRT.EQ.0.) GO TO 102
	#      LATEP=LATRT
	#      LONEP=LONRT
	#      GO TO 105
	#  102 IF (LATR .EQ. 0.) GO TO 104
	#      LATEP=LATR
	#      LONEP=LONR
	#      GO TO 105
	#  104 LATEP=LAT(K)+0.1
	#      LONEP=LON(K)+0.1

	## Trial focal depth
	Z = ZTR
	ZSV = ZTR

	## Trial origin (epicenter and origin time)
	ORG = PMIN -Z/5. - 1.

	if origin:
		## Trial epicenter or fixed origin
		LONEP, LATEP, _ORG = origin
		#if fix_origin:
		#	assert _ORG
		if _ORG:
			ORG = parse_datetime(_ORG).timestamp
	else:
		if LATRT != 0:
			## LATRT, LONRT are set in "CHECK FOR MULTIPLE SOLUTIONS OF THE
			## SAME EARTHQUAKE" section at the end, but this isn't implemented
			LATEP, LONEP = LATRT, LONRT
		else:
			## Use nearest station, adding 0.1 (minutes!) to lon/lat
			## to avoid division by zero
			if verbose:
				msg = 'Initializing location from nearest station %s'
				msg %= station_phases.stations[NEAR].code
				print(msg)
			LATEP = station_phases.stat_lats[NEAR] + 0.1/60
			LONEP = station_phases.stat_lons[NEAR] + 0.1/60

	#  105 ADJSQ=0.
	#      IPH=0
	#      NDEC=0
	#      PRMSSQ=100000.

	ADJSQ = 0.
	#IPH = 0
	NDEC = 0
	PRMSSQ = 1E+5

	## Note: ISW = selection card:
	## - blank: station delay model
	## - '1': variable first layer model
	## - ...
	#      IF (ISW .EQ. '1   ') KNO=MNO(K)
	#      IF(ISW .EQ. '1   ') FLTEP=FLT(KNO,K)

	## Variables related to variable first layer model (not implemented)
	KNO = 0
	if ISW.strip() == '1':
		KNO = MNO[NEAR]
		FLTEP = FLT[KNO, NEAR]


	## ------- GEIGER'S ITERATION TO FIND HYPOCENTRAL ADJUSTMENTS ------------

	#      NIMAX=TEST(11)+.0001

	#  777 NI = 1
	#      IF (INST .EQ. 9) NI=NIMAX
	#  111 IF(ERLMT .EQ. 0.) GO TO 110
	#      LATEP = LATSV + LA(NA)*DELAT
	#      LONEP = LONSV + LO(NA)*DELON
	#c	print *,'GI',lonep/60.,latep/60.
	#      Z = ZSV + ALZ(NA)*DEZ
	#      IF(Z .LT.test(15)) Z=test(15)
	#  110 FMO=0.
	#      FNO=0.
	#      DO 112 I=1,5
	#  112 SUM(I)=0.
	#c	print *,'GIi',lonep/60.,latep/60.

	## NOT_ENOUGH_DATA = GOTO96
	NOT_ENOUGH_DATA = False
	if NR < min_num_phases and not fix_origin:
		NOT_ENOUGH_DATA = True

	NIMAX = TEST[10] + .0001
	NI = NIMAX if fix_origin else 1
	#NIMAX = 2 if fix_origin else TEST[10]
	#NIMAX += .0001
	#NI = 1

	## Only used if compute_auxilliary_rms is True (not implemented)
	NA = 0
	DELON, DELAT, DEZ = 0., 0., 0.

	GOTO110 = False
	while True:
		## Main loop
		if NOT_ENOUGH_DATA:
			print('Insufficient data for locating this quake!')
			SE = np.zeros(4, dtype='f')
			#IEXIT = 1
			QSD = 'DD'
			return (0., 0., 0., UTCDateTime('0001-01-01 00:00:00'),
					SE, station_phases, QSD, NI, 0.)

		if verbose:
			print('NI=%d, NDEC=%d' % (NI, NDEC))

		if not GOTO110:
			# 111
			if ERLMT != 0:
				## Only used if compute_auxilliary_rms is True (not implemented)
				LATEP = LATSV + LA[NA] * DELAT
				LONEP = LONSV + LO[NA] * DELON
				Z = ZSV + ALZ[NA] * DEZ
				if Z < TEST[14]:
					Z = TEST[14]
		# 110
		#FMO, FNO = 0., 0.
		DELMIN = 99999.
		SUM = np.zeros(5, dtype='f')

		if verbose:
			print('Hypocenter: %.5f, %.5f, %.1f, %s'
				% (LONEP, LATEP, Z, UTCDateTime(ORG)))

		#C------- CALCULATE EPICENTRAL DISTANCE BY RICHTER'S METHOD -------------
		#      DO 120 I=1,NR
		#      JI=KDX(I)
		#      PHI = 0.0174532 * ((LAT(JI)+LATEP)/120.)
		#      SINPHI = SIN(PHI)
		#      SINP2  = SINPHI**2
		#      SINP4  = SINP2**2
		#      CA = 1.8553654 + 0.0062792*SINP2 + 0.0000319*SINP4
		#      CB = 1.8428071 + 0.0187098*SINP2 + 0.0001583*SINP4
		#      DX(I) = (LON(JI)-LONEP) * CA * COS(PHI)
		#      DY(I) = (LAT(JI)-LATEP) * CB
		#      DELTA(I)=SQRT(DX(I)**2+DY(I)**2)+0.000001
		#      WT(I)=W(I)
		#      IF (NI .LE. 1) GO TO 115
		#C------- DISTANCE WEIGHTING --------------------------------------------
		#      IF (DELTA(I) .LE. XNEAR) GO TO 115
		#      WT(I)=W(I)*(XFAR-DELTA(I))/XFN
		#      IF (WT(I) .LT. 0.005) WT(I)=0.
		#  115 IF (WT(I) .EQ. 0.) GO TO 120
		#      IF (KSMP(I) .EQ. 1) FMO=FMO+1.
		#      FNO=FNO+1.
		#      SUM(4)=SUM(4)+WT(I)
		#  120 CONTINUE

		## ------- CALCULATE EPICENTRAL DISTANCE BY RICHTER'S METHOD -------------
		DX, DY = CALC_EPICENTRAL_XY_OFFSET(LONEP, LATEP, LON, LAT)
		DELTA = np.sqrt(DX*DX + DY*DY) + 0.000001
		#print('DELTA', DELTA)

		## ------- DISTANCE WEIGHTING --------------------------------------------
		WT = W.copy()
		DWT = np.ones(NR, dtype='f')
		if apply_al_lindh_mod:
			## AL LINDH'S MODIFICATION (TO PREVENT STATIONS WEIGHTED OUT)
			DELMIN = min(DELTA.min(), DELMIN)
			for I in range(NR):
				## Note: this results in different YFAR for each phase...
				#DELMIN = min(DELTA[I], DELMIN)
				YFAR = max(XFAR, 3 * DELMIN)
				if NI > 3:
					if DELTA[I] > XNEAR:
						DWT[I] = ((YFAR - DELTA[I]) / float(YFAR - XNEAR))
						WT[I] *= DWT[I]
					if WT[I] < 0.005:
						WT[I] = 0
				#if WT[I] != 0:
				#	if KSMP[I] == 1:
				#		FMO += 1
		else:
			if NI > 1:
				idxs = DELTA > XNEAR
				DWT[idxs] = (XFAR - DELTA[idxs]) / XFN
				WT = W * DWT
				## Note: DWT may be negative, but will be set to 0 in next line
				WT[WT < 0.005] = 0

		zero_weight_idxs = np.isclose(WT, 0)
		FMO = float(np.sum((~zero_weight_idxs) & (KSMP == 1)))
		FNO = float(np.sum(~zero_weight_idxs))

		#      IF (FNO .LT. 3.) GO TO 96
		#      AVWT=SUM(4)/FNO
		#C------- NORMALIZE DISTANCE WEIGHTS ------------------------------------
		#      SUM(4)=0.0
		#      DO 122 I=1,NR
		#  122 WT(I)=WT(I)/AVWT
		#      IF ((NI.LE.2).OR.(KAZ.EQ.0)) GO TO 130
		#C------- AZIMUTHAL WEIGHTING -------------------------------------------
		#      CALL AZWTOS(DX,DY,NR,WT,KDX,AZ,TEMP,KEY,INS,IEW)
		#C------- COMPUTE TRAVEL TIMES & DERIVATIVES ----------------------------
		#  130 ZSQ=Z**2

		if FNO < min_num_phases and not fix_origin:
			NOT_ENOUGH_DATA = True
			continue

		## ------- NORMALIZE DISTANCE WEIGHTS -----------------------------------
		SUM[3] = np.sum(WT)
		AVWT = SUM[3] / FNO
		SUM[3] = 0
		WT /= AVWT

		## ------- AZIMUTHAL WEIGHTING ------------------------------------------
		AZ = np.degrees(np.pi/2. - np.arctan2(DY, DX))
		AZ = np.mod(AZ + 360, 360)
		AZ[(DX == 0) & (DY == 0)] = 999.
		AZWT = np.ones_like(WT)
		if azimuthal_weighting and NI > 2:
			#AZ[zero_weight_idxs] = 0.
			AZWT[~zero_weight_idxs], GAP = AZWTOS(AZ[~zero_weight_idxs],
												normalize=False)
			WT[~zero_weight_idxs] *= AZWT[~zero_weight_idxs]

		if verbose > 1:
			print('WT', WT)

		## Compute travel times T and partial derivatives X

		#ZSQ = Z * Z

		#      if(iflag.eq.0) CALL TRVDRV(ISW,V,D,DEPTH,VSQ,NL,THK,H,G,F,
		#     &  TID,DID,FLT,DELTA,DX,DY,NR,KDX,KNO,FLTEP,Z,ZSQ,X,T,ANIN)

		# TODO: check use of IFLAG in modified hypo71
		#IFLAG = 0
		#if IFLAG == 0:
		#ELEV = np.minimum(ELEV, 0)
		#ELEV *= 0
		#print('ELEV', ELEV)

		(T, X, AIN) = TRVDRV(velocity_model, Z, DELTA, ELEV, VIDXS, ISW=ISW)
		X[0] *= DX
		X[1] *= DY
		#print('DELTA: ', DELTA)
		#print('T', T)
		#print('X0', np.abs(X[0]).min(), np.abs(X[0]).max())
		#print('X1', np.abs(X[1]).min(), np.abs(X[1]).max())
		#print('X2', np.abs(X[2]).min(), np.abs(X[2]).max())

		#T2, tt_residuals = velocity_model.calc_tt_residuals(Z, ORG, DELTA, ELEV, TPS, VIDXS)
		#print(T2)
		#print(np.sqrt(np.mean(tt_residuals**2)))

		#import pylab
		#pylab.plot(DELTA, T, 'bo', label='FORTRAN')
		#pylab.plot(DELTA, T2, 'r.', label='python')
		#pylab.legend()
		#pylab.show()
		#exit()


		#		  131 FDLY=1.
		#      IF (ISW .EQ. '1   ') FDLY=0.
		#C------- CALCULATE TRAVEL TIME RESIDUALS X(4,I) & MODIFY THE DERIV'S ---
		#      DO 150 I=1,NR
		#      JI=KDX(I)
		#      IF (I .LE. NRP) GO TO 145
		#C------- S PHASE DATA --------------------------------------------------
		#      T(I)=POS*T(I)
		#      X(1,I)=POS*X(1,I)
		#      X(2,I)=POS*X(2,I)
		#      X(3,I)=POS*X(3,I)
		#      X(4,I)=TP(I)-T(I)-ORG-POS*DLY(KNO,JI)*FDLY
		#      GO TO 150
		#  145 IF (KSMP(I) .EQ. 0) GO TO 146
		#C------- S-P DATA ------------------------------------------------------
		#      X(1,I)=(POS-1.)*X(1,I)
		#      X(2,I)=(POS-1.)*X(2,I)
		#      X(3,I)=(POS-1.)*X(3,I)
		#      X(4,I)=TS(I)-TP(I)-(POS-1.)*(DLY(KNO,JI)*FDLY+T(I))
		#      GO TO 150
		#C------- P TRAVEL TIME RESIDUAL ----------------------------------------
		#  146 X(4,I)=TP(I)-T(I)-ORG-DLY(KNO,JI)*FDLY
		#  150 CONTINUE

		## ------- CALCULATE TRAVEL TIME RESIDUALS X(4,I)
		# Compute the time residual for each phase:
		#   X[3] = observed arrival (TPS)
		#         - predicted travel time from model (T)
		#         - origin time offset (ORG)
		#         - station delay correction (DLY * FDLY)
		# Positive values indicate the observed phase arrived later than predicted.
		FDLY = 1.
		if ISW.strip() == '1':
			FDLY = 0.
		X[3] = TPS - T - ORG - DLY[KNO] * FDLY

		## S-P data
		SMPIDXS = KSMP.astype('bool')
		if SMPIDXS.any():
			POS = 1.73
			X[:3,SMPIDXS] = (POS - 1.) * X[:3,SMPIDXS]
			X[3,SMPIDXS] = (TPS[SMPIDXS] - (POS - 1.) * T[SMPIDXS]
							- (POS - 1.) * DLY[KNO, SMPIDXS] * FDLY)

		#print('RMS', np.sqrt(np.mean(X[3]**2)))

		#C------- COMPUTE AVR, AAR, RMSSQ, & SDR --------------------------------
		#      ONF=0.0
		#      DO 152 I=1,NR
		#      ONF = ONF + WT(I)*(1-KSMP(I))
		#      XWT = X(4,I)*WT(I)
		#      SUM(1)=SUM(1)+XWT
		#      SUM(2)=SUM(2)+ABS(XWT)
		#      SUM(3)=SUM(3)+X(4,I)*XWT
		#      SUM(5)=SUM(5)+XWT*(1-KSMP(I))
		#  152 CONTINUE
		#      IF(FNO .GT. FMO) AVRPS=SUM(5)/(ONF)
		#      AVR=SUM(1)/FNO
		#      AAR=SUM(2)/FNO
		#      RMSSQ=SUM(3)/FNO
		#      SDR=SQRT(ABS(RMSSQ-AVR**2))
		#      DO 153 I=1,5
		#      SUM(I)= 0.0
		#  153 CONTINUE

		## ------- COMPUTE AVR, AAR, RMSSQ, & SDR --------------------------------
		ONF = 0.
		for I in range(NR):
			ONF += WT[I] * (1 - KSMP[I])
			XWT = X[3,I] * WT[I]
			SUM[0] += XWT
			SUM[1] += abs(XWT)
			SUM[2] += X[3,I] * XWT
			SUM[4] += XWT * (1 - KSMP[I])

		if FNO > FMO:
			AVRPS = SUM[4] / ONF
		AVR = SUM[0] / FNO
		AAR = SUM[1] / FNO
		if apply_al_lindh_mod:
			RMSSQ = SUM[2] / max(1., FNO - 4.)
		else:
			RMSSQ = SUM[2] / FNO
		SDR = np.sqrt(abs(RMSSQ - AVR * AVR))
		SUM[:] = 0.
		if verbose > 1:
			print('RMSSQ', RMSSQ)

		#      IF (RMSSQ .GE. TEST(1)) GO TO 154
		#      IF(ERLMT .EQ. 1.) GO TO 167
		#      IF(INST.EQ.9) GO TO 501
		#      IF(NI .GE. 2) GO TO 167
		#      GO TO 165
		#C------- JEFFREYS' WEIGHTING -------------------------------------------
		#  154 FMO=0.
		#      FNO=0.
		#      DO 160 I=1,NR
		#      WRK(I)='    '
		#      IF (WT(I) .EQ. 0.) GO TO 160
		#      K=10.*ABS(X(4,I)-AVR)/SDR+1.5
		#      IF (K .GT. 41) K=41
		#      WT(I)=WT(I)*WF(K)
		#      IF (K .GT. 30) WRK(I)='****'
		#      IF (WT(I) .LT. 0.005) WT(I)=0.
		#      IF (WT(I) .EQ. 0.) GO TO 160
		#      IF (KSMP(I) .EQ. 1) FMO=FMO+1.
		#      FNO=FNO+1.
		#      SUM(4)=SUM(4)+WT(I)
		#  160 CONTINUE
		#      IF (FNO .LT. 4.) GO TO 96
		#      AVWT=SUM(4)/FNO
		#      SUM(4)=0.0
		#      ONF=0.0
		#      DO 164 I=1,NR
		#      WT(I)=WT(I)/AVWT
		#      ONF = ONF + WT(I)*(1-KSMP(I))
		#      XWT=X(4,I)*WT(I)
		#      SUM(5)=SUM(5)+XWT*(1-KSMP(I))
		#  164 CONTINUE
		#C------- RECALCULATE AVRPS ---------------------------------------------
		#      IF(ERLMT .EQ. 1.) GO TO 163
		#      IF(INST .NE. 9) GO TO 163
		#      AVRPS = 0.0
		#      IF(FNO .NE. FMO) AVRPS = SUM(5)/ONF
		#      GO TO 501
		#  163 IF(FNO.EQ.FMO) AVRPS=0.0
		#      IF(FNO.EQ.FMO) GO TO 167
		#      AVRPS=SUM(5)/(ONF)
		#      SUM(5)=0.0
		#      IF(ERLMT .EQ. 1.) GO TO 167
		#C------- RESET FIRST ORIGIN TIME ---------------------------------------
		#      IF(NI.GE. 2) GO TO 167
		#  165 ORG=ORG+AVRPS
		#      DO 166 I=1,NR
		#      IF(KSMP(I) .EQ. 0) X(4,I)=X(4,I)-AVRPS
		#      XWT=WT(I)*X(4,I)
		#      SUM(5)=SUM(5)+XWT*(1 - KSMP(I))
		#      SUM(2)=SUM(2)+ABS(XWT)
		#      SUM(3)=SUM(3)+X(4,I)*XWT
		#  166 CONTINUE
		#      IF(FNO .GT. FMO) AVRPS=SUM(5)/(ONF)
		#      AAR=SUM(2)/FNO
		#      RMSSQ = SUM(3)/FNO
		#      GO TO 169

		#if RMSSQ < min_rms_jeffrey:
		if not (RMSSQ >= TEST[0] and (not apply_al_lindh_mod or NI > 3)):
			## RESET_ORG = GOTO165
			## Note: GOTO165 and GOTO167 are mutually exclusive,
			## as GOTO165 is followed by GOTO169
			if ERLMT == 1:
				#GOTO167 = True
				RESET_ORG = False
			elif fix_origin:
				break
			elif NI >= 2:
				#GOTO167 = True
				RESET_ORG = False
			else:
				#GOTO167 = False
				RESET_ORG = True
		else:
			## ------- JEFFREYS' WEIGHTING -------------------------------------------
			# 154
			if verbose:
				print("Jeffrey's weighting")
			FMO, FNO = 0., 0.
			WRK = np.zeros(NR, dtype='|S4')
			WRK[:] = '    '
			for I in range(NR):
				if WT[I] > 0:
					K = int(10. * abs(X[3,I] - AVR) / SDR + 1.5)
					K = min(41, K)
					WT[I] *= WF[K-1]
					if K > 30:
						WRK[I] = '****'
					if WT[I] < 0.005:
						WT[I] = 0
					if WT[I] > 0:
						if KSMP[I] == 1:
							FMO += 1
						FNO +=1
						SUM[3] += WT[I]
			if FNO < min_num_phases:
				NOT_ENOUGH_DATA = True
				continue

			AVWT = SUM[3] / FNO
			SUM[3] = 0.
			ONF = 0.
			for I in range(NR):
				WT[I] /= AVWT
				ONF += (WT[I] * (1-KSMP[I]))
				XWT = X[3,I] * WT[I]
				SUM[4] += (XWT * (1-KSMP[I]))
			if verbose > 1:
				print('WT', WT)

			## ------- RECALCULATE AVRPS ---------------------------------------------
			#if not (ERLMT == 1 or not fix_origin):
			if ERLMT != 1 and fix_origin:
				AVRPS = 0.
				if FNO != FMO:
					AVRPS = SUM[4] / ONF
				if not (fix_origin and not _ORG):
					# GOTO 501 (= break out of main loop !!)
					break

			## 163
			RESET_ORG = True
			if fix_origin and not _ORG:
				## Reset origin time if fix_origin is True, but _ORG is None
				RESET_ORG = True
			elif FNO == FMO:
				AVRPS = 0.
				#GOTO167 = True
				RESET_ORG = False
			else:
				AVRPS = SUM[4] / ONF
				SUM[4] = 0.
				if ERLMT == 1 or NI >= 2:
					#GOTO167 = True
					RESET_ORG = False

		## ------- RESET FIRST ORIGIN TIME ---------------------------------------
		# 165
		if RESET_ORG:
			ORG += AVRPS
			if verbose:
				print("Reset first origin time %+.3f -> %s"
						% (AVRPS, UTCDateTime(ORG)))
			for I in range(NR):
				if KSMP[I] == 0:
					X[3,I] -= AVRPS
				XWT = WT[I] * X[3,I]
				SUM[4] += (XWT * (1 - KSMP[I]))
				SUM[1] += abs(XWT)
				SUM[2] += (X[3,I] * XWT)
			if FNO > FMO:
				AVRPS = SUM[4] / ONF
			AAR = SUM[1] / FNO
			if apply_al_lindh_mod:
				RMSSQ = SUM[2] / max(1., FNO - 4.)
			else:
				RMSSQ = SUM[2] / FNO
			# GOTO 169


		#C------- FOR NI>1, COMPUTE AAR, & RMSSQ AS IF AVRPS=0. -----------------
		#  167 DO 168 I=1,NR
		#      XWT=WT(I)*(X(4,I)-AVRPS*(1-KSMP(I)))
		#      SUM(2)=SUM(2)+ABS(XWT)
		#      SUM(3)=SUM(3)+(X(4,I)-AVRPS*(1-KSMP(I)))*XWT
		#  168 CONTINUE
		#      AAR=SUM(2)/FNO
		#      RMSSQ=SUM(3)/FNO
		#      IF(ERLMT .EQ. 0.) GO TO 169

		## ------- FOR NI>1, COMPUTE AAR, & RMSSQ AS IF AVRPS=0. -----------------
		# 167
		else:
			for I in range(NR):
				XWT = WT[I] * (X[3,I] - AVRPS * (1 - KSMP[I]))
				SUM[1] += abs(XWT)
				SUM[2] += (X[3,I] - AVRPS * (1-KSMP[I])) * XWT
			AAR = SUM[1] / FNO
			if apply_al_lindh_mod:
				RMSSQ = SUM[2] / max(1., FNO - 4.)
			else:
				RMSSQ = SUM[2] / FNO

			if ERLMT != 0:
				if NA == 10:
					# GOTO 550
					break
				NA += 1
				# GOTO 111
				GOTO110 = False
				continue
			# GOTO 169

		#C------- CHECK IF SOLUTION IS BETTER THAN PREVIOUS ONE -----------------
		#  169 IF((NI .EQ. 1) .AND. (NDEC .EQ. 0)) GO TO 170
		#      IF(PRMSSQ.GE.RMSSQ) GO TO 170
		#      NDEC = NDEC +1
		#      IF(NDEC .GT. 1) GO TO 175
		#      DO 177 I= 1,3
		#      B(I) = 0.0
		#      AF(I)=-1.0
		#      SE(I) = 0.0
		#  177 CONTINUE
		#      NI = NI -1
		#      BM1=Y(1)
		#      BM2=Y(2)
		#      BM3=Y(3)
		#      BMAX = ABS(Y(1))
		#      IIMAX = 1
		#      DO 176 I = 2,3
		#      IF(ABS(Y(I)).LE.BMAX) GO TO 176
		#      BMAX = ABS(Y(I))
		#      IIMAX = I
		#  176 CONTINUE
		#      ISKP(IIMAX)=1
		#      Y(1)=-BM1/5.
		#      Y(2)=-BM2/5.
		#      Y(3)=-BM3/5.
		#      Y(4)=-Y(1)*XMEAN(1)-Y(2)*XMEAN(2)-Y(3)*XMEAN(3)
		#      XADJSQ=Y(1)**2+Y(2)**2+Y(3)**2
		#      KP=0
		#      IF(XADJSQ .LT. 4.*TEST(4)/25.) GO TO 170
		#  175 IF(NDEC .EQ. 5) GO TO 170
		#      GO TO 325

		## ------- CHECK IF SOLUTION IS BETTER THAN PREVIOUS ONE -----------------
		if fix_origin:
			break
		if verbose > 1:
			print('AVRPS=%.3f, PRMSSQ=%.3f, RMSSQ=%.3f' % (AVRPS, PRMSSQ, RMSSQ))
		if (NI == 1 and NDEC == 0) or PRMSSQ >= RMSSQ:
			## DO_SWMREG = GOTO170
			DO_SWMREG = True
		else:
			if verbose:
				print("Solution worse than previous one!")
			NDEC += 1
			if NDEC <= 1:
				## Note: B, SE and AF are returned by SWMREG,
				## not sure if they have to be initialized here too
				#B = np.zeros(4)
				#AF = -np.ones(3)
				SE = np.zeros(4, dtype='f')
				## Note: don't understand why NI should be decremented
				#NI -= 1
				BM1, BM2, BM3 = Y[:3]
				## BMAX is not used anywhere...
				IIMAX = np.argmax(np.abs(Y[:3]))
				BMAX = abs(Y[IIMAX])
				ISKP[IIMAX] = 1
				Y[:3] = (-BM1/5., -BM2/5., -BM3/5.)
				Y[3] = -Y[0] * XMEAN[0] - Y[1] * XMEAN[1] - Y[2] * XMEAN[2]
				XADJSQ = Y[0]**2 + Y[1]**2 + Y[2]**2
				KP = 0
				if (XADJSQ < 4 * TEST[3] / 25.):
					DO_SWMREG = True
				else:
					#if NDEC == 5:
					#	DO_SWMREG = True
					#else:
						# GOTO 325
						DO_SWMREG = False
						if verbose:
							print('XADJSQ', XADJSQ)
			else:
				# 175
				if NDEC == 5:
					DO_SWMREG = True
				else:
					# GOTO 325
					DO_SWMREG = False

		if DO_SWMREG:
			# 170
			#C------- STEPWISE MULTIPLE REGRESSION ANALYSIS OF TRAVEL TIME RESIDUALS-
			#  170 IF(NDEC .GE. 1) NI = NI + 1
			#      IF (INST.EQ.1) GO TO 250
			#      IF(ISKP(3) .EQ. 1) GO TO 250
			#      IF (INST .EQ. 9) GO TO 501
			#      IF ((FNO.EQ.4) .AND. (FMO.LT.4)) GO TO 250
			#C---- FREE SOLUTION
			#      KZ=0
			#      KF=0
			#      CALL SWMREG(TEST,IPRN,NR,KSMP,FNO,X,WT,ISKP,KF,KZ,XMEAN,
			#     &  B,Y,SE,AF,ONF,FLIM)
			#C------- AVOID CORRECTING DEPTH IF HORIZONTAL CHANGE IS LARGE ----------
			#      IF (Y(1)**2+Y(2)**2 .LT. TEST(2)) GO TO 300
			#C---- FIXED DEPTH SOLUTION
			#  250 KZ=1
			#      KF=0
			#      CALL SWMREG(TEST,IPRN,NR,KSMP,FNO,X,WT,ISKP,KF,KZ,XMEAN,
			#     &  B,Y,SE,AF,ONF,FLIM)

			## ------- STEPWISE MULTIPLE REGRESSION ANALYSIS OF TRAVEL TIME RESIDUALS-
			#if NDEC >= 1:
			#	NI += 1
			if fix_origin:
				# GOTO 501
				break
			if fix_depth or ISKP[2] == 1 or (FNO == 4 and FMO < 4):
				## FIXED_DEPTH_SOLUTION = GOTO250
				FIXED_DEPTH_SOLUTION = True
			else:
				## ---- FREE SOLUTION
				KZ, KF = 0, 0
				if verbose > 1:
					print('SWMREG (KZ=%d, FNO=%s, ISKP=%s)' % (KZ, FNO, ISKP))
				(XMEAN, Y, SE, FLIM) = SWMREG(X, WT, KSMP, FNO, ISKP, KF, KZ,
										TEST[2], TEST[5], verbose=(IPRN==3))
				## ------- AVOID CORRECTING DEPTH IF HORIZONTAL CHANGE IS LARGE ----------
				# NOTE: Fortran compares Y(1)**2+Y(2)**2 against TEST(2), where TEST(2)
				# was already squared in input1 (threshold = max_horizontal_adjustment_
				# for_depth_adjustment**2 = TEST[1] = 100 for default of 10 km).
				# Using the unsquared parameter here makes the threshold ~3x tighter.
				# Flagged for verification against Fortran reference.
				FIXED_DEPTH_SOLUTION = False
				if Y[0]**2 + Y[1]**2 >= max_horizontal_adjustment_for_depth_adjustment:
					FIXED_DEPTH_SOLUTION = True

			if FIXED_DEPTH_SOLUTION:
				## ---- FIXED DEPTH SOLUTION
				KZ, KF = 1, 0
				if verbose > 1:
					print('SWMREG (KZ=%d, FNO=%s, ISKP=%s)' % (KZ, FNO, ISKP))
				(XMEAN, Y, SE, FLIM) = SWMREG(X, WT, KSMP, FNO, ISKP, KF, KZ,
										TEST[2], TEST[5], verbose=(IPRN==3))

			if verbose > 1:
				print('Y', Y)
				print('FLIM', FLIM)
				print('XMEAN', XMEAN)
				#print('SE', SE)
				if np.allclose(Y, 0):
					print('X', X)

			#C------- LIMIT FOCAL DEPTH CHANGE & AVOID HYPOCENTER IN THE AIR --------
			#  300 DO 275 I= 1,3
			#      ISKP(I)=0
			#  275 CONTINUE
			#      OLDY1=Y(1)
			#      OLDY2=Y(2)
			#      OLDY3=Y(3)
			#      ABSY1=ABS(Y(1))
			#      ABSY2=ABS(Y(2))
			#      ABSY3=ABS(Y(3))
			#      IF(ABSY1.GT.ABSY2) GO TO 305
			#      ABSGR=ABSY2
			#      GO TO 308
			#  305 ABSGR=ABSY1
			#  308 IF(ABSY3.LE.TEST(5)) GO TO 310
			#      I=ABSY3/TEST(5)
			#      Y(3)=Y(3)/(I+1)
			#  310   continue
			#	 IF((Z+Y(3)).GT.test(15)) GO TO 315
			#      Y(3)=-Z*TEST(12)+.000001
			#      ISKP(3) = 1

			## ------- LIMIT FOCAL DEPTH CHANGE & AVOID HYPOCENTER IN THE AIR --------
			## 300
			ISKP[:3] = 0
			OLDY1, OLDY2, OLDY3 = Y[:3]
			ABSY1, ABSY2, ABSY3 = np.abs(Y[:3])
			ABSGR = max(ABSY1, ABSY2)

			if ABSY3 > TEST[4]:
				I = ABSY3 // TEST[4]
				Y[2] /= (I + 1.)
			if Z + Y[2] <= TEST[14]:
				Y[2] = -Z * TEST[11] + 0.000001
				ISKP[2] = 1

			#C------- LIMIT HORIZONTAL ADJUSTMENT OF EPICENTER ----------------------
			#  315 	continue
			#c	print *,z,y(3)
			#	IF(ABSGR.LE.TEST(10)) GO TO 320
			#      I=ABSGR/TEST(10)
			#      Y(1)=Y(1)/(I+1)
			#      Y(2)=Y(2)/(I+1)
			#  320 Y(4)=Y(4)-(Y(3)-OLDY3)*XMEAN(3)-(Y(1)-OLDY1)*XMEAN(1)
			#     1 -(Y(2)-OLDY2)*XMEAN(2)
			#      XADJSQ=Y(1)**2+Y(2)**2+Y(3)**2
			#      KP=0
			#      NDEC=0
			#      JPH=0

			## ------- LIMIT HORIZONTAL ADJUSTMENT OF EPICENTER ----------------------
			if verbose > 1:
				print('ABSGR', ABSGR)
			if ABSGR > TEST[9]:
				I = ABSGR // TEST[9]
				Y[0] /= (I + 1.)
				Y[1] /= (I + 1.)
			Y[3] = (Y[3] - (Y[2] - OLDY3) * XMEAN[2] - (Y[0] - OLDY1) * XMEAN[0]
							- (Y[1] - OLDY2) * XMEAN[1])
			XADJSQ = Y[0]**2 + Y[1]**2 + Y[2]**2
			if verbose > 1:
				print('XADJSQ', XADJSQ)
			KP = 0
			NDEC = 0
			#JPH = 0

		## output

		#      IF(NDEC .GE. 1) GO TO 330
		#C------- TERMINATE ITERATION IF HYPOCENTER ADJUSTMENT < TEST(4) --------
		#      IF (XADJSQ .LT. TEST(4)) GO TO 500
		#  330 IF(NI .EQ. NIMAX) GO TO 500

		## ------- TERMINATE ITERATION IF HYPOCENTER ADJUSTMENT < TEST(4) --------
		## 325
		if NDEC < 1:
			#if XADJSQ < TEST[3]:
			if (XADJSQ < TEST[3]
				and (not apply_al_lindh_mod or NI > 4)):
				if verbose:
					print('Terminating iteration because XADJSQ = %f' % XADJSQ)
				#print('X', X)
				# GOTO 500
				break

		## 330
		if NI == NIMAX:
			# GOTO 500
			break

		#C------- ADJUST HYPOCENTER ---------------------------------------------
		#      PHI = 0.0174532 * (LATEP/60.)
		#      SINPHI = SIN(PHI)
		#      SINP2  = SINPHI**2
		#      SINP4  = SINP2**2
		#      CA = 1.8553654 + 0.0062792*SINP2 + 0.0000319*SINP4
		#      CB = 1.8428071 + 0.0187098*SINP2 + 0.0001583*SINP4
		#      LATEP = LATEP + (Y(2)/CB)
		#      LONEP = LONEP + (Y(1)/(CA*COS(PHI)))
		#      Z=Z+Y(3)
		#      ORG=ORG+Y(4)
		#      SVY1 = Y(1)
		#      SVY2 = Y(2)
		#      SVY3 = Y(3)
		#      ADJSQ=XADJSQ
		#      IF(NDEC .EQ. 0) PRMSSQ=RMSSQ
		#      IF(NDEC.GE.1) GO TO 110
		#      NI = NI + 1
		#      IF(NI .LE. NIMAX) GO TO 111

		## ------- ADJUST HYPOCENTER ---------------------------------------------
		if verbose:
			print('Hypocentral adjustment: X=%.3f, Y=%.3f, Z=%.3f km, T=%.3f s'
				% (Y[0], Y[1], Y[2], Y[3]))
		LONEP, LATEP = SHIFT_LONLAT_BY_XY(LONEP, LATEP, Y[0], Y[1])
		Z += Y[2]
		ORG += Y[3]
		SVY1, SVY2, SVY3 = Y[:3]
		ADJSQ = XADJSQ
		if NDEC == 0:
			PRMSSQ = RMSSQ

		if NDEC >= 1:
			# GOTO 110
			GOTO110 = True
		else:
			NI += 1
			if NI <= NIMAX:
				GOTO110 = False
			else:
				break

	## end of Geiger's iteration


	#C------- RESET ORIGIN TIME ---------------------------------------------
	#  500 ORG=ORG+XMEAN(4)
	#      GO TO 502
	#  501 XMEAN(4)=0.0
	#  502 DO 505 I=1,5
	#  505 SUM(I)=0.0
	#      SUMM = 0.0
	#      DO 510 I=1,NR
	#      IF (KSMP(I) .EQ. 0) X(4,I)=X(4,I)-XMEAN(4)
	#      IF (WT(I) .EQ. 0.) GO TO 510
	#      IF(INST .NE. 9) GO TO 509
	#      XWTS=WT(I)*(X(4,I)**2)
	#      IF(KSMP(I) .EQ. 0) XWTS=WT(I)*((X(4,I)-AVRPS)**2)
	#      SUMM = SUMM + XWTS
	#  509 XWT=X(4,I)*WT(I)
	#      SUM(1)=SUM(1)+XWT
	#      SUM(2)=SUM(2)+ABS(XWT)
	#      SUM(3)=SUM(3)+X(4,I)*XWT
	#      SUM(5)=SUM(5)+XWT*(1-KSMP(I))
	#  510 CONTINUE
	#      RM9SV = SUMM/FNO
	#      AVR=SUM(1)/FNO
	#      AVRPS = 0.0
	#      IF(FNO .GT. FMO) AVRPS=SUM(5)/ONF
	#      AAR=SUM(2)/FNO
	#      RMSSQ=SUM(3)/FNO

	## ------- RESET ORIGIN TIME ---------------------------------------------
	## 500
	if not fix_origin:
		ORG += XMEAN[3]
	else:
		XMEAN[3] = 0

	SUM[:] = 0
	SUMM = 0.
	for I in range(NR):
		if KSMP[I] == 0:
			X[3,I] -= XMEAN[3]
		if WT[I] != 0:
			if fix_origin:
				if KSMP[I] == 0:
					XWTS = WT[I] * ((X[3,I] - AVRPS)**2)
				else:
					XWTS = WT[I] * X[3,I]**2
				SUMM += XWTS
			XWT = X[3,I] * WT[I]
			SUM[0] += XWT
			SUM[1] += abs(XWT)
			SUM[2] += (X[3,I] * XWT)
			SUM[4] += (XWT * (1 - KSMP[I]))
	RM9SV = SUMM / FNO
	AVR = SUM[0] / FNO
	AVRPS = 0.
	if FNO > FMO:
		AVRPS = SUM[4] / ONF
	AAR = SUM[1] / FNO
	if apply_al_lindh_mod:
		RMSSQ = SUM[2] / max(1., FNO - 4.)
	else:
		RMSSQ = SUM[2] / FNO

	#C------- COMPUTE ERROR ESTIMATES BY SOLVING FULL NORMAL EQUATION -------
	#      KF=2
	#      KP=1
	#      KZ=0
	#      CALL SWMREG(TEST,IPRN,NR,KSMP,FNO,X,WT,ISKP,KF,KZ,XMEAN,
	#     &  B,Y,SE,AF,ONF,FLIM)
	#      DO 521 I =1,3
	#  521 Y(I)=0.0
	#      IF(INST.EQ.1) KZ = 1

	## ------- COMPUTE ERROR ESTIMATES BY SOLVING FULL NORMAL EQUATION -------
	KF, KP, KZ = 2, 1, 0
	(XMEAN, Y, SE, FLIM) = SWMREG(X, WT, KSMP, FNO, ISKP, KF, KZ,
								TEST[2], TEST[5], verbose=(IPRN==3))
	if verbose > 1:
		print('XMEAN', XMEAN)
		#print('Y', Y)
		print('SE', SE)
	Y[:3] = 0
	if fix_depth:
		KZ = 1

	#      ERH=SQRT(SE(1)**2+SE(2)**2)

	#C------- COMPUTE SUMMARY OF TRAVEL TIME RESIDUALS ----------------------
	#      DO 522 I=1,NRP
	#      IF ((WT(I).EQ.0.) .OR. (KSMP(I).EQ.1))  GO TO 522
	#      JI=KDX(I)
	#      NRES(KNO,JI)=NRES(KNO,JI)+1
	#      SR(KNO,JI)=SR(KNO,JI)+X(4,I)*WT(I)
	#      SRSQ(KNO,JI)=SRSQ(KNO,JI)+X(4,I)**2*WT(I)
	#      SRWT(KNO,JI)=SRWT(KNO,JI)+WT(I)
	#  522 CONTINUE

	## ------- COMPUTE SUMMARY OF TRAVEL TIME RESIDUALS ----------------------
	for I in range(NRP):
		if not (WT[I] == 0 or KSMP[I] == 1):
			#JI = KDX[I]
			NRES[KNO,I] += 1
			SR[KNO,I] += (X[3,I] * WT[I])
			SRSQ[KNO,I] += (X[3,I]**2 * WT[I])
			SRWT[KNO,I] += WT[I]

	## ------- COMPUTE RMS AT AUXILIARY POINTS -------------------------------
	if compute_auxilliary_rms:
		# TODO, seems to require entering the main loop again...
		pass

	#C------- CHECK FOR MULTIPLE SOLUTIONS OF THE SAME EARTHQUAKE -----------
	#      IF(IPRO.NE.' ** ') RETURN
	#      NR=NRP
	#      NRP1=NR +1
	#      READ(4,600)  CHECK,IPRO,KNST,INST,ZRES,LAT1,LAT2,LON1,LON2,
	#     1 AZRES(NRP1)
	#      WRITE(8,601) CHECK,IPRO,KNST,INST,ZRES,LAT1,LAT2,LON1,LON2
	#  601 FORMAT(//2A4,9X,2I1,F5.2,1X,2(I4,F6.2),'--- RUN AGAIN ---')
	#  600 FORMAT(2A4,9X,2I1,F5.2,1X,2(I4,F6.2),T21,A4)
	#      LATRT=60.*LAT1+LAT2
	#      LONRT=60.*LON1+LON2
	#      IF(CHECK.EQ.'    ') GO TO 30
	#      WRITE(8,610) CHECK
	#  610 FORMAT(/' ERROR ',A4,' SKIPPED.   INST. CARD DID NOT FOLLOW ***')
	#      RETURN
	#      END

	## ------- CHECK FOR MULTIPLE SOLUTIONS OF THE SAME EARTHQUAKE -----------
	## Normally IPRO is blank
	## If it is equal to ' ** ', addtional instruction cards follow
	## with different KNST, INST, ZRES and trial hypocenter
	IPRO = ''
	if IPRO == ' ** ':
		pass

	#station_phases.data.loc[KIDXS, 'takeoff_angle'] = AIN
	#station_phases.data.loc[KIDXS, 'azimuth'] = np.round(AZ)
	#station_phases.data.loc[KIDXS, 'Repi'] = DELTA
	#station_phases.data.loc[KIDXS, 'Rhypo'] = np.hypot(DELTA, Z)
	#station_phases.data.loc[KIDXS, 'az_weight'] = AZWT
	#station_phases.data.loc[KIDXS, 'dist_weight'] = DWT
	#station_phases.data.loc[PIDXS, 'Pres'] = X[3][:NRP]
	#station_phases.data.loc[SIDXS, 'Sres'] = X[3][NRP:]
	#station_phases.data.loc[PIDXS, 'Ptot_weight'] = WT[:NRP]
	#station_phases.data.loc[SIDXS, 'Stot_weight'] = WT[NRP:]

	#Note: NRP = number of P phases, so X[3][:NRP] gives all P residuals.

	station_phases.data.loc[KIDXS, "takeoff_angle"] = np.asarray(AIN, dtype=np.float32)
	station_phases.data.loc[KIDXS, "azimuth"]       = np.round(AZ).astype(np.float32)
	station_phases.data.loc[KIDXS, "Repi"]          = np.asarray(DELTA, dtype=np.float32)
	station_phases.data.loc[KIDXS, "Rhypo"]         = np.hypot(DELTA, Z).astype(np.float32)
	station_phases.data.loc[KIDXS, "az_weight"]     = np.asarray(AZWT, dtype=np.float32)
	station_phases.data.loc[KIDXS, "dist_weight"]   = np.asarray(DWT, dtype=np.float32)
	station_phases.data.loc[PIDXS, "Pres"]          = np.asarray(X[3][:NRP], dtype=np.float32)
	station_phases.data.loc[SIDXS, "Sres"]          = np.asarray(X[3][NRP:], dtype=np.float32)
	station_phases.data.loc[PIDXS, "Ptot_weight"]   = np.asarray(WT[:NRP], dtype=np.float32)
	station_phases.data.loc[SIDXS, "Stot_weight"]   = np.asarray(WT[NRP:], dtype=np.float32)



	## Compute solution quality
	#      DATA CLASS/'A','B','C','D'/
	#      OFD=Z
	#      TFD=2.*Z
	#      IF (OFD .LT. 5.) OFD=5.
	#      IF (TFD .LT. 10.) TFD=10.
	#      JS=4
	#      IF ((RMS.LT.0.50).AND.(ERH.LE.5.0)) JS=3
	#      IF ((RMS.LT.0.30).AND.(ERH.LE.2.5).AND.(SE(3).LE.5.0)) JS=2
	#      IF ((RMS.LT.0.15).AND.(ERH.LE.1.0).AND.(SE(3).LE.2.0)) JS=1
	#      JD=4
	#      IF (NO .LT. 6) GO TO 30
	#      IF ((GAP.LE.180.).AND.(DMIN.LE.50.)) JD=3
	#      IF ((GAP.LE.135.).AND.(DMIN.LE.TFD)) JD=2
	#      IF ((GAP.LE. 90.).AND.(DMIN.LE.OFD)) JD=1
	#   30 JAV=(JS+JD+1)/2
	#      Q=CLASS(JAV)
	#      QS=CLASS(JS)
	#      QD=CLASS(JD)

	NO = FNO
	RMS = np.sqrt(RMSSQ)
	ERH = np.sqrt(SE[0]**2 + SE[1]**2)
	DMIN = DELTA.min()
	CLASS = ['A', 'B', 'C', 'D']

	OFD = max(5., Z)
	TFD = max(10., 2 * Z)
	JS = 4
	# Note: these are independent thresholds (sequential IFs in Fortran),
	# not elif — each can tighten the class further.
	if RMS < 0.5 and ERH <= 5.0:
		JS = 3
	if RMS < 0.3 and ERH <= 2.5 and SE[2] <= 5.0:
		JS = 2
	if RMS < 0.15 and ERH <= 1.0 and SE[2] <= 2.0:
		JS = 1

	# TODO: we should recompute AZ and GAP with new location
	JD = 4
	if NO >= 6:
		# Note: independent thresholds (sequential IFs in Fortran)
		if GAP <= 180 and DMIN <= 50:
			JD = 3
		if GAP <= 135 and DMIN <= TFD:
			JD = 2
		if GAP <= 90 and DMIN <= OFD:
			JD = 1

	JAV = (JS + JD + 1) // 2
	Q = CLASS[JAV-1]
	QS = CLASS[JS-1]
	QD = CLASS[JD-1]
	QSD = QS + QD

	if verbose:
		#      RMK2=' '
		#      RMKO=' '
		#C---- KZ=1 FOR FIXED DEPTH; ONF=0 FOR ORIGIN TIME BASED ON SMP'S
		#      IF (ONF .EQ. 0.) RMKO='*'
		#      IF (KZ .EQ. 1) RMK2='*'

		#      LAT1=LATEP/60.
		#      LAT2=LATEP-60.*LAT1
		#      LON1=LONEP/60.
		#      LON2=LONEP-60.*LON1
		#      ADJ=SQRT(ADJSQ)

		RMKO, RMK2 = ' ', ' '
		if ONF == 0:
			RMKO = '*'
		if KZ == 1:
			RMK2 = '*'

		t = UTCDateTime(ORG).datetime
		year, month, day = t.year, t.month, t.day
		year -= ((year // 100) * 100)
		hour, minute, second, ms = t.hour, t.minute, t.second, t.microsecond
		second += (ms * 1E-6)

		LAT1 = np.floor(LATEP)
		LAT2 = (LATEP - LAT1) * 60
		LON1 = np.floor(LONEP)
		LON2 = (LONEP - LON1) * 60

		IDMIN = np.round(DMIN)
		IGAP = np.round(GAP)
		ERZ = SE[2]
		ADJ = np.sqrt(ADJSQ)
		JNST = KNST * 10 + INST

		print('  DATE    ORIGIN    LAT N    LONG E    DEPTH    MAG NO DM '
				'GAP M  RMS  ERH  ERZ Q SQD  ADJ IN NR  AVR  AAR NM AVXM '
				'SDXM NF AVFM SDFM  I')
		msg = (' %02d%02d%02d%c%02d%02d %5.2f %2d-%5.2f %3d-%5.2f %6.2f%c'
				'      %3d%3d %3d%2d%5.2f%5.1f%5.1f %c %c|%c'
				'%5.2f %2d%3d%5.2f%5.2f  0  0.0  0.0  0  0.0  0.0 %2d')
		msg %= (year, month, day, RMKO, hour, minute, second, LAT1, LAT2, LON1,
				LON2, Z, RMK2, NO, IDMIN, GAP, KNO+1, RMS, ERH, ERZ, Q, QS, QD,
				ADJ, JNST, NR, AVR, AAR, NI-1)

		print(msg)
	
	#print("DEBUG ORG:", ORG, type(ORG))
	#ORG_abs = UTCDateTime(float(ORG))
	if not np.isfinite(ORG):
		if verbose:
			print(f"Invalid origin time (ORG={ORG}) at depth {ZTR:.2f} km — likely non-convergent.")
		ORG_abs = None
	else:
		ORG_abs = UTCDateTime(float(ORG))
	return (LONEP, LATEP, Z, ORG_abs, SE, station_phases, QSD, NI, RMS)



if __name__ == "__main__":
	## Test distance
	LONEP, LATEP = 4.04, 50.580
	DX, DY = 1.75, 4.25
	LON, LAT = SHIFT_LONLAT_BY_XY(LONEP, LATEP, DX, DY)
	print(LON, LAT)
	DX, DY = CALC_EPICENTRAL_XY_OFFSET(LONEP, LATEP, np.array(LON), np.array(LAT))
	print(DX, DY)
	exit()


	import eqcatalog
	from robspy.hypo.velocity_model import CrustalVelocityModel
	from robspy.rob.velocity_model import CAM93 as vmodel
	from robspy import parse_datetime

	id_earth = 9965
	[eq] = eqcatalog.rob.query_local_eq_catalog_by_id(id_earth)
	phase_pick_dict = eq.get_phase_picks()
	station_codes = phase_pick_dict.keys()
	print(station_codes)
	lons, lats, altitudes = eqcatalog.rob.get_station_coordinates(station_codes,
																include_z=True)
	stations = [Station(station_codes[i], lons[i], lats[i], altitudes[i]/1000.)
				for i in range(len(station_codes))]

	ZTR = eq.depth
	#ZTR = 0
	#origin = (eq.lon, eq.lat, parse_datetime(eq.datetime))
	origin = ()
	(LONEP, LATEP, Z, ORG, SE, station_phases, gap) = SINGLE(stations, phase_pick_dict, vmodel, ZTR,
									origin=origin, fix_depth=False,
									max_altitude=0)
	print(eq.lon, eq.lat, eq.depth, str(eq.datetime))
	print(LONEP, LATEP, Z, str(ORG))