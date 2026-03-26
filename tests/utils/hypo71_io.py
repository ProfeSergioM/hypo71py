"""
Write and parse HYPO71PC-format input/output files.

The Fortran input file has four sections, in order:
  1. TEST-variable reset cards  (terminated by a blank selection card)
  2. Station list               (terminated by a blank station-name)
  3. Velocity model             (terminated by a card with V < 0.01)
  4. Control card
  5. Phase cards                (terminated by ' ***'; data set end: ' $$$')

All column positions below are 1-indexed (Fortran convention).
"""

from pathlib import Path
from obspy import UTCDateTime


def _as_utc(pick):
    """Accept a PhasePick, UTCDateTime, or None; always return UTCDateTime or None."""
    if pick is None:
        return None
    if hasattr(pick, 'datetime'):   # PhasePick
        return pick.datetime
    return pick                     # already UTCDateTime


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _deg_to_degmin(decimal_deg):
    """
    Convert decimal degrees to (integer_degrees, decimal_minutes).
    Returns the sign separately as a 1/-1 multiplier.
    """
    sign = 1 if decimal_deg >= 0 else -1
    d = abs(decimal_deg)
    deg = int(d)
    minutes = (d - deg) * 60.0
    return sign, deg, minutes


def write_hypo71_input(path, stations, picks, velocity_model,
                       ztr=5.0, pos=1.73, xnear=50., xfar=200., use_s=False,
                       use_s_minus_p=False):
    """
    Write a HYPO71PC input file.

    Parameters
    ----------
    path : str or Path
        Output file path (will be named HYPO71.INP in its parent directory).
    stations : list of Station
        Station objects with .code, .lat, .lon, .elev (km, positive up).
    picks : dict
        {station_code: {'P': UTCDateTime, 'S': UTCDateTime or None}}
        All P picks must share the same UTC date and hour.
    velocity_model : CrustalVelocityModel
    ztr : float
        Trial focal depth (km).
    pos : float
        Vp/Vs ratio.
    xnear, xfar : float
        Distance-weighting limits (km).
    use_s_minus_p : bool
        If True, write phase cards with P weight digit = 5 for stations that
        have both P and S picks.  Weight digit > 4 → W = (4-w)/4 < 0, which
        triggers KSMP(L)=1 in HYPO71's INPUT2.  Both absolute P and S times
        are still written; the Fortran computes TS-TP internally.
    """
    lines = []

    # ------------------------------------------------------------------
    # 1. Selection card — a blank line stops the TEST-variable read loop
    #    (INPUT1 checks: IF ((ISW.EQ.'    ').OR.(ISW.EQ.'1   ')) GO TO 6)
    # ------------------------------------------------------------------
    lines.append(' ' * 80)

    # ------------------------------------------------------------------
    # 2. Station list
    #    FORMAT(1X,A1,A4,I2,F5.2,A1,I3,F5.2,A1,I4,F6.2,4X,F5.2,2X,F5.2,
    #           1X,I1,F5.2,F7.2,1X,I1,5X,I6,I4,f6.2)
    #
    #    Col  1     : blank (1X)
    #    Col  2     : IW weight flag (A1; blank = normal)
    #    Cols 3-6   : station name (A4)
    #    Cols 7-8   : latitude degrees (I2)
    #    Cols 9-13  : latitude minutes (F5.2)
    #    Col 14     : N/S (A1)
    #    Cols 15-17 : longitude degrees (I3)
    #    Cols 18-22 : longitude minutes (F5.2)
    #    Col 23     : E/W (A1)
    #    Cols 24-27 : elevation in metres (I4)
    #    Cols 28-33 : station delay (F6.2; 0 for synthetic tests)
    #    Cols 34-37 : blank (4X)
    #    Cols 38-42 : FMGC (F5.2; 0)
    #    Cols 43-44 : blank (2X)
    #    Cols 45-49 : XMGC (F5.2; 0)
    #    Col 50     : blank (1X)
    #    Col 51     : KLAS (I1; 0)
    #    Cols 52-56 : PRR (F5.2; 0)
    #    Cols 57-63 : CALR (F7.2; 0)
    #    Col 64     : blank (1X)
    #    Col 65     : ICAL (I1; 0)
    # ------------------------------------------------------------------
    for sta in stations:
        sign_lat, lat_deg, lat_min = _deg_to_degmin(sta.lat)
        sign_lon, lon_deg, lon_min = _deg_to_degmin(sta.lon)
        ns = 'N' if sign_lat >= 0 else 'S'
        ew = 'E' if sign_lon >= 0 else 'W'
        elev_m = int(round(sta.elevation * 1000))   # km → m

        line  = ' '                        # col  1: 1X
        line += ' '                        # col  2: IW (blank = normal weight)
        line += f'{sta.sta[:4]:<4s}'          # cols 3-6 (HYPO71 station names are 4 chars max)
        line += f'{lat_deg:2d}'            # cols 7-8
        line += f'{lat_min:5.2f}'          # cols 9-13
        line += ns                         # col 14
        line += f'{lon_deg:3d}'            # cols 15-17
        line += f'{lon_min:5.2f}'          # cols 18-22
        line += ew                         # col 23
        line += f'{elev_m:4d}'             # cols 24-27
        line += f'{0.0:6.2f}'             # cols 28-33: delay
        line += '    '                     # cols 34-37: 4X
        line += f'{0.0:5.2f}'             # cols 38-42: FMGC
        line += '  '                       # cols 43-44: 2X
        line += f'{0.0:5.2f}'             # cols 45-49: XMGC
        line += ' '                        # col 50: 1X
        line += '0'                        # col 51: KLAS
        line += f'{0.0:5.2f}'             # cols 52-56: PRR
        line += f'{0.0:7.2f}'             # cols 57-63: CALR
        line += ' '                        # col 64: 1X
        line += '0'                        # col 65: ICAL
        lines.append(line)

    # Blank station name terminates the station list
    lines.append('')

    # ------------------------------------------------------------------
    # 3. Velocity model
    #    FORMAT(2F7.3); terminated by a card with V < 0.01
    # ------------------------------------------------------------------
    depths = velocity_model.depths
    vp = velocity_model.get_velocities('P')
    for v, d in zip(vp, depths):
        lines.append(f'{v:7.3f}{d:7.3f}')
    lines.append(f'{0.0:7.3f}{0.0:7.3f}')   # terminator

    # ------------------------------------------------------------------
    # 4. Control card
    #    FORMAT(I1,F4.0,2F5.0,F5.2,7I5,5I1,2(I4,F6.2))
    #
    #    Col  1     : KSING (I1; 0 for normal processing)
    #    Cols 2-5   : ZTR  (F4.0; trial depth km)
    #    Cols 6-10  : XNEAR (F5.0)
    #    Cols 11-15 : XFAR  (F5.0)
    #    Cols 16-20 : POS   (F5.2; Vp/Vs)
    #    Cols 21-25 : IQ    (I5;  99 = print all quality classes)
    #    Cols 26-30 : KMS   (I5;  0)
    #    Cols 31-35 : KFM   (I5;  0)
    #    Cols 36-40 : IPUN  (I5;  1 = write punch file)
    #    Cols 41-45 : IMAG  (I5;  0)
    #    Cols 46-50 : IR    (I5;  0)
    #    Cols 51-55 : IPRN  (I5;  4 = verbose residuals)
    #    Col 56     : KPAPER (I1; 0)
    #    Col 57     : KTEST  (I1; 0)
    #    Col 58     : KAZ   (I1; 1 = azimuthal weighting on)
    #    Col 59     : KSORT  (I1; 0)
    #    Col 60     : KSEL   (I1; 0)
    #    Cols 61-64 : LATR deg (I4; 0 = no reference lat)
    #    Cols 65-70 : LATR min (F6.2)
    #    Cols 71-74 : LONR deg (I4)
    #    Cols 75-80 : LONR min (F6.2)
    # ------------------------------------------------------------------
    ctrl  = '0'                         # col 1:  KSING
    ctrl += f'{ztr:4.0f}'              # cols 2-5:  ZTR
    ctrl += f'{xnear:5.0f}'            # cols 6-10: XNEAR
    ctrl += f'{xfar:5.0f}'             # cols 11-15: XFAR
    ctrl += f'{pos:5.2f}'              # cols 16-20: POS
    ctrl += f'{99:5d}'                 # cols 21-25: IQ
    ctrl += f'{0:5d}'                  # cols 26-30: KMS
    ctrl += f'{0:5d}'                  # cols 31-35: KFM
    ctrl += f'{1:5d}'                  # cols 36-40: IPUN (punch on)
    ctrl += f'{0:5d}'                  # cols 41-45: IMAG
    ctrl += f'{0:5d}'                  # cols 46-50: IR
    ctrl += f'{4:5d}'                  # cols 51-55: IPRN
    ctrl += '00100'                    # cols 56-60: KPAPER,KTEST,KAZ,KSORT,KSEL
    ctrl += f'{0:4d}{0.0:6.2f}'       # cols 61-70: LATR
    ctrl += f'{0:4d}{0.0:6.2f}'       # cols 71-80: LONR
    lines.append(ctrl)

    # ------------------------------------------------------------------
    # 5. Phase cards
    #    FORMAT (via T column specifiers — key fields):
    #
    #    Cols 1-4   : station name (A4)
    #    Cols 5-6   : P remark, e.g. 'IP' (onset + phase)
    #    Col  7     : first motion (blank)
    #    Col  8     : P weight digit 0-4 (0 = full weight)
    #    Col  9     : blank
    #    Cols 10-15 : date YYMMDD (I6)
    #    Cols 16-17 : hour HH (I2)
    #    Cols 18-19 : minute MM (I2)
    #    Cols 20-24 : P seconds SS.ss (F5.2)
    #    Cols 32-36 : S seconds SS.ss from start of P's minute (F5.2; blank=no S)
    #    Cols 37-38 : S remark, e.g. 'ES' (blank=no S)
    #    Col  40    : S weight digit (0 = full weight)
    #
    #    TP(L) = 60*JMIN + P_seconds     (absolute within-hour time, seconds)
    #    TS(L) = 60*JMIN + S_seconds     (S_seconds can exceed 60)
    # ------------------------------------------------------------------

    # Determine the reference date/hour from the first P pick
    ref_p = next(
        (_as_utc(v.get('P')) for v in picks.values()
         if _as_utc(v.get('P')) is not None), None
    )
    if ref_p is None:
        raise ValueError("No P picks found in pick dictionary")

    year2 = ref_p.year % 100
    yymmdd = f'{year2:02d}{ref_p.month:02d}{ref_p.day:02d}'
    ref_hour = ref_p.hour

    for sta in stations:
        phase = picks.get(sta.code, {}) or picks.get(sta.sta, {})
        p_time = _as_utc(phase.get('P'))
        s_time = _as_utc(phase.get('S'))

        if p_time is None:
            continue

        p_min = p_time.minute
        p_sec = p_time.second + p_time.microsecond * 1e-6

        # Build an 80-character card as a list for easy column assignment
        card = [' '] * 80

        # Cols 1-4 (idx 0-3): station name — use bare sta name to match station list
        card[0:4] = list(f'{sta.sta[:4]:<4s}')

        # Cols 5-6 (idx 4-5): onset 'I' + phase 'P'
        card[4] = 'I'
        card[5] = 'P'
        # Col 7 (idx 6): first motion — blank
        card[6] = ' '
        # Col 8 (idx 7): P weight digit.
        # Weight digit > 4 → (4-w)/4 < 0 → HYPO71 triggers KSMP(L)=1 (S-P mode).
        # Both absolute P and S times are still written; the Fortran computes TS-TP.
        card[7] = '5' if (use_s_minus_p and s_time is not None) else '0'
        # Col 9 (idx 8): blank
        card[8] = ' '

        # Cols 10-15 (idx 9-14): YYMMDD
        card[9:15] = list(yymmdd)
        # Cols 16-17 (idx 15-16): hour
        card[15:17] = list(f'{ref_hour:02d}')
        # Cols 18-19 (idx 17-18): minute
        card[17:19] = list(f'{p_min:02d}')
        # Cols 20-24 (idx 19-23): P seconds F5.2
        card[19:24] = list(f'{p_sec:5.2f}')

        if s_time is not None:
            # S seconds relative to start of P's minute
            p_min_base = UTCDateTime(
                p_time.year, p_time.month, p_time.day,
                p_time.hour, p_time.minute, 0
            )
            s_sec_card = float(s_time - p_min_base)
            # Cols 32-36 (idx 31-35): S seconds F5.2
            card[31:36] = list(f'{s_sec_card:5.2f}')
            # Cols 37-38 (idx 36-37): S remark 'ES'
            card[36] = 'E'
            card[37] = 'S'
            # Col 40 (idx 39): S weight = 0 (full)
            card[39] = '0'

        lines.append(''.join(card))

    # Instruction/terminating card.  Cols 18-19 encode the processing mode:
    #   JMIN = KNST*10 + INST
    #   KNST=1: include S picks in the Geiger inversion (else S weights are zeroed)
    #   INST=0: free hypocentral solution
    # Blank station name (cols 1-4) terminates the phase list → INPUT2 label 350
    # → MJUMP=0 → SINGLE is called (do NOT use ' ***'/' $$$' dataset terminators).
    knst = 1 if use_s else 0
    inst = 0
    jmin_code = knst * 10 + inst
    inst_card = [' '] * 80
    inst_card[17] = str(jmin_code // 10)   # col 18: KNST digit
    inst_card[18] = str(jmin_code % 10)    # col 19: INST digit
    lines.append(''.join(inst_card))

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Parsing the punch output (HYPO71.PUN)
# ---------------------------------------------------------------------------

def parse_hypo71_pun(path):
    """
    Parse the HYPO71PC punch output file and return a list of solution dicts.

    The punch record format (FORMAT 87 in hypo1m2.f):
        I6,1X,2I2,F6.2,I3,'-',F5.2,I4,'-',F5.2,1X,F6.2,A1,A6,I3,
        I4,F5.1,F5.2,2A5,3A1

    Fields extracted per event:
        kdate   : int   YYMMDD
        khr     : int   hour
        kmin    : int   minute of origin
        sec     : float seconds of origin
        lat_deg : int   latitude integer degrees (negative = South)
        lat_min : float latitude minutes (always positive)
        lon_deg : int   longitude degrees (positive; abs applied in Fortran)
        lon_min : float longitude minutes (positive)
        depth   : float km
        rms     : float seconds
        no      : int   number of phases used
        gap     : int   azimuthal gap (degrees)
        dmin    : float min station distance (km)
        quality : str   e.g. 'AA'

    Derived:
        lat     : float decimal degrees (sign from lat_deg sign)
        lon     : float decimal degrees (always positive — sign unknown from
                  punch; caller must apply if needed)
        origin  : UTCDateTime (approximate; year reconstructed to 4 digits)
    """
    results = []
    with open(path) as fh:
        lines = fh.readlines()

    # The PUN header (written once per file by main.f FORMAT 41) encodes the
    # hemisphere for the whole file, e.g. "LAT S    LONG E".  HYPO71 strips the
    # sign from lat_deg/lon_deg in FORMAT 87 (via abs()/iabs()), so we must
    # recover it from the header.
    lat_sign = 1    # default: Northern hemisphere
    lon_sign = 1    # default: Eastern hemisphere
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith('DATE') and 'LAT' in stripped:
            # e.g. "DATE    ORIGIN    LAT S    LONG E    DEPTH ..."
            if 'LAT S' in stripped:
                lat_sign = -1
            if 'LONG W' in stripped:
                lon_sign = -1
            break

    for raw in lines:
        line = raw.rstrip('\n\r')

        # Skip the header, blank lines, and dataset markers
        if len(line) < 40:
            continue
        if line.lstrip().startswith('DATE') or '$$$' in line or '***' in line:
            continue

        try:
            pos = 0
            kdate    = int(line[pos:pos+6]);  pos += 6   # I6
            pos += 1                                       # 1X
            khr      = int(line[pos:pos+2]);  pos += 2   # I2
            kmin     = int(line[pos:pos+2]);  pos += 2   # I2
            sec      = float(line[pos:pos+6]); pos += 6  # F6.2
            lat_deg  = int(line[pos:pos+3]);  pos += 3   # I3
            pos += 1                                       # '-'
            lat_min  = float(line[pos:pos+5]); pos += 5  # F5.2
            lon_deg  = int(line[pos:pos+4]);  pos += 4   # I4
            pos += 1                                       # '-'
            lon_min  = float(line[pos:pos+5]); pos += 5  # F5.2
            pos += 1                                       # 1X
            depth    = float(line[pos:pos+6]); pos += 6  # F6.2
            pos += 1                                       # A1 (RMK2)
            pos += 6                                       # A6 (MAGOUT)
            no       = int(line[pos:pos+3]);  pos += 3   # I3
            gap      = int(line[pos:pos+4]);  pos += 4   # I4
            dmin     = float(line[pos:pos+5]); pos += 5  # F5.1
            rms      = float(line[pos:pos+5]); pos += 5  # F5.2
            erh_str  = line[pos:pos+5];       pos += 5   # A5
            erz_str  = line[pos:pos+5];       pos += 5   # A5
            quality  = line[pos:pos+3].strip()            # 3A1

            # HYPO71 FORMAT 87 uses abs()/iabs() for both lat and lon degrees,
            # so both are always positive in the punch record.  Apply the
            # hemisphere sign read from the file header.
            lat = lat_sign * (abs(lat_deg) + lat_min / 60.0)
            lon = lon_sign * (abs(lon_deg) + lon_min / 60.0)

            # Reconstruct approximate UTCDateTime.
            # kdate = YYMMDD; year is 2-digit — assume 2000s.
            # Fortran can produce sec >= 60 (meaning rollover to next minute).
            yy = kdate // 10000
            mm = (kdate % 10000) // 100
            dd = kdate % 100
            year = 2000 + yy
            base = UTCDateTime(year, mm, dd, khr, kmin, 0)
            origin = base + sec

            results.append(dict(
                kdate=kdate, khr=khr, kmin=kmin, sec=sec,
                lat=lat, lon=lon, depth=depth,
                rms=rms, no=no, gap=gap, dmin=dmin,
                quality=quality, origin=origin,
                erh=erh_str.strip(), erz=erz_str.strip(),
            ))
        except (ValueError, IndexError):
            # Skip malformed lines (overflow fields show as '***' etc.)
            continue

    return results
