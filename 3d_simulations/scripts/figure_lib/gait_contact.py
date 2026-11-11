"""Contact detection for cutting a continuous trace into gait cycles.

Every segmenter in this project used to do the same thing: threshold the raw
vertical GRF at 0.1 BW and take the rising edges as heelstrikes. That is one
comparison against one number on an unfiltered signal, and it fails whenever the
trace dips through the threshold mid-stance -- a false heelstrike, which cuts a
stance phase into a "cycle" that never unloads.

That failure was measured, not assumed: 48 of GaitNet's 424 walking cycles are
such fragments (13/57 at 0.8 m/s, 26/145 at 1.0, 9/145 at 1.2), each about half
the duration of a real cycle at the same speed. The existing `_single_grf_block`
screen cannot catch them -- it counts *extra* rising edges, and a fragment that
never unloads has none.

Three things are done here, in order:

  filter       the contact signal is low-passed before it is compared to
               anything. This is a DETECTION signal only -- the stored GRF is
               never filtered, so no reported curve is smoothed by this module.

  hysteresis   contact turns on at `on_bw` and off at the lower `off_bw`, so a
               dip has to be a real unloading, not a graze past one number, to
               end a stance phase.

  debounce     stance or flight phases shorter than `min_phase_s` are absorbed
               into their neighbour. Real toe-off/heelstrike events are tens of
               milliseconds apart at the very least.

and then `cycle_bounds` applies a duration guard relative to the trace's OWN
median cycle, which is what catches a fragment that survives all three: a
0.7 s cycle is obviously wrong beside a 1.28 s median, while an absolute bound
loose enough for both walking and running cannot see it.

The defaults are deliberately conservative: on data with no dip artefact they
reproduce the plain rising-edge result exactly (verified against the whole
GaitDynamics set, 594 cycles, unchanged).
"""
import numpy as np

# Contact on/off in body weight: one threshold of 5 % BW (no hysteresis gap, ON_BW == OFF_BW), the same as the
# heel-strike roll in plot_figure1.heel_strike_start. The low-pass and the debounce below still apply, so a graze of
# the threshold does not end a stance phase unless it lasts MIN_PHASE_S. Caveat: GaitDynamics generations carry a soft
# swing floor around 0.06 BW, which sits above 5 % BW, so their swing can read as contact (duty factor too high).
ON_BW = 0.05
OFF_BW = 0.05

# Shorter than this, a stance or flight phase is a detection artefact rather
# than an event: real double-support is ~100 ms even at speed.
MIN_PHASE_S = 0.04

# Contact detection only. Gait GRF power sits well below this; the transients it
# removes are precisely the ones that produce spurious threshold crossings.
CUTOFF_HZ = 10.0

# A cycle is kept if its duration lies within this factor of the trace's median
# cycle. A stance-only fragment is ~0.5x, so this catches it while leaving the
# real stride-to-stride variation (a few per cent) untouched.
DUR_BAND = (0.65, 1.55)

# Hard ceiling on the stance fraction of a cycle. A gait cycle runs heelstrike to
# heelstrike and so MUST contain an unloaded phase; running sits near 0.35, slow
# walking near 0.75, and even pathological shuffling stays under ~0.85. This is
# the screen of last resort for a stance fragment, and unlike DUR_BAND it does
# not depend on the rest of the trace being sane -- which matters, because the
# relative guard inverts on a rollout where the junk cycles are the majority and
# the median is therefore itself a fragment. That case is real: one GaitNet
# rollout yielded exactly two cycles, a duty-0.96 fragment and a duty-0.10 fall,
# and the fragment passed the band comfortably.
MAX_DUTY = 0.95


def _lowpass(x, fs, cutoff=CUTOFF_HZ, order=4):
    """Zero-phase Butterworth low-pass, with the edge cases handled.

    Zero-phase matters: a causal filter would delay the detected heelstrike by
    a few frames and so shift every cycle boundary, which is exactly the kind of
    silent phase error the overlay figures exist to catch.
    """
    nyq = 0.5 * fs
    if not np.isfinite(nyq) or cutoff >= nyq or len(x) < 4 * order:
        return np.asarray(x, dtype=float)      # too short or already band-limited
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff / nyq, btype='low')
    # padlen must stay under the signal length or filtfilt raises.
    return filtfilt(b, a, np.asarray(x, dtype=float),
                    padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def contact_mask(vgrf_bw, fs, on_bw=ON_BW, off_bw=OFF_BW,
                 min_phase_s=MIN_PHASE_S, cutoff=CUTOFF_HZ):
    """Boolean stance mask from a vertical GRF trace in body weight.

    `fs` is the sampling rate in Hz. Filtering, hysteresis and debouncing are
    applied in that order; see the module docstring.
    """
    v = _lowpass(vgrf_bw, fs, cutoff)

    # Hysteresis: walk the trace once, carrying the current state. Between the
    # two thresholds the state is held, which is the whole point -- a single
    # threshold has no state and so cannot tell a dip from an unloading.
    on, off = v > on_bw, v < off_bw
    mask = np.zeros(len(v), dtype=bool)
    state = bool(on[0]) if len(v) else False
    for i in range(len(v)):
        if state and off[i]:
            state = False
        elif not state and on[i]:
            state = True
        mask[i] = state

    min_len = max(int(round(min_phase_s * fs)), 1)
    return _debounce(mask, min_len)


def _debounce(mask, min_len):
    """Absorb runs shorter than `min_len` into the surrounding phase."""
    if min_len <= 1 or not len(mask):
        return mask
    out = mask.copy()
    edges = np.flatnonzero(np.diff(out.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(out)]
    for s, e in zip(starts, ends):
        # An interior run only; a short run at either end has no unambiguous
        # neighbour to be absorbed into and is left as it is.
        if e - s < min_len and s > 0 and e < len(out):
            out[s:e] = out[s - 1]
    return out


SNAP_S = 0.05


def _snap_to_raw(strikes, vgrf_bw, on_bw=ON_BW, tol=3):
    """Move each detected heelstrike onto the nearest RAW threshold crossing.

    Filtering decides *which* events are real; it must not decide *when* they
    happen. A zero-phase low-pass still rounds the shoulder of the loading ramp
    and so moves the 0.1 BW crossing by about a frame, which would shift every
    cycle boundary on data that had nothing wrong with it. Snapping back to the
    raw crossing keeps the boundaries bit-identical wherever the plain
    rising-edge rule was already correct, while the hysteresis and debouncing
    above still remove the spurious events.

    A candidate with no raw crossing within `tol` frames is left where it is.
    """
    v = np.asarray(vgrf_bw, dtype=float)
    raw = np.flatnonzero((v[1:] > on_bw) & (v[:-1] <= on_bw)) + 1
    if not len(raw) or not len(strikes):
        return strikes
    out = []
    for c in strikes:
        j = raw[np.argmin(np.abs(raw - c))]
        out.append(j if abs(j - c) <= tol else c)
    # Two candidates can snap onto the same raw crossing; keep one.
    return np.unique(out)


def has_single_contact_phase(vgrf_bw, fs, **kw):
    """True if a heelstrike-first, single-gait-cycle vGRF trace shows no more
    than one discrete contact phase.

    Same idea as `data_exploration_main.ipynb`'s `_is_not_2_peaks` exclusion
    criterion (drop trials with a spurious extra contact burst), but built on
    `contact_mask`'s hysteresis+lowpass+debounce rather than a raw single
    threshold: a raw 0.10 BW crossing flags ~40% of the published Falisse
    2022 benchmark as "bad", almost all of it boundary noise where the last
    frame sits a few thousandths above threshold as the next heelstrike
    starts loading in -- exactly the false-positive case hysteresis exists to
    reject (see the module docstring).

    A well-formed heelstrike-first cycle is already in contact at frame 0, so
    a clean single stance region produces ZERO internal rising edges in the
    debounced mask; a solve that lands on a bad local optimum can still
    report `converged == True` (IPOPT "Solved_To_Acceptable_Level") while its
    GRF trace shows a genuine spurious extra contact burst mid-cycle -- that
    shows up as >=1 rising edge here and gets dropped. `fs` should be the
    trial's own sampling rate (FS / dur), matching `contact_mask`'s other
    callers, so the debounce window scales to the trial's actual gait-cycle
    duration rather than assuming a fixed frame rate.

    A "heelstrike-first" storage convention means the frame-0/frame-(n-1)
    array boundary sits exactly ON the loading transition itself -- the worst
    possible place for it. `_lowpass`'s zero-phase filter has no data outside
    [0, n) to support that edge, so the last frame or two routinely lands a
    few thousandths either side of `on_bw` as the (same) next heelstrike
    loads in, and the hysteresis walk misreads it as a second, spurious
    contact phase. Rolling the trace half a cycle before detection moves that
    boundary into the middle of the opposite (well-supported, near-zero or
    near-peak) phase instead, so the expected count becomes 1 genuine rising
    edge rather than 0 -- an extra, real burst mid-cycle still adds a second.
    """
    v = np.asarray(vgrf_bw, dtype=float)
    n = len(v)
    if not n:
        return True
    rolled = np.roll(v, n // 2)
    mask = contact_mask(rolled, fs, **kw)
    n_rising = int(np.sum(np.diff(mask.astype(np.int8)) > 0))
    return n_rising <= 1


def heelstrike_indices(vgrf_bw, fs, **kw):
    """Indices where the foot transitions unloaded -> loaded."""
    mask = contact_mask(vgrf_bw, fs, **kw)
    strikes = np.flatnonzero(np.diff(mask.astype(np.int8)) == 1) + 1
    return _snap_to_raw(strikes, vgrf_bw, kw.get('on_bw', ON_BW),
                        tol=max(int(round(SNAP_S * fs)), 1))


def cycle_bounds(vgrf_bw, fs, min_cycle_s=0.30, max_cycle_s=2.00,
                 dur_band=DUR_BAND, max_duty=MAX_DUTY, **kw):
    """(start, end) index pairs of the complete gait cycles in a trace.

    Cycles are heelstrike to heelstrike, screened first by the absolute duration
    bounds and then against the median duration of the cycles that survived --
    the relative screen is what distinguishes a stance-only fragment from a
    genuinely slow cycle, since both pass any absolute bound wide enough to
    admit walking and running alike.
    """
    mask = contact_mask(vgrf_bw, fs, **kw)
    strikes = np.flatnonzero(np.diff(mask.astype(np.int8)) == 1) + 1
    strikes = _snap_to_raw(strikes, vgrf_bw, kw.get('on_bw', ON_BW),
                           tol=max(int(round(SNAP_S * fs)), 1))
    if len(strikes) < 2:
        return []

    pairs = [(a, b) for a, b in zip(strikes[:-1], strikes[1:])
             if min_cycle_s <= (b - a) / fs <= max_cycle_s
             and mask[a:b].mean() <= max_duty]
    if not pairs:
        return []

    durs = np.array([(b - a) / fs for a, b in pairs])
    med = float(np.median(durs))
    lo, hi = dur_band[0] * med, dur_band[1] * med
    return [p for p, d in zip(pairs, durs) if lo <= d <= hi]
