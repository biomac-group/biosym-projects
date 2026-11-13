"""
Give every MS-Human-700 muscle a name that pairs it with its left/right counterpart.

The torso muscles of MS-Human-700 do not pair by name (``_r``/``_l``, ``_R``/``_L``, right-side names without a suffix such
as ``splen_cerv_c3_T3`` next to ``splen_cerv_c3_T3_L``, midline muscles). biosym's mirror symmetry pairs states by the last
``_r``/``_l`` token of their name, so this script pairs the muscles by geometry instead: the path sites of a muscle,
mirrored through the sagittal plane of the pelvis (z -> -z), are compared with the path sites of every other muscle with
the same number of sites, in the default pose. Pairs the names imply (``x_r``/``x_l``, ``x_R``/``x_L``, ``x``/``x_L``) are
taken as they are (one of them, ``LTpT_T12``, is not mirror symmetric in the MJCF; pairs off by more than
``NAME_PAIR_MM`` are reported); the remaining muscles are paired by an optimal assignment on the mirror
distance (nearest neighbours alone are ambiguous between adjacent fascicles), and a muscle assigned to itself is a midline
muscle. The new name is the common stem with ``_r`` (the side with z > 0, OpenSim's right) or
``_l``; a midline muscle keeps its name, with any side token removed.

    uv run --with "mujoco>=3.4" python scripts/pair_muscles.py MODEL      # writes models/MODEL/muscle_names.csv

Columns of the csv: ``mujoco_name`` (the name in the MJCF), ``name`` (the new name), ``partner`` (the new name of the
counterpart, empty for a midline muscle), ``mirror_error_mm`` (mean site distance to the mirrored counterpart).
"""
import csv
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import PROJECT, from_argv  # noqa: E402

SIDE_SUFFIX = re.compile(r"[_-](r|l|R|L)$")
SIDE_TOKEN = re.compile(r"_([lr])(?=_|$)")      # biosym.constraints.periodicity._SIDE_TOKEN
MIDLINE_MM = 5.0                                  # a muscle this close to its own mirror image is a midline muscle
NAME_PAIR_MM = 5.0                                # a pair the names imply with a larger mirror distance is reported


def path_sites(m, d, actuator, pelvis):
    """
    Positions of the path sites of an actuator's tendon in the pelvis frame.

    Parameters
    ----------
    m, d : mujoco.MjModel, mujoco.MjData
        Model and data after ``mj_forward`` in the pose to compare in.
    actuator : int
        Actuator index (its transmission is a tendon).
    pelvis : int
        Body index of the pelvis.

    Returns
    -------
        ``(n_sites, 3)`` array, x forward, y up, z right.
    """
    tendon = m.actuator_trnid[actuator, 0]
    adr, num = m.tendon_adr[tendon], m.tendon_num[tendon]
    sites = [m.wrap_objid[w] for w in range(adr, adr + num) if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_SITE)]
    R, p = d.xmat[pelvis].reshape(3, 3), d.xpos[pelvis]
    return (d.site_xpos[sites] - p) @ R


def stem(name):
    """The name without a trailing side suffix (``_r``, ``_l``, ``_R``, ``_L``, also with ``-``)."""
    return SIDE_SUFFIX.sub("", name)


def pair_muscles(mjcf, pelvis_body="pelvis"):
    """
    Pair the muscles of an MJCF model by mirrored path geometry and name them by side.

    Parameters
    ----------
    mjcf : str or Path
        The MJCF.
    pelvis_body : str
        Body whose frame has z to the right (the mirror plane is its x-y plane).

    Returns
    -------
        List of dicts with ``mujoco_name``, ``name``, ``partner``, ``mirror_error_mm``, in actuator order.
    """
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pelvis_body)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    sites = [path_sites(m, d, i, pelvis) for i in range(m.nu)]
    mirror = np.diag([1.0, 1.0, -1.0])

    def distance(i, j):
        return 1e3 * float(np.mean(np.linalg.norm(sites[i] @ mirror - sites[j], axis=1)))

    def comparable(i, j):
        return len(sites[i]) == len(sites[j])

    # 1. pairs the names imply (stem_r / stem_l, stem_R / stem_L, or a right-side name without a suffix next to stem_L),
    #    accepted when the geometry agrees
    index = {n: i for i, n in enumerate(names)}
    pairs = []
    for i, n in enumerate(names):
        s = SIDE_SUFFIX.search(n)
        if s and s.group(1) in "rR":
            options = [stem(n) + suffix for suffix in ("_l", "_L", "-l", "-L")]
        elif s is None:
            options = [n + suffix for suffix in ("_l", "_L")]
        else:
            continue
        for o in options:
            j = index.get(o)
            if j is not None and comparable(i, j):
                pairs.append((i, j))
                if distance(i, j) > NAME_PAIR_MM:
                    print(f"warning: {n} and {o} pair by name, but their paths are {distance(i, j):.1f} mm from "
                          "mirror images of each other (asymmetric in the MJCF)")
                break
    paired = {k for p in pairs for k in p}
    if len(paired) != 2 * len(pairs):
        raise RuntimeError("a muscle is in two name pairs")

    # 2. the rest: optimal assignment on the mirror distance, a muscle assigned to itself is a midline muscle
    rest = [i for i in range(m.nu) if i not in paired]
    cost = np.full((len(rest), len(rest)), 1e6)
    for a, i in enumerate(rest):
        for b, j in enumerate(rest):
            if comparable(i, j):
                cost[a, b] = distance(i, j)
    rows_a, cols_b = linear_sum_assignment(cost)
    midline = []
    for a, b in zip(rows_a, cols_b):
        i, j = rest[a], rest[b]
        if cost[a, b] >= 1e6 or (i == j and cost[a, b] > MIDLINE_MM):
            raise RuntimeError(f"{names[i]}: no mirror partner ({cost[a, b]:.1f} mm)")
        if i == j:
            midline.append(i)
        elif i < j:
            if cost[b, a] != cost[a, b] and rest[cols_b[b]] != i:
                raise RuntimeError(f"{names[i]} -> {names[j]} is not a symmetric assignment")
            pairs.append((i, j))

    rows = [None] * m.nu
    for i in midline:
        new = SIDE_TOKEN.sub("", stem(names[i]))
        rows[i] = {"mujoco_name": names[i], "name": new, "partner": "", "mirror_error_mm": round(distance(i, i), 3)}
    for i, j in pairs:
        err = distance(i, j)
        right, left = (i, j) if sites[i][:, 2].mean() > 0 else (j, i)
        base = stem(names[left])
        rows[right] = {"mujoco_name": names[right], "name": base + "_r", "partner": base + "_l", "mirror_error_mm": round(err, 3)}
        rows[left] = {"mujoco_name": names[left], "name": base + "_l", "partner": base + "_r", "mirror_error_mm": round(err, 3)}

    new_names = [r["name"] for r in rows]
    duplicates = sorted({n for n in new_names if new_names.count(n) > 1})
    if duplicates:
        raise RuntimeError(f"new muscle names are not unique: {duplicates[:10]}")
    # the pairing biosym's periodicity will see: last _l/_r token swapped
    for r in rows:
        tokens = list(SIDE_TOKEN.finditer(r["name"]))
        partner = None
        if tokens:
            t = tokens[-1]
            partner = r["name"][:t.start()] + "_" + ("r" if t.group(1) == "l" else "l") + r["name"][t.end():]
        if (partner or "") != r["partner"]:
            raise RuntimeError(f"{r['name']}: biosym would pair it with {partner}, the geometry with {r['partner']}")
    return rows


def main():
    cfg = from_argv()
    OUT = cfg.names_csv
    rows = pair_muscles(cfg.mjcf, cfg.root_frame)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mujoco_name", "name", "partner", "mirror_error_mm"])
        writer.writeheader()
        writer.writerows(rows)
    renamed = [r for r in rows if r["name"] != r["mujoco_name"]]
    pairs = sum(1 for r in rows if r["partner"]) // 2
    errors = np.array([r["mirror_error_mm"] for r in rows if r["partner"]])
    print(f"{len(rows)} muscles: {pairs} pairs, {len(rows) - 2 * pairs} midline; {len(renamed)} renamed; "
          f"mirror error of the pairs: median {np.median(errors):.2f} mm, max {errors.max():.2f} mm")
    for r in renamed[:12]:
        print(f"   {r['mujoco_name']:28s} -> {r['name']}")
    print(f"written: {OUT.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
