"""
Run one simulation of the 3D pipeline: a half stride of walking or running of the standard SIPP runmad model.

    python run.py 1.2                                  # set speed 1.2 m/s, metabolic cost, seed 1
    python run.py                                      # self-chosen speed (free within --speed_range)
    python run.py 4.5 --objective effort --seed 3
    python run.py --muscle_weakness 50                 # bilateral hip abductors with 50 % of their strength
    python run.py 4.5 --objective effort --contact_stiffness 4 --contact_damping 0

The run starts from the standing solution of its seed (made first when ``<out_dir>/standing/seed<k>.pkl`` does not exist)
and is the same problem as ``paper_walking.py`` builds (``configs/walking.yaml``: Bhargava cost or muscle effort, learned
state prior, arm effort, heel-strike variance penalty), for one seed. The result is
``<out_dir>/<name>/v<speed>/seed<k>.pkl`` (``free`` instead of ``v<speed>`` for a self-chosen speed); ``<name>`` is the
objective (``walking`` for metabolic, ``effort``) followed by the changes below, e.g. ``effort_weak50_k4_c0``. An existing
result is not solved again. Exit code 0 when IPOPT found an optimal solution.

Model changes (all relative to the standard model; the changed model files are written to ``<out_dir>/_models/<name>/``):

``--muscle_weakness PCT``
    PCT percent of the strength (maximum isometric force) of the muscles in ``--weak_muscles`` is removed; default the seven
    hip abductors of both legs (glut_med1-3, glut_min1-3, tfl), as in the paper's bilateral abductor weakness (Figure 4b).
``--contact_stiffness FACTOR``
    Factor on the stiffness (1e6 N/m^2) of every contact sphere; 4 is the paper's "stiffer" contact (Figure 4c).
``--contact_damping FACTOR``
    Factor on the dissipation (2 s/m) of every contact sphere; 0 is the paper's contact without dissipation.

Every option also accepts the form with dashes (``--muscle-weakness``).
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import paper_common as pc  # noqa: E402  (changes to the project folder)
import paper_walking  # noqa: E402

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "models" / "sipp_runmad"
ABDUCTORS = tuple(f"{m}_{side}" for side in "rl" for m in ("glut_med1", "glut_med2", "glut_med3", "glut_min1", "glut_min2",
                                                          "glut_min3", "tfl"))


def number(value):
    """Short text of a number for a folder name: 4 -> ``4``, 0.5 -> ``0.5``."""
    return f"{value:g}"


def weaken_muscles(osim_text, muscles, weakness):
    """
    Scale the maximum isometric force of the given muscles in the text of an OpenSim model.

    Parameters
    ----------
    osim_text : str
        Content of the ``.osim`` file.
    muscles : iterable of str
        Names of the muscles to weaken; every one must exist in the model.
    weakness : float
        Fraction of the strength that is removed, 0 to below 1.

    Returns
    -------
        The changed text.
    """
    keep = 1.0 - weakness
    for name in muscles:
        block = re.search(rf'<(\w+) name="{re.escape(name)}">.*?</\1>', osim_text, re.S)
        if block is None:
            raise SystemExit(f"--weak_muscles: no muscle {name} in the model")
        force = re.search(r"<max_isometric_force>([^<]+)</max_isometric_force>", block.group(0))
        if force is None:
            raise SystemExit(f"--weak_muscles: {name} has no max_isometric_force")
        new = block.group(0).replace(force.group(0), f"<max_isometric_force>{float(force.group(1)) * keep:.10g}</max_isometric_force>")
        osim_text = osim_text[:block.start()] + new + osim_text[block.end():]
    return osim_text


def scale_contact(xml_text, stiffness, damping):
    """
    Scale the stiffness and the dissipation of every contact point in the text of a contact xml.

    Parameters
    ----------
    xml_text : str
        Content of the contact model xml.
    stiffness, damping : float
        Factors on ``stiffness`` and ``dissipation``.

    Returns
    -------
        The changed text.
    """
    def scaled(attribute, factor):
        return lambda m: f'{attribute}="{float(m.group(1)) * factor:.10g}"'

    xml_text, n_k = re.subn(r'\bstiffness="([^"]+)"', scaled("stiffness", stiffness), xml_text)
    xml_text, n_c = re.subn(r'\bdissipation="([^"]+)"', scaled("dissipation", damping), xml_text)
    if not n_k or not n_c:
        raise SystemExit("the contact xml has no stiffness / dissipation to scale")
    return xml_text


def write_model(directory, weakness, muscles, stiffness, damping):
    """
    Write a variant of the standard model to ``directory`` and return the path of its yaml.

    Unchanged files are links to the standard model's, so only a changed ``.osim`` or contact xml takes space.

    Parameters
    ----------
    directory : pathlib.Path
        Folder of the variant model.
    weakness : float
        Fraction of the strength removed from ``muscles``.
    muscles : iterable of str
        Muscles to weaken.
    stiffness, damping : float
        Factors on the contact stiffness and dissipation.

    Returns
    -------
        Path of the model yaml, relative to the project folder.
    """
    directory.mkdir(parents=True, exist_ok=True)

    def link(name):
        target = directory / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(MODEL_DIR / name)

    yaml_text = (MODEL_DIR / "sipp_runmad.yaml").read_text()
    link("Geometry")
    if weakness > 0:
        (directory / "sipp_runmad.osim").unlink(missing_ok=True)
        (directory / "sipp_runmad.osim").write_text(weaken_muscles((MODEL_DIR / "sipp_runmad.osim").read_text(), muscles, weakness))
    else:
        link("sipp_runmad.osim")
    contact = "contact_implicit.xml"
    if stiffness != 1 or damping != 1:
        (directory / contact).unlink(missing_ok=True)
        (directory / contact).write_text(scale_contact((MODEL_DIR / contact).read_text(), stiffness, damping))
    else:
        link(contact)
    path = directory / "sipp_runmad.yaml"
    path.write_text(yaml_text)
    return path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("speed", type=float, nargs="?", default=None,
                        help="set speed [m/s]; omit for a self-chosen speed (free within --speed_range)")
    parser.add_argument("--speed_range", "--speed-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.5, 2.0),
                        help="bounds [m/s] of the self-chosen speed when no speed is given (default 0.5 2.0)")
    parser.add_argument("--seed", type=int, default=1, help="random seed: the standing start of the run (default 1)")
    parser.add_argument("--objective", choices=("metabolic", "effort"), default="metabolic",
                        help="task objective: Bhargava cost of transport (default) or the cubed muscle effort")
    parser.add_argument("--muscle_weakness", "--muscle-weakness", type=float, default=0.0, metavar="PCT",
                        help="percent of the strength removed from --weak_muscles (default 0)")
    parser.add_argument("--weak_muscles", "--weak-muscles", nargs="+", default=list(ABDUCTORS), metavar="MUSCLE",
                        help="muscles that --muscle_weakness weakens (default: the hip abductors of both legs)")
    parser.add_argument("--contact_stiffness", "--contact-stiffness", type=float, default=1.0, metavar="FACTOR",
                        help="factor on the contact stiffness (default 1)")
    parser.add_argument("--contact_damping", "--contact-damping", type=float, default=1.0, metavar="FACTOR",
                        help="factor on the contact dissipation (default 1, 0 for none)")
    parser.add_argument("--nnodes", type=int, default=50, help="nodes of the half cycle (default 50)")
    parser.add_argument("--linear_solver", "--linear-solver", default=None, help="IPOPT linear solver, e.g. ma57, ma97, mumps")
    parser.add_argument("--max_iter", "--max-iter", type=int, default=None, help="IPOPT iteration limit (default: the config's)")
    parser.add_argument("--dur_min", "--dur-min", type=float, default=None, help="lower bound of the half-cycle duration [s]")
    parser.add_argument("--dur_max", "--dur-max", type=float, default=None,
                        help="upper bound of the half-cycle duration [s] (default: running3D.m's for the speed)")
    parser.add_argument("--weight", action="append", default=[], metavar="TERM=VALUE",
                        help="objective weight of a term (repeatable), e.g. --weight state_prior_term=2")
    parser.add_argument("--cot_arg", "--cot-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="extra argument of the cot_term objective (repeatable), e.g. --cot_arg work=absolute")
    parser.add_argument("--warm_from", "--warm-from", default=None, metavar="DIR",
                        help="start from the converged solution DIR/seed<k>.pkl instead of the standing solution")
    parser.add_argument("--log", action="store_true", help="iteration logger (terms and constraints every 25 iterations)")
    parser.add_argument("--dashboard", action="store_true", help="with --log: also the live dashboard (port 8050)")
    parser.add_argument("--out_dir", "--out-dir", default=str(pc.RESULT_DIR), help="results folder (default results)")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="write the model variant and print the run, do not solve")
    args = parser.parse_args()
    if not 0 <= args.muscle_weakness < 100:
        parser.error("--muscle_weakness is a percentage in [0, 100)")
    if args.contact_stiffness <= 0 or args.contact_damping < 0:
        parser.error("--contact_stiffness must be positive and --contact_damping not negative")

    name = "effort" if args.objective == "effort" else "walking"
    changed = []
    if args.muscle_weakness:
        changed.append(f"weak{number(args.muscle_weakness)}")
    if args.contact_stiffness != 1:
        changed.append(f"k{number(args.contact_stiffness)}")
    if args.contact_damping != 1:
        changed.append(f"c{number(args.contact_damping)}")
    variant = "_".join([name] + changed)

    command = ["--variant", variant, "--cost", args.objective, "--seeds", str(args.seed), "--n-converged", "1",
               "--tag", f"_s{args.seed}", "--nnodes", str(args.nnodes), "--out-dir", args.out_dir]
    command = ([str(args.speed)] if args.speed is not None else ["--free-speed", *map(str, args.speed_range)]) + command
    if changed:
        model = write_model(Path(args.out_dir) / "_models" / variant, args.muscle_weakness / 100, args.weak_muscles,
                            args.contact_stiffness, args.contact_damping)
        command += ["--model", str(model)]
    for flag, value in (("--linear-solver", args.linear_solver), ("--max-iter", args.max_iter), ("--dur-min", args.dur_min),
                        ("--dur-max", args.dur_max), ("--warm-from", args.warm_from)):
        if value is not None:
            command += [flag, str(value)]
    for item in args.weight:
        command += ["--weight", item]
    for item in args.cot_arg:
        command += ["--cot-arg", item]
    command += ["--log"] * args.log + ["--dashboard"] * (args.log and args.dashboard)
    if args.dashboard and not args.log:
        parser.error("--dashboard needs --log")

    folder = "free" if args.speed is None else f"v{args.speed:.2f}"
    print(f"run: {variant}, seed {args.seed}, speed {'free ' + str(list(args.speed_range)) if args.speed is None else args.speed}")
    print(f"result: {Path(args.out_dir) / variant / folder / f'seed{args.seed}.pkl'}")
    if args.dry_run:
        print("paper_walking.py " + " ".join(command))
        return 0
    sys.argv = ["paper_walking.py", *command]
    return paper_walking.main()


if __name__ == "__main__":
    sys.exit(main())
