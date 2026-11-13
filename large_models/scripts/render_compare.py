"""
Render the walking of the three models from the front, at the same speed, and put them side by side with headers.

    uv run --extra render python scripts/render_compare.py effort|metabolic [--camera front|side|isometric|top] [--force]
                                                           [--only sipp,ms_human_700,myofullbody] [--pair sipp,myofullbody] [--unlabeled]

Each clip is real time (30 fps), 4 strides, same camera, size and background, muscles drawn as tubes coloured by
activation with one scale (red at ``ACTIVATION_MAX``) for all three. The clips differ in the length of a stride, so they
are looped to 6 s before they are tiled left to right under a header each (model, then degrees of freedom, muscles and
speed) and above a footer that names the objective all simulations share:

* ``effort``: SIPP (3d_simulations, 1.3 m/s, effort objective, the converged seed with the lowest objective in
  ``effort/v1.30/summary_a.csv``), MS-Human-700 and MyoFullBody (1.3 m/s, cubed muscle effort weight 300 + prior);
* ``metabolic``: SIPP (3d_simulations, 1.3 m/s, metabolic objective, best seed of ``walking/v1.30``), MS-Human-700 and
  MyoFullBody (1.3 m/s, log cost of transport + prior).

``--pair a,b`` composes just those two models (rendering their clips when needed), each cropped to ``CROP[camera]`` around the
figure so that the two are close together: ``<camera>_<a>_vs_<b>_<objective>.mp4``.
``--unlabeled`` leaves out the headers and the footer: the clips side by side, nothing else
(``<camera>_comparison_<objective>_unlabeled.mp4``, or ``..._<a>_vs_<b>_<objective>_unlabeled.mp4`` with ``--pair``).
``--only`` renders just those clips and stops before the composition (a clip can be made as soon as its solution exists).
``--camera isometric`` looks from front-right above at equal angles to the three axes (35.3 deg elevation, 45 deg azimuth).
Clips already rendered (``results/comparison/<camera>_<model>_<objective>.mp4``) are reused unless ``--force`` is given or
the solution file is newer than the clip or is another file than the one it was rendered from (``<clip>.source``). Output: ``results/comparison/<camera>_comparison_<objective>.mp4``.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECTS = Path(__file__).resolve().parents[2]
RESULTS = PROJECTS / "large_models" / "results" / "comparison"
ACTIVATION_MAX = 0.5      # activation drawn fully red, the same for the three models
CAMERAS = {"front": "front", "side": "side", "isometric": (3.2, -3.2, 3.2), "top": (0.05, 0.0, 4.2)}   # name or offset (m, Blender axes: x forward, y left, z up)
SIZE = (720, 1080)
# x, y, width, height of the crop of a clip for ``--pair`` (around the followed pelvis); the side view needs more width
# because the legs swing along the picture
CROP = {"front": (140, 200, 440, 640), "isometric": (140, 200, 440, 640), "side": (90, 200, 540, 640),
        "top": (0, 0, SIZE[0], SIZE[1])}      # to be set after looking at a top frame
HEADER_HEIGHT = 100
FOOTER_HEIGHT = 56
SECONDS = 6
MODELS = {
    "sipp": ("RunMad-SIPP", "33 DoF, 92 muscles + 10 arm torques", PROJECTS / "3d_simulations/models/sipp_runmad/sipp_runmad.yaml"),
    "ms_human_700": ("MS-Human-700", "43 DoF, 700 muscles", PROJECTS / "large_models/models/ms_human_700/ms_human_700.yaml"),
    "myofullbody": ("MyoFullBody", "77 DoF, 416 muscles", PROJECTS / "large_models/models/myofullbody/myofullbody.yaml"),
}


def best_sipp(kind, speed):
    """
    The converged SIPP seed with the lowest objective of a speed in ``3d_simulations/results``.

    Parameters
    ----------
    kind : str
        ``effort`` or ``walking`` (the metabolic objective).
    speed : float
        Speed [m/s]; its folder is ``v<speed:.2f>``.

    Returns
    -------
        ``(path of the seed's pickle, speed text)``, read from ``summary_a.csv`` of the folder.
    """
    import csv

    folder = PROJECTS / "3d_simulations" / "results" / kind / f"v{speed:.2f}"
    rows = [r for r in csv.DictReader(open(folder / "summary_a.csv")) if r["converged"] == "1"
            and (folder / f"seed{r['seed']}.pkl").exists()]
    best = min(rows, key=lambda r: float(r["objective"]))
    return folder / f"seed{best['seed']}.pkl", f"{float(best['speed']):g} m/s"


RESULT = {
    "effort": {
        "sipp": best_sipp("effort", 1.3),
        "ms_human_700": (PROJECTS / "large_models/results/ms_human_700/walking_v1.30_seed1.pkl", "1.3 m/s"),
        "myofullbody": (PROJECTS / "large_models/results/myofullbody/walking_v1.30_seed1.pkl", "1.3 m/s"),
    },
    "metabolic": {
        "sipp": best_sipp("walking", 1.3),
        "ms_human_700": (PROJECTS / "large_models/results/ms_human_700/walking_metabolic_v1.30_seed1.pkl", "1.3 m/s"),
        "myofullbody": (PROJECTS / "large_models/results/myofullbody/walking_metabolic_v1.30_seed1.pkl", "1.3 m/s"),
    },
}
FOOTER = {"effort": "All simulations use BiomechPriorVAE + effort", "metabolic": "All simulations use BiomechPriorVAE + metabolic cost"}


def _fonts():
    import matplotlib
    from PIL import ImageFont

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return (ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 38),
            ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 24))


def wrap(text, font, width):
    """Break ``text`` into lines at spaces so that none is wider than ``width`` px (a single word is never split)."""
    words, lines = text.split(), []
    for word in words:
        if lines and font.getlength(f"{lines[-1]} {word}") <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def header_lines(detail, width):
    """The detail text of a header as lines that fit ``width`` (with a margin of 16 px on both sides)."""
    return wrap(detail, _fonts()[1], width - 32)


def header_image(path, title, lines, width=SIZE[0], height=HEADER_HEIGHT):
    """Write the header strip of one clip: dark background, bold title, the detail lines (DoF first)."""
    from PIL import Image, ImageDraw

    bold, regular = _fonts()
    img = Image.new("RGB", (width, height), (17, 22, 29))
    draw = ImageDraw.Draw(img)
    draw.text((width // 2, 36), title, font=bold, fill=(240, 240, 240), anchor="mm")
    for k, line in enumerate(lines):
        draw.text((width // 2, 78 + 30 * k), line, font=regular, fill=(185, 195, 210), anchor="mm")
    img.save(path)


def footer_image(path, text, width):
    """Write the footer strip under all clips."""
    from PIL import Image, ImageDraw

    _, regular = _fonts()
    img = Image.new("RGB", (width, FOOTER_HEIGHT), (17, 22, 29))
    ImageDraw.Draw(img).text((width // 2, FOOTER_HEIGHT // 2), text, font=regular, fill=(185, 195, 210), anchor="mm")
    img.save(path)


def main():
    objective = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in RESULT else None
    if objective is None:
        raise SystemExit("usage: render_compare.py effort|metabolic [--force]")
    force = "--force" in sys.argv
    camera = sys.argv[sys.argv.index("--camera") + 1] if "--camera" in sys.argv else "front"
    if camera not in CAMERAS:
        raise SystemExit(f"--camera must be one of {list(CAMERAS)}")
    unlabeled = "--unlabeled" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    pair = sys.argv[sys.argv.index("--pair") + 1].split(",") if "--pair" in sys.argv else None
    if pair and (len(pair) != 2 or not set(pair) <= set(MODELS)):
        raise SystemExit(f"--pair takes two models from {list(MODELS)}")
    if only and not set(only) <= set(MODELS):
        raise SystemExit(f"--only takes models from {list(MODELS)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    clips = []
    for key, (title, muscles, yaml) in MODELS.items():
        if (only and key not in only) or (pair and key not in pair):
            continue
        result, speed = RESULT[objective][key]
        path = RESULTS / f"{camera}_{key}_{objective}.mp4"
        source = path.with_suffix(".source")
        if force or not path.exists() or path.stat().st_mtime < Path(result).stat().st_mtime \
                or not source.exists() or source.read_text() != str(result):
            from biosym.model.model import BiosymModel
            from biosym.results import render

            model = BiosymModel(str(yaml), compile_eom=False, fit_muscle_paths=False)
            motion = render.prepare_motion(model, str(result), cycles=4, fps=30, slowdown=1.0)
            render.render_motion(motion, str(path), size=SIZE, camera=CAMERAS[camera], focal_length=45.0, muscles=True,
                                 muscle_radius=0.004, activation_max=ACTIVATION_MAX, contact_spheres=False,
                                 force_vectors=False)
            source.write_text(str(result))
            print("rendered", path, "from", result, flush=True)
        clips.append((path, title, muscles, speed))

    if only:
        return 0

    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    stem = f"{camera}_{pair[0]}_vs_{pair[1]}_{objective}" if pair else f"{camera}_comparison_{objective}"
    out = RESULTS / (stem + ("_unlabeled" if unlabeled else "") + ".mp4")
    cx, cy, cw, ch = CROP[camera] if pair else (0, 0, SIZE[0], SIZE[1])
    if unlabeled:
        cmd = [exe, "-y"]
        for path, *_ in clips:
            cmd += ["-stream_loop", "-1", "-t", str(SECONDS), "-i", str(path)]
        crops = "".join(f"[{i}:v]crop={cw}:{ch}:{cx}:{cy}[v{i}];" for i in range(len(clips)))
        stack = "".join(f"[v{i}]" for i in range(len(clips)))
        cmd += ["-filter_complex", f"{crops}{stack}hstack=inputs={len(clips)}", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        print("COMPARISON", out, flush=True)
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [exe, "-y"]
        details = [header_lines(f"{muscles}, {speed}", cw) for _, _, muscles, speed in clips]
        height = 48 + 30 * max(len(d) for d in details) + 30      # title, then the lines of the longest detail
        for i, (path, title, muscles, speed) in enumerate(clips):
            header = Path(tmp) / f"header{i}.png"
            header_image(header, title, details[i], cw, height)
            cmd += ["-loop", "1", "-framerate", "30", "-t", str(SECONDS), "-i", str(header),
                    "-stream_loop", "-1", "-t", str(SECONDS), "-i", str(path)]
        footer = Path(tmp) / "footer.png"
        footer_image(footer, FOOTER[objective], cw * len(clips))
        cmd += ["-loop", "1", "-framerate", "30", "-t", str(SECONDS), "-i", str(footer)]
        tiles = "".join(f"[{2 * i + 1}:v]crop={cw}:{ch}:{cx}:{cy}[v{i}];[{2 * i}:v][v{i}]vstack=inputs=2[c{i}];"
                        for i in range(len(clips)))
        stack = "".join(f"[c{i}]" for i in range(len(clips)))
        cmd += ["-filter_complex", f"{tiles}{stack}hstack=inputs={len(clips)}[row];[row][{2 * len(clips)}:v]vstack=inputs=2",
                "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
    print("COMPARISON", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
