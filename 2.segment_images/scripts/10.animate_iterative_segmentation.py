#!/usr/bin/env python
# coding: utf-8

# # Iterative segmentation visualization
#
# This notebook renders a napari animation that visualizes the segmentation approach used in this module as an **iterative, cumulative process**.
#
# Rather than loading pre-computed, *final* masks from disk (see `7.animate_segmentation_and_raw_signal.ipynb`), this notebook **re-runs the segmentation pipeline itself**, one step at a time, and keeps every intermediate array around so it can be loaded into napari as its own layer. The pipeline functions in `image_analysis_3D.segmentation_utils.*` are not changed - several of them (e.g. the nuclei/organoid mask-generation one-liner in `4.nuclei_segmentation.py` / `6.organoid_segmentation.py`, and `perform_morphology_dependent_segmentation` in `5.cell_segmentation.py`) chain many steps together into a single expression or function call that only returns the final array. Here those chains are "decoupled" back into their component calls, entirely within this notebook, so that every intermediate (pre-decoupled masks, pre-stitched masks, raw watershed output, etc.) is captured in its own variable.
#
# Two things are different from a simple "layer accumulates forever" build-up:
#
# - **Camera**: zoomed in once, to a fixed level, at the very start - never animated and never
#   re-applied per compartment (no more per-compartment zoom-in ramp). The *base* viewing angle is
#   only slightly above and at a slight angle (not a strong three-quarter oblique), with a modest
#   non-zero `camera.perspective` field of view. From there, every compartment's clip (nuclei,
#   cell, cytoplasm, organoid alike) spends its own length doing one full 360-degree rotation about
#   a single axis, then a full 360-degree rotation about a second axis - one at a time, never
#   simultaneously (`RotatingCamera`, see the "Camera" section below) - always landing back on that
#   same base angle by its last frame so the cut into the next compartment is seamless.
# - **Flythrough order, per compartment: black -> signal -> segmentation 1 -> segmentation 1 peel ->
#   segmentation 2 peel on -> segmentation 2 peel -> ... -> final mask.** Every compartment clip
#   starts from nothing visible (**black**). The first sub-stage is always that compartment's raw
#   channel - it simply fades in (**signal**), a plain opacity cross-fade, no clipping-plane sweep.
#   Every sub-stage after that is a segmentation mask, and each one gets the same two-beat
#   treatment, modeled on `layer_planes.py` in the
#   [napari-animation gallery](https://napari.org/napari-animation/) (which sweeps
#   `layer.plane.position` across keyframes to slice through a volume): the mask first **stacks up**
#   into view slice by slice (**"segmentation N peel on"** - sweeping an
#   `experimental_clipping_planes` plane through it along Z) and holds fully visible, then **peels
#   away** (sweeps back, hiding it again), holds hidden, then **adds back** (sweeps across once
#   more) and holds visible again (**"segmentation N peel"** covers this peel-away/add-back cycle).
#   The previous sub-stage's now-superseded intermediate is simply hidden (`visible = False`) at the
#   same time the next one starts its reveal, so each compartment's mask is visibly replaced by its
#   improved successor - segmentation 1 (raw window segmentation) -> segmentation 2 (decoupled
#   masks) -> final mask, for example. All of a compartment's sub-stages are chained onto one
#   `napari_animation.Animation`, so nuclei (for instance) renders as a **single clip**, not several
#   separate files. When moving to the next compartment, the *previous* compartment's final mask
#   smoothly **fades out** (via layer opacity, which napari-animation can actually ease - see
#   `animate_fade_out`) before the next compartment's signal starts fading in - the cytoplasm clip is
#   the one exception, and uses opacity crossfades throughout instead (see below).
#
# Compartments and their sub-stages, rendered and concatenated in this build-up order - **nuclei ->
# cell -> cytoplasm -> organoid** (not pipeline-computation order; every array below is already
# computed earlier in the notebook regardless of the order it's animated in here):
#
# 1. **Nuclei** (`01_nuclei.mp4`) - nuclei signal (DNA, 405 nm) with CLAHE contrast enhancement
#    fades in (never stripped - it stays visible underneath every mask for the rest of this clip)
#    -> raw 2.5D sliding-window CellposeSAM segmentation, before decoupling, peels on/peels
#    (stripped) -> decoupled per-slice masks, before 3D stitching, peels on/peels (stripped) ->
#    final mask (3D-stitched, border-cleaned, relabeled; used as watershed seeds) peels on/peels.
#    The final mask *and* the still-visible 405 nm signal both fade out once the cell compartment
#    starts. Like every compartment, the camera spends this whole clip doing one full 360-degree
#    rotation about a single axis, then the other - see the "Camera" section below.
# 2. **Cell** (`02_cell.mp4`) - the 405 nm nuclei signal (still visible from the nuclei clip) fades
#    away, replaced by the raw cell-boundary channel (AGP, 555 nm) fading in -> final mask
#    (hole-filled, size-filtered, post-hoc refined) peels on/peels, **once** - the
#    morphology-dependent elevation/threshold intermediate and the raw watershed are not shown as
#    their own steps. Neither the 555 nm signal nor the cell mask is stripped within this clip -
#    both are still fully visible at the end, going into the cytoplasm clip (which handles the
#    cell-mask-to-cytoplasm-mask build-up itself - see below - not this clip).
# 3. **Cytoplasm** (`03_cytoplasm.mp4`) - a sequence of opacity crossfades, not a peel: the 555 nm
#    channel (already visible, carried over from the cell clip - not re-faded from black, see the
#    note in `render_cytoplasm` below) stays up underneath the cell mask (also still visible from
#    the cell clip) -> the cell mask turns transparent-ish while the nuclei mask fades in underneath
#    it (translucent, visible "inside" the cell) -> the cytoplasm mask layers in at that same
#    opacity as the cell mask -> the cell mask drops out -> the nuclei mask drops out -> the
#    cytoplasm mask returns to full opacity as the final view (cell mask minus nuclei mask). The
#    555 nm channel is never stripped in this clip either. Fades out once the organoid compartment
#    starts.
# 4. **Organoid** (`04_organoid.mp4`) - the 555 nm channel (already visible, carried over from the
#    cytoplasm clip - not re-faded from black, see the note in `render_compartment` below; the
#    preprocessed Otsu+Gaussian thresholded signal and the decoupled per-slice intermediate are not
#    shown as their own steps) stays on screen, never stripped -> final mask (3D-stitched,
#    border-cleaned; bounds where the cell watershed is allowed to grow) stacks up into view **once
#    and stays** - unlike every other mask reveal in this notebook, it does not then peel away and
#    add back (`peel_once` on this compartment, see `animate_peel_step`'s `single_reveal`). This
#    completes the pipeline.
#
# The four compartment clips are also concatenated, in this same order, into a single "build-up" animation showing the full iterative approach end to end.
#
# Every one of those arrays is also cached to its own `.tiff` under `intermediates_dir` (see
# "Intermediate-array caching" below), keyed by variable name - so rerunning this notebook (e.g. to
# tweak the animation/camera code) loads the already-computed intermediates from disk instead of
# reprocessing them, including the two slow, GPU-bound CellposeSAM calls and the 3D watershed. Set
# `force_recompute = True` in the parameters cell to ignore the cache.

# ## Imports

# In[1]:


import os
import pathlib

import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
import tifffile
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.file_reading import (
    find_files_available,
    read_in_channels,
)
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from image_analysis_3D.file_utils.read_in_channel_mapping import (
    retrieve_channel_mapping,
)
from image_analysis_3D.segmentation_utils.general_segmentation_utils import (
    clean_border_objects,
    create_cytoplasm_masks,
    decouple_masks,
    get_labels_for_post_hoc_reassignment,
    reverse_sliding_window_max_projection,
    run_post_hoc_refinement,
    sliding_window_two_point_five_D,
)
from image_analysis_3D.segmentation_utils.nuclei_segmentation import (
    object_stitching_and_relation,
    segmentaion_on_two_D,
)
from moviepy import VideoFileClip, concatenate_videoclips
from napari_animation import Animation
from napari_animation.easing import Easing
from skimage.filters import sobel
from skimage.segmentation import relabel_sequential

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# ## Parameters

# In[2]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    clip_limit = args["clip_limit"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        input_subparent_name=input_subparent_name,
        clip_limit=clip_limit,
    )
    nuclei_clip_limit = clip_limit
    cyto_clip_limit = clip_limit
    amimation_subparent_name = "animations"
    intermediate_subparent_name = "iterative_segmentation_intermediates"

else:
    print("Running in a notebook")
    well_fov = "C4-2"
    patient = "NF0014_T1"
    input_subparent_name = "zstack_images"
    amimation_subparent_name = "animations"
    intermediate_subparent_name = "iterative_segmentation_intermediates"
    # same clip limits used by 4.nuclei_segmentation.py and 6.organoid_segmentation.py / 5.cell_segmentation.py
    nuclei_clip_limit = 0.01
    cyto_clip_limit = 0.03

# set this to True to force every step to be recomputed, ignoring any cached intermediates on disk
force_recompute = False

image_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}/"
).resolve(strict=True)

mp4_file_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/{amimation_subparent_name}/mp4/{well_fov}/iterative_segmentation/"
).resolve()
gif_file_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/{amimation_subparent_name}/gif/{well_fov}/iterative_segmentation/"
).resolve()
# every array computed below is cached here as its own .tiff, keyed by variable name, so a rerun
# of this notebook loads the already-computed intermediates instead of reprocessing them
intermediates_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{intermediate_subparent_name}/{well_fov}/"
).resolve()

mp4_file_dir.mkdir(parents=True, exist_ok=True)
gif_file_dir.mkdir(parents=True, exist_ok=True)
intermediates_dir.mkdir(parents=True, exist_ok=True)


# ## Helper functions

# In[3]:


def mp4_to_gif(input_mp4: pathlib.Path, output_gif: pathlib.Path, fps: int = 30):
    """
    Convert an mp4 file to a gif file using moviepy.
    Parameters
    ----------
    input_mp4 : pathlib.Path
        The path to the input mp4 file.
    output_gif : pathlib.Path
        The path to the output gif file.
    fps : int, optional
        The frames per second for the output gif file, by default 30.

    Returns
    -------
    None
    """
    with VideoFileClip(str(input_mp4)) as clip:
        width, height = clip.size
        side = min(width, height)

        # keep codec-friendly dimensions
        if side % 2 != 0:
            side -= 1

        x1 = int((width - side) // 2)
        y1 = int((height - side) // 2)

        square_clip = clip.cropped(
            x1=x1,
            y1=y1,
            x2=x1 + side,
            y2=y1 + side,
        )

        # Write only GIF (do not overwrite source mp4)
        square_clip.write_gif(
            str(output_gif),
            fps=fps,
            loop=0,
        )


# In[4]:


def animate_peel_step(
    animation: Animation,
    viewer: napari.Viewer,
    layer_names: list,
    axis_index: int = 0,
    n_positions: int = 24,
    hold_steps: int = 14,
    single_reveal: bool = False,
):
    """
    Append a "stack up -> peel away -> add back" reveal cycle for one or more 3D
    layers onto an *existing* `Animation`, with the camera held fixed throughout
    (see the camera setup in "Build the viewer" below for why it no longer rotates).

    With `single_reveal=True`, the mask stacks up into view once, holds, and stays -
    the peel-away/add-back half of the cycle is skipped entirely (used for the
    organoid compartment, which should not visibly hide its own final mask again
    after revealing it).

    This only captures keyframes - it does not create the `Animation` or export it,
    so several calls (one per pipeline sub-stage) can be chained onto the same
    `Animation` to build up one continuous clip per compartment, instead of a
    separate clip per sub-stage.

    The reveal is done by sweeping an `experimental_clipping_planes` plane back and
    forth along `axis_index` (0 = Z by default, since every array in this notebook is
    (Z, Y, X) and all three dims are displayed in 3D) - the plane position is swept
    between two values placed well outside the data's bounding box on either side, so
    one extreme always clips the whole volume away and the other always leaves it
    fully visible, regardless of which way the plane's `normal` happens to face.

    `experimental_clipping_planes` is a *list* attribute. napari-animation's built-in
    state interpolator (`interpolate_viewer_state`) only eases smoothly through plain
    numeric/tuple attributes nested in dicts (e.g. `layer.plane.position`, as in the
    napari-animation gallery's `layer_planes.py`) - it does not recurse into list
    elements, so it cannot ease between two different clipping-plane list values; it
    would just snap. The sweep below works around that by keyframing it by hand, one
    small position increment per frame (`capture_keyframe(steps=1)`), which is what
    actually produces the motion.

    A related quirk: a layer's `experimental_clipping_planes` starts out as an empty
    list (`[]`, no clip at all), and interpolating between an empty list and a
    one-element list zips to nothing - every frame of that *first* transition renders
    as if there were no clip, regardless of the clip position intended. So the very
    first clip position is captured twice: once with `steps=1` to snap the layer from
    "no clip" to "one clip plane" (matched list lengths from then on), then again with
    `steps=hold_steps` to actually hold there - without the first snap, "hold fully
    hidden" below would render as fully *visible* for its whole duration instead.

    Parameters
    ----------
    animation : napari_animation.Animation
        The in-progress animation to append keyframes to.
    viewer : napari.Viewer
        The napari viewer to animate.
    layer_names : list of str
        The layer(s) to peel - swept together, in sync.
    axis_index : int, optional
        Which of the 3 displayed array axes to sweep the clipping plane along, by
        default 0 (Z, so the mask appears to stack up/peel away slice by slice).
    n_positions : int, optional
        Number of discrete positions captured for each sweep direction, by default 15.
    hold_steps : int, optional
        Number of frames to hold at each fully-revealed/fully-hidden extreme, by
        default 8.
    single_reveal : bool, optional
        If True, stack up into view once and stop there (clipping cleared, mask left
        fully visible) instead of also peeling away and adding back, by default False.
    Returns
    -------
    None
    """
    layers = [viewer.layers[name] for name in layer_names]
    axis_size = max(layer.data.shape[axis_index] for layer in layers)

    # a point is visible where dot(point - position, normal) <= 0, i.e. (with this
    # normal) where its coordinate along `axis_index` is <= `position` - so a very
    # negative position clips everything away, and a very positive one leaves
    # everything visible, and sweeping position upward reveals the volume bottom-up
    normal = [0.0, 0.0, 0.0]
    normal[axis_index] = -1.0
    normal = tuple(normal)

    # placed well outside [0, axis_size) on each side, so one extreme always clips
    # everything away and the other always leaves everything visible
    position_hidden = -1.5 * axis_size
    position_visible = 2.5 * axis_size

    def plane_position(value):
        position = [0.0, 0.0, 0.0]
        position[axis_index] = value
        return tuple(position)

    def set_clip_position(value):
        for layer in layers:
            layer.experimental_clipping_planes = [
                {"normal": normal, "position": plane_position(value), "enabled": True}
            ]

    def sweep(start, end):
        for step_index in range(1, n_positions + 1):
            value = start + (end - start) * step_index / n_positions
            set_clip_position(value)
            animation.capture_keyframe(steps=1)

    set_clip_position(position_hidden)
    animation.capture_keyframe(
        steps=1
    )  # snap from "no clip" to "one clip plane" - see docstring
    animation.capture_keyframe(steps=hold_steps)  # hold fully hidden

    sweep(position_hidden, position_visible)  # "stacking": build the mask up
    animation.capture_keyframe(steps=hold_steps)  # hold fully revealed

    if not single_reveal:
        sweep(position_visible, position_hidden)  # peel it away
        animation.capture_keyframe(steps=hold_steps)  # hold fully hidden

        sweep(position_hidden, position_visible)  # add it back
        animation.capture_keyframe(steps=hold_steps)  # hold fully revealed

    for layer in layers:
        layer.experimental_clipping_planes = []
    animation.capture_keyframe(
        steps=1
    )  # clear clipping so the layer renders normally from here on


def animate_fade_out(
    animation: Animation,
    viewer: napari.Viewer,
    layer_names: list,
    steps: int = 12,
):
    """
    Append a smooth opacity fade-out for one or more layers onto an *existing*
    `Animation`, then hide them outright.

    `opacity` is a plain float, which napari-animation's built-in state
    interpolator eases smoothly between two captured keyframes (unlike `visible`,
    which is boolean and always snaps, or `experimental_clipping_planes`, which is a
    list - see `animate_peel_step`). The current, still fully-opaque state is
    captured explicitly first so the opacity change below has something to
    interpolate *from*; without that, the fade would have no effect; napari-animation
    only interpolates between two keyframes it has actually captured, and simply
    changing a property and then capturing a single keyframe records that as the
    starting state, not a transition.

    Parameters
    ----------
    animation : napari_animation.Animation
        The in-progress animation to append keyframes to.
    viewer : napari.Viewer
        The napari viewer to animate.
    layer_names : list of str
        The layer(s) to fade out together.
    steps : int, optional
        Number of frames over which to fade, by default 12.
    Returns
    -------
    None
    """
    animation.capture_keyframe(
        steps=1
    )  # capture the current, still fully-opaque state to fade *from*

    for layer_name in layer_names:
        viewer.layers[layer_name].opacity = 0.0
    animation.capture_keyframe(steps=steps)

    for layer_name in layer_names:
        viewer.layers[layer_name].visible = False
        viewer.layers[
            layer_name
        ].opacity = 1.0  # reset, in case the layer is ever shown again
    animation.capture_keyframe(steps=1)


def animate_opacity_transition(
    animation: Animation,
    viewer: napari.Viewer,
    target_opacities: dict,
    steps: int = 16,
):
    """
    Append a smooth opacity crossfade for one or more layers onto an *existing* `Animation`,
    toward whatever opacity each one should end up at - a lower-level, more general version of
    `animate_fade_out` (which only ever fades something out to 0 and hides it). Any targeted
    layer that isn't already visible is made visible first, so a layer can fade *in* from nothing
    just as easily as an already-visible one can cross-fade to a new opacity.

    `target_opacities` maps layer name -> target opacity in [0, 1]. As with `animate_fade_out`,
    the current opacity is captured first so there is something to ease *from* - `opacity` is a
    plain float, so napari-animation eases it smoothly between two captured keyframes.

    Parameters
    ----------
    animation : napari_animation.Animation
        The in-progress animation to append keyframes to.
    viewer : napari.Viewer
        The napari viewer to animate.
    target_opacities : dict
        Maps layer name -> target opacity in [0, 1].
    steps : int, optional
        Number of frames over which to ease, by default 16.
    Returns
    -------
    None
    """
    for layer_name in target_opacities:
        viewer.layers[layer_name].visible = True
    animation.capture_keyframe(steps=1)  # capture the current opacity to ease *from*

    for layer_name, target_opacity in target_opacities.items():
        viewer.layers[layer_name].opacity = target_opacity
    animation.capture_keyframe(steps=steps)


class RotatingCamera:
    """
    Proxy around a `napari_animation.Animation` that eases the camera through an explicit,
    ordered list of `camera.angles` keyframes - instead of a fixed axis/degrees formula -
    advancing a little further along that path each time `capture_keyframe` is called, instead
    of a camera that only moves once per sub-stage (or not at all). Every other `animate_*`
    helper above only ever calls `.capture_keyframe(...)` on the `animation` object it is given,
    so passing one of these in its place - duck-typed, no changes needed to those helpers - makes
    the camera ease continuously along `keyframe_angles` underneath whatever
    peel/fade/crossfade is actually happening, for the whole length of a compartment's clip.

    The keyframes are spaced evenly across `total_frames`: with N keyframes there are N - 1
    equal-length segments, and at any point in the clip the camera's current angles are a plain
    componentwise linear interpolation between the two keyframes bracketing that point.
    `camera_rotation_keyframes` below (defined once, right after `shared_camera_angles`, in the
    "Camera" section) is this notebook's one rotation path - a literal list of `camera.angles`
    waypoints, not a "start angle + N degrees" formula - so every orientation the camera actually
    passes through is visible and checkable right there, rather than re-derived from an axis
    index and a degrees figure on every call. Nothing stops a caller from handing this class any
    other hand-placed waypoint list instead - e.g. to steer around a known degenerate orientation
    on some other base angle - as long as it starts and ends on the orientation the clip needs to
    land back on.

    An earlier version of this camera design rotated through two sequential formula-driven
    phases - axis 0, then axis 1 - and that second phase is what made the rotation look "off":
    `camera.angles` are Euler angles, and sweeping axis 1 through 360 degrees at this notebook's
    base angle carries the view direction through a near-degenerate orientation (confirmed
    numerically via `napari.utils.camera_orientations.view_and_up_directions_from_angles` - at
    the midpoint of that phase the view direction collapses to almost exactly one data axis, e.g.
    `(0, -0.09, -1.0)`), which is the "stares straight down a single displayed axis" case called
    out in the "Camera" markdown section below. Combined with this notebook's anisotropic `scale`
    (Z stretched 10x relative to X/Y - see "Build the viewer" below), that near-degenerate
    orientation briefly renders the volume as a thin edge-on sliver before continuing to rotate -
    a visible glitch partway through every compartment's clip. `camera_rotation_keyframes` now
    carries `sequence_of_angles`: four hand-captured `camera.angles` orientations (defined in the
    "Camera" section below), each one checkable/reproducible by eye, rather than a degrees-based
    formula whose intermediate orientations have to be reasoned about indirectly.

    `camera.angles` is a plain tuple attribute, so (unlike `experimental_clipping_planes` - see
    `animate_peel_step`) napari-animation's built-in interpolator eases it smoothly between
    the sparse keyframes this proxy captures - there is no need to manually step every frame by
    hand the way the clipping-plane sweep does.

    `total_frames` must be this clip's exact total frame count (see `estimate_compartment_frames`
    below) so the keyframe path is paced evenly across the whole clip, landing on its last
    keyframe exactly on the clip's last frame - so every compartment's clip starts and ends on
    the same orientation (when the first and last keyframes match, as every call site below
    arranges) and the cut between compartments is seamless.

    Parameters
    ----------
    animation : napari_animation.Animation
        The real animation to forward every captured keyframe to.
    viewer : napari.Viewer
        The napari viewer to animate.
    total_frames : int
        This clip's total frame count, used to pace the keyframe path evenly across it.
    keyframe_angles : list of tuple of float
        The ordered `camera.angles` waypoints to ease through, spaced evenly across
        `total_frames`. Needs at least 2 entries (a 2-entry list - start and end - reproduces the
        old single-phase axis/degrees sweep exactly, since linear interpolation between two
        points *is* that formula). Pass the same tuple as the first and last entry so the clip
        starts and ends on the same orientation and the cut into the next compartment is
        seamless.
    """

    def __init__(
        self,
        animation: Animation,
        viewer: napari.Viewer,
        total_frames: int,
        keyframe_angles: list,
    ):
        if len(keyframe_angles) < 2:
            raise ValueError(
                "keyframe_angles needs at least 2 entries (a start and an end)."
            )
        self._animation = animation
        self._viewer = viewer
        self._keyframe_angles = [tuple(angles) for angles in keyframe_angles]
        self._total_frames = max(total_frames, 1)
        self._num_segments = len(self._keyframe_angles) - 1
        self._segment_length = self._total_frames / self._num_segments
        self._frame_count = 0

    def capture_keyframe(self, steps: int = 30):
        self._frame_count += steps
        segment_index = min(
            int(self._frame_count / self._segment_length), self._num_segments - 1
        )
        frames_into_segment = self._frame_count - segment_index * self._segment_length
        fraction = min(frames_into_segment / self._segment_length, 1.0)

        start_angles = self._keyframe_angles[segment_index]
        end_angles = self._keyframe_angles[segment_index + 1]
        angles = tuple(
            start + (end - start) * fraction
            for start, end in zip(start_angles, end_angles)
        )
        self._viewer.scene.camera.angles = angles
        self._animation.capture_keyframe(steps=steps)


# ### Intermediate-array caching
#
# Every array computed below (CLAHE signals, raw/decoupled/final masks, etc.) is cached to its own
# `.tiff` file in `intermediates_dir`, keyed by variable name. On a rerun, each step loads its cached
# array instead of reprocessing it - this matters most for the two CellposeSAM calls and the 3D
# watershed, which are the slow, GPU-bound steps. Set `force_recompute = True` in the parameters
# cell above to ignore the cache and reprocess everything from scratch (e.g. after changing
# `well_fov`/`patient`/clip limits, or after editing a pipeline step).

# In[5]:


def cached_array_path(name: str) -> pathlib.Path:
    """Path on disk for a cached intermediate array, keyed by variable name."""
    return intermediates_dir / f"{name}.tiff"


def load_cached_array(name: str) -> np.ndarray | None:
    """Load a cached intermediate array if present (and `force_recompute` is False), else None."""
    path = cached_array_path(name)
    if force_recompute or not path.exists():
        return None
    return tifffile.imread(path)


def save_cached_array(name: str, array: np.ndarray) -> np.ndarray:
    """Cache an intermediate array to disk under its variable name, and return it unchanged."""
    tifffile.imwrite(cached_array_path(name), array)
    return array


# `image_analysis_3D.segmentation_utils.cell_segmentation` (the module `5.cell_segmentation.py`
# uses for `segment_cells_with_3D_watershed` and `fill_holes_in_mask`) imports `cupy`/`cupyx` at
# module load time for GPU-accelerated hole filling. `cupy` is not part of `segmentation_env.yml`,
# so importing anything from that module fails with `ModuleNotFoundError: No module named 'cupy'`
# in the normal segmentation environment. To keep this notebook runnable there - and without
# touching `utils` - the two functions are reproduced below: the watershed call is copied verbatim
# (it never used `cupy`), and the hole-filling step is reimplemented with `scipy.ndimage` instead of
# `cupyx.scipy.ndimage` (same algorithm, CPU instead of GPU).

# In[6]:


def segment_cells_with_3D_watershed_cpu(
    cyto_signal: np.ndarray,
    nuclei_mask: np.ndarray,
    thresholded_signal: np.ndarray,
    connectivity: int = 1,
    compactness: float = 0,
) -> np.ndarray:
    """
    CPU-only copy of `cell_segmentation.segment_cells_with_3D_watershed` (unchanged logic,
    no cupy dependency - this function never used cupy in the first place).
    """
    labels = skimage.segmentation.watershed(
        image=cyto_signal,
        markers=nuclei_mask,
        connectivity=connectivity,
        compactness=compactness,
        mask=thresholded_signal,
    )

    # change the largest label (by area) to 0
    # cleans up the output and sets the background properly
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    labels[labels == largest_label] = 0
    return labels


def fill_holes_in_cell_mask_cpu(mask: np.ndarray) -> np.ndarray:
    """
    CPU-only reimplementation of `cell_segmentation.fill_holes_in_mask(mask, compartment="cell")`
    using `scipy.ndimage` instead of `cupyx.scipy.ndimage`. Same hybrid 3D + per-slice 2D
    enclosed-hole fill, run per label.
    """
    structure_3d = scipy.ndimage.generate_binary_structure(rank=3, connectivity=1)
    new_mask = np.zeros_like(mask)
    for label in np.unique(mask):
        label = int(label)
        if label == 0:
            continue  # skip background
        tmp_mask = mask == label

        # 3D enclosed-hole fill
        tmp_mask = scipy.ndimage.binary_fill_holes(tmp_mask, structure=structure_3d)

        # 2D per-z enclosed-hole fill (captures holes open across z in 3D)
        for z in range(tmp_mask.shape[0]):
            tmp_mask[z] = scipy.ndimage.binary_fill_holes(tmp_mask[z])
        new_mask[tmp_mask] = label

    return new_mask.astype(mask.dtype)


# ## Load the raw imaging channels

# In[7]:


channel_map = {
    "405": "Nuclei",
    "488": "Endoplasmic Reticulum",
    "555": "Actin, Golgi, and plasma membrane (AGP)",
    "640": "Mitochondria",
    "TRANS": "Brightfield",
}
scaling_values = [1, 0.1, 0.1]
image_metadata = f"{patient}_{well_fov}"

channel_dict = retrieve_channel_mapping(f"{root_dir}/config/channel_mapping.toml")
# only DNA (405 nm) and AGP are actually read: DNA is both the nuclei pipeline's input signal and
# the only raw channel shown in the nuclei clip (per-channel request: nuclei clip = 405 only), AGP
# is the organoid/cell pipelines' input signal. ER (488) and Mitochondria (640) are not used by any
# compartment's clip, so they are not read at all.
raw_channel_dict = {
    "DNA": channel_dict["DNA"],
    "AGP": channel_dict["AGP"],
}


# In[8]:


image_return_dict = read_in_channels(
    find_files_available(image_dir),
    channel_dict=raw_channel_dict,
    channels_to_read=list(raw_channel_dict.keys()),
)

dna_raw = image_return_dict["DNA"]
agp_raw = image_return_dict["AGP"]


# ## Nuclei segmentation pipeline
#
# `4.nuclei_segmentation.py` builds the final nuclei mask with a single chained expression:
#
# ```python
# nuclei_masks = np.array(
#     list(
#         decouple_masks(
#             reverse_sliding_window_max_projection(
#                 segmentaion_on_two_D(
#                     sliding_window_two_point_five_D(image_stack=nuclei, window_size=3)
#                 ),
#                 window_size=3,
#                 original_z_slice_count=nuclei_image_shape[0],
#             ),
#             original_img_shape=nuclei_image_shape,
#             distance_threshold=10,
#         ).values()
#     )
# )
# ```
#
# Below, each of those calls is pulled apart and assigned to its own variable - the `image_analysis_3D.segmentation_utils` functions themselves are unchanged, only the order in which this notebook calls them is different - so every intermediate array can be loaded into napari as its own layer.

# In[9]:


nuclei_clahe = load_cached_array("nuclei_clahe")
if nuclei_clahe is None:
    nuclei_clahe = skimage.exposure.equalize_adapthist(
        dna_raw, clip_limit=nuclei_clip_limit, kernel_size=None
    )
    save_cached_array("nuclei_clahe", nuclei_clahe)
nuclei_image_shape = nuclei_clahe.shape


# In[10]:


# the two arrays below require the slow, GPU-bound CellposeSAM call (`segmentaion_on_two_D`) -
# both are cached, so this block is skipped entirely once both are on disk
nuclei_raw_window_labels = load_cached_array("nuclei_raw_window_labels")
nuclei_masks_filtered = load_cached_array("nuclei_masks_filtered")

if nuclei_raw_window_labels is None or nuclei_masks_filtered is None:
    # 1. sliding-window 2.5D max projection
    nuclei_2p5d_stack = sliding_window_two_point_five_D(
        image_stack=nuclei_clahe, window_size=3
    )

    # 2. 2D CellposeSAM segmentation on each 2.5D max-projected slice
    nuclei_2d_seg_output = segmentaion_on_two_D(nuclei_2p5d_stack)

    # 3. reverse the sliding window to re-expand the projected labels back to the original z-slice count
    nuclei_reconstruction_dict = reverse_sliding_window_max_projection(
        nuclei_2d_seg_output,
        window_size=3,
        original_z_slice_count=nuclei_image_shape[0],
    )

    # grab the raw, pre-decoupled candidate mask for each slice (first duplicated-window copy)
    # this is the intermediate that `decouple_masks` consumes and resolves
    nuclei_raw_window_labels = np.array(
        [
            (
                nuclei_reconstruction_dict[zslice][0]
                if len(nuclei_reconstruction_dict[zslice]) > 0
                else np.zeros(nuclei_image_shape[1:], dtype=np.uint16)
            )
            for zslice in sorted(nuclei_reconstruction_dict)
        ]
    )

    # 4. decouple overlapping sliding-window masks into per-slice reassembled masks
    nuclei_masks_decoupled_dict = decouple_masks(
        nuclei_reconstruction_dict,
        original_img_shape=nuclei_image_shape,
        distance_threshold=10,
    )
    nuclei_masks_decoupled = np.array(list(nuclei_masks_decoupled_dict.values()))

    # 5. remove small objects per slice while preserving label IDs (same threshold as 4.nuclei_segmentation.py)
    nuclei_masks_filtered = nuclei_masks_decoupled.copy()
    for zslice in range(nuclei_masks_filtered.shape[0]):
        props = skimage.measure.regionprops(nuclei_masks_filtered[zslice])
        for prop in props:
            if prop.area < 250:  # ~15x15 pixel region for nuclei
                nuclei_masks_filtered[zslice] = np.where(
                    nuclei_masks_filtered[zslice] == prop.label,
                    0,
                    nuclei_masks_filtered[zslice],
                )

    save_cached_array("nuclei_raw_window_labels", nuclei_raw_window_labels)
    save_cached_array("nuclei_masks_filtered", nuclei_masks_filtered)


# In[11]:


nuclei_mask_final = load_cached_array("nuclei_mask_final")

if nuclei_mask_final is None:
    # 6. stitch the per-slice masks into 3D objects across z
    nuclei_mask_stitched, nuclei_stitch_diagnostics = object_stitching_and_relation(
        input_masks=nuclei_masks_filtered,
        max_match_distance=100,
        max_trajectory_length=15,  # 15 slice length (15 = 15 um)
        verbose=False,
    )
    print(
        f"Nuclei stitching diagnostics: {nuclei_stitch_diagnostics['num_trajectories']} "
        "trajectories found"
    )

    # 7. remove objects touching the image border, then relabel sequentially
    nuclei_mask_border_cleaned = clean_border_objects(
        nuclei_mask_stitched, border_width=5
    )
    nuclei_mask_final, _, _ = relabel_sequential(nuclei_mask_border_cleaned)

    save_cached_array("nuclei_mask_final", nuclei_mask_final)


# ## Organoid segmentation pipeline
#
# Same decoupling applied to the chained expression in `6.organoid_segmentation.py`.

# In[12]:


cyto2_clahe = load_cached_array("cyto2_clahe")
if cyto2_clahe is None:
    cyto2_clahe = skimage.exposure.equalize_adapthist(
        agp_raw, clip_limit=cyto_clip_limit, kernel_size=None
    )
    save_cached_array("cyto2_clahe", cyto2_clahe)
organoid_image_shape = cyto2_clahe.shape


# In[13]:


# the two arrays below require the slow, GPU-bound CellposeSAM call (`segmentaion_on_two_D`) -
# both are cached, so this block is skipped entirely once both are on disk
organoid_thresholded_signal = load_cached_array("organoid_thresholded_signal")
organoid_mask_decoupled = load_cached_array("organoid_mask_decoupled")

if organoid_thresholded_signal is None or organoid_mask_decoupled is None:
    # preprocessing: Otsu-threshold the AGP signal, then Gaussian-smooth the binary mask
    organoid_threshold = skimage.filters.threshold_otsu(cyto2_clahe)
    organoid_thresholded_signal = cyto2_clahe > organoid_threshold
    organoid_thresholded_signal = skimage.filters.gaussian(
        organoid_thresholded_signal, sigma=10
    )

    # 1-4. sliding-window 2.5D -> 2D CellposeSAM -> reverse sliding window -> decouple
    organoid_2p5d_stack = sliding_window_two_point_five_D(
        image_stack=organoid_thresholded_signal, window_size=3
    )
    organoid_2d_seg_output = segmentaion_on_two_D(organoid_2p5d_stack, diameter=200)
    organoid_reconstruction_dict = reverse_sliding_window_max_projection(
        organoid_2d_seg_output,
        window_size=3,
        original_z_slice_count=organoid_image_shape[0],
    )
    organoid_masks_decoupled_dict = decouple_masks(
        organoid_reconstruction_dict,
        original_img_shape=organoid_image_shape,
        distance_threshold=10,
    )
    organoid_mask_decoupled = np.array(list(organoid_masks_decoupled_dict.values()))

    save_cached_array("organoid_thresholded_signal", organoid_thresholded_signal)
    save_cached_array("organoid_mask_decoupled", organoid_mask_decoupled)


# In[14]:


organoid_mask_final = load_cached_array("organoid_mask_final")

if organoid_mask_final is None:
    # 5. stitch into 3D organoid objects, then remove objects touching the image border
    organoid_mask_stitched, organoid_stitch_diagnostics = object_stitching_and_relation(
        input_masks=organoid_mask_decoupled,
        max_match_distance=1000,
        verbose=False,
    )
    print(
        f"Organoid stitching diagnostics: {organoid_stitch_diagnostics['num_trajectories']} "
        "trajectories found"
    )
    organoid_mask_final = clean_border_objects(organoid_mask_stitched, border_width=5)

    save_cached_array("organoid_mask_final", organoid_mask_final)


# ## Cell segmentation pipeline
#
# `5.cell_segmentation.py` calls `perform_morphology_dependent_segmentation`, which internally
# chains the elevation-map construction, the seeded watershed call, hole filling, and small/large
# object removal into one function that only returns the final cell mask. The morphology-dependent
# branches below are reproduced unchanged from that function so each intermediate array (the
# elevation/threshold signal, the raw watershed labels, the hole-filled + size-filtered mask) can be
# captured on its own.

# In[15]:


# morphology lookup (same as 5.cell_segmentation.py)
image_label_path = pathlib.Path(
    f"{root_dir}/2.segment_images/image_labels/segmentation_classes.parquet"
).resolve(strict=True)
organoid_image_labels_df = pd.read_parquet(image_label_path)
morphology_class = organoid_image_labels_df.loc[
    (
        (organoid_image_labels_df["well_fov"] == well_fov)
        & (organoid_image_labels_df["patient"] == patient)
    ),
    "label",
]
if not morphology_class.empty and len(morphology_class) == 1:
    morphology_class = morphology_class.iloc[0]
elif morphology_class.empty:
    raise ValueError(f"No morphology label found for {well_fov} in {patient}.")
else:
    # multiple labels found - pick one at random, matching 5.cell_segmentation.py
    morphology_class = morphology_class.sample(n=1).iloc[0]

print(f"Organoid morphology for {well_fov}: {morphology_class}")


# In[16]:


# the preprocessing signal and the raw watershed mask are cached independently, so this block is
# skipped entirely once both are on disk (the watershed itself is a slow, seeded 3D computation)
cell_elevation_threshold_signal = load_cached_array("cell_elevation_threshold_signal")
cell_mask_raw_watershed = load_cached_array("cell_mask_raw_watershed")

if cell_elevation_threshold_signal is None or cell_mask_raw_watershed is None:
    # low-frequency elevation map - shared by every morphology branch
    butterworth_filter = skimage.filters.butterworth(
        cyto2_clahe,
        cutoff_frequency_ratio=0.08,
        order=2,
        high_pass=False,
        squared_butterworth=False,
    )

    if morphology_class in {"globular", "cluster"}:
        cell_elevation_threshold_signal = skimage.filters.gaussian(
            cyto2_clahe, sigma=2.5
        )
        cell_threshold = skimage.filters.threshold_otsu(cell_elevation_threshold_signal)
        cell_elevation_threshold_signal[
            cell_elevation_threshold_signal < cell_threshold
        ] = 0
        cell_elevation_threshold_signal[cell_elevation_threshold_signal > 0] = 1

        cell_elevation_map = sobel(butterworth_filter)
        cell_connectivity, cell_compactness = 1, 1

    elif morphology_class in {"small", "dissociated"}:
        cell_elevation_threshold_signal = skimage.filters.gaussian(cyto2_clahe, sigma=3)
        cell_threshold = skimage.filters.threshold_otsu(cell_elevation_threshold_signal)
        cell_elevation_threshold_signal[
            cell_elevation_threshold_signal < cell_threshold
        ] = 0
        cell_elevation_threshold_signal[cell_elevation_threshold_signal > 0] = 1
        cell_elevation_threshold_signal = skimage.morphology.dilation(
            cell_elevation_threshold_signal, skimage.morphology.ball(1)
        )

        cell_elevation_map = skimage.filters.gaussian(butterworth_filter, sigma=1)
        cell_elevation_map = sobel(cell_elevation_map)
        cell_connectivity, cell_compactness = 1, 0

    elif morphology_class == "elongated":
        cell_elevation_threshold_signal = skimage.filters.gaussian(cyto2_clahe, sigma=4)
        cell_threshold = skimage.filters.threshold_otsu(cell_elevation_threshold_signal)
        cell_elevation_threshold_signal[
            cell_elevation_threshold_signal < cell_threshold
        ] = 0
        cell_elevation_threshold_signal[cell_elevation_threshold_signal > 0] = 1
        cell_elevation_threshold_signal = skimage.morphology.dilation(
            cell_elevation_threshold_signal, skimage.morphology.ball(10)
        )

        cell_elevation_map = skimage.filters.gaussian(butterworth_filter, sigma=1)
        cell_elevation_map = sobel(cell_elevation_map)
        cell_connectivity, cell_compactness = 0, 0

    else:
        raise ValueError(f"Unknown morphology label: {morphology_class}")

    # raw seeded 3D watershed, before hole filling or size filtering
    cell_mask_raw_watershed = segment_cells_with_3D_watershed_cpu(
        cyto_signal=cell_elevation_map,
        nuclei_mask=nuclei_mask_final,
        thresholded_signal=cell_elevation_threshold_signal,
        connectivity=cell_connectivity,
        compactness=cell_compactness,
    )

    save_cached_array(
        "cell_elevation_threshold_signal", cell_elevation_threshold_signal
    )
    save_cached_array("cell_mask_raw_watershed", cell_mask_raw_watershed)


# In[17]:


cell_mask_final = load_cached_array("cell_mask_final")
cytoplasm_mask_final = load_cached_array("cytoplasm_mask_final")

if cell_mask_final is None or cytoplasm_mask_final is None:
    cell_mask_filled = fill_holes_in_cell_mask_cpu(cell_mask_raw_watershed)

    # remove small/large objects while preserving label IDs (mirrors perform_morphology_dependent_segmentation)
    cell_min_size, cell_max_size = 100, 10_000_000
    cell_mask_sized = cell_mask_filled.copy()
    for prop in skimage.measure.regionprops(cell_mask_sized):
        if prop.area < cell_min_size:
            cell_mask_sized[cell_mask_sized == prop.label] = 0
    unique_cell_labels, cell_label_counts = np.unique(
        cell_mask_sized[cell_mask_sized > 0], return_counts=True
    )
    for label, count in zip(unique_cell_labels, cell_label_counts):
        if count > cell_max_size:
            cell_mask_sized[cell_mask_sized == label] = 0

    # post-hoc nuclei/cell label reassignment and refinement (requires both masks to exist)
    cell_df = get_labels_for_post_hoc_reassignment(
        compartment_mask=cell_mask_sized, compartment_name="cell"
    )
    nuclei_df = get_labels_for_post_hoc_reassignment(
        compartment_mask=nuclei_mask_final, compartment_name="nuclei"
    )

    cell_mask_refined = run_post_hoc_refinement(
        mask_image=cell_mask_sized,
        sliding_window_context=3,
    )

    cytoplasm_mask_raw = create_cytoplasm_masks(
        nuclei_masks=nuclei_mask_final,
        cell_masks=cell_mask_refined,
    )

    # remove border-touching objects from the final cell and cytoplasm masks
    # (nuclei were already border-cleaned when nuclei_mask_final was produced)
    cell_mask_final = clean_border_objects(cell_mask_refined, border_width=5)
    cytoplasm_mask_final = clean_border_objects(cytoplasm_mask_raw, border_width=5)

    save_cached_array("cell_mask_final", cell_mask_final)
    save_cached_array("cytoplasm_mask_final", cytoplasm_mask_final)


# ## Build the viewer with every layer added but hidden

# In[18]:


headless = False
viewer = napari.Viewer(ndisplay=3, show=bool(not headless))

image_layers = {
    "Raw_AGP": agp_raw,
    "Nuclei_preprocessed_CLAHE": nuclei_clahe,
    "Organoid_thresholded_signal": organoid_thresholded_signal,
    "Cell_elevation_threshold_signal": cell_elevation_threshold_signal,
}

label_layers = {
    "Nuclei_raw_window_segmentation": nuclei_raw_window_labels,
    "Nuclei_decoupled_filtered": nuclei_masks_filtered,
    "Nuclei_final": nuclei_mask_final,
    "Organoid_decoupled": organoid_mask_decoupled,
    "Organoid_final": organoid_mask_final,
    "Cell_raw_watershed": cell_mask_raw_watershed,
    "Cell_final": cell_mask_final,
    "Cytoplasm_final": cytoplasm_mask_final,
}

image_layer_names = {name: f"{image_metadata}_{name}" for name in image_layers}
label_layer_names = {name: f"{image_metadata}_{name}" for name in label_layers}

for name, array in image_layers.items():
    viewer.add_image(
        array,
        name=image_layer_names[name],
        scale=scaling_values,
        visible=False,
    )

for name, array in label_layers.items():
    viewer.add_labels(
        np.asarray(array).astype(np.int32),
        name=label_layer_names[name],
        scale=scaling_values,
        visible=False,
    )


# In[19]:


viewer.reset_view()
viewer.scene.camera.zoom *= 2


# In[20]:


# a gentler viewpoint than a full three-quarter oblique: slightly above (a modest positive
# elevation) and at a slight angle (a small azimuth/roll offset) rather than a strong tilt, so
# the organoid's 3D structure still reads clearly without the view feeling rotated/banked - see
# the markdown cell above for why the camera is fixed rather than animated
viewer.reset_view()
viewer.scene.camera.angles = (141.0, 1, 136.0)  # z
viewer.scene.camera.perspective = 25.0


# ### Camera
#
# Fit the camera to the data with `reset_view()`, then **zoom in once, to a fixed level - not
# animated, and not re-applied per compartment.** Every clip starts already at this zoom; there is
# no more per-compartment zoom-in ramp (that used to make nuclei visibly zoom in further every
# sub-stage). Set a base viewing angle that is only **slightly above and at a slight angle** - a
# gentle lift and a small rotation, not a strong three-quarter oblique - plus a modest non-zero
# `perspective` field of view. `angles` (captured right after setting it, as `shared_camera_angles`
# below) is this clip's *base* orientation and matches `angle1` in `sequence_of_angles` (just
# below this cell).
#
# Every compartment's camera eases through `sequence_of_angles` - `angle1` through `angle4`, four
# explicit, hand-captured `camera.angles` orientations, not a "rotate N degrees" formula
# (`camera_rotation_keyframes = sequence_of_angles`, set right after `shared_camera_angles` below,
# is the exact list `RotatingCamera` eases through). An earlier version of this camera instead
# rotated formulaically through two sequential phases - full 360-degree sweeps about Euler axis 0,
# then axis 1 - and the axis-1 phase is what made the rotation look "off": sweeping axis 1 through
# 360 degrees at this base angle carries the view direction through a near-degenerate orientation
# (confirmed by probing `napari.utils.camera_orientations.view_and_up_directions_from_angles`
# directly - at the midpoint of that phase the view direction collapses to almost exactly
# `(0, -0.09, -1.0)`, i.e. staring straight down a single displayed axis with an orthographic
# projection - there is no amount of rotation that fixes that, since every angle along that sweep
# is still a flat, axis-aligned view). Combined with the anisotropic `scale` below (Z stretched
# 10x relative to X/Y), that rendered as a visible thin-sliver glitch partway through every clip.
# `sequence_of_angles` replaces that formula with explicit, hand-picked waypoints instead.
#
# **Open issue, left for a deliberate decision rather than silently patched:** `angle1` matches
# `shared_camera_angles`, so every compartment's clip starts on the shared baseline - but `angle4`
# does not match it back, so a clip no longer necessarily *ends* there. That breaks the "lands
# back on the same orientation it started on, so the cut into the next compartment is seamless"
# property the old formula-driven rotation guaranteed by construction - there will likely be a
# visible snap in camera angle at each compartment-to-compartment cut unless `sequence_of_angles`
# is extended with a closing waypoint back to `angle1`.
#
# Every compartment (nuclei, cell, cytoplasm, organoid) gets the exact same `sequence_of_angles`
# treatment - there is no camera exception for any one of them. `camera.angles` is a plain tuple
# attribute, so - unlike `experimental_clipping_planes` (a list; see `animate_peel_step`) -
# napari-animation's built-in interpolator eases it smoothly between the keyframes `RotatingCamera`
# captures, with no manual per-frame stepping needed.

# In[21]:


# make the viewer full screen
viewer.window._qt_window.showMaximized()
# hide the layer list and layer controls docks so they do not appear in the recorded canvas
viewer.window._qt_viewer.dockLayerList.setVisible(False)
viewer.window._qt_viewer.dockLayerControls.setVisible(False)

# set the viewer to a fixed window size
viewer.window._qt_window.resize(1000, 1000)

# fit the camera to the data, then zoom in - fixed, not animated: every compartment starts
# already at this zoom level (no per-compartment zoom-in ramp anymore, see `render_compartment`)
viewer.reset_view()
viewer.scene.camera.zoom *= 1.3

# a gentler viewpoint than a full three-quarter oblique: slightly above (a modest positive
# elevation) and at a slight angle (a small azimuth/roll offset) rather than a strong tilt, so
# the organoid's 3D structure still reads clearly without the view feeling rotated/banked - see
# the markdown cell above for why this is the *base* angle a full rotation returns to, not a
# frozen camera
viewer.scene.camera.angles = (141.0, 1, 136.0)  # z
angles = viewer.scene.camera.angles
viewer.scene.camera.perspective = 25.0

# viewer.canvas.overlays.text.visible = True
# viewer.canvas.overlays.text.position = "top_left"
# viewer.canvas.overlays.text.font_size = 16
# viewer.canvas.overlays.text.color = "white"


# In[ ]:


angle1 = (141.0, 1, 136.0)
angle2 = (-89.68578096393948, -1.5296463416538926, 99.53458035769405)
angle3 = (-3.302025772920807, -0.8873915915902288, 104.81038350856444)
angle4 = (75.69232122772563, 6.285358051359158, 115.16414147727976)

sequence_of_angles = [
    angle1,
    angle2,
    angle3,
    angle4,
]


# ## Define the compartments, each with its own sub-stages
#
# Instead of one clip per pipeline sub-stage, this renders **one clip per compartment**, played in
# **this build-up order: nuclei -> cell -> cytoplasm -> organoid** (not pipeline-computation order -
# the cell/organoid/cytoplasm arrays are all already computed earlier in the notebook regardless of
# the order they're animated in here). Within a compartment's clip, each sub-stage still lists the
# layer(s) that get **added** (peeled in) at that point and the layer(s) that get **stripped**
# (hidden outright) at that same point - the layer(s) they improve on - chained onto one `Animation`
# instead of exported as separate files. Each compartment (after the first) also lists
# `fade_out_layers`: the *previous* compartment's final mask, which smoothly fades out via opacity
# before this compartment's first sub-stage starts peeling in - see `animate_fade_out` above.
#
# `compartments` below only holds nuclei, cell, and organoid - the three that share this
# peel-in/fade-out pattern via `render_compartment` below. **Cytoplasm is handled separately**, by
# `render_cytoplasm` below, since what it needs to show (cell mask -> reveal nuclei inside it ->
# cross-fade to the cytoplasm mask -> drop the cell/nuclei layers -> settle on the cytoplasm mask
# alone) doesn't fit the peel-in-one-new-layer pattern the other three compartments use - it's a
# sequence of opacity crossfades between layers that are already in the viewer, not a single new
# layer being revealed. The two are stitched into the right order by calling them in sequence
# (nuclei, cell, cytoplasm, organoid) right before building `compartment_mp4_paths`, below.

# In[22]:


compartments = [
    {
        "name": "01_nuclei",
        "fade_out_layers": [],
        # camera motion for every compartment is now a single continuous 360-degree rotation,
        # applied uniformly in `render_compartment` via `RotatingCamera` - there is no more
        # per-compartment camera config here. The Z-axis peel is slower than the plain default
        # (more discrete sweep positions, longer holds) - see `peel_kwargs`.
        "peel_kwargs": {"n_positions": 64, "hold_steps": 20},
        # only the first and last nuclei mask iterations are shown - the raw 2.5D sliding-window
        # CellposeSAM output (before decoupling) and the final, 3D-stitched/border-cleaned mask -
        # skipping the decoupled-but-unstitched intermediate (`Nuclei_decoupled_filtered`) that
        # used to be its own sub-stage in between. That intermediate's array is still computed and
        # cached earlier in the notebook (other code depends on it); it's just not animated here.
        "sub_stages": [
            {
                "label": "Step 1: Nuclei signal (DNA, 405 nm) with CLAHE contrast enhancement",
                "add": [image_layer_names["Nuclei_preprocessed_CLAHE"]],
                "strip": [],
            },
            {
                # the 405 nm signal is NOT stripped here (or at the next sub-stage) - it stays
                # visible underneath every mask for the rest of this compartment's clip, so the
                # segmentation is always shown in context against the raw nuclei channel.
                # First iteration: raw 2.5D sliding-window CellposeSAM output, before decoupling.
                "label": "Step 2: Nuclei - first iteration (raw 2.5D sliding-window CellposeSAM, before decoupling)",
                "add": [label_layer_names["Nuclei_raw_window_segmentation"]],
                "strip": [],
            },
            {
                # last iteration: the final, 3D-stitched/border-cleaned mask used as watershed
                # seeds - the decoupled-but-unstitched intermediate in between is not shown
                "label": "Step 3: Nuclei - last iteration (final mask: 3D stitched, border-cleaned) - watershed seeds",
                "add": [label_layer_names["Nuclei_final"]],
                "strip": [label_layer_names["Nuclei_raw_window_segmentation"]],
            },
        ],
    },
    {
        "name": "02_cell",
        # the 405 nm nuclei signal is still visible (never stripped, see nuclei sub-stages above)
        # when this compartment starts, so it is faded out here too, together with the nuclei
        # final mask, before the 555 nm (AGP) signal below fades in to replace it
        "fade_out_layers": [
            label_layer_names["Nuclei_final"],
            image_layer_names["Nuclei_preprocessed_CLAHE"],
        ],
        # slower, smoother Z-axis peel than the plain default (more discrete sweep positions)
        "peel_kwargs": {"n_positions": 56, "hold_steps": 16},
        "sub_stages": [
            {
                # raw signal only - the morphology-dependent elevation/threshold intermediate the
                # actual pipeline computes from this channel is not shown; this clip goes straight
                # from the raw cell-boundary channel to the final cell mask (one peel cycle)
                "label": "Step 4: Raw cell-boundary channel (AGP, 555 nm)",
                "add": [image_layer_names["Raw_AGP"]],
                "strip": [],
            },
            {
                # the 555 nm signal is NOT stripped here - it stays visible underneath the cell
                # mask for the rest of this compartment's clip (same treatment as the 405 nm
                # nuclei signal in the nuclei compartment above), and the cell mask itself is
                # never stripped within this clip either - it's still fully visible at the end
                "label": "Step 5: Cell - final mask (hole-filled, size-filtered, post-hoc refined)",
                "add": [label_layer_names["Cell_final"]],
                "strip": [],
            },
        ],
    },
    {
        "name": "04_organoid",
        "fade_out_layers": [label_layer_names["Cytoplasm_final"]],
        # slower, smoother Z-axis peel than the plain default (more discrete sweep positions)
        "peel_kwargs": {"n_positions": 56, "hold_steps": 16},
        # the final mask should stack up into view once and stay - not peel away/add back again
        "peel_once": True,
        "sub_stages": [
            {
                # raw signal only - the decoupled per-slice intermediate is not shown; this clip
                # goes straight from the raw organoid channel to the final organoid mask (one
                # peel cycle)
                "label": "Step 10: Raw organoid channel (AGP, 555 nm)",
                "add": [image_layer_names["Raw_AGP"]],
                "strip": [],
            },
            {
                # the 555 nm signal is NOT stripped here either - it stays visible underneath
                # the organoid mask for the rest of the pipeline's clip
                "label": "Step 11: Organoid - final mask (3D stitched, border-cleaned) - bounds the watershed",
                "add": [label_layer_names["Organoid_final"]],
                "strip": [],
            },
        ],
    },
]


# ## Render one clip per compartment
#
# `render_compartment` renders one of the three peel-in/fade-out compartments (nuclei, cell,
# organoid) onto its own `Animation` and its own mp4, following the **black -> signal ->
# segmentation 1 -> segmentation 1 peel -> segmentation 2 peel on -> segmentation 2 peel -> ...**
# order described above: the previous compartment's final mask (if any) fades out first; then the
# compartment's first sub-stage (its raw signal image) is revealed with a plain opacity fade-in
# (`animate_opacity_transition`), never a peel; then every sub-stage after that hides its
# superseded intermediate outright (`visible = False`) while its newly-added mask gets the
# `animate_peel_step` stack-up/peel-away/add-back reveal cycle - all chained onto that one
# `Animation`, so the whole compartment plays as a single continuous clip.
#
# Every compartment gets the same camera treatment now: `render_compartment` first calls
# `estimate_compartment_frames` to work out exactly how many frames this clip's fades/peels will
# capture, then wraps a real `Animation` in a `RotatingCamera` sized to that frame count, so the
# camera eases through `sequence_of_angles` across the *whole* clip (see the "Camera" section
# above - including the open question of whether that path currently lands back on the shared
# base angle for a seamless cut into the next compartment). `animate_fade_out`/
# `animate_opacity_transition`/`animate_peel_step` are handed this
# `RotatingCamera` in place of the raw `Animation` and never know the difference, since it only
# ever calls `.capture_keyframe(...)` the same way they already do. A compartment can still opt
# into a different peel speed via `peel_kwargs` in its entry in `compartments` above, and
# `peel_once: True` (currently only `04_organoid` does) makes every segmentation sub-stage's
# `animate_peel_step` call run with `single_reveal=True` - the mask stacks up into view and stays,
# instead of also peeling away and adding back.
#
# `render_cytoplasm` renders the bespoke cytoplasm clip (see the markdown above `compartments`) with
# the same `RotatingCamera` treatment (its frame count is a fixed formula instead of
# `estimate_compartment_frames`, since its sequence of opacity crossfades never varies): the raw
# 555 nm channel fades back in under the cell mask (still visible from the cell clip) -> the cell
# mask turns transparent-ish while the nuclei mask fades in underneath it (translucent, so it reads
# as "visible inside" the cell) -> the cytoplasm mask layers in at that same opacity -> the cell
# mask drops out -> the nuclei mask drops out -> the cytoplasm mask comes up to full opacity as the
# final view. Every step is an opacity crossfade via `animate_opacity_transition`, never a peel -
# every layer involved is already loaded in the viewer, nothing new needs revealing slice-by-slice.

# In[23]:


# the fixed camera baseline every compartment's clip starts on - `angle1` in `sequence_of_angles`
# below matches this value (see the "Camera" section's open issue about whether the clip also
# ends back here, for a seamless cut between compartments)
shared_camera_angles = viewer.scene.camera.angles
shared_camera_zoom = viewer.scene.camera.zoom

# the one rotation path every compartment's `RotatingCamera` eases through: the hand-captured
# `sequence_of_angles` defined above (`angle1` through `angle4`) - four explicit `camera.angles`
# orientations, not a "base angle + N degrees" formula, so the exact pose at every stop is
# whatever was captured by hand, visible right there rather than re-derived on every call.
# `angle1` matches `shared_camera_angles` (both `(141.0, 1, 136.0)`), so the path starts on the
# shared baseline - but `angle4` does NOT match it back, so (unlike the earlier axis-0-only
# formula this replaced) a compartment's clip no longer necessarily ends on the same orientation
# it started on. That breaks the "lands back on `shared_camera_angles` so the cut into the next
# compartment is seamless" invariant described in the "Camera" section above and in
# `RotatingCamera`'s docstring - there will be a visible snap in camera angle at each
# compartment-to-compartment cut unless `sequence_of_angles` is extended to close the loop back
# to `angle1`. Left as given rather than silently "fixed" - decide deliberately whether to close
# the loop.
camera_rotation_keyframes = sequence_of_angles


def estimate_compartment_frames(compartment: dict) -> int:
    """
    The exact total frame count `render_compartment` will capture for this compartment -
    needed up front so `RotatingCamera` can pace its `sequence_of_angles` keyframe path evenly
    across the *whole* clip, independent of how many sub-stages it has or how long each
    fade/peel takes. Mirrors the frame counts `animate_fade_out`,
    `animate_opacity_transition`, and `animate_peel_step` actually produce - see their
    docstrings/bodies above for where each term comes from.
    """
    total = 0
    if compartment["fade_out_layers"]:
        total += (
            1 + 12 + 1
        )  # animate_fade_out: capture(1) + its default fade steps=12 + hide(1)

    peel_kwargs = compartment.get("peel_kwargs", {})
    n_positions = peel_kwargs.get("n_positions", 24)
    hold_steps = peel_kwargs.get("hold_steps", 14)
    peel_once = compartment.get("peel_once", False)

    for sub_stage_index in range(len(compartment["sub_stages"])):
        if sub_stage_index == 0:
            total += (
                1 + hold_steps
            )  # animate_opacity_transition: capture(1) + steps(hold_steps)
        else:
            total += 1 + hold_steps  # snap from "no clip" + hold fully hidden
            total += n_positions + hold_steps  # stack up + hold fully revealed
            if not peel_once:
                total += n_positions + hold_steps  # peel away + hold fully hidden
                total += n_positions + hold_steps  # add back + hold fully revealed
            total += 1  # clear clipping planes
    return total


def render_compartment(compartment: dict) -> pathlib.Path:
    real_animation = Animation(viewer)
    total_frames = estimate_compartment_frames(compartment)
    # every capture_keyframe call below goes through this proxy instead of `real_animation`
    # directly, so the camera eases along `camera_rotation_keyframes` (a full 360-degree
    # revolution about the one clean axis - see `RotatingCamera` above) across the whole clip
    animation = RotatingCamera(
        real_animation, viewer, total_frames, camera_rotation_keyframes
    )

    if compartment["fade_out_layers"]:
        animate_fade_out(animation, viewer, compartment["fade_out_layers"])

    peel_kwargs = compartment.get("peel_kwargs", {})
    # when set, every segmentation sub-stage in this compartment stacks up once and stays,
    # instead of also peeling away and adding back (see organoid's entry in `compartments`)
    peel_once = compartment.get("peel_once", False)

    for sub_stage_index, sub_stage in enumerate(compartment["sub_stages"]):
        for layer_name in sub_stage["strip"]:
            viewer.layers[layer_name].visible = False

        viewer.canvas.overlays.text.text = sub_stage["label"]

        if sub_stage_index == 0:
            # this compartment's very first sub-stage is always its raw signal image, not a
            # segmentation mask - the flythrough shows it as a plain "black -> signal" opacity
            # fade-in, not the stack-up/peel-away/add-back cycle every segmentation stage below
            # gets from `animate_peel_step`. BUT a signal layer that is already visible (carried
            # over, still fully opaque, from an earlier compartment - e.g. the 555 nm channel
            # persisting from cell/cytoplasm into organoid) must NOT be hard-reset to invisible
            # here: that reset is what used to make the view flash completely black for a frame
            # right as the previous compartment's fade-out finished. Only force the opacity to 0
            # (so there is something to fade *from*) the first time this layer is ever shown.
            for layer_name in sub_stage["add"]:
                layer = viewer.layers[layer_name]
                if not layer.visible:
                    layer.opacity = 0.0
                layer.visible = True
            animate_opacity_transition(
                animation,
                viewer,
                {layer_name: 1.0 for layer_name in sub_stage["add"]},
                steps=peel_kwargs.get("hold_steps", 14),
            )
        else:
            # every sub-stage after the signal is a segmentation mask: "peel on" (stack up into
            # view), hold, then - unless this compartment is `peel_once` - peel away, hold, add
            # back - see `animate_peel_step`
            for layer_name in sub_stage["add"]:
                viewer.layers[layer_name].visible = True
            animate_peel_step(
                animation,
                viewer,
                sub_stage["add"],
                single_reveal=peel_once,
                **peel_kwargs,
            )

    # the rotation above should already land back here (one full revolution per axis, paced by
    # the exact frame count `estimate_compartment_frames` predicted) - reset explicitly anyway as
    # cheap insurance against any floating-point drift before the next compartment starts
    viewer.scene.camera.angles = shared_camera_angles
    viewer.scene.camera.zoom = shared_camera_zoom

    compartment_mp4_path = pathlib.Path(f"{mp4_file_dir}/{compartment['name']}.mp4")
    real_animation.animate(str(compartment_mp4_path), canvas_only=True)
    return compartment_mp4_path


def render_cytoplasm(crossfade_steps: int = 20) -> pathlib.Path:
    agp_layer = image_layer_names["Raw_AGP"]
    cell_layer = label_layer_names["Cell_final"]
    nuclei_layer = label_layer_names["Nuclei_final"]
    cytoplasm_layer = label_layer_names["Cytoplasm_final"]

    real_animation = Animation(viewer)
    # this clip always captures exactly: 1 (agp opacity snap) + 4 opacity transitions of
    # (1 + crossfade_steps) frames each + 2 (drop cell, drop nuclei) - see
    # `estimate_compartment_frames` above for the same idea applied to the peel-based clips
    total_frames = 1 + 4 * (1 + crossfade_steps) + 2
    animation = RotatingCamera(
        real_animation, viewer, total_frames, camera_rotation_keyframes
    )

    # the cell mask is already visible (and fully opaque) from the end of the cell clip, and the
    # 555 nm channel is already visible too (never stripped in the cell clip - see `compartments`)
    # - only force its opacity to 0 if it somehow isn't visible yet, so a layer that is already
    # on screen doesn't flash to black for a frame before "fading back in" to where it already was
    if not viewer.layers[agp_layer].visible:
        viewer.layers[agp_layer].opacity = 0.0
    viewer.layers[agp_layer].visible = True
    animation.capture_keyframe(steps=1)
    viewer.canvas.overlays.text.text = "Step 6: Cytoplasm - raw cell-boundary channel (AGP, 555 nm), with the cell mask"
    animate_opacity_transition(
        animation, viewer, {agp_layer: 1.0}, steps=crossfade_steps
    )

    # the cell mask turns transparent-ish while the nuclei mask fades in underneath it
    # (translucent, visible "inside" the cell)
    viewer.canvas.overlays.text.text = (
        "Step 7: Cytoplasm - cell mask turns transparent, nuclei mask revealed inside"
    )
    animate_opacity_transition(
        animation, viewer, {cell_layer: 0.85, nuclei_layer: 0.55}, steps=crossfade_steps
    )

    # bring in the cytoplasm mask at the same opacity the cell mask is currently at
    viewer.canvas.overlays.text.text = (
        "Step 8: Cytoplasm - cytoplasm mask layered in (cell mask minus nuclei)"
    )
    animate_opacity_transition(
        animation, viewer, {cytoplasm_layer: 0.85}, steps=crossfade_steps
    )

    # drop the cell mask
    viewer.layers[cell_layer].visible = False
    viewer.layers[cell_layer].opacity = 1.0  # reset, in case it is ever shown again
    animation.capture_keyframe(steps=1)

    # drop the nuclei mask
    viewer.layers[nuclei_layer].visible = False
    viewer.layers[nuclei_layer].opacity = 1.0  # reset, in case it is ever shown again
    animation.capture_keyframe(steps=1)

    # bring the cytoplasm mask up to full opacity - the final view
    viewer.canvas.overlays.text.text = (
        "Step 9: Cytoplasm - final mask (cell mask minus nuclei mask)"
    )
    animate_opacity_transition(
        animation, viewer, {cytoplasm_layer: 1.0}, steps=crossfade_steps
    )

    # the rotation above should already land back here - reset explicitly anyway, same as
    # `render_compartment`, as cheap insurance against any floating-point drift
    viewer.scene.camera.angles = shared_camera_angles
    viewer.scene.camera.zoom = shared_camera_zoom

    cytoplasm_mp4_path = pathlib.Path(f"{mp4_file_dir}/03_cytoplasm.mp4")
    real_animation.animate(str(cytoplasm_mp4_path), canvas_only=True)
    return cytoplasm_mp4_path


# render in the requested build-up order: nuclei -> cell -> cytoplasm -> organoid
compartments_by_name = {
    compartment["name"]: compartment for compartment in compartments
}
compartment_mp4_paths = [
    render_compartment(compartments_by_name["01_nuclei"]),
    render_compartment(compartments_by_name["02_cell"]),
    render_cytoplasm(),
    render_compartment(compartments_by_name["04_organoid"]),
]

print("All compartments animated")


# ### Sanity check: make sure each clip actually rendered content
#
# Some of this animation's clips legitimately start on a black frame - the clipping plane starts
# fully hidden before the first mask peels in - so a quick look at frame 0 isn't a reliable way to
# tell whether a clip rendered correctly. This instead samples frames across each compartment's mp4
# and checks that *some* of them are well above black. If a clip comes back entirely black, the
# `animate_peel_step`/`animate_fade_out` logic itself is not the likely cause (it's covered by the
# sign/interpolation notes in their docstrings above, and was verified against this well/FOV's own
# cached masks) - the two most common real causes are a **stale kernel** still running an older,
# already-fixed version of these functions (restart the kernel and re-run every cell from the top,
# not just this one), or the napari **window being minimized, occluded, or not focused** while this
# cell was rendering (some Linux window managers hand back a black framebuffer for a Qt window that
# isn't actually on screen - leave the window visible and in the foreground until rendering
# finishes).

# In[24]:


def check_mp4_not_blank(
    mp4_path: pathlib.Path, max_sampled_frames: int = 60, black_threshold: float = 1.0
) -> None:
    """Warn if an exported mp4 looks entirely black across a sample of its frames."""
    frames = iio.imread(str(mp4_path))
    sample = frames[:: max(1, len(frames) // max_sampled_frames)]
    brightness = [float(frame.mean()) for frame in sample]
    if max(brightness) < black_threshold:
        print(
            f"WARNING: {mp4_path.name} looks entirely black (max sampled mean brightness "
            f"{max(brightness):.2f}). See the markdown cell above for the likely causes."
        )
    else:
        print(
            f"{mp4_path.name}: OK (max sampled mean brightness {max(brightness):.2f})"
        )


for compartment_mp4_path in compartment_mp4_paths:
    check_mp4_not_blank(compartment_mp4_path)


# ## Concatenate the per-compartment clips into a single iterative build-up animation
#
# This stitches the four compartment clips together in the same nuclei -> cell -> cytoplasm ->
# organoid order they were just rendered in, producing one continuous video that shows the
# segmentation approach being built up iteratively.

# In[25]:


combined_mp4_path = pathlib.Path(
    f"{mp4_file_dir}/00_iterative_segmentation_buildup.mp4"
)

clips = [VideoFileClip(str(p)) for p in compartment_mp4_paths]
combined_clip = concatenate_videoclips(clips, method="compose")
combined_clip.write_videofile(str(combined_mp4_path), fps=30)
combined_clip.close()
for clip in clips:
    clip.close()


# ## Convert every rendered mp4 (per-step and combined) to a gif

# In[ ]:


mp4_file_paths = list(pathlib.Path(mp4_file_dir).rglob("*.mp4"))
for mp4_file in mp4_file_paths:
    mp4_file = pathlib.Path(mp4_file)
    gif_file = pathlib.Path(gif_file_dir / f"{mp4_file.stem}.gif")
    mp4_to_gif(str(mp4_file), str(gif_file))

print(f"mp4s written to: {mp4_file_dir}")
print(f"gifs written to: {gif_file_dir}")


# In[ ]:


for partial_mp4_path in compartment_mp4_paths:
    partial_mp4_path.unlink(missing_ok=True)

print(
    f"cleaned up {len(compartment_mp4_paths)} per-compartment partial mp4s, "
    f"keeping only {combined_mp4_path.name} in {mp4_file_dir}"
)


# In[ ]:


viewer.close()
