"""Train UNet++ on a ConvNeXt or EfficientNet encoder for thin and faint crack segmentation.

Expected dataset layout::

    dataset/
      train/images, train/masks
      val/images,   val/masks
      test/images,  test/masks

The project is organised and run like the other hairline trainers (the Swin
U-Net, FiLM U-Net and multi-decoder U-Net projects). Data pairing, the
augmentation library, the loss, metrics, checkpoints and figures are theirs,
so the models can be compared directly; the network is new. It has no
conditioning on the ground sampling distance. Training and evaluation only:
there is no full-orthomosaic inference, skeletonisation or DXF export here.

The network (``UNetPlusPlus``) is the UNet++ of segmentation_models_pytorch
(smp), built from smp's own encoder, nested decoder and output convolution:

* Encoder (``--encoder``, default ``tu-convnext_tiny``): any smp encoder,
  ImageNet-pretrained. ``tu-<name>`` is a timm model (``tu-convnext_tiny``,
  ``tu-convnext_small``, ``tu-convnextv2_tiny``, ``tu-tf_efficientnetv2_s``);
  ``efficientnet-b0`` to ``efficientnet-b7`` are smp's own EfficientNets. The
  images are normalised with the statistics the encoder was pretrained with.
  ``--drop-path-rate`` (0.1) is the stochastic depth of a timm encoder.
* Decoder: the nested UNet++ decoder with dense skip connections on the
  features at 1/2 to 1/32 scale, ``--decoder-channels`` wide (256 128 64 32
  16), with scSE attention (``--decoder-attention``) and nearest-neighbour
  upsampling (``--decoder-interpolation``), then a 3x3 output convolution.
* 1/2-scale stem, for ConvNeXt only: ConvNeXt downsamples four times in its
  first layer and has no features at 1/2 scale, which UNet++ needs
  (``smp.UnetPlusPlus`` fails on it in the forward pass). Two 3x3
  convolutions on the image, the first with stride 2, trained from scratch,
  supply that feature (``--half-channels``, 48). EfficientNet has its own and
  gets no such stem.
* Hairline stem, as in the other trainers: the last UNet++ node upsamples
  from 1/2 scale without a skip connection, which is coarse for cracks one or
  two pixels wide. A full-resolution stem, fed with fixed morphological
  black-hat and top-hat responses (``ThinLineResponse``), is given to that
  node as its skip. ``--no-line-prior`` disables the thin-line channels and
  ``--no-hairline-stem`` the whole stem.
* Deep supervision as in the UNet++ paper: a 1x1 output on each of the
  intermediate 1/2-scale nodes X(0,1), X(0,2) and X(0,3), trained on the crack
  mask (``--aux-weights``). All output biases start at the crack prior.
* With an EfficientNet encoder, ``--no-hairline-stem`` and
  ``--no-deep-supervision`` the network is ``smp.UnetPlusPlus`` itself.
* ``forward(images)``: eval mode returns the crack logits (N, 1, H, W). Train
  mode returns ``(logits, auxiliary)`` with the deep-supervision logits.

Loss: the reference loss (BCE + recall-leaning Tversky + clDice with warm-up)
on the network output, plus ``--aux-weights`` (0.2 each) times BCE + Tversky
on the three deep-supervision outputs. ``--loss focal`` replaces BCE + Tversky
by the binary focal loss everywhere. ``training_history.csv`` has the whole
objective as ``train_loss`` and the part of the network output as
``train_output_loss`` (the terms ``val_loss`` has).

Thresholds: the threshold of the network output is calibrated on validation
with ``--select-metric`` (the 2-px tolerant F1) on a grid from 0.01 to 0.99.
Besides that shared threshold, every source family gets its own, if it has
at least ``--source-threshold-min-images`` (10) validation images; a smaller
source keeps the shared one. On the first UNet++ run the shared threshold
(0.875) was set by the benchmark photos, whose cracks are thick and
contrasted, while the orthomosaic tiles scored best at the bottom of the
grid (0.05), 0.05 to 0.09 tolerant F1 higher than at 0.875. The test
metrics are reported both ways: at the shared threshold as before, and under
``with_source_thresholds`` with every source at its own. ``best.pt`` stores
both (``optimal_threshold``, ``source_thresholds`` and their ``_tta``
variants), so a detector can use the threshold of the source it is run on.

GSD table (``--gsd-csv``, optional): the ground sampling distance (mm per
pixel) of each source, as a CSV with the columns ``pattern,gsd_mm_per_px``.
``pattern`` is a regular expression searched in the image file name (for
example ``^EH104_``); the first matching row gives the GSD and a tile without
a match is unknown (NaN). The network never sees the GSD. Only the data
pipeline uses the table: the mixing augmentations combine tiles of one source
family and one GSD, and the test metrics are also reported per GSD value. A
table of sources, tile counts and GSD values per split is printed at start-up.

The augmentations keep track of the GSD for that purpose: it follows every
scale change. An augmentation that enlarges the image by a
factor s shows each crack s times wider in pixels, so the tile's GSD becomes
gsd / s. That holds for the reference rescale (0.75-1.25) and affine
(0.9-1.1), the random resized crop, zoom-out, perspective (mean local scale at
the tile centre) and mosaic (gsd x 2). Crops, flips, 90-degree turns,
photometric changes, elastic and lens-type warps, cutout, fading, synthetic
cracks and the downscale-upscale degradation leave it unchanged, and an
unknown GSD stays unknown. Copy-paste donors, CutMix and MixUp partners and
mosaic tiles come from the same source family and the same GSD (or all
unknown), and are brought to the scale of the tile they are mixed into.

Kept from the other hairline trainers, unchanged:

* Dataset pairing, Kaggle dataset discovery and the default paths.
* Native-resolution random crops for training (no resizing to 256 px), full
  unresized images for validation and test, tiled evaluation of large images
  with blended logits, evaluation batches grouped by shape and capped in
  megapixels.
* The reference augmentation and the whole extended library (below).
* Model selection and threshold calibration on validation only with the 2-px
  tolerance F1; strict precision, recall, F1 and IoU; per-source and per-GSD
  metrics; EMA weights; flip TTA; warm-up + cosine schedule, AMP, gradient
  clipping, early stopping, resumable ``last.pt`` / ``best.pt``,
  ``--evaluate-only``.
* Optimiser defaults: learning rate 3e-4, encoder factor 0.1, weight decay
  1e-4; ``--optimizer adam`` for plain Adam. The decoder and the stems train
  at the full rate.
* The server workflow: checkpoints written to a temporary file and renamed,
  ``validation_by_source.csv`` with the three weakest sources in the log,
  ``--init-weights`` (a ``best.pt`` of this trainer with the same network),
  ``--label-tolerance`` and test overlays spread over the sources.
* Every table and figure, in the same plot style.

Augmentation: every technique of the library is on by default
(``--aug-preset all``). The reference augmentation always runs (except in the
``none`` preset): rescale, crack-focused crop, crack copy-paste, synthetic
hairlines, crack fading, flips and turns, affine, elastic, the photometric
chain (contrast levels, brightness, gamma, illumination gradient, CLAHE, HSV,
grayscale, Gaussian and motion blur, Gaussian noise, JPEG) and cutout. The
extended library adds the operations of the table below. Each is a registered
function with its own probability ``--<name>-p`` and, where it has one, a
magnitude ``--<name>-m`` in [0, 1] that scales its range (1 = the full range
named in its help text).

* Geometric operations move image, mask and valid together: masks are warped
  linearly and cut at 0.4 so 1-px cracks stay connected, valid is warped with
  nearest neighbour, and pixels warped in from outside the image get valid = 0
  instead of a background label.
* An occlusion that hides a crack sets valid = 0 there.
* Label safety (``--label-safety``, on by default): after an operation that
  can reduce crack contrast (blur, noise, fog, compression, exposure, shadows,
  stains), the crack contrast against the local background is measured as in
  the copy-paste donor test. Where a crack that was visible has dropped below
  4 grey levels, its pixels and a 2-px band around them get valid = 0, so the
  model is never trained on a label it cannot see. The measure is taken per
  32-px cell. Cracks that were already below the limit in the source image are
  the dataset's own faint labels and are left as they are.
* Fixed order: scale -> crop -> mixing (mosaic / CutMix / MixUp, copy-paste)
  -> synthetic cracks and distractors -> geometric -> orthomosaic artefacts ->
  photometric -> weather and lighting -> blur -> noise -> compression ->
  occlusion.

Presets (``--aug-preset``); an individual ``--<name>-p`` / ``--<name>-m`` flag
overrides the preset:

* ``all`` (default): every operation of the table at probability 0.05 and
  magnitude 0.75 with ``--max-rotation`` 180, on top of the reference
  augmentation. That includes the seven operations marked ``*``, which can
  harm crack labels (the reason is in the code); ``heavy`` is the strongest
  preset without them.
* ``reference``: the reference augmentation, nothing added. This is the
  default of the Swin U-Net and FiLM U-Net projects, for a comparison in which
  only the network differs.
* ``none``: a random native-resolution crop (with the reference crack focus,
  which is sampling rather than augmentation) and nothing else.
* ``light``, ``medium``, ``heavy``: the reference plus the operations below.
  Magnitude of every enabled operation: 0.5, 0.75 and 1.0. ``--max-rotation``:
  20, 45 and 180 degrees.

Probability of each operation per preset (``*`` = off in light, medium and
heavy because it harms crack labels)::

    operation            stage        light  medium  heavy  all
    mosaic               mixing       0      0.05    0.10   0.05
    cutmix               mixing       0      0.10    0.15   0.05
    mixup*               mixing       0      0       0      0.05
    branch-crack         cracks       0.04   0.06    0.08   0.05
    varwidth-crack       cracks       0.03   0.05    0.06   0.05
    joint                distractors  0.05   0.08    0.10   0.05
    sawcut               distractors  0      0.04    0.06   0.05
    scratch              distractors  0      0.04    0.06   0.05
    painted-line         distractors  0      0.03    0.05   0.05
    cable-shadow         distractors  0      0.05    0.08   0.05
    stain                distractors  0.05   0.08    0.10   0.05
    efflorescence        distractors  0      0.05    0.08   0.05
    dirt                 distractors  0      0.05    0.08   0.05
    spall                distractors  0      0.04    0.06   0.05
    resized-crop         rescaling    0      0.15    0.25   0.05
    zoom-out             rescaling    0      0.08    0.12   0.05
    shear                warp         0.10   0.15    0.20   0.05
    translate            warp         0.10   0.15    0.20   0.05
    perspective          warp         0.10   0.15    0.20   0.05
    optical              warp         0      0.08    0.12   0.05
    grid-distortion      warp         0      0.05    0.10   0.05
    piecewise-affine*    warp         0      0       0      0.05
    seam                 ortho        0.05   0.08    0.12   0.05
    nodata-border        ortho        0      0.03    0.05   0.05
    exposure-step        ortho        0.05   0.08    0.12   0.05
    blur-patch           ortho        0      0.05    0.08   0.05
    ghosting*            ortho        0      0       0      0.05
    rgb-shift            photometric  0.10   0.12    0.15   0.05
    channel-shuffle      photometric  0      0       0.02   0.05
    color-jitter         photometric  0.15   0.20    0.25   0.05
    white-balance        photometric  0.15   0.20    0.25   0.05
    tone-curve           photometric  0.10   0.15    0.20   0.05
    autocontrast         photometric  0      0.05    0.08   0.05
    equalize             photometric  0      0       0.05   0.05
    sharpen              photometric  0.05   0.08    0.10   0.05
    emboss               photometric  0      0       0.02   0.05
    posterize            photometric  0      0       0.02   0.05
    solarize*            photometric  0      0       0      0.05
    overexposure         photometric  0      0.04    0.06   0.05
    underexposure        photometric  0      0.04    0.06   0.05
    shadow               weather      0.08   0.12    0.15   0.05
    sun-flare            weather      0      0       0.03   0.05
    vignette             weather      0.05   0.08    0.10   0.05
    spotlight            weather      0      0.05    0.08   0.05
    fog                  weather      0      0       0.03   0.05
    wet                  weather      0      0.05    0.08   0.05
    rain*                weather      0      0       0      0.05
    defocus              blur         0.03   0.05    0.08   0.05
    median-blur          blur         0      0.02    0.04   0.05
    zoom-blur*           blur         0      0       0      0.05
    glass-blur*          blur         0      0       0      0.05
    poisson-noise        noise        0.08   0.10    0.15   0.05
    iso-noise            noise        0.05   0.08    0.10   0.05
    speckle              noise        0      0.05    0.08   0.05
    salt-pepper          noise        0      0       0.03   0.05
    pixel-dropout        noise        0      0.03    0.05   0.05
    chromatic-aberration noise        0      0.05    0.08   0.05
    ringing              noise        0      0.03    0.05   0.05
    webp                 compression  0.08   0.10    0.15   0.05
    downscale            compression  0.08   0.10    0.15   0.05
    random-erasing       occlusion    0.05   0.08    0.10   0.05
    gridmask             occlusion    0      0.04    0.06   0.05
    hide-and-seek        occlusion    0      0.04    0.06   0.05

``--randaugment N M`` additionally applies N randomly chosen photometric,
weather, blur, noise or compression operations (never geometric or mixing, and
never a ``*`` operation) at magnitude M (0-10) to every training image.

Augmentation tooling: ``--preview-augmentations K`` writes, for K training
tiles, one panel per enabled augmentation (image | label overlay | valid map,
with the GSD before and after) to ``<output>/augmentation_preview`` before
training; ``--preview-only`` stops there. ``augmentation_counts.json`` records
how often each augmentation fired and how many crack pixels the label-safety
check excluded. The resolved augmentation configuration is stored in
``configuration.json`` and ``best.pt``.

Outputs, in ``unetpp_hairline_output`` by default:
``configuration.json``, ``training_history.csv``, ``validation_by_source.csv``,
``last.pt``, ``best.pt``, ``val_threshold_sweep.csv``, ``test_metrics.json``,
``contrast_coverage.json``, ``augmentation_counts.json``,
``test_predictions/`` and ``plots/``, plus ``augmentation_preview/`` when
requested. ``test_predictions/`` holds, for every saved test image, the
probability map and the overlay of the network output. ``best.pt`` also
stores the architecture with its encoder and decoder options, the GSD table
and the augmentation configuration. ``--evaluate-only`` writes its metrics
to ``test_metrics_evaluate_only.json`` and leaves the run's other files as
they are, so an earlier run can be scored with the per-source thresholds
without training again.

``--self-test`` checks the network (against ``smp.UnetPlusPlus``) and the
loss, and the alignment, binary labels and GSD bookkeeping of every
augmentation through dataset and model, then exits.
"""

from __future__ import annotations
from .common import (
    BASIC_METRICS, COUNT_NAMES, Counter, DEFAULT_DATASET_PATH, DEFAULT_ENCODER, DEFAULT_OUTPUT_PATH, DataLoader,
    ENCODER_EXAMPLES, METRIC_NAMES, Path, WeightedRandomSampler, argparse, csv, defaultdict, json, math, nn, np, os,
    random, re, seed_everything, seed_worker, sys, time, torch,
)
from .gsd import GSDTable, gsd_groups, print_source_table
from .augmentations import (
    AUGMENTATIONS, AugmentationOptions, LABEL_SAFETY_CELL, LABEL_SAFETY_GREY_LEVELS, NO_AUGMENTATION,
    PRESET_MAGNITUDE, PRESET_MAX_ROTATION, PRESET_NAMES, RANDAUGMENT_POOL, preset_probabilities,
)
from .data import (
    CrackSegmentationDataset, collect_pairs, save_augmentation_preview, shape_grouped_batches, source_family,
)
from .models import ARCHITECTURES, ModelEMA, different_arguments
from .losses import FocalCrackLoss, HairlineCrackLoss
from .evaluation import (
    ThresholdSweep, at_threshold, basic_metrics, counts_at_indices, evaluate, metrics_from_counts, prediction_writer,
    source_threshold_indices,
)
from .plots import save_evaluation_tables, save_plots

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parameter_groups(model, learning_rate, encoder_factor, weight_decay):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # No weight decay on BatchNorm weights and biases.
            # Everything outside the pretrained encoder (stems, decoder, outputs) trains at the full rate.
            no_decay = parameter.ndim <= 1
            groups[(name.startswith("encoder."), no_decay)].append(parameter)
    return [
        {"params": parameters, "lr": learning_rate * (encoder_factor if is_encoder else 1.0),
         "weight_decay": 0.0 if no_decay else weight_decay}
        for (is_encoder, no_decay), parameters in groups.items()
    ]


def warmup_cosine(total_steps: int, warmup_steps: int, minimum_ratio: float):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


MODE_NAMES = {"no_tta": "no TTA", "tta": "flip TTA"}


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def save_checkpoint(state, path: Path) -> None:
    """Write to a temporary file, then rename it over `path`.

    A job killed while saving (time limit, scancel) then leaves the previous
    checkpoint intact instead of a truncated file that cannot be resumed.
    """
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def open_epoch_log(path: Path, fields, first_epoch: int):
    """Open a per-epoch CSV (epoch in the first column) for the rows from `first_epoch` on.

    A fresh run starts the file. A resumed run keeps the rows of the epochs before
    `first_epoch` and drops later ones: a job killed between writing an epoch's
    row and saving last.pt repeats that epoch, which would otherwise be logged twice.
    """
    rows = []
    if first_epoch > 1 and path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows = rows[:1] + [row for row in rows[1:] if row and row[0].isdigit() and int(row[0]) < first_epoch]
    handle = path.open("w", newline="", encoding="utf-8")
    csv.writer(handle).writerows(rows or [list(fields)])
    handle.flush()
    return handle


def initial_weights(path: Path, kwargs: dict) -> dict:
    """The best.pt of an earlier run for --init-weights, checked against the network this run builds.

    The network must be the same: encoder, decoder options, stems and deep supervision.
    """
    initial = torch.load(path, map_location="cpu", weights_only=False)
    different = different_arguments(initial["architecture"]["kwargs"], kwargs)
    if different:
        raise SystemExit(f"--init-weights holds a different network ({different}); this run builds "
                         f"{ {key: kwargs[key] for key in different} }. Pass the matching model options.")
    return initial


def flag_name(name: str) -> str:
    """Attribute name of an operation's arguments: 'zoom-out' -> 'zoom_out' (+ '_p' / '_m')."""
    return name.replace("-", "_")


# Reference augmentation arguments. Their argparse default is None, meaning "not given": the
# preset decides, and for every preset but "none" that is the reference default.
REFERENCE_AUGMENTATION_ARGUMENTS = (
    "scale_p", "focus_p", "copy_paste_p", "synthetic_p", "fade_p", "affine_p", "elastic_p", "cutout_p",
    "max_rotation", "synthetic_label_width", "contrast_p", "contrast_min_change", "contrast_max_change",
    "contrast_mode", "contrast_step",
)


def resolve_augmentation(args) -> tuple[AugmentationOptions, dict]:
    """Preset first, then the individual flags on top of it.

    Fills the unset augmentation arguments of `args` with their resolved values and returns the
    options for the dataset and the full configuration that is stored with the run.
    """
    defaults, preset = AugmentationOptions(), args.aug_preset
    overrides = dict(NO_AUGMENTATION) if preset == "none" else {}
    if preset in PRESET_MAX_ROTATION:
        overrides["max_rotation"] = PRESET_MAX_ROTATION[preset]
    for name in REFERENCE_AUGMENTATION_ARGUMENTS:
        if getattr(args, name) is None:
            setattr(args, name, overrides.get(name, getattr(defaults, name)))
    probabilities, operations = preset_probabilities(preset), {}
    for name, operation in AUGMENTATIONS.items():
        probability = getattr(args, flag_name(name) + "_p")
        magnitude = getattr(args, flag_name(name) + "_m", None)
        probability = probabilities[name] if probability is None else probability
        magnitude = PRESET_MAGNITUDE[preset] if magnitude is None else magnitude
        if not (0 <= probability <= 1 and 0 <= magnitude <= 1):
            raise SystemExit(f"--{name}-p and --{name}-m must lie in [0, 1].")
        operations[name] = (float(probability), float(magnitude))
        setattr(args, flag_name(name) + "_p", float(probability))
        if operation.scaled:
            setattr(args, flag_name(name) + "_m", float(magnitude))
    randaugment = None
    if args.randaugment is not None:
        count, magnitude = args.randaugment
        if count != int(count) or not 0 <= count <= len(RANDAUGMENT_POOL) or not 0 <= magnitude <= 10:
            raise SystemExit(f"--randaugment N M needs a whole N in 0-{len(RANDAUGMENT_POOL)} and M in 0-10.")
        randaugment = (int(count), float(magnitude)) if count > 0 else None
    options = AugmentationOptions(
        args.scale_p, args.focus_p, args.copy_paste_p, args.synthetic_p, args.fade_p,
        args.affine_p, args.elastic_p, args.cutout_p, args.max_rotation, args.synthetic_label_width,
        args.contrast_p, args.contrast_min_change, args.contrast_max_change,
        args.contrast_mode, args.contrast_step,
        dihedral=preset != "none", photometric=preset != "none", ops=operations,
        label_safety=args.label_safety, randaugment=randaugment,
    )
    configuration = {
        "preset": preset,
        "order": ["scale", "crop", "mixing", "synthetic cracks and distractors", "geometric",
                  "orthomosaic artefacts", "photometric", "weather and lighting", "blur", "noise",
                  "compression", "occlusion"],
        "reference": {**{name: getattr(args, name) for name in REFERENCE_AUGMENTATION_ARGUMENTS},
                      "dihedral": options.dihedral, "photometric": options.photometric},
        "operations": {name: {"stage": AUGMENTATIONS[name].stage, "p": probability, "m": magnitude}
                       for name, (probability, magnitude) in operations.items()},
        "enabled_operations": [name for name, (probability, _) in operations.items() if probability > 0],
        "label_safety": {"enabled": bool(args.label_safety), "threshold_grey_levels": LABEL_SAFETY_GREY_LEVELS,
                         "cell_px": LABEL_SAFETY_CELL},
        "randaugment": None if randaugment is None else {"n": randaugment[0], "m": randaugment[1],
                                                         "pool": RANDAUGMENT_POOL},
    }
    return options, configuration


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET_PATH,
                        help=f"80/10/10 dataset root (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/last.pt")
    parser.add_argument("--init-weights", type=Path,
                        help="Start from the weights of an earlier run's best.pt of this trainer instead of "
                             "ImageNet, to fine-tune on new data. It must hold the same network (encoder, decoder "
                             "options, stems). "
                             "The run itself is fresh (new schedule, EMA and history); not used when --resume "
                             "continues from last.pt")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="No training: evaluate <output>/best.pt on validation and test, write the "
                             "metrics to <output>/test_metrics_evaluate_only.json and the figures, curves and "
                             "confusion matrices to <output>/plots (other files untouched)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the built-in checks (network and loss, augmentation alignment "
                             "and GSD bookkeeping) on a temporary dummy dataset, then exit")
    # Model
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, metavar="NAME",
                        help="Encoder of segmentation_models_pytorch: tu-<timm model> or one of smp's own, for "
                             f"example {', '.join(ENCODER_EXAMPLES)} (default: {DEFAULT_ENCODER})")
    parser.add_argument("--no-pretrained", action="store_true", help="Random encoder initialisation")
    parser.add_argument("--drop-path-rate", type=float, default=0.1,
                        help="Stochastic depth of a timm encoder (tu-...). smp's own EfficientNets have theirs "
                             "built in and ignore this")
    parser.add_argument("--decoder-channels", type=int, nargs=5, default=[256, 128, 64, 32, 16], metavar="C",
                        help="Channels of the UNet++ decoder at 1/16, 1/8, 1/4, 1/2 and 1/1 scale")
    parser.add_argument("--decoder-attention", choices=("scse", "none"), default="scse",
                        help="scSE attention in every decoder node (default), or none")
    parser.add_argument("--decoder-interpolation", choices=("nearest", "bilinear"), default="nearest",
                        help="Upsampling inside the decoder (default: nearest, as in smp)")
    parser.add_argument("--half-channels", type=int, default=48,
                        help="Channels of the 1/2-scale stem given to an encoder without 1/2-scale features "
                             "(ConvNeXt); not used with EfficientNet")
    parser.add_argument("--no-hairline-stem", action="store_true",
                        help="No full-resolution stem: the last decoder node upsamples from 1/2 scale without "
                             "a skip connection, as in smp")
    parser.add_argument("--no-line-prior", action="store_true",
                        help="Disable the black-hat/top-hat thin-line input channels of the hairline stem")
    parser.add_argument("--no-deep-supervision", action="store_true",
                        help="No outputs on the intermediate UNet++ nodes")
    # GSD table of the sources: for the data pipeline only, the network does not see it
    parser.add_argument("--gsd-csv", type=Path, default=None,
                        help="CSV with the columns pattern,gsd_mm_per_px. pattern is a regular expression "
                             "searched in the image file name (e.g. ^EH104_); the first matching row gives "
                             "the GSD, no match means unknown. Without it every tile is unknown. Used for "
                             "mixing tiles of one GSD only and for the per-GSD test metrics")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Training crop size at native resolution; also the inference tile size")
    # CNN settings, those of train_unet_hairline.py.
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw",
                        help="adamw: decoupled weight decay (default, as the reference). adam: plain Adam, "
                             "where --weight-decay acts as L2 regularisation")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else min(4, os.cpu_count() or 1))
    # Loss
    parser.add_argument("--loss", choices=("hairline", "focal"), default="hairline",
                        help="hairline: BCE + Tversky + clDice (default, as the reference). focal: binary "
                             "focal loss + the same clDice term. The deep-supervision outputs use the same "
                             "region term, without clDice")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="--loss focal: weight of the crack class")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="--loss focal: focusing exponent")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--tversky-weight", type=float, default=1.0)
    parser.add_argument("--cldice-weight", type=float, default=0.5)
    parser.add_argument("--cldice-warmup-epochs", type=int, default=5,
                        help="Ramp clDice in linearly; it is noisy while predictions are still blobs")
    parser.add_argument("--false-positive-weight", type=float, default=0.4)
    parser.add_argument("--false-negative-weight", type=float, default=0.6,
                        help="> false-positive weight favours recall of faint cracks")
    parser.add_argument("--skeleton-iterations", type=int, default=10)
    parser.add_argument("--aux-weights", type=float, nargs=3, default=(0.2, 0.2, 0.2),
                        help="Deep supervision: weights of the outputs on the UNet++ nodes X(0,1), X(0,2) and X(0,3)")
    parser.add_argument("--label-tolerance", type=int, default=0,
                        help="Leave the ring of this many pixels around every label out of the loss: "
                             "hand-drawn labels sit a few pixels off the crack, and the tol_* metrics "
                             "already accept a prediction that close. 0 = strict loss (default)")
    # Evaluation
    parser.add_argument("--tolerance", type=int, default=2, help="Pixel tolerance for tol_* metrics")
    parser.add_argument("--select-metric", choices=METRIC_NAMES, default="tol_f1",
                        help="Metric for checkpoint selection and threshold calibration")
    parser.add_argument("--select-pattern", default="",
                        help="Regex on validation file names defining the target domain (model selection "
                             r"and threshold), e.g. '^EH\d' for the EH104 tiles only. Default: empty = "
                             "all validation data, every source in the dataset")
    parser.add_argument("--source-threshold-min-images", type=int, default=10,
                        help="A source family gets its own validation-calibrated threshold if it has at least "
                             "this many validation images; smaller ones keep the shared threshold")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA for calibration/test")
    parser.add_argument("--eval-batch-megapixels", type=float, default=2.0,
                        help="An evaluation batch holds at most this many megapixels "
                             "(and at most --eval-batch-size images)")
    parser.add_argument("--eval-tile", type=int, default=1024,
                        help="Validation/test images with a side above this are evaluated as overlapping "
                             "tiles of this size, because full-resolution photos do not fit in memory "
                             "whole. Smaller images still run whole; 0 disables tiling")
    parser.add_argument("--eval-tile-overlap", type=int, default=128, help="Overlap of those tiles in pixels")
    parser.add_argument("--save-predictions", type=int, default=40,
                        help="Test overlays to save, spread evenly over the sources; a number at least the "
                             "size of the test set saves every image")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug: subsample train")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Debug: subsample val/test")
    # Augmentation: the reference's arguments. No default here means "as the preset says", which is
    # the reference default shown in the help for every preset except "none".
    defaults = AugmentationOptions()
    parser.add_argument("--aug-preset", choices=PRESET_NAMES, default="all",
                        help="all: the reference augmentation plus every operation of the extended library "
                             "at probability 0.05 (default). heavy / medium / light: the reference plus the "
                             "library without the operations that can harm crack labels (table above). "
                             "reference: the reference augmentation only, as in the Swin and FiLM U-Net "
                             "projects. none: random crop only")
    parser.add_argument("--target-weight", type=float, default=1.5,
                        help="Sampling weight of target-domain training images relative to others")
    parser.add_argument("--scale-p", type=float, help=f"(reference default {defaults.scale_p})")
    parser.add_argument("--focus-p", type=float, help=f"(reference default {defaults.focus_p})")
    parser.add_argument("--copy-paste-p", type=float, help=f"(reference default {defaults.copy_paste_p})")
    parser.add_argument("--synthetic-p", type=float, help=f"(reference default {defaults.synthetic_p})")
    parser.add_argument("--fade-p", type=float, help=f"(reference default {defaults.fade_p})")
    parser.add_argument("--affine-p", type=float, help=f"(reference default {defaults.affine_p})")
    parser.add_argument("--elastic-p", type=float, help=f"(reference default {defaults.elastic_p})")
    parser.add_argument("--cutout-p", type=float, help=f"(reference default {defaults.cutout_p})")
    parser.add_argument("--max-rotation", type=float,
                        help=f"Largest rotation of the affine in degrees, up to 180 for a full rotation with "
                             f"reflect padding (reference default {defaults.max_rotation:g})")
    parser.add_argument("--synthetic-label-width", type=int,
                        help=f"(reference default {defaults.synthetic_label_width})")
    parser.add_argument("--contrast-mode", choices=("all", "random", "off"),
                        help="all: every training image is cycled through every contrast level from "
                             "-max to +max in --contrast-step steps, including the original (reference "
                             "default, 21 levels). random: a random change on a share --contrast-p of the "
                             "images. off: no contrast change (the 'none' preset)")
    parser.add_argument("--contrast-step", type=float,
                        help="Spacing of the contrast levels in 'all' mode (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-p", type=float,
                        help="'random' mode only: probability of a contrast change per training image")
    parser.add_argument("--contrast-min-change", type=float,
                        help="Smallest contrast change, as a fraction (default 0.01 = 1 %%)")
    parser.add_argument("--contrast-max-change", type=float,
                        help="Largest contrast change, as a fraction (default 0.10 = 10 %%); "
                             "each change is an increase or a decrease with equal probability")
    # Augmentation: the extended library, one probability (and magnitude) per registered operation.
    library = parser.add_argument_group(
        "extended augmentation library",
        "--<name>-p: probability per training image (default: the preset's, 0.05 in 'all'). "
        "--<name>-m: magnitude in [0, 1] scaling the range given below (default: the preset's).")
    for name, operation in AUGMENTATIONS.items():
        note = " [OFF in light/medium/heavy: harms crack labels]" if operation.default_off else ""
        library.add_argument(f"--{name}-p", type=float, metavar="P", help=f"{operation.stage}: {operation.help}{note}")
        if operation.scaled:
            library.add_argument(f"--{name}-m", type=float, metavar="M", help=argparse.SUPPRESS)
    parser.add_argument("--label-safety", action=argparse.BooleanOptionalAction, default=True,
                        help="After a contrast-reducing operation of the extended library, exclude cracks "
                             "that are no longer visible (below 4 grey levels of local contrast) from the loss")
    parser.add_argument("--randaugment", type=float, nargs=2, metavar=("N", "M"),
                        help="RandAugment: N random photometric / weather / blur / noise / compression "
                             "operations per image at magnitude M (0-10). Default: off")
    parser.add_argument("--preview-augmentations", type=int, default=0, metavar="K",
                        help="Before training, save one panel per enabled augmentation for K training tiles "
                             "to <output>/augmentation_preview")
    parser.add_argument("--preview-only", action="store_true", help="Exit after --preview-augmentations")
    # Jupyter injects ``-f <kernel.json>``. Because ``dataset`` is an optional
    # positional argument, remove the complete kernel pair before parsing.
    raw_arguments, filtered_arguments, index = sys.argv[1:], [], 0
    while index < len(raw_arguments):
        if raw_arguments[index] == "-f":
            index += 2
        else:
            filtered_arguments.append(raw_arguments[index])
            index += 1
    args = parser.parse_args(filtered_arguments)
    if min(args.decoder_channels) < 1 or args.half_channels < 1:
        parser.error("--decoder-channels and --half-channels must be positive.")
    if not 0 <= args.drop_path_rate < 1:
        parser.error("--drop-path-rate must lie in [0, 1).")
    if min(args.aux_weights) < 0:
        parser.error("--aux-weights must not be negative.")
    if args.image_size % 32:
        parser.error("--image-size must be divisible by 32 (the encoder halves the image five times).")
    if args.eval_tile % 32 or not 0 <= args.eval_tile_overlap < max(1, args.eval_tile):
        parser.error("--eval-tile must be divisible by 32 and larger than --eval-tile-overlap.")
    if args.eval_batch_megapixels <= 0:
        parser.error("--eval-batch-megapixels must be greater than zero.")
    if args.label_tolerance < 0:
        parser.error("--label-tolerance must not be negative.")
    if args.init_weights is not None and not args.evaluate_only and not args.init_weights.is_file():
        parser.error(f"--init-weights file not found: {args.init_weights}")
    if args.preview_only and args.preview_augmentations <= 0:
        parser.error("--preview-only needs --preview-augmentations K.")
    return args


def check_augmentation_arguments(args) -> None:
    """The reference's checks, on the resolved values."""
    if not 0 <= args.contrast_min_change <= args.contrast_max_change < 1:
        raise SystemExit("Contrast changes must satisfy 0 <= --contrast-min-change <= --contrast-max-change < 1.")
    if args.contrast_step <= 0:
        raise SystemExit("--contrast-step must be greater than zero.")
    if not 0 <= args.max_rotation <= 180:
        raise SystemExit("--max-rotation must lie in [0, 180] degrees.")


def main():
    args = parse_arguments()
    if args.self_test:
        from .self_test import run_self_tests  # here, because self_test imports this module

        run_self_tests()
        return
    options, augmentation = resolve_augmentation(args)
    check_augmentation_arguments(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    stored = None
    if args.evaluate_only:
        if not best_path.is_file():
            raise SystemExit(f"--evaluate-only needs a trained model, but {best_path} does not exist.")
        stored = torch.load(best_path, map_location="cpu", weights_only=False)

    # ---------------- Network ---------------- #
    architecture = {"name": "unetplusplus", "kwargs": {
        "encoder_name": args.encoder, "decoder_channels": list(args.decoder_channels),
        "decoder_attention": None if args.decoder_attention == "none" else args.decoder_attention,
        "decoder_interpolation": args.decoder_interpolation, "half_channels": args.half_channels,
        "stem_channels": 0 if args.no_hairline_stem else 32,
        "line_prior": not (args.no_line_prior or args.no_hairline_stem),
        "deep_supervision": not args.no_deep_supervision, "drop_path_rate": args.drop_path_rate,
    }}
    # Checked before anything is written, so that a run with the wrong options leaves the folder as it was.
    resume_state, initial = None, None
    if args.evaluate_only:  # rebuild exactly what was trained; its weights come from best.pt
        architecture = stored["architecture"]
    elif args.resume and last_path.is_file():
        resume_state = torch.load(last_path, map_location=device, weights_only=False)
        different = different_arguments(resume_state["architecture"]["kwargs"], architecture["kwargs"])
        if different:
            raise SystemExit(f"last.pt holds a different network ({different}); this run builds "
                             f"{ {key: architecture['kwargs'][key] for key in different} }. "
                             "Resume with the options of the first run.")
    elif args.init_weights is not None:
        initial = initial_weights(args.init_weights, architecture["kwargs"])
    kwargs = architecture["kwargs"]
    # Only a fresh run needs the pretrained encoder: every other one loads all of its weights below.
    fresh = not (args.evaluate_only or initial is not None or resume_state is not None)

    # ---------------- GSD table ---------------- #
    if args.gsd_csv is not None:
        gsd_table = GSDTable.from_csv(args.gsd_csv)
        print(f"GSD table: {len(gsd_table.rows)} pattern(s) from {args.gsd_csv}.")
    elif stored is not None and stored.get("gsd_csv"):
        gsd_table = GSDTable(stored["gsd_csv"])  # evaluate with the GSD values the model was trained with
        print(f"GSD table: {len(gsd_table.rows)} pattern(s) stored in {best_path.name}.")
    else:
        gsd_table = GSDTable()
    if not args.evaluate_only:
        (args.output / "configuration.json").write_text(json.dumps(json_safe(
            {**vars(args), "augmentation": augmentation, "gsd_csv_rows": gsd_table.rows}), indent=2))

    splits = {name: collect_pairs(args.dataset / name) for name in ("train", "val", "test")}
    subsample = random.Random(args.seed)
    if args.max_train_samples:
        splits["train"] = subsample.sample(splits["train"], min(args.max_train_samples, len(splits["train"])))
    if args.max_eval_samples:
        for name in ("val", "test"):
            splits[name] = subsample.sample(splits[name], min(args.max_eval_samples, len(splits[name])))
    families = {name: [source_family(image) for image, _ in pairs] for name, pairs in splits.items()}
    domains = sorted(set().union(*families.values()))
    domain_index = {domain: index for index, domain in enumerate(domains)}
    # The GSD of a tile is that of the first CSV pattern found in its image file name; NaN without one.
    gsds = {name: [gsd_table.lookup(image.name) for image, _ in pairs] for name, pairs in splits.items()}

    pattern = re.compile(args.select_pattern) if args.select_pattern else None
    target_domains = sorted({
        family for family, (image, _) in zip(families["val"], splits["val"])
        if pattern is None or pattern.search(image.name)
    })
    if not target_domains:
        print(f"No validation file matches --select-pattern {args.select_pattern!r}; using all validation data.")
        target_domains = sorted(set(families["val"]))
    target_ids = [domain_index[domain] for domain in target_domains]

    print_source_table(domains, families, gsds, target_domains)

    datasets = {
        name: CrackSegmentationDataset(
            splits[name], [domain_index[family] for family in families[name]], gsds[name],
            training=name == "train", crop_size=args.image_size, options=options,
        )
        for name in ("train", "val", "test")
    }
    train_set = datasets["train"]
    enabled = augmentation["enabled_operations"]
    print(f"Augmentation preset: {args.aug_preset} | extended operations enabled: {len(enabled)}"
          + (f" ({', '.join(enabled)})" if 0 < len(enabled) <= 12 else "")
          + (f" | RandAugment N={options.randaugment[0]} M={options.randaugment[1]:g}" if options.randaugment else "")
          + (f" | label safety: {'on' if options.label_safety else 'off'}" if enabled or options.randaugment else ""))
    if args.preview_augmentations > 0:
        preview_folder = args.output / "augmentation_preview"
        shown = save_augmentation_preview(train_set, preview_folder, args.preview_augmentations, args.seed)
        print(f"Augmentation preview: {len(shown)} augmentation(s) on "
              f"{min(args.preview_augmentations, len(train_set))} tile(s) in {preview_folder}.")
        if args.preview_only:
            return

    pin = device.type == "cuda"
    loader_options = dict(num_workers=args.workers, pin_memory=pin, worker_init_fn=seed_worker,
                          persistent_workers=args.workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    sample_weights = [args.target_weight if family in target_domains else 1.0 for family in families["train"]]
    if args.target_weight != 1.0 and len(set(sample_weights)) > 1:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        train_loader = DataLoader(train_set, args.batch_size, sampler=sampler, drop_last=True, **loader_options)
    else:
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                                  generator=generator, **loader_options)
    # Evaluation memory is bounded twice: a batch holds at most --eval-batch-megapixels,
    # and an image with a side above --eval-tile is run as overlapping tiles.
    batch_pixels = args.eval_batch_megapixels * 1e6
    tiling = dict(tile=args.eval_tile, overlap=args.eval_tile_overlap,
                  tile_batch=max(1, int(batch_pixels // max(1, args.eval_tile) ** 2)))
    val_loader, test_loader = (
        DataLoader(datasets[name], batch_sampler=shape_grouped_batches(
            datasets[name].shapes, args.eval_batch_size, max_pixels=batch_pixels), **loader_options)
        for name in ("val", "test")
    )
    # The threshold is calibrated on the target-domain validation images only,
    # so the final passes do not need the other validation sources.
    target_set = set(target_ids)
    calibration_loader = DataLoader(datasets["val"], batch_sampler=shape_grouped_batches(
        datasets["val"].shapes, args.eval_batch_size,
        [index for index, domain in enumerate(datasets["val"].domain_ids) if domain in target_set],
        max_pixels=batch_pixels,
    ), **loader_options)
    if len(train_loader) == 0:
        raise SystemExit("Training set is smaller than one batch; lower --batch-size.")

    model = ARCHITECTURES[architecture["name"]](pretrained=fresh and not args.no_pretrained, **kwargs)
    if initial is not None:
        keys = model.load_state_dict(initial["model"], strict=False)
        unfit = keys.missing_keys + keys.unexpected_keys
        if unfit:
            raise SystemExit(f"--init-weights does not fit this network: {len(unfit)} parameters differ, "
                             f"for example {unfit[:3]}.")
        print(f"Initial weights: {args.init_weights} (epoch {initial.get('epoch')}, "
              f"{initial.get('select_metric')} {initial.get('best_val_score', float('nan')):.4f}).")
        del initial
    model = model.to(device, memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        parameter_groups(model, args.learning_rate, args.encoder_lr_factor, args.weight_decay)
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(
        args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch), args.min_lr_ratio
    ))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.loss == "focal":
        loss_function = FocalCrackLoss(args.focal_alpha, args.focal_gamma, args.cldice_weight,
                                       args.skeleton_iterations, args.aux_weights, args.label_tolerance)
    else:
        loss_function = HairlineCrackLoss(
            args.bce_weight, args.tversky_weight, args.cldice_weight,
            args.false_positive_weight, args.false_negative_weight,
            args.skeleton_iterations, args.aux_weights, args.label_tolerance,
        )
    # 0.05 to 0.95 in steps of 0.025, and 0.01 steps beyond: on the first run the orthomosaic tiles scored
    # best at 0.05 and the benchmark photos at 0.95, the ends of the grid of the earlier trainers.
    thresholds = np.round(np.concatenate((np.arange(0.01, 0.045, 0.01), np.arange(0.05, 0.951, 0.025),
                                          np.arange(0.96, 0.995, 0.01))), 3)
    sweep = ThresholdSweep(thresholds, args.tolerance, len(domains), device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"Device: {device} | AMP: {use_amp} | parameters: {parameter_count:.1f} M | "
          f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")
    encoder_count = sum(parameter.numel() for parameter in model.encoder.parameters()) / 1e6
    weights = "weights from the checkpoint" if not fresh else "random" if args.no_pretrained else "pretrained"
    added = [name for name, present in (
        ("1/2-scale stem", model.half_resolution is not None),
        ("hairline stem" + (" with thin-line prior" if model.thin_lines is not None else ""),
         model.full_resolution is not None),
        ("deep supervision on X(0,1) to X(0,3)", model.auxiliary_heads is not None),
    ) if present]
    print(f"Network: UNet++ on {kwargs['encoder_name']} (encoder {encoder_count:.1f} M parameters, {weights}), "
          f"decoder channels {' '.join(str(channels) for channels in kwargs['decoder_channels'])}, "
          f"attention {kwargs['decoder_attention'] or 'none'}; "
          + (f"added to smp's network: {', '.join(added)}." if added else "smp.UnetPlusPlus, nothing added."))
    if train_set.contrast_factors is not None:
        factors = train_set.contrast_factors
        print(f"Contrast: all {len(factors)} levels from {100 * (factors[0] - 1):+.0f} % to "
              f"{100 * (factors[-1] - 1):+.0f} % in {100 * args.contrast_step:g} % steps; every training image "
              f"cycles through all of them (needs at least {len(factors)} epochs per image).")
    elif args.contrast_mode == "off" or not options.photometric:
        print("Contrast: unchanged.")
    else:
        print(f"Contrast: random change of {100 * args.contrast_min_change:g}-{100 * args.contrast_max_change:g} % "
              f"on {100 * args.contrast_p:g} % of the training images.")

    start_epoch, best_score, best_epoch, stale = 1, -1.0, 0, 0
    if resume_state is not None:
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
            ema.updates = state["ema_updates"]
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        draws = state.get("contrast_draws")
        if (draws is not None and train_set.contrast_factors is not None
                and draws[0].shape == train_set.image_draws.shape and draws[1].shape == train_set.level_draws.shape):
            train_set.image_draws.copy_(draws[0])  # continue each image's cycle where it stopped
            train_set.level_draws.copy_(draws[1])
        fired = state.get("augmentation_fired")
        if fired is not None and fired[0].shape == train_set.fired.shape:
            train_set.fired.copy_(fired[0])  # the counts cover the whole run, not only this session
            train_set.safety.copy_(fired[1])
        print(f"Resumed from epoch {state['epoch']} (best {args.select_metric} {best_score:.4f}).")

    # What best.pt needs, besides the weights, to rebuild the model and to document the run.
    run_record = {
        "architecture": architecture,
        "gsd_csv": [list(row) for row in gsd_table.rows], "augmentation": augmentation,
    }
    # History columns: train_*/val_* use the fixed threshold 0.5 so the two curves are
    # comparable; the unprefixed metrics are validation at that epoch's best threshold.
    # train_loss is the whole objective; train_output_loss is its part on the network output,
    # the terms val_loss has (no deep supervision).
    fields = ["epoch", "lr", "train_loss", "train_output_loss", "val_loss",
              *(f"train_{name}" for name in BASIC_METRICS), *(f"val_{name}" for name in BASIC_METRICS),
              "threshold", *METRIC_NAMES, "all_val_f1", "all_val_tol_f1", "seconds"]
    history_path = args.output / "training_history.csv"
    # One row per validation source and epoch, at that epoch's selected threshold; best_* is
    # the source's own optimum. The pooled history hides a source the model fails on.
    source_path = args.output / "validation_by_source.csv"
    source_fields = ["epoch", "source", "images", "target", "threshold", *METRIC_NAMES,
                     "best_threshold", f"best_{args.select_metric}"]
    val_sources, val_images = sorted(set(datasets["val"].domain_ids)), Counter(families["val"])
    labelled_column = COUNT_NAMES.index("labelled")
    half_index = int(np.argmin(np.abs(thresholds - 0.5)))
    if not args.evaluate_only:
        with open_epoch_log(history_path, fields, start_epoch) as handle, \
                open_epoch_log(source_path, source_fields, start_epoch) as source_handle:
            writer, source_writer = csv.DictWriter(handle, fieldnames=fields), csv.writer(source_handle)
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.time()
                model.train()
                cldice_scale = min(1.0, epoch / max(1, args.cldice_warmup_epochs))
                running_loss, output_loss, seen = 0.0, 0.0, 0
                train_counts = torch.zeros(4, dtype=torch.float64, device=device)  # tp, fp, fn, pixels
                for images, masks, valid, _, _, _ in train_loader:
                    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    masks, valid = masks.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        outputs = model(images)
                    loss = loss_function(outputs, masks, valid, cldice_scale)
                    if not torch.isfinite(loss):
                        print("Warning: non-finite loss; batch skipped.")
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                    running_loss += loss.item() * images.size(0)
                    output_loss += loss_function.output_term.item() * images.size(0)
                    seen += images.size(0)
                    with torch.no_grad():  # training metrics at threshold 0.5 (logit 0)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        supervised = valid > 0.5
                        prediction, truth = (logits.detach() >= 0) & supervised, (masks >= 0.5) & supervised
                        train_counts += torch.stack((
                            (prediction & truth).sum(), (prediction & ~truth).sum(),
                            (~prediction & truth).sum(), supervised.sum(),
                        )).double()

                evaluation_model = ema.module if ema is not None else model
                val_loss = evaluate(evaluation_model, val_loader, device, sweep, use_amp, loss_function, **tiling)
                target_metrics, all_metrics = sweep.metrics(target_ids), sweep.metrics()
                best_index = int(np.argmax(target_metrics[args.select_metric]))
                score = float(target_metrics[args.select_metric][best_index])
                train_metrics = basic_metrics(*train_counts.tolist())
                row = {
                    "epoch": epoch, "lr": max(group["lr"] for group in optimizer.param_groups),
                    "train_loss": running_loss / max(1, seen), "train_output_loss": output_loss / max(1, seen),
                    "val_loss": val_loss,
                    **{f"train_{name}": train_metrics[name] for name in BASIC_METRICS},
                    **{f"val_{name}": float(target_metrics[name][half_index]) for name in BASIC_METRICS},
                    "threshold": float(thresholds[best_index]), **at_threshold(target_metrics, best_index),
                    "all_val_f1": float(all_metrics["f1"].max()),
                    "all_val_tol_f1": float(all_metrics["tol_f1"].max()),
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                source_scores = []  # (score at the selected threshold, name) of target sources with labels
                for domain_id in val_sources:
                    name, source_counts = domains[domain_id], sweep.count_array([domain_id])
                    source_metrics = metrics_from_counts(source_counts)
                    own_index = int(np.argmax(source_metrics[args.select_metric]))
                    source_writer.writerow([
                        epoch, name, val_images[name], int(name in target_domains),
                        f"{thresholds[best_index]:.3f}",
                        *(f"{source_metrics[metric][best_index]:.5f}" for metric in METRIC_NAMES),
                        f"{thresholds[own_index]:.3f}", f"{source_metrics[args.select_metric][own_index]:.5f}",
                    ])
                    if name in target_domains and source_counts[0, labelled_column] > 0:
                        source_scores.append((float(source_metrics[args.select_metric][best_index]), name))
                source_handle.flush()
                improved = score > best_score
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | loss {row['train_loss']:.4f}/{val_loss:.4f} | "
                    f"thr {row['threshold']:.3f} P {row['precision']:.4f} R {row['recall']:.4f} "
                    f"F1 {row['f1']:.4f} IoU {row['iou']:.4f} | tolF1 {row['tol_f1']:.4f} | "
                    f"{row['seconds']:.0f}s{'  *best*' if improved else ''}"
                )
                if len(source_scores) > 1:
                    print(f"    weakest sources ({args.select_metric}): "
                          + " | ".join(f"{name} {value:.3f}" for value, name in sorted(source_scores)[:3]))
                if improved:
                    best_score, best_epoch, stale = score, epoch, 0
                    save_checkpoint({
                        "model": evaluation_model.state_dict(), **run_record,
                        "arguments": json_safe(vars(args)), "epoch": epoch,
                        "best_val_score": best_score, "select_metric": args.select_metric,
                        "optimal_threshold": float(thresholds[best_index]),
                    }, best_path)
                else:
                    stale += 1
                save_checkpoint({
                    "model": model.state_dict(), "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "stale": stale, "architecture": architecture,
                    "arguments": json_safe(vars(args)),
                    "contrast_draws": None if train_set.contrast_factors is None else
                    [train_set.image_draws.clone(), train_set.level_draws.clone()],
                    "augmentation_fired": [train_set.fired.clone(), train_set.safety.clone()],
                }, last_path)
                if stale >= args.patience:
                    print(f"Early stopping: no {args.select_metric} improvement for {args.patience} epochs.")
                    break
        coverage = train_set.contrast_coverage()
        if coverage is not None:
            (args.output / "contrast_coverage.json").write_text(json.dumps(coverage, indent=2))
            print(f"Contrast coverage: {coverage['images_trained_at_every_level']} of "
                  f"{coverage['training_images']} training images were trained at all {coverage['levels']} "
                  f"levels (draws per image: minimum {coverage['minimum_draws_per_image']}, "
                  f"mean {coverage['mean_draws_per_image']:.1f}).")
        counts = train_set.augmentation_counts()
        (args.output / "augmentation_counts.json").write_text(json.dumps(counts, indent=2))
        safety = counts["label_safety"]
        print(f"Augmentation counts: {counts['training_samples_drawn']} training samples drawn; label safety "
              f"excluded {safety['crack_pixels_excluded']} of {safety['crack_pixels_checked']} checked crack "
              f"pixels in {safety['checks']} checks.")

    # ---------------- Calibration on validation, then a single test pass ---------------- #
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"\nBest epoch {checkpoint['epoch']} ({args.select_metric} {checkpoint['best_val_score']:.4f}).")
    use_tta = not args.no_tta
    calibration, evaluation, source_indices = {}, {}, {}
    validation_images = {domain_index[name]: count for name, count in Counter(families["val"]).items()}
    for tta in sorted({False, use_tta}):
        evaluate(model, calibration_loader, device, sweep, use_amp, tta=tta, **tiling)
        validation_counts = sweep.count_array(target_ids)
        metrics = metrics_from_counts(validation_counts)
        index = int(np.argmax(metrics[args.select_metric]))
        stored_threshold = checkpoint.get("optimal_threshold_tta" if tta else "optimal_threshold")
        if args.evaluate_only and stored_threshold is not None:
            # Report at the threshold the training run calibrated (and the detector uses).
            # Recalibrating on another machine can move it by a grid step, because the
            # validation curve is flat near its maximum and CPU/GPU precision differs.
            index = int(np.argmin(np.abs(thresholds - stored_threshold)))
            print(f"Threshold stored in best.pt ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        else:
            print(f"Validation-calibrated threshold ({'flip TTA' if tta else 'no TTA'}): {thresholds[index]:.3f}")
        calibration[tta] = (index, metrics)
        evaluation["tta" if tta else "no_tta"] = {"threshold_index": index, "val": validation_counts}
        # Every source's own threshold, besides the shared one; evaluation only reuses those of best.pt.
        stored_sources = checkpoint.get("source_thresholds_tta" if tta else "source_thresholds")
        calibrated = not (args.evaluate_only and stored_sources)
        if not calibrated:
            source_indices[tta] = {domain_index[name]: int(np.argmin(np.abs(thresholds - value)))
                                   for name, value in stored_sources.items() if name in domain_index}
        else:
            source_indices[tta] = source_threshold_indices(sweep, target_ids, args.select_metric, validation_images,
                                                           args.source_threshold_min_images, index)
        print(f"{'Own threshold per source' if calibrated else 'Thresholds per source stored in best.pt'} "
              f"({'flip TTA' if tta else 'no TTA'}), validation {args.select_metric} at the shared "
              f"{thresholds[index]:.3f} -> at its own:")
        for domain, own in sorted(source_indices[tta].items(), key=lambda item: domains[item[0]]):
            source_metrics = metrics_from_counts(sweep.count_array([domain]))[args.select_metric]
            few = calibrated and validation_images.get(domain, 0) < args.source_threshold_min_images
            print(f"    {domains[domain]:28s} {source_metrics[index]:.3f} -> {source_metrics[own]:.3f} "
                  f"at {thresholds[own]:.3f}" + (" (too few images: the shared threshold)" if few else ""))
    if not args.evaluate_only:
        with (args.output / "val_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            sweep_writer = csv.writer(handle)
            sweep_writer.writerow(["tta", "threshold", *METRIC_NAMES])
            for tta, (_, metrics) in calibration.items():
                for index, threshold in enumerate(thresholds):
                    sweep_writer.writerow(
                        [tta, f"{threshold:.3f}", *(f"{metrics[name][index]:.5f}" for name in METRIC_NAMES)]
                    )
        checkpoint["optimal_threshold"] = float(thresholds[calibration[False][0]])
        checkpoint["optimal_threshold_tta"] = float(thresholds[calibration[use_tta][0]])
        checkpoint["threshold_metrics"] = {
            str(tta): {name: calibration[tta][1][name].tolist() for name in METRIC_NAMES} for tta in calibration
        }
        checkpoint["thresholds"] = thresholds.tolist()
        for key, tta in (("source_thresholds", False), ("source_thresholds_tta", use_tta)):
            checkpoint[key] = {domains[domain]: float(thresholds[own]) for domain, own in source_indices[tta].items()}
        save_checkpoint(checkpoint, best_path)

    results = {
        "best_epoch": checkpoint["epoch"], "select_metric": args.select_metric,
        "best_validation_score": checkpoint["best_val_score"], "tolerance_px": args.tolerance,
        "target_sources": target_domains,
    }
    # The same pixel counts as per source, grouped by the GSD of the test tiles instead.
    test_set = datasets["test"]
    gsd_labels, gsd_group_ids = gsd_groups(test_set.gsds)
    gsd_sweep = ThresholdSweep(thresholds, args.tolerance, len(gsd_labels), device)
    for tta in sorted(calibration):
        index, mode = calibration[tta][0], "tta" if tta else "no_tta"
        hook = None
        if tta == use_tta and args.save_predictions > 0 and not args.evaluate_only:
            hook = prediction_writer(
                test_set, args.output / "test_predictions", args.save_predictions, float(thresholds[index]))
        test_loss = evaluate(model, test_loader, device, sweep, use_amp, loss_function, tta=tta, hook=hook,
                             gsd_sweep=gsd_sweep, gsd_group_ids=torch.tensor(gsd_group_ids), **tiling)
        evaluation[mode]["test"] = sweep.count_array()
        per_source = {domains[d]: at_threshold(sweep.metrics([d]), index)
                      for d in sorted(set(test_set.domain_ids))}
        per_gsd = {gsd_labels[group]: {"images": gsd_group_ids.count(group),
                                       **at_threshold(gsd_sweep.metrics([group]), index)}
                   for group in sorted(set(gsd_group_ids))}
        # Each source at its own validation-calibrated threshold; a source without one at the shared threshold.
        own = {d: source_indices[tta].get(d, index) for d in sorted(set(test_set.domain_ids))}
        with_source_thresholds = {
            **at_threshold(metrics_from_counts(counts_at_indices(sweep, own)), 0),
            "per_source": {domains[d]: {"threshold": float(thresholds[i]), **at_threshold(sweep.metrics([d]), i)}
                           for d, i in own.items()},
        }
        results[mode] = {
            "threshold": float(thresholds[index]), "loss": test_loss,
            **at_threshold(sweep.metrics(), index), "per_source": per_source, "per_gsd": per_gsd,
            "with_source_thresholds": with_source_thresholds,
        }
    # An evaluation-only run never overwrites the training run's files.
    metrics_name = "test_metrics_evaluate_only.json" if args.evaluate_only else "test_metrics.json"
    (args.output / metrics_name).write_text(json.dumps(results, indent=2))
    print("Test:", json.dumps(results, indent=2))
    metric = args.select_metric
    for mode in ("no_tta", "tta"):
        if mode not in results:
            continue
        shared, own = results[mode], results[mode]["with_source_thresholds"]
        print(f"Test {metric} ({MODE_NAMES[mode]}), at the shared threshold {shared['threshold']:.3f} -> "
              f"at each source's own:")
        for name in sorted(own["per_source"]):
            print(f"    {name:28s} {shared['per_source'][name][metric]:.3f} -> {own['per_source'][name][metric]:.3f} "
                  f"at {own['per_source'][name]['threshold']:.3f}")
        print(f"    {'all sources pooled':28s} {shared[metric]:.3f} -> {own[metric]:.3f}")
    print(f"Test metrics in {args.output / metrics_name}.")

    # ---------------- Tables and figures ---------------- #
    plot_folder = args.output / "plots"
    save_evaluation_tables(plot_folder, thresholds, evaluation)
    try:
        written = save_plots(args.output, thresholds, evaluation, checkpoint["epoch"], args.tolerance)
        print(f"Figures in {plot_folder}: {', '.join(written)}")
    except ImportError:
        print("matplotlib is not installed, so no figures were drawn (python -m pip install matplotlib). "
              f"The curve and confusion-matrix tables are in {plot_folder}.")


__all__ = [
    'parameter_groups', 'warmup_cosine', 'MODE_NAMES', 'json_safe', 'save_checkpoint', 'open_epoch_log',
    'initial_weights', 'flag_name', 'REFERENCE_AUGMENTATION_ARGUMENTS',
    'resolve_augmentation', 'parse_arguments', 'check_augmentation_arguments', 'main',
]
