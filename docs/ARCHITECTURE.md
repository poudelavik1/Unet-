# UNet++ hairline crack segmentation architecture

Organization follows the Swin U-Net, FiLM U-Net and multi-decoder U-Net
projects (configs, src, scripts, slurm, docs). The data pipeline, augmentation
library, loss, metrics, figures and training loop are those of the other
hairline trainers; the network is new. It is not conditioned on the ground
sampling distance.

| Module | Responsibility |
| --- | --- |
| `src/unetpp/common.py` | Shared imports, constants (default encoder), default paths, random seeds |
| `src/unetpp/gsd.py` | GSD table (`pattern,gsd_mm_per_px`) and per-GSD groups, for the data pipeline and the reports |
| `src/unetpp/augmentations.py` | Reference augmentation, extended library, label safety, presets |
| `src/unetpp/data.py` | Image/mask pairing, dataset with GSD bookkeeping, augmentation preview, batching |
| `src/unetpp/models.py` | `UNetPlusPlus` (smp encoder and nested decoder, stems, deep supervision), EMA, checkpoint reconstruction |
| `src/unetpp/losses.py` | The loss terms (BCE, focal, Dice, Tversky, clDice) and their weighted sum, with deep-supervision terms |
| `src/unetpp/evaluation.py` | Threshold sweeps, tiled inference, tolerant metrics, predictions |
| `src/unetpp/plots.py` | Epoch curves and final evaluation figures/tables |
| `src/unetpp/comparison.py` | Figures and tables comparing finished runs (the optimiser and loss study) |
| `src/unetpp/training.py` | Argument parsing, optimizer/scheduler, training/validation, resume |
| `src/unetpp/self_test.py` | The trainer's built-in checks (`--self-test`) |
| `scripts/train.py` | TOML defaults and training CLI |
| `scripts/encoder_weights.py` | Cache the pretrained encoder, or check that it is cached |
| `scripts/submit_ablation.sh` | Train every optimiser and loss combination, one job after another |
| `scripts/compare_runs.py` | Draw finished runs in one set of comparison figures |
| `slurm/train_unetpp.sbatch` | ASL GPU allocation and execution |

`augmentations.py`, `data.py` and `gsd.py` are unchanged copies of the
multi-decoder U-Net project's files.

## The network

```text
image ──┬─ encoder (ConvNeXt or EfficientNet) ── features at 1/2*, 1/4, 1/8, 1/16, 1/32
        ├─ 1/2-scale stem, ConvNeXt only ─────── * the 1/2-scale features ConvNeXt lacks
        └─ hairline stem (black-hat / top-hat) ── features at 1/1

 scale   encoder     nested UNet++ decoder (smp)
 1/2     feature ──▶ X(0,1) ──▶ X(0,2) ──▶ X(0,3) ──▶ X(0,4) ──▶ last node 1/1 ──▶ 3x3 conv ──▶ crack
 1/4     feature ──▶ X(1,1) ──▶ X(1,2) ──▶ X(1,3) ─┘                ▲
 1/8     feature ──▶ X(2,1) ──▶ X(2,2) ─┘                    hairline stem
 1/16    feature ──▶ X(3,1) ─┘
 1/32    feature ─┘
                     └── 1x1 outputs on X(0,1), X(0,2), X(0,3): deep supervision
```

Every node X(i,j) upsamples the node below it and concatenates all earlier
nodes of its own row, which is what distinguishes UNet++ from U-Net: the skip
connections pass through convolutions instead of being copied.

- **Encoder** (`--encoder`): any encoder of
  [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch),
  ImageNet-pretrained. `tu-<name>` is a timm model. The default is
  `tu-convnext_tiny` (27.8 M parameters); `efficientnet-b4` (17.5 M) is the
  EfficientNet to run next to it. The image is normalised inside the model
  with the statistics the encoder was pretrained with, so the model is called
  with RGB in [0, 1].
- **Decoder**: smp's `UnetPlusPlusDecoder`, 256, 128, 64, 32 and 16 channels
  from 1/16 to 1/1 scale, scSE attention in every node, then smp's 3x3 output
  convolution. Built and initialised as `smp.UnetPlusPlus` builds it.
- **1/2-scale stem, ConvNeXt only.** ConvNeXt downsamples four times in its
  first layer and has no features at 1/2 scale. smp marks that scale with a
  feature of zero channels, and the forward pass of `smp.UnetPlusPlus` fails on
  it. Two 3x3 convolutions on the image, the first with stride 2
  (`--half-channels`, 48), trained from scratch, supply the top row of the
  decoder. EfficientNet has 1/2-scale features of its own and gets no stem.
- **Hairline stem**, as in the other trainers. In smp the last node upsamples
  from 1/2 scale without a skip connection. Here a full-resolution stem (two
  3x3 convolutions on the image and on fixed black-hat and top-hat responses)
  is the skip of that node. `--no-line-prior` drops the thin-line channels,
  `--no-hairline-stem` the whole stem.
- **Deep supervision**, as in the UNet++ paper: a 1x1 output on each of
  X(0,1), X(0,2) and X(0,3), upsampled to full size and trained on the crack
  mask. `--no-deep-supervision` removes them. They are used in training only.

With an EfficientNet encoder, `--no-hairline-stem` and `--no-deep-supervision`
the network is `smp.UnetPlusPlus` itself; the tests load its weights into smp's
model and compare the outputs.

| Network | Parameters |
| --- | --- |
| UNet++ on `tu-convnext_tiny`, default | 35.1 M |
| UNet++ on `efficientnet-b4`, default | 20.9 M |
| Multi-decoder U-Net on ResNet-34, three decoders | 31.0 M |

## Loss

```text
loss = L(output, mask)                                    the --loss terms
     + sum over X(0,1..3) of  w_k * R(deep output k, mask)   its region terms (all but clDice)
```

`--loss` names one or more terms, added with their `--<term>-weight`:

| Term | What it measures | Weight |
| --- | --- | --- |
| `bce` | Binary cross-entropy per pixel | 1 |
| `focal` | Binary focal loss: cross-entropy turned down on pixels already right (alpha 0.25, gamma 2) | 1 |
| `dice` | 1 - soft Dice over the batch | 1 |
| `tversky` | 1 - Tversky index, misses weighted 0.6 and false positives 0.4 | 1 |
| `cldice` | Soft clDice: continuity of the crack centreline, ramped in over 5 epochs | 0.5 |

The default, `hairline`, is BCE + Tversky + clDice, the loss of the other
projects. `w_k` is `--aux-weights` (0.2 for each). clDice alone is not a
usable loss: a prediction of crack everywhere has an empty soft skeleton and
scores 0, so it only makes sense next to a region term.

`training_history.csv` records the whole objective (`train_loss`) and the part
of the network output (`train_output_loss`, the terms `val_loss` has).

## Thresholds

The first run (ConvNeXt-Tiny, the 12,736-tile dataset) showed that one
threshold cannot serve every source. The shared threshold, calibrated on all
validation images pooled by pixel, came out at 0.875, because the benchmark
photos hold most of the labelled crack pixels. The orthomosaic tiles are 71 %
of the training tiles but hold only 21 % of their labelled crack pixels; their cracks are faint
and the network is less sure of them, and on validation they scored best at
0.05, the bottom of the grid:

| Validation source | Tolerant F1 at 0.875 | At its own threshold |
| --- | --- | --- |
| RAMA8 faces 3 to 8 (`f#_r#_c#`, 736 images) | 0.284 | 0.350 at 0.05 |
| RAMA8 west (`rama#_west-...`, 94) | 0.235 | 0.329 at 0.05 |
| RAMA8 east (`rama#_east-...`, 46) | 0.265 | 0.315 at 0.05 |
| EH104 (`eh#_r#_c#`, 299) | 0.698 | 0.707 at 0.75 |
| Benchmark and close-range sets | 0.88 to 0.99 | at most 0.003 higher, 0.825 to 0.95 |

So the trainer now:

- searches thresholds from 0.01 to 0.99 (0.05 to 0.95 in steps of 0.025, and
  steps of 0.01 beyond);
- besides the shared threshold, calibrates one threshold per source family on
  validation, for every family with at least `--source-threshold-min-images`
  (10) validation images; a smaller family keeps the shared threshold;
- reports the test metrics both ways in `test_metrics.json`: at the shared
  threshold as before (`per_source`, `per_gsd`, the pooled values), and under
  `with_source_thresholds` with every source at its own threshold, pooled and
  per source;
- stores the thresholds in `best.pt` (`optimal_threshold`,
  `source_thresholds` and their `_tta` variants), so a detector run on a
  facade can use the facade's threshold instead of 0.875.

The thresholds are chosen on validation only, so the test numbers under
`with_source_thresholds` are as fair as the shared ones. The model and the
training are unchanged by this: an earlier run is scored the new way with
`--evaluate-only` (`EVALUATE_ONLY=1` on the server), which writes
`test_metrics_evaluate_only.json` and leaves the run's other files as they are.

## Optimiser and loss study

To show what the optimiser and the loss contribute, `scripts/submit_ablation.sh`
trains one run per combination, all with the same network, data, augmentation
and schedule:

| | BCE | Focal | Dice | Tversky | BCE + Tversky + clDice |
| --- | --- | --- | --- | --- | --- |
| AdamW | run | run | run | run | run (the default) |
| Adam | run | run | run | run | run |

AdamW decays the weights separately from the gradient step; Adam adds the
weight decay (1e-4) to the gradient as L2 regularisation. Both use the same
learning rates. Each run trains on the loss named in its column alone, on the
network output and on the deep-supervision outputs.

`scripts/compare_runs.py` reads the finished runs and writes, to
`runs/ablation/comparison/`:

- `comparison_test_metrics.png`: tolerant F1, F1 and IoU on the test set, one
  row per loss and one dot per optimiser (AdamW blue, Adam orange);
- `comparison_heatmap.png`: the same runs as an optimiser x loss grid of
  tolerant F1 and F1, every value written in its cell;
- `comparison_validation_curves.png`: validation tolerant F1 per epoch, one
  panel per loss, both optimisers in each;
- `comparison_precision_recall.png`: tolerant precision against recall, with
  curves of equal tolerant F1, which shows what each loss trades between the two;
- `comparison_summary.csv` (every metric, the tolerant F1 at the per-source
  thresholds and per source) and `comparison_summary.md` (a table to paste).

Every run is evaluated at its own validation-calibrated threshold, so a loss
that gives lower probabilities is not penalised by a threshold set for another.
The loss values themselves are not compared: the terms have different scales.

## Training settings

The settings are those of the other CNN trainers, so that the runs differ in
the network only: AdamW, learning rate 3e-4 with the pretrained encoder at a
tenth of it, weight decay 1e-4, two warm-up epochs and cosine decay, EMA
weights, mixed precision, gradient clipping at 1, crops of 256 px, batch 12,
80 epochs with early stopping after 15 epochs without improvement. The stems,
the decoder and the outputs train at the full rate. `--drop-path-rate` (0.1)
is the stochastic depth of a timm encoder; smp's own EfficientNets have theirs
built in.

## Augmentation

The augmentation library of the earlier trainers is used whole: the reference
augmentation plus 63 registered operations (mixing, synthetic cracks and distractors,
rescaling, warps, orthomosaic artefacts, photometric, weather, blur, noise,
compression, occlusion), RandAugment, the label-safety check and the
augmentation preview. The default preset is `all`: every operation at
probability 0.05 and magnitude 0.75 on top of the reference augmentation.

Seven of the operations (mixup, piecewise-affine, ghosting, solarize, rain,
zoom-blur, glass-blur) can harm crack labels and are left out of the `light`,
`medium` and `heavy` presets; `all` includes them. `--aug-preset heavy` is the
strongest preset without them, and `--aug-preset reference` is the
augmentation the Swin and FiLM U-Net runs used. The table of probabilities is
in `python scripts/train.py --help`.

## The GSD table

The network has no GSD input. `configs/gsd.csv` (the pixel size of each
source) is used by the data pipeline only: the mixing augmentations (mosaic,
CutMix, MixUp, copy-paste) combine tiles of one source and one GSD, the
augmentation preview shows the GSD of each tile, and the test metrics are also
reported per GSD value. Without `--gsd-csv` every tile counts as unknown and
training is otherwise the same.

## Checkpoints

`last.pt` and `best.pt` are written to a temporary file and renamed. `best.pt`
stores the architecture name and arguments (encoder, decoder options, stems,
deep supervision), the GSD table, the augmentation configuration and the
shared and per-source thresholds, and
`unetpp.models.load_model_from_checkpoint` rebuilds the network from them
without downloading anything. `--init-weights` and `--resume` need the same
network as the checkpoint and stop with the differing options otherwise.

The dataset stays outside the repository and is supplied by absolute path.
Each fresh job writes into its own `runs/` directory. Resume uses the same
directory with `RESUME=1`. Do not run concurrent jobs into one output folder.

## Inference

There is no full-orthomosaic inference for this model yet. A model loaded with
`load_model_from_checkpoint` is called as `model(images)` with RGB in [0, 1]
and a height and width that are multiples of 32, and returns the crack logits;
`checkpoint["optimal_threshold"]` is the threshold calibrated on validation,
and `checkpoint["source_thresholds"]` maps every source family (for example
`f#_r#_c#`, the RAMA8 face tiles) to its own.
