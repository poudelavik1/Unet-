# UNet++ hairline crack training on ASL

## Project layout

```text
configs/training.toml         Core training defaults
configs/gsd.csv               GSD (mm per pixel) of each source
src/unetpp/                   Augmentation, data, models, losses, evaluation, plots, training
scripts/                      Conda setup, encoder weights, submission, training CLI, epoch monitor
slurm/train_unetpp.sbatch     ASL CUDA training job
slurm/gpu_check.sbatch        Ten-minute GPU, environment and network check
docs/ARCHITECTURE.md          Encoder, nested decoder, stems and deep supervision
tests/                        Workflow, trainer and model checks
```

The workflow is that of the Swin U-Net, FiLM U-Net and multi-decoder U-Net
projects.

## Upload to GitHub from Windows

Repository: https://github.com/poudelavik1/Unet-
For future local updates, run from this project directory:

```powershell
git add .gitignore .gitattributes README.md requirements.txt environment.yml pyproject.toml configs src scripts slurm docs tests
git commit -m "Update training code"
git push origin main
```

Do not commit the dataset, ZIP, checkpoints, or credentials. A private repository
requires GitHub authentication when cloning (SSH key or credential helper).

## Server setup (once)

Connect to the ASL login node as for the Swin U-Net project. The repository is
named `Unet-`; it is cloned into `~/unetpp`, the folder name used below.

```bash
cd ~
[ -d unetpp ] || git clone https://github.com/poudelavik1/Unet-.git unetpp
cd unetpp
bash scripts/setup_conda.sh
```

Setup uses the `swinunet` Conda environment of the Swin U-Net project (Python
3.11, PyTorch 2.9.1 / torchvision 0.24.1 CUDA 12.6 wheels) and creates it if it
does not exist. It adds `segmentation-models-pytorch` 0.5.0 and `timm` to that
environment; the packages of the other projects stay as they are. Set
`ENV_NAME` to use another environment, and set the same `ENV_NAME` (or
`ENV_DIR`) when submitting. Jobs do not call `conda activate`, because
`/opt/miniconda3` exists only on the login node; they put
`~/.conda/envs/$ENV_NAME/bin` on `PATH` directly.

Setup also downloads the pretrained ConvNeXt-Tiny encoder into
`~/.cache/huggingface`, so compute nodes need no internet. Every other encoder
needs its own weights cached from the login node before it is used:

```bash
ENCODER=efficientnet-b4 bash scripts/setup_conda.sh
```

Jobs run with `HF_HUB_OFFLINE=1` and read the weights from that cache.

## Check the GPU (optional)

```bash
sbatch slurm/gpu_check.sbatch
squeue -u "$USER"
cat logs/gpu_check_JOBID.out logs/gpu_check_JOBID.err
```

The output shows the GPU, driver, PyTorch CUDA build and package versions,
loads the cached encoder weights, runs the network once on a 1024 x 1024 tile
in half precision and prints the GPU memory that took. It ends with
`GPU check passed.` Check another encoder with
`ENCODER=efficientnet-b4 sbatch --export=ALL slurm/gpu_check.sbatch`.

## Submit CUDA training

The dataset must contain `train/images`, `train/masks`, `val/images`,
`val/masks`, `test/images`, and `test/masks`.

```bash
bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

That trains the default network: UNet++ on ConvNeXt-Tiny with every
augmentation on (`all` preset). Core defaults live in `configs/training.toml`.
CLI flags take precedence. Direct training on an allocated compute node:

```bash
python scripts/train.py "$HOME/dataset_20261011/split_80_10_10" --output runs/manual --gsd-csv configs/gsd.csv
```

Default request: ASL-gpu, one GPU, four CPU cores, 32 GB host RAM, two days.
Request a shorter job with `SBATCH_TIMELIMIT=1-00:00:00` before the command.
Submission prints the job ID, result directory, and monitoring commands.
Each new job uses a separate output directory and does not resume old weights.
On ASL a second job of the same user has been removed from the queue while the first was
running, so submit the next run after the last one ends.

Variables in front of the command change the run:

| Variable | Effect | Default |
| --- | --- | --- |
| `EPOCHS`, `BATCH_SIZE`, `WORKERS` | Core settings | `configs/training.toml` (80, 12, 4) |
| `ENCODER` | Any smp encoder: `tu-convnext_tiny`, `tu-convnext_small`, `efficientnet-b4`, `efficientnet-b5`, ... | `tu-convnext_tiny` |
| `AUG_PRESET` | `all`, `heavy`, `medium`, `light`, `reference`, `none` | `all` |
| `LOSS` | `hairline` (BCE + Tversky + clDice) or `focal` | `hairline` |
| `TRAIN_ARGS` | Further trainer options, in one quoted string | none |
| `GSD_CSV` | GSD table (data pipeline and reports only) | `configs/gsd.csv` |
| `SELECT_PATTERN`, `TARGET_WEIGHT` | Select the model and threshold on one source | all sources |
| `INIT_WEIGHTS`, `LEARNING_RATE` | Fine-tune an earlier `best.pt` of this trainer | ImageNet encoder |
| `LABEL_TOLERANCE` | Pixels around each label left out of the loss | 0 |
| `SAVE_PREDICTIONS` | Test overlays to save | 40 |
| `EVALUATE_ONLY` | `1`: score the `best.pt` of an earlier run again, no training (below) | `0` |

The two encoders are two runs, one after the other, with the same augmentation:

```bash
bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

```bash
ENCODER=efficientnet-b4 bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

`TRAIN_ARGS` passes any option of `python scripts/train.py --help` to the job.
The unmodified network of smp (no hairline stem, no deep supervision) on
EfficientNet-B4 is:

```bash
ENCODER=efficientnet-b4 TRAIN_ARGS="--no-hairline-stem --no-deep-supervision" bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

To compare with the Swin and FiLM U-Net runs, which used the reference
augmentation, add `AUG_PRESET=reference`.

`AUG_PRESET=all` (the default) applies each of the 63 operations of the
extended library with probability 0.05 on top of the reference augmentation.
Seven of them can harm crack labels; `AUG_PRESET=heavy` is the strongest preset
without those seven.

By default the best checkpoint and the threshold are chosen on the validation
images of every source pooled by pixel, which the thick-crack sources dominate.
Every source family also gets its own threshold, calibrated on its validation
images, and the test metrics are reported at both (see
[the thresholds](ARCHITECTURE.md#thresholds)).
`SELECT_PATTERN` (a regex on validation file names) chooses them on a target
source instead and samples that source's training images `TARGET_WEIGHT` times
as often (default 1.5). For the row/column tiles:

```bash
SELECT_PATTERN='_r[0-9]+_c[0-9]+' bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

Only use `SLURM_ACCOUNT` / `SLURM_PARTITION` values confirmed by the
administrator. A pending job can show `InvalidAccount` on this server and still
start on its own. Pending jobs produce no epoch metrics until they start.

The first run (ConvNeXt-Tiny, 12,736 training tiles, batch 12) took 210 s per
epoch on an RTX A6000, 4.7 hours for 80 epochs. Its memory use was not logged;
`slurm/gpu_check.sbatch` prints the memory of one evaluation tile. Reduce the
batch size if GPU memory is exhausted (`BATCH_SIZE=8`); if evaluation exhausts
memory, reduce the tile (`TRAIN_ARGS="--eval-tile 768"`). Early stopping
defaults to 15 epochs without improvement, so training can stop before the
requested epoch count.

## The GSD table

`configs/gsd.csv` gives the GSD of each source in mm per pixel:

```text
pattern,gsd_mm_per_px
^EH104_,0.6511
^RAMA8_EAST-,2.5
```

`pattern` is a regular expression searched in the image file name. The first
matching row gives the GSD, and a file without a match (CRACK500, the photos)
has an unknown GSD. Write `[0-9]` rather than `\d`: the submission script
checks the patterns with `grep -E` and the trainer applies them as Python
regular expressions.

The network does not use the GSD. The table serves the data pipeline: the
mixing augmentations (mosaic, CutMix, MixUp, copy-paste) only combine tiles of
one source and one GSD, and the test metrics are reported per GSD value.
Submission prints how many training images each pattern matches.
`GSD_CSV=/absolute/path.csv` uses another table.

## Monitor performance

Use the actual job ID and output path printed by the submission script:

```bash
squeue -u "$USER"
tail -F logs/unetpp_JOBID.out
python scripts/monitor_epochs.py /absolute/path/to/run --watch
cat logs/unetpp_JOBID.err
```

Ctrl+C stops the viewer; the submitted job keeps running after SSH disconnects.
The monitor prints a row only after training AND validation finish an epoch.
All epoch values remain in `training_history.csv`, and `validation_by_source.csv`
holds the same metrics for every source separately (the log prints the three
weakest after each epoch).

At the start the log prints the source table, the number of enabled
augmentations and the network with what was added to smp's. After training,
the run directory holds:

- `best.pt`, resumable `last.pt`, `configuration.json`, `training_history.csv`,
  `validation_by_source.csv`, `val_threshold_sweep.csv`
- `test_metrics.json`: pooled, per source and per GSD value at the shared
  threshold, and under `with_source_thresholds` with every source at its own
- `test_predictions/`: for each saved test image the probability map and the
  overlay (yellow = hit, red = false positive, green = missed)
- `augmentation_counts.json`, `contrast_coverage.json`, `plots/`, and the Git
  commit and package versions of the job

## Score an earlier run again

A run trained before the per-source thresholds existed is scored with them,
without training, by an evaluation-only job on its run directory:

```bash
EVALUATE_ONLY=1 GSD_CSV="$HOME/gsd_20261011.csv" bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10" "$HOME/unetpp/runs/20261011_212847_430380"
```

It writes `test_metrics_evaluate_only.json` and the log prints, for every
source, the validation and test score at the shared threshold and at its own.
`best.pt`, `test_metrics.json` and the history stay as they were; the figures
in `plots/` are drawn again.

## Resume and update

Resume into the same run directory using its checkpoint:

```bash
RESUME=1 bash scripts/submit.sh /absolute/path/to/split_80_10_10 /absolute/path/to/existing/run
```

Pass the same variables as in the first submission (`ENCODER` and `TRAIN_ARGS`
included): a resume with another network stops before it changes the run
directory. Never submit two jobs writing into the same output directory. Cancel
a job using `scancel JOBID`. Update the repo only after active jobs finish:

```bash
git pull --ff-only
```

Retrieve results with `scp -r` from the server, as for the Swin U-Net project.

ASL instructions: https://asl.ait.ac.th/package-slurm.html
Slurm examples: https://asl.ait.ac.th/package-script.html
PyTorch wheels: https://pytorch.org/get-started/previous-versions/
