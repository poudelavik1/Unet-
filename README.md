# UNet++ hairline crack segmentation

Training for thin and faint concrete cracks with UNet++ on a stronger encoder:
ConvNeXt or EfficientNet, both through
[segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
(smp). Native-resolution crops, BCE + Tversky + clDice, tolerant crack metrics,
and every augmentation of the extended library switched on by default.

The project is organised and run like the
[Swin U-Net](https://github.com/poudelavik1/swinunet),
[FiLM U-Net](https://github.com/poudelavik1/filmunet) and
[multi-decoder U-Net](https://github.com/poudelavik1/addunet) projects and
shares their data pipeline, augmentation library, loss, metrics and figures, so
the runs can be compared directly. Only the network is new.

```text
configs/             Training defaults and the GSD table
src/unetpp/          Augmentation, data, model, loss, evaluation, plotting, training modules
scripts/             Conda setup, encoder weights, training CLI, submission, epoch monitoring
slurm/               ASL GPU check and training jobs
tests/               Workflow, trainer and model checks
docs/                Architecture and server guide
```

- [Architecture: encoder, nested decoder, stems and deep supervision](docs/ARCHITECTURE.md)
- [ASL setup, submission, monitoring, and resume](docs/ASL_SETUP.md)
- [Training defaults](configs/training.toml) and [GSD table](configs/gsd.csv)

## The network in short

| Encoder (`--encoder`, `ENCODER=` on the server) | Parameters | Notes |
| --- | --- | --- |
| `tu-convnext_tiny` (default) | 35.1 M | ConvNeXt has no 1/2-scale features, so a small stem supplies them |
| `efficientnet-b4` | 20.9 M | smp's UNet++ as it is, plus the hairline stem and deep supervision |

Any other smp encoder name works too: `tu-convnext_small`, `tu-convnextv2_tiny`,
`efficientnet-b3`, `efficientnet-b5`, `tu-tf_efficientnetv2_s`, or `resnet34`
for a UNet++ on the encoder of the earlier U-Net.

The encoder, the nested UNet++ decoder (with scSE attention) and the output
convolution are smp's. Three things are added, each with a switch:

- A 1/2-scale stem for ConvNeXt only. `smp.UnetPlusPlus` fails on ConvNeXt,
  which has no features at that scale.
- The hairline stem of the other trainers: full-resolution features with a
  thin-line prior as the skip of the last decoder node (`--no-hairline-stem`).
- Deep supervision on the intermediate UNet++ nodes (`--no-deep-supervision`).

With EfficientNet and the last two switched off, the network is
`smp.UnetPlusPlus` itself, weight for weight.

`--loss` takes one or more loss terms (BCE, focal, Dice, Tversky, clDice) and
`--optimizer` AdamW or Adam. `scripts/submit_ablation.sh` trains every
combination of the two optimisers and five losses on the server, and
`scripts/compare_runs.py` draws all runs in one set of comparison figures
([the study](docs/ARCHITECTURE.md#optimiser-and-loss-study)).

Every source family also gets its own validation-calibrated threshold next to
the shared one: on the first run the facade tiles scored best near 0.05 while
the shared threshold was 0.875. Both are reported and stored in `best.pt`
([details](docs/ARCHITECTURE.md#thresholds)).

## Run locally

```bash
python -m pip install -r requirements.txt
```

```bash
python scripts/train.py --self-test
```

```bash
python scripts/train.py "E:\STRUCTURAL ENGINEERING\thesis\crackdetection\DATASETS\split_80_10_10" --output runs/local --gsd-csv configs/gsd.csv
```

`python scripts/train.py --help` lists every option, including one
`--<name>-p` probability per augmentation.

## First time on the server

```bash
cd ~
[ -d unetpp ] || git clone https://github.com/poudelavik1/Unet-.git unetpp
cd unetpp
bash scripts/setup_conda.sh
```

Setup reuses the `swinunet` Conda environment of the Swin U-Net project, adds
`segmentation-models-pytorch` and `timm` to it, and caches the pretrained
ConvNeXt-Tiny encoder, which the compute nodes cannot download. Cache another
encoder the same way before using it:

```bash
ENCODER=efficientnet-b4 bash scripts/setup_conda.sh
```

## Train on the server

```bash
cd ~/unetpp
git pull --ff-only
bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

Submission checks the dataset, the GSD table and the cached encoder weights
before queueing, then prints commands for following the log and epoch metrics.
A variable in front of the command changes the run, for example the
EfficientNet-B4 encoder:

```bash
ENCODER=efficientnet-b4 bash scripts/submit.sh "$HOME/dataset_20261011/split_80_10_10"
```

All variables are listed in the [server guide](docs/ASL_SETUP.md).

Run the tests with `python -m unittest discover -s tests -v`.
