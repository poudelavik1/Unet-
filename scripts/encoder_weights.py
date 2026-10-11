"""Download the pretrained weights of an encoder into the local cache, or check that they are there.

Compute nodes cannot download, so the weights must be cached from the login node
(scripts/setup_conda.sh runs this). With HF_HUB_OFFLINE=1 in the environment nothing is
downloaded from the Hugging Face Hub: the call then succeeds only if the weights are cached,
which is how scripts/submit.sh and the jobs use it.
"""
import sys
import warnings


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/encoder_weights.py ENCODER   (for example tu-convnext_tiny, efficientnet-b4)")
    name = sys.argv[1]
    import segmentation_models_pytorch as smp

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder = smp.encoders.get_encoder(name, in_channels=3, depth=5, weights="imagenet")
    except Exception as exc:
        sys.exit(f"Pretrained {name} weights are not available: {type(exc).__name__}: {exc}")
    parameters = sum(parameter.numel() for parameter in encoder.parameters()) / 1e6
    print(f"Pretrained {name} weights are cached ({parameters:.1f} M parameters; "
          f"feature channels at 1/2 to 1/32 scale: {list(encoder.out_channels)[1:]}).")


if __name__ == "__main__":
    main()
