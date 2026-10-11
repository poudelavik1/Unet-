"""Validate the trainer's self-test, the UNet++ network and loss, and the server additions
(tolerance ring, crash-safe checkpoints, resumed epoch logs, saved overlays, initial weights)."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    import segmentation_models_pytorch as smp
    from unetpp.evaluation import prediction_writer, spread_over_sources
    from unetpp.losses import FocalCrackLoss, HairlineCrackLoss
    from unetpp.models import UNetPlusPlus, load_model_from_checkpoint
    from unetpp.self_test import run_self_tests
    from unetpp.training import initial_weights, open_epoch_log, parameter_groups, save_checkpoint
except ImportError:  # the workflow tests still run without PyTorch and segmentation_models_pytorch
    torch = None


@unittest.skipIf(torch is None, 'PyTorch or segmentation_models_pytorch is not installed')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.zeros(1, 1, 32, 32)
        self.target[0, 0, 16, 4:28] = 1  # a one-pixel horizontal crack on row 16
        self.valid = torch.ones_like(self.target)

    def gradient(self, tolerance, loss=None):
        logits = torch.randn(1, 1, 32, 32, requires_grad=True)
        (loss or HairlineCrackLoss)(label_tolerance=tolerance)(logits, self.target, self.valid).backward()
        return logits.grad[0, 0]

    def test_built_in_self_test_passes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_self_tests()  # raises SystemExit with the failed checks
        self.assertIn('All self-tests passed.', output.getvalue())

    def test_strict_loss_supervises_pixels_beside_the_label(self):
        self.assertTrue((self.gradient(0)[14:19, 4:28] != 0).all())

    def test_tolerance_ring_is_left_out_of_the_loss(self):
        for loss in (HairlineCrackLoss, FocalCrackLoss):
            gradient = self.gradient(2, loss)
            self.assertTrue((gradient[[14, 15, 17, 18], 4:28] == 0).all())  # within 2 px of the label
            self.assertTrue((gradient[16, 4:28] != 0).all())                # the label itself
            self.assertTrue((gradient[[13, 19], 4:28] != 0).all())          # 3 px away: background again

    def test_checkpoint_is_replaced_whole(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            save_checkpoint({'epoch': 1}, path)
            save_checkpoint({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 2)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ['last.pt'])

    def test_resumed_log_drops_the_repeated_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training_history.csv'
            # Killed after logging epoch 3 but before saving last.pt, so epoch 3 runs again.
            path.write_text('epoch,loss\n1,0.9\n2,0.8\n3,0.7\n', encoding='utf-8')
            with open_epoch_log(path, ['epoch', 'loss'], 3) as handle:
                handle.write('3,0.6\n')
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss', '1,0.9', '2,0.8', '3,0.6'])
            with open_epoch_log(path, ['epoch', 'loss'], 1):  # a fresh run starts the file again
                pass
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss'])

    def test_saved_overlays_cover_every_source(self):
        domain_ids = [0] * 300 + [1] * 50 + [2] * 3  # one large source first, as in the size-grouped loader
        chosen = spread_over_sources(domain_ids, 40)
        per_source = [sum(domain_ids[index] == domain for index in chosen) for domain in range(3)]
        self.assertEqual(per_source, [19, 18, 3])  # the small source is saved whole, the rest shared
        self.assertGreater(max(index for index in chosen if domain_ids[index] == 0), 250)  # not just the first files
        self.assertEqual(spread_over_sources(domain_ids, 10**6), set(range(len(domain_ids))))  # a large limit saves all

    def test_overlays_are_written_for_the_selected_images_only(self):
        class Tiles:  # the parts of CrackSegmentationDataset the writer reads
            domain_ids = [0, 0, 0, 1]
            shapes = [(32, 32)] * 4
            pairs = [(Path(f'source{domain}_{index}.png'), None) for index, domain in enumerate(domain_ids)]

        with tempfile.TemporaryDirectory() as folder:
            write = prediction_writer(Tiles, Path(folder), 2, 0.5)
            write(torch.rand(4, 3, 32, 32), torch.rand(4, 1, 32, 32), self.target.expand(4, -1, -1, -1),
                  torch.arange(4))
            names = sorted(item.name for item in Path(folder).iterdir())
            self.assertEqual(names, ['source0_0_overlay.png', 'source0_0_probability.png',
                                     'source1_3_overlay.png', 'source1_3_probability.png'])


@unittest.skipIf(torch is None, 'PyTorch or segmentation_models_pytorch is not installed')
class UNetPlusPlusTests(unittest.TestCase):
    # The smallest encoder of each family; the arguments are those the trainer stores in a checkpoint.
    convnext, efficientnet = 'tu-convnext_atto', 'efficientnet-b0'
    kwargs = dict(encoder_name=convnext, decoder_channels=[256, 128, 64, 32, 16], decoder_attention='scse',
                  decoder_interpolation='nearest', half_channels=48, stem_channels=32, line_prior=True,
                  deep_supervision=True, drop_path_rate=0.1)

    @staticmethod
    def build(**changes):
        torch.manual_seed(0)
        return UNetPlusPlus(pretrained=False, **{**UNetPlusPlusTests.kwargs, **changes})

    def test_without_the_additions_it_is_the_network_of_smp(self):
        images = torch.rand(2, 3, 64, 96)
        for attention, interpolation in (('scse', 'nearest'), (None, 'bilinear')):
            model = self.build(encoder_name=self.efficientnet, stem_channels=0, deep_supervision=False,
                               decoder_attention=attention, decoder_interpolation=interpolation).eval()
            reference = smp.UnetPlusPlus(self.efficientnet, encoder_weights=None, decoder_attention_type=attention,
                                         decoder_interpolation=interpolation).eval()
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()),
                             sum(parameter.numel() for parameter in reference.parameters()))
            reference.encoder.load_state_dict(model.encoder.state_dict())
            reference.decoder.load_state_dict(model.decoder.state_dict())
            reference.segmentation_head.load_state_dict(model.head.state_dict())
            with torch.no_grad():
                self.assertTrue(torch.equal(model(images), reference((images - model.image_mean) / model.image_std)))

    def test_convnext_gets_the_half_scale_features_it_lacks(self):
        model = self.build().eval()
        self.assertEqual(list(model.encoder.out_channels)[1], 0)   # smp's mark of the missing 1/2 scale
        self.assertIsNotNone(model.half_resolution)
        self.assertEqual(model.decoder.skip_channels[3], 48)       # the stem's channels are the 1/2-scale skip
        self.assertIsNone(self.build(encoder_name=self.efficientnet).half_resolution)
        with torch.no_grad():
            self.assertEqual(model(torch.rand(1, 3, 64, 96)).shape, (1, 1, 64, 96))

    def test_deep_supervision_outputs_follow_the_intermediate_nodes(self):
        images = torch.rand(2, 3, 64, 96)
        for encoder in (self.convnext, self.efficientnet):
            model = self.build(encoder_name=encoder).train()
            logits, auxiliary = model(images)
            self.assertEqual([output.shape for output in (logits, *auxiliary)], [(2, 1, 64, 96)] * 4)
        self.assertEqual(self.build(deep_supervision=False).train()(images)[1], [])
        masks = (torch.rand(2, 1, 64, 96) > 0.9).float()
        loss = HairlineCrackLoss(auxiliary_weights=(0.5, 0.0, 0.0))
        model.zero_grad()
        loss(model(images), masks, torch.ones_like(masks)).backward()
        first, second = (head.weight.grad.abs().sum().item() for head in model.auxiliary_heads[:2])
        self.assertGreater(first, 0.0)
        self.assertEqual(second, 0.0)  # a zero weight leaves that output untrained

    def test_the_hairline_stem_can_be_left_out(self):
        with_stem, without = self.build(), self.build(stem_channels=0)
        self.assertEqual(with_stem.decoder.blocks['x_0_4'].conv1[0].in_channels, 64)
        self.assertEqual(without.decoder.blocks['x_0_4'].conv1[0].in_channels, 32)
        self.assertIsNone(without.full_resolution)
        self.assertIsNone(self.build(line_prior=False).thin_lines)
        self.assertEqual(self.build(line_prior=False).full_resolution[0][0].in_channels, 3)
        self.assertEqual(with_stem.full_resolution[0][0].in_channels, 7)  # RGB and four thin-line responses

    def test_only_the_encoder_trains_at_the_reduced_rate(self):
        model = self.build()
        groups = parameter_groups(model, 3e-4, 0.1, 1e-4)
        rate = {id(parameter): group['lr'] for group in groups for parameter in group['params']}
        for name, parameter in model.named_parameters():
            self.assertAlmostEqual(rate[id(parameter)], 3e-5 if name.startswith('encoder.') else 3e-4, msg=name)
        self.assertTrue(any(name.startswith('half_resolution.') for name, _ in model.named_parameters()))

    def test_checkpoint_rebuilds_the_network(self):
        for changes in ({}, {'encoder_name': self.efficientnet, 'stem_channels': 0, 'line_prior': False}):
            model = self.build(**changes).eval()
            architecture = {'name': 'unetplusplus', 'kwargs': {**self.kwargs, **changes}}
            images = torch.rand(1, 3, 64, 64)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'best.pt'
                save_checkpoint({'model': model.state_dict(), 'architecture': architecture}, path)
                rebuilt, _ = load_model_from_checkpoint(path)
                with torch.no_grad():
                    self.assertTrue(torch.equal(rebuilt(images), model(images)))

    def test_initial_weights_need_the_same_network(self):
        trained = self.build().eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint({'model': trained.state_dict(),
                             'architecture': {'name': 'unetplusplus', 'kwargs': self.kwargs}}, path)
            # Tuples for lists and another drop-path rate are the same network.
            wanted = {**self.kwargs, 'decoder_channels': (256, 128, 64, 32, 16), 'drop_path_rate': 0.0}
            model = UNetPlusPlus(pretrained=False, **wanted).eval()
            keys = model.load_state_dict(initial_weights(path, wanted)['model'], strict=False)
            self.assertEqual((keys.missing_keys, keys.unexpected_keys), ([], []))
            images = torch.rand(2, 3, 64, 96)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(images), trained(images)))
            for change in ({'encoder_name': self.efficientnet}, {'stem_channels': 0}, {'deep_supervision': False},
                           {'decoder_attention': None}, {'decoder_channels': [128, 64, 32, 32, 16]}):
                with self.assertRaises(SystemExit):
                    initial_weights(path, {**self.kwargs, **change})


if __name__ == '__main__':
    unittest.main()
