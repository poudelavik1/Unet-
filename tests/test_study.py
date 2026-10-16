"""Validate the loss terms and the optimiser and loss study: single terms, their sum, the
trainer's options, and the comparison of finished runs in tables and figures."""
from pathlib import Path
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    import torch.nn.functional as F
    from unetpp.comparison import compare_runs, run_terms
    from unetpp.losses import CrackLoss, FocalCrackLoss, HairlineCrackLoss, loss_label, loss_terms, soft_cldice_loss
except ImportError:  # the workflow tests still run without PyTorch
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class LossTermTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 1, 32, 32) * 2
        self.target = (torch.rand(2, 1, 32, 32) > 0.85).float()
        self.valid = torch.ones_like(self.target)

    def loss(self, *terms, **options):
        return float(CrackLoss(terms, **options)(self.logits, self.target, self.valid))

    def test_names_and_aliases(self):
        self.assertEqual(loss_terms('hairline'), ('bce', 'tversky', 'cldice'))
        self.assertEqual(loss_terms(['tversky', 'bce', 'bce']), ('bce', 'tversky'))  # once each, in a fixed order
        self.assertEqual(loss_label(['cldice', 'focal']), 'Focal + clDice')
        with self.assertRaises(ValueError):
            loss_terms(['bce', 'lovasz'])

    def test_each_term_alone_is_its_formula(self):
        probability = torch.sigmoid(self.logits)
        tp = float((probability * self.target).sum())
        fp = float((probability * (1 - self.target)).sum())
        fn = float(((1 - probability) * self.target).sum())
        self.assertAlmostEqual(self.loss('bce'), float(F.binary_cross_entropy_with_logits(self.logits, self.target)), 5)
        self.assertAlmostEqual(self.loss('dice'), 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1), 5)
        self.assertAlmostEqual(self.loss('tversky'), 1 - (tp + 1) / (tp + 0.4 * fp + 0.6 * fn + 1), 5)
        self.assertAlmostEqual(self.loss('cldice', weights={'cldice': 1.0}),
                               float(soft_cldice_loss(probability, self.target, 10)), 5)
        focal = self.loss('focal')
        self.assertTrue(0 < focal < self.loss('bce'))  # focal down-weights the easy pixels

    def test_combined_loss_is_the_weighted_sum_of_its_terms(self):
        combined = self.loss('hairline')
        self.assertAlmostEqual(combined, self.loss('bce') + self.loss('tversky') + self.loss('cldice'), 5)
        self.assertAlmostEqual(combined, float(HairlineCrackLoss()(self.logits, self.target, self.valid)), 6)
        self.assertAlmostEqual(self.loss('focal', 'cldice'),
                               float(FocalCrackLoss()(self.logits, self.target, self.valid)), 6)
        self.assertAlmostEqual(self.loss('bce', 'dice', weights={'bce': 2.0, 'dice': 0.5}),
                               2 * self.loss('bce') + 0.5 * self.loss('dice'), 5)

    def test_deep_supervision_gets_the_region_terms_only(self):
        auxiliary = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        loss = CrackLoss('hairline', auxiliary_weights=(0.2, 0.2, 0.2))
        total = float(loss((self.logits, auxiliary), self.target, self.valid))
        region = sum(float(CrackLoss(('bce', 'tversky'))(extra, self.target, self.valid)) for extra in auxiliary)
        self.assertAlmostEqual(total, float(loss.output_term) + 0.2 * region, 5)

    def test_cldice_alone_is_minimised_by_crack_everywhere(self):
        # The reason the study leaves clDice alone out: it cannot tell a crack from the wall.
        self.assertEqual(float(soft_cldice_loss(torch.ones(1, 1, 32, 32), self.target[:1], 10)), 0.0)

    def test_trainer_options(self):
        from unetpp import training

        for given, expected in ((['--loss', 'dice'], ['dice']), ([], ['bce', 'tversky', 'cldice']),
                                (['--loss', 'focal', 'cldice', '--optimizer', 'adam'], ['focal', 'cldice'])):
            sys.argv = ['train.py', 'dataset', *given]
            args = training.parse_arguments()
            self.assertEqual(args.loss, expected)
        self.assertEqual(args.optimizer, 'adam')


def write_run(folder: Path, optimizer, loss, tol_f1, epochs=4):
    """A finished run as the trainer leaves it, with only the files the comparison reads."""
    folder.mkdir(parents=True)
    (folder / 'configuration.json').write_text(json.dumps({'optimizer': optimizer, 'loss': loss}))
    block = {'threshold': 0.5, 'loss': 1.0, 'precision': tol_f1 - 0.1, 'recall': tol_f1 - 0.05, 'f1': tol_f1 - 0.08,
             'iou': tol_f1 - 0.25, 'tol_precision': tol_f1 + 0.02, 'tol_recall': tol_f1 - 0.02, 'tol_f1': tol_f1,
             'accuracy': 0.98, 'per_source': {'eh#_r#_c#': {'tol_f1': tol_f1 - 0.1}},
             'with_source_thresholds': {'tol_f1': tol_f1 + 0.01}}
    (folder / 'test_metrics.json').write_text(json.dumps({'best_epoch': epochs, 'tta': block, 'no_tta': block}))
    with (folder / 'training_history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'tol_f1'])
        writer.writerows([epoch, tol_f1 * epoch / epochs] for epoch in range(1, epochs + 1))


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ComparisonTests(unittest.TestCase):
    def test_runs_of_a_study_are_compared_in_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / 'ablation'
            write_run(study / 'adamw_hairline', 'adamw', ['bce', 'tversky', 'cldice'], 0.82)
            write_run(study / 'adam_hairline', 'adam', 'hairline', 0.81)  # an earlier run stored the name
            write_run(study / 'adamw_dice', 'adamw', ['dice'], 0.79)
            write_run(study / 'adam_bce', 'adam', ['bce'], 0.76)
            (study / 'adam_dice').mkdir()  # still training: no configuration or metrics yet
            (study / 'adamw_bce').mkdir()
            (study / 'adamw_bce' / 'configuration.json').write_text(json.dumps({'optimizer': 'adamw', 'loss': ['bce']}))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                written = compare_runs([study], study / 'comparison')
            self.assertIn('Left out, no test metrics yet', output.getvalue())
            files = sorted(item.name for item in (study / 'comparison').iterdir())
            self.assertEqual(files, sorted(written))
            for name in ('comparison_summary.csv', 'comparison_summary.md', 'comparison_test_metrics.png',
                         'comparison_heatmap.png', 'comparison_validation_curves.png',
                         'comparison_precision_recall.png'):
                self.assertIn(name, files)
            with (study / 'comparison' / 'comparison_summary.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row['optimizer'], row['loss']) for row in rows],
                             [('Adam', 'BCE'), ('AdamW', 'Dice'), ('AdamW', 'BCE + Tversky + clDice'),
                              ('Adam', 'BCE + Tversky + clDice')])
            self.assertEqual(rows[2]['tol_f1_source_thresholds'], '0.8300')
            table = (study / 'comparison' / 'comparison_summary.md').read_text()
            self.assertIn('**0.820**', table)  # the best run is marked

    def test_old_loss_names(self):
        self.assertEqual(run_terms({'loss': 'focal'}), ('focal', 'cldice'))
        self.assertEqual(run_terms({}), ('bce', 'tversky', 'cldice'))


if __name__ == '__main__':
    unittest.main()
