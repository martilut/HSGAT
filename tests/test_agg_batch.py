import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_agg_batch_module():
    tg = types.ModuleType('torch_geometric')
    graphgym = types.ModuleType('torch_geometric.graphgym')
    utils = types.ModuleType('torch_geometric.graphgym.utils')
    agg_runs = types.ModuleType('torch_geometric.graphgym.utils.agg_runs')
    agg_runs.agg_batch = lambda *_args, **_kwargs: None

    with patch.dict(
        'sys.modules',
        {
            'torch_geometric': tg,
            'torch_geometric.graphgym': graphgym,
            'torch_geometric.graphgym.utils': utils,
            'torch_geometric.graphgym.utils.agg_runs': agg_runs,
        },
    ):
        path = Path('/home/runner/work/HSGAT/HSGAT/graphgym/agg_batch.py')
        spec = importlib.util.spec_from_file_location('agg_batch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestAggBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agg_batch_module()

    def test_parse_args_defaults(self):
        with patch('sys.argv', ['agg_batch.py', '--dir', '/tmp/results']):
            args = self.module.parse_args()
        self.assertEqual(args.dir, '/tmp/results')
        self.assertEqual(args.metric, 'auto')

    def test_parse_args_custom_metric(self):
        with patch('sys.argv', ['agg_batch.py', '--dir', '/tmp/results', '--metric', 'auc']):
            args = self.module.parse_args()
        self.assertEqual(args.metric, 'auc')


if __name__ == '__main__':
    unittest.main()
