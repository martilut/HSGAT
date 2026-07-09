import importlib.util
import ast
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_configs_gen_module():
    numpy = types.ModuleType('numpy')
    numpy.array = lambda x: x
    numpy.sum = lambda x: sum(x)
    numpy.round = lambda x: x
    yaml = types.ModuleType('yaml')
    yaml.FullLoader = object
    yaml.load = lambda *_args, **_kwargs: {}
    yaml.dump = lambda *_args, **_kwargs: None
    tg = types.ModuleType('torch_geometric')
    graphgym = types.ModuleType('torch_geometric.graphgym')
    utils = types.ModuleType('torch_geometric.graphgym.utils')
    comp_budget = types.ModuleType('torch_geometric.graphgym.utils.comp_budget')
    io = types.ModuleType('torch_geometric.graphgym.utils.io')

    comp_budget.match_baseline_cfg = lambda cfg, *_args, **_kwargs: cfg
    io.makedirs_rm_exist = lambda *_args, **_kwargs: None
    io.string_to_python = ast.literal_eval

    with patch.dict(
        'sys.modules',
        {
            'torch_geometric': tg,
            'torch_geometric.graphgym': graphgym,
            'torch_geometric.graphgym.utils': utils,
            'torch_geometric.graphgym.utils.comp_budget': comp_budget,
            'torch_geometric.graphgym.utils.io': io,
            'numpy': numpy,
            'yaml': yaml,
        },
    ):
        path = Path('/home/runner/work/HSGAT/HSGAT/graphgym/configs_gen.py')
        spec = importlib.util.spec_from_file_location('configs_gen', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestConfigsGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_configs_gen_module()

    def test_get_fname(self):
        self.assertEqual(self.module.get_fname('/a/b/c.yaml'), 'c')
        self.assertEqual(self.module.get_fname(None), 'default')

    def test_grid2list(self):
        out = self.module.grid2list([[1, 2], ['x', 'y']])
        self.assertEqual(out, [[1, 'x'], [2, 'x'], [1, 'y'], [2, 'y']])

    def test_lists_distance(self):
        self.assertEqual(self.module.lists_distance([1, 2, 3], [1, 0, 3]), 1)

    def test_exclude_list_id(self):
        self.assertEqual(self.module.exclude_list_id(['a', 'b', 'c'], 1), ['a', 'c'])

    def test_load_search_file(self):
        content = "model.type t ['gcn','gat']\n\noptim.base_lr lr [0.01,0.1]\n"
        with tempfile.NamedTemporaryFile('w+', delete=True) as f:
            f.write(content)
            f.flush()
            out = self.module.load_search_file(f.name)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0][0], 'model.type')
        self.assertEqual(out[1][0][1], 'lr')


if __name__ == '__main__':
    unittest.main()
