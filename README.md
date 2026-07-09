# HSGAT

Heterogeneous Sparse Graph Attention.

## GraphGym base setup

This repository starts from a GraphGym customization workflow. Initialize from the GraphGym base by cloning PyG and moving into its GraphGym directory:

```bash
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric/graphgym
```

## Quick start

Run a single GraphGym experiment (default: node classification on Planetoid):

```bash
bash run_single.sh
```

Run a batch/grid experiment:

```bash
bash run_batch.sh
```

To run with CPU backend, add this to the selected `*.yaml` config:

```yaml
accelerator: cpu
```

## In-depth GraphGym usage

### Single experiment

- Configuration is defined in a `*.yaml` file, with missing fields filled from GraphGym defaults (`set_cfg()`).
- Example launch:

```bash
python main.py --cfg configs/pyg/example_node.yaml --repeat 3
```

- Results are written to `results/${CONFIG_NAME}/`, including per-seed runs and aggregated statistics (`agg`, `stats.json`, `best.json`).

### Batch experiments

- Select a base config (`--config`) to define the starting architecture.
- Optionally set `--config_budget` to auto-control trainable-parameter budget.
- Provide a grid file (`--grid`) describing perturbations for generated runs.
- Generate config files:

```bash
python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --config_budget configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs
```

- Launch queued parallel runs:

```bash
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
```

- Results are written to `results/${CONFIG_NAME}_grid_${GRID_NAME}/`, with aggregate CSV summaries under `agg/`.

## Customizing GraphGym

GraphGym supports project-level customization without modifying PyG internals:

- External customization: `graphgym/custom_graphgym/`
- In-PyG contribution path: `torch_geometric/graphgym/contrib/`

Supported customizable module areas include:

- `custom_graphgym/act/`
- `custom_graphgym/config/`
- `custom_graphgym/feature_augment/`
- `custom_graphgym/feature_encoder/`
- `custom_graphgym/head/`
- `custom_graphgym/layer/`
- `custom_graphgym/loader/`
- `custom_graphgym/loss/`
- `custom_graphgym/network/`
- `custom_graphgym/optimizer/`
- `custom_graphgym/pooling/`
- `custom_graphgym/stage/`
- `custom_graphgym/train/`
- `custom_graphgym/transform/`
