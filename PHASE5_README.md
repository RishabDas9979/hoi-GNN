# Phase 5 files (training and evaluation)

| File | Plan item | Tested by author? |
|---|---|---|
| evaluation/metrics.py | 5.5 | yes, 13 tests passed |
| evaluation/vcoco_evaluator.py | 5.6 | yes, 14 tests passed (own reimplementation: diff against s-gupta/v-coco before calling it official) |
| evaluation/aggregate.py | 5.8, 5.9 | yes, 5 tests passed |
| training/losses.py | 5.1 | NO (needs torch) |
| training/trainer.py | 5.2, 5.3 | NO (needs torch) |
| training/train.py | 5.4 | NO (needs torch) |
| training/data_hooks.py | glue | NO; `_real_datasets` MUST be adapted to data/dataset.py |
| training/batch_adapter.py | glue | NO; field names `y`, `y_mask`, `pair_index` are assumptions |
| training/fake_data.py, utils/tiny_model.py | dev stubs | NO |
| utils/checkpoint.py, utils/seed.py | 5.3 | NO |
| configs/training_config.yaml | config | n/a |
| tests/test_phase5.py | tests | written, never run |

## Run order on your machine
    pytest tests/test_metrics.py tests/test_vcoco_evaluator.py tests/test_aggregate.py -q
    pytest tests/test_phase5.py -v
    python -m training.train --model tiny --data fake --seed 0 --epochs 3 --out results_smoke
    python -m evaluation.aggregate --results results_smoke

## Contract with Phase 4 (tell your teammate)
- `models.build_model(name, cfg)` where `cfg["runtime"]` = {num_actions, node_dim, edge_dim}
- `model(data) -> logits [P, A]` (raw logits, no sigmoid, no masking)
- Prior baseline: optional `fit(dataset)` method, buffers instead of parameters
- Registry names expected by aggregate.py: prior, mlp, gcn, gat, edge_aware_gatv2
