# Benchmark script

The benchmark script provides a way to run a single model on all predefined benchmark tasks. 

It handles training the model and logging the results to wandb.

### Usage

The benchmark should be run as a python module.

```bash
python -m benchmarks.benchmark \
  --models_to_run microsoft/layoutlm-base-uncased \
  --output_path /home/ib/experiments/test-run \
  --benchmark_to_run receipts driver_licenses
```

Arguments:
* `models_to_run` - path to the model, or its huggingface name. Default: models defined in `benchmarks.benchmark.DEFAULT_MODELS_TO_RUN`
* `output_path` - the path where the model output will be stored. Default: temporary directory that till be deleted upon finishing the run.
* `benchmark_to_run` - the names of the benchmark tasks that should be run. Default: all defined benchmark tasks.

### Task configuration

Each task is defined as a json file in `benchmarks/configs`. The json should contain following fields:
* `benchmark_name` - full name of the benchmark that will be used in wandb
* `benchmark_shared_path` - the path to the benchmark on the shared directory. Currently, the shared dir does not work so it containts the path on my private dir
* `hyperparams` - the default hyperparams for the training process