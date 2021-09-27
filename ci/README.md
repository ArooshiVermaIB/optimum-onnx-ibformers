# Publishing

- Add Instabase environments where you want the package published to `environment.yaml`
- From `ib_deep_learning`, run `python -m ci.publish`. The package in `ib_deep_learning/layoutlm-mtt` will be published to all environments where the current version as specified in (`ib_deep_learning/layoutlm-mtt/package.json`) is not already present

## environments.yaml
```yaml
dogfood:
  host: https://dogfood.instabase.com
  path: daniel.cahn/my-repo/fs/Instabase Drive/test_location # This is a root location used for test files. It should be a safe location where things can be saved and deleted
  token: SOME_TOKEN # TODO we should ideally use environment variables here 

```

# Model Testing

Add tests to `model_tests.yaml`.

Run tests with `python -m ci.run_tests`. May take a while to run.

## model_tests.yaml

Example test config

```yaml
Test Name:
  env: dogfood
  # The .ibannotator file specified should be accessible using the API key in environments.yaml
  ibannotator: ib_annotation/data/fs/Prod Drive/datasets/DLs Clean/Drivers Licenses Clean2.ibannotator
  time_limit: 1200 # 1200 seconds = 20 minutes
  config:
    batch_size: 2
    learning_rate: 5.e-5 # Make sure to keep the . to make sure the yaml knows this is a float, not a string
    use_mixed_precision: true
    upload: false # Unless you need the weights downloaded after training, always include this
    use_gpu: true # Generally always true
    # Anything missing will use defaults on the server. Note that if defaults change, the results may change too.
  metrics: # all metrics currently assuming higher is better
    # Current metrics supported are "exact_match", "precision", "recall", "f1"
    exact_match:
      DL Number: 0.9
      Name: 0.3
      Address: 0.9
    precision:
      DL Number: 0.9 # Note that not all fields must be specified for every metric
    # Note that not all metrics must be tested 
```

# Unpublish

```bash
python -m ci.unpublish --package ib_layout_lm_trainer --version 0.0.7
```

# Advanced

If something goes wrong, change the log-level on line 51 of `config.py` to debug.
