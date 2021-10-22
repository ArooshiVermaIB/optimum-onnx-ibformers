# ibformers
DL models and data processing pipelines. Facilitate prototyping new solutions and ideas.

### Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

You'll need `instabase`'s `model-training-tasks` as well. In `instabase` repo execute:

```bash
cd distributed-tasks/celery/model-training-tasks/service-cpu
make clean build
```

From there, either add `distributed-tasks/celery/model-training-tasks/service-cpu/build/py` to PYTHONPATH.


#### Pycharm setup

For imports to properly show up in PyCharm, try the following:
1. Add `instabase` project as another Content Root (Preferences -> Project: ibformers -> Project Structure -> Add Content Root)
2. In the same panel, mark `distributed-tasks/celery/model-training-tasks/service-cpu/build/py` as Sources Root. 

### Main assumptions
- uses hf/datasets library for storing and preprocessing data
    - data manipulation process is defined as a pipeline of 
      functions which are applied on the dataset with the `map` method
    - both raw dataset and postprocessed datasets (e.g. by tokenization) are memory-mapped using an 
      efficient zero-serialization cost backend (Apache Arrow). No RAM limitation troubles - 
      especially painful if we load images of the pages.
- supported models
    - layoutlm
    - layoutlmv2 (TBD)
- supported tasks
    - extraction - token classification
    - extraction - token classification for QA (TBD)
    - extraction - seq2seq (TBD)
    - classification - with classification head (TBD)
    - classification - seq2seq (TBD)
    
