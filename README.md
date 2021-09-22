# ibformers
DL models and data processing pipelines. Facilitate prototyping new solutions and ideas.


### Main assumtions
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
    
