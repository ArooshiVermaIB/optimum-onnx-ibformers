# Changelog

All notable changes to this repository will be documented in this file.

# 1.2.0 - 2022-05-20
### Added
- Added data augmentation for split classifier, with a new pipeline

### Fixed
- Fixed memory issue caused by improper image handling in table data preprocessing
- Fixed incorrect calculation of exact match metric for document without annotated fields

# 1.1.1 - 2022-05-12
### Fixed
- Set default writer_batch_size to 4 to avoid OOMs

## 1.1.0 - 2022-05-12

### Added
- Added a model/pipeline for table extraction - DETR-based table tranformer
- Added ce_ins loss type to classifier and split classififer
- Added logic to filter first n pages for classification
- Added layoutlm v3 pipeline for extraction

### Fixed
- Fix the bug related to order of records which do not correspond to page order

## [1.0.6] - 2022-04-19
### Fixed
- Bug fix: classifier generation code skips over the first class label by mistake.
- Avoid expanduser when determining cache dir path
- Better message if there are no records processed for inference
- Fix bug in split-classifier if all records are skipped

## [1.0.5] - 2022-04-01

### Changed
- Removed `datasets.load_dataset` call from inference code to eliminate disk IO operations.
- Change the way how parameters are passed to pipeline components

### Fixed
- Bug fix in extraction dataset creation, when the length of annotations is zero.
- Fixed a bug during fp16 prediction in split classification

## [1.0.4] - 2022-03-25

### Changed
- Disabled early stopping and validation set size in default parameters

### Fixed
- Improved classification inference speed for large datasets

## [1.0.3] - 2022-03-18

### Fixed
- Fixed metric formatting when non-number is returned

## [1.0.2] - 2022-03-17

### Added
- Results from hyperparameter search are logged as a table in instabase>=22.03

### Changed
- Adapted the metric formatting to changes in instabase>=22.03

### Fixed
- Replaced ibopen calls with read_file
- Set `classes` when generating classifier and split classifier modules.

## [1.0.1] - 2022-03-14

### Fixed
- Optuna import does not break when the library is not present.

## [1.0.0] - 2022-03-11

### Added
- Hyperparameter tuning
- Add new loss type ce_ins (apply class_weight based on inverse number of samples)
- Add new base model - instalm-base
- Add classification pipeline
- Add split-classification pipeline

### Fixed
- Fix exact match metric displaying 100% when no annotations are present
- Fix sending too long information to metadata and logs
- Refactor datasets to use same components for all 3 tasks

### Deprecated
- Remove MQA model

## [0.3.0] - 2022-02-07

### Fixed
- Optimize memory usage during data preparation for long documents

## [0.2.0] - 2022-01-27

### Changed
- Progress bar in ML Studio will not show 100% until job complete.
- Improved status messages in ML Studio.
- Changed the naming of the package (ibformers_extraction)
- Changed default hyperparameters for base models

### Fixed
- Optimize memory usage during prediction for long documents

## [0.1.0] - 2022-01-13
### Added
- Add early stopping with validation dataset split.
- QnA pipeline
- Add possibility to limit chunks without annotations - useful for long documents

### Fixed
- Decreased writer batch size for ML Studio dataset to reduce memory usage

### Changed
- Files uploaded on metric read/write are compressed for uploading

## [0.0.12] - 2022-01-05
### Fixed
- Fix exact_match metric
### Added
- Add option to download base model from Instabase drive
- Add validation of dataset sizes.

## [0.0.11] - 2021-12-30
### Added
- Add `class_weights` support
- Add retry logic when loading images from ibdoc
- Add options for early stopping and evaluation dataset creation
### Fixed
- Add fix inside data processing to correct bboxes with negative coordinates
### Changed
- Migrate existing benchmarks to ML Studio format
### Removed
- remove support for annotator datasets

## [0.0.10] - 2021-12-09
### Added
- Initial version of ibformers which include multiple base models for extraction


The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).
