# Changelog

All notable changes to this repository will be documented in this file.


## [Unreleased]
### Added
- Add early stopping with validation dataset split.
- QnA pipeline
- Add possibility to limit chunks without annotations - useful for long documents

### Fixed
- Decreased writer batch size for ML Studio dataset to reduce memory usage

### Changed
- Files uploaded on metric read/write are compressed for uploading

## [0.0.12] - 2022-1-5
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
