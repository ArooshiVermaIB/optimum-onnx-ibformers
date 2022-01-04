# Changelog

All notable changes to this repository will be documented in this file.


## [Unreleased]
### Added
- Add validation of dataset sizes.
### Fixed
- Fix exact_match metric

## [0.0.11] - 2021-12-30
### Added
- Add `class_weights` support
- Add retry logic when loading images from ibdoc
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
