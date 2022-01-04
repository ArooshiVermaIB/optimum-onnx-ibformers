## OCR Script

### Usage:

```shell
pythom -m examples.rvl_ocr.rvl_ocr_script /path/to/input/documents /path/to/target/dir
```

### Process description

1. Create an index in the input directory, or read one if it exists.
2. Split the documents in the index into batches, and create Task for each batch.
3. Discard already completed tasks.
4. For each task:
   1. Create output directory
   2. Zip the documents assigned to the task.
   3. Send the zip to Instabase instance
   4. Unzip documents at Instabase and delete the zip
   5. Run OCR flow at Instabase
   6. Download processed documents
   7. Remove output from OCR flow
   8. Remove raw documents from Instabase.

### Options


Additional options can be set via env variables:

* `ENV_HOST`, `ENV_TOKEN` - params for env setup
* `ENV_DATA_PATH` - base path for data on Instabase instance
* `ENV_OCR_BINARY_FLOW_PATH` - path to binary flow that performs the processing
* `NUM_DOCS_PER_FOLDER` - number of documents per single folder
* `MAX_CONCURRENT_TASKS` - max number of tasks that can be run at the same time
* `MAX_CONCURRENT_FLOWS` - max number of OCR flows that can be run at the same time
* `DEBUG_LIMIT_DOC_NUMBER` - if set to an integer, only that many documents will be processed.

See `config.py` for other options.

Example call with options:
```shell
NUM_DOCS_PER_FOLDER=100 pythom -m examples.rvl_ocr.rvl_ocr_script /path/to/input/documents /path/to/target/dir
```