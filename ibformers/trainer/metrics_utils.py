from datasets import Dataset

from instabase.metrics_utils.counter import Counter


_NUM_PAGES_TRAINED = u"apps/services/training/num_pages_trained"
_NUM_FILES_TRAINED = u"apps/services/training/num_files_trained"


def increment_document_counter(train_dataset: Dataset, training_job_id: str, model_name: str, user_name: str) -> None:
    """Increments the counter which tracks the number of files trained."""
    attributes = {
        u"training_job_id": training_job_id,
        u"model_name": model_name,
        u"user_name": user_name,
    }

    file_count = len(train_dataset)
    file_counter = Counter(_NUM_FILES_TRAINED)
    file_counter.increment_by(file_count, attributes)

    page_count = sum([len(page_in_doc) for page_in_doc in train_dataset["page_spans"]])
    page_counter = Counter(_NUM_PAGES_TRAINED)
    page_counter.increment_by(page_count, attributes)
