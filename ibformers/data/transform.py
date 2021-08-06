import numpy as np

from ibformers.data.utils import convert_to_dict_of_lists, tag_answer_in_doc


def fuzzy_tag_in_document(example_batch):
    # try to find an answer inside the text of the document
    # example will be skipped in case of no spans found

    new_batch = []
    for example_num in range(len(example_batch['id'])):
        example = {k: v[example_num] for k, v in example_batch.items()}
        answers = tag_answer_in_doc(words=example['words'], answer=example['answer'])
        if len(answers) == 0:
            continue
        else:
            example['detected_answers'] = answers
            new_batch.append(example)

    new_keys = list(example.keys()) + ['detected_answers']
    dict_batch = convert_to_dict_of_lists(new_batch, new_keys)

    return dict_batch


def add_token_labels_qa(example_batch, **kwargs):
    batch_token_labels = []

    for token_starts, answers in zip(example_batch["token_starts"], example_batch["detected_answers"]):
        token_label_ids = np.zeros((len(token_starts)))
        for ans in answers:
            # look for indexes of the tokens which contain start and end of matched text
            start_idx = np.searchsorted(token_starts, ans["start"] + 1, 'left') - 1
            end_idx = np.searchsorted(token_starts, ans["end"], 'left')
            token_label_ids[start_idx:end_idx] = 1

        batch_token_labels.append(token_label_ids)

    return {'token_label_ids': batch_token_labels}