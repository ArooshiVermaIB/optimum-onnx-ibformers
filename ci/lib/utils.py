import logging
from typing import Dict, Any


def do_comparison(
    evaluation_results: Dict[str, Dict[str, Any]],
    expected_metrics: Dict[str, Dict[str, Any]],
    *,
    test_name: str,
) -> bool:
    """
    Check that the evaluation_results are each at least as high as the expected metrics
    >>> evaluation_results = {
    >>>     "f1_score": {
    >>>         "name": 0.95,
    >>>         "address": 0.9,
    >>>     }, "accuracy": {
    >>>         "name": 0.6,
    >>>         "address": 0.7,
    >>>     }, "pecision": {
    >>>         "name": 0.7,
    >>>         "address": 0.8,
    >>>     }
    >>> }
    >>> expected_metrics = {
    >>>        "f1_score": {
    >>>         "name": 0.9
    >>>     }, "accuracy": {
    >>>         "name": 0.5
    >>>     }
    >>> }
    >>> do_comparison(evaluation_results, expected_metrics)
    True
    """
    logger = logging.getLogger(test_name)
    success = True
    for metric, fields in expected_metrics.items():
        if metric not in evaluation_results:
            logger.error(f"evaluation results did not contain expected metric {metric}")
            success = False
            continue
        evaluation_results_for_metric = evaluation_results[metric]
        for field, value in fields.items():
            if field not in evaluation_results_for_metric:
                logger.error(f"evaluation results did not contain expected field {field} for metric {metric}")
                success = False
                continue
            observed = evaluation_results_for_metric[field]
            try:
                observed = float(observed)
            except ValueError:
                logger.error(
                    f"for field '{field}' and metric '{metric}' observed value '{observed}' "
                    f"could not be converted to a float"
                )
                success = False
                continue

            if (
                not isinstance(evaluation_results_for_metric[field], float)
                or value > evaluation_results_for_metric[field]
            ):
                logger.error(
                    f"for field '{field}' and metric '{metric}' expected at "
                    f"least '{value}' but observed '{evaluation_results_for_metric[field]}'"
                )
                success = False
    return success