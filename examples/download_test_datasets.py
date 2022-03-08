from ci.lib.config import (
    load_environments,
    load_model_tests
)
import fire
import os
from download_dataset import ProjectDownloader

environments = load_environments()

model_tests = load_model_tests()

            
def download_test_dataset(test_dataset, dest_path):
    model_test = model_tests[test_dataset]
    dataset_path = model_test['dataset_project_path']
    for env in model_test['env']:
        final_path = os.path.join(dest_path, test_dataset, env)
        os.makedirs(final_path, exist_ok=True)
        complete_dataset_path = os.path.join(environments[env]['path'], dataset_path)
        downloader = ProjectDownloader(env, complete_dataset_path, final_path)
        downloader.ml_studio()

if __name__ == "__main__":
    fire.Fire(download_test_dataset)