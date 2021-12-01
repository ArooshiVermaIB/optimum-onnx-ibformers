#!/bin/bash

# add more mypy check when we fix typings for other modules
/root/.local/bin/mypy -p ibformers.models
/root/.local/bin/mypy -p ibformers.datasets

# run unit tests for ibformers
/root/.local/bin/green -vv tests