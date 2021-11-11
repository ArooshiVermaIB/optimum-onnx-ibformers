test:
	# add more mypy check when we fix typings for other modules
	mypy -p ibformers.models
	mypy -p ibformers.datasets

	# run unit tests for ibformers
	green -vv tests
