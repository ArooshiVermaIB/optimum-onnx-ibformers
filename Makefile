test:
	# add more mypy check when we fix typings for other modules
	mypy -p ibformers.models
	mypy -p ibformers.datasets

	# run unit tests for ibformers
	green -vv tests


build-docker:
	docker build -f `pwd`/Dockerfile -t ibformers-ci-unittests:latest `pwd`/


run-docker-test:
	# example: IB_TEST_ENV=doc-insights-sandbox make run-docker-test
	docker run --rm -t ibformers-ci-unittests:latest
