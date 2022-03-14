build-docker:
	docker build -f `pwd`/Dockerfile -t ibformers-ci:latest `pwd`/../

run-docker-test:
	# example: IB_TEST_ENV=doc-insights-sandbox make run-docker-test
	docker run --rm -t ibformers-ci:latest ci.run_tests --environment $(IB_TEST_ENV)

run-docker-publish:
	# example: IB_TEST_ENV=doc-insights-sandbox make run-docker-publish
	docker run --rm -t ibformers-ci:latest ci.publish --environment $(IB_TEST_ENV) --package ibformers_extraction
	docker run --rm -t ibformers-ci:latest ci.publish --environment $(IB_TEST_ENV) --package ibformers_classification
