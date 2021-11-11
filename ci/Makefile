build-docker:
	docker build -f `pwd`/Dockerfile -t docker-test `pwd`/../

run-docker-test:
	# example: IB_TEST_ENV=doc-insights-sandbox make run-docker-test
	docker run docker-test ci.run_tests --environment $(IB_TEST_ENV)

run-docker-publish:
	# example: IB_TEST_ENV=doc-insights-sandbox make run-docker-publish
	docker run docker-test ci.publish --environment $(IB_TEST_ENV)
