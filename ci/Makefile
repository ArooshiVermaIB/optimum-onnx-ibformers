build-docker:
	docker build -f `pwd`/Dockerfile -t docker-test `pwd`/../

run-docker-ci-test-doc-insights-sandbox:
	docker run docker-test ci.run_tests --environment doc-insights-sandbox

run-docker-ci-test-dogfood:
	docker run docker-test ci.run_tests --environment dogfood

run-docker-ci-publish-doc-insights-sandbox:
	docker run docker-test ci.publish --environment doc-insights-sandbox

run-docker-ci-publish-dogfood:
	docker run docker-test ci.publish --environment dogfood
