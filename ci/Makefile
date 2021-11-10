build-docker:
	docker build -f . -t docker-test ../

run-docker-ci-test-doc-insights-sandbox:
	docker run docker-test --environment doc-insights-sandbox --no-train --no-inference

run-docker-ci-test-dogfood:
	docker run docker-test --environment dogfood
