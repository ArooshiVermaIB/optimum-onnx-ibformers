FROM python:3.8

WORKDIR /ci_test

COPY . /ci_test/

RUN pip install -r ci/requirements.txt

ENTRYPOINT ["sh", "ci/entrypoint.sh"]