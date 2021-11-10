FROM python:3.7

WORKDIR /ci_test

COPY . /ci_test/

RUN python -m pip install --upgrade pip && pip install --user -r ci/requirements.txt

ENTRYPOINT ["sh", "ci/entrypoint.sh"]