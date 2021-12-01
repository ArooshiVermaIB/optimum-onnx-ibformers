FROM python:3.7

WORKDIR /ci_unittest

COPY . /ci_unittest/

RUN python -m pip install --upgrade pip && pip install --user -r ./requirements-dev.txt && pip install --user -r ./requirements.txt

ENTRYPOINT ["sh", "entrypoint.sh"]
