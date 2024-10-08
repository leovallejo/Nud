FROM amd64/python:3.9-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN mkdir /app/data
RUN chown -R 1000:1000 /app/data

EXPOSE 8000

ENV NAME sample

CMD ["gunicorn", "-b", ":8000", "app:app"]
