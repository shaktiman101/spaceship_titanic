FROM python:3.7

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install lazypredict
COPY app.py /app/app.py
COPY train.csv /app/train.csv
COPY test.csv /app/test.csv
CMD python app.py