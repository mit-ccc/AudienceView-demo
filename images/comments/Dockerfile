FROM python:3.10-slim

RUN apt-get -qq update
RUN apt-get install -y apt-utils
RUN apt-get -y install ca-certificates build-essential git curl wget
RUN apt-get -qq clean

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . .

CMD ["run.py"]
