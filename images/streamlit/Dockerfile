FROM python:3.10-slim

RUN apt-get -qq update
RUN apt-get install -y apt-utils
RUN apt-get -y install ca-certificates build-essential git curl wget
RUN apt-get -qq clean

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords

COPY . .

EXPOSE 8504

HEALTHCHECK --interval=30s --timeout=30s --retries=3 CMD ["/healthcheck.sh"]

ENTRYPOINT ["/app/entrypoint.sh"]
