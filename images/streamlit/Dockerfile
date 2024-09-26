FROM python:3.10-slim

RUN apt-get -qq update
RUN apt-get install -y apt-utils
RUN apt-get -y install ca-certificates build-essential git curl wget
RUN apt-get -qq clean

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r /app/requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN rm -f /app/requirements.txt

ARG USERID=0
ARG GROUPID=0

# Check if the group exists, and create it if it doesn't
RUN if ! getent group ${GROUPID} >/dev/null; then \
        groupadd -g ${GROUPID} mygroup; \
    fi

# Check if the user exists, and create it if it doesn't
RUN if ! getent passwd ${USERID} >/dev/null; then \
        useradd -m -u ${USERID} -g ${GROUPID} myuser; \
    fi

USER ${USERID}:${GROUPID}

COPY --chown=${USERID}:${GROUPID} . .

EXPOSE 8504

HEALTHCHECK --interval=30s --timeout=30s --retries=3 CMD ["/healthcheck.sh"]

ENTRYPOINT ["/app/entrypoint.sh"]
