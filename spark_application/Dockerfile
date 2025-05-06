FROM bde2020/spark-master:3.1.1-hadoop3.2

RUN apk --no-cache update && \
    apk --no-cache add python3 python3-dev py3-pip gcc musl-dev linux-headers

WORKDIR /opt/spark_application

RUN pip3 install psutil

COPY . /opt/spark_application/
