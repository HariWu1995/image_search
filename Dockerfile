# Download base image
# FROM python:3.9.14-slim-bullseye
FROM python:3.8-slim

# Add source code and saved file
ENV HOME /root
ENV PYTHONPATH "/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/site-packages"

# RUN apt-get update
# RUN apt-get install -y git-core
# RUN apt remove -y python3-pip

RUN python3 -m pip install --upgrade --user pip
RUN python3 -m pip install --no-deps torch==1.10.2 torchmetrics==0.9.3 \
                                     torchaudio==0.10.2 torchtext==0.11.2 torchvision==0.11.3 \
                            --extra-index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install transformers==4.17.0

COPY ./requirements_api.txt ./
RUN python3 -m pip install -r requirements_api.txt --no-deps

COPY ./requirements_model.txt ./
RUN python3 -m pip install -r requirements_model.txt --no-deps

# COPY ./requirements_model_2.txt ./
# RUN python3 -m pip install -r requirements_model_2.txt --no-deps

# Install GPU
# https://towardsdatascience.com/how-to-set-up-deep-learning-machine-on-aws-gpu-instance-3bb18b0a2579

# RUN mkdir -p /usr/local/artefacts
COPY ./artefacts /usr/local/artefacts
COPY ./database  /usr/local/database
COPY ./config    /usr/local/config
COPY ./src       /usr/local/src

# Define environment variables
ENV BUCKET_MODEL dataspire-model-storage-dev

EXPOSE 8080
WORKDIR /usr/local
CMD python3 -m src.api.main -p 8080

# http://127.0.0.1:8080/docs