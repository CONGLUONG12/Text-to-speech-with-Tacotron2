FROM python:3.7
#FROM nvidia/cuda:11.4.0-base-ubuntu20.04
RUN apt update
RUN apt-get install -y python3-pip
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app
#RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
run pip install tensorflow==1.15.2
RUN apt-get update && apt-get install -y git && apt-get install -y libsndfile1 
COPY . /app
ENTRYPOINT ["python3","main.py"]  

