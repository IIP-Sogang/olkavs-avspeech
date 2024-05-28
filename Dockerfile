FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

USER root