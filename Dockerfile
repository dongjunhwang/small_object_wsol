# Please fix the pytorch version same as your environment.
FROM pytorch/pytorch:latest

RUN pip install munch
RUN pip install sklearn
RUN pip install opencv-python
RUN apt-get update
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
