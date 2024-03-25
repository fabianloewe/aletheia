FROM ubuntu:latest

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    octave \
    octave-image \
    octave-signal \
    octave-nan \
    liboctave-dev \
    imagemagick \
    steghide \
    outguess \
    libmagic1

WORKDIR /app

COPY ./aletheia-cache /app/aletheia-cache
COPY ./aletheia-models /app/aletheia-models
COPY ./aletheia-resources /app/aletheia-resources
COPY ./aletheialib /app/aletheialib
COPY ./aletheia.py /app/aletheia.py
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "aletheia.py"]