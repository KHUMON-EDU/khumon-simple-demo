# app/Dockerfile

FROM python:3.11

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    libgl1 \
    ffmpeg

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app

COPY ./static /app/static

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
