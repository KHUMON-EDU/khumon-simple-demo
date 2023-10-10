# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get install -y \
    libgl1 \
    libgl1-mesa-glxx


RUN git clone https://github.com/KHUMON-EDU/khumon-simple-demo .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
