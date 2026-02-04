FROM python:3.12-slim-trixie

WORKDIR /app

COPY Readme.md .
COPY picsellia_token .
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/
COPY configs/ configs/

COPY main.py .

ENTRYPOINT [ "python", "main.py" ]

