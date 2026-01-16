FROM python:3.13-slim-trixie

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/

COPY main.py .

ENTRYPOINT [ "python", "main.py" ]

