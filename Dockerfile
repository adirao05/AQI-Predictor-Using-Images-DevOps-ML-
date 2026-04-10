FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=300 --retries 5 torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=300 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "aaa/src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]