FROM python:3.11-slim
RUN pip install --no-cache-dir requests
COPY forwarder.py /app/forwarder.py
WORKDIR /app
CMD ["python", "-u", "forwarder.py"]
