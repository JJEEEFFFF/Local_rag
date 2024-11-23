FROM python:3.12-slim

WORKDIR /FINAL_LOCAL_RAG

COPY ./requirements.txt ./
RUN  pip install --no-cache-dir -r requirements.txt

COPY ./modules ./modules

CMD ["uvicorn", "FINAL_LOCAL_RAG.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]

