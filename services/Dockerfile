FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app
COPY . .

RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]