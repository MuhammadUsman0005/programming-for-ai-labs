FROM python:3.11-slim

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements_deploy.txt

COPY . .

EXPOSE 7860

CMD ["bash", "start.sh"]
