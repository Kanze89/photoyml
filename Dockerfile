FROM python:3.10-slim

WORKDIR /app

# Install git (needed to install CLIP from GitHub)
RUN apt-get update && apt-get install -y git && apt-get clean

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
