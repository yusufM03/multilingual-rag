# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#  all-MiniLM-L6-v2 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy entire project into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit app on container start
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.headless=true"]
