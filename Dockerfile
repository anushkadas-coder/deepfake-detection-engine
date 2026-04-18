FROM python:3.9

# Standard HF user setup
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /app

# Install the updated system dependencies for OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
USER user
# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the code
COPY --chown=user . .

# Run the FastAPI server on port 7860
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]