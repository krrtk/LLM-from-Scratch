FROM python:3.10-slim

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install dependencies first (layer-cache friendly)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=user:user . .

# Tell main.py where to find the weights
ENV MODEL_WEIGHTS_PATH=/home/user/app/verdict_weights.pth

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
