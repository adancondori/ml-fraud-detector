# Stage 1: builder — install all dependencies
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-scorer.txt .
RUN pip install --no-cache-dir -r requirements-scorer.txt

# Stage 2: runtime — copy only what's needed
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY scorer/ ./scorer/
COPY src/ ./src/
COPY setup.py pyproject.toml ./

# Install the fraud_detector package in editable mode (no-deps since all deps are already installed)
RUN pip install --no-cache-dir -e . --no-deps

# Exec form CMD — required for graceful shutdown and lifespan events
CMD ["uvicorn", "scorer.main:app", "--host", "0.0.0.0", "--port", "8000"]
