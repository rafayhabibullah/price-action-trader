FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production deps only
RUN uv sync --frozen --no-dev

# Copy source code
COPY . .

# Cloud Run expects port 8080
ENV PORT=8080

CMD ["uv", "run", "python", "-m", "trading.runner"]
