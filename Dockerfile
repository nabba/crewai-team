# Pinned digest prevents silent base image substitution (supply-chain attack)
# To update: docker pull python:3.13-slim && docker inspect --format '{{index .RepoDigests 0}}'
# Note: Python 3.14 blocked by crewai's Requires-Python >=3.10,<3.14 pin.
FROM python:3.13-slim@sha256:d168b8d9eb761f4d3fe305ebd04aeb7e7f2de0297cec5fb2f8f6403244621664

WORKDIR /app

# Install system dependencies (gosu for privilege dropping)
RUN apt-get update && apt-get install -y \
    curl \
    gosu \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir yt-dlp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy dashboards: Firebase (legacy) + React control plane
COPY dashboard/public/index.html dashboard/index.html
COPY dashboard/build/ dashboard/build/

# Create workspace directories
RUN mkdir -p workspace/output workspace/memory workspace/skills workspace/proposals workspace/applied_code workspace/philosophy/texts

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app/workspace

# Copy entrypoint (runs as root to fix perms, then drops to appuser)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
