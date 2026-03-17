# Pinned digest — keep in sync with main Dockerfile
FROM python:3.11-slim@sha256:6d98ca198cea726f2c86da2699594339a7b7ff08e49728797b4ed6e3b5c3b62a

# Install only required runtimes — keep image minimal
RUN apt-get update && apt-get install -y \
    nodejs npm ruby-full bash \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Common Python libraries agents are likely to need
RUN pip install --no-cache-dir \
    requests numpy pandas matplotlib

# Non-root user for extra isolation
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /sandbox

# Default: do nothing (command passed at runtime)
CMD ["bash"]
