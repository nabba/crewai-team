# Pinned digest prevents silent base image substitution (supply-chain attack)
# To update: docker pull python:3.13-slim && docker inspect --format '{{index .RepoDigests 0}}'
# Note: Python 3.14 blocked by crewai's Requires-Python >=3.10,<3.14 pin.
FROM python:3.13-slim@sha256:d168b8d9eb761f4d3fe305ebd04aeb7e7f2de0297cec5fb2f8f6403244621664

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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

# Install ShinkaEvolve separately (--no-deps avoids httpx version
# conflict with crewai — shinka declares httpx==0.27, crewai needs
# >=0.28, and shinka actually works fine with 0.28 in practice).
#
# The transitive deps below are the ones shinka actually imports at
# session start. Without them every shinka session crashed at LLM-init
# with cryptic "Requested model(s) are unavailable" / "No module named
# 'psutil'" — and because shinka writes no ledger record on those
# crashes, the engine selector's days_since_engine_run("shinka")
# stayed at infinity and forced-rotation kept picking shinka forever
# (2026-04-30 diagnosis). Bake the deps into the image so this can't
# regress on rebuild.
#
# hydra-core / omegaconf / antlr4 / unidiff / radon / mando keep
# --no-deps to avoid pulling in conflicting transitive versions; they
# are direct shinka requirements with stable APIs.
#
# google-genai / python-Levenshtein / seaborn / psutil get full
# dependency resolution because their deps are well-behaved and
# google-genai needs anyio/google-auth/websockets pulled cleanly.
RUN pip install --no-cache-dir --no-deps \
    shinka-evolve@git+https://github.com/SakanaAI/ShinkaEvolve.git && \
    pip install --no-cache-dir --no-deps \
    hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
    unidiff radon mando && \
    pip install --no-cache-dir \
    google-genai python-Levenshtein seaborn psutil || true

# Install Playwright + Chromium for browser automation tools (T1-4).
# Chromium + its system libs are large; build skips the fonts pack to save space.
RUN playwright install --with-deps chromium || true

# Copy application code
COPY app/ app/

# Copy LLM Wiki subsystem (Karpathy pattern)
COPY wiki/ wiki/
COPY raw/ raw/
COPY wiki_schema/ wiki_schema/

# Copy dashboards: Firebase (legacy) + React control plane
COPY dashboard/public/index.html dashboard/index.html
COPY dashboard/build/ dashboard/build/

# Copy scripts (wiki ingest, etc.) and tests
COPY scripts/ scripts/
COPY tests/ tests/

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
