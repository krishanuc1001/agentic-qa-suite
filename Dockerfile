# Lightweight, reproducible image for Hugging Face Spaces / Render / Railway / Fly.io.
# Streamlit Community Cloud does NOT use this Dockerfile (it builds from
# requirements.txt + .python-version directly).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Hugging Face Spaces requires a non-root user with UID 1000.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Ensure runtime artifacts directory exists and is writable.
RUN mkdir -p /app/QAReports && chown -R appuser:appuser /app/QAReports

USER appuser

# Hugging Face Spaces injects $PORT (typically 7860); default for other hosts.
ENV PORT=7860
EXPOSE 7860

# Use the shell form so $PORT expands at container start.
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
