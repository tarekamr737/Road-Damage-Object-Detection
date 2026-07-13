FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-deploy.txt \
    && pip install --no-cache-dir --no-deps .

COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY models/exports/production_road_damage_model.pt ./models/exports/production_road_damage_model.pt

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5).read()"

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
