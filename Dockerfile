FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/phast
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

# The image deliberately performs only the bounded installation check. It does
# not launch a fracture simulation during image builds or CI smoke tests.
CMD ["python", "run_sanitizer.py"]
