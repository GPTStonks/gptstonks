FROM ubuntu:22.04

# Update system
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential libssl-dev python3-pip python3-venv python3-dev libwebkit2gtk-4.0-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install requirements with PDM
COPY ./pyproject.toml /api/pyproject.toml
COPY ./pdm.lock /api/pdm.lock
WORKDIR /api
RUN pip install --no-cache-dir setuptools==68.2.2 wheel==0.41.3 pdm==2.10.0 && \
    pdm install --no-editable --no-self
ENV PATH="/api/.venv/bin:$PATH"

# Download index with gdown
RUN --mount=type=secret,id=vsi_gdrive_uri \
    pdm add gdown && \
    VSI_GDRIVE_URI=$(cat /run/secrets/vsi_gdrive_uri) python -c "import os, gdown; gdown.download_folder(os.getenv('VSI_GDRIVE_URI'))" >/dev/null 2>&1 && \
    pdm cache clear

# Copy __init__ in parent dir to enable relative imports
COPY ./__init__.py /api/__init__.py

# Copy FastAPI app
COPY ./gptstonks_api /api/gptstonks_api
WORKDIR /api/gptstonks_api

# Expose port for FastAPI app to run on
EXPOSE 8000

# Create OpenSSL conf file for OECD data access within openbb
RUN echo 'openssl_conf = openssl_init\n\
\n\
[openssl_init]\n\
ssl_conf = ssl_sect\n\
\n\
[ssl_sect]\n\
system_default = system_default_sect\n\
\n\
[system_default_sect]\n\
Options = UnsafeLegacyRenegotiation'\
> /api/gptstonks_api/openssl.cnf
ENV OPENSSL_CONF="/api/gptstonks_api/openssl.cnf"

WORKDIR /api

# Run the FastAPI app
CMD ["uvicorn", "gptstonks_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
