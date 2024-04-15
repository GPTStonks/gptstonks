# Dockerfile for production environment. TLS certificate and key are required as volumes.
FROM ubuntu:22.04

# Update system
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential libssl-dev python3-pip python3-venv python3-dev libwebkit2gtk-4.0-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install requirements with PDM and download index
COPY ./projects/gptstonks_api/pyproject.toml /api/pyproject.toml
COPY ./projects/gptstonks_api/pdm.lock /api/pdm.lock
WORKDIR /api
RUN pip install --no-cache-dir setuptools==68.2.2 wheel==0.41.3 pdm==2.12.3 && \
    pdm install --no-editable --no-self && \
    pdm cache clear
ENV PATH="/api/.venv/bin:$PATH"

# Copy FastAPI app
COPY ./projects/gptstonks_api/gptstonks/api /api/gptstonks_api
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

# Run the FastAPI app with SSL configuration
CMD ["uvicorn", "gptstonks_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "/api/key.pem", "--ssl-certfile", "/api/cert.pem"]
