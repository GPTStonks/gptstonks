FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Update system
RUN apt update && apt install -y python3-pip python3-venv

#Install graphical tool to avoid interaction
RUN apt install -y libwebkit2gtk-4.0-dev

# Install requirements.txt
COPY ./requirements.txt /api/requirements.txt
WORKDIR /api
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy __init__ in parent dir to enable relative imports
COPY /__init__.py /api/__init__.py

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

WORKDIR /api

# Run the FastAPI app
CMD ["uvicorn", "gptstonks_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--env-file", "gptstonks_api/.env"]
