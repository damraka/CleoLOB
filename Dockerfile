FROM python:3.12-slim
WORKDIR /opt/cleolob
COPY pyproject.toml requirements.txt ./
COPY lob ./lob
RUN pip install --no-cache-dir . && pip check
WORKDIR /work
ENTRYPOINT ["cleo"]
CMD ["--help"]
