FROM python:3.12-slim-bookworm

# Select the target CPU architecture
ARG TARGETARCH

# Install Java runtime required by Apache Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-${TARGETARCH}
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Configure Python and Spark
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV SPARK_LOCAL_IP=127.0.0.1

# Set application directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default application
CMD ["python", "evaluate_models.py"]