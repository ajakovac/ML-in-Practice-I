from invoke import task
import time
import json
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
MONGO_DIR = os.path.join(root_dir,"Data","mongo")
CONTAINER_NAME = "mongodb"
IMAGE = "mongo:7.0"

@task
def mongo(c):
    """Run MongoDB in Docker with local data persistence."""
    # 1. Ensure local mongo directory exists
    os.makedirs(MONGO_DIR, exist_ok=True)
    print(f"📂 Mongo data directory: {MONGO_DIR}")

    # 2. Pull latest image if not present
    print(f"🐋 Pulling image {IMAGE} if necessary...")
    c.run(f"docker pull {IMAGE}")

    # 3. Run container (remove old one if exists)
    print(f"🚀 Starting MongoDB container '{CONTAINER_NAME}'...")
    c.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)  # remove if exists
    c.run(
        f'docker run -d '
        f'--name {CONTAINER_NAME} '
        f'-p 27017:27017 '
        f'-v "{MONGO_DIR}:/data/db" '
        f'{IMAGE}'
    )

    # 4. Show running container
    c.run(f"docker ps --filter name={CONTAINER_NAME}")

@task
def stop_mongo(c):
    """Stop the MongoDB Docker container (data is kept)."""
    print("🛑 Stopping MongoDB container...")
    c.run(f"docker stop {CONTAINER_NAME}", warn=True)
    c.run(f"docker ps --filter name={CONTAINER_NAME}")