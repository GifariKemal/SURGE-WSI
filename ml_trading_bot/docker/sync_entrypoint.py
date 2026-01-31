"""
Sync Service Entrypoint
=======================

Runs the data sync service with health check API
"""

import os
import sys
import threading
import logging

# Add src to path
sys.path.insert(0, '/app')

from src.data.sync_service import DataSyncService, create_health_app
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration from environment
    pg_config = {
        'host': os.getenv('POSTGRES_HOST', 'timescaledb'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'surge_wsi'),
        'user': os.getenv('POSTGRES_USER', 'surge_wsi'),
        'password': os.getenv('POSTGRES_PASSWORD', 'surge_wsi_secret')
    }

    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    sync_interval = int(os.getenv('SYNC_INTERVAL', 60))

    logger.info("=" * 60)
    logger.info("SURGE-WSI Data Sync Service")
    logger.info("=" * 60)
    logger.info(f"PostgreSQL: {pg_config['host']}:{pg_config['port']}")
    logger.info(f"Qdrant: {qdrant_host}:{qdrant_port}")
    logger.info(f"Redis: {redis_host}:{redis_port}")
    logger.info(f"Sync interval: {sync_interval}s")
    logger.info("=" * 60)

    # Create sync service
    sync_service = DataSyncService(
        pg_config=pg_config,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        redis_host=redis_host,
        redis_port=redis_port,
        sync_interval=sync_interval
    )

    # Create health check app
    app = create_health_app(sync_service)

    # Start sync service in background thread
    sync_thread = threading.Thread(target=sync_service.start, daemon=True)
    sync_thread.start()
    logger.info("Sync service started in background")

    # Run health check API
    logger.info("Starting health check API on port 8765...")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
