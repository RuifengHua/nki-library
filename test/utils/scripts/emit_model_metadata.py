# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone script to emit targeted model metadata to SQS.

Reads all JSON config files from configuration/test_vector_metadata/ and
sends them to the kernel-perf SQS queue for OpenSearch ingestion.

Designed to run as a separate pipeline approval step, decoupled from test
execution, so that metadata is emitted exactly once per pipeline run
regardless of how many test steps exist or whether tests crash.

Runs on Hydra Fargate where the invocation role provides SQS credentials.

Usage:
    python -m test.utils.scripts.emit_model_metadata --queue-url <SQS_QUEUE_URL>
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3

from ..metadata_loader import compute_config_version_id, default_metadata_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SQS_BATCH_SIZE = 10
SQS_MAX_RETRIES = 3


METADATA_DIR = default_metadata_dir()


@dataclass
class MetadataMessage:
    """A single metadata config ready for SQS emission."""

    label: str
    """Human-readable identifier for logging (e.g. 'filename.json:abcd1234')."""
    body: dict
    """Full SQS message body including type and payload."""


def collect_metadata_messages(metadata_dir: Path) -> list[MetadataMessage]:
    """Read all JSON config files and build SQS messages."""
    messages: list[MetadataMessage] = []
    indexed_at = datetime.now(timezone.utc).isoformat()

    for config_file in sorted(metadata_dir.glob("*.json")):
        try:
            with open(config_file, "r") as f:
                configs = json.load(f)

            if not isinstance(configs, list):
                configs = [configs]

            for config in configs:
                version_id = compute_config_version_id(config)
                body = {
                    "type": "targeted_model",
                    "payload": {
                        **config,
                        "version_id": version_id,
                        "source_file": config_file.name,
                        "indexed_at": indexed_at,
                    },
                }
                messages.append(
                    MetadataMessage(
                        label=f"{config_file.name}:{version_id[:8]}",
                        body=body,
                    )
                )
        except Exception as e:
            logger.error(f"Failed to parse config file {config_file.name}: {e}")

    return messages


def send_to_sqs(queue_url: str, messages: list[MetadataMessage]) -> tuple[int, list[str]]:
    """Send messages to SQS in batches. Returns (emitted_count, failed_labels)."""
    region = queue_url.split(".")[1] if "amazonaws.com" in queue_url else "us-east-1"
    sqs = boto3.client("sqs", region_name=region)
    emitted = 0
    failed: list[str] = []

    for i in range(0, len(messages), SQS_BATCH_SIZE):
        batch = messages[i : i + SQS_BATCH_SIZE]
        entries = [{"Id": str(idx), "MessageBody": json.dumps(msg.body)} for idx, msg in enumerate(batch)]

        for attempt in range(SQS_MAX_RETRIES):
            try:
                response = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
                emitted += len(response.get("Successful", []))
                for failure in response.get("Failed", []):
                    failed.append(batch[int(failure["Id"])].label)
                break
            except Exception as e:
                if attempt < SQS_MAX_RETRIES - 1:
                    logger.warning(f"Batch send attempt {attempt + 1}/{SQS_MAX_RETRIES} failed: {e}")
                else:
                    logger.error(f"Batch send failed after {SQS_MAX_RETRIES} retries: {e}")
                    failed.extend([msg.label for msg in batch])

    return emitted, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit targeted model metadata to SQS")
    parser.add_argument("--queue-url", required=True, help="SQS queue URL")
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR, help="Path to metadata JSON directory")
    args = parser.parse_args()

    if not args.metadata_dir.exists():
        logger.error(f"Metadata directory not found: {args.metadata_dir}")
        return 1

    messages = collect_metadata_messages(args.metadata_dir)
    if not messages:
        logger.warning("No metadata configs found")
        return 0

    logger.info(f"Collected {len(messages)} metadata configs from {args.metadata_dir}")

    emitted, failed = send_to_sqs(args.queue_url, messages)
    logger.info(f"Emitted {emitted}/{len(messages)} metadata configs to SQS")

    if failed:
        logger.error(f"Failed to emit {len(failed)} configs: {failed[:10]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
