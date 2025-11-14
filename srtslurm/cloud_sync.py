"""
Cloud storage sync for benchmark results using S3-compatible storage
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


class CloudSyncManager:
    """Manager for syncing benchmark results with S3-compatible cloud storage."""

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """Initialize CloudSyncManager.

        Args:
            endpoint_url: S3-compatible endpoint URL
            bucket: Bucket name
            prefix: Optional prefix for all keys (e.g., "benchmark-results/")
            aws_access_key_id: Access key (defaults to env var)
            aws_secret_access_key: Secret key (defaults to env var)
        """
        if boto3 is None:
            raise ImportError(
                "boto3 is required for cloud sync. Install with: pip install boto3"
            )

        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def push_run(self, run_dir: str, progress_callback=None) -> bool:
        """Upload a single run directory to cloud storage.

        Args:
            run_dir: Path to run directory (e.g., "3667_1P_1D_20251110_192145")
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            True if successful, False otherwise
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            logger.error(f"Run directory does not exist: {run_dir}")
            return False

        run_name = run_path.name
        logger.info(f"Pushing run {run_name} to cloud storage...")

        # Get all files in run directory (including subdirectories)
        files_to_upload = []
        for root, dirs, files in os.walk(run_path):
            for file in files:
                file_path = Path(root) / file
                files_to_upload.append(file_path)

        if not files_to_upload:
            logger.warning(f"No files found in {run_dir}")
            return False

        # Upload each file
        uploaded = 0
        for file_path in files_to_upload:
            # Calculate relative path within run directory
            rel_path = file_path.relative_to(run_path)
            # S3 key includes run name and relative path
            s3_key = f"{self.prefix}{run_name}/{rel_path}"

            try:
                self.s3.upload_file(str(file_path), self.bucket, s3_key)
                uploaded += 1

                if progress_callback:
                    progress_callback(uploaded, len(files_to_upload), str(rel_path))

                logger.debug(f"Uploaded {rel_path}")
            except Exception as e:
                logger.error(f"Failed to upload {rel_path}: {e}")
                return False

        logger.info(f"Successfully pushed {uploaded} files from {run_name}")
        return True

    def pull_run(
        self, run_id: str, local_dir: str, progress_callback=None
    ) -> Optional[str]:
        """Download a single run from cloud storage.

        Args:
            run_id: Run directory name (e.g., "3667_1P_1D_20251110_192145")
            local_dir: Local directory to download to
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            Path to downloaded run directory, or None if failed
        """
        logger.info(f"Pulling run {run_id} from cloud storage...")

        # List all objects with this run's prefix
        prefix = f"{self.prefix}{run_id}/"
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        except ClientError as e:
            logger.error(f"Failed to list objects for {run_id}: {e}")
            return None

        if "Contents" not in response:
            logger.warning(f"No files found for run {run_id}")
            return None

        objects = response["Contents"]
        run_path = Path(local_dir) / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Download each file
        downloaded = 0
        for obj in objects:
            s3_key = obj["Key"]
            # Remove prefix and run_id from key to get relative path
            rel_path = s3_key[len(prefix) :]

            if not rel_path:  # Skip directory markers
                continue

            local_file = run_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.s3.download_file(self.bucket, s3_key, str(local_file))
                downloaded += 1

                if progress_callback:
                    progress_callback(downloaded, len(objects), rel_path)

                logger.debug(f"Downloaded {rel_path}")
            except Exception as e:
                logger.error(f"Failed to download {rel_path}: {e}")
                return None

        logger.info(f"Successfully pulled {downloaded} files to {run_path}")
        return str(run_path)

    def list_remote_runs(self) -> list[str]:
        """List all runs available in cloud storage.

        Returns:
            List of run directory names
        """
        try:
            # Use delimiter to get "directories" at the prefix level
            response = self.s3.list_objects_v2(
                Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"
            )
        except ClientError as e:
            logger.error(f"Failed to list remote runs: {e}")
            return []

        if "CommonPrefixes" not in response:
            return []

        # Extract run names from common prefixes
        runs = []
        for prefix_obj in response["CommonPrefixes"]:
            prefix_str = prefix_obj["Prefix"]
            # Remove the base prefix and trailing slash
            run_name = prefix_str[len(self.prefix) :].rstrip("/")
            if run_name:
                runs.append(run_name)

        return sorted(runs, reverse=True)

    def sync_missing_runs(self, local_dir: str, progress_callback=None) -> int:
        """Download runs that exist in cloud but not locally.

        Args:
            local_dir: Local logs directory
            progress_callback: Optional callback function(run_name, current, total)

        Returns:
            Number of runs downloaded
        """
        # Get list of remote runs
        remote_runs = self.list_remote_runs()
        if not remote_runs:
            logger.info("No remote runs found")
            return 0

        # Get list of local runs
        local_path = Path(local_dir)
        local_runs = set()
        if local_path.exists():
            for entry in local_path.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    local_runs.add(entry.name)

        # Find missing runs
        missing_runs = [run for run in remote_runs if run not in local_runs]

        if not missing_runs:
            logger.info("All remote runs are already downloaded")
            return 0

        logger.info(f"Found {len(missing_runs)} missing runs to download")

        # Download missing runs
        downloaded = 0
        for i, run_id in enumerate(missing_runs, 1):
            if progress_callback:
                progress_callback(run_id, i, len(missing_runs))

            result = self.pull_run(run_id, local_dir)
            if result:
                downloaded += 1

        logger.info(f"Downloaded {downloaded}/{len(missing_runs)} runs")
        return downloaded

    def run_exists_in_cloud(self, run_id: str) -> bool:
        """Check if a run exists in cloud storage.

        Args:
            run_id: Run directory name

        Returns:
            True if run exists, False otherwise
        """
        prefix = f"{self.prefix}{run_id}/"
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=1
            )
            return "Contents" in response and len(response["Contents"]) > 0
        except ClientError as e:
            logger.error(f"Failed to check if run exists: {e}")
            return False

    def test_connection(self) -> bool:
        """Test connection to cloud storage.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            logger.info("Successfully connected to cloud storage")
            return True
        except NoCredentialsError:
            logger.error("No credentials provided for cloud storage")
            return False
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                logger.error(f"Bucket '{self.bucket}' does not exist")
            else:
                logger.error(f"Failed to connect to cloud storage: {e}")
            return False


def load_cloud_config(config_path: str = "cloud_config.toml") -> Optional[dict]:
    """Load cloud configuration from TOML file.

    Args:
        config_path: Path to cloud_config.toml

    Returns:
        Dict with cloud config, or None if file doesn't exist
    """
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli
        except ImportError:
            logger.error("tomli/tomllib required for config loading")
            return None

    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        with open(config_file, "rb") as f:
            config = tomli.load(f)
        return config.get("cloud", {})
    except Exception as e:
        logger.error(f"Failed to load cloud config: {e}")
        return None


def create_sync_manager_from_config(
    config_path: str = "cloud_config.toml",
) -> Optional[CloudSyncManager]:
    """Create CloudSyncManager from config file.

    Args:
        config_path: Path to cloud_config.toml

    Returns:
        CloudSyncManager instance, or None if config doesn't exist or is invalid
    """
    config = load_cloud_config(config_path)
    if not config:
        return None

    required_keys = ["endpoint_url", "bucket"]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return None

    try:
        return CloudSyncManager(
            endpoint_url=config["endpoint_url"],
            bucket=config["bucket"],
            prefix=config.get("prefix", ""),
        )
    except Exception as e:
        logger.error(f"Failed to create sync manager: {e}")
        return None

