from pathlib import Path

from google.cloud import storage


def upload_file(
    bucket_name: str,
    local_path: str | Path,
    destination_path: str,
) -> None:
    """Upload a local file to a Google Cloud Storage bucket."""
    local_file = Path(local_path)

    if not local_file.is_file():
        raise FileNotFoundError(f"File not found: {local_file}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)

    blob.upload_from_filename(str(local_file))

    print(f"Uploaded {local_file} to gs://{bucket_name}/{destination_path}")
