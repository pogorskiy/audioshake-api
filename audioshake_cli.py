#!/usr/bin/env python3
"""
AudioShake CLI - Command line interface for AudioShake API.

Upload audio files, run processing jobs, and download results.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

import requests


# Supported audio file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".aiff", ".aif", ".mp4", ".mov"}


# =============================================================================
# AudioShake Models Documentation
# =============================================================================
#
# INSTRUMENT STEM SEPARATION (1 credit/min, max 3 hours)
# These models isolate or extract musical components from a mixed track.
# Useful for remixing, immersive audio, gaming, and music education.
#
# | Model Key        | Description                                          | Variants      |
# |------------------|------------------------------------------------------|---------------|
# | vocals           | Extracts vocal elements from a mix                   | high_quality  |
# | vocals_lead      | Lead vocals (primary melodic/lyrical content)        | -             |
# | vocals_backing   | Backing vocals, harmonies, chants, ad-libs, choirs   | -             |
# | instrumental     | Instrumental-only version (removes vocals)           | high_quality  |
# | drums            | Isolates percussion and rhythmic elements            | -             |
# | bass             | Separates bass instruments and low-frequency sounds  | -             |
# | guitar           | Isolates all guitar stems (acoustic, electric, etc.) | -             |
# | guitar_electric  | Isolates electric guitar only                        | -             |
# | guitar_acoustic  | Isolates acoustic/classical guitar                   | -             |
# | piano            | Extracts piano or keyboard instruments               | -             |
# | strings          | Isolates orchestral strings (violin, cello, viola)   | -             |
# | wind             | Extracts wind instruments (flute, saxophone, etc.)   | -             |
# | other            | Remaining instrumentation after main stems removed   | -             |
# | other-x-guitar   | Residual after removing vocals, drums, bass, guitar  | -             |
#
# DIALOGUE, MUSIC & EFFECTS (1.5 credits/min, max 3 hours unless noted)
# Isolate voices or remove background elements for film, TV, and dubbing.
#
# | Model Key        | Description                                          | Credits/Min | Max Length |
# |------------------|------------------------------------------------------|-------------|------------|
# | dialogue         | Isolates speech/vocals from other sounds             | 1.5         | 3 Hours    |
# | effects          | Retains ambience, sound effects, environmental noise | 1.5         | 3 Hours    |
# | music_removal    | Removes music, retains dialogue/effects/natural sound| N/A         | 1 Hour     |
# | music_fx         | Background stem (music + effects, no dialogue)       | 1.5         | 3 Hours    |
# | music_detection  | Detects portions containing music                    | 0.5         | 3 Hours    |
#
# Note: music_removal and multi_voice require access request.
#       Contact support@audioshake.ai to enable these models.
#
# TRANSCRIPTION & ALIGNMENT (1 credit/min, max 1 hour)
# Convert spoken content into synchronized text and timestamps.
#
# | Model Key        | Description                                          |
# |------------------|------------------------------------------------------|
# | transcription    | Text representation of spoken words                  |
# | alignment        | Synchronization of audio and corresponding text      |
#
# Note: Running both together costs 1.5 credits/min (Premium pricing).
#
# VARIANTS
# Certain models offer variants optimized for specific use-cases.
# Use --variant to specify.
#
# | Model        | Variant       | Description                              | Credits/Min |
# |--------------|---------------|------------------------------------------|-------------|
# | vocals       | high_quality  | Higher quality, longer processing time   | 1.5         |
# | instrumental | high_quality  | Higher quality, longer processing time   | 1.5         |
# | multi_voice  | two_speaker   | Optimized for 2 speakers (default)       | N/A*        |
# | multi_voice  | n_speaker     | Creates stems for any number of speakers | N/A*        |
#
# * multi_voice not available via /tasks API
#
# =============================================================================

# Available models for validation and help text
AVAILABLE_MODELS = {
    # Instrument Stem Separation
    "vocals": "Extracts vocal elements (supports high_quality variant)",
    "vocals_lead": "Lead vocals - primary melodic/lyrical content",
    "vocals_backing": "Backing vocals, harmonies, chants, ad-libs, choirs",
    "instrumental": "Instrumental-only version (supports high_quality variant)",
    "drums": "Percussion and rhythmic elements",
    "bass": "Bass instruments and low-frequency sounds",
    "guitar": "All guitar stems (acoustic, electric, classical)",
    "guitar_electric": "Electric guitar only",
    "guitar_acoustic": "Acoustic/classical guitar only",
    "piano": "Piano and keyboard instruments",
    "strings": "Orchestral strings (violin, cello, viola)",
    "wind": "Wind instruments (flute, saxophone, etc.)",
    "other": "Remaining instrumentation after main stems",
    "other-x-guitar": "Residual after vocals, drums, bass, guitar removed",
    # Dialogue, Music & Effects
    "dialogue": "Isolates speech/vocals from other sounds",
    "effects": "Ambience, sound effects, environmental noise",
    "music_removal": "Removes music, retains dialogue/effects/natural sound (request access: support@audioshake.ai)",
    "music_fx": "Background stem (music + effects, no dialogue)",
    "music_detection": "Detects portions containing music",
    "multi_voice": "Separates dialogue from multiple speakers (variants: two_speaker, n_speaker)",
    # Transcription & Alignment
    "transcription": "Text representation of spoken words",
    "alignment": "Synchronization of audio and text",
}

# Models that support variants
MODEL_VARIANTS = {
    "vocals": ["high_quality"],
    "instrumental": ["high_quality"],
    "multi_voice": ["two_speaker", "n_speaker"],
}


class AudioShakeError(Exception):
    """Base exception for AudioShake API errors."""
    pass


class AudioShakeClient:
    """Client for interacting with AudioShake API."""

    def __init__(self, api_key: str, base_url: str = "https://api.audioshake.ai"):
        """
        Initialize AudioShake client.

        Args:
            api_key: AudioShake API key for authentication
            base_url: Base URL for AudioShake API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        files: Optional[dict] = None,
    ) -> dict:
        """
        Make an authenticated request to the AudioShake API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /assets)
            json_data: JSON body for POST requests
            files: Files for multipart/form-data uploads

        Returns:
            Response JSON as dict

        Raises:
            AudioShakeError: If the API returns an error
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "x-api-key": self.api_key,
            "accept": "application/json",
        }

        try:
            if files:
                response = requests.request(method, url, headers=headers, files=files)
            elif json_data:
                headers["Content-Type"] = "application/json"
                response = requests.request(method, url, headers=headers, json=json_data)
            else:
                response = requests.request(method, url, headers=headers)

            if response.status_code >= 400:
                try:
                    data = response.json()
                    print(f"API Error Response (HTTP {response.status_code}):", file=sys.stderr)
                    print(f"{data}", file=sys.stderr)
                    
                    if response.status_code == 401:
                        raise AudioShakeError("Authentication failed: Invalid API key")
                    elif response.status_code == 404:
                        raise AudioShakeError(f"Not found: {data.get('message', 'Resource not found')}")
                    elif response.status_code == 400:
                        raise AudioShakeError(f"Bad request: {data.get('message', 'Invalid request')}")
                    else:
                        raise AudioShakeError(f"API error: {data.get('message', response.text)}")
                except ValueError:
                    print(f"API Error Response (HTTP {response.status_code}):", file=sys.stderr)
                    print(f"{response.text}", file=sys.stderr)
                    raise AudioShakeError(f"API error: {response.text}")

            return response.json()

        except requests.exceptions.ConnectionError:
            raise AudioShakeError("Network error: Unable to connect to AudioShake API")
        except requests.exceptions.Timeout:
            raise AudioShakeError("Network error: Request timed out")
        except requests.exceptions.RequestException as e:
            raise AudioShakeError(f"Network error: {str(e)}")

    def upload_asset(self, file_path: str) -> dict:
        """
        Upload an audio file to AudioShake.

        Args:
            file_path: Path to the audio file to upload

        Returns:
            Asset dict with id, format, link, name

        Raises:
            FileNotFoundError: If the file does not exist
            AudioShakeError: If the upload fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        
        # Determine MIME type based on extension
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
            ".mp4": "video/mp4",
            ".mov": "video/mov",
        }
        mime_type = mime_types.get(ext, "application/octet-stream")

        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            return self._make_request("POST", "/assets", files=files)

    def create_task(
        self,
        asset_id: str,
        model: str,
        output_format: str = "wav",
        residual: bool = False,
        variant: Optional[str] = None,
    ) -> dict:
        """
        Create a processing task for an asset.

        Args:
            asset_id: ID of the uploaded asset
            model: Model to use (e.g., vocals, drums, bass)
            output_format: Output format (wav, mp3, flac)
            residual: Include residual audio
            variant: Model variant (e.g., high_quality)

        Returns:
            Task dict with id and targets

        Raises:
            AudioShakeError: If task creation fails
        """
        target = {
            "model": model,
            "formats": [output_format],
            "residual": residual,
        }
        if variant:
            target["variant"] = variant

        payload = {
            "assetId": asset_id,
            "targets": [target],
        }

        return self._make_request("POST", "/tasks", json_data=payload)

    def get_task(self, task_id: str) -> dict:
        """
        Get task status by ID.

        Args:
            task_id: ID of the task

        Returns:
            Task dict with current status

        Raises:
            AudioShakeError: If task not found or API error
        """
        return self._make_request("GET", f"/tasks/{task_id}")

    def wait_for_task(self, task_id: str, poll_interval: int = 5) -> dict:
        """
        Poll task until all targets are completed or failed.

        Args:
            task_id: ID of the task to wait for
            poll_interval: Seconds between status checks

        Returns:
            Completed task dict

        Raises:
            AudioShakeError: If task fails or API error
        """
        while True:
            task = self.get_task(task_id)
            targets = task.get("targets", [])

            # Check status of all targets
            all_done = True
            has_failed = False
            statuses = []

            for target in targets:
                status = target.get("status", "unknown")
                model = target.get("model", "unknown")
                statuses.append(f"{model}: {status}")

                if status == "failed":
                    has_failed = True
                elif status != "completed":
                    all_done = False

            # Display progress
            print(f"Status: {', '.join(statuses)}")

            if has_failed:
                # Find the error message
                for target in targets:
                    if target.get("status") == "failed":
                        error = target.get("error", "Unknown error")
                        raise AudioShakeError(f"Task failed: {error}")

            if all_done:
                return task

            time.sleep(poll_interval)

    def download_outputs(
        self,
        task: dict,
        output_dir: str,
        source_name: Optional[str] = None,
        model: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> list:
        """
        Download all output files from a completed task.

        Args:
            task: Completed task dict with targets containing output links
            output_dir: Directory to save downloaded files
            source_name: Original source file name (without extension) for output naming
            model: Model used for processing
            variant: Model variant used (if any)

        Returns:
            List of downloaded file paths

        Raises:
            AudioShakeError: If download fails
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        downloaded_files = []
        targets = task.get("targets", [])
        used_filenames = set()  # Track used filenames to avoid collisions

        for target in targets:
            outputs = target.get("output", [])
            target_model = target.get("model", model or "output")
            target_variant = target.get("variant", variant) or variant
            
            for output in outputs:
                link = output.get("link")
                fmt = output.get("format", "wav")

                if not link:
                    continue

                # Build filename: {source_name}-{model}-{variant}.{ext}
                # or {source_name}-{model}.{ext} if no variant
                if source_name:
                    if target_variant:
                        base_filename = f"{source_name}-{target_model}-{target_variant}"
                    else:
                        base_filename = f"{source_name}-{target_model}"
                else:
                    # Fallback to original name from API
                    name = output.get("name", "output")
                    base_filename = name
                
                # Handle filename collisions by adding a counter
                filename = f"{base_filename}.{fmt}"
                counter = 1
                while filename in used_filenames:
                    filename = f"{base_filename}_{counter}.{fmt}"
                    counter += 1
                used_filenames.add(filename)
                
                filepath = os.path.join(output_dir, filename)

                try:
                    print(f"Downloading {filename}...")
                    print(f"URL: {link}")
                    response = requests.get(link, stream=True)
                    response.raise_for_status()

                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded_files.append(filepath)
                    print(f"Saved: {filepath}")

                except requests.exceptions.RequestException as e:
                    raise AudioShakeError(f"Failed to download {filename}: {str(e)}")

        return downloaded_files


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    
    # Build models help text
    models_help = "\n\nAvailable Models:\n"
    models_help += "\n  INSTRUMENT STEM SEPARATION (1 cr/min, max 3h):\n"
    stem_models = ["vocals", "vocals_lead", "vocals_backing", "instrumental", 
                   "drums", "bass", "guitar", "guitar_electric", "guitar_acoustic",
                   "piano", "strings", "wind", "other", "other-x-guitar"]
    for m in stem_models:
        models_help += f"    {m:18} - {AVAILABLE_MODELS[m]}\n"
    
    models_help += "\n  DIALOGUE, MUSIC & EFFECTS (1.5 cr/min, max 3h unless noted):\n"
    dme_models = ["dialogue", "effects", "music_removal", "music_fx", "music_detection", "multi_voice"]
    for m in dme_models:
        models_help += f"    {m:18} - {AVAILABLE_MODELS[m]}\n"
    
    models_help += "\n  TRANSCRIPTION & ALIGNMENT (1 cr/min, max 1h):\n"
    ta_models = ["transcription", "alignment"]
    for m in ta_models:
        models_help += f"    {m:18} - {AVAILABLE_MODELS[m]}\n"
    
    models_help += "\n  VARIANTS (use with --variant):\n"
    models_help += "    vocals --variant high_quality       - Higher quality (1.5 cr/min)\n"
    models_help += "    instrumental --variant high_quality - Higher quality (1.5 cr/min)\n"
    models_help += "    multi_voice --variant two_speaker   - Optimized for 2 speakers (default)\n"
    models_help += "    multi_voice --variant n_speaker     - Stems for any number of speakers\n"
    
    parser = argparse.ArgumentParser(
        prog="audioshake_cli",
        description="CLI for AudioShake API - upload audio, process with ML models, download results.",
        epilog=f"""
Examples:
  # Upload a file only (get asset ID for later use)
  %(prog)s --api-key YOUR_KEY --input song.mp3

  # Upload all audio files from a directory
  %(prog)s --api-key YOUR_KEY --input ./my_songs/

  # Process an existing asset with a model
  %(prog)s --api-key YOUR_KEY --asset-id abc123 --model vocals --output ./results

  # Upload and process in one command
  %(prog)s --api-key YOUR_KEY --input song.mp3 --model vocals --output ./results

  # Use high_quality variant for better results
  %(prog)s --api-key YOUR_KEY --input song.mp3 --model vocals --variant high_quality --output ./results

  # Process all audio files in a directory (creates subdirs per file)
  %(prog)s --api-key YOUR_KEY --input ./my_songs/ --model vocals --output ./results

  # Specify output format
  %(prog)s --api-key YOUR_KEY --input song.mp3 --model drums --output ./results --format mp3

  # Extract dialogue from video
  %(prog)s --api-key YOUR_KEY --input video.mp4 --model dialogue --output ./results
{models_help}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="AudioShake API key for authentication",
    )

    # Input source group - either file/directory or existing asset
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        metavar="PATH",
        help="Path to input audio file or directory containing audio files",
    )
    input_group.add_argument(
        "--asset-id",
        metavar="ID",
        help="ID of previously uploaded asset",
    )

    parser.add_argument(
        "--model",
        metavar="MODEL",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model for processing (see list below)",
    )

    parser.add_argument(
        "--variant",
        metavar="VARIANT",
        help="Model variant (e.g., high_quality for vocals/instrumental)",
    )

    parser.add_argument(
        "--output",
        metavar="DIR",
        help="Output directory for downloaded results",
    )

    parser.add_argument(
        "--format",
        choices=["wav", "mp3", "flac"],
        default="wav",
        help="Output format (default: wav)",
    )

    parser.add_argument(
        "--source-name",
        metavar="NAME",
        help="Source file name (without extension) for output naming when using --asset-id",
    )

    return parser


def get_audio_files_from_path(path: str) -> list[str]:
    """
    Get list of audio files from a path.
    
    If path is a file, returns [path].
    If path is a directory, returns all audio files in it.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of audio file paths
    """
    if os.path.isfile(path):
        return [path]
    
    if os.path.isdir(path):
        audio_files = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    audio_files.append(full_path)
        return sorted(audio_files)
    
    return []


def validate_args(args: argparse.Namespace) -> tuple[bool, str]:
    """
    Validate argument combinations.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Must have either --input or --asset-id to do anything useful
    if not args.input and not args.asset_id:
        return False, "Either --input or --asset-id is required"

    # If --model is provided, --output is required
    if args.model and not args.output:
        return False, "--output is required when --model is specified"

    # If --asset-id is provided, --model must also be provided
    if args.asset_id and not args.model:
        return False, "--model is required when using --asset-id"
    
    # Validate input path exists
    if args.input:
        if not os.path.exists(args.input):
            return False, f"Input path not found: {args.input}"
        
        # If directory, check it has audio files
        if os.path.isdir(args.input):
            audio_files = get_audio_files_from_path(args.input)
            if not audio_files:
                return False, f"No audio files found in directory: {args.input}"
    
    # Validate variant is supported for the model
    if args.variant and args.model:
        supported_variants = MODEL_VARIANTS.get(args.model, [])
        if not supported_variants:
            return False, f"Model '{args.model}' does not support variants"
        if args.variant not in supported_variants:
            return False, f"Invalid variant '{args.variant}' for model '{args.model}'. Supported: {', '.join(supported_variants)}"

    return True, ""


def get_mode(args: argparse.Namespace) -> str:
    """
    Determine execution mode based on arguments.
    
    Returns:
        'upload' - upload only mode (--input without --model)
        'process' - process mode (--model with --input or --asset-id)
    """
    if args.input and not args.model:
        return "upload"
    return "process"


def run_upload_only(client: AudioShakeClient, input_path: str) -> int:
    """
    Upload-only mode: upload file(s) and print asset ID(s).
    
    Args:
        client: AudioShake API client
        input_path: Path to audio file or directory
        
    Returns:
        Exit code (0 for success, 2 for file not found, 3 for API error)
    """
    audio_files = get_audio_files_from_path(input_path)
    
    if not audio_files:
        print(f"Error: No audio files found at {input_path}", file=sys.stderr)
        return 2
    
    total = len(audio_files)
    success_count = 0
    
    for i, input_file in enumerate(audio_files, 1):
        if total > 1:
            print(f"\n[{i}/{total}] Processing: {os.path.basename(input_file)}")
        
        try:
            print(f"Uploading {input_file}...")
            asset = client.upload_asset(input_file)
            asset_id = asset.get("id")
            print(f"Upload successful!")
            print(f"Asset ID: {asset_id}")
            success_count += 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            if total == 1:
                return 2
        except AudioShakeError as e:
            print(f"Error: {e}", file=sys.stderr)
            if total == 1:
                return 3
    
    if total > 1:
        print(f"\nCompleted: {success_count}/{total} files uploaded successfully")
    
    return 0 if success_count == total else 3


def run_process_mode(
    client: AudioShakeClient,
    asset_id: str,
    model: str,
    output_dir: str,
    output_format: str,
    variant: Optional[str] = None,
    source_name: Optional[str] = None,
) -> int:
    """
    Process mode: create task, wait for completion, download results.
    
    Args:
        client: AudioShake API client
        asset_id: ID of the asset to process
        model: Model to use for processing
        output_dir: Directory to save results
        output_format: Output format (wav, mp3, flac)
        variant: Model variant (e.g., high_quality)
        source_name: Original source file name (without extension) for output naming
        
    Returns:
        Exit code (0 for success, 3 for API error, 4 for task failed)
    """
    try:
        variant_str = f" (variant: {variant})" if variant else ""
        print(f"Creating task with model '{model}'{variant_str}...")
        task = client.create_task(asset_id, model, output_format, variant=variant)
        task_id = task.get("id")
        print(f"Task created: {task_id}")
        
        print("Waiting for task to complete...")
        completed_task = client.wait_for_task(task_id)
        
        print("\nTask result JSON:")
        print(json.dumps(completed_task, indent=2))
        print()
        
        print("Downloading results...")
        downloaded = client.download_outputs(
            completed_task, output_dir,
            source_name=source_name, model=model, variant=variant
        )
        
        print(f"Done! Downloaded {len(downloaded)} file(s) to {output_dir}")
        return 0
    except AudioShakeError as e:
        error_msg = str(e)
        if "Task failed" in error_msg:
            print(f"Error: {e}", file=sys.stderr)
            return 4
        print(f"Error: {e}", file=sys.stderr)
        return 3


def run_full_workflow(
    client: AudioShakeClient,
    input_path: str,
    model: str,
    output_dir: str,
    output_format: str,
    variant: Optional[str] = None,
) -> int:
    """
    Full workflow: upload, create task, wait, download.
    Supports both single files and directories.
    
    Args:
        client: AudioShake API client
        input_path: Path to audio file or directory
        model: Model to use for processing
        output_dir: Directory to save results
        output_format: Output format (wav, mp3, flac)
        variant: Model variant (e.g., high_quality)
        
    Returns:
        Exit code (0 for success, 2 for file not found, 3 for API error, 4 for task failed)
    """
    audio_files = get_audio_files_from_path(input_path)
    
    if not audio_files:
        print(f"Error: No audio files found at {input_path}", file=sys.stderr)
        return 2
    
    total = len(audio_files)
    success_count = 0
    
    for i, input_file in enumerate(audio_files, 1):
        if total > 1:
            print(f"\n{'='*60}")
            print(f"[{i}/{total}] Processing: {os.path.basename(input_file)}")
            print(f"{'='*60}")
        
        # Create per-file output directory if processing multiple files
        if total > 1:
            file_basename = os.path.splitext(os.path.basename(input_file))[0]
            file_output_dir = os.path.join(output_dir, file_basename)
        else:
            file_output_dir = output_dir
        
        result = _process_single_file(
            client, input_file, model, file_output_dir, output_format, variant
        )
        
        if result == 0:
            success_count += 1
        elif total == 1:
            return result
    
    if total > 1:
        print(f"\n{'='*60}")
        print(f"Batch complete: {success_count}/{total} files processed successfully")
        print(f"{'='*60}")
    
    return 0 if success_count == total else 3


def _process_single_file(
    client: AudioShakeClient,
    input_file: str,
    model: str,
    output_dir: str,
    output_format: str,
    variant: Optional[str] = None,
) -> int:
    """
    Process a single file: upload, create task, wait, download.
    
    Args:
        client: AudioShake API client
        input_file: Path to audio file to upload
        model: Model to use for processing
        output_dir: Directory to save results
        output_format: Output format (wav, mp3, flac)
        variant: Model variant (e.g., high_quality)
        
    Returns:
        Exit code (0 for success, 2 for file not found, 3 for API error, 4 for task failed)
    """
    asset_id = None
    # Get source file name without extension for output naming
    source_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Step 1: Upload
    try:
        print(f"Uploading {input_file}...")
        asset = client.upload_asset(input_file)
        asset_id = asset.get("id")
        print(f"Upload successful! Asset ID: {asset_id}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except AudioShakeError as e:
        print(f"Error during upload: {e}", file=sys.stderr)
        return 3
    
    # Step 2: Create task and process
    try:
        variant_str = f" (variant: {variant})" if variant else ""
        print(f"Creating task with model '{model}'{variant_str}...")
        task = client.create_task(asset_id, model, output_format, variant=variant)
        task_id = task.get("id")
        print(f"Task created: {task_id}")
        
        print("Waiting for task to complete...")
        completed_task = client.wait_for_task(task_id)
        
        print("\nTask result JSON:")
        print(json.dumps(completed_task, indent=2))
        print()
        
        print("Downloading results...")
        downloaded = client.download_outputs(
            completed_task, output_dir,
            source_name=source_name, model=model, variant=variant
        )
        
        print(f"Done! Downloaded {len(downloaded)} file(s) to {output_dir}")
        print(f"Asset ID for reuse: {asset_id}")
        return 0
    except AudioShakeError as e:
        error_msg = str(e)
        # Always print asset ID if upload succeeded
        print(f"Asset ID for reuse: {asset_id}", file=sys.stderr)
        if "Task failed" in error_msg:
            print(f"Error: {e}", file=sys.stderr)
            return 4
        print(f"Error: {e}", file=sys.stderr)
        return 3


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    is_valid, error_msg = validate_args(args)
    if not is_valid:
        print(f"Error: {error_msg}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 1

    # Create API client
    client = AudioShakeClient(args.api_key)
    
    # Determine mode and execute
    mode = get_mode(args)
    
    if mode == "upload":
        # Upload-only mode: --input without --model
        return run_upload_only(client, args.input)
    else:
        # Process mode
        if args.input:
            # Full workflow: --input with --model
            return run_full_workflow(
                client, args.input, args.model, args.output, args.format,
                variant=args.variant
            )
        else:
            # Process existing asset: --asset-id with --model
            return run_process_mode(
                client, args.asset_id, args.model, args.output, args.format,
                variant=args.variant, source_name=args.source_name
            )


if __name__ == "__main__":
    sys.exit(main())
