#!/usr/bin/env python3
"""
AudioShake Batch Processor - Batch processing wrapper for audioshake_cli.py.

Uploads WAV files once to get Asset IDs, then runs multiple models on each asset.
For vocals and instrumental models, automatically runs both standard and high_quality variants.
"""

import argparse
import os
import re
import subprocess
import sys


def validate_models(models_str: str) -> list[str]:
    """
    Validate and parse the models string.
    
    Args:
        models_str: Comma-separated list of models
        
    Returns:
        List of model names
        
    Raises:
        ValueError: If models string is empty or contains only whitespace
    """
    if not models_str or not models_str.strip():
        raise ValueError("Models list cannot be empty")
    
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    
    if not models:
        raise ValueError("Models list cannot be empty")
    
    return models


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="audioshake_batch",
        description="Batch process WAV files through AudioShake API.",
        epilog="""
Examples:
  # Process all WAV files in a folder with vocals and drums models
  %(prog)s --input ./songs --api-key YOUR_KEY --models vocals,drums --output ./results

  # Process with multiple models including high_quality variants
  %(prog)s --input ./songs --api-key YOUR_KEY --models vocals,instrumental,bass --output ./results

  # Process specific files in order from a text file
  %(prog)s --input ./songs --api-key YOUR_KEY --models vocals --output ./results --file-list files.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Path to folder containing WAV files",
    )
    
    parser.add_argument(
        "--api-key",
        required=True,
        metavar="KEY",
        help="AudioShake API key for authentication",
    )
    
    parser.add_argument(
        "--models",
        required=True,
        metavar="MODELS",
        help="Comma-separated list of models (e.g., vocals,drums,bass)",
    )
    
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--file-list",
        metavar="FILE",
        help="Path to text file with list of filenames to process (one per line, in order)",
    )
    
    parsed = parser.parse_args(args)
    
    # Validate models
    try:
        parsed.models_list = validate_models(parsed.models)
    except ValueError as e:
        parser.error(str(e))
    
    return parsed


def read_file_list(file_list_path: str) -> list[str]:
    """
    Read list of filenames from a text file.
    
    Args:
        file_list_path: Path to text file with filenames (one per line)
        
    Returns:
        List of filenames in the order they appear in the file
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty or contains no valid entries
    """
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"File list does not exist: {file_list_path}")
    
    filenames = []
    with open(file_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                filenames.append(line)
    
    if not filenames:
        raise ValueError(f"File list is empty: {file_list_path}")
    
    return filenames


def scan_wav_files(input_dir: str, file_list: list[str] | None = None) -> list[str]:
    """
    Scan directory and return list of WAV files.
    
    If file_list is provided, returns files in that order (validating they exist).
    Otherwise, returns sorted list of all WAV files in directory.
    
    Args:
        input_dir: Path to directory to scan
        file_list: Optional list of filenames to process in order
        
    Returns:
        List of paths to WAV files
        
    Raises:
        FileNotFoundError: If directory does not exist or file from list not found
        ValueError: If no WAV files found
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input path is not a directory: {input_dir}")
    
    if file_list is not None:
        # Use provided file list in order
        wav_files = []
        for filename in file_list:
            file_path = os.path.join(input_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File from list not found: {file_path}")
            wav_files.append(file_path)
        
        if not wav_files:
            raise ValueError("File list is empty")
        
        return wav_files
    
    # Default: scan directory for all WAV files
    wav_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            wav_files.append(os.path.join(input_dir, filename))
    
    if not wav_files:
        raise ValueError(f"No WAV files found in directory: {input_dir}")
    
    return sorted(wav_files)


def extract_asset_id(output: str) -> str | None:
    """
    Extract Asset ID from audioshake_cli.py output.
    
    Args:
        output: stdout from audioshake_cli.py
        
    Returns:
        Asset ID string or None if not found
    """
    # Look for "Asset ID: <id>" pattern
    match = re.search(r"Asset ID:\s*(\S+)", output)
    if match:
        return match.group(1)
    return None


def expand_models_with_variants(models: list[str]) -> list[tuple[str, str | None]]:
    """
    Expand model list by adding variants for vocals and instrumental.
    
    For vocals and instrumental models, both standard (None) and high_quality
    variants are added. For other models, only the standard variant is added.
    
    Args:
        models: List of model names
        
    Returns:
        List of tuples (model, variant) where variant is None or "high_quality"
        
    Example:
        ["vocals", "drums"] -> [("vocals", None), ("vocals", "high_quality"), ("drums", None)]
    """
    result: list[tuple[str, str | None]] = []
    
    for model in models:
        if model in ("vocals", "instrumental"):
            result.append((model, None))
            result.append((model, "high_quality"))
        else:
            result.append((model, None))
    
    return result


def build_upload_command(api_key: str, file_path: str) -> list[str]:
    """
    Build command list for uploading a file via audioshake_cli.py.
    
    Args:
        api_key: AudioShake API key
        file_path: Path to file to upload
        
    Returns:
        List of command arguments
    """
    return [
        sys.executable,
        "audioshake_cli.py",
        "--api-key", api_key,
        "--input", file_path,
    ]


def upload_file(api_key: str, file_path: str) -> str | None:
    """
    Upload a file via audioshake_cli.py and extract Asset ID.
    
    Args:
        api_key: AudioShake API key
        file_path: Path to file to upload
        
    Returns:
        Asset ID or None on error
    """
    cmd = build_upload_command(api_key, file_path)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for upload
        )
        
        if result.returncode != 0:
            print(f"Error uploading {file_path}: {result.stderr}", file=sys.stderr)
            return None
        
        asset_id = extract_asset_id(result.stdout)
        if not asset_id:
            print(f"Error: Could not extract Asset ID from output", file=sys.stderr)
            return None
        
        return asset_id
        
    except subprocess.TimeoutExpired:
        print(f"Error: Upload timed out for {file_path}", file=sys.stderr)
        return None
    except subprocess.SubprocessError as e:
        print(f"Error running subprocess: {e}", file=sys.stderr)
        return None


def build_model_command(
    api_key: str,
    asset_id: str,
    model: str,
    variant: str | None,
    output_dir: str,
    source_name: str | None = None,
) -> list[str]:
    """
    Build command list for running a model via audioshake_cli.py.
    
    Args:
        api_key: AudioShake API key
        asset_id: ID of the uploaded asset
        model: Model name to run
        variant: Model variant or None for standard
        output_dir: Output directory for results
        source_name: Source file name (without extension) for output naming
        
    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,
        "audioshake_cli.py",
        "--api-key", api_key,
        "--asset-id", asset_id,
        "--model", model,
        "--output", output_dir,
    ]
    
    if variant:
        cmd.extend(["--variant", variant])
    
    if source_name:
        cmd.extend(["--source-name", source_name])
    
    return cmd


def run_single_model(
    api_key: str,
    asset_id: str,
    model: str,
    variant: str | None,
    output_dir: str,
    source_name: str | None = None,
) -> bool:
    """
    Run a single model on an Asset ID via audioshake_cli.py.
    
    Args:
        api_key: AudioShake API key
        asset_id: ID of the uploaded asset
        model: Model name to run
        variant: Model variant or None for standard
        output_dir: Output directory for results
        source_name: Source file name (without extension) for output naming
        
    Returns:
        True if successful, False on error
    """
    cmd = build_model_command(api_key, asset_id, model, variant, output_dir, source_name)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for processing
        )
        
        if result.returncode != 0:
            variant_str = f" ({variant})" if variant else ""
            print(f"Error running model {model}{variant_str}: {result.stderr}", file=sys.stderr)
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        variant_str = f" ({variant})" if variant else ""
        print(f"Error: Processing timed out for model {model}{variant_str}", file=sys.stderr)
        return False
    except subprocess.SubprocessError as e:
        print(f"Error running subprocess: {e}", file=sys.stderr)
        return False


def process_files(
    api_key: str,
    wav_files: list[str],
    models: list[str],
    output_dir: str,
) -> tuple[int, int]:
    """
    Process all files: upload + run models.
    
    Args:
        api_key: AudioShake API key
        wav_files: List of paths to WAV files
        models: List of models to process
        output_dir: Output directory for results
        
    Returns:
        Tuple (successful_files, failed_files)
    """
    total_files = len(wav_files)
    successful_files = 0
    failed_files = 0
    
    # Expand models with variants (vocals/instrumental get high_quality variant too)
    expanded_models = expand_models_with_variants(models)
    total_models = len(expanded_models)
    
    for file_idx, file_path in enumerate(wav_files, start=1):
        filename = os.path.basename(file_path)
        # Extract source name without extension for output naming
        source_name = os.path.splitext(filename)[0]
        print(f"Uploading file {file_idx}/{total_files}: {filename}")
        
        # Upload file and get Asset ID
        asset_id = upload_file(api_key, file_path)
        
        if asset_id is None:
            print(f"  Upload error, skipping file")
            failed_files += 1
            continue
        
        print(f"  Asset ID: {asset_id}")
        
        # Track if any model failed for this file
        file_had_errors = False
        
        # Run all models on this asset
        for model_idx, (model, variant) in enumerate(expanded_models, start=1):
            variant_str = f" ({variant})" if variant else ""
            print(f"  Model {model_idx}/{total_models}: {model}{variant_str}")
            
            success = run_single_model(api_key, asset_id, model, variant, output_dir, source_name)
            
            if not success:
                file_had_errors = True
        
        if file_had_errors:
            failed_files += 1
        else:
            successful_files += 1
    
    return successful_files, failed_files


def main() -> int:
    """
    Main entry point.
    
    Parses arguments, creates output directory, scans for WAV files,
    processes all files through the specified models, and returns
    appropriate exit code.
    
    Returns:
        Exit code: 0 if all operations successful, 1 if any errors occurred
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        return 1
    
    # Read file list if provided
    file_list = None
    if args.file_list:
        try:
            file_list = read_file_list(args.file_list)
            print(f"Loaded file list from: {args.file_list}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Scan for WAV files
    try:
        wav_files = scan_wav_files(args.input, file_list)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    print(f"Found WAV files: {len(wav_files)}")
    print(f"Models to process: {', '.join(args.models_list)}")
    print(f"Output folder: {args.output}")
    print()
    
    # Process all files
    successful, failed = process_files(
        api_key=args.api_key,
        wav_files=wav_files,
        models=args.models_list,
        output_dir=args.output,
    )
    
    # Print final statistics
    print()
    print("=" * 50)
    print("Final statistics:")
    print(f"  Total files: {len(wav_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  With errors: {failed}")
    print("=" * 50)
    
    # Return exit code: 0 if all successful, 1 if any errors
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
