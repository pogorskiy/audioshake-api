#!/usr/bin/env python3
"""
Property-based tests for audioshake_cli.py using hypothesis.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioshake_cli import create_parser, get_mode, AudioShakeClient, AVAILABLE_MODELS


# Strategy for generating valid API keys (non-empty alphanumeric strings)
# Using lists to ensure min_size works correctly
api_key_strategy = st.lists(
    st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=8, max_size=64
).map(lambda chars: "".join(chars))

# Strategy for generating valid file paths (simple names without special chars)
filename_strategy = st.lists(
    st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=1, max_size=30
).map(lambda chars: "".join(chars))

# Strategy for valid model names from AVAILABLE_MODELS
model_strategy = st.sampled_from(list(AVAILABLE_MODELS.keys()))

# Strategy for valid output formats
format_strategy = st.sampled_from(["wav", "mp3", "flac"])

# Strategy for valid asset IDs
asset_id_strategy = st.lists(
    st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=8, max_size=50
).map(lambda chars: "".join(chars))


@given(
    api_key=api_key_strategy,
    input_file=filename_strategy,
    model=model_strategy,
    output_dir=filename_strategy,
    output_format=format_strategy,
)
@settings(max_examples=100)
def test_parser_accepts_all_required_parameters(api_key, input_file, model, output_dir, output_format):
    """
    Property 1: Argument Parser Accepts All Required Parameters.
    
    For any valid combination of CLI arguments (--api-key, --input, --model, --output, --format),
    the argument parser SHALL successfully parse them without errors.
    
    **Validates: Requirements 4.1**
    """
    parser = create_parser()
    
    # Build argument list with all parameters
    args_list = [
        "--api-key", api_key,
        "--input", f"/tmp/{input_file}.mp3",
        "--model", model,
        "--output", f"/tmp/{output_dir}",
        "--format", output_format,
    ]
    
    # Parse arguments - should not raise any exception
    args = parser.parse_args(args_list)
    
    # Verify all arguments are correctly parsed
    assert args.api_key == api_key, f"api_key mismatch: expected {api_key}, got {args.api_key}"
    assert args.input == f"/tmp/{input_file}.mp3", f"input mismatch"
    assert args.model == model, f"model mismatch: expected {model}, got {args.model}"
    assert args.output == f"/tmp/{output_dir}", f"output mismatch"
    assert args.format == output_format, f"format mismatch: expected {output_format}, got {args.format}"


@given(
    api_key=api_key_strategy,
    asset_id=asset_id_strategy,
    model=model_strategy,
    output_dir=filename_strategy,
)
@settings(max_examples=100)
def test_parser_accepts_asset_id_parameter(api_key, asset_id, model, output_dir):
    """
    Property 1 (asset-id variant): Parser accepts --asset-id instead of --input.
    
    For any valid combination with --asset-id, the parser SHALL successfully parse.
    
    **Validates: Requirements 4.1**
    """
    parser = create_parser()
    
    args_list = [
        "--api-key", api_key,
        "--asset-id", asset_id,
        "--model", model,
        "--output", f"/tmp/{output_dir}",
    ]
    
    args = parser.parse_args(args_list)
    
    assert args.api_key == api_key
    assert args.asset_id == asset_id
    assert args.model == model
    assert args.output == f"/tmp/{output_dir}"
    assert args.input is None  # --input should not be set


@given(
    api_key=api_key_strategy,
    input_file=filename_strategy,
)
@settings(max_examples=50)
def test_parser_accepts_upload_only_mode(api_key, input_file):
    """
    Property 1 (upload-only variant): Parser accepts --input without --model for upload-only mode.
    
    **Validates: Requirements 4.1**
    """
    parser = create_parser()
    
    args_list = [
        "--api-key", api_key,
        "--input", f"/tmp/{input_file}.wav",
    ]
    
    args = parser.parse_args(args_list)
    
    assert args.api_key == api_key
    assert args.input == f"/tmp/{input_file}.wav"
    assert args.model is None
    assert args.output is None


@given(
    api_key=api_key_strategy,
    input_file=filename_strategy,
    model=model_strategy,
    output_dir=filename_strategy,
    variant=st.sampled_from(["high_quality"]),
)
@settings(max_examples=50)
def test_parser_accepts_variant_parameter(api_key, input_file, model, output_dir, variant):
    """
    Property 1 (variant parameter): Parser accepts --variant parameter.
    
    **Validates: Requirements 4.1**
    """
    parser = create_parser()
    
    args_list = [
        "--api-key", api_key,
        "--input", f"/tmp/{input_file}.mp3",
        "--model", model,
        "--output", f"/tmp/{output_dir}",
        "--variant", variant,
    ]
    
    args = parser.parse_args(args_list)
    
    assert args.api_key == api_key
    assert args.variant == variant


# =============================================================================
# Property 2: Mode Selection Based on Arguments
# =============================================================================


@given(
    api_key=api_key_strategy,
    input_file=filename_strategy,
)
@settings(max_examples=100)
def test_mode_selection_upload_only(api_key, input_file):
    """
    Property 2: Mode Selection Based on Arguments - Upload Only Mode.
    
    For any set of CLI arguments where --input is provided without --model,
    the CLI SHALL correctly determine upload-only mode.
    
    **Validates: Requirements 4.3**
    """
    parser = create_parser()
    
    # Build args with --input but no --model
    args_list = [
        "--api-key", api_key,
        "--input", f"/tmp/{input_file}.mp3",
    ]
    
    args = parser.parse_args(args_list)
    
    # Mode should be "upload" when --input provided without --model
    mode = get_mode(args)
    assert mode == "upload", f"Expected 'upload' mode when --input without --model, got '{mode}'"


@given(
    api_key=api_key_strategy,
    input_file=filename_strategy,
    model=model_strategy,
    output_dir=filename_strategy,
)
@settings(max_examples=100)
def test_mode_selection_process_with_input(api_key, input_file, model, output_dir):
    """
    Property 2: Mode Selection Based on Arguments - Process Mode with Input.
    
    For any set of CLI arguments where --model is provided with --input,
    the CLI SHALL correctly determine process mode.
    
    **Validates: Requirements 4.3**
    """
    parser = create_parser()
    
    # Build args with --input and --model
    args_list = [
        "--api-key", api_key,
        "--input", f"/tmp/{input_file}.mp3",
        "--model", model,
        "--output", f"/tmp/{output_dir}",
    ]
    
    args = parser.parse_args(args_list)
    
    # Mode should be "process" when --model is provided
    mode = get_mode(args)
    assert mode == "process", f"Expected 'process' mode when --model with --input, got '{mode}'"


@given(
    api_key=api_key_strategy,
    asset_id=asset_id_strategy,
    model=model_strategy,
    output_dir=filename_strategy,
)
@settings(max_examples=100)
def test_mode_selection_process_with_asset_id(api_key, asset_id, model, output_dir):
    """
    Property 2: Mode Selection Based on Arguments - Process Mode with Asset ID.
    
    For any set of CLI arguments where --model is provided with --asset-id,
    the CLI SHALL correctly determine process mode.
    
    **Validates: Requirements 4.3**
    """
    parser = create_parser()
    
    # Build args with --asset-id and --model
    args_list = [
        "--api-key", api_key,
        "--asset-id", asset_id,
        "--model", model,
        "--output", f"/tmp/{output_dir}",
    ]
    
    args = parser.parse_args(args_list)
    
    # Mode should be "process" when --model is provided with --asset-id
    mode = get_mode(args)
    assert mode == "process", f"Expected 'process' mode when --model with --asset-id, got '{mode}'"


# =============================================================================
# Property 3: Output Files Match Task Response
# =============================================================================

# Strategy for generating output items in task response
output_item_strategy = st.fixed_dictionaries({
    "name": st.lists(
        st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
        min_size=3, max_size=15
    ).map(lambda chars: "".join(chars)),
    "format": st.sampled_from(["wav", "mp3", "flac"]),
    "type": st.sampled_from(["audio/wav", "audio/mpeg", "audio/flac"]),
    "link": st.just("https://cdn.audioshake.ai/test/output.wav"),
})

# Strategy for generating target items
target_strategy = st.fixed_dictionaries({
    "id": st.lists(
        st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
        min_size=10, max_size=20
    ).map(lambda chars: "".join(chars)),
    "model": model_strategy,
    "status": st.just("completed"),
    "output": st.lists(output_item_strategy, min_size=1, max_size=3),
})

# Strategy for generating complete task responses
task_response_strategy = st.fixed_dictionaries({
    "id": st.lists(
        st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
        min_size=10, max_size=30
    ).map(lambda chars: "".join(chars)),
    "targets": st.lists(target_strategy, min_size=1, max_size=3),
})


@given(task_response=task_response_strategy)
@settings(max_examples=50)
def test_download_outputs_file_count_matches_response(task_response):
    """
    Property 3: Output Files Match Task Response.
    
    For any completed task response containing output assets, the download function
    SHALL save exactly the same number of files as listed in the response to the
    specified output directory.
    
    **Validates: Requirements 2.2**
    """
    # Count total expected output files from task response
    expected_file_count = sum(
        len(target.get("output", []))
        for target in task_response.get("targets", [])
    )
    
    # Skip if no outputs (edge case)
    assume(expected_file_count > 0)
    
    # Create a temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create client (API key doesn't matter for this test)
        client = AudioShakeClient("test-api-key")
        
        # Mock the requests.get to return fake file content
        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b"fake audio content"]
        mock_response.raise_for_status = MagicMock()
        
        with patch("requests.get", return_value=mock_response):
            # Call download_outputs
            downloaded_files = client.download_outputs(task_response, temp_dir)
        
        # Property: number of downloaded files equals number of outputs in response
        assert len(downloaded_files) == expected_file_count, (
            f"Expected {expected_file_count} files, but downloaded {len(downloaded_files)}. "
            f"Task response had {len(task_response.get('targets', []))} targets."
        )
        
        # Verify all files actually exist on disk
        for filepath in downloaded_files:
            assert os.path.exists(filepath), f"Downloaded file does not exist: {filepath}"
        
        # Verify the count of files in the output directory matches
        actual_files_on_disk = [
            f for f in os.listdir(temp_dir) 
            if os.path.isfile(os.path.join(temp_dir, f))
        ]
        assert len(actual_files_on_disk) == expected_file_count, (
            f"Expected {expected_file_count} files on disk, found {len(actual_files_on_disk)}"
        )
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
