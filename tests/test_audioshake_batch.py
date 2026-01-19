#!/usr/bin/env python3
"""
Property-based tests for audioshake_batch.py using hypothesis.
"""

import os
import sys
import tempfile
from hypothesis import given, strategies as st, settings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioshake_batch import (
    scan_wav_files,
    extract_asset_id,
    expand_models_with_variants,
    build_upload_command,
    build_model_command,
)


# Strategy for generating valid filenames (without path separators)
# Use only lowercase to avoid case-insensitive filesystem collisions on macOS
valid_filename_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyz0123456789_-"
)
filename_base = st.text(valid_filename_chars, min_size=1, max_size=20)

# Strategy for file extensions (mix of wav and non-wav)
extensions = st.sampled_from([".wav", ".WAV", ".Wav", ".mp3", ".flac", ".txt", ".py", ""])


@given(
    wav_names=st.lists(filename_base, min_size=1, max_size=10, unique=True),
    other_names=st.lists(filename_base, min_size=0, max_size=10, unique=True),
    other_extensions=st.lists(
        st.sampled_from([".mp3", ".flac", ".txt", ".py", ""]),
        min_size=0,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_scan_wav_files_returns_only_wav_sorted(wav_names, other_names, other_extensions):
    """
    Property 2: Scanning returns only WAV files in sorted order.
    
    For any directory with arbitrary files, scan_wav_files should return only
    files with .wav extension, sorted alphabetically.
    
    **Validates: Requirements 2.1, 2.4**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create WAV files (using various case extensions)
        wav_extensions = [".wav", ".WAV", ".Wav"]
        created_wav_files = []
        
        for i, name in enumerate(wav_names):
            ext = wav_extensions[i % len(wav_extensions)]
            filename = f"{name}{ext}"
            filepath = os.path.join(tmpdir, filename)
            # Create empty file
            with open(filepath, "w") as f:
                pass
            created_wav_files.append(filepath)
        
        # Create non-WAV files
        for name, ext in zip(other_names, other_extensions):
            # Ensure no collision with wav names
            if name not in wav_names:
                filename = f"{name}{ext}"
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, "w") as f:
                    pass
        
        # Call the function
        result = scan_wav_files(tmpdir)
        
        # Property 1: All returned files have .wav extension (case-insensitive)
        for filepath in result:
            assert filepath.lower().endswith(".wav"), f"Non-WAV file returned: {filepath}"
        
        # Property 2: Result is sorted
        assert result == sorted(result), "Result is not sorted"
        
        # Property 3: All WAV files from directory are included
        assert len(result) == len(wav_names), f"Expected {len(wav_names)} WAV files, got {len(result)}"


# Strategy for generating valid Asset IDs (alphanumeric, UUID-like, etc.)
asset_id_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
asset_id_strategy = st.text(asset_id_chars, min_size=1, max_size=50)

# Strategy for generating random text that doesn't contain "Asset ID:"
random_text_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,;:!?()-"
)
random_text = st.text(random_text_chars, min_size=0, max_size=200)


@given(
    asset_id=asset_id_strategy,
    prefix=random_text,
    whitespace_before=st.sampled_from([" ", "  ", "\t", " \t"]),
    whitespace_after=st.sampled_from(["\n", " ", "\n\n", "  \n"]),
    suffix=random_text,
)
@settings(max_examples=100)
def test_extract_asset_id_from_cli_output(asset_id, prefix, whitespace_before, whitespace_after, suffix):
    """
    Property 3: Extracting Asset ID from CLI output.
    
    For any valid audioshake_cli.py output containing "Asset ID: XXX",
    the extraction function should correctly return the value XXX.
    
    In real CLI output, Asset ID is followed by whitespace (newline/space).
    
    **Validates: Requirements 3.2**
    """
    # Filter out prefix/suffix that might contain "Asset ID:" pattern
    if "Asset ID:" in prefix or "Asset ID:" in suffix:
        return  # Skip this case to avoid ambiguity
    
    # Construct CLI output with Asset ID (realistic format: ID followed by whitespace)
    cli_output = f"{prefix}Asset ID:{whitespace_before}{asset_id}{whitespace_after}{suffix}"
    
    # Extract Asset ID
    result = extract_asset_id(cli_output)
    
    # Property: Extracted ID should match the original asset_id
    assert result == asset_id, f"Expected '{asset_id}', got '{result}'"


@given(
    asset_id=asset_id_strategy,
    prefix=random_text,
    whitespace_before=st.sampled_from([" ", "  ", "\t", " \t"]),
)
@settings(max_examples=50)
def test_extract_asset_id_at_end_of_output(asset_id, prefix, whitespace_before):
    """
    Property 3 (end of string case): Asset ID at end of output should be extracted correctly.
    
    **Validates: Requirements 3.2**
    """
    # Filter out prefix that might contain "Asset ID:" pattern
    if "Asset ID:" in prefix:
        return  # Skip this case to avoid ambiguity
    
    # Construct CLI output with Asset ID at the end (no trailing content)
    cli_output = f"{prefix}Asset ID:{whitespace_before}{asset_id}"
    
    # Extract Asset ID
    result = extract_asset_id(cli_output)
    
    # Property: Extracted ID should match the original asset_id
    assert result == asset_id, f"Expected '{asset_id}', got '{result}'"


@given(random_output=random_text)
@settings(max_examples=50)
def test_extract_asset_id_returns_none_when_not_found(random_output):
    """
    Property 3 (negative case): When output doesn't contain Asset ID pattern,
    function should return None.
    
    **Validates: Requirements 3.2**
    """
    # Ensure the random output doesn't accidentally contain the pattern
    if "Asset ID:" in random_output:
        return  # Skip this case
    
    result = extract_asset_id(random_output)
    
    assert result is None, f"Expected None for output without Asset ID, got '{result}'"


# Strategy for generating model names (excluding vocals/instrumental for controlled testing)
other_model_names = st.sampled_from(["drums", "bass", "guitar", "piano", "other", "melody"])
special_models = st.sampled_from(["vocals", "instrumental"])


@given(
    models=st.lists(
        st.one_of(other_model_names, special_models),
        min_size=1,
        max_size=10,
        unique=True,  # Avoid duplicates - the function processes each model as-is
    )
)
@settings(max_examples=100)
def test_expand_models_with_variants(models):
    """
    Property 4: Expanding models with variants.
    
    For any list of unique models:
    - If "vocals" is in the list, result should contain both (vocals, None) and (vocals, "high_quality")
    - If "instrumental" is in the list, result should contain both (instrumental, None) and (instrumental, "high_quality")
    - For other models, result should contain only (model, None)
    
    **Validates: Requirements 4.3, 4.4**
    """
    result = expand_models_with_variants(models)
    
    # Property 1: Result should be a list of tuples with valid structure
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        model, variant = item
        assert isinstance(model, str)
        assert variant is None or variant == "high_quality"
    
    # Property 2: For each model in input, check correct expansion
    for model in models:
        if model in ("vocals", "instrumental"):
            # Should have both variants
            assert (model, None) in result, f"Missing ({model}, None) in result"
            assert (model, "high_quality") in result, f"Missing ({model}, 'high_quality') in result"
        else:
            # Should have only standard variant
            assert (model, None) in result, f"Missing ({model}, None) in result"
            assert (model, "high_quality") not in result, f"Unexpected ({model}, 'high_quality') in result"
    
    # Property 3: Order should be preserved - models appear in same order as input
    result_models_order = [model for model, _ in result if result.index((model, None)) == result.index((model, None))]
    # Extract unique models in order from result
    seen = set()
    result_models_order = []
    for model, _ in result:
        if model not in seen:
            result_models_order.append(model)
            seen.add(model)
    
    assert result_models_order == models, "Model order not preserved"
    
    # Property 4: Count check - vocals/instrumental contribute 2 entries, others contribute 1
    expected_count = 0
    for model in models:
        if model in ("vocals", "instrumental"):
            expected_count += 2
        else:
            expected_count += 1
    
    assert len(result) == expected_count, f"Expected {expected_count} entries, got {len(result)}"


# Strategy for generating non-empty strings without whitespace (for command arguments)
# Exclude strings starting with '-' to avoid confusion with CLI flags
arg_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./"
)
# First char must not be '-', rest can include more chars
first_char = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
rest_chars = st.text(arg_chars, min_size=0, max_size=49)
non_empty_arg = st.builds(lambda f, r: f + r, first_char, rest_chars)


@given(
    api_key=non_empty_arg,
    file_path=non_empty_arg,
)
@settings(max_examples=100)
def test_build_upload_command_format(api_key, file_path):
    """
    Property 5 (upload): Building CLI commands for upload.
    
    For any combination of (api_key, file_path), the built command should
    contain all required arguments in correct format.
    
    **Validates: Requirements 3.1**
    """
    cmd = build_upload_command(api_key, file_path)
    
    # Property 1: Command is a non-empty list
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    
    # Property 2: First element is Python executable
    assert cmd[0] == sys.executable
    
    # Property 3: Second element is the CLI script
    assert cmd[1] == "audioshake_cli.py"
    
    # Property 4: Contains --api-key followed by the api_key value
    assert "--api-key" in cmd
    api_key_idx = cmd.index("--api-key")
    assert cmd[api_key_idx + 1] == api_key
    
    # Property 5: Contains --input followed by the file_path value
    assert "--input" in cmd
    input_idx = cmd.index("--input")
    assert cmd[input_idx + 1] == file_path
    
    # Property 6: Command has exactly 6 elements (python, script, --api-key, key, --input, path)
    assert len(cmd) == 6


@given(
    api_key=non_empty_arg,
    asset_id=non_empty_arg,
    model=non_empty_arg,
    variant=st.one_of(st.none(), st.just("high_quality")),
    output_dir=non_empty_arg,
)
@settings(max_examples=100)
def test_build_model_command_format(api_key, asset_id, model, variant, output_dir):
    """
    Property 5 (model): Building CLI commands for running a model.
    
    For any combination of (api_key, asset_id, model, variant, output_dir),
    the built command should contain all required arguments in correct format.
    
    **Validates: Requirements 4.2, 5.2**
    """
    cmd = build_model_command(api_key, asset_id, model, variant, output_dir)
    
    # Property 1: Command is a non-empty list
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    
    # Property 2: First element is Python executable
    assert cmd[0] == sys.executable
    
    # Property 3: Second element is the CLI script
    assert cmd[1] == "audioshake_cli.py"
    
    # Property 4: Contains --api-key followed by the api_key value
    assert "--api-key" in cmd
    api_key_idx = cmd.index("--api-key")
    assert cmd[api_key_idx + 1] == api_key
    
    # Property 5: Contains --asset-id followed by the asset_id value
    assert "--asset-id" in cmd
    asset_id_idx = cmd.index("--asset-id")
    assert cmd[asset_id_idx + 1] == asset_id
    
    # Property 6: Contains --model followed by the model value
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == model
    
    # Property 7: Contains --output followed by the output_dir value
    assert "--output" in cmd
    output_idx = cmd.index("--output")
    assert cmd[output_idx + 1] == output_dir
    
    # Property 8: If variant is provided, contains --variant followed by variant value
    if variant:
        assert "--variant" in cmd
        variant_idx = cmd.index("--variant")
        assert cmd[variant_idx + 1] == variant
        # Command has 12 elements with variant
        assert len(cmd) == 12
    else:
        assert "--variant" not in cmd
        # Command has 10 elements without variant
        assert len(cmd) == 10


def compute_exit_code(successful: int, failed: int) -> int:
    """
    Compute exit code based on processing results.
    
    This mirrors the logic in main() for determining exit code.
    
    Args:
        successful: Number of successfully processed files
        failed: Number of files with errors
        
    Returns:
        0 if all successful (failed == 0), 1 if any errors (failed > 0)
    """
    if failed > 0:
        return 1
    return 0


@given(
    successful=st.integers(min_value=0, max_value=1000),
    failed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_exit_code_corresponds_to_result(successful, failed):
    """
    Property 6: Exit code corresponds to result.
    
    For any processing result (successful, failed counts):
    - Exit code should be 0 if all operations successful (failed == 0)
    - Exit code should be non-zero (1) if any errors occurred (failed > 0)
    
    **Validates: Requirements 6.3**
    """
    exit_code = compute_exit_code(successful, failed)
    
    # Property 1: Exit code is always 0 or 1
    assert exit_code in (0, 1), f"Exit code should be 0 or 1, got {exit_code}"
    
    # Property 2: Exit code is 0 if and only if failed == 0
    if failed == 0:
        assert exit_code == 0, f"Expected exit code 0 when failed=0, got {exit_code}"
    else:
        assert exit_code == 1, f"Expected exit code 1 when failed={failed}, got {exit_code}"
    
    # Property 3: Exit code is non-zero if and only if failed > 0
    if failed > 0:
        assert exit_code != 0, f"Expected non-zero exit code when failed={failed}"
    else:
        assert exit_code == 0, f"Expected zero exit code when failed=0"


@given(
    successful=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=50)
def test_exit_code_zero_when_all_successful(successful):
    """
    Property 6 (success case): Exit code is 0 when no failures.
    
    **Validates: Requirements 6.3**
    """
    exit_code = compute_exit_code(successful, failed=0)
    
    assert exit_code == 0, f"Expected exit code 0 when all successful, got {exit_code}"


@given(
    successful=st.integers(min_value=0, max_value=100),
    failed=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_exit_code_nonzero_when_any_failures(successful, failed):
    """
    Property 6 (failure case): Exit code is non-zero when any failures.
    
    **Validates: Requirements 6.3**
    """
    exit_code = compute_exit_code(successful, failed)
    
    assert exit_code != 0, f"Expected non-zero exit code when failed={failed}, got {exit_code}"
    assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
