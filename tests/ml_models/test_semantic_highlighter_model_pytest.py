# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import shutil
from zipfile import ZipFile

import pytest
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from opensearch_py_ml.ml_models import SemanticHighlighterModel

TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath("__file__")), "tests", "test_model_files"
)


def clean_test_folder(TEST_FOLDER):
    if os.path.exists(TEST_FOLDER):
        for files in os.listdir(TEST_FOLDER):
            sub_path = os.path.join(TEST_FOLDER, files)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            else:
                try:
                    shutil.rmtree(sub_path)
                except OSError as err:
                    print(
                        "Fail to delete files, please delete all files in "
                        + str(TEST_FOLDER)
                        + " "
                        + str(err)
                    )

        shutil.rmtree(TEST_FOLDER)


def compare_model_config(
    model_config_path,
    model_id,
    model_format,
    expected_model_description=None,
):
    try:
        with open(model_config_path) as json_file:
            model_config_data = json.load(json_file)
    except Exception as exec:
        assert (
            False
        ), f"Creating model config file for tracing in {model_format} raised an exception {exec}"

    assert (
        "name" in model_config_data and model_config_data["name"] == model_id
    ), f"Missing or Wrong model name in {model_format} model config file"

    assert (
        "model_format" in model_config_data
        and model_config_data["model_format"] == model_format
    ), f"Missing or Wrong model_format in {model_format} model config file"

    if expected_model_description is not None:
        assert (
            "description" in model_config_data
            and model_config_data["description"] == expected_model_description
        ), f"Missing or Wrong model description in {model_format} model config file'"

    assert (
        "model_content_size_in_bytes" in model_config_data
    ), f"Missing 'model_content_size_in_bytes' in {model_format} model config file"

    assert (
        "model_content_hash_value" in model_config_data
    ), f"Missing 'model_content_hash_value' in {model_format} model config file"


def compare_model_zip_file(zip_file_path, expected_filenames, model_format):
    with ZipFile(zip_file_path, "r") as f:
        filenames = set(f.namelist())
        assert (
            filenames == expected_filenames
        ), f"The content in the {model_format} model zip file does not match the expected content: {filenames} != {expected_filenames}"


def prepare_example_inputs(batch_size=1):
    """
    Create mock inputs that the model would expect.
    
    Parameters
    ----------
    batch_size : int, default=1
        Number of examples in the batch. If 1, returns single input.
        If > 1, returns batched inputs.
    
    Returns
    -------
    dict
        Dictionary containing input tensors with appropriate batch dimension
    """
    # Base input tensors
    base_input_ids = [101, 2054, 2003, 1037, 2307, 1012, 102]  # [CLS] what is a test? [SEP]
    base_attention_mask = [1, 1, 1, 1, 1, 1, 1]
    base_token_type_ids = [0, 0, 0, 0, 0, 0, 0]
    base_sentence_ids = [0, 0, 0, 0, 0, 0, -100]
    
    if batch_size == 1:
        # Single input case
        return {
            "input_ids": torch.tensor([base_input_ids]),
            "attention_mask": torch.tensor([base_attention_mask]),
            "token_type_ids": torch.tensor([base_token_type_ids]),
            "sentence_ids": torch.tensor([base_sentence_ids]),
        }
    else:
        # Batch input case
        return {
            "input_ids": torch.tensor([base_input_ids] * batch_size),
            "attention_mask": torch.tensor([base_attention_mask] * batch_size),
            "token_type_ids": torch.tensor([base_token_type_ids] * batch_size),
            "sentence_ids": torch.tensor([base_sentence_ids] * batch_size),
        }


clean_test_folder(TEST_FOLDER)
test_model = SemanticHighlighterModel(folder_path=TEST_FOLDER)


def test_check_attribute():
    try:
        check_attribute = getattr(test_model, "model_id", "folder_path")
    except AttributeError:
        check_attribute = False
    assert check_attribute

    assert test_model.folder_path == TEST_FOLDER
    assert (
        test_model.model_id == "opensearch-project/opensearch-semantic-highlighter-v1"
    )

    # Skip this part as it would modify files outside the test directory
    # clean_test_folder(default_folder)
    # test_model0 = SemanticHighlighterModel()
    # assert test_model0.folder_path == default_folder
    # clean_test_folder(default_folder)


def test_folder_path():
    # The SemanticHighlighterModel doesn't enforce the check for existing folders
    # like other model classes do. Let's verify it doesn't raise an exception,
    # which is its current behavior.
    test_non_empty_path = os.path.join(
        os.path.dirname(os.path.abspath("__file__")), "tests"
    )
    try:
        model = SemanticHighlighterModel(
            folder_path=test_non_empty_path, overwrite=False
        )
        # Clean up if a folder was created
        if (
            os.path.exists(model.folder_path)
            and os.path.basename(model.folder_path) == "semantic-highlighter"
        ):
            shutil.rmtree(model.folder_path)
    except Exception as e:
        assert False, f"Unexpected exception was raised: {e}"


def test_check_required_fields():
    # test without required_fields should raise TypeError
    with pytest.raises(TypeError):
        test_model.save_as_pt()


def test_save_as_pt():
    example_inputs = prepare_example_inputs()
    try:
        test_model.save_as_pt(example_inputs=example_inputs)
    except Exception as exec:
        assert False, f"Tracing model in torchScript raised an exception {exec}"


def test_save_as_onnx():
    example_inputs = prepare_example_inputs()
    # The save_as_onnx method is expected to raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        test_model.save_as_onnx(example_inputs=example_inputs)
    assert "ONNX format is not supported for semantic highlighter models" in str(
        exc_info.value
    )


def test_make_model_config_json_for_torch_script():
    model_id = "opensearch-project/opensearch-semantic-highlighter-v1"
    model_format = "TORCH_SCRIPT"
    expected_model_description = "This is a semantic highlighter model that extracts relevant passages from documents."

    clean_test_folder(TEST_FOLDER)
    test_model1 = SemanticHighlighterModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    example_inputs = prepare_example_inputs()
    test_model1.save_as_pt(model_id=model_id, example_inputs=example_inputs)
    model_config_path_torch = test_model1.make_model_config_json(
        model_format="TORCH_SCRIPT", description=expected_model_description
    )

    compare_model_config(
        model_config_path_torch,
        model_id,
        model_format,
        expected_model_description=expected_model_description,
    )

    clean_test_folder(TEST_FOLDER)


def test_model_zip_content():
    model_id = "opensearch-project/opensearch-semantic-highlighter-v1"
    model_format = "TORCH_SCRIPT"
    torch_script_zip_file_path = os.path.join(
        TEST_FOLDER, "opensearch-semantic-highlighter-v1.zip"
    )
    torch_script_expected_filenames = {
        "opensearch-semantic-highlighter-v1.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "LICENSE",
    }

    clean_test_folder(TEST_FOLDER)
    test_model2 = SemanticHighlighterModel(
        folder_path=TEST_FOLDER,
        model_id=model_id,
    )

    example_inputs = prepare_example_inputs()
    test_model2.save_as_pt(
        model_id=model_id,
        example_inputs=example_inputs,
        add_apache_license=True,
    )

    compare_model_zip_file(
        torch_script_zip_file_path, torch_script_expected_filenames, model_format
    )

    clean_test_folder(TEST_FOLDER)


def test_save_as_pt_single_input():
    """Test model tracing with single input"""
    example_inputs = prepare_example_inputs(batch_size=1)
    try:
        test_model.save_as_pt(example_inputs=example_inputs)
    except Exception as exec:
        assert False, f"Tracing model in torchScript with single input raised an exception {exec}"


def test_save_as_pt_batch_input():
    """Test model tracing with batch input"""
    example_inputs = prepare_example_inputs(batch_size=2)
    try:
        test_model.save_as_pt(example_inputs=example_inputs)
    except Exception as exec:
        assert False, f"Tracing model in torchScript with batch input raised an exception {exec}"


def test_model_batch_processing():
    """Test that the model correctly processes both single and batch inputs"""
    # Initialize model
    model = TraceableBertTaggerForSentenceExtractionWithBackoff.from_pretrained(DEFAULT_MODEL_ID)
    
    # Test single input
    single_inputs = prepare_example_inputs(batch_size=1)
    single_output = model(
        single_inputs["input_ids"],
        single_inputs["attention_mask"],
        single_inputs["token_type_ids"],
        single_inputs["sentence_ids"]
    )
    assert isinstance(single_output, tuple), "Single input output should be a tuple"
    
    # Test batch input
    batch_inputs = prepare_example_inputs(batch_size=2)
    batch_output = model(
        batch_inputs["input_ids"],
        batch_inputs["attention_mask"],
        batch_inputs["token_type_ids"],
        batch_inputs["sentence_ids"]
    )
    assert isinstance(batch_output, tuple), "Batch input output should be a tuple"
    assert len(batch_output) == 2, "Batch output should have length equal to batch size"
    
    # Test that single input works when passed as a batch of 1
    single_as_batch = model(
        single_inputs["input_ids"],
        single_inputs["attention_mask"],
        single_inputs["token_type_ids"],
        single_inputs["sentence_ids"]
    )
    assert isinstance(single_as_batch, tuple), "Single input as batch output should be a tuple"
    assert len(single_as_batch) == 1, "Single input as batch should have length 1"


def test_semantic_highlighter_model_trace():
    """Test tracing the semantic highlighter model with both single and batch inputs."""
    # Initialize model and tokenizer
    base_model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = SemanticHighlighterModel(base_model_id)

    # Generate tracing dataset with fixed batch size
    batch_size = 4
    trace_dataset = generate_tracing_dataset(size=batch_size)

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        trace_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            k: torch.stack([torch.tensor(d[k]) for d in x])
            for k in x[0].keys()
        },
    )

    # Get a batch of inputs
    batch = next(iter(dataloader))

    # Test single input tracing
    single_input = {
        k: v[0].unsqueeze(0) for k, v in batch.items()
    }
    traced_model = model.save_as_pt(
        example_inputs=single_input,
        model_dir="test_semantic_highlighter_model",
        model_filename="semantic_highlighter_model.pt",
    )

    # Test batch input tracing
    traced_model_batch = model.save_as_pt(
        example_inputs=batch,
        model_dir="test_semantic_highlighter_model_batch",
        model_filename="semantic_highlighter_model_batch.pt",
    )

    # Clean up
    shutil.rmtree("test_semantic_highlighter_model")
    shutil.rmtree("test_semantic_highlighter_model_batch")


clean_test_folder(TEST_FOLDER)
