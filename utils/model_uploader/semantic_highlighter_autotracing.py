# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Semantic Highlighter Auto-tracing Utility for OpenSearch

This module provides functionality for auto-tracing semantic highlighter models
for deployment in OpenSearch. It handles creating example inputs, tracing the model,
packaging it for upload, and optionally testing the deployment in a test environment.

Key components:
- Example input generation for model tracing
- Model tracing to TorchScript format
- Configuration file generation
- Optional deployment testing
- Preparing files for upload to OpenSearch model repository

This utility is used to prepare semantic highlighter models for efficient deployment
in OpenSearch ML Commons.
"""

import argparse
import json
import os
import sys
from typing import Optional

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from functools import partial

import nltk
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models import SemanticHighlighterModel
from tests import OPENSEARCH_TEST_CLIENT
from utils.model_uploader.autotracing_utils import (
    QUESTION_ANSWERING_ALGORITHM,
    TORCH_SCRIPT_FORMAT,
    autotracing_warning_filters,
    check_model_status,
    prepare_files_for_uploading,
    preview_model_config,
    register_and_deploy_model,
    store_description_variable,
    store_license_verified_variable,
    verify_license_by_hfapi,
)


def prepare_train_features(
    tokenizer, examples, max_seq_length=512, stride=128, padding=False
):
    """
    Prepare tokenized training features for the semantic highlighter model.

    This function tokenizes the input examples and extracts sentence-level information
    required for training and tracing the semantic highlighter model.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer to use for processing text
    examples : dict
        Dictionary containing questions, contexts, and sentence annotation data
    max_seq_length : int, default=512
        Maximum sequence length for tokenization
    stride : int, default=128
        Stride length for tokenization with overlap
    padding : bool, default=False
        Whether to pad sequences to max_seq_length

    Returns
    -------
    dict
        Dictionary containing tokenized and processed features including:
        - input_ids, attention_mask, token_type_ids: standard BERT inputs
        - sentence_ids: token-level sentence IDs
        - sentence_labels: binary labels for sentences
        - example_id: example identifiers
    """
    # jointly tokenize questions and context
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=padding,
        is_split_into_words=True,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # Create data structures to hold processed features
    tokenized_examples["example_id"] = []
    tokenized_examples["word_ids"] = []
    tokenized_examples["sentence_ids"] = []
    tokenized_examples["answer_sentence_ids"] = []
    tokenized_examples["sentence_labels"] = []

    for i, sample_index in enumerate(sample_mapping):
        # Get word ids for current feature
        word_ids = tokenized_examples.word_ids(i)
        # Get marked answer sentences from original data
        answer_ids = set(np.where(examples["orig_sentence_labels"][sample_index])[0])
        # Get sentence mappings for each word
        word_level_sentence_ids = examples["word_level_sentence_ids"][sample_index]

        # Identify the context start position (after question tokens)
        sequence_ids = tokenized_examples.sequence_ids(i)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # Map each token to its corresponding sentence id
        # Use -100 for special tokens and question tokens
        sentences_ids = [-100] * token_start_index
        for word_idx in word_ids[token_start_index:]:
            if word_idx is not None:
                sentences_ids.append(word_level_sentence_ids[word_idx])
            else:
                sentences_ids.append(-100)

        sentence_labels = [0] * (max(sentences_ids) + 1)
        answer_ids = set()
        for sentence_id in sentences_ids:
            if (
                sentence_id >= 0
                and examples["orig_sentence_labels"][sample_index][sentence_id] == 1
            ):
                sentence_labels[sentence_id] = 1
                answer_ids.add(sentence_id)

        tokenized_examples["sentence_ids"].append(sentences_ids)
        tokenized_examples["sentence_labels"].append(sentence_labels)
        tokenized_examples["answer_sentence_ids"].append(answer_ids)
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["word_ids"].append(word_ids)

    return tokenized_examples


def generate_tracing_dataset(size=10):
    """
    Generate a sample dataset for tracing the semantic highlighter model.

    This function creates a small example dataset with a question and context passage
    about OpenSearch highlighting, with sentence-level annotations. The dataset is
    processed with the tokenizer to generate inputs suitable for model tracing.

    Parameters
    ----------
    size : int, default=10
        Number of examples to generate in the dataset (batch size)

    Returns
    -------
    datasets.Dataset
        A processed dataset containing tokenized inputs ready for model tracing,
        including input_ids, attention_mask, token_type_ids, and sentence_ids.
    """
    # Define a question and corresponding passage about OpenSearch highlighting
    question = "When does OpenSearch use text reanalysis for highlighting?"
    passage = "To highlight the search terms, the highlighter needs the start and end character offsets of each term. The offsets mark the term's position in the original text. The highlighter can obtain the offsets from the following sources: Postings: When documents are indexed, OpenSearch creates an inverted search index—a core data structure used to search for documents. Postings represent the inverted search index and store the mapping of each analyzed term to the list of documents in which it occurs. If you set the index_options parameter to offsets when mapping a text field, OpenSearch adds each term's start and end character offsets to the inverted index. During highlighting, the highlighter reruns the original query directly on the postings to locate each term. Thus, storing offsets makes highlighting more efficient for large fields because it does not require reanalyzing the text. Storing term offsets requires additional disk space, but uses less disk space than storing term vectors. Text reanalysis: In the absence of both postings and term vectors, the highlighter reanalyzes text in order to highlight it. For every document and every field that needs highlighting, the highlighter creates a small in-memory index and reruns the original query through Lucene's query execution planner to access low-level match information for the current document. Reanalyzing the text works well in most use cases. However, this method is more memory and time intensive for large fields."

    # Split passage into words and assign sentence IDs to each word
    sentence_ids = []
    context = []
    passage_sents = nltk.sent_tokenize(passage)
    for sent_id, sent in enumerate(passage_sents):
        sent_words = sent.split(" ")
        context += sent_words
        sentence_ids += [sent_id] * len(sent_words)

    # Mark the relevant sentence (sentence 8 contains the answer)
    orig_sentence_labels = [0] * len(passage_sents)
    orig_sentence_labels[8] = 1

    # Create dataset with the question, context and sentence annotations
    # Repeat the example to create a batch of the specified size
    trace_dataset = Dataset.from_dict(
        {
            "question": [[question]] * size,
            "context": [context] * size,
            "word_level_sentence_ids": [sentence_ids] * size,
            "orig_sentence_labels": [orig_sentence_labels] * size,
            "id": [f"test_{i}" for i in range(size)],
        }
    )

    # Initialize tokenizer and process the dataset
    base_model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    preprocess_fn = partial(prepare_train_features, tokenizer)
    trace_dataset = trace_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=trace_dataset.column_names,
        desc="Preparing model inputs",
    )
    return trace_dataset


def main(
    model_id: str,
    version: str,
    model_format: str,
    upload: bool = False,
    model_name: str = None,
    model_description: str = "",
):
    """
    Main function to trace and optionally upload the semantic highlighter model.

    Parameters
    ----------
    model_id : str
        Model ID to use from Hugging Face
    version : str
        Version of the model
    model_format : str
        Format to save the model in (TORCH_SCRIPT or ONNX)
    upload : bool, optional
        Whether to upload the model to OpenSearch
    model_name : str, optional
        Name for the traced model file
    model_description : str, optional
        Description of the model
    """
    # Initialize model and tokenizer
    base_model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = SemanticHighlighterModel(base_model_id)

    # Generate tracing dataset
    trace_dataset = generate_tracing_dataset()

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        trace_dataset,
        batch_size=1,  # Use batch size 1 for tracing
        shuffle=False,
        collate_fn=lambda x: {
            k: torch.stack([torch.tensor(d[k]) for d in x])
            for k in x[0].keys()
        },
    )

    # Get a batch of inputs
    batch = next(iter(dataloader))

    # Generate default model name if not provided
    if model_name is None:
        model_name = str(model_id.split("/")[-1] + ".pt")

    # Create output directories
    model_dir = os.path.join(model.folder_path, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Trace and save the model
    torchscript_model_path = model.save_as_pt(
        example_inputs=batch,
        model_dir=model_dir,
        model_filename=model_name,
    )

    # Save tokenizer files
    tokenizer_path = os.path.join(model.folder_path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer files saved to {tokenizer_path}")

    # Create zip file with model and tokenizer
    zip_file_name = str(model_id.split("/")[-1] + ".zip")
    zip_file_path = os.path.join(model.folder_path, zip_file_name)
    with ZipFile(str(zip_file_path), "w") as zipObj:
        model_path = os.path.join(model_dir, model_name)
        zipObj.write(model_path, arcname=str(model_name))

        for file in os.listdir(tokenizer_path):
            file_path = os.path.join(tokenizer_path, file)
            zipObj.write(file_path, arcname=file)

    # Add Apache license if needed
    model._add_apache_license_to_model_zip_file(zip_file_path)

    model.torch_script_zip_file_path = zip_file_path
    print(f"Zip file saved to {zip_file_path}")

    if upload:
        # Initialize MLCommonClient for deployment
        print("--- Initializing MLCommonClient for deployment test ---")
        ml_client = MLCommonClient()

        # Verify license
        print(f"--- Verifying license for model {model_id} ---")
        license = ml_client.verify_model_license(model_id)
        print(f"License verified as {license}.")

        # Upload model
        print("--- Uploading model to OpenSearch ---")
        ml_client.upload_model(
            model_id=model_id,
            version=version,
            model_format=model_format,
            model_path=zip_file_path,
            model_name=model_name,
            model_description=model_description,
        )
        print("Model uploaded successfully.")

    return zip_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace semantic highlighter model")
    parser.add_argument("model_id", type=str, help="Model ID to use from Hugging Face")
    parser.add_argument("version", type=str, help="Version of the model")
    parser.add_argument("model_format", type=str, help="Format to save the model in (TORCH_SCRIPT or ONNX)")
    parser.add_argument("-up", "--upload", action="store_true", help="Whether to upload the model to OpenSearch")
    parser.add_argument("-mn", "--model_name", type=str, help="Name for the traced model file")
    parser.add_argument("-md", "--model_description", type=str, default="", help="Description of the model")
    args = parser.parse_args()

    main(
        model_id=args.model_id,
        version=args.version,
        model_format=args.model_format,
        upload=args.upload,
        model_name=args.model_name,
        model_description=args.model_description,
    )
