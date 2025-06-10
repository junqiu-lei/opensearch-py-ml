# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from typing import List
from zipfile import ZipFile

import torch
from torch import nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel

from opensearch_py_ml.ml_commons.ml_common_utils import (
    _generate_model_content_hash_value,
)
from opensearch_py_ml.ml_models.base_models import BaseUploadModel

DEFAULT_MODEL_ID = "opensearch-project/opensearch-semantic-highlighter-v1"


class TraceableBertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """
    A torch.jit-compatible version of the sentence highlighter model for inference.

    This model extends BERT to perform sentence-level tagging with a backoff mechanism
    that ensures at least one sentence is selected when confidence exceeds a minimum
    threshold (alpha=0.05). Supports both single and batch inputs.
    """

    def __init__(self, config):
        """
        Initialize the model with BERT base and a classification head.

        Parameters
        ----------
        config : BertConfig
            Configuration object containing model hyperparameters
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
    ):
        """
        Forward pass of the model for sentence highlighting.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of input sequences (shape: [batch_size, seq_length])
        attention_mask : torch.Tensor
            Mask to avoid attention on padding tokens (shape: [batch_size, seq_length])
        token_type_ids : torch.Tensor
            Segment token indices for input portions (shape: [batch_size, seq_length])
        sentence_ids : torch.Tensor
            IDs assigning tokens to sentences (shape: [batch_size, seq_length])

        Returns
        -------
        tuple
            Indices of sentences to highlight for each item in the batch
        """
        # Input validation
        if input_ids is None or attention_mask is None or token_type_ids is None or sentence_ids is None:
            raise ValueError("All input tensors must be provided")

        # Ensure inputs are properly batched
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if token_type_ids.dim() == 1:
            token_type_ids = token_type_ids.unsqueeze(0)
        if sentence_ids.dim() == 1:
            sentence_ids = sentence_ids.unsqueeze(0)

        # Validate batch dimensions match
        batch_size = input_ids.size(0)
        if not all(t.size(0) == batch_size for t in [attention_mask, token_type_ids, sentence_ids]):
            raise ValueError("All input tensors must have the same batch size")

        # Validate sequence lengths match
        seq_length = input_ids.size(1)
        if not all(t.size(1) == seq_length for t in [attention_mask, token_type_ids, sentence_ids]):
            raise ValueError("All input tensors must have the same sequence length")

        # Pass inputs through the BERT model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get token-level embeddings and apply dropout
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        @torch.jit.script_if_tracing
        def _get_agg_output(ids, sequence_output):
            """
            Aggregates token-level embeddings into sentence-level embeddings.

            Parameters
            ----------
            ids : torch.Tensor
                Tensor containing sentence IDs for each token
            sequence_output : torch.Tensor
                Token-level embeddings from the BERT model

            Returns
            -------
            tuple
                Contains aggregated sentence embeddings, offsets, and sentence counts
            """
            max_sentences = torch.max(ids) + 1
            d_model = sequence_output.shape[-1]

            agg_output = []
            global_offset_per_item = []
            num_sentences_per_item = []
            for i, sen_ids in enumerate(ids):
                out = []
                local_sen_ids = sen_ids.clone()
                mask = local_sen_ids != -100
                offset = local_sen_ids[mask].min()
                global_offset_per_item.append(offset)
                local_sen_ids[mask] = local_sen_ids[mask] - offset
                num_sentences = torch.max(local_sen_ids) + 1
                num_sentences_per_item.append(num_sentences)

                for j in range(int(num_sentences)):
                    out.append(
                        sequence_output[i, local_sen_ids == j].mean(
                            dim=-2, keepdim=True
                        )
                    )
                padding = torch.zeros(
                    (int(max_sentences - num_sentences), d_model),
                    device=sequence_output.device,
                )
                out.append(padding)
                out = torch.cat(out, dim=0)
                agg_output.append(out)
            agg_output = torch.stack(agg_output)
            return (agg_output, global_offset_per_item, num_sentences_per_item)

        # Aggregate token embeddings into sentence embeddings
        agg_output, global_offset_per_item, num_sentences_per_item = _get_agg_output(
            sentence_ids, sequence_output
        )
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)
        pos_probs = probs[:, :, 1]

        @torch.jit.script_if_tracing
        def _get_sentence_preds(
            pos_probs,
            global_offset_per_item: List[torch.Tensor],
            num_sentences_per_item: List[torch.Tensor],
            threshold: float = 0.5,
            alpha: float = 0.05,
        ):
            """
            Converts sentence probabilities into predictions with backoff logic.

            Parameters
            ----------
            pos_probs : torch.Tensor
                Positive class probabilities for each sentence
            global_offset_per_item : List[torch.Tensor]
                Minimum sentence ID for each batch item
            num_sentences_per_item : List[torch.Tensor]
                Number of sentences for each batch item
            threshold : float, default=0.5
                Probability threshold for sentence selection
            alpha : float, default=0.05
                Minimum confidence threshold for backoff selection

            Returns
            -------
            List[torch.Tensor]
                List of selected sentence indices for each item in the batch
            """
            sentences_preds = []
            for i, (probs_i, offset_i, num_sentences_i) in enumerate(
                zip(pos_probs, global_offset_per_item, num_sentences_per_item)
            ):
                # Get predictions above threshold
                preds = torch.where(probs_i > threshold)[0]
                
                # If no predictions above threshold, use backoff
                if len(preds) == 0:
                    max_prob, max_idx = torch.max(probs_i, dim=0)
                    if max_prob > alpha:
                        preds = torch.tensor([max_idx], device=probs_i.device)
                
                # Adjust indices by offset
                preds = preds + offset_i
                sentences_preds.append(preds)
            
            return sentences_preds

        return _get_sentence_preds(
            pos_probs, global_offset_per_item, num_sentences_per_item
        )


class SemanticHighlighterModel(BaseUploadModel):
    """
    Model class for preparing and packaging the OpenSearch semantic highlighter.

    This class handles model conversion to TorchScript, packaging model with tokenizer,
    and generating configuration for OpenSearch ML Commons deployment.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize a SemanticHighlighterModel instance.

        Parameters
        ----------
        model_id : str, optional
            The Hugging Face model ID to use
        folder_path : str, optional
            Directory path to save model files and configuration
        overwrite : bool, optional
            Whether to overwrite existing files
        """
        if folder_path is None:
            folder_path = "semantic-highlighter/"

        super().__init__(
            model_id=model_id, folder_path=folder_path, overwrite=overwrite
        )
        # Path to the generated zip file, populated after calling save_as_pt
        self.torch_script_zip_file_path = None

    def save_as_pt(self, example_inputs, model_dir, model_filename):
        """
        Save the model as a traced PyTorch model.

        Parameters
        ----------
        example_inputs : dict
            Dictionary containing example inputs for tracing. Must include:
            - input_ids: Tensor of shape (batch_size, sequence_length)
            - attention_mask: Tensor of shape (batch_size, sequence_length)
            - token_type_ids: Tensor of shape (batch_size, sequence_length)
            - sentence_ids: Tensor of shape (batch_size, sequence_length)
        model_dir : str
            Directory to save the model in
        model_filename : str
            Name of the model file

        Returns
        -------
        torch.jit.ScriptModule
            The traced model
        """
        # Validate input tensors
        required_keys = ["input_ids", "attention_mask", "token_type_ids", "sentence_ids"]
        for key in required_keys:
            if key not in example_inputs:
                raise ValueError(f"Missing required input tensor: {key}")
            if not isinstance(example_inputs[key], torch.Tensor):
                raise ValueError(f"Input {key} must be a torch.Tensor")

        # Validate batch dimensions
        batch_size = example_inputs["input_ids"].size(0)
        for key in required_keys:
            if example_inputs[key].size(0) != batch_size:
                raise ValueError(f"All input tensors must have the same batch size. Got {key} with batch size {example_inputs[key].size(0)}")

        # Validate sequence lengths
        seq_length = example_inputs["input_ids"].size(1)
        for key in required_keys:
            if example_inputs[key].size(1) != seq_length:
                raise ValueError(f"All input tensors must have the same sequence length. Got {key} with sequence length {example_inputs[key].size(1)}")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Trace the model
        traced_model = torch.jit.trace(self, example_inputs)

        # Save the traced model
        model_path = os.path.join(model_dir, model_filename)
        torch.jit.save(traced_model, model_path)

        return traced_model

    def save_as_onnx(
        self,
        example_inputs: dict,
        model_id: str = DEFAULT_MODEL_ID,
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = True,
    ) -> str:
        """
        ONNX format is not supported for semantic highlighter models.
        This method is implemented to satisfy the BaseUploadModel interface.

        Raises
        ------
        NotImplementedError
            Always raises this error as ONNX format is not supported.
        """
        raise NotImplementedError(
            "ONNX format is not supported for semantic highlighter models"
        )

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = "1.0.0",
        model_format: str = "TORCH_SCRIPT",
        description: str = None,
        model_zip_file_path: str = None,
    ) -> str:
        """
        Create the model configuration file for OpenSearch ML Commons.

        Parameters
        ----------
        model_name : str, optional
            Name of the model for OpenSearch
        version_number : str, optional
            Version of the model
        model_format : str, optional
            Format of the model
        description : str, optional
            Model description
        model_zip_file_path : str, optional
            Path to the model zip file

        Returns
        -------
        str
            Path to the created config file
        """
        # Use model_id as the model name if none provided
        if model_name is None:
            model_name = self.model_id

        model_config_content = {
            "name": model_name,
            "version": version_number,
            "model_format": model_format,
            "function_name": "QUESTION_ANSWERING",
            "description": (
                description
                if description
                else "A semantic highlighter model that identifies relevant sentences in a document given a query."
            ),
            "model_config": {
                "model_type": "sentence_highlighting",
                "framework_type": "huggingface_transformers",
            },
        }

        if model_zip_file_path is None:
            model_zip_file_path = self.torch_script_zip_file_path

        if model_zip_file_path:
            model_config_content["model_content_size_in_bytes"] = os.stat(
                model_zip_file_path
            ).st_size
            model_config_content["model_content_hash_value"] = (
                _generate_model_content_hash_value(model_zip_file_path)
            )

        model_config_file_path = os.path.join(
            self.folder_path, "ml-commons_model_config.json"
        )
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)

        with open(model_config_file_path, "w") as file:
            json.dump(model_config_content, file, indent=4)

        print(f"Model config file saved at: {model_config_file_path}")
        return model_config_file_path

    def _test_traced_model(self, traced_model_cpu, original_inputs, model_path):
        """
        Test the traced model on both CPU and GPU to ensure compatibility.
        
        Parameters
        ----------
        traced_model_cpu : torch.jit.ScriptModule
            The traced model on CPU
        original_inputs : dict
            Original inputs used for tracing
        model_path : str
            Path where the model was saved
        """
        print("🧪 Testing traced model compatibility...")
        
        # Test on CPU
        try:
            loaded_model_cpu = torch.jit.load(model_path, map_location=torch.device("cpu"))
            cpu_inputs = {k: v.cpu() for k, v in original_inputs.items()}
            
            cpu_output = loaded_model_cpu(
                cpu_inputs["input_ids"],
                cpu_inputs["attention_mask"], 
                cpu_inputs["token_type_ids"],
                cpu_inputs["sentence_ids"]
            )
            print("✅ CPU inference test passed")
            
        except Exception as e:
            print(f"❌ CPU inference test failed: {e}")
            raise
        
        # Test on GPU (if available)
        if torch.cuda.is_available():
            try:
                loaded_model_gpu = torch.jit.load(model_path, map_location=torch.device("cuda"))
                gpu_inputs = {k: v.cuda() for k, v in original_inputs.items()}
                
                gpu_output = loaded_model_gpu(
                    gpu_inputs["input_ids"],
                    gpu_inputs["attention_mask"],
                    gpu_inputs["token_type_ids"], 
                    gpu_inputs["sentence_ids"]
                )
                print("✅ GPU inference test passed")
                
                # Compare outputs (move GPU output to CPU for comparison)
                gpu_output_cpu = tuple(tensor.cpu() for tensor in gpu_output)
                if len(cpu_output) == len(gpu_output_cpu):
                    print("✅ CPU and GPU outputs have matching structure")
                else:
                    print("⚠️  CPU and GPU outputs have different structures")
                    
            except Exception as e:
                print(f"❌ GPU inference test failed: {e}")
                print("⚠️  Model may not work properly on GPU")
        else:
            print("⏭️  GPU not available, skipping GPU test")
