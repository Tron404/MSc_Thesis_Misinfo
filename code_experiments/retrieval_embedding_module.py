from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from sentence_transformers import SentenceTransformer


@dataclass
class InstanceSample:
	claim_text: str
	claim_embedding: torch.Tensor
	reports_content: Dict[int, List[int]]
	sentence_texts: Dict[int, Dict[str, Any]]  # inner dict: is_evidence, is_repeat, report_idx
	evidence_sentences_embeddings: torch.Tensor
	gt_explanation: str
	gt_label: str


@dataclass
class InstanceSampleNli(InstanceSample):
	nli_tuples: tuple


@dataclass
class InstanceSamplePredicted(InstanceSampleNli):
	pred_label: str


class EmbeddingModule:
	def __init__(
		self, model_name: str, device: str | torch.device | None, batch_size, **kwargs
	) -> None:
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		elif isinstance(device, torch.device) is True:
			device = device.type

		self.batch_size = batch_size
		self.embedding_model = SentenceTransformer(
			model_name_or_path=model_name, device=device, **kwargs
		)

	# @TODO: normalize from here?
	def __call__(self, sentences: List[str]) -> Any:
		return self.embedding_model.encode(
			sentences, batch_size=self.batch_size, convert_to_tensor=True
		)
