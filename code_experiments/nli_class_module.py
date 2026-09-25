from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BaseNliModel(ABC):
	def __init__(self, model_name: str, device: torch.device, **kwargs) -> None:
		self.device = device
		self.batch_size = kwargs.pop("batch_size")
		self.load_model(model_name, **kwargs)

	def batcher(self, p: list[str], h: list[str]):
		for batch_idx in range(0, len(p), self.batch_size):
			yield (
				p[batch_idx : batch_idx + self.batch_size],
				h[batch_idx : batch_idx + self.batch_size],
			)

	@abstractmethod
	def load_model(self, model_name: str, **kwargs):
		pass

	@abstractmethod
	def get_nli_label(
		self, premises: list[str], hypotheses: list[str]
	) -> tuple[list[str], list[float]]:
		pass


class NliModel:
	# factory class
	model_registry = {}

	@classmethod
	def register(cls, name: str) -> Callable:
		def inner_register(class_to_register: BaseNliModel) -> BaseNliModel:
			if name not in cls.model_registry:
				cls.model_registry[name] = class_to_register
			return class_to_register

		return inner_register

	@classmethod
	def create_model(cls, name, model_name, device, **kwargs) -> BaseNliModel:
		model_class = cls.model_registry[name]
		model_instance = model_class(model_name=model_name, device=device, **kwargs)
		return model_instance


@NliModel.register("finetuned_nli")
class FineTunedNli(BaseNliModel):
	model2hfrepo = {
		"albert-xxl": "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
		"deberta-large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
	}

	# !! @TODO: add logging
	def load_model(self, model_name: str, **kwargs):
		print(f"Loading {model_name}")
		model_repo_path = self.model2hfrepo[model_name]
		self.tokenizer = AutoTokenizer.from_pretrained(model_repo_path)
		self.model = AutoModelForSequenceClassification.from_pretrained(
			model_repo_path, device_map=self.device
		)

	@torch.no_grad()
	def get_nli_label(self, premises: list[str], hypotheses: list[str]):
		batches = self.batcher(premises, hypotheses)

		pred_labels = []
		pred_probs = []
		pred_all_probs = []
		for batch_premises, batch_hypothesis in batches:
			inputs = self.tokenizer(
				batch_premises,
				batch_hypothesis,
				return_tensors="pt",
				padding="longest",
				truncation=True,
			).to(self.device)
			outputs = self.model(**inputs)
			logits = outputs.logits

			pred_id = torch.softmax(logits, dim=1).argmax(dim=1).cpu().tolist()
			pred_id_probs = torch.softmax(logits, dim=1).cpu().max(dim=1).values.tolist()

			pred_labels += pred_id
			pred_probs += pred_id_probs
			pred_all_probs += torch.softmax(logits, dim=1).cpu().tolist()

		return pred_labels, pred_probs, pred_all_probs


# ? MAYBE USE LLM AS NLI RELATION PREDICTOR
@NliModel.register("llm_nli")
class LlmNli(BaseNliModel):
	def load_model(self, model_name: str, **kwargs):
		print(model_name)

	def get_nli_label(
		self, premises: list[str], hypotheses: list[str]
	) -> tuple[list[str], list[float]]:
		pass
