import glob
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from retrieval_embedding_module import InstanceSampleNli
from torch.utils.data import Dataset


class InstanceDataset(ABC, Dataset):
	def __init__(self, input_dir: str, label2id_map: dict):
		self.path_files = glob.glob(f"{input_dir}/*.pkl")
		self.label2id_map = label2id_map
		self.id2label_map = {val: key for key, val in label2id_map.items()}

	def __len__(self):
		return len(self.path_files)

	@abstractmethod
	def __getitem__(self, index):
		pass


class InstanceDatasetEmb(InstanceDataset):
	def __getitem__(self, index):
		path_file = self.path_files[index]
		sample: InstanceSampleNli = pickle.load(open(path_file, "rb"))
		claim_embedding = sample.claim_embedding.unsqueeze(0)
		evidence_sentences_embeddings = sample.evidence_sentences_embeddings
		label = self.label2id_map[sample.gt_label]

		return claim_embedding, evidence_sentences_embeddings, label, path_file


class InstanceDatasetText(InstanceDataset):
	def __getitem__(self, index):
		path_file = self.path_files[index]
		sample: InstanceSampleNli = pickle.load(open(path_file, "rb"))
		label = self.label2id_map[sample.gt_label]
		claim_text = sample.claim_text
		evidence_text = [
			[sample.sentence_texts[s_idx]["text"] for s_idx in sentence_idx]
			for report_idx, sentence_idx in sample.reports_content.items()
		]  # higher sim reports to lower sim reports, ordered from evidence selection
		nli_tuples = sample.nli_tuples
		global_report_idx = sample.reports_content.keys()

		return claim_text, evidence_text, label, nli_tuples, global_report_idx, path_file


@dataclass
class DataBatch:
	file_id: list[str]
	x: dict[str, torch.Tensor]
	y: torch.Tensor

	def cast_input(self, x: dict, device: str):
		inputs = {}
		for tensor_key, tensor_val in x.items():
			if hasattr(tensor_val, "to"):
				inputs[tensor_key] = tensor_val.to(device)
			elif isinstance(tensor_val, list) and hasattr(tensor_val[0], "to"):
				inputs[tensor_key] = [t_v.to(device) for t_v in tensor_val]
			else:
				inputs[tensor_key] = tensor_val

		return inputs

	def to(self, device: str):

		return DataBatch(
			file_id=self.file_id,
			x=self.cast_input(self.x, device),
			y=self.y.to(device),
		)
