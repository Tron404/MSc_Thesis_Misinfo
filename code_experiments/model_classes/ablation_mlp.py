from collections import Counter
from functools import partial

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.modules import Dropout, Linear
from torch.utils.data import DataLoader
from training_utils import DataBatch, InstanceDatasetText
from transformers import AutoModel, AutoTokenizer


class Mlp(torch.nn.Module):
	def __init__(self, num_classes, device, **model_parameters):
		super().__init__()

		model_plm_id = model_parameters["plm_id"]
		self.device = device
		self.finetuning_type = model_parameters["finetuning_type"]
		self.use_cls_token = model_parameters["use_cls_token"]
		self.plm_dropout_prob = model_parameters["plm_dropout_prob"]

		lora_rank = 16
		lora_alpha = 32  # usually 2x the rank
		lora_dropout = 0.1
		target_modules = model_parameters["target_modules"]

		peft_config = LoraConfig(
			task_type=TaskType.FEATURE_EXTRACTION,
			r=lora_rank,
			lora_alpha=lora_alpha,
			lora_dropout=lora_dropout,
			target_modules=target_modules,
			inference_mode=False,
		)

		self.plm_model = AutoModel.from_pretrained(model_plm_id, device_map=self.device)
		num_emb_dim = self.plm_model.config.hidden_size
		self.plm_dropout = Dropout(p=self.plm_dropout_prob)

		match self.finetuning_type:
			case "none":
				for plm_param in self.plm_model.parameters():
					plm_param.requires_grad = False
			case "full":
				for plm_param in self.plm_model.parameters():
					plm_param.requires_grad = True
			case "lora":
				self.plm_model = get_peft_model(self.plm_model, peft_config)

		# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
		print("==============================")
		trainable_params = 0
		all_params = 0
		for param in self.parameters():
			all_params += param.numel()
			if param.requires_grad:
				trainable_params += param.numel()
		print(
			f"PLM Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
		)
		print(f"PLM All parameters: {all_params:,}")

		self._create_network(num_classes=num_classes, num_emb_dim=num_emb_dim, **model_parameters)

		trainable_params = 0
		all_params = 0
		for param in self.parameters():
			all_params += param.numel()
			if param.requires_grad:
				trainable_params += param.numel()
		print(
			f"Full trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
		)
		print(f"Full parameters: {all_params:,}")

		print("==============================")

	def _create_network(self, num_classes, num_emb_dim, **model_parameters):
		self.classification_head = Linear(num_emb_dim * 2, num_classes)

	def plm_pooling(self, x: torch.Tensor, attention_mask: torch.Tensor):
		if self.use_cls_token is False:
			attention_mask_exp = attention_mask.unsqueeze(-1).expand(x.size())
			sum_x = torch.sum(x * attention_mask_exp, dim=1)
			sum_mask = torch.clamp(attention_mask_exp.sum(1), min=1e-9)
			pooled_ex = sum_x / sum_mask
		else:
			pooled_ex = x[:, 0].to(torch.float32)

		return pooled_ex

	def _get_embeddings(self, batch: DataBatch):
		# claim tokens, !PER REPORT! tokens

		grad_context = (
			torch.no_grad
			if self.finetuning_type == "none" or self.training is False
			else torch.enable_grad
		)  # while there is an model.eval() in the run_epoch for validation, better to be sure the correct context is applied

		with grad_context():
			### get embedding of claim text
			claim_tokens = batch.x["claim_tokens"]  # [B,TOKENS]
			claim_attention_masks = claim_tokens["attention_mask"]

			claim_hidden_states = self.plm_model(**claim_tokens).last_hidden_state
			claim_hidden_states = self.plm_pooling(claim_hidden_states, claim_attention_masks)
			claim_hidden_states = self.plm_dropout(claim_hidden_states)

			evidence_reports_tokens = batch.x["evidence_tokens"]  # [B,TOPK_R,TOKENS]
			evidence_reports_attention_masks = [
				ev_report["attention_mask"] for ev_report in evidence_reports_tokens
			]

			### get embedding of evidence reports
			evidence_reports_hidden_states = []
			for item_evidence_tokens, item_evidence_att_mask in zip(
				evidence_reports_tokens, evidence_reports_attention_masks
			):
				item_evidence_hidden_states = self.plm_model(
					**item_evidence_tokens
				).last_hidden_state
				item_evidence_hidden_states = self.plm_pooling(
					item_evidence_hidden_states, item_evidence_att_mask
				)
				item_evidence_hidden_states = self.plm_dropout(item_evidence_hidden_states)

				evidence_reports_hidden_states += [item_evidence_hidden_states]

		return claim_hidden_states, evidence_reports_hidden_states

	def forward(self, batch: DataBatch):
		### get embeddings and graph structure
		claim_hidden_states, evidence_reports_hidden_states = self._get_embeddings(batch)

		### combine all embeddings
		batch_report_ev = []
		for item_ev_emb in evidence_reports_hidden_states:
			batch_report_ev += [torch.mean(item_ev_emb, dim=0)]
		batch_report_ev = torch.stack(batch_report_ev, dim=0)

		final_emb = torch.cat([claim_hidden_states, batch_report_ev], dim=-1)

		logits = self.classification_head(final_emb)

		return logits


def create_subset(dataset: InstanceDatasetText, num_samples: int) -> InstanceDatasetText:
	new_subset = []  # idx
	class_distribution = Counter()
	per_class_num = int(num_samples / len(dataset.id2label_map.keys()))

	for idx, (_, _, label, _, _, _) in enumerate(dataset):
		if class_distribution[label] < per_class_num:
			class_distribution[label] += 1
			new_subset += [idx]

		if class_distribution.total() >= num_samples:
			break

	dataset.path_files = [dataset.path_files[subset_idx] for subset_idx in new_subset]
	return dataset


def create_dataloader(data_path: str, label2id_map: dict, **params):
	def custom_collate_fn(batch: list[InstanceDatasetText], tokenizer: AutoTokenizer):
		batch_claim = []  # [B]
		batch_evidence = []  # [B,TOPK_R]
		batch_labels = []  # [B]
		batch_file_paths = []  # [B]
		batch_nli_tuples = []  # [B,4]
		batch_global_report_idx = []  # !@TODO: JUST RE-INDEX THESE WHEN SELECTING EVIDENCE
		for item in batch[:]:
			claim, evidence_reports, label, nli_tuples, global_report_idx, path_file = item
			batch_labels += [label]
			batch_claim += [claim]
			batch_file_paths += [path_file]
			batch_nli_tuples += [nli_tuples]
			batch_global_report_idx += [
				global_report_idx
			]  # !@TODO: JUST RE-INDEX THESE WHEN SELECTING EVIDENCE

			batch_evidence += [[" ".join(report_sents) for report_sents in evidence_reports]]

		batch_labels = torch.as_tensor(batch_labels)
		tokenized_claim_input = tokenizer(
			batch_claim,
			padding="longest",
			return_tensors="pt",
			max_length=params["max_len"],
			truncation=True,
		)
		tokenized_evidence_reports_input = [
			tokenizer(
				evidence_reports_local,
				padding="longest",
				return_tensors="pt",
				max_length=params["max_len"],
				truncation=True,
			)
			for evidence_reports_local in batch_evidence
		]

		return DataBatch(
			file_id=batch_file_paths,
			y=batch_labels,
			x={
				"claim_tokens": tokenized_claim_input,
				"evidence_tokens": tokenized_evidence_reports_input,
				"nli_tuples": batch_nli_tuples,
				"global_report_idx": batch_global_report_idx,  # !@TODO: JUST RE-INDEX THESE WHEN SELECTING EVIDENCE
			},
		)

	batch_size = params.get("batch_size")
	shuffle = params.get("shuffle")
	model_id = params.get("plm_id")
	subset_train_data = params.get("subset_train_data")

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	if subset_train_data is not None:
		dataset = InstanceDatasetText(data_path, label2id_map)
		dataset = create_subset(dataset, num_samples=subset_train_data)
	else:
		dataset = InstanceDatasetText(data_path, label2id_map)

	custom_collate_partial = partial(custom_collate_fn, tokenizer=tokenizer)

	dataloader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_partial
	)

	return dataloader
