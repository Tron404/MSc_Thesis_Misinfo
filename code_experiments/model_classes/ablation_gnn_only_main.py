from collections import Counter, defaultdict
from functools import partial

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.modules import Dropout, Linear, ModuleList
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
	GATv2Conv,
	global_mean_pool,
	to_hetero,
)
from training_utils import DataBatch, InstanceDatasetText
from transformers import AutoModel, AutoTokenizer


class GraphNetwork(torch.nn.Module):
	def __init__(self, num_edge_features, graph_model_layout):
		super().__init__()

		self.gat1 = GATv2Conv(
			graph_model_layout["conv"][0],
			graph_model_layout["conv"][1],
			heads=graph_model_layout["conv"][2],
			edge_dim=num_edge_features,
			add_self_loops=False,
			concat=False,
			dropout=0.1,
			residual=False,
		)

		self.graph_classifier = ModuleList([self.gat1])

	def forward(self, x, edge_index, edge_attr):
		x = self.gat1(x, edge_index, edge_attr=edge_attr)

		return x


class GnnOnly(torch.nn.Module):
	def __init__(self, num_classes, device, **model_parameters):
		super().__init__()

		model_plm_id = model_parameters["plm_id"]
		self.device = device
		self.finetuning_type = model_parameters["finetuning_type"]
		self.use_cls_token = model_parameters["use_cls_token"]
		self.graph_type = model_parameters["graph_type"]
		self.plm_dropout_prob = model_parameters["plm_dropout_prob"]
		self.make_graph_func = (
			self.make_graph_data_hetero
			if self.graph_type == "heterogeneous"
			else self.make_graph_data_homo
		)

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
		num_edge_features = model_parameters["num_edge_features"]
		num_feat1 = 64

		self.data_metadata = (
			["claim", "evidence"],
			[
				("evidence", "relation_0", "claim"),
				("evidence", "relation_1", "claim"),
				("evidence", "relation_2", "claim"),
				("evidence", "relation_0", "evidence"),
				("evidence", "relation_1", "evidence"),
				("evidence", "relation_2", "evidence"),
			],
		)
		self.plm_dropout = Dropout(p=self.plm_dropout_prob)

		num_heads1 = 4
		graph_model_layout = {
			"conv": (num_emb_dim, num_feat1, num_heads1),
		}
		num_edge_features = None if self.graph_type == "homogeneous" else num_edge_features
		self.graph_model = GraphNetwork(
			num_edge_features=num_edge_features, graph_model_layout=graph_model_layout
		)

		#!! this is more parameter efficient than individual convs with HeteroConv
		if self.graph_type == "heterogeneous":
			self.graph_model = to_hetero(self.graph_model, self.data_metadata, aggr="sum")

		self.classification_head = Linear(num_feat1, num_classes)

	def make_graph_data_hetero(
		self,
		claim_embeddings: torch.Tensor,
		report_embeddings: torch.Tensor,
		nli_tuples: list,
		global_report_idx: list,
	) -> Batch:
		batched_graph_data = []

		for claim_emb, ev_report_emb, nli_tuples_local, global_report_idx_local in zip(
			claim_embeddings, report_embeddings, nli_tuples, global_report_idx
		):
			global_report_idx_2_local_idx = {
				g_report_idx: local_idx
				for local_idx, g_report_idx in enumerate(global_report_idx_local, start=0)
			}
			edge_idx_per_nli = defaultdict(list)
			edge_attr_per_nli = defaultdict(list)
			for nli_tuple in nli_tuples_local:
				premise_idx, hypothesis_idx, top_nli_label, all_probs = nli_tuple
				# idx must start at 0 => claim = 0, else > 0
				if premise_idx == -1:  # is the claim
					premise_idx = 0
					left_side = "claim"
					right_side = "evidence"
					hypothesis_idx = global_report_idx_2_local_idx[hypothesis_idx]
				elif hypothesis_idx == -1:
					left_side = "evidence"
					right_side = "claim"
					hypothesis_idx = 0
					premise_idx = global_report_idx_2_local_idx[premise_idx]
				else:
					left_side = "evidence"
					right_side = "evidence"
					hypothesis_idx = global_report_idx_2_local_idx[hypothesis_idx]
					premise_idx = global_report_idx_2_local_idx[premise_idx]

				edge_type = (left_side, f"relation_{top_nli_label}", right_side)

				edge_idx_per_nli[edge_type] += [(premise_idx, hypothesis_idx)]
				edge_attr_per_nli[edge_type] += [max(all_probs)]

			graph_item = HeteroData()
			graph_item["claim"].x = claim_emb.unsqueeze(dim=0)
			graph_item["evidence"].x = ev_report_emb

			for edge_type in edge_idx_per_nli.keys():
				graph_item[*edge_type].edge_index = torch.as_tensor(
					edge_idx_per_nli[edge_type], dtype=torch.long
				).T
				graph_item[*edge_type].edge_attr = torch.as_tensor(edge_attr_per_nli[edge_type])

			# create dummy values to be used by the PyG collator for batching
			all_expected_edge_types = self.data_metadata[1]
			for expected_edge_type in all_expected_edge_types:
				if expected_edge_type not in edge_attr_per_nli.keys():
					graph_item[*expected_edge_type].edge_index = torch.empty(
						(2, 0), dtype=torch.long
					)
					graph_item[*expected_edge_type].edge_attr = torch.empty(
						(0), dtype=torch.float32
					)

			batched_graph_data += [graph_item]

		return Batch.from_data_list(batched_graph_data).to(self.device)

	def make_graph_data_homo(
		self,
		claim_embeddings: torch.Tensor,
		report_embeddings: torch.Tensor,
		nli_tuples: list,
		global_report_idx: list,
	) -> Batch:
		batched_graph_data = []

		for claim_emb, ev_report_emb, nli_tuples_local, global_report_idx_local in zip(
			claim_embeddings, report_embeddings, nli_tuples, global_report_idx
		):
			edge_idx = []

			global_report_idx_2_local_idx = {
				g_report_idx: local_idx
				for local_idx, g_report_idx in enumerate(global_report_idx_local, start=1)
			}

			for nli_tuple in nli_tuples_local:
				premise_idx, hypothesis_idx, top_nli_label, all_probs = nli_tuple
				# idx must start at 0 => claim = 0, else > 0
				if premise_idx == -1:  # is the claim
					premise_idx = 0
					hypothesis_idx = global_report_idx_2_local_idx[hypothesis_idx]
				elif hypothesis_idx == -1:
					hypothesis_idx = 0
					premise_idx = global_report_idx_2_local_idx[premise_idx]
				else:
					hypothesis_idx = global_report_idx_2_local_idx[hypothesis_idx]
					premise_idx = global_report_idx_2_local_idx[premise_idx]

				edge_idx += [(premise_idx, hypothesis_idx)]

			node_features = torch.concat(
				[
					claim_emb.unsqueeze(0),  # [1,D]
					ev_report_emb,  # [TOPK_R,D]
				],
				dim=0,
			).to(torch.float)

			edge_idx = torch.as_tensor(edge_idx, dtype=torch.long).T

			graph_item = Data(
				x=node_features,
				edge_index=edge_idx,
			)

			batched_graph_data += [graph_item]

		return Batch.from_data_list(batched_graph_data).to(self.device)

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

				evidence_reports_hidden_states += [item_evidence_hidden_states]  # [B,TOPK_R,768]

			nli_tuples = batch.x["nli_tuples"]
			global_report_idx = batch.x["global_report_idx"]

		batched_graph_data = self.make_graph_func(
			claim_hidden_states, evidence_reports_hidden_states, nli_tuples, global_report_idx
		)

		return claim_hidden_states, evidence_reports_hidden_states, batched_graph_data

	def forward_graph_emb(self, batched_graph_data):
		if self.graph_type == "heterogeneous":
			x_dict = batched_graph_data.x_dict
			edge_index_dict = batched_graph_data.edge_index_dict
			edge_attr_dict = batched_graph_data.edge_attr_dict

			graph_emb_dict = self.graph_model(x_dict, edge_index_dict, edge_attr_dict)

			graph_emb_claim = graph_emb_dict["claim"]

			graph_emb_evidence = global_mean_pool(
				graph_emb_dict["evidence"], batched_graph_data["evidence"].batch
			)

			graph_emb = torch.add(graph_emb_claim, graph_emb_evidence)
		else:
			x = batched_graph_data.x
			edge_index = batched_graph_data.edge_index
			batch = batched_graph_data.batch

			graph_emb = self.graph_model(x, edge_index, edge_attr=None)

			graph_emb = global_mean_pool(graph_emb, batch)

		return graph_emb

	def forward(self, batch: DataBatch):
		### get embeddings and graph structure
		claim_hidden_states, evidence_reports_hidden_states, batched_graph_data = (
			self._get_embeddings(batch)
		)

		### compute graph_emb
		graph_emb = self.forward_graph_emb(batched_graph_data)
		logits = self.classification_head(graph_emb)

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
