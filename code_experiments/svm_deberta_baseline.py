import json
import os
import pickle
from argparse import ArgumentParser
from collections import Counter
from functools import partial

import numpy as np
import torch
from retrieval_embedding_module import InstanceSampleNli, InstanceSamplePredicted
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_utils import DataBatch, InstanceDatasetText
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(42)
DEVICE = "cuda:0"


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


def plm_pooling(x: torch.Tensor, attention_mask: torch.Tensor):
	attention_mask_exp = attention_mask.unsqueeze(-1).expand(x.size())
	sum_x = torch.sum(x * attention_mask_exp, dim=1)
	sum_mask = torch.clamp(attention_mask_exp.sum(1), min=1e-9)
	pooled_ex = sum_x / sum_mask

	return pooled_ex


@torch.no_grad()
def run_embedding(model, dataloader):
	true_labels = []
	sample_ids = []
	claim_embeddings = []
	evidence_embeddings = []
	true_labels = []
	for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
		batch_data = batch_data.to(DEVICE)
		sample_ids += batch_data.file_id
		true_labels += batch_data.y.cpu().numpy().tolist()

		c_emb = model(**batch_data.x["claim_tokens"]).last_hidden_state
		c_emb = plm_pooling(c_emb, batch_data.x["claim_tokens"]["attention_mask"])
		claim_embeddings += [c_emb]

		batch_ev_emb = []
		for ev_tokens in batch_data.x["evidence_tokens"]:
			ev_emb = model(**ev_tokens).last_hidden_state
			ev_emb = plm_pooling(ev_emb, ev_tokens["attention_mask"])
			batch_ev_emb += [torch.mean(ev_emb, dim=0)]

		evidence_embeddings += [torch.stack(batch_ev_emb, dim=0)]

	true_labels = np.asarray(true_labels, dtype=int)

	claim_embeddings = torch.cat(claim_embeddings, dim=0)
	evidence_embeddings = torch.cat(evidence_embeddings, dim=0)

	embeddings = torch.cat([claim_embeddings, evidence_embeddings], dim=1).cpu().numpy()

	return embeddings, true_labels, sample_ids


def init_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset", type=str, choices=["RAWFC", "LIAR-RAW"])

	args = arg_parser.parse_args()
	return vars(args)


if __name__ == "__main__":
	data_params = {
		"shuffle": True,
		"batch_size": 4,
		"subset_train_data": None,
		"plm_id": "microsoft/deberta-v3-base",
		"max_len": 512,
	}
	args = init_args()

	dataset = args["dataset"]
	data_dir = "nli_relations_r10-s5_deberta-large/"
	model_plm_id = data_params["plm_id"]

	classification_lab2id = json.load(
		open(f"{data_dir}/{dataset}/classification_lab2id_map.json", "r")
	)
	classification_id2lab = {idx: lab for lab, idx in classification_lab2id.items()}
	train_dataloader = create_dataloader(
		f"{data_dir}/{dataset}/train", label2id_map=classification_lab2id, **data_params
	)
	test_dataloader = create_dataloader(
		f"{data_dir}/{dataset}/test", label2id_map=classification_lab2id, **data_params
	)

	plm_model = AutoModel.from_pretrained(model_plm_id, device_map=DEVICE)
	plm_model.eval()
	num_emb_dim = plm_model.config.hidden_size

	train_embeddings, train_labels, train_sample_ids = run_embedding(plm_model, train_dataloader)
	pickle.dump(
		{"emb": train_embeddings, "labels": train_labels},
		open(f"train_embeddings_{dataset}.pkl", "wb"),
	)
	test_embeddings, test_labels, test_sample_ids = run_embedding(plm_model, test_dataloader)
	pickle.dump(
		{"emb": test_embeddings, "labels": test_labels},
		open(f"test_embeddings_{dataset}.pkl", "wb"),
	)

	svc = SVC(random_state=42)
	svc.fit(train_embeddings, train_labels)

	pred_labels = svc.predict(test_embeddings)

	output_dir_root = f"predicted_test_samples/svm/{dataset}/test"
	os.makedirs(output_dir_root, exist_ok=True)
	for true_lab, pred_lab, file_id in zip(test_labels, pred_labels, test_sample_ids):
		sample: InstanceSampleNli = pickle.load(open(file_id, "rb"))
		output_path = os.path.join(output_dir_root, file_id.split("/")[-1])
		assert classification_lab2id[sample.gt_label] == true_lab, (
			classification_lab2id[sample.gt_label],
			true_lab,
		)
		pickle.dump(
			InstanceSamplePredicted(pred_label=classification_id2lab[pred_lab], **vars(sample)),
			open(output_path, "wb"),
		)
