import glob
import json
import os
import pickle
from argparse import ArgumentParser
from dataclasses import asdict
from itertools import permutations
from typing import List, Tuple

import numpy as np
import torch
from nli_class_module import NliModel
from retrieval_embedding_module import InstanceSample, InstanceSampleNli
from tqdm import tqdm


def read_all_instance_emb(dir_path: str) -> Tuple[List["InstanceSample"], List[str]]:
	files = glob.glob(f"{dir_path}/*.pkl")
	files = sorted(files, key=lambda x: x.split("/")[-1].removesuffix(".pkl"))
	instances = [
		pickle.load(open(file_path, "rb")) for file_path in tqdm(files, desc="Loading data...")
	]
	return instances, files


def init_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--topk_reports", type=int)
	arg_parser.add_argument("--topk_sents", type=int)

	args = arg_parser.parse_args()
	return vars(args)


if __name__ == "__main__":
	model_name = "deberta-large"
	args = init_args()
	topk_reports = args["topk_reports"]
	topk_sentences = args["topk_sents"]

	device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
	model_params = {"batch_size": 2}
	model = NliModel.create_model(
		name="finetuned_nli", model_name=model_name, device=device, **model_params
	)

	id2lab = model.model.config.id2label

	datasets = ["RAWFC", "LIAR-RAW"][:]
	splits = ["train", "val", "test"][:]
	output_dir = f"nli_relations_r{topk_reports}-s{topk_sentences}_{model_name}"
	input_dir = f"selected_evidence_r{topk_reports}-s{topk_sentences}"
	os.makedirs(output_dir, exist_ok=True)
	json.dump(id2lab, open(f"{output_dir}/id2lab_map.json", "w"))

	for dataset in datasets:
		for split in splits:
			dir_path = f"{input_dir}/{dataset}/{split}"
			output_path = f"{output_dir}/{dataset}/{split}"
			os.makedirs(output_path, exist_ok=True)

			instances, file_paths = read_all_instance_emb(dir_path)
			labels = sorted(set([sample.gt_label for sample in instances]))
			label2idx = {label: idx for idx, label in enumerate(labels)}
			json.dump(
				label2idx, open(f"{output_dir}/{dataset}/classification_lab2id_map.json", "w")
			)

			for sample, path in tqdm(
				zip(instances, file_paths), desc="Creating NLI relations...", total=len(file_paths)
			):
				output_path = path.replace(input_dir, output_dir)
				if os.path.exists(output_path):
					continue

				claim_id = -1
				evidence_ids = list(sample.reports_content.keys())

				evidence_claim_pairs_ids = [(ev_id, claim_id) for ev_id in evidence_ids]
				evidence_claim_pairs_text = [
					(
						" ".join([
							sample.sentence_texts[sent_id]["text"]
							for sent_id in sample.reports_content[ev_id]
						]),
						sample.claim_text,
					)
					for ev_id, _ in evidence_claim_pairs_ids
				]

				evidence_evidence_pairs_ids = list(permutations(evidence_ids, 2))
				evidence_evidence_pairs_text = [
					(
						" ".join([
							sample.sentence_texts[sent_id]["text"]
							for sent_id in sample.reports_content[ev_id_premise]
						]),
						" ".join([
							sample.sentence_texts[sent_id]["text"]
							for sent_id in sample.reports_content[ev_id_hypothesis]
						]),
					)
					for ev_id_premise, ev_id_hypothesis in evidence_evidence_pairs_ids
				]

				all_pairs_text = evidence_claim_pairs_text + evidence_evidence_pairs_text
				all_pairs_text = np.asarray(all_pairs_text)

				premises = all_pairs_text[:, 0].tolist()
				hypotheses = all_pairs_text[:, 1].tolist()
				pred_labels, pred_probs, pred_all_probs = model.get_nli_label(premises, hypotheses)

				nli_tuples = []
				for id_pair, label, prob, all_prob in zip(
					evidence_claim_pairs_ids + evidence_evidence_pairs_ids,
					pred_labels,
					pred_probs,
					pred_all_probs,
				):
					nli_tuples += [(*id_pair, label, all_prob)]

				nli_instance = InstanceSampleNli(**asdict(sample), nli_tuples=nli_tuples)
				pickle.dump(nli_instance, open(output_path, "wb"))
