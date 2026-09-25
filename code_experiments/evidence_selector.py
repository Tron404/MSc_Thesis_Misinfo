import glob
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from retrieval_embedding_module import InstanceSample
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_all_instance_emb(dir_path: str) -> Tuple[List["InstanceSample"], List[str]]:
	files = glob.glob(f"{dir_path}/*.pkl")
	files = sorted(files, key=lambda x: x.split("/")[-1].removesuffix(".pkl"))
	instances = [
		pickle.load(open(file_path, "rb")) for file_path in tqdm(files, desc="Loading data...")
	]
	return instances, files


@torch.no_grad()
def cosine_sim(query_emb: torch.Tensor, candidate_emb: torch.Tensor):
	def check_norm(x: torch.Tensor):
		dim = 0
		num_x = 1
		if len(x.shape) > 1:
			dim = 1
			num_x = x.shape[0]
		x_norm = x.norm(dim=dim)
		check_ones = torch.ones(num_x, dtype=torch.float32, device=DEVICE)
		if not torch.all(torch.isclose(x_norm, check_ones)):
			x_norm = x_norm.unsqueeze(1).repeat(1, x.shape[dim])
			x = torch.div(x, x_norm)
		return x

	query_emb = check_norm(query_emb)
	candidate_emb = check_norm(candidate_emb)

	query_emb = query_emb.unsqueeze(0)

	sim_scores = torch.matmul(query_emb, candidate_emb.T)
	return sim_scores.squeeze(0).cpu().numpy()


def evidence_selection(
	sim_scores: np.ndarray,
	instance: InstanceSample,
	topk_reports: int = 10,
	topk_sentences: int = 10,
):
	sent_evidence_idx = np.argsort(sim_scores)[::-1].tolist()

	# organize sent idx into their corresponding reports
	evidence_reports = defaultdict(list)
	for sent_idx in sent_evidence_idx:
		cand_report_idx = instance.sentence_texts[sent_idx]["report_idx"]
		evidence_reports[cand_report_idx] += [sent_idx]

	# take topk sentences by similarity
	report_avg_sim = {}
	chosen_idx = []

	for report_idx, sent_idx in evidence_reports.items():
		report_sim_scores = sim_scores[sent_idx]
		ordered_local_sent_idx = np.argsort(report_sim_scores)[::-1][:topk_sentences]
		sent_idx = [sent_idx[ordered_idx] for ordered_idx in ordered_local_sent_idx]
		evidence_reports[report_idx] = sent_idx
		report_avg_sim[report_idx] = np.mean(sim_scores[sent_idx])
		chosen_idx += sent_idx

	# take topk reports by avg sent similarity

	report_avg_sim = sorted(report_avg_sim.items(), key=lambda x: x[1], reverse=True)[:topk_reports]
	evidence_reports = {
		report_idx: evidence_reports[report_idx] for report_idx, report_sim in report_avg_sim
	}

	subset_instance = {
		"claim_text": instance.claim_text,
		"claim_embedding": instance.claim_embedding.cpu(),
		"reports_content": evidence_reports,
		"sentence_texts": {
			sent_idx: {**instance.sentence_texts[sent_idx], "cos_sim": sim_scores[sent_idx]}
			for sent_idx in chosen_idx
		},
		"evidence_sentences_embeddings": instance.evidence_sentences_embeddings.cpu(),
		"gt_explanation": instance.gt_explanation,
		"gt_label": instance.gt_label,
	}

	subset_instance = InstanceSample(**subset_instance)
	return subset_instance


def init_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--topk_reports", type=int)
	arg_parser.add_argument("--topk_sents", type=int)

	args = arg_parser.parse_args()
	return vars(args)


if __name__ == "__main__":
	datasets = ["RAWFC", "LIAR-RAW"][:]
	splits = ["train", "val", "test"]

	args = init_args()
	topk_reports = args["topk_reports"]
	topk_sentences = args["topk_sents"]
	output_root_dir = f"selected_evidence_r{topk_reports}-s{topk_sentences}"

	for dataset in datasets:
		for split in splits:
			dir_path = f"embeddings/{dataset}/{split}"
			output_dir = f"{output_root_dir}/{dataset}/{split}"
			os.makedirs(output_dir, exist_ok=True)

			instances, file_paths = read_all_instance_emb(dir_path)
			for sample, path in tqdm(
				zip(instances, file_paths), desc="Selecting evidence...", total=len(file_paths)
			):
				output_path = path.replace("embeddings", output_root_dir)

				if not isinstance(sample.claim_embedding, torch.Tensor):
					sample.claim_embedding = torch.from_numpy(sample.claim_embedding)
					sample.evidence_sentences_embeddings = torch.from_numpy(
						sample.evidence_sentences_embeddings
					)

				claim_emb = sample.claim_embedding.to(DEVICE)
				evidence_emb = sample.evidence_sentences_embeddings.to(DEVICE)

				sim_scores = cosine_sim(claim_emb, evidence_emb)
				subset_sample = evidence_selection(
					sim_scores, sample, topk_reports=topk_reports, topk_sentences=topk_sentences
				)
				pickle.dump(subset_sample, open(output_path, "wb"))
