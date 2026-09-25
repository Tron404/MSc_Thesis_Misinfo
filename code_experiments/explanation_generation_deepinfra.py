import glob
import json
import os
import pickle
import time
from argparse import ArgumentParser
from dataclasses import dataclass

import requests
from env_utils import load_env_from_file
from retrieval_embedding_module import InstanceSamplePredicted
from tqdm import tqdm

load_env_from_file(".")

model_url_map = {"meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}


def init_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset", type=str, choices=["RAWFC", "LIAR-RAW"])
	arg_parser.add_argument("--model_url", type=str)

	args = arg_parser.parse_args()

	return vars(args)


@dataclass
class GraphDescTemplate:
	nodes: dict[int, dict[str, str]]
	edges: dict[str, dict[str, str]]

	def make_str(self):
		return json.dumps(vars(self), indent=1)


def format_graph_desc(sample: InstanceSamplePredicted, nli_id2label: dict):
	def convert_to_likert_desc(conf_score):
		increment = (1 - 0.33) / 3

		idx2likert = {0: "weak", 1: "moderate", 2: "strong"}
		for idx in range(0, 3):
			left_th = 0.33 + idx * increment
			right_th = 0.33 + (idx + 1) * increment
			if conf_score > left_th and conf_score < right_th:
				return idx2likert[idx]
		return None

	nodes = {}
	edges = {}

	nodes[0] = {"text": sample.claim_text, "node_type": "claim"}
	for report_idx, sent_idx in sample.reports_content.items():
		nodes[report_idx + 1] = {
			"text": " ".join([sample.sentence_texts[s_idx]["text"] for s_idx in sent_idx]),
			"node_type": "evidence",
		}
	for premise_idx, hypo_idx, top_label, prob_dist in sample.nli_tuples:
		premise_idx += 1
		hypo_idx += 1
		top_label = nli_id2label[str(top_label)]
		likert_desc = convert_to_likert_desc(max(prob_dist))

		edges[f"({premise_idx}, {hypo_idx})"] = {"edge_type": top_label, "confidence": likert_desc}

	return GraphDescTemplate(nodes=nodes, edges=edges).make_str()


def format_prompt(
	prompt_type,
	prompt_template,
	pred_label,
	class_explanation,
	nli_id2label,
	sample: InstanceSamplePredicted,
):
	claim_text = sample.claim_text
	match prompt_type:
		case "claim_only":
			prompt = prompt_template.format(
				class_label=pred_label,
				class_explanation=class_explanation,
				claim_text=claim_text,
			)
		case "claim_evidence":
			evidence_text = "\n".join([
				"- " + "".join([sample.sentence_texts[s_idx]["text"] for s_idx in sent_idx])
				for _, sent_idx in sample.reports_content.items()
			])
			prompt = prompt_template.format(
				class_label=pred_label,
				class_explanation=class_explanation,
				claim_text=claim_text,
				evidence_text=evidence_text,
			)
		case "claim_graph":
			graph_desc = format_graph_desc(sample, nli_id2label)
			prompt = prompt_template.format(
				class_label=pred_label,
				class_explanation=class_explanation,
				claim_text=claim_text,
				graph_desc=graph_desc,
			)

	return prompt


def model_explanation(model_url, data_dir_root, use_reasoning=False):
	data_dir = f"{data_dir_root}/test"
	class_explanations = json.load(open(f"{data_dir_root}/class_def.json", "r"))
	nli_id2label = json.load(open(f"{data_dir_root}/id2lab_map.json", "r"))

	model_url_deepinfra = model_url_map[model_url]
	for prompt_type, prompt_template in all_prompt_templates.items():
		other_id = "" if use_reasoning is False else "_reasoning"
		output_dir = (
			f"llm_explanations/{dataset}/{model_url.split('/')[-1]}{other_id}/{prompt_type}/"
		)
		os.makedirs(output_dir, exist_ok=True)

		for sample_path in tqdm(
			glob.glob(f"{data_dir}/*.pkl"),
			desc=f"{model_url}; {prompt_type}",
		):
			sample: InstanceSamplePredicted = pickle.load(open(sample_path, "rb"))
			output_path = sample_path.replace(data_dir, output_dir).replace(".pkl", ".json")
			if os.path.exists(output_path) is True:
				continue
			pred_label = sample.pred_label
			class_explanation = class_explanations[pred_label]

			prompt = format_prompt(
				prompt_type, prompt_template, pred_label, class_explanation, nli_id2label, sample
			)

			messages = [
				{
					"role": "system",
					"content": "You are an expert fact-checker.",
				},
				{
					"role": "user",
					"content": prompt,
				},
			]

			try:
				response = requests.post(
					url="https://api.deepinfra.com/v1/openai/chat/completions",
					headers={
						"Authorization": f"Bearer {os.environ['DEEPINFRA_KEY']}",
					},
					data=json.dumps({
						"model": model_url_deepinfra,
						"messages": messages,
						"temperature": 0.0,
					}),
				)
				response = response.json()
				response = {
					**response,
					"input_messages": messages,
					"pred_label": pred_label,
					"gt_label": sample.gt_label,
					"gt_explanation": sample.gt_explanation,
				}
				json.dump(response, open(output_path, "w"), indent=2)
			except Exception as e:
				print(e)
				print(sample_path)
			time.sleep(0.5)


if __name__ == "__main__":
	args = init_args()
	dataset = args["dataset"]
	model_url = args["model_url"]

	all_prompt_templates = {
		template_path.split("/")[-1].removesuffix(".json"): json.load(open(template_path, "r"))[
			"content"
		]
		for template_path in glob.glob("prompt_explanations/*.json")
	}

	model_explanation(model_url, f"predicted_test_samples/FcGalm/{dataset}", use_reasoning=False)
