import glob
import json
import os
import pickle
import time
from argparse import ArgumentParser

import requests
from env_utils import load_env_from_file
from retrieval_embedding_module import InstanceSample
from tqdm import tqdm

load_env_from_file(".")

MODEL_PROVIDER_DICT = {
	"openai/gpt-4o-mini": ["openai"],  # azure has stronger guardrails
	"google/gemini-2.5-flash": ["google-vertex/eu", "google-vertex/global"],
	"openai/gpt-5.6-luna": ["openai/flex"],
	"meta-llama/llama-3.3-70b-instruct": ["deepinfra/turbo", "novita/bf16"],
}

MODEL_HAS_REASONING = {
	"google/gemini-2.5-flash": True,
	"openai/gpt-5.6-luna": True,
	"openai/gpt-4o-mini": False,
	"meta-llama/llama-3.3-70b-instruct": False,
}


def init_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset", type=str, choices=["RAWFC", "LIAR-RAW"])
	arg_parser.add_argument("--model_url", type=str)

	args = arg_parser.parse_args()

	return vars(args)


def format_prompt(
	prompt_type, prompt_template, class_labels, class_explanations, sample: InstanceSample
):
	match prompt_type:
		case "claim_only":
			claim_text = sample.claim_text
			prompt = prompt_template.format(
				class_labels=class_labels,
				class_explanations=class_explanations,
				claim_text=claim_text,
			)
		case "claim_evidence":
			claim_text = sample.claim_text
			evidence_text = "\n".join([
				"- " + "".join([sample.sentence_texts[s_idx]["text"] for s_idx in sent_idx])
				for _, sent_idx in sample.reports_content.items()
			])
			prompt = prompt_template.format(
				class_labels=class_labels,
				class_explanations=class_explanations,
				claim_text=claim_text,
				evidence_text=evidence_text,
			)
	return prompt


def model_classification(model_url, data_dir_root, use_filtered_evidence=True, use_reasoning=False):
	data_dir = f"{data_dir_root}/test"
	classifcation_lab2id_map = json.load(
		open(f"{data_dir_root}/classification_lab2id_map.json", "r")
	)
	class_labels = ", ".join(list(classifcation_lab2id_map.keys()))
	class_explanations = json.load(open(f"{data_dir_root}/class_def.json", "r"))
	class_explanations = "\n".join([
		f"- {class_label}: {class_def}" for class_label, class_def in class_explanations.items()
	])

	providers = MODEL_PROVIDER_DICT[model_url]

	for prompt_type, prompt_template in all_prompt_templates.items():
		prompt_dir_id = f"{prompt_type}"
		other_id = "" if use_reasoning is False else "_reasoning"
		if prompt_type != "claim_only" and use_filtered_evidence is True:
			prompt_dir_id = f"{prompt_dir_id}_filtered_evidence"

		output_dir = f"llm_answers/{dataset}/{model_url.split('/')[-1]}{other_id}/{prompt_dir_id}/"
		os.makedirs(output_dir, exist_ok=True)

		iterator_desc = (
			f"{model_url}; {prompt_type}"
			if prompt_type == "claim_only"
			else f"{model_url}; {prompt_type}; filtered_evidence={use_filtered_evidence}"
		)
		for sample_path in tqdm(
			glob.glob(f"{data_dir}/*.pkl"),
			desc=iterator_desc,
		):
			sample: InstanceSample = pickle.load(open(sample_path, "rb"))
			output_path = sample_path.replace(data_dir, output_dir).replace(".pkl", ".json")
			if os.path.exists(output_path) is True:
				continue

			prompt = format_prompt(
				prompt_type, prompt_template, class_labels, class_explanations, sample
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
					url="https://openrouter.ai/api/v1/chat/completions",
					headers={
						"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
					},
					data=json.dumps({
						"model": model_url,
						"messages": messages,
						"provider": {"data_collection": "deny", "only": providers},
						"reasoning": {"enabled": use_reasoning},
						"temperature": 0.0,
					}),
				)
				response = response.json()
				response = {
					**response,
					"input_messages": messages,
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
		for template_path in glob.glob("llm_prompts/*.json")
	}

	all_data_dir_root = [f"nli_relations_r10-s5_deberta-large/{dataset}", f"embeddings/{dataset}"]

	use_reasoning = False

	# filtered evidence
	model_classification(
		model_url,
		f"nli_relations_r10-s5_deberta-large/{dataset}",
		use_filtered_evidence=True,
		use_reasoning=use_reasoning,
	)

	# unfiltered evidence
	model_classification(
		model_url, f"embeddings/{dataset}", use_filtered_evidence=False, use_reasoning=use_reasoning
	)
