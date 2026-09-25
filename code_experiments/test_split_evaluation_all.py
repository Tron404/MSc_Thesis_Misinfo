import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml
from retrieval_embedding_module import InstanceSampleNli, InstanceSamplePredicted
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

torch.manual_seed(42)
DEVICE = "cuda:0"
BATCH_SIZE = 4


@torch.no_grad()
def run_evaluation(model, dataloader):
	pred_labels, true_labels = [], []
	sample_ids = []
	all_other_outputs = []
	for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
		batch_data = batch_data.to(DEVICE)
		sample_ids += batch_data.file_id
		labels = batch_data.y

		outputs = model(batch_data)
		if isinstance(outputs, tuple):
			logits, other = outputs
			all_other_outputs += [other]
		else:
			logits = outputs

		pred_labels += logits.argmax(dim=-1).tolist()
		true_labels += labels.tolist()
	pred_labels = np.asarray(pred_labels, dtype=int)
	true_labels = np.asarray(true_labels, dtype=int)

	p, r, f1, _ = precision_recall_fscore_support(
		y_true=true_labels,
		y_pred=pred_labels,
		# labels=labels,
		average="macro",
		zero_division=0,
	)
	p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
		y_true=true_labels,
		y_pred=pred_labels,
		# labels=labels,
		average=None,
		zero_division=0,
	)

	per_class_metrics = {
		f"class_{class_idx}": {
			"precision": p_c.item(),
			"recall": r_c.item(),
			"f1": f1_c.item(),
		}
		for class_idx, (p_c, r_c, f1_c) in enumerate(zip(p_per_class, r_per_class, f1_per_class))
	}
	metrics = {
		"test_macro_precision": p,
		"test_macro_recall": r,
		"test_macro_f1": f1,
		**per_class_metrics,
	}
	return pred_labels, true_labels, metrics, sample_ids


if __name__ == "__main__":
	data_root_dir = "nli_relations_r10-s5_deberta-large"
	model_root_dir = "BEST_MODEL_CKPT"

	for data_modelclass in os.listdir(model_root_dir):
		dataset_name, class_name = data_modelclass.split("_")

		data_dir = f"{data_root_dir}/{dataset_name}"
		classification_lab2id = json.load(open(f"{data_dir}/classification_lab2id_map.json", "r"))
		nli_id2lab = json.load(open(f"{data_root_dir}/id2lab_map.json", "r"))
		class_explanation_dict = json.load(open(f"{data_dir}/class_def.json", "r"))

		classification_id2lab = {idx_id: lab for lab, idx_id in classification_lab2id.items()}
		num_classes = len(classification_lab2id.keys())

		data_modelclass_dir = os.path.join(model_root_dir, data_modelclass)
		for exp_condition in os.listdir(data_modelclass_dir):
			output_root_dir = f"predicted_test_samples_CURRENT/{class_name}"
			if len(exp_condition.split("_")) > 1:
				output_root_dir = f"{output_root_dir}_{exp_condition}"
			output_data_root_dir = f"{output_root_dir}/{dataset_name}/test"

			if os.path.exists(output_data_root_dir) is True:
				continue

			exp_condition_dir = os.path.join(data_modelclass_dir, exp_condition)
			model_config_file_path = [
				os.path.join(exp_condition_dir, "configuration", file)
				for file in os.listdir(f"{exp_condition_dir}/configuration")
				if file != "train_paras.yaml"
				and file.endswith(".py") is False
				and os.path.isdir(f"{exp_condition_dir}/configuration/{file}")
				is False  # __pycache__ gets generated for some reason
			][0]
			model_config = yaml.safe_load(open(model_config_file_path, "r"))
			processing_parameters = model_config["processing_parameters"]

			model_parameters = model_config["model_parameters"]
			model_parameters["plm_dropout_prob"] = model_parameters.get("plm_dropout", 0.0)

			if "fcgalm" not in exp_condition.lower():
				model_parameters["finetuning_type"] = "lora" if "lora" in exp_condition else "none"
				model_parameters["graph_type"] = (
					"heterogeneous" if "hetero" in exp_condition else "homogeneous"
				)
			else:
				model_parameters["finetuning_type"] = "lora"
				model_parameters["graph_type"] = "heterogeneous"

			model_arch_class_path = [
				os.path.join(exp_condition_dir, "configuration", file)
				for file in os.listdir(f"{exp_condition_dir}/configuration")
				if file.endswith(".py") is True
				and os.path.isdir(f"{exp_condition_dir}/configuration/{file}") is False
				and "train" not in file
			][0]
			script_file = importlib.util.spec_from_file_location(class_name, model_arch_class_path)
			class_module = importlib.util.module_from_spec(script_file)
			sys.modules[class_name] = class_module
			script_file.loader.exec_module(class_module)
			print(exp_condition, class_module, exp_condition_dir, class_name)
			class_arch = getattr(class_module, class_name)

			ckpt_path = f"{exp_condition_dir}/best_checkpoint/state_dict.pth"

			model = class_arch(num_classes=num_classes, device=DEVICE, **model_parameters)
			model.load_state_dict(
				torch.load(ckpt_path, map_location={"cuda:1": "cuda:0"}), strict=True
			)
			model = model.to(DEVICE)
			model.eval()

			data_params = {"batch_size": BATCH_SIZE, **processing_parameters}
			test_dataloader = class_module.create_dataloader(
				data_path=f"{data_dir}/test", label2id_map=classification_lab2id, **data_params
			)

			print(output_root_dir)
			os.makedirs(output_data_root_dir, exist_ok=True)
			json.dump(
				classification_lab2id,
				open(f"{output_root_dir}/{dataset_name}/classification_lab2id_map.json", "w"),
				indent=2,
			)
			json.dump(
				nli_id2lab, open(f"{output_root_dir}/{dataset_name}/id2lab_map.json", "w"), indent=2
			)
			json.dump(
				class_explanation_dict,
				open(f"{output_root_dir}/{dataset_name}/class_def.json", "w"),
				indent=2,
			)

			pred_labels, true_labels, metrics, sample_ids = run_evaluation(model, test_dataloader)
			for pred_lab, true_lab, sample_path in zip(pred_labels, true_labels, sample_ids):
				sample_nli: InstanceSampleNli = pickle.load(open(sample_path, "rb"))
				output_path = sample_path.replace(data_root_dir, output_root_dir)

				sample_predicted = InstanceSamplePredicted(
					pred_label=classification_id2lab[pred_lab],
					**vars(sample_nli),
				)

				pickle.dump(sample_predicted, open(output_path, "wb"))
