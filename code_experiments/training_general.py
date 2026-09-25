import inspect
import io
import json
import logging
import os
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import yaml
from model_classes import import_model_class
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader as PtDataloader
from torch_geometric.loader import DataLoader as PygDataloader
from tqdm import tqdm

IS_DEBUG = False
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class EarlyStopper:
	def __init__(self, patience=1, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_validation_f1 = float("-inf")

	def early_stop(self, validation_f1):
		if validation_f1 > self.min_validation_f1:
			self.min_validation_f1 = validation_f1
			self.counter = 0
		elif validation_f1 < (self.min_validation_f1 + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


def get_cm_heatmap(y_pred, y_true, id2label):
	cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

	plt.figure(figsize=(16, 9))
	heatmap = sns.heatmap(cm, annot=True, linewidths=0.5, fmt="d")
	plt.ylabel("True Label")
	plt.xlabel("Predicted Label")
	plt.xticks(ticks=[k + 0.5 for k in id2label.values()], labels=list(id2label.keys()))
	plt.yticks(ticks=[k + 0.5 for k in id2label.values()], labels=list(id2label.keys()))
	plt.tight_layout()

	return heatmap


def parse_args():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset", type=str, choices=["RAWFC", "LIAR-RAW"])
	arg_parser.add_argument("--path_data", type=str)
	arg_parser.add_argument("--seed", type=int, default=42)
	arg_parser.add_argument("--model_class", type=str)
	arg_parser.add_argument("--device", type=str)
	arg_parser.add_argument("--model_config_path", type=str, help="Path to YAML file")

	#### overriding YAML model config settings
	arg_parser.add_argument(
		"--graph_type", type=str, nargs="?", choices=["homogeneous", "heterogeneous"]
	)
	arg_parser.add_argument("--finetuning_type", type=str, nargs="?", choices=["none", "lora"])
	arg_parser.add_argument("--plm_dropout_prob", type=float, default=0.0)

	args = arg_parser.parse_args()
	args = vars(args)

	model_paras_args = ["graph_type", "finetuning_type", "plm_dropout_prob"]
	model_parameters = {}
	for mp in model_paras_args:
		if mp in args.keys():
			mp_val = args.pop(mp)
			if mp_val is not None:
				model_parameters[mp] = mp_val
	args["model_parameters"] = model_parameters

	return args


def run_epoch(
	dataset_dataloader: PygDataloader | PtDataloader,
	model,
	optimizer,
	loss_func,
	epoch_idx,
	classification_id2lab,
	device,
	is_train=False,
):
	running_loss = 0
	last_loss = 0
	num_batches = len(dataset_dataloader)

	if is_train is True:
		model.train()
		grad_tracking = torch.enable_grad
	else:
		model.eval()
		grad_tracking = torch.no_grad

	pred_labels, true_labels = [], []
	sample_ids = []
	all_other_outputs = []

	with grad_tracking():
		for idx, batch_data in enumerate(dataset_dataloader):
			batch_data = batch_data.to(device)
			sample_ids += batch_data.file_id
			labels = batch_data.y

			if is_train is True:
				optimizer.zero_grad(set_to_none=True)

			outputs = model(batch_data)
			if isinstance(outputs, tuple):
				logits, other = outputs
				all_other_outputs += [other]
			else:
				logits = outputs
			loss = loss_func(logits, labels)

			if is_train is True:
				loss.backward()
				optimizer.step()

			pred_labels += logits.argmax(dim=-1).tolist()
			true_labels += labels.tolist()

			running_loss += loss.item()

	last_loss = running_loss / num_batches

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

	eval_type = "train" if is_train is True else "validation"
	metrics = {
		f"{eval_type}_macro_precision": p,
		f"{eval_type}_macro_recall": r,
		f"{eval_type}_macro_f1": f1,
	}
	if IS_DEBUG is False:
		heatmap = get_cm_heatmap(
			y_true=true_labels, y_pred=pred_labels, id2label=classification_id2lab
		)
		with io.BytesIO() as buffer:
			heatmap.figure.savefig(buffer, bbox_inches="tight")
			buffer.seek(0)
			image_figure = Image.open(buffer)
			image_figure.load()
			mlflow.log_image(image_figure, key=f"confusion_matrix_{eval_type}", step=epoch_idx)

		plt.close()

		sample_ids = np.asarray(sample_ids)
		misses_idx = np.where(pred_labels != true_labels)
		hits_idx = np.where(pred_labels == true_labels)
		exp_name = mlflow.get_experiment(mlflow.active_run().info.experiment_id).name
		run_name = mlflow.active_run().info.run_name
		with open(f"eval_fails/label_diagnostics_{exp_name}---{run_name}.jsonl", "a") as file:
			label_diagnostics = {
				"epoch": epoch_idx,
				"eval_type": eval_type,
				"misses": {
					"pred": pred_labels[misses_idx].tolist(),
					"true": true_labels[misses_idx].tolist(),
					"file_ids": sample_ids[misses_idx].tolist(),
				},
				"hits": {
					"pred": pred_labels[hits_idx].tolist(),
					"true": true_labels[hits_idx].tolist(),
					"file_ids": sample_ids[hits_idx].tolist(),
				},
				"other": all_other_outputs,
			}
			file.write(json.dumps(label_diagnostics) + "\n")
			file.flush()
	else:
		print(f"Epoch: {epoch_idx} --- {metrics}")

	return last_loss, metrics


def warmup_lr_func(current_epoch, warmup_epochs):
	if current_epoch < warmup_epochs:
		aux_lr = current_epoch / warmup_epochs
		return aux_lr


# ! MODEL OVERFIT VERY EASILY ON THE EMBEDDINGS -> USE SIMPLER/SHALLOWER MODELS
def train_torch_model(
	data_dir: str,
	model_class_name: str,
	train_config_path: dict,
	model_config_path: dict,
	device: str,
	model_override_config: dict,
):
	model_class, create_dataloader = import_model_class(model_class_name)

	model_parameters_config = yaml.safe_load(open(model_config_path, "r"))
	train_parameters = yaml.safe_load(open(train_config_path, "r"))
	subset_train_data = train_parameters["subset_train_data"]

	if IS_DEBUG is False:
		mlflow.log_artifact(model_config_path, "configuration")
		mlflow.log_artifact(train_config_path, "configuration")
		mlflow.log_artifact(__file__, "configuration")
		mlflow.log_artifact(inspect.getfile(model_class), "configuration")
		os.makedirs("eval_fails", exist_ok=True)

		# add some tags to filter in mlflow
		mlflow.set_tag("use_data_subset", subset_train_data is not None)
		mlflow.set_tag("data_dir", data_dir)

	lr = train_parameters.get("lr")
	num_epochs = train_parameters.get("num_epochs")
	batch_size = train_parameters.get("batch_size")

	processing_parameters = {}
	model_parameters = {}
	if model_parameters_config is not None:
		processing_parameters = (
			model_parameters_config["processing_parameters"]
			if "processing_parameters" in model_parameters_config.keys()
			else {}
		)
		model_parameters = (
			model_parameters_config["model_parameters"]
			if "model_parameters" in model_parameters_config.keys()
			else {}
		)
		for mp_key, mp_val in model_override_config.items():
			model_parameters[mp_key] = mp_val

	data_params = {
		"shuffle": True,
		"batch_size": batch_size,
		"subset_train_data": subset_train_data,
		**processing_parameters,
	}

	classification_lab2id = json.load(open(f"{data_dir}/classification_lab2id_map.json", "r"))
	train_dataloader = create_dataloader(
		data_path=f"{data_dir}/train", label2id_map=classification_lab2id, **data_params
	)
	val_dataloader = create_dataloader(
		data_path=f"{data_dir}/val", label2id_map=classification_lab2id, **data_params
	)

	num_classes = len(classification_lab2id.keys())
	model = model_class(num_classes=num_classes, device=device, **model_parameters).to(device)

	weight_decay = train_parameters.get("adam_w_decay")
	loss_func = CrossEntropyLoss(weight=None)
	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	use_lr_scheduler = train_parameters.get("use_lr_scheduler")
	if use_lr_scheduler is True:
		target_lr = train_parameters.get("target_lr")
		target_epoch = int(num_epochs * 0.75)
		warmup_epochs = 5
		lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=target_lr)
		# lr_scheduler = SequentialLR(
		# 	optimizer,
		# 	[
		# 		LambdaLR(
		# 			optimizer,
		# 			partial(warmup_lr_func, warmup_epochs=warmup_epochs),
		# 		),
		# 		CosineAnnealingLR(optimizer=optimizer, T_max=target_epoch, eta_min=target_lr),
		# 	],
		# 	milestones=[warmup_epochs],
		# )
		# target_epoch += warmup_epochs  # for cosine annealing, to account for the warmup

	if IS_DEBUG is False:
		training_sessions_params = {
			"validation_shuffle": data_params["shuffle"],
			"train_parameters": {**train_parameters},
			"model_parameters": {**model_parameters},
			"processing_parameters": {**processing_parameters},
			"optimizer": type(optimizer).__name__,
			"optimizer_parameters": [
				{key: val for key, val in para_group.items() if key != "params"}
				for para_group in optimizer.param_groups
			],
			"loss_function": type(loss_func).__name__,
			"model_architecture": model,
		}
		mlflow.log_params(training_sessions_params)
		if "plm_id" in model_parameters.keys():
			mlflow.set_tag("model_id", model_parameters["plm_id"])

	best_val_score = {"macf1": -1, "epoch": -1}
	best_ckpt_state = None
	early_stopper = EarlyStopper(patience=10, min_delta=1e-5)
	for epoch_idx in tqdm(range(num_epochs)):
		avg_train_loss, train_metrics = run_epoch(
			train_dataloader,
			model,
			optimizer,
			loss_func,
			epoch_idx,
			classification_lab2id,
			device,
			is_train=True,
		)
		avg_val_loss, val_metrics = run_epoch(
			val_dataloader,
			model,
			optimizer,
			loss_func,
			epoch_idx,
			classification_lab2id,
			device,
			is_train=False,
		)
		if IS_DEBUG is False:
			mlflow.log_metric(
				key="decaying_lr", value=lr_scheduler.get_last_lr()[-1], step=epoch_idx
			)
		if use_lr_scheduler is True and epoch_idx < target_epoch:
			lr_scheduler.step()

		if IS_DEBUG is False:
			mlflow.log_metrics(
				metrics={
					"train_loss": avg_train_loss,
					**train_metrics,
					"val_loss": avg_val_loss,
					**val_metrics,
				},
				step=epoch_idx,
			)

			if val_metrics["validation_macro_f1"] > best_val_score["macf1"]:
				best_val_score["macf1"] = val_metrics["validation_macro_f1"]
				best_val_score["epoch"] = epoch_idx

				best_ckpt_state = deepcopy(model.state_dict())
			if early_stopper.early_stop(best_val_score["macf1"]) is True:
				print("EARLY STOPPING")
				break

	if IS_DEBUG is False:
		mlflow.log_dict(best_val_score, "best_val_score.json")
		mlflow.pytorch.log_state_dict(best_ckpt_state, artifact_path="best_checkpoint")


if __name__ == "__main__":
	args = parse_args()

	seed = args["seed"]
	dataset_name = args["dataset"]
	path_data = args["path_data"]
	model_class_name = args["model_class"]
	model_config_path = args["model_config_path"]
	device = args["device"]
	model_override_config = args["model_parameters"]

	torch.manual_seed(seed)

	data_dir = f"{path_data}/{dataset_name}"
	train_config_path = "model_configs/train_paras.yaml"

	### start MLFlow setup
	current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
	exp_name = f"{dataset_name}_{model_class_name}"

	if IS_DEBUG is False:
		mlflow.set_experiment(exp_name)
		mlflow.config.enable_system_metrics_logging()
		logging.getLogger("mlflow").setLevel(logging.ERROR)
		mlflow.config.set_system_metrics_sampling_interval(1)

		with mlflow.start_run(run_name=f"run_seed={seed}_{current_date}") as run:
			torch.manual_seed(seed)
			train_torch_model(
				data_dir=data_dir,
				model_class_name=model_class_name,
				train_config_path=train_config_path,
				model_config_path=model_config_path,
				device=device,
				model_override_config=model_override_config,
			)
	else:
		torch.manual_seed(seed)
		train_torch_model(
			data_dir=data_dir,
			model_class_name=model_class_name,
			train_config_path=train_config_path,
			model_config_path=model_config_path,
			device=device,
			model_override_config=model_override_config,
		)
