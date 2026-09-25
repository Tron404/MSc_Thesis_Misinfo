import glob
import json
import os
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from retrieval_embedding_module import InstanceSamplePredicted
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def get_cm_heatmap(y_pred, y_true, label2id):
	cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
	labels = [lab.title() for lab in label2id.keys()]
	print(labels)
	fp = np.sum(cm, axis=0) - np.diag(cm)
	print("FP", fp)
	fn = np.sum(cm, axis=1) - np.diag(cm)
	print("FN", fn)
	tp = np.diag(cm)
	print("TP", tp)
	total_pred = dict(sorted(Counter(y_pred).items(), key=lambda x: x[0]))
	total_true = dict(sorted(Counter(y_true).items(), key=lambda x: x[0]))
	print(f"Total: {total_pred} --- {total_true}")

	plt.figure(figsize=(25, 14))
	heatmap = sns.heatmap(cm, annot=True, linewidths=0.5, fmt="d", annot_kws={"fontsize": 60})
	cbar = heatmap.collections[0].colorbar
	cbar.ax.tick_params(labelsize=60)

	plt.ylabel("True Label", fontsize=55, labelpad=10)
	plt.xlabel("Predicted Label", fontsize=55, labelpad=10)
	plt.xticks(
		ticks=[k + 0.5 for k in label2id.values()],
		labels=labels,
		fontsize=55,
		rotation=45,
		ha="right",
		rotation_mode="anchor",
	)
	plt.yticks(ticks=[k + 0.5 for k in label2id.values()], labels=labels, fontsize=55, rotation=0)
	plt.tight_layout()

	return heatmap


pred_responses_collection = defaultdict(list)
for pred_path in glob.glob("predicted_test_samples_CURRENT/**/*.pkl", recursive=True):
	pred_responses_collection[os.path.join(*pred_path.split("/")[:-2])] += [pred_path]

# skip_exp = ["llama", "gpt"]
skip_exp = []
only_exp = [
	# "gpt-4o-mini_claim_evidence_filtered_evidence",
	# "llama-3.3-70b-instruct_claim_evidence_filtered_evidence",
	# "FcGalm",
	# "TextOnly_text_acc",
	# "GnnOnly_gnn_homo",
	# "GnnOnlyMain_gnn_lora_homo",
]

for exp_condition, pred_responses_path in pred_responses_collection.items():
	do_skip_exp = False
	for sk_exp in skip_exp:
		if sk_exp in exp_condition.lower():
			do_skip_exp = True
			break
	if len(only_exp) > 0 and exp_condition.split("/")[-2] not in only_exp:
		continue
	if do_skip_exp is True:
		continue

	classification_lab2id = json.load(open(f"{exp_condition}/classification_lab2id_map.json"))

	pred_labels = []
	true_labels = []
	for file_path in pred_responses_path:
		sample: InstanceSamplePredicted = pickle.load(open(file_path, "rb"))

		pred_labels += [classification_lab2id[sample.pred_label]]
		true_labels += [classification_lab2id[sample.gt_label]]

	p, r, f1, _ = precision_recall_fscore_support(
		y_pred=pred_labels, y_true=true_labels, average="macro", zero_division=0
	)

	print(f"=== {exp_condition} {len(pred_labels)} ===")
	print(f"{p * 100:0.2f} & {r * 100:0.2f} & {f1 * 100:0.2f}")
	# cm_figure = get_cm_heatmap(
	# 	y_pred=pred_labels, y_true=true_labels, label2id=classification_lab2id
	# )
	# model_name = exp_condition.split("/")[-2]
	# cm_figure.figure.savefig(f"{exp_condition}_{model_name}_cm.pdf", transparent=True)
	# plt.show()
	print()
