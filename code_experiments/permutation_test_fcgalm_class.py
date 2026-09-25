import glob
import json
import os
import pickle
from collections import defaultdict
from itertools import product

import numpy as np
from nli_dt_baseline import get_dt_preds
from retrieval_embedding_module import InstanceSamplePredicted
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

np.random.seed(42)


def get_svm_preds(dataset):
	train_data = pickle.load(open(f"train_embeddings_{dataset}.pkl", "rb"))
	train_embeddings, train_labels = train_data["emb"], train_data["labels"]

	test_data = pickle.load(open(f"test_embeddings_{dataset}.pkl", "rb"))
	test_embeddings, test_labels = test_data["emb"], test_data["labels"]

	svc_file_path = f"svc_{dataset}.pkl"
	if os.path.exists(svc_file_path) is False:
		svc = SVC(random_state=42)
		svc.fit(train_embeddings, train_labels)
		pickle.dump(svc, open(svc_file_path, "wb"))
	else:
		svc = pickle.load(open(svc_file_path, "rb"))

	pred_labels = svc.predict(test_embeddings)

	p, r, f1, _ = precision_recall_fscore_support(
		y_pred=pred_labels, y_true=test_labels, zero_division=0, average="macro"
	)

	print(p, r, f1)

	return {
		"true_lab": test_labels,
		"pred_lab": pred_labels,
	}


results_dir_root = "predicted_test_samples_CURRENT"
results_collections = defaultdict(list)

skip_exp = ["gpt", "llama"]

for pred_file in glob.glob(f"{results_dir_root}/**/*.pkl", recursive=True):
	do_skip = False
	for sk_exp in skip_exp:
		if sk_exp in pred_file.lower():
			do_skip = True
			break
	if do_skip is True:
		continue
	exp_condition = "---".join(pred_file.split("/")[1:-2])
	results_collections[exp_condition] += [pred_file]


def random_classifier_baseline(num_classes, num_samples):
	preds = np.random.choice(range(num_classes), num_samples).tolist()
	return preds


exp_to_compare = [
	"FcGalm---RAWFC",
	"FcGalm---LIAR-RAW",
	"GnnOnly_gnn_lora_hetero---RAWFC",
	"GnnOnly_gnn_lora_hetero---LIAR-RAW",
	"TextOnly_text_lora---RAWFC",
	"TextOnly---LIAR-RAW",
	"Mlp_mlp_lora---RAWFC",
	"Mlp_mlp_lora---LIAR-RAW",
]

labels_collections = defaultdict(dict)
unique_exp = defaultdict(set)
for exp_condition, files in results_collections.items():
	if exp_condition not in exp_to_compare:
		continue
	pred_labels = []
	true_labels = []
	dataset_name = exp_condition.split("---")[-1]
	classification_lab2id = json.load(
		open(
			f"{results_dir_root}/{exp_condition.replace('---', '/')}/classification_lab2id_map.json",
			"r",
		)
	)
	exp_condition = exp_condition.split("---")[0]
	unique_exp[dataset_name] |= set([exp_condition])

	for pred_file in files:
		test_results: InstanceSamplePredicted = pickle.load(open(pred_file, "rb"))
		pred_labels += [classification_lab2id[test_results.pred_label]]
		true_labels += [classification_lab2id[test_results.gt_label]]
	labels_collections[dataset_name][exp_condition] = {
		"true_lab": np.asarray(true_labels),
		"pred_lab": np.asarray(pred_labels),
	}


# labels_collections["SVM---RAWFC"] = get_svm_preds(dataset="RAWFC")
# labels_collections["SVM---LIAR-RAW"] = get_svm_preds(dataset="LIAR-RAW")
# labels_collections["DT---RAWFC"] = get_dt_preds(
# 	dataset="RAWFC", data_dir="nli_relations_r10-s5_deberta-large"
# )
# labels_collections["DT---LIAR-RAW"] = get_dt_preds(
# 	dataset="LIAR-RAW", data_dir="nli_relations_r10-s5_deberta-large"
# )

unique_exp["RAWFC"] |= set(["random_classifier"])
unique_exp["LIAR-RAW"] |= set(["random_classifier"])

labels_collections["RAWFC"]["random_classifier"] = {
	"pred_lab": random_classifier_baseline(3, 200),
	"true_lab": labels_collections["RAWFC"]["FcGalm"]["true_lab"],
}

labels_collections["LIAR-RAW"]["random_classifier"] = {
	"pred_lab": random_classifier_baseline(6, 1251),
	"true_lab": labels_collections["LIAR-RAW"]["FcGalm"]["true_lab"],
}


def permutation_test(pred_a, pred_b, true_labs, num_permutations=10):
	f1_pred_a = f1_score(true_labs, pred_a, average=None, zero_division=0)
	f1_pred_b = f1_score(true_labs, pred_b, average=None, zero_division=0)
	observed_stat = np.abs(f1_pred_a - f1_pred_b)

	permuted_stats = []
	for _ in range(num_permutations):
		swap_idx = np.random.random(len(true_labs)) < 0.5

		permutation_pred_a = np.where(swap_idx, pred_b, pred_a)
		permutation_pred_b = np.where(swap_idx, pred_a, pred_b)

		permutation_f1_pred_a = f1_score(
			true_labs, permutation_pred_a, average=None, zero_division=0
		)
		permutation_f1_pred_b = f1_score(
			true_labs, permutation_pred_b, average=None, zero_division=0
		)
		permuted_stats += [np.abs(permutation_f1_pred_a - permutation_f1_pred_b)]

	permuted_stats = np.asarray(permuted_stats)
	p_value = (np.sum(permuted_stats >= observed_stat, axis=0) + 1) / (num_permutations + 1)

	return p_value, observed_stat, f1_pred_a, f1_pred_b


for dataset, model_exp in labels_collections.items():
	print(dataset)
	actual_compared_models = unique_exp[dataset] - set(["FcGalm"])
	comparison_pairs = product(["FcGalm"], actual_compared_models)

	classification_lab2id = json.load(
		open(
			f"predicted_test_samples/FcGalm/{dataset}/classification_lab2id_map.json",
			"r",
		)
	)

	p_scores = {}
	stats = {}
	for model_a, model_b in comparison_pairs:
		pred_a = model_exp[model_a]["pred_lab"]
		pred_b = model_exp[model_b]["pred_lab"]

		true_a = model_exp[model_a]["true_lab"]
		true_b = model_exp[model_b]["true_lab"]

		assert False not in (true_a == true_b), "order of true labels is not the same!!"

		p_value, observed_stat, f1_pred_a, f1_pred_b = permutation_test(
			pred_a, pred_b, true_a, num_permutations=10000
		)

		pair = f"{model_a}---{model_b}"
		p_scores[pair] = p_value
		stats[pair] = {
			"delta_f1": observed_stat,
			f"f1_{model_a}": f1_pred_a,
			f"f1_{model_b}": f1_pred_b,
			"p_val": p_value,
		}

	print(actual_compared_models)
	p_scores_mat = np.asarray(list(p_scores.values()))
	print(p_scores_mat)
	print("---")
	print(p_scores_mat.T)
	print("============")

	all_corrected_p = []
	for p_model_class in p_scores_mat.T:
		can_reject, corrected_p, p, corr_alpha = multipletests(
			p_model_class, method="holm", alpha=0.05
		)
		all_corrected_p += [corrected_p]
	all_corrected_p = np.asarray(all_corrected_p).T

	print(p_scores_mat)
	print("----")
	print(all_corrected_p)
	print(classification_lab2id)

	final_stats = {}
	for idx, compared_model in enumerate(actual_compared_models):
		final_stats[compared_model] = {
			class_lab: p_val_class
			for class_lab, p_val_class in zip(classification_lab2id.keys(), all_corrected_p[idx])
		}

	pickle.dump(final_stats, open(f"all_p_values_{dataset}_fcgalm_perclass.pkl", "wb"))

	# print(f"=== correction alpha={corr_alpha}===")
	# for reject, cor_p, (pair, orig_p) in zip(can_reject, corrected_p, p_scores.items()):
	# 	stats[pair] = {"to_reject": reject, "corrected_p": cor_p, **stats[pair]}
	# 	print(round(cor_p, 5), round(orig_p, 5), reject, pair)
	# print("=======================")
