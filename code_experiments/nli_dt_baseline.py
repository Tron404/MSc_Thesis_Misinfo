import glob
import json
import os
import pickle
from collections import Counter

import numpy as np
from retrieval_embedding_module import InstanceSampleNli
from sklearn.tree import DecisionTreeClassifier


def load_data_counts(path):
	files = glob.glob(f"{path}/*.pkl")
	lab2id_map = json.load(open(f"{os.path.join(*path.split('/')[:-2])}/id2lab_map.json", "r"))
	classificaiton_lab2id_map = json.load(
		open(f"{os.path.join(*path.split('/')[:-1])}/classification_lab2id_map.json", "r")
	)

	x, y = [], []
	for f in files:
		sample: InstanceSampleNli = pickle.load(open(f, "rb"))
		claim_ev_labels = Counter({int(lab): 0 for lab in lab2id_map.keys()})
		for premise_id, hypo_id, top_label, probs in sample.nli_tuples:
			#! only consider claim<-evidence pairs
			if hypo_id != -1:
				continue

			claim_ev_labels[top_label] += 1
		x += [list(claim_ev_labels.values())]
		y += [classificaiton_lab2id_map[sample.gt_label]]

	return x, y


def get_dt_preds(dataset, data_dir):
	x_train, y_train = load_data_counts(f"{data_dir}/{dataset}/train")
	x_val, y_val = load_data_counts(f"{data_dir}/{dataset}/test")

	dt_classifier = DecisionTreeClassifier(random_state=42)
	dt_classifier = dt_classifier.fit(x_train, y_train)

	y_pred = dt_classifier.predict(x_val)

	return {"pred_lab": y_pred, "true_lab": np.asarray(y_val)}
