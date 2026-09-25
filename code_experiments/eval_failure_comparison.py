import glob
import pickle
from collections import Counter, defaultdict

from retrieval_embedding_module import InstanceSamplePredicted
from tqdm import tqdm

test_fails = defaultdict(list)
common_fails = Counter()
prediction_fails = defaultdict(lambda: defaultdict(list))
for pred_file_path in tqdm(glob.glob("predicted_test_samples_CURRENT/**/*.pkl", recursive=True)):
	if "LIAR-RAW" not in pred_file_path:
		continue
	model_name = pred_file_path.split("/")[1]
	file_id = pred_file_path.split("/")[-1].removesuffix(".pkl")
	sample: InstanceSamplePredicted = pickle.load(open(pred_file_path, "rb"))
	if sample.pred_label != sample.gt_label:
		common_fails[file_id] += 1
		prediction_fails[file_id][sample.pred_label] += [model_name]

print(common_fails.most_common(10))
most_common_id = common_fails.most_common(2)[1][
	0
]  # top 3 are equal, i just chose the ones i liked best in terms of claims
print(most_common_id)
print(prediction_fails[most_common_id])


# RAWFC = 175243 - most misclassified in test set; most misclassified as false (11 times) is actually true

## false (12): ['gpt-4o-mini_claim_evidence_filtered_evidence', 'llama-3.3-70b-instruct_claim_evidence_filtered_evidence',
##### 'llama-3.3-70b-instruct_claim_evidence', 'gpt-4o-mini_claim_evidence', 'dt_classifier', 'GnnOnlyMain_gnn_lora_hetero',
##### 'GnnOnlyMain_gnn_lora_homo', 'llama-3.3-70b-instruct_claim_only', 'FcGalm', 'GnnOnlyMain_gnn_hetero', 'gpt-4o-mini_claim_only', 'TextOnly_text_lora']

## half (5): ['Mlp', 'GnnOnlyMain_gnn_homo', , 'Mlp_mlp_lora', 'svm', 'TextOnly']

# ----

# LIAR-RAW = 8963; most confused label: false (8), is actually pants-fire


## false (8): ['gpt-4o-mini_claim_evidence_filtered_evidence', 'GnnOnlyMain_gnn_homo',
##### 'gpt-4o-mini_claim_evidence', 'GnnOnlyMain_gnn_lora_hetero',
##### , 'svm', 'gpt-4o-mini_claim_only', 'TextOnly', 'FcGalm']

## half-true (4): ['llama-3.3-70b-instruct_claim_evidence_filtered_evidence', 'llama-3.3-70b-instruct_claim_evidence',
##### 'dt_classifier', 'llama-3.3-70b-instruct_claim_only']

## barely-true (5): ['Mlp', 'Mlp_mlp_lora', , 'GnnOnlyMain_gnn_hetero', 'GnnOnlyMain_gnn_lora_homo', 'TextOnly_text_lora']
