import os
import pickle
from collections import defaultdict

import torch
from retrieval_embedding_module import EmbeddingModule, InstanceSample
from retrieval_utils import load_liaraw_data, load_rawfc_data
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = DEVICE.type
BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
	embedder = EmbeddingModule(model_name=MODEL_NAME, batch_size=BATCH_SIZE, device=DEVICE_STR)

	datasets = ["RAWFC", "LIAR-RAW"]
	splits = ["train", "val", "test"]

	for dataset_name in datasets:
		for dataset_split in splits:
			match dataset_name:
				case "LIAR-RAW":
					data = load_liaraw_data(dataset_split)
				case "RAWFC":
					data = load_rawfc_data(dataset_split)

			output_dir = f"embeddings/{dataset_name}/{dataset_split}"
			os.makedirs(output_dir, exist_ok=True)
			for sample in tqdm(data, desc=f"{dataset_name}:{dataset_split}"):
				event_id = sample["event_id"]
				event_id = event_id.split(".")[0]  # to remove any existing file extensions

				sent_to_emb = []
				sents_dict = {}
				reports_dict = defaultdict(list)
				sent_idx = 0
				for report_idx, report in enumerate(sample["reports"]):
					for sentence in report["tokenized"]:
						if (
							"is_repeat" in sentence.keys()
						):  # from RAWFC - apparently not all sentences have this...
							sentence.pop("is_repeat")

						text = sentence.pop("sent")
						sent_to_emb += [text]
						reports_dict[report_idx] += [sent_idx]
						sents_dict[sent_idx] = {
							"text": text,
							"sent_idx": sent_idx,
							"report_idx": report_idx,
							**sentence,
						}
						sent_idx += 1

				sent_to_emb = [sample["claim"], *sent_to_emb]
				all_embeddings = embedder(sent_to_emb).cpu()
				instance_embedded = InstanceSample(
					claim_text=sample["claim"],
					claim_embedding=all_embeddings[0],
					reports_content=reports_dict,
					sentence_texts=sents_dict,
					evidence_sentences_embeddings=all_embeddings[1:],
					gt_explanation=sample["explain"],
					gt_label=sample["label"],
				)
				pickle.dump(
					instance_embedded, open(os.path.join(output_dir, f"{event_id}.pkl"), "wb")
				)
