import json
import os


def load_rawfc_data(split):
	input_path_dir = f"../datasets/data/RAWFC/{split}"
	data = [
		json.load(open(f"{input_path_dir}/{json_file}", "r"))
		for json_file in os.listdir(input_path_dir)
	]
	return data


def load_liaraw_data(split):
	input_path_dir = f"../datasets/data/LIAR-RAW/{split}.json"
	data = json.load(open(input_path_dir, "r"))
	return data


def load_rawfc_sents(split):
	input_path_dir = f"../datasets/data/RAWFC/{split}"
	data = [
		json.load(open(f"{input_path_dir}/{json_file}", "r"))
		for json_file in os.listdir(input_path_dir)
	]
	sents = [
		sent["sent"]
		for sample in data
		for report in sample["reports"]
		for sent in report["tokenized"]
	]
	return list(set(sents))


def load_liaraw_sents(split):
	input_path_dir = f"../datasets/data/LIAR-RAW/{split}.json"
	data = json.load(open(input_path_dir, "r"))
	sents = [
		sent["sent"]
		for sample in data
		for report in sample["reports"]
		for sent in report["tokenized"]
	]
	return list(set(sents))
