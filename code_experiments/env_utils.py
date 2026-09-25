import os


def load_env_from_file(directory_path):
	for file_path in os.listdir(directory_path):
		if ".env" != file_path:
			continue
		with open(directory_path + "/" + file_path, "r") as file:
			for line in file.readlines():
				if line[0] == "#":
					continue
				(key_name, key_value) = line.split("=")

				os.environ[key_name] = key_value.strip().strip('"')
