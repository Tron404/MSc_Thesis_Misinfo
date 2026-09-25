import ast
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

package_dir = Path(__file__).resolve().parent


# https://stackoverflow.com/questions/44698193/how-to-get-a-list-of-classes-and-functions-from-a-python-file-without-importing
def find_arch_collate(module_filepath):
	with open(module_filepath, "r") as module_file:
		syntax_tree = ast.parse(module_file.read())
		nodes_interest = {"classes": [], "method": None}
		for node in syntax_tree.body:
			if isinstance(node, ast.ClassDef):
				nodes_interest["classes"] += [node.name]
			elif isinstance(node, ast.FunctionDef):
				nodes_interest["method"] = node.name
	return nodes_interest["classes"], nodes_interest["method"]


_model_architecture_modules = {}
for _, module_name, _ in iter_modules(__path__):
	module_filepath = f"{package_dir}/{module_name}.py"
	architecures, collate_func = find_arch_collate(module_filepath)
	for a_name in architecures:
		_model_architecture_modules[a_name] = (module_name, collate_func)


def import_model_class(model_class: str):
	module_name, collate_func = _model_architecture_modules[model_class]
	module = import_module(f"{__name__}.{module_name}")
	return getattr(module, model_class), getattr(module, collate_func)
