import os
import shutil


def export_library(library_dir, destination_folder):
	"""Copy the whole Library folder into destination_folder."""
	if not os.path.isdir(library_dir):
		raise FileNotFoundError(f"Library folder not found: {library_dir}")
	if not destination_folder:
		raise ValueError("Destination folder is empty")

	target = os.path.join(destination_folder, os.path.basename(library_dir.rstrip('/\\')))
	if os.path.abspath(target) == os.path.abspath(library_dir):
		raise ValueError("Destination points to the current Library folder")
	shutil.copytree(library_dir, target, dirs_exist_ok=True)
	return target


def import_library(source_folder, library_dir):
	"""Copy all .mdl files from source_folder into library_dir with overwrite."""
	if not source_folder:
		raise ValueError("Source folder is empty")
	if not os.path.isdir(source_folder):
		raise FileNotFoundError(f"Source folder not found: {source_folder}")

	os.makedirs(library_dir, exist_ok=True)
	copied = 0
	for name in os.listdir(source_folder):
		src_path = os.path.join(source_folder, name)
		if os.path.isfile(src_path) and name.lower().endswith('.mdl'):
			dst_path = os.path.join(library_dir, name)
			shutil.copy2(src_path, dst_path)
			copied += 1
	return copied

