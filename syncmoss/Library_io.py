import os
import re
import shutil
from collections import Counter

from syncmoss.constants import numco

LIBRARY_METADATA_FIELDS = [
	'Chemical composition',
	'Temperature (K)',
	'Pressure (GPa)',
	'External field (T)',
	'DOI',
]

LIBRARY_METADATA_DEFAULTS = {
	'Chemical composition': '',
	'Temperature (K)': '',
	'Pressure (GPa)': '',
	'External field (T)': '',
	'DOI': '',
}

# Must stay aligned with names shown by ParametersTable.auto_fill_params().
MODEL_PARAMETER_NAMES = {
	'baseline': ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr'],
	'Singlet': ['T', 'δ, mm/s', 'L, mm/s', 'G, mm/s'],
	'Doublet': ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1'],
	'Sextet': ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'A', 'a+', 'a-', 'GH, T', 'I1/I3'],
	'MDGD': ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'GH, T', 'Dδε', 'DδH', 'DεH', 'A', 'a+', 'a-', 'I1/I3'],
	'Hamilton_mc': ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °', 'θ, °', 'φ, °'],
	'Hamilton_pc': ['T', 'δ, mm/s', 'Q, mm/s', 'H, T', 'L, mm/s', 'G, mm/s', 'η', 'θH, °', 'φH, °'],
	'Relax_MS': ['T', 'δ, mm/s', 'ε, mm/s', 'H, T', 'L, mm/s', 'A', 'R', 'alfa', 'S'],
	'Relax_2S': ['T', 'δ1, mm/s', 'ε1, mm/s', 'H1, T', 'δ2, mm/s', 'ε2, mm/s', 'H2, T', 'L, mm/s', 'A', 'Ω12', 'P1/P2'],
	'ASM': ['T', 'δ, mm/s', 'εm, mm/s', 'εl, mm/s', 'His, T', 'Han, T', 'L, mm/s', 'G, mm/s', 'm', 'A', 'Num', 'I13'],
	'Be': ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1'],
	'KB_nano': ['T', 'δ, mm/s', 'ε, mm/s', 'L, mm/s', 'G, mm/s', 'A', 'G2/G1'],
	'Variables': [f'V{i + 1}' for i in range(numco)],
	'Nbaseline': ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr'],
	'Expression': ['Expression'],
	'Average_H': ['T', 'δ, mm/s', 'ε, mm/s', 'Hin, T', 'L, mm/s', 'G, mm/s', 'Hex, T', 'K', 'J', 'θ, °', 'N'],
	'Distr': ['par', 'L', 'R', 'Num', 'Probability density function'],
	'Corr': ['par', 'Dependency function'],
}


def _parse_metadata_comment_line(line):
	"""Parse '#@<key> <value>' metadata line."""
	stripped = line.strip()
	if not stripped.startswith('#@'):
		return None, None

	payload = stripped[2:].strip()
	if not payload:
		return None, None

	for key in LIBRARY_METADATA_FIELDS:
		if payload.lower().startswith(key.lower()):
			value = payload[len(key):].strip()
			return key, value

	return None, None


def parse_library_model_file(file_path):
	"""Parse a library model file into metadata and preview-friendly structures."""
	metadata = dict(LIBRARY_METADATA_DEFAULTS)
	comments = []
	data_lines = []

	with open(file_path, 'r', encoding='utf-8') as file:
		for raw_line in file:
			line = raw_line.rstrip('\n')
			stripped = line.strip()
			if not stripped:
				continue

			if stripped.startswith('#@'):
				key, value = _parse_metadata_comment_line(stripped)
				if key is not None:
					metadata[key] = value
				else:
					payload = stripped[2:].strip()
					if payload.lower().startswith('comment'):
						payload = payload[7:].strip()
					if payload:
						comments.append(payload)
				continue

			if stripped.startswith('#'):
				comments.append(stripped.lstrip('#').strip())
				continue

			data_lines.append(line)

	models = []
	model_colors = []
	table_rows = []
	validation_warnings = []

	if data_lines:
		m_list = [line.split('\t') for line in data_lines]
		models = m_list[0]

		# Backward compatibility: detect color row.
		has_colors = False
		if len(m_list) > 1:
			color_names = ['blue', 'red', 'yellow', 'cyan', 'fuchsia', 'lime', 'darkorange', 'blueviolet', 'green', 'tomato', 'white', 'silver', 'lightgreen', 'pink']
			second_line = m_list[1]
			has_colors = all((field.startswith('#') and len(field) == 7) or field in color_names or field == '' for field in second_line)
			if has_colors:
				model_colors = list(second_line)

		if not model_colors:
			model_colors = [''] * len(models)

		param_start_idx = 2 if has_colors else 1
		row_data_lines = m_list[param_start_idx:]

		for row_idx, model_name in enumerate(models):
			if model_name in ['', 'None']:
				continue

			row_data = row_data_lines[row_idx] if row_idx < len(row_data_lines) else []
			row_param_blocks = len(row_data) // 5
			meaningful_param_indices = []
			for p_idx in range(row_param_blocks):
				base_idx = p_idx * 5
				if base_idx + 4 >= len(row_data):
					break
				value = str(row_data[base_idx]).strip()
				lower = str(row_data[base_idx + 1]).strip()
				upper = str(row_data[base_idx + 2]).strip()
				stored_name = str(row_data[base_idx + 3]).strip()
				# Treat fully empty parameter slots as absent.
				if value == '' and lower == '' and upper == '' and stored_name == '':
					continue
				meaningful_param_indices.append(p_idx)

			row_param_count = len(meaningful_param_indices)
			expected_names = list(MODEL_PARAMETER_NAMES.get(model_name, []))
			expected_count = len(expected_names)

			if expected_count > 0 and row_param_count != expected_count:
				validation_warnings.append(
					f"{model_name}: file has {row_param_count} parameter(s), expected {expected_count}"
				)

			for p_idx in meaningful_param_indices:
				base_idx = p_idx * 5
				if base_idx + 4 >= len(row_data):
					break

				value = str(row_data[base_idx]).strip()
				if value == '':
					# Requested behavior: if value is missing, parameter name must be empty too.
					param_name = ''
				elif p_idx < expected_count:
					param_name = expected_names[p_idx]
				else:
					stored_name = row_data[base_idx + 3].strip() if base_idx + 3 < len(row_data) else ''
					param_name = stored_name or f'p{p_idx + 1}'

				table_rows.append({
					'row_index': row_idx,
					'model': model_name,
					'param_name': param_name,
					'value': row_data[base_idx],
					'lower': row_data[base_idx + 1],
					'upper': row_data[base_idx + 2],
					'fixed': row_data[base_idx + 4],
				})

	active_models = [m for m in models if m and m not in ['None', 'baseline']]
	submodel_counts = Counter(active_models)

	return {
		'file_name': os.path.basename(file_path),
		'file_path': file_path,
		'title': os.path.splitext(os.path.basename(file_path))[0],
		'metadata': metadata,
		'comments': comments,
		'models': models,
		'model_colors': model_colors,
		'active_models': active_models,
		'submodel_counts': submodel_counts,
		'table_rows': table_rows,
		'validation_warnings': validation_warnings,
	}


def _ensure_library_metadata(file_path):
	"""Ensure all library metadata lines exist in the model file.

	Missing fields are added as empty '#@<key> ' lines at file top.
	"""
	if not os.path.isfile(file_path):
		return
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()

		existing = set()
		for raw in lines:
			stripped = raw.strip()
			if not stripped.startswith('#@'):
				continue
			payload = stripped[2:].strip()
			for key in LIBRARY_METADATA_FIELDS:
				if payload.lower().startswith(key.lower()):
					existing.add(key)
					break

		missing = [k for k in LIBRARY_METADATA_FIELDS if k not in existing]
		if not missing:
			return

		prefix = [f"#@{k} \n" for k in missing]
		with open(file_path, 'w', encoding='utf-8') as f:
			f.writelines(prefix + lines)
	except Exception:
		# Keep import/export resilient; skip broken files silently here.
		return


def _files_are_exactly_equal(path_a, path_b):
	"""Return True when two files are equivalent for Library dedup.

	For .mdl files this comparison is metadata-normalization aware, so a file
	without explicit metadata lines matches the same file after import-time
	normalization added missing '#@...' lines.
	"""
	try:
		ext_a = os.path.splitext(path_a)[1].lower()
		ext_b = os.path.splitext(path_b)[1].lower()
		if ext_a == '.mdl' and ext_b == '.mdl':
			ca = _canonical_library_text_for_compare(path_a)
			cb = _canonical_library_text_for_compare(path_b)
			if ca is not None and cb is not None:
				return ca == cb

		if os.path.getsize(path_a) != os.path.getsize(path_b):
			return False
		with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
			while True:
				ca = fa.read(1024 * 1024)
				cb = fb.read(1024 * 1024)
				if ca != cb:
					return False
				if not ca:
					return True
	except Exception:
		return False


def _canonical_library_text_for_compare(file_path):
	"""Build canonical text for .mdl comparison, normalizing known metadata lines."""
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()

		metadata_values = {}
		other_lines = []
		for raw in lines:
			stripped = raw.strip()
			if stripped.startswith('#@'):
				key, value = _parse_metadata_comment_line(stripped)
				if key is not None:
					metadata_values[key] = value
					continue
			other_lines.append(raw.replace('\r\n', '\n').replace('\r', '\n'))

		prefix = [f"#@{key} {metadata_values.get(key, '')}\n" for key in LIBRARY_METADATA_FIELDS]
		return ''.join(prefix + other_lines)
	except Exception:
		return None


def _base_stem_without_copy_suffix(stem):
	"""Return stem without trailing ' (N)' suffix to avoid 'name (2) (3)' forms."""
	m = re.match(r'^(.*?)(?:\s\((\d+)\))?$', stem)
	if not m:
		return stem
	base = (m.group(1) or stem).rstrip()
	return base or stem


def _split_stem_and_version(stem):
	"""Split title stem into (base_stem, version_index) where base has no trailing ' (N)'."""
	m = re.match(r'^(.*?)(?:\s\((\d+)\))?$', stem)
	if not m:
		clean = stem.strip()
		return clean or stem, 1
	base = (m.group(1) or stem).rstrip() or stem
	idx = int(m.group(2)) if m.group(2) else 1
	return base, max(1, idx)


def library_model_sort_key(name_or_title):
	"""Sort key for library model names/titles: alphabetical by base, then version index.

	Examples:
	- test, test (2), test (3)
	- alpha, beta, beta (2)
	"""
	text = str(name_or_title or '')
	stem = os.path.splitext(text)[0]
	base, idx = _split_stem_and_version(stem)
	return (base.lower(), idx, stem.lower())


def _next_version_index_for_base(library_dir, base_stem, ext='.mdl'):
	"""Return next version index for base title among existing files in library_dir."""
	base_l = base_stem.lower()
	max_idx = 1
	found_any = False
	for existing_name in os.listdir(library_dir):
		existing_path = os.path.join(library_dir, existing_name)
		if not os.path.isfile(existing_path):
			continue
		existing_stem, existing_ext = os.path.splitext(existing_name)
		if existing_ext.lower() != ext.lower():
			continue
		existing_base, existing_idx = _split_stem_and_version(existing_stem)
		if existing_base.lower() == base_l:
			found_any = True
			if existing_idx > max_idx:
				max_idx = existing_idx
	return (max_idx + 1) if found_any else 2


def compute_versioned_title_if_needed(library_dir, requested_title):
	"""Resolve title collisions for save-to-library.

	Returns:
		(tuple): (final_title, version_index_or_None)
		- version_index_or_None is the numeric suffix when a rename happened.
	"""
	title = (requested_title or '').strip()
	if not title:
		return title, None

	direct_path = os.path.join(library_dir, f"{title}.mdl")
	if not os.path.exists(direct_path):
		return title, None

	base, _ = _split_stem_and_version(title)
	next_idx = _next_version_index_for_base(library_dir, base, ext='.mdl')
	return f"{base} ({next_idx})", next_idx


def _source_sort_key(name):
	"""Sort source models by base title and numeric version to preserve intuitive order."""
	stem, ext = os.path.splitext(name)
	base, idx = _split_stem_and_version(stem)
	return (base.lower(), idx, stem.lower(), ext.lower())


def _find_identical_existing_for_base(src_path, library_dir, base_stem, ext='.mdl'):
	"""Check whether src_path content already exists in any same-base library version."""
	base_l = base_stem.lower()
	for existing_name in os.listdir(library_dir):
		existing_path = os.path.join(library_dir, existing_name)
		if not os.path.isfile(existing_path):
			continue
		existing_stem, existing_ext = os.path.splitext(existing_name)
		if existing_ext.lower() != ext.lower():
			continue
		existing_base, _ = _split_stem_and_version(existing_stem)
		if existing_base.lower() != base_l:
			continue
		if _files_are_exactly_equal(src_path, existing_path):
			return existing_name
	return None


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
	for name in os.listdir(target):
		path = os.path.join(target, name)
		if os.path.isfile(path) and name.lower().endswith('.mdl'):
			_ensure_library_metadata(path)
	return target


def import_library(source_folder, library_dir):
	"""Copy .mdl files into library_dir with exact-content dedup and numbered renaming.

	Returns:
		dict with keys:
		- copied: number of copied files
		- skipped_identical: number skipped because content already existed
		- renamed: list of (original_name, new_name)
	"""
	if not source_folder:
		raise ValueError("Source folder is empty")
	if not os.path.isdir(source_folder):
		raise FileNotFoundError(f"Source folder not found: {source_folder}")

	os.makedirs(library_dir, exist_ok=True)
	renamed = []
	copied = 0
	skipped_identical = 0

	source_names = [
		name for name in os.listdir(source_folder)
		if os.path.isfile(os.path.join(source_folder, name)) and name.lower().endswith('.mdl')
	]
	source_names = sorted(source_names, key=_source_sort_key)

	for name in source_names:
		src_path = os.path.join(source_folder, name)
		stem, ext = os.path.splitext(name)
		base_stem, _ = _split_stem_and_version(stem)

		# Deduplicate against any existing version for this base title.
		identical_name = _find_identical_existing_for_base(src_path, library_dir, base_stem, ext=ext or '.mdl')
		if identical_name is not None:
			skipped_identical += 1
			continue

		direct_path = os.path.join(library_dir, name)
		if os.path.exists(direct_path):
			next_idx = _next_version_index_for_base(library_dir, base_stem, ext=ext or '.mdl')
			target_name = f"{base_stem} ({next_idx}){ext}"
		else:
			target_name = name

		dst_path = os.path.join(library_dir, target_name)
		shutil.copy2(src_path, dst_path)
		_ensure_library_metadata(dst_path)
		copied += 1
		if target_name != name:
			renamed.append((name, target_name))

	return {
		'copied': copied,
		'skipped_identical': skipped_identical,
		'renamed': renamed,
	}

