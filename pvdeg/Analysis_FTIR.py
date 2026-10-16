# Created: 9-23-2025
# Author: Trenton Wells
# Organization: NLR
# NLR Contact: trenton.wells@nrel.gov
# Personal Contact: trentonwells73@gmail.com

import os, re, json, math, contextlib, threading, traceback, time, collections, importlib, copy
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from pybaselines.whittaker import arpls
from pybaselines.spline import irsqr
from pybaselines.classification import fabc
from scipy.interpolate import CubicSpline
from pybaselines.utils import optimize_window
from scipy.signal import find_peaks
from lmfit.models import PseudoVoigtModel
from IPython.display import clear_output, display
import ast, html

# Environment detection (used for some interactive behaviors)
try:
    import google.colab  # type: ignore
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

# Lazy peak widget mode (defer heavy slider construction until user expands a peak).
LAZY_PEAK_WIDGETS = True
# Deconvolution debug instrumentation flag
DECONV_DEBUG = False
_DECONV_DEBUG_ERRORS = []  # list of traceback strings for failed materializations
try:
    _DECONV_DEBUG_LOG  # type: ignore[name-defined]
except Exception:
    _DECONV_DEBUG_LOG = []  # persistent list of debug lines

def _debug_log(msg: str):
    if not isinstance(msg, str):
        try:
            msg = str(msg)
        except Exception:
            if (
                y_fit is None
                or y is None
                or x is None
                or len(y) != len(x)
                or len(y_fit) != len(x)
            ):
                if tsfr in (None, "", [], {}):
                    note = note or "Fit not ran"
                else:
                    note = note or "unusable fit"
                err = np.inf
            else:
                arr_y = np.array(y, dtype=float)
                arr_fit = np.array(y_fit, dtype=float)
                dif = arr_y - arr_fit
                # Reduced Chi-Square: chi2 / dof, where dof = N - m
                N = float(arr_y.size)
                # Estimate m as parameters used per component (amplitude, center, sigma, fraction)
                try:
                    m = 0.0
                    if isinstance(tsfr, list):
                        m = float(sum(1 for comp in tsfr if isinstance(comp, dict)) * 4)
                    elif isinstance(tsfr, dict):
                        # If direct fit vector provided, assume minimal parameters
                        m = 1.0
                except Exception:
                    m = 0.0
                dof = max(N - m, 1.0)
                chi2 = float(np.sum(dif ** 2))
                err = chi2 / dof
            print(f"Deconvolution debug {'ENABLED' if DECONV_DEBUG else 'DISABLED'}")
    pass

def _lazy_debug(msg: str):
    try:
        if DECONV_DEBUG:
            print(f"[DECONV][lazy] {msg}")
    except Exception:
        pass

class _LazyPlaceholder:
    """Lightweight stand-in for an ipywidgets widget prior to materialization.
    Provides a ``value`` attribute and a no-op ``observe`` so existing code paths
    can interact without errors. ``value`` may be updated programmatically.
    """
    def __init__(self, value):
        self.value = value
    def observe(self, *_, **__):
        # No-op: real widgets attach callbacks; placeholder ignores.
        return

class _LazyModePlaceholder(_LazyPlaceholder):
    pass

# Global caches for peak widgets so nested helpers and module-level utilities share the
# same references during a deconvolution session.
peak_box_cache = {}
peak_accordion = None


def _ensure_cache_from_accordion(idx: int):
    """Module-level fallback to sync a peak's cache entry from accordion widgets.

    Safe no-op if globals not yet initialized or peak already materialized.
    """
    try:
        peak_box_cache = globals().get('peak_box_cache')
        peak_accordion = globals().get('peak_accordion')
        if not isinstance(peak_box_cache, dict) or peak_accordion is None:
            return
        entry = peak_box_cache.get(idx)
        if entry and entry.get('materialized'):
            return
        for child in getattr(peak_accordion, 'children', []):
            try:
                kids = list(getattr(child, 'children', []))
                if len(kids) < 2:
                    continue
                toggle_local, details_local = kids[:2]
                desc = getattr(toggle_local, 'description', '')
                if isinstance(desc, str) and desc.startswith(f"Peak {idx+1} "):
                    rows = list(getattr(details_local, 'children', []))
                    include_cb = alpha_w = amp_w = center_w = sigma_w = win_w = None
                    amp_mode_w = center_mode_w = sigma_mode_w = None
                    try:
                        if len(rows) >= 5:
                            include_row = rows[0]
                            alpha_row = rows[1]
                            mu_row = rows[2]
                            amplitude_row = rows[3]
                            sigma_row = rows[4]
                            include_cb = list(getattr(include_row, 'children', []))[0]
                            alpha_children = list(getattr(alpha_row, 'children', []))
                            if len(alpha_children) >= 2:
                                alpha_w = alpha_children[1]
                            amp_children = list(getattr(amplitude_row, 'children', []))
                            if len(amp_children) >= 3:
                                amp_mode_w = amp_children[1]
                                amp_w = amp_children[2]
                            mu_children = list(getattr(mu_row, 'children', []))
                            if len(mu_children) >= 6:
                                center_mode_w = mu_children[1]
                                center_w = mu_children[2]
                                win_w = mu_children[5]
                            sigma_children = list(getattr(sigma_row, 'children', []))
                            if len(sigma_children) >= 3:
                                sigma_mode_w = sigma_children[1]
                                sigma_w = sigma_children[2]
                    except Exception:
                        pass
                    # No structured rows? fallback descriptions
                    if not (center_w and sigma_w and amp_w):
                        flat = rows[:]
                        def _find(prefix):
                            for w in flat:
                                try:
                                    d = getattr(w, 'description', '')
                                    if isinstance(d, str) and d.startswith(prefix):
                                        return w
                                except Exception:
                                    continue
                            return None
                        include_cb = include_cb or (entry.get('include') if entry else None)
                        center_w = center_w or _find(f"Center {idx+1}") or (entry.get('center') if entry else None)
                        sigma_w = sigma_w or _find(f"Sigma {idx+1}") or (entry.get('sigma') if entry else None)
                        amp_w = amp_w or _find(f"Amp {idx+1}") or (entry.get('amplitude') if entry else None)
                        alpha_w = alpha_w or _find(f"α {idx+1}") or (entry.get('alpha') if entry else None)
                        win_w = win_w or _find(f"Win {idx+1}") or (entry.get('center_window') if entry else None)
                        amp_mode_w = amp_mode_w or _find(f"A-mode {idx+1}") or (entry.get('amp_mode') if entry else None)
                        center_mode_w = center_mode_w or _find(f"C-mode {idx+1}") or (entry.get('center_mode') if entry else None)
                        sigma_mode_w = sigma_mode_w or _find(f"S-mode {idx+1}") or (entry.get('sigma_mode') if entry else None)
                    if center_w and sigma_w and amp_w:
                        peak_box_cache[idx] = {
                            'box': child,
                            'toggle': toggle_local,
                            'details': details_local,
                            'include': include_cb,
                            'alpha': alpha_w,
                            'center': center_w,
                            'sigma': sigma_w,
                            'amplitude': amp_w,
                            'center_window': win_w,
                            'amp_mode': amp_mode_w,
                            'center_mode': center_mode_w,
                            'sigma_mode': sigma_mode_w,
                            'materialized': True,
                        }
                        try:
                            _debug_log(f"[CACHE_ACCORDION_SYNC] peak={idx+1} materialized_via_scan=module")
                        except Exception:
                            pass
                        # Refresh slider lists if helper exists in current scope
                        try:
                            globals().get('_refresh_slider_lists', lambda: None)()
                        except Exception:
                            pass
                    break
            except Exception:
                continue
    except Exception:
        pass
    
def _convert_dates_iso(directory: str, dry_run: bool = False):
    """Convert date substrings in folder and file names under ``directory`` to ISO
    format (YYYY-MM-DD). Recognizes MM-DD-YYYY (US) patterns and variants with
    separators '-', '_', '.', or space. Already-ISO dates are left unchanged.

    Parameters:
        directory: Root directory to scan.
        dry_run: If True, only print planned changes.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Regex to capture potential date tokens (month/day/year with varied separators)
    date_token = re.compile(r"\b\d{1,2}[-_\. ]\d{1,2}[-_\. ]\d{4}\b")

    def _normalize_token(token: str) -> str:
        # unify separators to '-'
        clean = re.sub(r"[-_\. ]", "-", token)
        parts = clean.split("-")
        if len(parts) != 3:
            return token
        a, b, c = parts
        # If already ISO (year first)
        if len(a) == 4:
            return f"{a}-{b.zfill(2)}-{c.zfill(2)}"  # ensure zero padding
        # Otherwise treat as month-day-year
        if len(c) == 4 and len(a) <= 2 and len(b) <= 2:
            # Basic bounds check
            try:
                m = int(a); d = int(b)
                if not (1 <= m <= 12 and 1 <= d <= 31):
                    return token
            except Exception:
                return token
            return f"{c}-{a.zfill(2)}-{b.zfill(2)}"
        return token

    def _rename_entry(parent: str, name: str) -> str:
        new_name = name
        for match in date_token.findall(name):
            iso = _normalize_token(match)
            if iso != match:
                print(f"Found date '{match}' -> '{iso}' in '{name}'")
                new_name = new_name.replace(match, iso)
        return new_name

    print("Converting date substrings to ISO format..." if not dry_run else "(dry-run) Simulating date conversion...")
    # Rename directories first so file paths remain valid
    for current_root, dirnames, filenames in os.walk(directory):
        # Directories
        for d in list(dirnames):
            new_d = _rename_entry(current_root, d)
            if new_d != d:
                old_path = os.path.join(current_root, d)
                new_path = os.path.join(current_root, new_d)
                if dry_run:
                    print(f"(dry-run) Would rename directory: {old_path} -> {new_path}")
                else:
                    print(f"Renaming directory: {old_path} -> {new_path}")
                    os.rename(old_path, new_path)
                # Update dirnames list for continued traversal
                try:
                    dir_index = dirnames.index(d)
                    dirnames[dir_index] = new_d
                except ValueError:
                    pass
        # Files
        for fname in filenames:
            new_fname = _rename_entry(current_root, fname)
            if new_fname != fname:
                old_fp = os.path.join(current_root, fname)
                new_fp = os.path.join(current_root, new_fname)
                if dry_run:
                    print(f"(dry-run) Would rename file: {old_fp} -> {new_fp}")
                else:
                    print(f"Renaming file: {old_fp} -> {new_fp}")
                    os.rename(old_fp, new_fp)
    print("Date renaming to ISO format complete." if not dry_run else "(dry-run) Date renaming simulation complete.")

def rename_files(
    directory=None,
    replace_spaces=None,
    iso_date_rename=None,
    file_rename=None,
    character_to_use=None,
    pairs_input=None,
    dry_run: bool = False,
):
    """Rename files (and folder/file dates) within a directory.

    Parameters (None prompts interactively):
        directory: root folder to scan.
        replace_spaces: replace spaces in filenames.
        iso_date_rename: convert date substrings to ISO.
        file_rename: perform old:new word replacements.
        character_to_use: replacement for spaces (default prompted when needed).
        pairs_input: comma-separated old:new pairs (prompted if needed).
        dry_run: when True, only print planned changes (no filesystem writes).

    Actions (each optional):
        - Replace spaces in filenames with chosen character.
        - Convert date substrings in folder & file names to ISO (YYYY-MM-DD).
        - Replace specified substrings via old:new pairs.
    """
    # Directory
    if directory is None:
        directory = input("Enter the directory to scan: ").strip()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Scanning directory: {directory}")

    # Replace spaces
    if replace_spaces is None:
        ans = input("Replace spaces in filenames? (y/n): ").strip().lower()
        replace_spaces = ans == "y"
    if replace_spaces:
        if character_to_use is None:
            character_to_use = input("Separator to use (e.g. _): ").strip() or "_"
        print("Replacing spaces in filenames...")
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if " " in fname:
                    old_fp = os.path.join(root, fname)
                    new_fname = fname.replace(" ", character_to_use)
                    new_fp = os.path.join(root, new_fname)
                    if dry_run:
                        print(f"(dry-run) Would rename: {old_fp} -> {new_fp}")
                    else:
                        print(f"Renaming: {old_fp} -> {new_fp}")
                        os.rename(old_fp, new_fp)
        print("Space replacement complete." if not dry_run else "(dry-run) Space replacement simulation complete.")
    else:
        print("Spaces replacement skipped.")

    # Date conversion
    if iso_date_rename is None:
        ans = input("Convert dates in names to ISO (YYYY-MM-DD)? (y/n): ").strip().lower()
        iso_date_rename = ans == "y"
    if iso_date_rename:
        _convert_dates_iso(directory, dry_run=dry_run)
    else:
        print("Date conversion skipped.")

    # Word replacement
    if file_rename is None:
        ans = input("Perform word replacements (old:new)? (y/n): ").strip().lower()
        file_rename = ans == "y"
    if file_rename:
        if pairs_input is None:
            pairs_input = input("Enter old:new pairs (comma-separated): ").strip()
        word_pairs = [p.split(":") for p in pairs_input.split(",") if ":" in p]
        print("Replacing specified substrings in filenames...")
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                new_fname = fname
                for old, new in word_pairs:
                    new_fname = new_fname.replace(old, new)
                if new_fname != fname:
                    old_fp = os.path.join(root, fname)
                    new_fp = os.path.join(root, new_fname)
                    if dry_run:
                        print(f"(dry-run) Would rename: {old_fp} -> {new_fp}")
                    else:
                        print(f"Renaming: {old_fp} -> {new_fp}")
                        os.rename(old_fp, new_fp)
        print("Batch word replacement complete." if not dry_run else "(dry-run) Batch word replacement simulation complete.")
    else:
        print("Word replacement skipped.")


def extract_file_info(
    FTIR_DataFrame,
    file_types=None,
    separators=None,
    material_terms=None,
    conditions_terms=None,
    directory=None,
    append_missing=None,
    access_subdirectories=True,
    track_replicates=False,
):
    """
    Use file info to create or update a structured DataFrame of scan details.

    Main function to gather file information and update the provided FTIR_DataFrame in
    memory.

    Parameters:
    -----------
    FTIR_DataFrame : pd.DataFrame
        The existing DataFrame to append new data to (will be updated in memory).
    file_types : str or None
        Comma-separated string of file extensions to consider (e.g. '.csv,.0,.dpt'). If
        None, prompts user for input.
    separators : str or None
        Comma-separated string of separator characters used in filenames and folder
        names (e.g. '_ , space , -'). If None, prompts user for input.
    material_terms : str or None
        Comma-separated string of material terms to search for in filenames and folder
        names. If None, prompts user for input.
    conditions_terms : str or None
        Comma-separated string of condition terms to search for in filenames and folder
        names. If None, prompts user for input.
    directory : str or None
        The root directory to scan. If None, prompts user for input.
    append_missing : bool or None
        Whether to append rows with missing values. If None, prompts user for input.
    access_subdirectories : bool or None
        If False, only descend into immediate subfolders of 'directory' whose names
        contain a date label (MM-DD-YYYY or YYYY-MM-DD). If None, prompts user for
        input. Default is True.
    track_replicates : bool or None
        Whether to print groups of replicate files. If None, prompts user for input.

    Returns:
    --------
    FTIR_DataFrame : pd.DataFrame
        The updated DataFrame with new file info appended.
    """
    # Guard for undefined / invalid FTIR_DataFrame
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")

    # --- Helper functions (scoped to extract_file_info) --- #
    def _find_term(term, text):
        """
        Find whole word matches of a term in text, case-insensitive.

        Adds spaces around text to catch terms at the start/end, since term-finding uses
        spaces on either side to detect whole words.
        """
        return (
            re.search(rf"(?<!\S){re.escape(term)}(?!\S)", f" {text} ", re.IGNORECASE)
            is not None
        )

    def _gather_file_info(
        FTIR_DataFrame,
        file_types,
        separators,
        material_terms,
        conditions_terms,
        directory,
        append_missing,
        access_subdirectories=True,
        track_replicates=False,
    ):
        """
        Gather file information from a specified root directory and its subdirectories.

        Helps extract_file_info() create a structured DataFrame by extracting details
        from filenames and parent folder names.
        If "ignore" is in the filename, the file will be skipped.
        """
        # Info is first derived from parent folder names, then filenames if not found

        data = []
        grouped_files = {}
        # Consistent spaced ellipsis formatting
        print("Scanning directory for spectral files . . .")
        # Build set of processed files
        # This prevents re-processing files that are already in the DataFrame
        processed_files = set()
        for _, row in FTIR_DataFrame.iterrows():
            processed_files.add((row["File Location"], row["File Name"]))
        # Pre-count candidate files for progress tracking
        total_candidates = 0
        for _file_path, _dirnames, _filenames in os.walk(directory):
            if not access_subdirectories:
                if os.path.normpath(_file_path) == os.path.normpath(directory):
                    date_regex = re.compile(r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})")
                    _dirnames[:] = [d for d in _dirnames if date_regex.search(d)]
            for _filename in _filenames:
                # Apply the same core file filters as the main loop
                if (_file_path, _filename) in processed_files:
                    continue
                if _filename.startswith("."):
                    continue
                if "ignore" in _filename.lower():
                    continue
                if not any(
                    _filename.lower().endswith(file_type.lower())
                    for file_type in file_types
                ):
                    continue
                total_candidates += 1
        print(f"Found {total_candidates} new spectral files to parse . . .")

        parsed_count = 0

        for file_path, dirnames, filenames in os.walk(directory):
            # Optionally restrict traversal to only date-labeled immediate subfolders
            if not access_subdirectories:
                # Only filter when we are at the root directory level
                if os.path.normpath(file_path) == os.path.normpath(directory):
                    date_regex = re.compile(r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})")
                    dirnames[:] = [d for d in dirnames if date_regex.search(d)]
            parent_folder = os.path.basename(file_path)
            for filename in filenames:
                # Skip files already in DataFrame
                if (file_path, filename) in processed_files:
                    continue
                first_column_list = []
                second_column_list = []
                # Skip hidden files, system files, and files with 'ignore' in the name
                if filename.startswith("."):
                    continue
                if "ignore" in filename.lower():
                    continue
                # Skip files that do not match the specified file types
                if not any(
                    filename.lower().endswith(file_type.lower())
                    for file_type in file_types
                ):
                    continue
                # Update progress for each candidate file encountered
                parsed_count += 1
                # Print a single-line progress indicator
                try:
                    pct = (
                        (parsed_count / total_candidates * 100)
                        if total_candidates
                        else 0
                    )
                    print(
                        f"\rParsed {parsed_count}/{total_candidates} files ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                except Exception:
                    pass
                # Read first and second columns from the file, save as list of floats
                full_file_path = os.path.join(file_path, filename)
                try:
                    with open(full_file_path, "r") as data_file:
                        for line in data_file:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    first_column_list.append(float(parts[0]))
                                    second_column_list.append(float(parts[1]))
                                except ValueError:
                                    continue
                except Exception:
                    pass

                # Normalize filename and parent folder by removing file extension and
                # replacing separators with spaces
                # Makes for easier term-finding
                filename_no_ext = filename
                for file_type in file_types:
                    if filename_no_ext.lower().endswith(file_type.lower()):
                        filename_no_ext = filename_no_ext[: -(len(file_type))]
                normalized_filename = filename_no_ext
                normalized_parent_folder = parent_folder
                for sep in separators:
                    normalized_parent_folder = normalized_parent_folder.replace(
                        sep, " "
                    )
                    normalized_filename = normalized_filename.replace(sep, " ")
                # Extract date from parent folder or filename
                # All date formats accepted, as long as they have 2 digits for month and
                # day, and 4 digits for year, separated by hyphens
                date_match = re.search(
                    r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", parent_folder
                )
                date = date_match.group(0) if date_match else None
                if not date:
                    date_match_filename = re.search(
                        r"(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", filename
                    )
                    date = date_match_filename.group(0) if date_match_filename else None
                conditions = next(
                    (
                        term
                        for term in conditions_terms
                        if _find_term(term, normalized_parent_folder)
                    ),
                    None,
                )
                # Extract conditions from parent folder or filename
                if not conditions:
                    conditions = next(
                        (
                            term
                            for term in conditions_terms
                            if _find_term(term, normalized_filename)
                        ),
                        None,
                    )
                material = next(
                    (
                        term
                        for term in material_terms
                        if _find_term(term, normalized_parent_folder)
                    ),
                    None,
                )
                # Extract material from parent folder or filename
                if not material:
                    material = next(
                        (
                            term
                            for term in material_terms
                            if _find_term(term, normalized_filename)
                        ),
                        None,
                    )
                time_match = re.search(r"(\d+)(?:H|hr)", parent_folder, re.IGNORECASE)
                # Extract time(duration) from parent folder or filename
                if time_match:
                    time = int(time_match.group(1))
                else:
                    time_match = re.search(r"(\d+)(?:H|hr)", filename, re.IGNORECASE)
                    time = int(time_match.group(1)) if time_match else None
                # If condition is 'unexposed', set time to 0
                if conditions is not None and conditions.lower() == "unexposed":
                    time = 0
                missing_any = (
                    date is None
                    or conditions is None
                    or material is None
                    or time is None
                )
                # Print a warning if any value is missing
                if missing_any:
                    message = (
                        f"ValueError: Missing value for file '{filename}'. Results: "
                        f"date={date}, conditions={conditions}, material={material}, "
                        f"time={time}"
                    )
                    print(message)

                data_row = {
                    "File Location": file_path,
                    "File Name": filename,
                    "Date": date,
                    "Conditions": conditions,
                    "Material": material,
                    "Time": time,
                    "X-Axis": first_column_list,
                    "Raw Data": second_column_list,
                }
                if append_missing:
                    data.append(data_row)
                else:
                    if not missing_any:
                        data.append(data_row)

        # Group files by (material, conditions, time) after all files are processed

        # Optionally print replicate groups to the console
        if track_replicates:
                # Build replicate groups including BOTH existing rows already present in
                # FTIR_DataFrame and any newly discovered rows in `data`.
                # This allows the replicate reporting to reflect the full dataset state,
                # not just files added in this invocation.
                replicate_groups = {}

                def _add_row_like(mat, cond, t, fname, fpath):
                    group_key = (mat, cond, t)
                    replicate_groups.setdefault(group_key, []).append(
                        (fname, os.path.basename(fpath))
                    )

                # 1. Existing DataFrame rows (if provided)
                try:
                    if FTIR_DataFrame is not None and len(FTIR_DataFrame) > 0:
                        # Only consider rows that have the required grouping columns
                        needed = {"Material", "Conditions", "Time", "File Name", "File Location"}
                        available = set(FTIR_DataFrame.columns)
                        if needed.issubset(available):
                            for _idx, _r in FTIR_DataFrame.iterrows():
                                _add_row_like(
                                    _r.get("Material"),
                                    _r.get("Conditions"),
                                    _r.get("Time"),
                                    _r.get("File Name"),
                                    _r.get("File Location"),
                                )
                except Exception:
                    pass

                # 2. Newly gathered rows (in-memory `data` list of dicts)
                for row in data:
                    _add_row_like(
                        row.get("Material"),
                        row.get("Conditions"),
                        row.get("Time"),
                        row.get("File Name"),
                        row.get("File Location"),
                    )

                # Emit only groups with more than one member (sorted by material, conditions, time)
                print("Replicate groups (>=2 files across existing + new):")
                any_groups = False
                def _group_sort_key(k):
                    m, c, t = k
                    def s(v):
                        return "" if v is None else str(v).lower()
                    def tkey(v):
                        if v is None:
                            return (1, "")
                        try:
                            return (0, float(v))
                        except Exception:
                            return (1, str(v).lower())
                    return (s(m), s(c), tkey(t))
                for group_key, file_list in sorted(replicate_groups.items(), key=lambda item: _group_sort_key(item[0])):
                    if len(file_list) > 1:
                        any_groups = True
                        formatted = [
                            f"{fname} (parent folder: {pfolder})" for fname, pfolder in file_list
                        ]
                        print(f"  {group_key}: {formatted}")
                if not any_groups:
                    print("  (No replicate groups found.)")

        # Finish progress line with newline for clean output
        try:
            print("\nBasic file information extraction complete.")
        except Exception:
            pass

        return data, grouped_files

    # Ensure required columns exist
    required_columns = [
        "File Location",
        "File Name",
        "Date",
        "Quality",
        "Conditions",
        "Material",
        "Time",
        "Sample Humidity",
        "Sample Temperature",
        "X-Axis",
        "Raw Data",
        "Baseline Function",
        "Baseline Parameters",
        "Baseline",
        "Baseline-Corrected Data",
        "Normalization Peak Wavenumber",
        "Normalized and Corrected Data",
        "Peak Wavenumbers",
        "Peak Absorbances",
        "Using Canon Peaks",
        "Deconvolution Results",
        "Deconvolution X-Ranges",
        "Material Fit Results",
    ]
    for column in required_columns:
        if column not in FTIR_DataFrame.columns:
            FTIR_DataFrame[column] = None
    # Ensure 'Quality' column exists and defaults to 'good'; migrate legacy 'quality' if present
    try:
        if (
            "Quality" not in FTIR_DataFrame.columns
            and "quality" in FTIR_DataFrame.columns
        ):
            try:
                FTIR_DataFrame.rename(columns={"quality": "Quality"}, inplace=True)
            except Exception:
                pass
        # If both exist, coalesce values into 'Quality' and drop legacy 'quality'
        if "Quality" in FTIR_DataFrame.columns and "quality" in FTIR_DataFrame.columns:
            try:
                q_legacy = FTIR_DataFrame["quality"].astype("string")
                q_new = FTIR_DataFrame["Quality"].astype("string")
                FTIR_DataFrame["Quality"] = q_new.fillna(q_legacy).fillna("good")
                with contextlib.suppress(Exception):
                    FTIR_DataFrame.drop(columns=["quality"], inplace=True)
            except Exception:
                pass
        if "Quality" not in FTIR_DataFrame.columns:
            FTIR_DataFrame["Quality"] = "good"
        else:
            FTIR_DataFrame["Quality"] = (
                FTIR_DataFrame["Quality"].astype("string").fillna("good")
            )
    except Exception:
        try:
            FTIR_DataFrame["Quality"] = "good"
        except Exception:
            pass

    # Cast columns to correct dtype
    # String columns
    string_cols = [
        "File Location",
        "File Name",
        "Date",
        "Conditions",
        "Material",
        "Baseline Function",
        "Baseline Parameters",
        "Using Canon Peaks",
    ]
    for col in string_cols:
        if col in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = FTIR_DataFrame[col].astype("string")

    # Integer columns
    if "Time" in FTIR_DataFrame.columns:
        FTIR_DataFrame["Time"] = pd.to_numeric(
            FTIR_DataFrame["Time"], errors="coerce"
        ).astype("Int64")
    # Numeric sample environment columns
    for env_col in ("Sample Humidity", "Sample Temperature"):
        if env_col in FTIR_DataFrame.columns:
            try:
                FTIR_DataFrame[env_col] = pd.to_numeric(
                    FTIR_DataFrame[env_col], errors="coerce"
                ).astype("float")
            except Exception:
                pass

    # Dictionary columns
    if "Baseline Parameters" in FTIR_DataFrame.columns:

        def _to_dict(val):
            try:
                if isinstance(val, dict):
                    return val
                if val is None:
                    return None
                # Handle scalar NA safely (avoid array truth-value checks)
                try:
                    import numpy as np  # local import in case top-level not loaded yet
                    if isinstance(val, (float, np.floating)) and np.isnan(val):
                        return None
                    # Only invoke pandas isna for scalars
                    from pandas.api.types import is_scalar

                    if is_scalar(val) and pd.isna(val):
                        return None
                except Exception:
                    pass
                if isinstance(val, str):
                    s = val.strip()
                    if not s:
                        return None
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                return val
            except Exception:
                return val

        FTIR_DataFrame["Baseline Parameters"] = FTIR_DataFrame[
            "Baseline Parameters"
        ].apply(_to_dict)
    if "Deconvolution Results" in FTIR_DataFrame.columns:

        def _to_dict(val):
            try:
                if isinstance(val, dict):
                    return val
                if val is None:
                    return None
                try:
                    import numpy as np
                    if isinstance(val, (float, np.floating)) and np.isnan(val):
                        return None
                    from pandas.api.types import is_scalar

                    if is_scalar(val) and pd.isna(val):
                        return None
                except Exception:
                    pass
                if isinstance(val, str):
                    s = val.strip()
                    if not s:
                        return None
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                return val
            except Exception:
                return val

        FTIR_DataFrame["Deconvolution Results"] = FTIR_DataFrame[
            "Deconvolution Results"
        ].apply(_to_dict)
    if "Material Fit Results" in FTIR_DataFrame.columns:

        def _to_dict(val):
            try:
                if isinstance(val, dict):
                    return val
                if val is None:
                    return None
                try:
                    import numpy as np
                    if isinstance(val, (float, np.floating)) and np.isnan(val):
                        return None
                    from pandas.api.types import is_scalar

                    if is_scalar(val) and pd.isna(val):
                        return None
                except Exception:
                    pass
                if isinstance(val, str):
                    s = val.strip()
                    if not s:
                        return None
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                return val
            except Exception:
                return val

        FTIR_DataFrame["Material Fit Results"] = FTIR_DataFrame[
            "Material Fit Results"
        ].apply(_to_dict)
    # Float columns
    if "Normalization Peak Wavenumber" in FTIR_DataFrame.columns:
        FTIR_DataFrame["Normalization Peak Wavenumber"] = pd.to_numeric(
            FTIR_DataFrame["Normalization Peak Wavenumber"], errors="coerce"
        ).astype("float")

    # Columns that are lists of floats (leave as object, but ensure lists of floats)
    list_float_cols = [
        "X-Axis",
        "Raw Data",
        "Baseline",
        "Baseline-Corrected Data",
        "Normalized and Corrected Data",
        "Peak Wavenumbers",
        "Peak Absorbances",
    ]
    for col in list_float_cols:
        if col in FTIR_DataFrame.columns:

            def to_float_list(val):
                if isinstance(val, list):
                    return [float(x) for x in val]
                elif pd.isnull(val):
                    return val
                try:
                    import ast

                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except Exception:
                    pass

                return val

            FTIR_DataFrame[col] = FTIR_DataFrame[col].apply(to_float_list)

    # Ensure list-like columns are stored with object dtype (per-row lists)
    try:
        existing_list_cols = [c for c in list_float_cols if c in FTIR_DataFrame.columns]
        if existing_list_cols:
            FTIR_DataFrame[existing_list_cols] = FTIR_DataFrame[
                existing_list_cols
            ].astype(object)
    except Exception:
        # Fallback: coerce individually if bulk coercion fails
        for c in list_float_cols:
            if c in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[c] = FTIR_DataFrame[c].astype(object)
                except Exception:
                    pass

    # Option for if DataFrame should append rows with missing values or not
    if append_missing is None:
        message = (
            f"Do you want to append rows with missing values into the DataFrame?"
            f" (y/n): "
        )
        append_missing = input(message).strip().lower()
        append_missing = True if append_missing == "y" else False

    # Option for whether to access non-date-labeled subdirectories
    if access_subdirectories is None:
        message = (
            f"Limit scan to only subfolders with date labels"
            f"(MM-DD-YYYY or YYYY-MM-DD)? (y/n): "
        )
        resp = input(message).strip().lower()
        access_subdirectories = False if resp == "y" else True

    # Get file types
    if file_types is None:
        file_types = input(
            "Enter file types to scan, separated by commas (e.g. .csv,.0,.dpt): "
        ).strip()
    file_types = [ft.strip() for ft in file_types.split(",") if ft.strip()]

    # Get separators
    if separators is None:
        separators = input(
            "Enter separator(s) used in filenames and folder names (e.g. _ or space): "
        ).strip()
    if separators.lower() == "space":
        separators = [" "]
    separators = [sep.strip() for sep in separators.split(",") if sep.strip()]

    # Get material terms
    if material_terms is None:
        material_terms = (
            input(
                "Enter material terms to find, separated by commas (e.g. CPC,PPE,PO): "
            )
            .strip()
            .lower()
        )
    material_terms = [
        term.strip() for term in material_terms.split(",") if term.strip()
    ]

    # Get condition terms
    if conditions_terms is None:
        conditions_terms = (
            input(
                "Enter condition terms to find, separated by commas (e.g. A3,A4,A5): "
            )
            .strip()
            .lower()
        )
    conditions_terms = [
        term.strip() for term in conditions_terms.split(",") if term.strip()
    ]

    if directory is None:
        directory = input("Enter the path to the folder to be scanned: ").strip()
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

    # Gather new file info
    data, grouped_files = _gather_file_info(
        FTIR_DataFrame=FTIR_DataFrame,
        file_types=file_types,
        separators=separators,
        material_terms=material_terms,
        conditions_terms=conditions_terms,
        directory=directory,
        append_missing=append_missing,
        access_subdirectories=access_subdirectories,
        track_replicates=track_replicates,
    )

    # Append new data to FTIR_DataFrame
    if data:
        new_data = pd.DataFrame(data)
        # Ensure 'Quality' exists for new rows; migrate legacy 'quality' if present
        try:
            if "Quality" not in new_data.columns and "quality" in new_data.columns:
                try:
                    new_data.rename(columns={"quality": "Quality"}, inplace=True)
                except Exception:
                    pass
            # If both exist, coalesce and drop legacy 'quality'
            if "Quality" in new_data.columns and "quality" in new_data.columns:
                try:
                    q_legacy = new_data["quality"].astype("string")
                    q_new = new_data["Quality"].astype("string")
                    new_data["Quality"] = q_new.fillna(q_legacy).fillna("good")
                    with contextlib.suppress(Exception):
                        new_data.drop(columns=["quality"], inplace=True)
                except Exception:
                    pass
            if "Quality" not in new_data.columns:
                new_data["Quality"] = "good"
            else:
                new_data["Quality"] = (
                    new_data["Quality"].astype("string").fillna("good")
                )
        except Exception:
            try:
                new_data["Quality"] = "good"
            except Exception:
                pass
        FTIR_DataFrame = pd.concat([FTIR_DataFrame, new_data], ignore_index=True)
        FTIR_DataFrame.drop_duplicates(
            subset=["File Location", "File Name"], inplace=True
        )
        FTIR_DataFrame.reset_index(drop=True, inplace=True)

    # --- Enforce canonical column ordering (always run, even if no new data) ---
    try:
        existing_cols = list(FTIR_DataFrame.columns)
        desired_order = [
            "File Location",
            "File Name",
            "Date",
            "Conditions",
            "Material",
            "Time",
            "Sample Humidity",
            "Sample Temperature",
            "Quality",
            "X-Axis",
            "Raw Data",
            "Baseline Function",
            "Baseline Parameters",
            "Baseline",
            "Baseline-Corrected Data",
            "Normalization Peak Wavenumber",
            "Normalized and Corrected Data",
            "Peak Wavenumbers",
            "Peak Absorbances",
            "Using Canon Peaks",
            "Deconvolution Results",
            "Deconvolution X-Ranges",
            "Material Fit Results",
        ]
        ordered = [c for c in desired_order if c in existing_cols]
        others = [c for c in existing_cols if c not in ordered]
        FTIR_DataFrame = FTIR_DataFrame[ordered + others]
    except Exception:
        pass

    # Replicate tracking after ordering (so columns exist in final DataFrame form).
    if track_replicates and FTIR_DataFrame is not None and len(FTIR_DataFrame) > 0:
        try:
            needed = {"Material", "Conditions", "Time", "File Name", "File Location"}
            if needed.issubset(set(FTIR_DataFrame.columns)):
                replicate_groups_df = {}
                for _idx, _r in FTIR_DataFrame.iterrows():
                    key = (_r.get("Material"), _r.get("Conditions"), _r.get("Time"))
                    replicate_groups_df.setdefault(key, []).append(
                        (
                            _r.get("File Name"),
                            os.path.basename(_r.get("File Location", "")),
                        )
                    )
                print("Replicate groups in DataFrame (>=2 files):")
                any_df = False
                def _group_sort_key2(k):
                    m, c, t = k
                    def s(v):
                        return "" if v is None else str(v).lower()
                    def tkey(v):
                        if v is None:
                            return (1, "")
                        try:
                            return (0, float(v))
                        except Exception:
                            return (1, str(v).lower())
                    return (s(m), s(c), tkey(t))
                for gk, flist in sorted(replicate_groups_df.items(), key=lambda item: _group_sort_key2(item[0])):
                    if len(flist) > 1:
                        any_df = True
                        formatted = [
                            f"{fn} (parent folder: {pf})" for fn, pf in flist
                        ]
                        print(f"  {gk}: {formatted}")
                if not any_df:
                    print("  (No replicate groups found in DataFrame.)")
        except Exception:
            pass

    return FTIR_DataFrame


# ---- Validation helpers for clearer, user-friendly errors ---- #
def _require_columns(df, columns, context="DataFrame"):
    """
    Ensure the DataFrame contains the given columns, else raise a KeyError.

    Helper function.
    """
    if df is None:
        raise ValueError(f"{context} is None. A valid pandas DataFrame is required.")
    try:
        cols = list(df.columns)
    except Exception:
        raise TypeError(
            f"{context} must be a pandas DataFrame with a 'columns' " f"attribute."
        )
    missing = [c for c in columns if c not in cols]
    if missing:
        raise KeyError(
            f"Missing required column(s) in {context}: {missing}. Available columns: "
            f"{cols}"
        )


def _safe_literal_eval(val, value_name="value"):
    """
    Safely parse string representations of Python literals, with descriptive errors.

    Helper function.
    """
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            raise ValueError(
                f"Could not parse {value_name} from string: {val!r}. Error: {e}"
            )
    return val


def _quality_column_name(df):
    try:
        if "Quality" in df.columns:
            return "Quality"
        if "quality" in df.columns:
            try:
                df.rename(columns={"quality": "Quality"}, inplace=True)
            except Exception:
                pass
            return "Quality"
    except Exception:
        pass
    # Default canonical name
    return "Quality"


def _quality_series(df):
    """Return a normalized, lowercase quality Series; defaults to 'good' when missing."""
    try:
        col = _quality_column_name(df)
        if col in df.columns:
            return df[col].astype("string").str.lower().fillna("good")
    except Exception:
        pass
    # Default to all 'good' if column missing
    try:
        return pd.Series(["good"] * len(df), index=df.index, dtype="string")
    except Exception:
        return pd.Series(["good"] * len(df), index=df.index)


def _quality_good_mask(df):
    """Boolean mask where True indicates rows not marked as 'bad'.

    Falls back to all True if the Quality column is missing or parsing fails.
    """
    try:
        qs = _quality_series(df)
        return qs != "bad"
    except Exception:
        return pd.Series([True] * len(df), index=getattr(df, "index", None))


# ------------------------ Session summary helpers ------------------------- #
def _session_summary_lines(changes: dict, *, context: str = ""):
    """Build concise summary lines from a per-session changes dict.

    The changes dict may contain keys like:
      - range_material: list[(material, count_rows, range_str)]
      - normalized_materials: list[(material, updated_count, skipped_count_or_None)]
      - saved_file: list[(idx, count)]
      - saved_filtered: int
      - quality: list[(idx, new_quality)]
      - saved: list[(idx, count_components)] (deconvolution)
      - iter: list[(idx, start_rc, final_rc, changes)] (deconvolution)

    Returns a list of human-friendly strings.
    """
    lines = []
    try:
        rm = changes.get("range_material") or []
        if rm:
            lines.append(
                "Saved normalization range for {} materials: {}".format(
                    len(rm), ", ".join([f"{m} ({c} rows)" for m, c, _ in rm])
                )
            )
    except Exception:
        pass
    try:
        nm = changes.get("normalized_materials") or []
        if nm:
            mats = [m for m, _u, _s in nm]
            lines.append("Normalized materials: " + ", ".join(mats))
    except Exception:
        pass
    # Baseline-correction: emit grouped summary by function with filenames/materials
    try:
        _ctx = str(context).lower().strip()
        if _ctx.startswith("baseline"):  # only for baseline_correct_spectra context
            bcm = changes.get("baseline_corrected_material") or []  # list[(material, function, updated_count)]
            bcf = changes.get("baseline_corrected_file") or []      # list[(idx, function[, filename])]

            # Normalize bcf tuples to (function, filename)
            per_file = []
            try:
                for t in bcf:
                    if len(t) >= 3:
                        _idx, fnc, fname = t[0], str(t[1]), str(t[2])
                    elif len(t) == 2:
                        _idx, fnc = t
                        fname = str(_idx)
                    else:
                        continue
                    per_file.append((fnc.upper(), fname))
            except Exception:
                pass

            # Group by function
            grouped = {}
            total = 0
            # Material-level summaries
            try:
                for mat, fnc, cnt in bcm:
                    fnc_u = str(fnc).upper()
                    grouped.setdefault(fnc_u, {"materials": set(), "files": []})
                    grouped[fnc_u]["materials"].add(str(mat))
                    try:
                        total += int(cnt)
                    except Exception:
                        pass
            except Exception:
                pass
            # File-level summaries
            try:
                for fnc_u, fname in per_file:
                    grouped.setdefault(fnc_u, {"materials": set(), "files": []})
                    grouped[fnc_u]["files"].append(str(fname))
                    total += 1
            except Exception:
                pass

            noun = "spectrum" if total == 1 else "spectra"
            lines.append(f"Baseline-corrected {total} {noun}:")
            # Emit per-function subheadings with items
            for fnc_u in sorted(grouped.keys()):
                lines.append(f"{fnc_u}:")
                items = []
                try:
                    if grouped[fnc_u]["materials"]:
                        for m in sorted(grouped[fnc_u]["materials"]):
                            items.append(f"  - {m}")
                    if grouped[fnc_u]["files"]:
                        for fname in grouped[fnc_u]["files"]:
                            items.append(f"  - {fname}")
                except Exception:
                    pass
                if not items:
                    items.append("  - (none)")
                lines.extend(items)
        else:
            # Non-baseline tools keep their original saved summaries
            try:
                sf = changes.get("saved_file") or []
                if sf:
                    head = ", ".join([f"{i}" for i in sf[:5]])
                    tail = " ..." if len(sf) > 5 else ""
                    lines.append(
                        f"Saved results for {len(sf)} spectra (first 5: {head}{tail})"
                    )
            except Exception:
                pass
            try:
                sfilt = int(changes.get("saved_filtered") or 0)
                if sfilt:
                    lines.append(f"Bulk-saved results for {sfilt} filtered spectra.")
            except Exception:
                pass
    except Exception:
        pass
    try:
        sv = changes.get("saved") or []
        if sv:
            head = ", ".join([f"{i}:{c}" for i, c in sv[:5]])
            tail = " ..." if len(sv) > 5 else ""
            lines.append(
                f"Saved deconvolution for {len(sv)} spectra (first 5: {head}{tail})"
            )
    except Exception:
        pass
    try:
        iters = changes.get("iter") or []
        if iters:
            lines.append(f"Ran iterative correction {len(iters)} time(s).")
    except Exception:
        pass
    try:
        qev = changes.get("quality") or []
        if qev:
            bad = [i for i, v in qev if str(v).lower() == "bad"]
            good = [i for i, v in qev if str(v).lower() == "good"]
            if bad:
                head = bad[:10]
                tail = " ..." if len(bad) > 10 else ""
                lines.append(f"Marked {len(bad)} spectra bad: {head}{tail}")
            if good:
                head = good[:10]
                tail = " ..." if len(good) > 10 else ""
                lines.append(f"Marked {len(good)} spectra good: {head}{tail}")
    except Exception:
        pass
    if not lines:
        lines.append("No DataFrame modifications were made in this session.")
    return lines


def _emit_session_summary(target, lines, *, title: str = "Session Summary"):
    """Emit summary lines into either an Output widget (msg_out) or an HTML widget.

    - target: ipywidgets.Output | ipywidgets.HTML
    - lines: list[str]
    - For Output: prints plain text lines.
    - For HTML: sets monospaced text with escaping.
    """
    try:
        from ipywidgets import Output, HTML
    except Exception:
        Output, HTML = None, None
    # Normalize title capitalization and add underline
    try:
        # Replace leading 'Session summary' (any case) with 'Session Summary'
        if title.lower().startswith("session summary"):
            # Preserve any suffix after the phrase (e.g. '(Normalization)')
            suffix = title[len("Session summary"):]
            title = "Session Summary" + suffix
        elif title.lower().startswith("session summary"):
            title = "Session Summary" + title[len("session summary"):]
    except Exception:
        pass
    underline = "-" * len(title)

    # Output
    try:
        if Output is not None and isinstance(target, Output):
            with target:
                clear_output(wait=True)
                print(title)
                print(underline)
                for line in lines:
                    print(" - " + str(line))
            return
    except Exception:
        pass
    # HTML
    try:
        if HTML is not None and isinstance(target, HTML):
            safe_lines = "\n".join([html.escape(str(l)) for l in lines])
            target.value = (
                "<div style='font-family:monospace; white-space:pre-wrap;'><strong>"
                + html.escape(title)
                + "</strong>\n"
                + html.escape(underline)
                + "\n"
                + safe_lines
                + "</div>"
            )
            return
    except Exception:
        pass
    # Fallback to print
    try:
        print(title)
        print(underline)
        for line in lines:
            print(" - " + str(line))
    except Exception:
        pass


# Alias to match universal helper naming used across tools
def _emit_function_summary(target, lines, *, title: str = "Session Summary"):
    return _emit_session_summary(target, lines, title=title)


# ----------------------- Session selection persistence ----------------------- #
# Persist last-used selections across interactive tools within this module.
_SESSION_SELECTIONS = {"material": "any", "conditions": "any", "time": "any"}
# ^ Persist last-used filter selections across interactive tools so a user's context
#   (material / conditions / time) carries between normalization, peak finding, etc.

# Track active widgets/figures created by baseline_correct_spectra (interactive) to ensure clean re-entry
_TB_WIDGETS = []
# ^ Bookkeeping list of active ipywidgets objects created by baseline_correct_spectra so they can
#   be cleanly closed before rebuilding the UI (prevents stale comm warnings).

# Persist the "Parameter Details" toggle state across baseline_correct_spectra rebuilds
_TB_PARAM_DETAILS_OPEN = False
# ^ Remembers whether the "Parameter Details" toggle was open in baseline_correct_spectra to keep
#   user preference when the parameter UI is dynamically rebuilt.

# Session change trackers for interactive tools
# Used by plot_spectra to collect per-session quality marks across separate plot UIs
_PLOT_SPECTRA_SESSION_CHANGES = None  # type: ignore[var-annotated]
# ^ Per-session change log (dict) for plot_spectra interactive mode. Collects
#   quality mark events so a summary can be rendered on Close.


# ----------------------- Reusable Quality Button Helper ----------------------- #
def _make_quality_controls(df, row_getter, *, margin="10px 10px 0 0", status_out=None):
    """Return mutually exclusive quality buttons ("Mark spectrum as bad" / "Mark spectrum as good").

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a quality column (auto-detected or created).
    row_getter : Callable[[], pd.Series | None]
        Function returning the currently selected DataFrame row or None.
    margin : str, optional
        CSS margin applied to each button for consistent spacing.

    Returns
    -------
    (mark_bad_btn, mark_good_btn, refresh_fn)
        Widgets and a refresh function to sync visibility with current row state.

    Notes
    -----
    - If no row selected, only "Mark spectrum as bad" is shown (default assumption is good).
    - When a spectrum is marked bad, the bad button hides and the good button appears (and vice versa).
    - Quality column name resolved via _quality_column_name; created if missing.
    - Exceptions are swallowed for notebook UI resilience.
    """
    # Ensure sufficient width so full text is visible in all contexts
    _btn_width = "210px"
    mark_bad_btn = widgets.Button(
        description="Mark spectrum as bad",
        button_style="danger",
        layout=widgets.Layout(margin=margin, width=_btn_width),
    )
    mark_good_btn = widgets.Button(
        description="Mark spectrum as good",
        button_style="success",
        layout=widgets.Layout(margin=margin, width=_btn_width),
    )

    def _quality_col():
        try:
            return _quality_column_name(df)
        except Exception:
            return "Quality"

    def refresh():
        try:
            row = row_getter()
        except Exception:
            row = None
        # Re-fetch the latest row from the DataFrame so quality changes made via df.at
        # are visible immediately (selected_row Series objects are stale snapshots).
        try:
            if row is not None:
                row = df.loc[row.name]
        except Exception:
            pass
        # If no row selected, show only bad button
        if row is None:
            try:
                mark_bad_btn.layout.display = ""
                mark_good_btn.layout.display = "none"
            except Exception:
                pass
            return
        qcol = _quality_col()
        try:
            val = str(row.get(qcol, "good")).strip().lower()
        except Exception:
            val = "good"
        is_bad = val == "bad"
        try:
            mark_bad_btn.layout.display = "none" if is_bad else ""
            mark_good_btn.layout.display = "" if is_bad else "none"
        except Exception:
            pass

    def _set_quality(status):
        try:
            row = row_getter()
            if row is None:
                return
            qcol = _quality_col()
            df.at[row.name, qcol] = status
        except Exception:
            pass
        # Optional persistent status output
        try:
            if status_out is not None:
                from IPython.display import display as _ipd  # noqa: F401
                with status_out:
                    print(f"Marked index {getattr(row, 'name', 'row')} as {status}.")
        except Exception:
            pass
        refresh()

    try:
        mark_bad_btn.on_click(lambda _b=None: _set_quality("bad"))
        mark_good_btn.on_click(lambda _b=None: _set_quality("good"))
    except Exception:
        pass

    refresh()
    return mark_bad_btn, mark_good_btn, refresh


# ----------------------- Quality dropdown helper (decoupling) ----------------------- #
def _quality_dropdown_handle(action, *, dropdown, include_bad_flag, idx, label_builder, observer_fn):
    """Remove or reinsert a dropdown option for a spectrum while keeping its plot visible.

    Parameters
    ----------
    action : str
        'bad' or 'good'.
    dropdown : ipywidgets.Dropdown
        Spectrum selection dropdown.
    include_bad_flag : bool
        Checkbox state indicating whether bad spectra are shown.
    idx : Any
        DataFrame index of spectrum.
    label_builder : Callable[[Any], str]
        Builds label string for reinsertion.
    observer_fn : Callable[[dict], None]
        Function registered via dropdown.observe; temporarily detached during mutation.

    Notes
    -----
    - Skips modification when include_bad_flag is True.
    - On 'bad': removes option matching idx if present and clears dropdown value if it was selected.
    - On 'good': reinserts option if missing and selects it.
    - All exceptions swallowed for resilience in interactive notebooks.
    """
    try:
        if dropdown is None or include_bad_flag:
            return
        opts = list(getattr(dropdown, 'options', []))
        if action == 'bad':
            try:
                dropdown.unobserve(observer_fn, names='value')
            except Exception:
                pass
            try:
                opts = [o for o in opts if not (isinstance(o, tuple) and o[1] == idx)]
                dropdown.options = opts
                if getattr(dropdown, 'value', None) == idx:
                    dropdown.value = None if opts else None
            except Exception:
                pass
            try:
                dropdown.observe(observer_fn, names='value')
            except Exception:
                pass
        elif action == 'good':
            ids = [o[1] for o in opts if isinstance(o, tuple)]
            if idx not in ids:
                try:
                    dropdown.unobserve(observer_fn, names='value')
                except Exception:
                    pass
                try:
                    label = label_builder(idx)
                except Exception:
                    label = f"Row {idx}"
                try:
                    # Reinsert option without changing current selection to avoid flicker
                    dropdown.options = opts + [(label, idx)] if opts else [(label, idx)]
                    # Do NOT set dropdown.value here; keep current selection stable
                except Exception:
                    pass
                try:
                    dropdown.observe(observer_fn, names='value')
                except Exception:
                    pass
    except Exception:
        pass


# ----------------------- Common Dataset/Parsing Helpers ----------------------- #
def _conditions_column_name(df):
    """Return the conditions column name if present: 'Conditions' | 'Condition' | None."""
    try:
        if df is None or not hasattr(df, "columns"):
            return None
        if "Conditions" in df.columns:
            return "Conditions"
        if "Condition" in df.columns:
            return "Condition"
    except Exception:
        pass
    return None


def _extract_material_condition_lists(df, *, exclude_unexposed=True):
    """Return (materials, conditions) lists from df with optional 'unexposed' filtering.

    - materials: sorted unique string values of 'Material' if present, else []
    - conditions: sorted unique string values of Conditions/Condition column if present, else []
    - exclude_unexposed: drop any condition equal to 'unexposed' (case-insensitive)
    """
    materials = []
    conditions = []
    if df is None or len(df) == 0:
        return materials, conditions
    try:
        if "Material" in df.columns:
            materials = sorted(
                {str(v) for v in df["Material"].dropna().astype(str).unique().tolist()}
            )
    except Exception:
        pass
    try:
        ccol = _conditions_column_name(df)
        if ccol and ccol in df.columns:
            vals = [str(v) for v in df[ccol].dropna().astype(str).unique().tolist()]
            if exclude_unexposed:
                vals = [v for v in vals if v.strip().lower() != "unexposed"]
            conditions = sorted(vals)
    except Exception:
        pass
    return materials, conditions


def _filter_spectra_dataframe(
    df,
    *,
    material="any",
    condition="any",
    include_bad=True,
    include_unexposed=True,
    normalized_column=None,
):
    """Return a filtered subset of *df* for spectrum dropdowns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing spectra metadata.
    material : str
        Specific material to filter on; use "any" to skip material filtering.
    condition : str
        Specific condition to filter on; use "any" to skip condition filtering.
    include_bad : bool
        When False, exclude rows flagged as bad quality (via _quality_good_mask).
    include_unexposed : bool
        When True and a condition filter is applied, retain "unexposed" rows for the
        selected material (or all materials when material == "any").
    normalized_column : str | None
        Optional column name that must be non-null for rows to be retained (e.g.,
        "Normalized and Corrected Data").
    """

    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        try:
            return df.iloc[0:0]
        except Exception:
            return df

    try:
        result = df.copy()
    except Exception:
        result = df

    if not include_bad:
        try:
            quality_mask = _quality_good_mask(result)
            if isinstance(quality_mask, pd.Series):
                result = result[quality_mask]
        except Exception:
            pass

    material_val = str(material).strip() if material is not None else "any"
    material_lower = material_val.lower()
    if material_lower != "any":
        try:
            mat_series = result.get("Material")
            if mat_series is None:
                return result.iloc[0:0]
            mat_series = mat_series.fillna("").astype(str).str.strip().str.lower()
            result = result[mat_series == material_lower]
        except Exception:
            try:
                return result.iloc[0:0]
            except Exception:
                return df.iloc[0:0]

    cond_val = str(condition).strip() if condition is not None else "any"
    cond_lower = cond_val.lower()
    if cond_lower != "any":
        cond_col = _conditions_column_name(result)
        if cond_col and cond_col in getattr(result, "columns", []):
            try:
                cond_series = result[cond_col].fillna("").astype(str).str.strip().str.lower()
            except Exception:
                cond_series = pd.Series([""] * len(result), index=result.index)
            cond_mask = cond_series == cond_lower
            if include_unexposed and material_lower != "any":
                try:
                    cond_mask = cond_mask | (cond_series == "unexposed")
                except Exception:
                    pass
            result = result[cond_mask]
        else:
            try:
                return result.iloc[0:0]
            except Exception:
                return df.iloc[0:0]

    if normalized_column:
        if normalized_column in getattr(result, "columns", []):
            try:
                result = result[result[normalized_column].notna()]
            except Exception:
                try:
                    result = result[pd.notna(result[normalized_column])]
                except Exception:
                    try:
                        return result.iloc[0:0]
                    except Exception:
                        return df.iloc[0:0]
        else:
            try:
                return result.iloc[0:0]
            except Exception:
                return df.iloc[0:0]

    return result


# ---------------------- Shared Palette Helpers ---------------------- #
def _time_gradient_color(times_unique, val):
    """Return an RGB string color for a given time value using a shared
    blue→purple→red gradient consistent across time-series plotting functions.

    Equal-step mapping: color steps are based on the ordinal position
    in the unique time list (index 0..N-1), not absolute time magnitude,
    so consecutive distinct times are visually separated evenly.

    - times_unique: list of sorted unique time values (numeric preferred)
    - val: time value to map (exact match preferred; nearest used otherwise)
    """
    # Anchor colors (approximate to plot_spectra gradient)
    _blue = (50, 100, 220)
    _purple = (160, 80, 200)
    _red = (220, 60, 60)

    try:
        if not times_unique:
            r, g, b = _blue
            return f"rgb({r},{g},{b})"

        # Normalize times_unique to numeric when possible, preserving order
        tu = []
        for t in times_unique:
            try:
                tu.append(float(t))
            except Exception:
                # fallback: keep as-is (string)
                tu.append(t)

        # Build an index map for exact matches where possible
        index_map = {}
        for i, t in enumerate(tu):
            # Use float key for numeric values; else use string key
            key = t if isinstance(t, (int, float)) else str(t)
            if key not in index_map:
                index_map[key] = i

        # Resolve val to an index: exact match first, else nearest by numeric distance
        idx = 0
        try:
            # attempt numeric compare
            vnum = float(val)
            if vnum in tu:
                idx = tu.index(vnum)
            else:
                # nearest by absolute numeric difference
                diffs = []
                for t in tu:
                    try:
                        diffs.append(abs(vnum - float(t)))
                    except Exception:
                        # non-numeric: treat as large diff
                        diffs.append(float('inf'))
                idx = int(diffs.index(min(diffs))) if diffs else 0
        except Exception:
            # non-numeric val: try exact string match
            sval = str(val)
            if sval in index_map:
                idx = index_map[sval]
            else:
                idx = 0

        n = max(1, len(tu) - 1)
        pos = float(idx) / float(n)

        def _blend(c1, c2, w):
            return (
                int(c1[0] + (c2[0] - c1[0]) * w),
                int(c1[1] + (c2[1] - c1[1]) * w),
                int(c1[2] + (c2[2] - c1[2]) * w),
            )

        if pos <= 0.5:
            r, g, b = _blend(_blue, _purple, pos / 0.5)
        else:
            r, g, b = _blend(_purple, _red, (pos - 0.5) / 0.5)
        return f"rgb({r},{g},{b})"
    except Exception:
        return "rgb(50,100,220)"


def _parse_seq(val):
    """Parse a value into a 1D list/array of floats, or return None if invalid.

    Accepts python-literal strings (e.g., "[1,2,3]") via _safe_literal_eval.
    """
    try:
        v = _safe_literal_eval(val, value_name="sequence")
    except Exception:
        v = val
    try:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == 0:
                return None
            return arr
        return None
    except Exception:
        return None


def _get_session_defaults():
    """Return a shallow copy of the last-used material/conditions/time selections."""
    try:
        return dict(_SESSION_SELECTIONS)
    except Exception:
        return {"material": "any", "conditions": "any", "time": "any"}


def _set_session_selection(material=None, conditions=None, time=None):
    """Update persisted selections; ignores None. Accepts 'any' or concrete values.

    - material, conditions: coerced to str.
    - time: left as-is if numeric; otherwise coerced to str (including 'any').
    """
    try:
        if material is not None:
            _SESSION_SELECTIONS["material"] = str(material)
        if conditions is not None:
            _SESSION_SELECTIONS["conditions"] = str(conditions)
        if time is not None:
            # keep numeric times numeric when possible for easier matching
            try:
                if isinstance(time, str) and time.strip().lower() == "any":
                    _SESSION_SELECTIONS["time"] = "any"
                else:
                    _SESSION_SELECTIONS["time"] = int(time)
            except Exception:
                _SESSION_SELECTIONS["time"] = str(time)
    except Exception:
        # Best-effort; do not raise in UX path
        pass


def _ensure_1d_numeric_array(name, seq):
    """Coerce a sequence into a 1D float numpy array; raise clear error if invalid."""
    seq = _safe_literal_eval(seq, value_name=name)
    try:
        arr = np.asarray(seq, dtype=float)
    except Exception as e:
        raise ValueError(
            f"{name} must be a sequence of numbers. Got: "
            f"{type(seq).__name__}. Error: {e}"
        )
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got array with shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} is empty. A non-empty sequence is required.")
    return arr


def _parse_parameters(parameter_str):
    """
    Parse a parameter string into a dictionary.

    Example input: "lam=100, quantile=0.05"
    Example output: {'lam': 100, 'quantile': 0.05}

    Parameters
    ----------
    param_str : str
        A string containing key=value pairs separated by commas.

    Returns
    -------
    parameter_dictionary : dict
        A dictionary with parameter names as keys and their corresponding values.
    """
    # Converts 'lam=100, quantile=0.05' to a dictionary
    if parameter_str is None:
        return {}
    if not isinstance(parameter_str, str):
        raise TypeError(
            f"parameter_str must be a string of key=value pairs, got "
            f"{type(parameter_str).__name__}."
        )
    tokens = [tok.strip() for tok in parameter_str.split(",") if tok.strip()]
    if not tokens:
        return {}

    def _coerce_scalar(s):
        low = s.strip().lower()
        if low in {"none", "null", "nan"}:
            return None
        if low in {"true", "false"}:
            return low == "true"
        try:
            if any(ch in s for ch in ".eE"):
                return float(s)
            return int(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

    parameter_dictionary = {}
    for item in tokens:
        if "=" not in item:
            raise ValueError(
                f"Invalid parameter token {item!r}. Expected format 'key=value'. Full "
                f"string: {parameter_str!r}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Found empty parameter name in token {item!r}.")
        parameter_dictionary[key] = _coerce_scalar(value)
    return parameter_dictionary


def _get_default_parameters(function_name):
    """
    Input the name of a baseline function and return its default parameters as a
    dictionary.

    Parameters
    ----------
    function_name : str
        The name of the baseline function.

    Returns
    -------
    BASELINE_DEFAULTS.get(function_name.upper(), {}) : dict
        A dictionary of default parameters for the given function.
    """
    BASELINE_DEFAULTS = {
        "ARPLS": {
            "lam": 1e5,
            "diff_order": 2,
            "max_iter": 50,
            "tol": 1e-3,
            "weights": None,
        },
        "IRSQR": {
            "lam": 1e6,
            "quantile": 0.05,
            "num_knots": 100,
            "spline_degree": 3,
            "diff_order": 3,
            "max_iter": 100,
            "tol": 1e-6,
            "weights": None,
            "eps": None,
        },
        "FABC": {
            "lam": 1e6,
            "scale": None,
            "num_std": 3.0,
            "diff_order": 2,
            "min_length": 2,
            "weights": None,
            "weights_as_mask": False,
            "pad_kwargs": None,
        },
        "MANUAL": {},
    }
    return BASELINE_DEFAULTS.get(function_name.upper(), {})


def _cast_parameter_types(function_name, parameters):
    """
    Cast parameter types for each function based on known parameter types.

    Parameters
    ----------
    function_name : str
        The name of the baseline function.
    parameters : dict
        A dictionary of parameters to cast.

    Returns
    -------
    parameters : dict
        The dictionary with casted parameter types.
    """
    function = function_name.upper()
    if function == "ARPLS":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
    elif function == "IRSQR":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if "quantile" in parameters:
            parameters["quantile"] = float(parameters["quantile"])
        if "num_knots" in parameters:
            parameters["num_knots"] = int(parameters["num_knots"])
        if "spline_degree" in parameters:
            parameters["spline_degree"] = int(parameters["spline_degree"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "max_iter" in parameters:
            parameters["max_iter"] = int(parameters["max_iter"])
        if "tol" in parameters:
            parameters["tol"] = float(parameters["tol"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "eps" in parameters:
            if str(parameters["eps"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["eps"] = float(parameters["eps"])
                except Exception:
                    pass
    elif function == "FABC":
        if "lam" in parameters:
            parameters["lam"] = float(parameters["lam"])
        if str(parameters["scale"]).lower() not in ["none", "null", ""]:
            try:
                parameters["scale"] = ast.literal_eval(parameters["scale"])
            except Exception:
                pass
        if "num_std" in parameters:
            parameters["num_std"] = float(parameters["num_std"])
        if "diff_order" in parameters:
            parameters["diff_order"] = int(parameters["diff_order"])
        if "min_length" in parameters:
            parameters["min_length"] = int(parameters["min_length"])
        if "weights" in parameters:
            if str(parameters["weights"]).lower() not in ["none", "null", ""]:
                try:
                    parameters["weights"] = ast.literal_eval(parameters["weights"])
                except Exception:
                    pass
        if "weights_as_mask" in parameters:
            if str(parameters["weights_as_mask"]).lower() in ["true"]:
                parameters["weights_as_mask"] = True
            else:
                parameters["weights_as_mask"] = False
        if "pad_kwargs" in parameters:
            if parameters["pad_kwargs"] is not None:
                try:
                    parameters["pad_kwargs"] = ast.literal_eval(
                        parameters["pad_kwargs"]
                    )
                except Exception:
                    pass
    return parameters


def _processing_column_errors(df, want_baseline, want_baseline_corrected, want_normalized):
    """Return list of user-facing error messages only when there is ZERO usable data.

    Previous logic required all rows to be populated (flagged errors if ANY NaN / None
    existed). This prevented plotting partially processed DataFrames. Now we instead
    emit an error only if there are no valid (non-empty, non-NaN) entries in the
    requested column.
    """
    errors = []

    checks = [
        (want_baseline, "Baseline", "You need to baseline-correct the spectra before this will be available for plotting."),
        (want_baseline_corrected, "Baseline-Corrected Data", "You need to baseline-correct the spectra before this will be available for plotting."),
        (want_normalized, "Normalized and Corrected Data", "You need to normalize the spectra before this will be available for plotting."),
    ]

    for flag, col_name, msg in checks:
        if not flag:
            continue  # Column not requested for plotting
        if col_name not in df.columns:
            errors.append(msg)
            continue
        col = df[col_name]
        if not isinstance(col, pd.Series):
            errors.append(msg)
            continue

        usable = 0
        try:
            for v in col:
                if v is None:
                    continue
                # Accept list/tuple/ndarray with at least one non-NaN value
                if isinstance(v, (list, tuple)):
                    if len(v) == 0:
                        continue
                    if any(pd.isna(x) for x in v):
                        # Allow partially valid lists as long as at least one value is not NaN
                        if all(pd.isna(x) for x in v):
                            continue
                    usable += 1
                    continue
                try:
                    import numpy as np  # local import to avoid issues if numpy missing earlier
                    if isinstance(v, np.ndarray):
                        if v.size == 0:
                            continue
                        if np.isnan(v).all():
                            continue
                        usable += 1
                        continue
                except Exception:
                    pass
                # Scalar numeric
                if isinstance(v, (int, float)) and not pd.isna(v):
                    usable += 1
        except Exception:
            # On unexpected failure treat as no usable data
            usable = 0

        if usable == 0:
            errors.append(msg)

    return errors


def plot_spectra(
    FTIR_DataFrame,
    materials=None,
    conditions=None,
    times=None,
    raw_data=True,
    baseline=False,
    baseline_corrected=False,
    normalized=False,
    deconv_fit=False,
    downsample=False,
    separate_plots=False,
    include_replicates=True,
    mark_bad=None,
    mark_good=None,
    show_bad=False,
    interactive=True,
):
    """
    Plot spectra grouped by material, condition, and time.

    Accepts lists or 'any' for each category.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    material : str, list, or 'any'
        The material(s) to filter by, or 'any' to include all.
    condition : str, list, or 'any'
        The condition(s) to filter by, or 'any' to include all.
    time : str, int, list, or 'any'
            try:
                filtered = filtered[_quality_good_mask(filtered)]
            except Exception:
                pass
            if filtered.empty:
    raw_data : bool, optional
        Whether to plot the raw data (default is True).
    baseline : bool, optional
        Whether to plot the baseline (default is False).
            try:
                filtered = filtered[_quality_good_mask(filtered)]
            except Exception:
                pass
    baseline_corrected : bool, optional
        Whether to plot the baseline-corrected data (default is False).
    normalized : bool, optional
        Whether to plot the normalized-and-corrected data from the column
        'Normalized and Corrected Data' (default is False).
    separate_plots : bool, optional
        Whether to create separate plots for each spectrum (default is False).
    deconv_fit : bool, optional
        When True, overlay the stored deconvolution fit reconstructed from
        'Deconvolution Results' (default False).
    downsample : bool, optional
        When True, adaptively decimate x/y to reduce points per trace (default False).
    include_replicates : bool, optional
        Whether to include all replicates or just the first of each group (default is
        True).

    Returns
    -------
    None

    mark_bad (optional)
        If provided, mark rows as bad quality:
        - 'all': mark all plotted rows as bad
        - list of DataFrame indices
        - list of file names (matches 'File Name')
    mark_good (optional)
        If provided, mark rows as good quality (same accepted value forms as mark_bad).
    show_bad : bool, optional
        When True, include spectra marked as bad in the plots; when False (default),
        bad spectra are excluded.
    """
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    # Interactive widget UI
    # - When interactive=True (default) or no filters are provided, build an ipywidgets
    #   control panel for filtering (Material/Conditions/Time), toggling which traces
    #   to show, and choosing separate vs grouped plots.
    # - This branch also tracks per-session quality changes and shows a Close summary.
    if interactive or (materials is None and conditions is None and times is None):
        try:
            # Initialize per-session change log (quality marks only for this UI)
            global _PLOT_SPECTRA_SESSION_CHANGES
            # Initialize or reuse per-session quality change log for interactive plotting
            if _PLOT_SPECTRA_SESSION_CHANGES is None:
                _PLOT_SPECTRA_SESSION_CHANGES = {"quality": []}
            else:
                # Reuse existing dict but don't wipe previous events until summary emitted on Close
                _PLOT_SPECTRA_SESSION_CHANGES.setdefault("quality", [])
            # Build options from DataFrame
            try:
                materials_opts = sorted(
                    set(
                        str(x)
                        for x in FTIR_DataFrame.get("Material", []).dropna().astype(str)
                    )
                )
            except Exception:
                materials_opts = []
            try:
                conditions_col = (
                    "Conditions"
                    if "Conditions" in FTIR_DataFrame.columns
                    else (
                        "Condition" if "Condition" in FTIR_DataFrame.columns else None
                    )
                )
                if conditions_col is not None:
                    conditions_opts = sorted(
                        set(
                            str(x)
                            for x in FTIR_DataFrame[conditions_col].dropna().astype(str)
                        )
                    )
                else:
                    conditions_opts = []
            except Exception:
                conditions_opts = []
            try:
                times_raw = FTIR_DataFrame.get("Time", [])
                times_set = set()
                seq = list(
                    getattr(times_raw, "dropna", lambda: [])().tolist()
                    if hasattr(times_raw, "dropna")
                    else times_raw
                )
                for t in seq:
                    try:
                        times_set.add(int(t))
                    except Exception:
                        try:
                            times_set.add(str(t))
                        except Exception:
                            pass
                times_opts = sorted(times_set, key=lambda v: (isinstance(v, str), v))
            except Exception:
                times_opts = []

            # Widgets
            materials_dd = widgets.Dropdown(
                options=["any"] + materials_opts,
                value="any",
                description="Material",
                layout=widgets.Layout(width="40%"),
            )

            conditions_dd = widgets.Dropdown(
                options=["any"] + conditions_opts,
                value="any",
                description="Conditions",
                layout=widgets.Layout(width="40%"),
            )

            # For Time, allow 'any' or a single specific time
            # Keep underlying values numeric when possible
            time_options = [("any", "any")] + [(str(v), v) for v in times_opts]
            times_dd = widgets.Dropdown(
                options=time_options,
                value="any",
                description="Time",
                layout=widgets.Layout(width="30%"),
            )

            # Apply persisted defaults if available
            try:
                _sess = _get_session_defaults()
                if _sess.get("material") in materials_dd.options:
                    materials_dd.value = _sess.get("material")
                if _sess.get("conditions") in conditions_dd.options:
                    conditions_dd.value = _sess.get("conditions")
                saved_time = _sess.get("time", "any")
                time_values = [v for (_lab, v) in time_options]
                if saved_time in time_values:
                    times_dd.value = saved_time
            except Exception:
                pass

            # Compact multi-select checkboxes for trace types
            raw_cb = widgets.Checkbox(
                value=True if materials is None else bool(raw_data),
                description="Raw",
                layout=widgets.Layout(width="auto"),
            )
            base_cb = widgets.Checkbox(
                value=False if materials is None else bool(baseline),
                description="Baseline",
                layout=widgets.Layout(width="auto"),
            )
            blc_cb = widgets.Checkbox(
                value=False if materials is None else bool(baseline_corrected),
                description="Baseline-corrected",
                layout=widgets.Layout(width="auto"),
            )
            norm_cb = widgets.Checkbox(
                value=False if materials is None else bool(normalized),
                description="Normalized",
                layout=widgets.Layout(width="auto"),
            )
            deconv_cb = widgets.Checkbox(
                value=False if materials is None else bool(deconv_fit),
                description="Deconvolution fit",
                layout=widgets.Layout(width="auto"),
            )
            downsample_cb = widgets.Checkbox(
                value=False if materials is None else bool(downsample),
                description="Downsample spectra",
                layout=widgets.Layout(width="auto"),
            )
            # Traces as a column (left block) with a subtle frame
            traces_col = widgets.VBox(
                [
                    widgets.HTML(value="<b>Traces</b>"),
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                    deconv_cb,
                ],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    padding="8px",
                    margin="0 10px 0 0",
                ),
            )
            separate_plots_chk = widgets.Checkbox(
                value=True if materials is None else bool(separate_plots),
                description="Separate plots",
            )
            include_replicates_chk = widgets.Checkbox(
                value=True if materials is None else bool(include_replicates),
                description="Include replicates",
            )
            show_bad_chk = widgets.Checkbox(
                value=False if materials is None else bool(show_bad),
                description="Include bad spectra",
            )

            # Actions & summary area
            plot_button = widgets.Button(description="Plot", button_style="primary")
            close_button = widgets.Button(description="Close", button_style="danger")
            out = widgets.Output()  # persists session summary on Close

            def _parse_mark(text):
                s = (text or "").strip()
                if not s:
                    return None
                if s.lower() == "all":
                    return "all"
                parts = [p.strip() for p in s.split(",") if p.strip()]
                parsed = []
                for p in parts:
                    try:
                        parsed.append(int(p))
                    except Exception:
                        parsed.append(p)
                return parsed if parsed else None

            def _materials_value():
                v = materials_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _conditions_value():
                v = conditions_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _times_value():
                v = times_dd.value
                return str(v) if v not in (None, "", "any") else "any"

            def _on_plot(_b):
                with out:
                    clear_output(wait=True)
                    try:
                        # Map trace checkboxes to boolean flags
                        tr_raw = bool(raw_cb.value)
                        tr_base = bool(base_cb.value)
                        tr_blc = bool(blc_cb.value)
                        tr_norm = bool(norm_cb.value)
                        tr_deconv = bool(deconv_cb.value)
                        tr_down = bool(downsample_cb.value)
                        # --- Validate requested trace types within the CURRENT FILTERED SUBSET ---
                        def _valid_mask_for_col(df, col):
                            if df is None or len(df) == 0:
                                return pd.Series([False] * (0 if df is None else len(df)))
                            if col not in df.columns:
                                return pd.Series([False] * len(df), index=df.index)
                            vals = df[col]
                            valid = []
                            for v in vals:
                                # Try to parse sequences robustly (handles stringified lists, etc.)
                                seq = None
                                try:
                                    seq = _parse_seq(v)
                                except Exception:
                                    seq = None
                                if seq is None:
                                    # Fallback for already-sequence values
                                    try:
                                        import numpy as np
                                        if isinstance(v, np.ndarray):
                                            seq = v
                                        elif isinstance(v, (list, tuple)):
                                            seq = v
                                    except Exception:
                                        pass
                                if seq is not None:
                                    try:
                                        import numpy as np
                                        arr = np.asarray(seq, dtype=float).ravel()
                                        valid.append(bool(arr.size > 0 and not np.isnan(arr).all()))
                                        continue
                                    except Exception:
                                        try:
                                            ok = any((x is not None) and not pd.isna(x) for x in seq)
                                            valid.append(bool(ok))
                                            continue
                                        except Exception:
                                            valid.append(False)
                                            continue
                                # Scalar numeric fallback
                                if isinstance(v, (int, float)) and not pd.isna(v):
                                    valid.append(True)
                                else:
                                    valid.append(False)
                            return pd.Series(valid, index=vals.index)

                        def _valid_mask_for_deconv(df):
                            # Specialized validation for 'Deconvolution Results': expect a non-empty list
                            if df is None or len(df) == 0:
                                return pd.Series([False] * (0 if df is None else len(df)))
                            if "Deconvolution Results" not in df.columns:
                                return pd.Series([False] * len(df), index=df.index)
                            vals = df["Deconvolution Results"]
                            valid = []
                            for v in vals:
                                obj = v
                                if isinstance(obj, str):
                                    try:
                                        obj = ast.literal_eval(obj)
                                    except Exception:
                                        obj = None
                                ok = isinstance(obj, list) and len(obj) > 0
                                valid.append(bool(ok))
                            return pd.Series(valid, index=vals.index)

                        # Build filtered subset for validation using current UI selections
                        try:
                            m_val = _materials_value()
                            c_val = _conditions_value()
                            t_val = _times_value()
                            dfv = FTIR_DataFrame
                            mask_val = pd.Series([True] * len(dfv))
                            # Respect quality filter
                            try:
                                if not bool(show_bad_chk.value):
                                    mask_val &= _quality_good_mask(dfv).values
                            except Exception:
                                pass
                            if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                try:
                                    mask_val &= dfv["Material"].astype(str).isin(mats_list)
                                except Exception:
                                    mask_val &= dfv.get("Material", pd.Series([])).astype(str).isin(mats_list)
                            if isinstance(c_val, str) and c_val.strip().lower() != "any":
                                cond_list = [s.strip() for s in c_val.split(",") if s.strip()]
                                try:
                                    cond_mask = dfv["Conditions"].astype(str).isin(cond_list)
                                except Exception:
                                    cond_mask = dfv.get("Conditions", pd.Series([])).astype(str).isin(cond_list)
                                # If Time == 'any', include 'unexposed' across conditions
                                if isinstance(t_val, str) and t_val.strip().lower() == "any":
                                    try:
                                        cond_series = dfv["Conditions"].astype(str).str.lower()
                                        cond_mask = cond_mask | (cond_series == "unexposed")
                                    except Exception:
                                        cond_mask = cond_mask | (dfv.get("Conditions", pd.Series([])) == "unexposed")
                                mask_val &= cond_mask
                            if isinstance(t_val, str) and t_val.strip().lower() != "any":
                                t_list = []
                                for t in t_val.split(","):
                                    ts = t.strip()
                                    if not ts:
                                        continue
                                    try:
                                        t_list.append(int(ts))
                                    except Exception:
                                        t_list.append(ts)
                                try:
                                    mask_val &= dfv["Time"].isin(t_list)
                                except Exception:
                                    mask_val &= dfv.get("Time", pd.Series([])).isin(t_list)
                                # If time selection includes 0, also include 'unexposed' spectra for the selected material(s)
                                try:
                                    if any((isinstance(x, int) and x == 0) or (isinstance(x, str) and x.strip() == "0") for x in t_list):
                                        try:
                                            unexp_series = dfv["Conditions"].astype(str).str.lower()
                                            unexp_mask = unexp_series == "unexposed"
                                        except Exception:
                                            unexp_mask = dfv.get("Conditions", pd.Series([])).astype(str).str.lower() == "unexposed"
                                        # Constrain to selected materials if not 'any'
                                        if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                            mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                            try:
                                                unexp_mask &= dfv["Material"].astype(str).isin(mats_list)
                                            except Exception:
                                                unexp_mask &= dfv.get("Material", pd.Series([])).astype(str).isin(mats_list)
                                        # OR in the unexposed spectra
                                        try:
                                            mask_val |= unexp_mask
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            filtered_val = dfv[mask_val]
                        except Exception:
                            filtered_val = FTIR_DataFrame

                        validation_errors = []
                        try:
                            if tr_base and not _valid_mask_for_col(filtered_val, "Baseline").any():
                                validation_errors.append(
                                    "You need to baseline-correct the spectra before this will be available for plotting."
                                )
                            if tr_blc and not _valid_mask_for_col(filtered_val, "Baseline-Corrected Data").any():
                                validation_errors.append(
                                    "You need to baseline-correct the spectra before this will be available for plotting."
                                )
                            if tr_norm and not _valid_mask_for_col(filtered_val, "Normalized and Corrected Data").any():
                                validation_errors.append(
                                    "You need to normalize the spectra before this will be available for plotting."
                                )
                            if tr_deconv and not _valid_mask_for_deconv(filtered_val).any():
                                validation_errors.append(
                                    "You need to deconvolute peaks before the deconvolution fit will be available for plotting."
                                )
                        except Exception:
                            validation_errors = []  # fail open on unexpected issues

                        if validation_errors:
                            for msg in validation_errors:
                                try:
                                    display(widgets.HTML(value=f"<b>Error:</b> {msg}"))
                                except Exception:
                                    print(f"Error: {msg}")
                            return
                        # Prepare a plotting function we can call after any warnings
                        def _do_plot():
                            try:
                                plot_spectra(
                                    FTIR_DataFrame=FTIR_DataFrame,
                                    materials=_materials_value(),
                                    conditions=_conditions_value(),
                                    times=_times_value(),
                                    raw_data=tr_raw,
                                    baseline=tr_base,
                                    baseline_corrected=tr_blc,
                                    normalized=tr_norm,
                                    deconv_fit=tr_deconv,
                                    downsample=tr_down,
                                    separate_plots=separate_plots_chk.value,
                                    include_replicates=include_replicates_chk.value,
                                    show_bad=show_bad_chk.value,
                                    interactive=False,
                                    # Gradient colors applied automatically based on time.
                                )
                            except Exception as e:
                                with out:
                                    print(f"Error while plotting: {e}")

                        # Estimate how many spectra will be plotted with current filters
                        try:
                            m_val = _materials_value()
                            c_val = _conditions_value()
                            t_val = _times_value()
                            df = FTIR_DataFrame
                            mask_est = pd.Series([True] * len(df))
                            # Respect quality filter
                            try:
                                if not bool(show_bad_chk.value):
                                    mask_est &= _quality_good_mask(df).values
                            except Exception:
                                pass
                            if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                try:
                                    mask_est &= df["Material"].astype(str).isin(mats_list)
                                except Exception:
                                    mask_est &= df.get("Material", pd.Series([])).astype(str).isin(mats_list)
                            if isinstance(c_val, str) and c_val.strip().lower() != "any":
                                cond_list = [s.strip() for s in c_val.split(",") if s.strip()]
                                try:
                                    cond_mask = df["Conditions"].astype(str).isin(cond_list)
                                except Exception:
                                    cond_mask = df.get("Conditions", pd.Series([])).astype(str).isin(cond_list)
                                # If Time == 'any', include 'unexposed' across conditions
                                if isinstance(t_val, str) and t_val.strip().lower() == "any":
                                    try:
                                        cond_series = df["Conditions"].astype(str).str.lower()
                                        cond_mask = cond_mask | (cond_series == "unexposed")
                                    except Exception:
                                        cond_mask = cond_mask | (df.get("Conditions", pd.Series([])) == "unexposed")
                                mask_est &= cond_mask
                            if isinstance(t_val, str) and t_val.strip().lower() != "any":
                                t_list = []
                                for t in t_val.split(","):
                                    ts = t.strip()
                                    if not ts:
                                        continue
                                    try:
                                        t_list.append(int(ts))
                                    except Exception:
                                        t_list.append(ts)
                                try:
                                    mask_est &= df["Time"].isin(t_list)
                                except Exception:
                                    mask_est &= df.get("Time", pd.Series([])).isin(t_list)
                                # If time selection includes 0, also include 'unexposed' spectra for the selected material(s)
                                try:
                                    if any((isinstance(x, int) and x == 0) or (isinstance(x, str) and x.strip() == "0") for x in t_list):
                                        try:
                                            unexp_series = df["Conditions"].astype(str).str.lower()
                                            unexp_mask = unexp_series == "unexposed"
                                        except Exception:
                                            unexp_mask = df.get("Conditions", pd.Series([])).astype(str).str.lower() == "unexposed"
                                        if isinstance(m_val, str) and m_val.strip().lower() != "any":
                                            mats_list = [s.strip() for s in m_val.split(",") if s.strip()]
                                            try:
                                                unexp_mask &= df["Material"].astype(str).isin(mats_list)
                                            except Exception:
                                                unexp_mask &= df.get("Material", pd.Series([])).astype(str).isin(mats_list)
                                        try:
                                            mask_est |= unexp_mask
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            filtered_est = df[mask_est]
                            if not bool(include_replicates_chk.value) and not filtered_est.empty:
                                try:
                                    filtered_est = filtered_est.sort_values(by=["Material", "Conditions", "Time"]).drop_duplicates(
                                        subset=["Material", "Conditions", "Time"], keep="first"
                                    )
                                except Exception:
                                    pass
                            match_count = int(len(filtered_est))
                        except Exception:
                            match_count = 0

                        # Choose thresholds (lower in Colab)
                        warn_threshold = 40 if _IN_COLAB else 75

                        if match_count > warn_threshold:
                            # Build a warning prompt with proceed/cancel options
                            try:
                                # Disable original Plot button (grey out) while warning is active
                                try:
                                    plot_button.disabled = True
                                    plot_button.button_style = ''  # remove primary styling
                                    plot_button.description = 'Plot (warning active)'
                                except Exception:
                                    pass
                                msg = (
                                    f"This selection will plot {match_count} spectra"
                                    + (" as separate figures" if bool(separate_plots_chk.value) else "")
                                    + ". This may be slow or cause a crash."
                                )
                                tips = (
                                    "Tips: narrow filters, disable replicates, disable separate plots, or enable the downsample option."
                                )
                                display(
                                    widgets.VBox(
                                        [
                                            widgets.HTML(
                                                value=f"<b>Warning:</b> {msg}<br><i>{tips}</i>"
                                            ),
                                            widgets.HBox(
                                                [
                                                    widgets.Button(
                                                        description="Plot Anyways",
                                                        button_style="primary",
                                                        layout=widgets.Layout(margin="5px 10px 5px 0"),
                                                    ),
                                                    widgets.Button(
                                                        description="Redo Selection",
                                                        button_style="warning",
                                                        layout=widgets.Layout(margin="5px 0 5px 0"),
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                )

                                # Wire up buttons after creation
                                # Need references; rebuild HBox to capture them
                                proceed_btn = widgets.Button(
                                    description="Plot Anyways", button_style="primary"
                                )
                                redo_btn = widgets.Button(
                                    description="Redo Selection", button_style="warning"
                                )
                                # Re-render with wired instances and replace previous prompt
                                clear_output(wait=True)
                                display(
                                    widgets.VBox(
                                        [
                                            widgets.HTML(
                                                value=f"<b>Warning:</b> {msg}<br><i>{tips}</i>"
                                            ),
                                            widgets.HBox([proceed_btn, redo_btn]),
                                        ]
                                    )
                                )

                                def _proceed(_b=None):
                                    with out:
                                        clear_output(wait=True)
                                        # Re-enable Plot button
                                        try:
                                            plot_button.disabled = False
                                            plot_button.button_style = 'primary'
                                            plot_button.description = 'Plot'
                                        except Exception:
                                            pass
                                        _do_plot()

                                def _redo(_b=None):
                                    with out:
                                        clear_output(wait=True)
                                        # Re-enable Plot button so user can adjust and plot again
                                        try:
                                            plot_button.disabled = False
                                            plot_button.button_style = 'primary'
                                            plot_button.description = 'Plot'
                                        except Exception:
                                            pass
                                        display(
                                            widgets.HTML(
                                                value="Selection not plotted. Adjust filters above and click Plot."
                                            )
                                        )

                                proceed_btn.on_click(_proceed)
                                redo_btn.on_click(_redo)
                                return
                            except Exception:
                                # If warning UI fails, fall back to plotting directly
                                pass

                        # If below threshold (or warning failed), plot directly
                        _do_plot()
                        # After plotting separate plots, info: individual mark buttons
                        # append to the global session dict in the non-interactive branch below.
                        try:
                            if separate_plots_chk.value:
                                pass
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Error while plotting: {e}")

            plot_button.on_click(_on_plot)

            # Persist selections on Plot and on changes
            def _persist(_=None):
                try:
                    _set_session_selection(
                        material=materials_dd.value,
                        conditions=conditions_dd.value,
                        time=times_dd.value,
                    )
                except Exception:
                    pass

            plot_button.on_click(lambda _b: _persist())
            materials_dd.observe(lambda ch: _persist(), names="value")
            conditions_dd.observe(lambda ch: _persist(), names="value")
            times_dd.observe(lambda ch: _persist(), names="value")

            def _on_close(_b):
                """Close the interactive UI and clear outputs, similar to other modules."""
                try:
                    # Emit session summary before closing if any quality changes occurred
                    global _PLOT_SPECTRA_SESSION_CHANGES
                    if isinstance(_PLOT_SPECTRA_SESSION_CHANGES, dict):
                        lines = _session_summary_lines(
                            _PLOT_SPECTRA_SESSION_CHANGES,
                            context="plot_spectra",
                        )
                        # Use the existing output widget if still present; else fallback to print.
                        try:
                            _emit_session_summary(out, lines, title="Session Summary (Plot Spectra)")
                        except Exception:
                            print("plot_spectra session summary:")
                            for l in lines:
                                print(" - " + str(l))
                        # Reset tracker for next session
                        _PLOT_SPECTRA_SESSION_CHANGES = {"quality": []}
                except Exception:
                    pass
                try:
                    # Clear the visible area first
                    clear_output(wait=True)
                except Exception:
                    pass
                # Attempt to close widgets to free comms and avoid duplicate UIDs
                for w in [
                    # Individual selector widgets
                    materials_dd,
                    conditions_dd,
                    times_dd,
                    # Trace checkboxes and their column container
                    raw_cb,
                    base_cb,
                    blc_cb,
                    norm_cb,
                    downsample_cb,
                    traces_col,
                    options_col,
                    # Toggle checkboxes and their container rows
                    separate_plots_chk,
                    include_replicates_chk,
                    show_bad_chk,
                    # Top-level layout containers
                    None,  # placeholder if we later add more dynamic containers
                    # Action buttons & output
                    plot_button,
                    close_button,
                    out,
                ] + [
                    # Higher-level composite containers (selectors/toggles/controls)
                    "selectors_placeholder",  # will be swapped below if defined
                    "toggles_placeholder",
                    "controls_placeholder",
                ]:
                    # Replace placeholder strings with actual widget objects if they exist
                    if w == "selectors_placeholder":
                        w = selectors
                    elif w == "toggles_placeholder":
                        w = toggles
                    elif w == "controls_placeholder":
                        w = controls
                    if w is None:
                        continue
                    try:
                        # Keep 'out' visible to preserve the session summary
                        if w is out:
                            continue
                        w.close()
                    except Exception:
                        pass

            close_button.on_click(_on_close)

            # Layout and display the controls and the summary output area (out)
            selectors = widgets.HBox([materials_dd, conditions_dd, times_dd])
            # Options as a column (right block) with a subtle frame
            options_col = widgets.VBox(
                [
                    widgets.HTML(value="<b>Options</b>"),
                    separate_plots_chk,
                    include_replicates_chk,
                    show_bad_chk,
                    downsample_cb,
                ],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    padding="8px",
                ),
            )
            toggles = widgets.HBox([traces_col, options_col])
            controls = widgets.VBox(
                [
                    selectors,
                    toggles,
                    widgets.HBox([plot_button, close_button]),
                ]
            )
            display(controls, out)
            return
        except Exception as e:
            # If widgets are unavailable or something fails, fall back to non-interactive path with a note
            try:
                print(
                    f"Interactive controls unavailable, falling back to static plot: {e}"
                )
            except Exception:
                pass

    # Non-interactive path: coalesce None to 'any'
    if materials is None:
        materials = "any"
    if conditions is None:
        conditions = "any"
    if times is None:
        times = "any"

    # Parse comma-separated strings into lists, handle 'any' (case-insensitive)
    mask = pd.Series([True] * len(FTIR_DataFrame))
    # Optionally exclude rows marked as bad quality
    try:
        if not show_bad:
            mask &= _quality_good_mask(FTIR_DataFrame).values
    except Exception:
        pass
    if isinstance(materials, str) and materials.strip().lower() != "any":
        material_list = [m.strip() for m in materials.split(",") if m.strip()]
        mask &= FTIR_DataFrame["Material"].isin(material_list)
    if isinstance(conditions, str) and conditions.strip().lower() != "any":
        condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
        # Base condition mask for selected conditions
        cond_mask = FTIR_DataFrame["Conditions"].isin(condition_list)
        # If Time == 'any', always include 'unexposed' spectra for the selected material(s),
        # regardless of the chosen condition(s) (applies across conditions)
        if isinstance(times, str) and times.strip().lower() == "any":
            try:
                cond_series = FTIR_DataFrame["Conditions"].astype(str).str.lower()
                cond_mask = cond_mask | (cond_series == "unexposed")
            except Exception:
                # Fallback without case normalization
                cond_mask = cond_mask | (FTIR_DataFrame["Conditions"] == "unexposed")
        mask &= cond_mask
    if isinstance(times, str) and times.strip().lower() != "any":
        # Try to convert to int if possible, else keep as string
        time_list = []
        for t in times.split(","):
            t = t.strip()
            if t:
                try:
                    time_list.append(int(t))
                except ValueError:
                    time_list.append(t)
        mask &= FTIR_DataFrame["Time"].isin(time_list)
        # If time selection includes 0, also include 'unexposed' spectra for the selected material(s)
        try:
            if any((isinstance(x, int) and x == 0) or (isinstance(x, str) and x.strip() == "0") for x in time_list):
                try:
                    unexp_series = FTIR_DataFrame["Conditions"].astype(str).str.lower()
                    unexp_mask = unexp_series == "unexposed"
                except Exception:
                    unexp_mask = FTIR_DataFrame["Conditions"].astype(str).str.lower() == "unexposed"
                if isinstance(materials, str) and materials.strip().lower() != "any":
                    # material_list defined earlier when materials filter applied
                    try:
                        material_list_local = [m.strip() for m in materials.split(",") if m.strip()]
                        unexp_mask &= FTIR_DataFrame["Material"].astype(str).isin(material_list_local)
                    except Exception:
                        unexp_mask &= FTIR_DataFrame.get("Material", pd.Series([])).astype(str).isin(material_list_local)
                mask |= unexp_mask
        except Exception:
            pass

    filtered_data = FTIR_DataFrame[mask]

    # Validate within filtered_data; show descriptive errors only if zero usable rows
    def _valid_mask_for_col(df, col):
        if df is None or len(df) == 0:
            return pd.Series([False] * (0 if df is None else len(df)))
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        vals = df[col]
        valid = []
        for v in vals:
            seq = None
            try:
                seq = _parse_seq(v)
            except Exception:
                seq = None
            if seq is None:
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        seq = v
                    elif isinstance(v, (list, tuple)):
                        seq = v
                except Exception:
                    pass
            if seq is not None:
                try:
                    import numpy as np
                    arr = np.asarray(seq, dtype=float).ravel()
                    valid.append(bool(arr.size > 0 and not np.isnan(arr).all()))
                    continue
                except Exception:
                    try:
                        ok = any((x is not None) and not pd.isna(x) for x in seq)
                        valid.append(bool(ok))
                        continue
                    except Exception:
                        valid.append(False)
                        continue
            if isinstance(v, (int, float)) and not pd.isna(v):
                valid.append(True)
            else:
                valid.append(False)
        return pd.Series(valid, index=vals.index)

    def _valid_mask_for_deconv(df):
        # Specialized validation for 'Deconvolution Results': expect a non-empty list
        if df is None or len(df) == 0:
            return pd.Series([False] * (0 if df is None else len(df)))
        if "Deconvolution Results" not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        vals = df["Deconvolution Results"]
        valid = []
        for v in vals:
            obj = v
            if isinstance(obj, str):
                try:
                    obj = ast.literal_eval(obj)
                except Exception:
                    obj = None
            ok = isinstance(obj, list) and len(obj) > 0
            valid.append(bool(ok))
        return pd.Series(valid, index=vals.index)

    noninteractive_validation_errors = []
    try:
        if baseline and not _valid_mask_for_col(filtered_data, "Baseline").any():
            noninteractive_validation_errors.append(
                "You need to baseline-correct the spectra before this will be available for plotting."
            )
        if baseline_corrected and not _valid_mask_for_col(filtered_data, "Baseline-Corrected Data").any():
            noninteractive_validation_errors.append(
                "You need to baseline-correct the spectra before this will be available for plotting."
            )
        if normalized and not _valid_mask_for_col(filtered_data, "Normalized and Corrected Data").any():
            noninteractive_validation_errors.append(
                "You need to normalize the spectra before this will be available for plotting."
            )
        if deconv_fit and not _valid_mask_for_deconv(filtered_data).any():
            noninteractive_validation_errors.append(
                "You need to deconvolute peaks before the deconvolution fit will be available for plotting."
            )
    except Exception:
        noninteractive_validation_errors = []

    if noninteractive_validation_errors:
        for msg in noninteractive_validation_errors:
            try:
                display(widgets.HTML(value=f"<b>Error:</b> {msg}"))
            except Exception:
                print(f"Error: {msg}")
        return

    # Row-level warnings (non-blocking): enumerate rows in filtered subset lacking required data
    try:
        row_level_messages = []
        # Build per-trace validity masks only for requested trace types
        trace_specs = [
            (baseline, "Baseline"),
            (baseline_corrected, "Baseline-Corrected Data"),
            (normalized, "Normalized and Corrected Data"),
        ]
        # Precompute validity masks (avoid re-parsing column multiple times)
        validity_cache = {}
        for flag, col_name in trace_specs:
            if not flag:
                continue
            try:
                validity_cache[col_name] = _valid_mask_for_col(filtered_data, col_name)
            except Exception:
                validity_cache[col_name] = pd.Series([False] * len(filtered_data), index=filtered_data.index)

        for idx, row in filtered_data.iterrows():
            mat = str(row.get("Material", "NA"))
            cond = str(row.get("Conditions", row.get("Condition", "NA")))
            t_val = row.get("Time", "NA")
            for flag, col_name in trace_specs:
                if not flag:
                    continue
                mask_series = validity_cache.get(col_name)
                is_valid = False
                try:
                    is_valid = bool(mask_series.loc[idx])
                except Exception:
                    is_valid = False
                if not is_valid:
                    # Generic user-facing guidance (no granular reason details)
                    advice_map = {
                        "Baseline": "You need to baseline-correct the spectra before this will be available for plotting.",
                        "Baseline-Corrected Data": "You need to baseline-correct the spectra before this will be available for plotting.",
                        "Normalized and Corrected Data": "You need to normalize the spectra before this will be available for plotting.",
                    }
                    advice = advice_map.get(col_name, "Required processing step missing.")
                    row_level_messages.append(
                        f"Row {idx} (Material={mat}, Conditions={cond}, Time={t_val}) missing {col_name}: {advice}"
                    )
        if row_level_messages:
            # Present as a collapsible-ish block; simple HTML formatting
            try:
                warn_html = (
                    "<div style='border:1px solid #e0a800;padding:8px;margin:6px 0;background:#fffbe6'>"
                    "<b>Warning:</b> Some requested processed traces are missing for specific rows.<br>"
                    + "<br>".join(row_level_messages)
                    + "</div>"
                )
                display(widgets.HTML(value=warn_html))
            except Exception:
                print("Warning: Some requested processed traces are missing for specific rows:")
                for m in row_level_messages:
                    print(" - " + m)
    except Exception:
        # Swallow any unexpected issues; plotting should continue
        pass

    # If nothing matches, explain why and bail early instead of showing a blank plot
    if filtered_data.empty:
        mats = sorted(
            set(
                map(
                    str, FTIR_DataFrame.get("Material", pd.Series([])).dropna().unique()
                )
            )
        )
        conds_col = (
            "Conditions"
            if "Conditions" in FTIR_DataFrame.columns
            else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
        )
        conds = (
            sorted(
                set(
                    map(
                        str,
                        FTIR_DataFrame.get(conds_col, pd.Series([])).dropna().unique(),
                    )
                )
            )
            if conds_col
            else []
        )
        times_avail = sorted(
            set(FTIR_DataFrame.get("Time", pd.Series([])).dropna().unique())
        )
        print(
            "No spectra matched the current filters.\n"
            f" - materials={materials!r}, conditions={conditions!r}, times={times!r}\n"
            "Try relaxing one or more filters (e.g., set to 'any').\n"
            f"Available Materials: {mats}\n"
            f"Available Conditions: {conds}\n"
            f"Available Times: {times_avail}"
        )
        return

    # If not including replicates, keep only the first member of each (Material,
    # Conditions, Time) group
    if not include_replicates:
        filtered_data = filtered_data.sort_values(by=["Material", "Conditions", "Time"])
        filtered_data = filtered_data.drop_duplicates(
            subset=["Material", "Conditions", "Time"], keep="first"
        )

    # Sort by time once for both legend and plotting (assume all times are integers)
    filtered_data_sorted = filtered_data.sort_values(by="Time")
    x_axis_col = "X-Axis" if "X-Axis" in filtered_data_sorted.columns else "Wavelength"

    # Helper: adaptive stride to cap points per trace when downsampling
    def _stride_for(n):
        try:
            n = int(n)
        except Exception:
            return 1
        target = 2000
        try:
            from math import ceil as _ceil

            return max(1, _ceil(n / float(target)))
        except Exception:
            return 1

    # Build shared palette mapping via _time_gradient_color and apply dash styles for replicates.
    time_values = [t for t in filtered_data_sorted.get("Time", []) if pd.notna(t)]
    try:
        time_values_unique = sorted(set(time_values))
    except Exception:
        time_values_unique = []

    def _time_to_color(val):
        try:
            # Use shared blue→purple→red equal-step palette
            return _time_gradient_color(time_values_unique, val)
        except Exception:
            return "rgb(160,80,200)"  # fallback purple

    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    replicate_counts = {}
    _row_line = {}
    for i, row in filtered_data_sorted.iterrows():
        mat_val = row.get("Material", "")
        cond_val = row.get("Conditions", row.get("Condition", ""))
        time_val = row.get("Time")
        key = (str(mat_val), str(cond_val), time_val)
        replicate_counts.setdefault(key, 0)
        rep_idx = replicate_counts[key]
        replicate_counts[key] += 1
        color = _time_to_color(time_val)
        if include_replicates:
            dash = dash_styles[rep_idx % len(dash_styles)]
        else:
            dash = "solid"
        _row_line[i] = {"color": color, "dash": dash}

    # Plot all together (legend in time order) with Plotly
    fig_group = go.FigureWidget()
    _row_plot_warnings = []
    for idx, spectrum_row in filtered_data_sorted.iterrows():
        try:
            material_val = spectrum_row.get("Material", "")
            condition_val = spectrum_row.get(
                "Conditions", spectrum_row.get("Condition", "")
            )
            time_val = spectrum_row.get("Time", "")
            spectrum_label = f"{material_val}, {condition_val}, {time_val}"
            # Parse x-axis
            x_axis = spectrum_row.get(x_axis_col)
            if isinstance(x_axis, str):
                try:
                    x_axis = ast.literal_eval(x_axis)
                except Exception:
                    x_axis = None
            if x_axis is None:
                _row_plot_warnings.append(
                    f"Row {idx} skipped: missing X-axis ('{x_axis_col}')."
                )
                continue

            # Plot selected series
            def _add_series(y, name_suffix):
                try:
                    if isinstance(y, str):
                        try:
                            y_v = ast.literal_eval(y)
                        except Exception:
                            y_v = None
                    else:
                        y_v = y
                    # Suppress scalar numeric (float/int) mistaken as iterable; treat as missing
                    if isinstance(y_v, (int, float)) and not isinstance(y_v, bool):
                        y_v = None
                    if y_v is not None:
                        x_list = list(x_axis)
                        y_list = list(y_v)
                        if downsample:
                            s = _stride_for(len(x_list))
                            if s > 1:
                                x_list = x_list[::s]
                                y_list = y_list[::s]
                        fig_group.add_scatter(
                            x=x_list,
                            y=y_list,
                            mode="lines",
                            name=f"{name_suffix}: {spectrum_label}",
                            line=_row_line.get(idx),
                        )
                    else:
                        _row_plot_warnings.append(
                            f"Row {idx} missing data for '{name_suffix}' trace."
                        )
                except Exception as e:
                    # Suppress verbose 'float object is not iterable' errors (already covered by higher-level messages)
                    if isinstance(e, TypeError) and "float" in str(e) and "iterable" in str(e):
                        pass
                    else:
                        _row_plot_warnings.append(
                            f"Row {idx} error while adding '{name_suffix}' trace: {e}"
                        )

            def _add_deconv_fit():
                try:
                    res = spectrum_row.get("Deconvolution Results")
                    if isinstance(res, str):
                        try:
                            res = ast.literal_eval(res)
                        except Exception:
                            res = None
                    if not isinstance(res, list) or len(res) == 0:
                        _row_plot_warnings.append(
                            f"Row {idx} missing 'Deconvolution Results' for deconvolution fit."
                        )
                        return
                    # Build composite pseudo-Voigt model from stored parameters
                    y_fit = None
                    x_arr = np.asarray(list(x_axis), dtype=float)
                    y_fit = np.zeros_like(x_arr, dtype=float)
                    for j, p in enumerate(res):
                        try:
                            amp = float(p.get("amplitude", 0.0))
                        except Exception:
                            amp = 0.0
                        try:
                            cen = float(p.get("center", float("nan")))
                        except Exception:
                            cen = float("nan")
                        # Prefer unified sigma; fallback to sigma_l or sigma_g
                        try:
                            sig = p.get("sigma", None)
                            if sig is None:
                                sig = p.get("sigma_l", p.get("sigma_g", None))
                            sig = float(sig) if sig is not None else float("nan")
                        except Exception:
                            sig = float("nan")
                        try:
                            frac = p.get("alpha", p.get("fraction", 0.5))
                            frac = float(frac)
                        except Exception:
                            frac = 0.5
                        # Skip invalid components
                        if not (np.isfinite(amp) and np.isfinite(cen) and np.isfinite(sig)):
                            continue
                        try:
                            mdl = PseudoVoigtModel(prefix=f"p{j}_")
                            params = mdl.make_params(
                                amplitude=max(0.0, amp), center=cen, sigma=max(1e-9, sig), fraction=min(max(frac, 0.0), 1.0)
                            )
                            y_fit = y_fit + mdl.eval(params, x=x_arr)
                        except Exception:
                            # Fallback manual PV evaluation if lmfit model fails
                            try:
                                g = np.exp(-((x_arr - cen) ** 2) / (2.0 * (sig ** 2)))
                                l = (sig ** 2) / (((x_arr - cen) ** 2) + (sig ** 2))
                                pv = frac * l + (1.0 - frac) * g
                                # Scale pv to amplitude approximately (peak area vs height ambiguity); use height scaling
                                pv = amp * pv / (np.max(pv) if np.max(pv) > 0 else 1.0)
                                y_fit = y_fit + pv
                            except Exception:
                                continue
                    # Add trace
                    x_list = list(x_arr)
                    y_list = list(y_fit)
                    if downsample:
                        s = _stride_for(len(x_list))
                        if s > 1:
                            x_list = x_list[::s]
                            y_list = y_list[::s]
                    # Style: thicker line and same color
                    line_style = dict(_row_line.get(idx)) if isinstance(_row_line.get(idx), dict) else {}
                    line_style.update({"width": 2})
                    fig_group.add_scatter(
                        x=x_list,
                        y=y_list,
                        mode="lines",
                        name=f"Deconv fit: {spectrum_label}",
                        line=line_style,
                    )
                except Exception as e:
                    _row_plot_warnings.append(
                        f"Row {idx} error while adding deconvolution fit: {e}"
                    )

            if raw_data and ("Raw Data" in spectrum_row):
                _add_series(spectrum_row.get("Raw Data"), "Raw")
            if baseline and (spectrum_row.get("Baseline") is not None):
                _add_series(spectrum_row.get("Baseline"), "Baseline")
            if baseline_corrected and (
                spectrum_row.get("Baseline-Corrected Data") is not None
            ):
                _add_series(
                    spectrum_row.get("Baseline-Corrected Data"), "Baseline-Corrected"
                )
            if normalized and (
                spectrum_row.get("Normalized and Corrected Data") is not None
            ):
                _add_series(
                    spectrum_row.get("Normalized and Corrected Data"),
                    "Normalized and Corrected",
                )
            if deconv_fit:
                _add_deconv_fit()
        except Exception as e:
            _row_plot_warnings.append(
                f"Row {idx} skipped entirely due to unexpected error: {e}"
            )

    if _row_plot_warnings:
        try:
            warn_html = (
                "<div style='border:1px solid #e0a800;padding:8px;margin:6px 0;background:#fffbe6'>"
                "<b>Warning:</b> Some rows were skipped or partially plotted due to missing/invalid data.<br>"
                + "<br>".join(_row_plot_warnings)
                + "</div>"
            )
            display(widgets.HTML(value=warn_html))
        except Exception:
            print("Warning: issues encountered while plotting some rows:")
            for m in _row_plot_warnings:
                print(" - " + m)
    fig_group.update_layout(
        title=f"Spectra for Material: {materials} | Condition: {conditions} | Time: {times}",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
        # Place legend on the right to avoid overlapping x-axis title
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        margin=dict(r=140),
    )
    display(fig_group)

    # Display plotted file locations in the exact order they were plotted
    try:
        file_paths = []
        for _i, _row in filtered_data_sorted.iterrows():
            folder = str(_row.get("File Location", "") or "")
            fname = str(_row.get("File Name", "") or "")
            full_path = (
                os.path.join(folder, fname) if folder and fname else (folder or fname)
            )
            if full_path:
                file_paths.append(full_path)
        if file_paths:
            header = f"Plotted file order ({len(file_paths)}):"
            html = "<b>{}</b><br><pre style='margin:0'>{}</pre>".format(
                header, "\n".join(f"{i+1}. {p}" for i, p in enumerate(file_paths))
            )
            display(widgets.HTML(value=html))
    except Exception as e:
        try:
            print(f"Note: could not display file list: {e}")
        except Exception:
            pass

    # Optional: mark selected rows as good/bad after plotting
    try:
        qcol = _quality_column_name(FTIR_DataFrame)

        # Local function to append quality change to session log
        def _log_quality(idx_val, status_val):
            try:
                global _PLOT_SPECTRA_SESSION_CHANGES
                if isinstance(_PLOT_SPECTRA_SESSION_CHANGES, dict):
                    _PLOT_SPECTRA_SESSION_CHANGES.setdefault(
                        "quality", []
                    ).append((idx_val, status_val))
            except Exception:
                pass

        # Mark good first
        if mark_good is not None and not filtered_data.empty:
            if isinstance(mark_good, str) and mark_good.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "good"
                for _i in filtered_data.index.tolist():
                    _log_quality(_i, "good")
            elif isinstance(mark_good, (list, tuple, set)):
                to_mark_idx = []
                for item in mark_good:
                    try:
                        if isinstance(item, (int, np.integer)):
                            to_mark_idx.append(int(item))
                        else:
                            matches = filtered_data.index[
                                filtered_data.get("File Name", "").astype(str)
                                == str(item)
                            ].tolist()
                            to_mark_idx.extend(matches)
                    except Exception:
                        pass
                if to_mark_idx:
                    FTIR_DataFrame.loc[list(set(to_mark_idx)), qcol] = "good"
                    for _i in set(to_mark_idx):
                        _log_quality(_i, "good")
        # Then mark bad
        if mark_bad is not None and not filtered_data.empty:
            if isinstance(mark_bad, str) and mark_bad.strip().lower() == "all":
                FTIR_DataFrame.loc[filtered_data.index, qcol] = "bad"
                for _i in filtered_data.index.tolist():
                    _log_quality(_i, "bad")
            elif isinstance(mark_bad, (list, tuple, set)):
                to_mark_idx = []
                for item in mark_bad:
                    try:
                        if isinstance(item, (int, np.integer)):
                            to_mark_idx.append(int(item))
                        else:
                            matches = filtered_data.index[
                                filtered_data.get("File Name", "").astype(str)
                                == str(item)
                            ].tolist()
                            to_mark_idx.extend(matches)
                    except Exception:
                        pass
                if to_mark_idx:
                    FTIR_DataFrame.loc[list(set(to_mark_idx)), qcol] = "bad"
                    for _i in set(to_mark_idx):
                        _log_quality(_i, "bad")
    except Exception:
        pass

    # Plot each file individually if requested, in sequential order by time
    if separate_plots:
        for idx, row in filtered_data_sorted.iterrows():
            # Create individual Plotly figure
            fig_i = go.FigureWidget()
            x_axis = row.get(x_axis_col)
            if isinstance(x_axis, str):
                try:
                    x_axis = ast.literal_eval(x_axis)
                except Exception:
                    x_axis = None
            if x_axis is None:
                continue

            def _add_series_i(y, name_suffix):
                if isinstance(y, str):
                    try:
                        y_v = ast.literal_eval(y)
                    except Exception:
                        y_v = None
                else:
                    y_v = y
                if y_v is not None:
                    x_list = list(x_axis)
                    y_list = list(y_v)
                    if downsample:
                        s = _stride_for(len(x_list))
                        if s > 1:
                            x_list = x_list[::s]
                            y_list = y_list[::s]
                    fig_i.add_scatter(
                        x=x_list,
                        y=y_list,
                        mode="lines",
                        name=name_suffix,
                        line=_row_line.get(idx),
                    )

            if raw_data:
                _add_series_i(row.get("Raw Data"), "Raw")
            if baseline and (row.get("Baseline") is not None):
                _add_series_i(row.get("Baseline"), "Baseline")
            if baseline_corrected and (row.get("Baseline-Corrected Data") is not None):
                _add_series_i(row.get("Baseline-Corrected Data"), "Baseline-Corrected")
            if normalized and (row.get("Normalized and Corrected Data") is not None):
                _add_series_i(
                    row.get("Normalized and Corrected Data"), "Normalized and Corrected"
                )
            # Deconvolution fit per-spectrum
            if deconv_fit:
                try:
                    res = row.get("Deconvolution Results")
                    if isinstance(res, str):
                        try:
                            res = ast.literal_eval(res)
                        except Exception:
                            res = None
                    if isinstance(res, list) and len(res) > 0:
                        x_arr = np.asarray(list(x_axis), dtype=float)
                        y_fit = np.zeros_like(x_arr, dtype=float)
                        for j, p in enumerate(res):
                            try:
                                amp = float(p.get("amplitude", 0.0))
                            except Exception:
                                amp = 0.0
                            try:
                                cen = float(p.get("center", float("nan")))
                            except Exception:
                                cen = float("nan")
                            try:
                                sig = p.get("sigma", None)
                                if sig is None:
                                    sig = p.get("sigma_l", p.get("sigma_g", None))
                                sig = float(sig) if sig is not None else float("nan")
                            except Exception:
                                sig = float("nan")
                            try:
                                frac = p.get("alpha", p.get("fraction", 0.5))
                                frac = float(frac)
                            except Exception:
                                frac = 0.5
                            if not (np.isfinite(amp) and np.isfinite(cen) and np.isfinite(sig)):
                                continue
                            try:
                                mdl = PseudoVoigtModel(prefix=f"p{j}_")
                                params = mdl.make_params(
                                    amplitude=max(0.0, amp), center=cen, sigma=max(1e-9, sig), fraction=min(max(frac, 0.0), 1.0)
                                )
                                y_fit = y_fit + mdl.eval(params, x=x_arr)
                            except Exception:
                                try:
                                    g = np.exp(-((x_arr - cen) ** 2) / (2.0 * (sig ** 2)))
                                    l = (sig ** 2) / (((x_arr - cen) ** 2) + (sig ** 2))
                                    pv = frac * l + (1.0 - frac) * g
                                    pv = amp * pv / (np.max(pv) if np.max(pv) > 0 else 1.0)
                                    y_fit = y_fit + pv
                                except Exception:
                                    continue
                        x_list = list(x_arr)
                        y_list = list(y_fit)
                        if downsample:
                            s = _stride_for(len(x_list))
                            if s > 1:
                                x_list = x_list[::s]
                                y_list = y_list[::s]
                        fig_i.add_scatter(
                            x=x_list,
                            y=y_list,
                            mode="lines",
                            name="Deconvolution fit",
                            line=dict(width=2, color=_row_line.get(idx, {}).get("color", "#000000")),
                        )
                    else:
                        pass
                except Exception:
                    pass
            material_val = row.get("Material", "")
            condition_val = row.get("Conditions", row.get("Condition", ""))
            time_val = row.get("Time", "")
            fig_i.update_layout(
                title=f"Spectrum: {material_val}, {condition_val}, {time_val}",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Absorbance (AU)",
                legend=dict(orientation="h", y=-0.2),
            )

            # Unified quality controls via shared helper (closure captures this row index)
            mark_bad_btn, mark_good_btn, _refresh_row_btns = _make_quality_controls(
                FTIR_DataFrame, lambda i=idx: FTIR_DataFrame.loc[i]
            )
            # Add logging callbacks (helper already updates DataFrame & toggles buttons)
            try:
                mark_bad_btn.on_click(lambda _b=None, i=idx: _log_quality(i, "bad"))
                mark_good_btn.on_click(lambda _b=None, i=idx: _log_quality(i, "good"))
            except Exception:
                pass
            # Ensure initial visibility reflects current row status (helper ran refresh once)
            try:
                _refresh_row_btns()
            except Exception:
                pass
            # Display figure with its quality mark buttons BELOW the plot inside a bordered container
            try:
                container = widgets.VBox(
                    [
                        fig_i,
                        widgets.HBox([mark_bad_btn, mark_good_btn], layout=widgets.Layout(margin="4px 0 0 0")),
                    ],
                    layout=widgets.Layout(
                        border="1px solid #ccc",
                        padding="8px",
                        margin="6px 0",
                    ),
                )
                display(container)
            except Exception:
                # Fallback: show buttons with upward arrow indicating association if styling fails
                mark_bad_btn.description = "Mark spectrum as bad ↑"
                mark_good_btn.description = "Mark spectrum as good ↑"
                display(widgets.VBox([fig_i, widgets.HBox([mark_bad_btn, mark_good_btn])]))


def baseline_correct_spectra(
    FTIR_DataFrame,
    material=None,
    baseline_function=None,
    filepath=None,
):
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """
    Apply a modifiable baseline to a single spectrum from the DataFrame.

    Allows for on-the-fly parameter adjustments via interactive widgets and
    experimentation with different baseline functions.

    Parameters
    ----------
    FTIR_DataFrame (pd.DataFrame): The in-memory DataFrame containing all spectra.
    material (str, optional): Material name to analyze (ignored if filepath is
        provided).
    baseline_function (str): Baseline function to use ('ARPLS', 'IRSQR', 'FABC').
    filepath (str, optional): If provided, only process this file (by 'File Location'
        + 'File Name').

    Returns
    -------
    FTIR_DataFrame : pd.DataFrame
        The updated DataFrame with baseline corrections applied, if user chooses to save
        choices. Otherwise, the DataFrame remains unchanged.
    """
    # Proactively close any leftover widgets from a prior session to ensure fresh UI renders
    try:
        for _w in list(_TB_WIDGETS):
            try:
                _w.close()
            except Exception:
                pass
        _TB_WIDGETS.clear()
    except Exception:
        pass
    try:
        clear_output(wait=True)
    except Exception:
        pass

    # Per-session change log for this baseline_correct_spectra (interactive) session
    # Added baseline_corrected_* keys so the summary can explicitly report
    # which files/materials had baseline-corrected data generated inline.
    baseline_session_changes = {
        "quality": [],
        "saved_file": [],
        "saved_filtered": 0,
        "baseline_corrected_file": [],  # list[(idx, function_name)]
        "baseline_corrected_material": [],  # list[(material, function_name, count_updated)]
    }

    # --- Helper: robustly extract numeric x/y arrays from a DataFrame row ---
    def _row_xy(
        row,
    ):  # local helper (only used inside interactive baseline_correct_spectra)
        """Return (x_array, y_array) as float numpy arrays for a row.

        Tries literal_eval for string-stored lists; tolerates already-list/array.
        Falls back to empty arrays on any failure (so caller can skip row cleanly).
        """
        try:
            x_raw = row.get("X-Axis")
            y_raw = row.get("Raw Data")
        except Exception:
            return np.array([], dtype=float), np.array([], dtype=float)

        # Decode strings representing arrays; try literal_eval first, then a robust fallback
        def _to_array_from_string(s):
            # First attempt: literal_eval on python-literal list strings
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            # Fallback: find bracketed content and parse with numpy.fromstring
            try:
                txt = str(s)
                lb = txt.find("[")
                rb = txt.rfind("]")
                if lb != -1 and rb != -1 and rb > lb:
                    inner = txt[lb + 1 : rb]
                    # Normalize common tokens
                    inner = (
                        inner.replace("NaN", "nan")
                        .replace("INF", "inf")
                        .replace("-INF", "-inf")
                    )
                    arr = np.fromstring(inner, sep=",")
                    return arr.tolist()
            except Exception:
                pass
            return s

        if isinstance(x_raw, str):
            x_raw = _to_array_from_string(x_raw)
        if isinstance(y_raw, str):
            y_raw = _to_array_from_string(y_raw)
        # Coerce iterables to list first to avoid pandas Series dtype surprises
        try:
            if hasattr(x_raw, "__iter__") and not isinstance(x_raw, (str, bytes)):
                x_raw = list(x_raw)
        except Exception:
            x_raw = []
        try:
            if hasattr(y_raw, "__iter__") and not isinstance(y_raw, (str, bytes)):
                y_raw = list(y_raw)
        except Exception:
            y_raw = []
        try:
            x_arr = np.asarray(x_raw, dtype=float)
        except Exception:
            x_arr = np.array([], dtype=float)
        try:
            y_arr = np.asarray(y_raw, dtype=float)
        except Exception:
            y_arr = np.array([], dtype=float)
        # Basic shape validation; allow proceeding only if 1D and same length
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size == 0 or y_arr.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if x_arr.size != y_arr.size:
            return np.array([], dtype=float), np.array([], dtype=float)
        return x_arr, y_arr

    if baseline_function is None:
        # Default to ARPLS when not specified; user can change via dropdown below
        baseline_function = "ARPLS"
    # Do NOT auto-launch manual baseline; user must still select a spectrum first (minimal mode preserved)
    # Initialize selection placeholders; user will pick a spectrum via dropdowns
    selected_row = None
    # Track currently displayed spectrum independent of dropdown selection so plot persists
    # when its option is removed after marking bad quality with 'Include bad spectra' unchecked.
    current_idx_bc = None
    x_values = np.array([])
    y_values = np.array([])
    # If neither material nor filepath provided, try session defaults for material
    if material is None and filepath is None:
        try:
            _sess = _get_session_defaults()
            sess_mat = _sess.get("material")
            if isinstance(sess_mat, str) and sess_mat.strip().lower() != "any":
                material = sess_mat
        except Exception:
            pass

    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            filtered_df = FTIR_DataFrame[
                (FTIR_DataFrame["File Location"] == folder)
                & (FTIR_DataFrame["File Name"] == fname)
            ]
        else:
            filtered_df = FTIR_DataFrame[FTIR_DataFrame["File Name"] == filepath]
        # Exclude rows marked as bad quality
        try:
            filtered_df = filtered_df[_quality_good_mask(filtered_df)]
        except Exception:
            pass
        if filtered_df.empty:
            raise ValueError(f"No entry found for file '{filepath}'.")
        selected_row = filtered_df.iloc[0]
        material = selected_row.get("Material", "Unknown")
        # Persist this selection to session state
        try:
            _set_session_selection(
                material=selected_row.get("Material"),
                conditions=selected_row.get("Conditions"),
                time=selected_row.get("Time"),
            )
        except Exception:
            pass
    # If a specific file is selected, compute x/y; otherwise wait for user selection
    if selected_row is not None:
        x_values = (
            ast.literal_eval(selected_row["X-Axis"])
            if isinstance(selected_row["X-Axis"], str)
            else selected_row["X-Axis"]
        )
        y_values = (
            ast.literal_eval(selected_row["Raw Data"])
            if isinstance(selected_row["Raw Data"], str)
            else selected_row["Raw Data"]
        )
        y_values = np.array(y_values, dtype=float)

    parameters = _get_default_parameters(baseline_function)
    parameters = _cast_parameter_types(baseline_function, parameters)

    # Print selected file path only after a specific file is chosen
    if selected_row is not None:
        file_path = os.path.join(
            selected_row.get("File Location", ""), selected_row.get("File Name", "")
        )
        try:
            print(f"Plotting: {file_path}")
        except Exception:
            pass

    # Widget setup for live parameter editing
    baseline_parameter_widgets = {}
    # Explicitly define widgets for each baseline function and parameter
    if baseline_function.upper() == "ARPLS":
        # lam: float, iterations: int (diff_order fixed internally; not user-editable)
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e5),
            min=1e4,
            max=1e6,
            step=1e4,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 50),
            min=1,
            max=200,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-3),
            min=1e-6,
            max=1e-1,
            step=1e-4,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "IRSQR":
        # lam: float, quantile: float, num_knots: int, spline_degree: int, diff_order:
        # int, max_iterations: int, tolerance: float, eps: float
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e5,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
            value=parameters.get("quantile", 0.05),
            min=0.001,
            max=0.5,
            step=0.001,
            description="Quantile",
            readout_format=".3f",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
            value=parameters.get("num_knots", 100),
            min=5,
            max=500,
            step=5,
            description="Knots",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
            value=parameters.get("spline_degree", 3),
            min=1,
            max=5,
            step=1,
            description="Spline Degree",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 3),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
            value=parameters.get("max_iter", 100),
            min=1,
            max=1000,
            step=1,
            description="Max Iterations",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["tol"] = widgets.FloatSlider(
            value=parameters.get("tol", 1e-6),
            min=1e-10,
            max=1e-2,
            step=1e-7,
            description="Tolerance",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
    elif baseline_function.upper() == "FABC":
        # lam: float, scale: int or None, num_std: float, diff_order: int, min_length:
        # int
        baseline_parameter_widgets["lam"] = widgets.FloatSlider(
            value=parameters.get("lam", 1e6),
            min=1e4,
            max=1e7,
            step=1e5,
            description="Smoothness (lam)",
            readout_format=".1e",
            style={"description_width": "auto"},
        )
        # If no spectrum is selected yet, use a generic default for scale; recomputed on selection
        if selected_row is not None and y_values is not None and len(y_values) > 0:
            try:
                scale_default = int(
                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                )
            except Exception:
                scale_default = 50
        else:
            scale_default = 50
        scale_val = parameters.get("scale", None)
        if scale_val is None:
            scale_val = scale_default
        baseline_parameter_widgets["scale"] = widgets.IntSlider(
            value=int(scale_val),
            min=2,
            max=500,
            step=1,
            description="Scale",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
            value=parameters.get("num_std", 3.0),
            min=1.5,
            max=4.5,
            step=0.1,
            description="Standard Deviations",
            readout_format=".2f",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
            value=parameters.get("diff_order", 2),
            min=1,
            max=3,
            step=1,
            description="Differential Order",
            style={"description_width": "auto"},
        )
        baseline_parameter_widgets["min_length"] = widgets.IntSlider(
            value=parameters.get("min_length", 2),
            min=1,
            max=6,
            step=1,
            description="Min Baseline Span Length",
            style={"description_width": "auto"},
        )

    # -------------------
    # Filtering and spectrum selection controls (Material/Conditions/Spectrum)
    # -------------------
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    # Build base DataFrame for options: restrict to filepath if provided
    if filepath is not None:
        base_df = filtered_df.copy()
    else:
        base_df = FTIR_DataFrame.copy()
        try:
            base_df = base_df[_quality_good_mask(base_df)]
        except Exception:
            pass
    # Unique materials
    try:
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in base_df.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
            if "Material" in base_df.columns
            else []
        )
    except Exception:
        unique_materials = []
    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    # Conditions list (exclude 'unexposed' from selector values)
    try:
        cond_series = (
            base_df["Conditions"]
            if "Conditions" in base_df.columns
            else (
                base_df["Condition"]
                if "Condition" in base_df.columns
                else pd.Series([], dtype=object)
            )
        )
        _all_conditions = [
            str(v) for v in cond_series.dropna().astype(str).unique().tolist()
        ]
        unique_conditions = sorted(
            [c for c in _all_conditions if c.strip().lower() != "unexposed"]
        )
    except Exception:
        unique_conditions = []
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    # Apply session defaults to filters
    try:
        _sess = _get_session_defaults()
        sess_mat = _sess.get("material")
        if isinstance(sess_mat, str) and sess_mat in unique_materials:
            material_dd.value = sess_mat
        sess_cond = _sess.get("conditions")
        if isinstance(sess_cond, str) and sess_cond in unique_conditions:
            conditions_dd.value = sess_cond
    except Exception:
        pass
    # Baseline function dropdown (user can switch between methods)
    baseline_dd = widgets.Dropdown(
        options=["ARPLS", "IRSQR", "FABC", "MANUAL"],
        value=(
            str(baseline_function).upper()
            if str(baseline_function).upper() in ["ARPLS", "IRSQR", "FABC", "MANUAL"]
            else "ARPLS"
        ),
        description="Baseline",
        layout=widgets.Layout(width="30%"),
    )
    # Normalize baseline_function to a valid string and sync with dropdown to avoid None-related crashes on first selection
    try:
        if not isinstance(baseline_function, str) or str(
            baseline_function
        ).upper() not in ("ARPLS", "IRSQR", "FABC", "MANUAL"):
            baseline_function = str(baseline_dd.value).upper()
        else:
            baseline_function = str(baseline_function).upper()
    except Exception:
        baseline_function = "ARPLS"
    # Spectrum dropdown (built via helper)
    spectrum_sel = widgets.Dropdown(
        options=[("Select a spectrum…", None)],
        value=None,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )

    def _rebuild_conditions_options():
        """Rebuild the Conditions dropdown based on current Material filter and data."""
        try:
            if material_dd.value == "any":
                dfm = base_df
            else:
                dfm = base_df[
                    base_df.get("Material", "").astype(str) == str(material_dd.value)
                ]
            cs = (
                dfm["Conditions"]
                if "Conditions" in dfm.columns
                else (
                    dfm["Condition"]
                    if "Condition" in dfm.columns
                    else pd.Series([], dtype=object)
                )
            )
            cvals = [str(v) for v in cs.dropna().astype(str).unique().tolist()]
            cvals = sorted([c for c in cvals if c.strip().lower() != "unexposed"])
            curr = (
                conditions_dd.value
                if conditions_dd.value in (["any"] + cvals)
                else "any"
            )
            conditions_dd.options = ["any"] + cvals
            conditions_dd.value = curr
        except Exception:
            pass

    def _build_spectrum_options():
        """Recompute the Spectrum dropdown options using current filters and session defaults."""
        # Update current selection
        nonlocal selected_row, x_values, y_values, material
        try:
            df = base_df.copy()
            if not include_bad_cb.value:
                try:
                    df = df[_quality_good_mask(df)]
                except Exception:
                    pass
            if material_dd.value != "any":
                df = df[df.get("Material", "").astype(str) == str(material_dd.value)]
            # Filter by conditions if chosen
            if conditions_dd.value != "any":
                cond_col = (
                    "Conditions"
                    if "Conditions" in df.columns
                    else ("Condition" if "Condition" in df.columns else None)
                )
                if cond_col is not None:
                    # Always include 'unexposed' spectra in addition to the selected condition
                    sel_val = str(conditions_dd.value)
                    cond_series = df.get(cond_col, pd.Series([], dtype=object)).astype(
                        str
                    )
                    unexp_mask = cond_series.str.strip().str.lower() == "unexposed"
                    cond_mask = cond_series == sel_val
                    df = df[cond_mask | unexp_mask]
            # Sort by time if present
            if "Time" in df.columns:
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
            options = []
            for idx2, r2 in df.iterrows():
                label = (
                    f"{r2.get('Material','')} | {r2.get('Conditions', r2.get('Condition',''))}"
                    f" | T={r2.get('Time','')} | {r2.get('File Name','')}"
                )
                options.append((label, idx2))
            if not options:
                spectrum_sel.options = [("<no spectra>", None)]
                spectrum_sel.value = None
                return
            # Prefer previously selected idx if still present; else prefer session time
            prev = spectrum_sel.value
            values = [v for (_l, v) in options]
            chosen = None
            if prev in values:
                chosen = prev
            else:
                # try session time
                try:
                    _sess2 = _get_session_defaults()
                    stime = _sess2.get("time")
                except Exception:
                    stime = None
                if stime is not None:
                    for _l, v in options:
                        try:
                            if float(FTIR_DataFrame.loc[v].get("Time")) == float(stime):
                                chosen = v
                                break
                        except Exception:
                            continue
            # Present a placeholder + options; do not auto-select if no match
            spectrum_sel.options = [("Select a spectrum…", None)] + options
            spectrum_sel.value = chosen if chosen in values else None
            if chosen is not None:
                # Update current selection variables only when a specific spectrum is chosen
                rsel = FTIR_DataFrame.loc[chosen]
                selected_row = rsel
                material = rsel.get("Material", material)
                x_values = (
                    ast.literal_eval(rsel["X-Axis"])
                    if isinstance(rsel["X-Axis"], str)
                    else rsel["X-Axis"]
                )
                y_values = (
                    ast.literal_eval(rsel["Raw Data"])
                    if isinstance(rsel["Raw Data"], str)
                    else rsel["Raw Data"]
                )
                y_values = np.array(y_values, dtype=float)
                # Persist session
                try:
                    _set_session_selection(
                        material=selected_row.get("Material"),
                        conditions=selected_row.get("Conditions"),
                        time=selected_row.get("Time"),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # Seed initial lists based on current state
    _rebuild_conditions_options()
    _build_spectrum_options()

    # Force minimal mode on entry: avoid auto-selected spectrum from session state
    # so the UI always renders and the user explicitly confirms the selection.
    try:
        if spectrum_sel.value is not None:
            spectrum_sel.value = None
            selected_row = None
    except Exception:
        pass

    baseline_output_area = widgets.Output()
    # Persist a single Plotly FigureWidget and update its traces for low flicker
    baseline_figure_widget = None

    def _plot_baseline(**widget_params):
        """Compute and display baseline and residual using current params and selected spectrum."""
        nonlocal baseline_figure_widget
        # Merge and cast widget parameters
        param_vals = parameters.copy()
        param_vals.update(widget_params)
        param_vals = _cast_parameter_types(baseline_function, param_vals)
        with baseline_output_area:
            # If no spectrum has been selected yet, prompt once
            # Allow continued display even if dropdown value cleared after marking bad.
            if selected_row is None:
                # Do not clear an existing plot; only show prompt if nothing rendered yet.
                if baseline_figure_widget is None:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print("Select a spectrum to preview the baseline.")
                else:
                    # Leave existing figure visible; optionally could append a note.
                    try:
                        print("(Select a spectrum to update the preview.)")
                    except Exception:
                        pass
                return

            # Compute baseline safely
            try:
                if baseline_function.upper() == "ARPLS":
                    baseline_result = arpls(y_values, **param_vals)
                elif baseline_function.upper() == "IRSQR":
                    baseline_result = irsqr(y_values, **param_vals, x_data=x_values)
                elif baseline_function.upper() == "FABC":
                    baseline_result = fabc(y_values, **param_vals)
                else:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print(f"Unknown baseline function: {baseline_function}")
                    return
            except Exception as e:
                try:
                    clear_output(wait=True)
                except Exception:
                    pass
                print(f"Baseline computation error: {e}")
                return

            # Normalize baseline output
            if isinstance(baseline_result, tuple):
                baseline = baseline_result[0]
            elif isinstance(baseline_result, dict):
                baseline = baseline_result.get("baseline")
                if baseline is None:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass
                    print("Error: Baseline function did not return a baseline array.")
                    return
            else:
                baseline = baseline_result

            try:
                # Prepare arrays (parameter summary logic removed per simplification request)
                x_arr = np.asarray(x_values)
                y_arr = np.asarray(y_values)
                baseline_arr = np.asarray(baseline)
                residual = y_arr - baseline_arr

                # Build or update Plotly FigureWidget
                title_top = "Raw Data and Baseline"
                if (
                    baseline_figure_widget is None
                    or len(getattr(baseline_figure_widget, "data", [])) < 3
                ):
                    # Do not clear the whole cell output; this figure renders outside the Output widget.
                    # Create subplots: top (raw + baseline), bottom (baseline-corrected)
                    base_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=(title_top, "Baseline-Corrected"),
                    )
                    baseline_figure_widget = go.FigureWidget(base_fig)
                    # Raw spectrum
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=y_arr,
                        mode="lines",
                        name="Spectrum",
                        line=dict(color="black"),
                        row=1,
                        col=1,
                    )
                    # Baseline
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=baseline_arr,
                        mode="lines",
                        name="Baseline",
                        line=dict(color="red", width=1.5, dash="dash"),
                        row=1,
                        col=1,
                    )
                    # Baseline-corrected (spectrum - baseline)
                    baseline_figure_widget.add_scatter(
                        x=x_arr,
                        y=residual,
                        mode="lines",
                        name="Baseline-Corrected",
                        line=dict(color="blue"),
                        row=2,
                        col=1,
                    )
                    # Axes labels and layout
                    baseline_figure_widget.update_yaxes(
                        title_text="Absorbance (AU)", row=1, col=1
                    )
                    baseline_figure_widget.update_yaxes(title_text="", row=2, col=1)
                    baseline_figure_widget.update_xaxes(
                        title_text="Wavenumber (cm⁻¹)", row=2, col=1
                    )
                    baseline_figure_widget.update_layout(
                        legend=dict(orientation="h", y=-0.2), height=800
                    )
                    display(baseline_figure_widget)
                else:
                    # Update data traces in-place (no redraw flicker)
                    try:
                        baseline_figure_widget.data[0].x = x_arr
                        baseline_figure_widget.data[0].y = y_arr
                        baseline_figure_widget.data[1].x = x_arr
                        baseline_figure_widget.data[1].y = baseline_arr
                        baseline_figure_widget.data[2].x = x_arr
                        baseline_figure_widget.data[2].y = residual
                    except Exception:
                        # Fall back to rebuild if trace shapes changed unexpectedly
                        baseline_figure_widget = None
                        _plot_baseline(**widget_params)
                        return
                    # Update subplot titles
                    try:
                        if (
                            hasattr(baseline_figure_widget.layout, "annotations")
                            and len(baseline_figure_widget.layout.annotations) >= 2
                        ):
                            baseline_figure_widget.layout.annotations[0].text = (
                                title_top
                            )
                            baseline_figure_widget.layout.annotations[1].text = (
                                "Baseline-Corrected"
                            )
                    except Exception:
                        pass
            except Exception as e:
                # Keep any existing figure; report error in the Output widget only.
                try:
                    clear_output(wait=True)
                    print(f"Plot error: {e}")
                except Exception:
                    pass

    # Minimal UI when no spectrum is selected: show only filters, spectrum dropdown, and Close button
    try:
        _no_selection = (selected_row is None) or (spectrum_sel.value is None)
    except Exception:
        _no_selection = True
    if _no_selection:
        # Simple message prompting selection
        with baseline_output_area:
            try:
                clear_output(wait=True)
                print("Select a spectrum to preview the baseline.")
            except Exception:
                pass

    close_btn = widgets.Button(
        description="Close",
        button_style="danger",
        layout=widgets.Layout(margin="10px 0 0 0"),
    )

    # Filters and spectrum rows (no sliders or other buttons)
    filters_row = widgets.HBox(
        [material_dd, conditions_dd, baseline_dd]
    )
    spectrum_row = widgets.HBox([spectrum_sel, include_bad_cb])
    ui = widgets.VBox([filters_row, spectrum_row, close_btn])

    container = widgets.VBox([ui, baseline_output_area])
    display(container)
    try:
        _TB_WIDGETS.extend([container])
    except Exception:
        pass

    # Wire minimal interactions: rebuild options on filter change
    def _on_mat_min(change):
        """Minimal-mode observer: when Material changes, refresh Conditions and Spectrum lists."""
        if change.get("name") == "value":
            _rebuild_conditions_options()
            _build_spectrum_options()

    def _on_cond_min(change):
        """Minimal-mode observer: when Conditions changes, refresh Spectrum list."""
        if change.get("name") == "value":
            _build_spectrum_options()

    def _on_inc_min(change):
        """Minimal-mode observer: when Include-bad toggles, refresh Spectrum list."""
        if change.get("name") == "value":
            _build_spectrum_options()

    def _on_base_min(change):
        """Minimal-mode observer: sync baseline function selection; manual builds later upon selection."""
        if change.get("name") == "value":
            nonlocal baseline_function
            try:
                baseline_function = str(change.get("new")).upper()
            except Exception:
                baseline_function = "ARPLS"
            # For MANUAL, defer anchor point UI until a spectrum is chosen.

    def _on_spec_min(change):
        """Minimal-mode observer: on Spectrum select, build the full parameter or manual UI in place."""
        if change.get("name") == "value" and change.get("new") is not None:
            # A spectrum has been chosen; build full UI in-place without recursive re-entry
            try:
                sel_idx = change.get("new")
                nonlocal selected_row, x_values, y_values, material, current_idx_bc
                selected_row = FTIR_DataFrame.loc[sel_idx]
                current_idx_bc = sel_idx
                material = selected_row.get("Material", material)
                x_values = (
                    ast.literal_eval(selected_row["X-Axis"])
                    if isinstance(selected_row["X-Axis"], str)
                    else selected_row["X-Axis"]
                )
                y_values = (
                    ast.literal_eval(selected_row["Raw Data"])
                    if isinstance(selected_row["Raw Data"], str)
                    else selected_row["Raw Data"]
                )
                y_values = np.array(y_values, dtype=float)
                try:
                    _set_session_selection(
                        material=selected_row.get("Material"),
                        conditions=selected_row.get("Conditions"),
                        time=selected_row.get("Time"),
                    )
                except Exception:
                    pass
            except Exception:
                return
            # Build integrated MANUAL mode or parameter UI depending on selection
            # Ensure baseline_function is a valid string in sync with dropdown (fresh-kernel safety)
            try:
                nonlocal baseline_function
            except Exception:
                pass
            try:
                baseline_function = (
                    str(baseline_function).upper()
                    if isinstance(baseline_function, str)
                    else str(baseline_dd.value).upper()
                )
                if baseline_function not in ("ARPLS", "IRSQR", "FABC", "MANUAL"):
                    baseline_function = str(baseline_dd.value).upper()
            except Exception:
                baseline_function = "ARPLS"

            def _build_manual_ui():
                """Build or rebuild the inline manual baseline UI, reusing the existing container.

                Replaces prior parameter UI without closing the shared parent container to allow
                switching between MANUAL and automated baselines without UI disappearance.
                """
                nonlocal baseline_parameter_widgets, baseline_function
                # Clear parameter widgets (manual mode uses anchor workflow instead)
                baseline_parameter_widgets = {}
                baseline_function = "MANUAL"
                # Detach any automated baseline observer while in manual mode to avoid double firing
                try:
                    baseline_dd.unobserve(_on_base_full, names="value")
                except Exception:
                    pass
                manual_out = widgets.Output()
                anchor_points = []
                baseline_active = False  # Flag: baseline preview active after Continue
                # Buttons
                continue_btn = widgets.Button(
                    description="Continue", button_style="success"
                )
                redo_btn = widgets.Button(
                    description="Undo all", button_style="warning"
                )
                undo_btn = widgets.Button(description="Undo last")
                save_file_btn_m = widgets.Button(
                    description="Save for spectrum", button_style="success"
                )
                save_mat_btn_m = widgets.Button(
                    description="Save for material", button_style="info"
                )
                close_btn_m = widgets.Button(description="Close", button_style="danger")
                # Reusable quality controls
                mark_bad_btn_m, mark_good_btn_m, _refresh_mark_btns_m = (
                    _make_quality_controls(FTIR_DataFrame, lambda: selected_row)
                )

                # Track quality changes for session summary
                def _log_mark_bad_m(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append((selected_row.name, "bad"))
                            _quality_dropdown_handle(
                                "bad",
                                dropdown=spectrum_sel,
                                include_bad_flag=include_bad_cb.value,
                                idx=selected_row.name,
                                label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                                observer_fn=_on_spec_m,
                            )
                    except Exception:
                        pass

                def _log_mark_good_m(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append((selected_row.name, "good"))
                            _quality_dropdown_handle(
                                "good",
                                dropdown=spectrum_sel,
                                include_bad_flag=include_bad_cb.value,
                                idx=selected_row.name,
                                label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                                observer_fn=_on_spec_m,
                            )
                    except Exception:
                        pass

                try:
                    mark_bad_btn_m.on_click(_log_mark_bad_m)
                    mark_good_btn_m.on_click(_log_mark_good_m)
                except Exception:
                    pass
                # Figures
                fig_m = go.FigureWidget()
                fig_m.add_scatter(
                    x=np.asarray(x_values, dtype=float),
                    y=np.asarray(y_values, dtype=float),
                    mode="lines",
                    name="Spectrum",
                    line=dict(color="black"),
                )
                fig_m.add_scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    name="Anchor Points",
                    marker=dict(color="red", size=10),
                )
                fig_m.update_layout(
                    title="Manual Baseline: click to add anchor points",
                    xaxis_title="Wavenumber (cm⁻¹)",
                    yaxis_title="Absorbance (AU)",
                    height=450,
                )
                # Prevent autoscaling on subsequent baseline updates by fixing initial ranges
                try:
                    _x_min = float(np.min(x_values))
                    _x_max = float(np.max(x_values))
                    _y_min = float(np.min(y_values))
                    _y_max = float(np.max(y_values))
                    fig_m.update_xaxes(range=[_x_min, _x_max], autorange=False)
                    fig_m.update_yaxes(range=[_y_min, _y_max], autorange=False)
                except Exception:
                    pass
                fig_corr = go.FigureWidget()
                fig_corr.add_scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name="Baseline-Corrected",
                    line=dict(color="blue"),
                )
                fig_corr.update_layout(
                    title="Baseline-Corrected",
                    xaxis_title="Wavenumber (cm⁻¹)",
                    yaxis_title="Absorbance (AU)",
                    height=350,
                )
                # Match corrected figure axes to primary figure to avoid autoscale jumps
                try:
                    fig_corr.update_xaxes(range=[_x_min, _x_max], autorange=False)
                    fig_corr.update_yaxes(range=[_y_min, _y_max], autorange=False)
                except Exception:
                    pass

                # (Handlers wired within helper)

                # --- Colab-safe manual anchor input ---
                # In Colab, Plotly click callbacks are unreliable; provide a
                # Text input for anchor x-values and an Add button instead.
                anchor_input = widgets.Text(
                    placeholder="e.g., 400, 750, 1080",
                    description="Anchors",
                    layout=widgets.Layout(width="50%"),
                )
                add_anchor_btn = widgets.Button(
                    description="Add",
                    button_style="info",
                    tooltip="Add anchor points from the text box",
                )

                def _add_anchors_from_text(_b=None):
                    raw = str(anchor_input.value or "").strip()
                    if not raw:
                        return
                    try:
                        vals = [float(v) for v in raw.split(",") if v.strip()]
                    except Exception:
                        with manual_out:
                            print("Invalid anchor list. Use comma-separated numbers.")
                        return
                    # Merge into anchor_points, keep unique, sorted
                    for v in vals:
                        try:
                            if v not in anchor_points:
                                anchor_points.append(float(v))
                        except Exception:
                            continue
                    ap_sorted = sorted(anchor_points)
                    # Update anchor markers on raw figure
                    try:
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        fig_m.data[1].x = ap_sorted
                        fig_m.data[1].y = [
                            float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                            for ax in ap_sorted
                        ]
                    except Exception:
                        pass
                    # If preview active, recompute
                    if baseline_active and len(anchor_points) >= 2:
                        _preview_baseline()

                add_anchor_btn.on_click(_add_anchors_from_text)

                # Click handler to add anchor point at nearest x
                def _on_click(trace, points, selector):
                    try:
                        if not points.xs:
                            return
                        x_click = float(points.xs[0])
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        idx_near = int(np.nanargmin(np.abs(xs - x_click)))
                        apx = float(xs[idx_near])
                        # Avoid duplicates
                        if apx not in anchor_points:
                            anchor_points.append(apx)
                            ap_sorted = sorted(anchor_points)
                            fig_m.data[1].x = ap_sorted
                            fig_m.data[1].y = [
                                float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                                for ax in ap_sorted
                            ]
                            # If baseline already active, recompute immediately for live update
                            if baseline_active and len(anchor_points) >= 2:
                                _preview_baseline()
                    except Exception:
                        pass

                # Attach click to raw trace only outside Colab
                try:
                    if not _IN_COLAB:
                        fig_m.data[0].on_click(_on_click)
                except Exception:
                    pass

                # Helper: compute preview baseline and show
                def _preview_baseline():
                    with manual_out:
                        try:
                            clear_output(wait=True)
                        except Exception:
                            pass
                    # Capture current axis ranges to preserve user zoom
                    x_range_main = y_range_main = x_range_corr = y_range_corr = None
                    try:
                        if (
                            fig_m.layout.xaxis.autorange is not True
                            and fig_m.layout.xaxis.range
                        ):
                            x_range_main = list(fig_m.layout.xaxis.range)
                        if (
                            fig_m.layout.yaxis.autorange is not True
                            and fig_m.layout.yaxis.range
                        ):
                            y_range_main = list(fig_m.layout.yaxis.range)
                    except Exception:
                        pass
                    try:
                        if (
                            fig_corr.layout.xaxis.autorange is not True
                            and fig_corr.layout.xaxis.range
                        ):
                            x_range_corr = list(fig_corr.layout.xaxis.range)
                        if (
                            fig_corr.layout.yaxis.autorange is not True
                            and fig_corr.layout.yaxis.range
                        ):
                            y_range_corr = list(fig_corr.layout.yaxis.range)
                    except Exception:
                        pass
                    xs = np.asarray(x_values, dtype=float)
                    ys = np.asarray(y_values, dtype=float)
                    if len(anchor_points) < 2:
                        with manual_out:
                            print("Select at least two anchor points to preview.")
                        return
                    ap_sorted = np.array(sorted(anchor_points), dtype=float)
                    y_anchor = np.array(
                        [ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted],
                        dtype=float,
                    )
                    try:
                        spline = CubicSpline(
                            ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0))
                        )
                    except Exception:
                        spline = CubicSpline(ap_sorted, y_anchor)
                    baseline_vals = spline(xs)
                    corrected = ys - baseline_vals
                    # Update raw/baseline figure (keep 0: raw, 1: anchors)
                    while len(fig_m.data) > 2:
                        fig_m.data = tuple(fig_m.data[:2])
                    fig_m.add_scatter(
                        x=xs,
                        y=baseline_vals,
                        mode="lines",
                        name="Baseline",
                        line=dict(color="red", width=1.5, dash="dash"),
                    )
                    # Update corrected figure (single trace)
                    try:
                        fig_corr.data[0].x = xs
                        fig_corr.data[0].y = corrected
                    except Exception:
                        pass
                    # Reapply previous axis ranges to avoid automatic rescaling
                    try:
                        if x_range_main:
                            fig_m.update_xaxes(range=x_range_main, autorange=False)
                        if y_range_main:
                            fig_m.update_yaxes(range=y_range_main, autorange=False)
                    except Exception:
                        pass
                    try:
                        if x_range_corr:
                            fig_corr.update_xaxes(range=x_range_corr, autorange=False)
                        if y_range_corr:
                            fig_corr.update_yaxes(range=y_range_corr, autorange=False)
                    except Exception:
                        pass

                def _continue(_b=None):
                    nonlocal baseline_active
                    baseline_active = True
                    # Hide Continue after first activation
                    try:
                        continue_btn.layout.display = "none"
                    except Exception:
                        pass
                    _preview_baseline()

                def _redo(_b=None):
                    # Rebuild manual UI inline without closing shared container
                    try:
                        fig_m.close()
                    except Exception:
                        pass
                    try:
                        fig_corr.close()
                    except Exception:
                        pass
                    _build_manual_ui()

                def _undo(_b=None):
                    # Remove the most recently added anchor point and update plots
                    try:
                        if not anchor_points:
                            return
                        # Capture axis ranges to preserve user zoom
                        x_range_main = y_range_main = x_range_corr = y_range_corr = None
                        try:
                            if (
                                fig_m.layout.xaxis.autorange is not True
                                and fig_m.layout.xaxis.range
                            ):
                                x_range_main = list(fig_m.layout.xaxis.range)
                            if (
                                fig_m.layout.yaxis.autorange is not True
                                and fig_m.layout.yaxis.range
                            ):
                                y_range_main = list(fig_m.layout.yaxis.range)
                        except Exception:
                            pass
                        try:
                            if (
                                fig_corr.layout.xaxis.autorange is not True
                                and fig_corr.layout.xaxis.range
                            ):
                                x_range_corr = list(fig_corr.layout.xaxis.range)
                            if (
                                fig_corr.layout.yaxis.autorange is not True
                                and fig_corr.layout.yaxis.range
                            ):
                                y_range_corr = list(fig_corr.layout.yaxis.range)
                        except Exception:
                            pass
                        # Pop last added point (reverse chronological)
                        last = anchor_points.pop()
                        xs = np.asarray(x_values, dtype=float)
                        ys = np.asarray(y_values, dtype=float)
                        ap_sorted = sorted(anchor_points)
                        # Update anchor markers
                        try:
                            fig_m.data[1].x = ap_sorted
                            fig_m.data[1].y = [
                                float(ys[int(np.nanargmin(np.abs(xs - ax)))])
                                for ax in ap_sorted
                            ]
                        except Exception:
                            pass
                        # Recompute or clear baseline/corrected
                        if baseline_active and len(anchor_points) >= 2:
                            _preview_baseline()
                        else:
                            # Clear baseline preview and corrected plot if insufficient points
                            try:
                                while len(fig_m.data) > 2:
                                    fig_m.data = tuple(fig_m.data[:2])
                                fig_corr.data[0].x = []
                                fig_corr.data[0].y = []
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Reapply stored ranges
                    try:
                        if x_range_main:
                            fig_m.update_xaxes(range=x_range_main, autorange=False)
                        if y_range_main:
                            fig_m.update_yaxes(range=y_range_main, autorange=False)
                    except Exception:
                        pass
                    try:
                        if x_range_corr:
                            fig_corr.update_xaxes(range=x_range_corr, autorange=False)
                        if y_range_corr:
                            fig_corr.update_yaxes(range=y_range_corr, autorange=False)
                    except Exception:
                        pass

                def _save_file(_b=None):
                    if selected_row is None or len(anchor_points) < 2:
                        with manual_out:
                            print("Need at least two anchor points to save.")
                        return
                    xs = np.asarray(x_values, dtype=float)
                    ys = np.asarray(y_values, dtype=float)
                    ap_sorted = np.array(sorted(anchor_points), dtype=float)
                    y_anchor = np.array(
                        [ys[int(np.nanargmin(np.abs(xs - ap)))] for ap in ap_sorted],
                        dtype=float,
                    )
                    try:
                        spline = CubicSpline(
                            ap_sorted, y_anchor, bc_type=((1, 0.0), (1, 0.0))
                        )
                    except Exception:
                        spline = CubicSpline(ap_sorted, y_anchor)
                    baseline_vals = spline(xs).astype(float)
                    corrected = (ys - baseline_vals).astype(float)
                    # Persist to DataFrame for this file
                    try:
                        FTIR_DataFrame.at[selected_row.name, "Baseline Function"] = (
                            "Manual"
                        )
                        FTIR_DataFrame.at[selected_row.name, "Baseline Parameters"] = (
                            str(
                                {
                                    "anchor_points": [
                                        float(v) for v in ap_sorted.tolist()
                                    ]
                                }
                            )
                        )
                        FTIR_DataFrame.at[selected_row.name, "Baseline"] = (
                            baseline_vals.tolist()
                        )
                        FTIR_DataFrame.at[
                            selected_row.name, "Baseline-Corrected Data"
                        ] = corrected.tolist()
                    except Exception:
                        pass
                    with manual_out:
                        print("Saved manual baseline for this file.")
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault(
                                "saved_file", []
                            ).append((selected_row.name, None))
                            # Also record as baseline-corrected (Manual) with filename for summary output
                            try:
                                fn = FTIR_DataFrame.at[selected_row.name, "File Name"]
                                if not isinstance(fn, str) or not fn.strip():
                                    _loc = FTIR_DataFrame.at[selected_row.name, "File Location"]
                                    fn = os.path.basename(_loc) if isinstance(_loc, str) else str(selected_row.name)
                            except Exception:
                                fn = str(selected_row.name)
                            baseline_session_changes.setdefault(
                                "baseline_corrected_file", []
                            ).append((selected_row.name, "MANUAL", str(fn)))
                    except Exception:
                        pass

                def _save_material(_b=None):
                    """Save manual baseline anchor points and compute baseline/corrected arrays for all rows of the material.

                    For each spectrum of the selected material, we:
                      1. Re-sample anchor point y-values from that spectrum's raw data.
                      2. Fit a cubic spline through those anchor points.
                      3. Store the spline-evaluated baseline and (raw - baseline) in the DataFrame.
                    """
                    if selected_row is None or len(anchor_points) < 2:
                        with manual_out:
                            print("Need at least two anchor points to save.")
                        return
                    mat_val = selected_row.get("Material", material)
                    ap_sorted = sorted(anchor_points)
                    # Ensure destination columns exist and are object dtype (lists per row)
                    try:
                        for _col in ("Baseline", "Baseline-Corrected Data"):
                            if _col not in FTIR_DataFrame.columns:
                                FTIR_DataFrame[_col] = None
                        FTIR_DataFrame["Baseline"] = FTIR_DataFrame["Baseline"].astype(
                            object
                        )
                        FTIR_DataFrame["Baseline-Corrected Data"] = FTIR_DataFrame[
                            "Baseline-Corrected Data"
                        ].astype(object)
                    except Exception:
                        pass
                    # Identify rows for this material
                    try:
                        mat_series = FTIR_DataFrame.get(
                            "Material",
                            pd.Series(index=FTIR_DataFrame.index, dtype=object),
                        ).astype(str)
                        msk = (
                            mat_series.str.strip().str.casefold()
                            == str(mat_val).strip().casefold()
                        )
                        # Coerce to boolean ndarray with same length to avoid index alignment pitfalls
                        msk = pd.Series(msk, index=FTIR_DataFrame.index)
                    except Exception:
                        msk = pd.Series(
                            [False] * len(FTIR_DataFrame), index=FTIR_DataFrame.index
                        )
                    matched_total = int(msk.sum())
                    updated_count = 0
                    skipped_count = 0
                    # Persist function + parameters first
                    try:
                        FTIR_DataFrame.loc[msk, "Baseline Function"] = "Manual"
                        FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str(
                            {"anchor_points": [float(v) for v in ap_sorted]}
                        )
                    except Exception:
                        pass
                    ap_sorted_arr = np.array(ap_sorted, dtype=float)
                    for _ridx in FTIR_DataFrame.loc[msk].index:
                        try:
                            rloc = FTIR_DataFrame.loc[_ridx]
                            xs, ys = _row_xy(rloc)
                            if xs.size == 0:
                                skipped_count += 1
                                continue
                            y_anchor = np.array(
                                [
                                    ys[int(np.nanargmin(np.abs(xs - ap)))]
                                    for ap in ap_sorted_arr
                                ],
                                dtype=float,
                            )
                            try:
                                spline = CubicSpline(
                                    ap_sorted_arr,
                                    y_anchor,
                                    bc_type=((1, 0.0), (1, 0.0)),
                                )
                            except Exception:
                                spline = CubicSpline(ap_sorted_arr, y_anchor)
                            baseline_vals = spline(xs).astype(float)
                            if baseline_vals.size != ys.size:
                                skipped_count += 1
                                continue
                            corrected_vals = (ys - baseline_vals).astype(float)
                            FTIR_DataFrame.at[_ridx, "Baseline"] = (
                                baseline_vals.tolist()
                            )
                            FTIR_DataFrame.at[_ridx, "Baseline-Corrected Data"] = (
                                corrected_vals.tolist()
                            )
                            updated_count += 1
                        except Exception:
                            skipped_count += 1
                            continue
                    with manual_out:
                        print(
                            f"Saved manual baseline (anchors -> arrays) for material '{mat_val}' on {updated_count}/{matched_total} rows (skipped {skipped_count})."
                        )
                    try:
                        baseline_session_changes["saved_filtered"] = (
                            int(baseline_session_changes.get("saved_filtered", 0))
                            + matched_total
                        )
                    except Exception:
                        pass
                    try:
                        baseline_session_changes.setdefault(
                            "baseline_corrected_material", []
                        ).append((mat_val, "Manual", updated_count))
                    except Exception:
                        pass

                def _close_m(_b=None):
                    # Show session summary and collapse UI to summary only
                    try:
                        lines = _session_summary_lines(
                            baseline_session_changes, context="baseline_correct_spectra"
                        )
                        _emit_session_summary(
                            manual_out, lines, title="Session Summary (Baseline Correction)"
                        )
                    except Exception:
                        pass
                    try:
                        # Replace UI with summary output
                        container.children = (widgets.VBox([manual_out]),)
                    except Exception:
                        # Fallback: close figures
                        try:
                            fig_m.close()
                            fig_corr.close()
                        except Exception:
                            pass

                continue_btn.on_click(_continue)
                redo_btn.on_click(_redo)
                undo_btn.on_click(_undo)
                save_file_btn_m.on_click(_save_file)
                save_mat_btn_m.on_click(_save_material)
                close_btn_m.on_click(_close_m)

                # Filter/spectrum observers for manual mode
                def _on_mat_m(change):
                    if change.get("name") == "value":
                        _rebuild_conditions_options()
                        _build_spectrum_options()
                        # Reset selection on filter change
                        try:
                            anchor_points.clear()
                            fig_m.data[1].x = []
                            fig_m.data[1].y = []
                            while len(fig_m.data) > 2:
                                fig_m.data = tuple(fig_m.data[:2])
                            # clear corrected plot
                            fig_corr.data[0].x = []
                            fig_corr.data[0].y = []
                            # Reset preview state and show Continue again
                            nonlocal baseline_active
                            baseline_active = False
                            try:
                                continue_btn.layout.display = ""
                            except Exception:
                                pass
                            _refresh_mark_btns_m()
                        except Exception:
                            pass

                def _on_cond_m(change):
                    if change.get("name") == "value":
                        _build_spectrum_options()
                        try:
                            anchor_points.clear()
                            fig_m.data[1].x = []
                            fig_m.data[1].y = []
                            while len(fig_m.data) > 2:
                                fig_m.data = tuple(fig_m.data[:2])
                            fig_corr.data[0].x = []
                            fig_corr.data[0].y = []
                            nonlocal baseline_active
                            baseline_active = False
                            try:
                                continue_btn.layout.display = ""
                            except Exception:
                                pass
                            _refresh_mark_btns_m()
                        except Exception:
                            pass

                def _on_inc_m(change):
                    if change.get("name") == "value":
                        _build_spectrum_options()

                def _on_spec_m(change):
                    if change.get("name") == "value" and change.get("new") is not None:
                        try:
                            sel_idx3 = change.get("new")
                            r3 = FTIR_DataFrame.loc[sel_idx3]
                            nonlocal selected_row, x_values, y_values, material, current_idx_bc
                            selected_row = r3
                            current_idx_bc = sel_idx3
                            material = r3.get("Material", material)
                            x_values = (
                                ast.literal_eval(r3["X-Axis"])
                                if isinstance(r3["X-Axis"], str)
                                else r3["X-Axis"]
                            )
                            y_values = (
                                ast.literal_eval(r3["Raw Data"])
                                if isinstance(r3["Raw Data"], str)
                                else r3["Raw Data"]
                            )
                            y_arr = np.asarray(y_values, dtype=float)
                            x_arr = np.asarray(x_values, dtype=float)
                            fig_m.data[0].x = x_arr
                            fig_m.data[0].y = y_arr
                            # reset anchors and preview
                            anchor_points.clear()
                            fig_m.data[1].x = []
                            fig_m.data[1].y = []
                            while len(fig_m.data) > 2:
                                fig_m.data = tuple(fig_m.data[:2])
                            fig_corr.data[0].x = []
                            fig_corr.data[0].y = []
                            nonlocal baseline_active
                            baseline_active = False
                            try:
                                continue_btn.layout.display = ""
                            except Exception:
                                pass
                            _refresh_mark_btns_m()
                            _set_session_selection(
                                material=selected_row.get("Material"),
                                conditions=selected_row.get("Conditions"),
                                time=selected_row.get("Time"),
                            )
                        except Exception:
                            pass

                def _on_base_m(change):
                    if change.get("name") == "value":
                        new_b = str(change.get("new")).upper()
                        if new_b == "MANUAL":
                            return  # already in manual mode
                        # Switch to automated baseline inline: detach manual observer, attach full observer
                        try:
                            baseline_dd.unobserve(_on_base_m, names="value")
                        except Exception:
                            pass
                        # Close manual figures
                        try:
                            fig_m.close()
                            fig_corr.close()
                        except Exception:
                            pass
                        # Reattach automated observer if not present
                        try:
                            baseline_dd.observe(_on_base_full, names="value")
                        except Exception:
                            pass
                        # Invoke automated baseline rebuild logic
                        try:
                            _on_base_full({"name": "value", "new": new_b})
                        except Exception:
                            # Fallback: set dropdown value triggers original observer
                            try:
                                baseline_dd.value = new_b
                            except Exception:
                                pass
                        return

                material_dd.observe(_on_mat_m, names="value")
                conditions_dd.observe(_on_cond_m, names="value")
                include_bad_cb.observe(_on_inc_m, names="value")
                spectrum_sel.observe(_on_spec_m, names="value")
                baseline_dd.observe(_on_base_m, names="value")

                # Compose UI
                controls_row_top = widgets.HBox(
                    [material_dd, conditions_dd, baseline_dd]
                )
                spec_row = widgets.HBox([spectrum_sel, include_bad_cb])
                mark_row_m = widgets.HBox([mark_bad_btn_m, mark_good_btn_m])
                # Row for Save/Close actions (outside bordered plot area)
                btn_row_m = widgets.HBox(
                    [save_file_btn_m, save_mat_btn_m, close_btn_m]
                )
                # Split action rows so anchor input can sit between Continue and Redo/Undo
                manual_continue_row = widgets.HBox([continue_btn])
                anchor_row_m = widgets.HBox([anchor_input, add_anchor_btn])
                manual_redo_undo_row = widgets.HBox([redo_btn, undo_btn])
                # Bordered plot + mark section: Continue row, anchor entry, redo/undo, then plots and mark buttons
                bordered_manual = widgets.VBox(
                    [
                        manual_continue_row,
                        anchor_row_m,
                        manual_redo_undo_row,
                        fig_m,
                        fig_corr,
                        mark_row_m,
                    ],
                    layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
                )
                # Assemble full manual UI
                manual_ui = widgets.VBox(
                    [
                        controls_row_top,
                        spec_row,
                        btn_row_m,
                        bordered_manual,
                        manual_out,
                    ]
                )
                try:
                    container.children = (manual_ui,)
                except Exception:
                    display(manual_ui)
                try:
                    _TB_WIDGETS.extend([fig_m, fig_corr])
                except Exception:
                    pass
                _refresh_mark_btns_m()

            # If MANUAL, build manual UI and return
            if baseline_function.upper() == "MANUAL":
                try:
                    _build_manual_ui()
                except Exception as e:
                    with baseline_output_area:
                        try:
                            clear_output(wait=True)
                            print(f"Error building manual UI: {e}")
                        except Exception:
                            pass
                return

            # Otherwise, proceed to rebuild parameter widgets (refresh defaults for selected baseline)
            nonlocal baseline_parameter_widgets
            baseline_parameter_widgets = {}
            parameters_local = _get_default_parameters(baseline_function)
            parameters_local = _cast_parameter_types(
                baseline_function, parameters_local
            )
            if baseline_function.upper() == "ARPLS":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e5),
                    min=1e4,
                    max=1e6,
                    step=1e4,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                    value=parameters_local.get("max_iter", 50),
                    min=1,
                    max=200,
                    step=1,
                    description="Max Iterations",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                    value=parameters_local.get("tol", 1e-3),
                    min=1e-6,
                    max=1e-1,
                    step=1e-4,
                    description="Tolerance",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            elif baseline_function.upper() == "IRSQR":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e6),
                    min=1e5,
                    max=1e7,
                    step=1e5,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
                    value=parameters_local.get("quantile", 0.05),
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    description="Quantile",
                    readout_format=".3f",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
                    value=parameters_local.get("num_knots", 100),
                    min=5,
                    max=500,
                    step=5,
                    description="Knots",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
                    value=parameters_local.get("spline_degree", 3),
                    min=1,
                    max=5,
                    step=1,
                    description="Spline Degree",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                    value=parameters_local.get("diff_order", 3),
                    min=1,
                    max=3,
                    step=1,
                    description="Differential Order",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                    value=parameters_local.get("max_iter", 100),
                    min=1,
                    max=1000,
                    step=1,
                    description="Max Iterations",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                    value=parameters_local.get("tol", 1e-6),
                    min=1e-10,
                    max=1e-2,
                    step=1e-7,
                    description="Tolerance",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            elif baseline_function.upper() == "FABC":
                baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                    value=parameters_local.get("lam", 1e6),
                    min=1e4,
                    max=1e7,
                    step=1e5,
                    description="Smoothness (lam)",
                    readout_format=".1e",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                # scale recompute from selected spectrum
                try:
                    scale_default2 = int(
                        np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                    )
                except Exception:
                    scale_default2 = 50
                scale_val2 = parameters_local.get("scale") or scale_default2
                baseline_parameter_widgets["scale"] = widgets.IntSlider(
                    value=int(scale_val2),
                    min=2,
                    max=500,
                    step=1,
                    description="Scale",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
                    value=parameters_local.get("num_std", 3.0),
                    min=1.5,
                    max=4.5,
                    step=0.1,
                    description="Standard Deviations",
                    readout_format=".2f",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                    value=parameters_local.get("diff_order", 2),
                    min=1,
                    max=3,
                    step=1,
                    description="Differential Order",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
                baseline_parameter_widgets["min_length"] = widgets.IntSlider(
                    value=parameters_local.get("min_length", 2),
                    min=1,
                    max=6,
                    step=1,
                    description="Min Baseline Span Length",
                    continuous_update=False,
                    style={"description_width": "auto"},
                )
            # Build full UI identical to main branch
            defaults_full = _get_default_parameters(baseline_function)
            widget_rows_full = []
            for k, w in baseline_parameter_widgets.items():
                rb = widgets.Button(
                    description="Reset",
                    button_style="info",
                    layout=widgets.Layout(width="70px", margin="0 0 6px 8px"),
                )
                if k == "scale":

                    def _reset_scale2(_b=None, w=w):
                        try:
                            new_def = int(
                                np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                            )
                            w.value = new_def
                        except Exception:
                            pass

                    rb.on_click(_reset_scale2)
                else:
                    rv = defaults_full.get(k, w.value)

                    def make_reset_f(w2, val2):
                        return lambda _b=None: setattr(w2, "value", val2)

                    rb.on_click(make_reset_f(w, rv))
                widget_rows_full.append(widgets.HBox([w, rb]))

            # Add a collapsible 'Parameter Details' toggle below the parameter widgets.
            # Clicking toggles visibility of the explanatory text; button persists for hide/show.
            def _make_param_details_row(name: str):
                name = str(name).upper().strip()
                if name == "ARPLS":
                    txt = (
                        "lam (float): Smoothness parameter (higher = smoother baseline).\n\n"
                        "max_iter (integer): Max number of fit iterations.\n\n"
                        "tol (float): Exit criteria (accuracy goal)."
                    )
                elif name == "IRSQR":
                    txt = (
                        "lam (float): The smoothing parameter (higher = smoother baseline).\n\n"
                        "quantile (float): The quantile at which to fit the baseline (0 < quantile < 1).\n\n"
                        "num_knots (integer): The number of knots for the spline.\n\n"
                        "spline_degree (integer): The degree of the spline.\n\n"
                        "diff_order (integer): The order of the differential matrix. Typical values are 3, 2, or 1.\n\n"
                        "max_iter (integer): The max number of fit iterations.\n\n"
                        "tol (float): Exit criteria (accuracy goal)."
                    )
                elif name == "FABC":
                    txt = (
                        "lam (float): The smoothing parameter (higher = smoother baseline).\n\n"
                        "scale (integer): The scale at which to calculate the continuous wavelet transform. Should be approximately equal to the index-based full-width-at-half-maximum of the peaks or features in the data. Default is None, which will use half of the value from :func:`.optimize_window`, which is not always a good value, but at least scales with the number of data points and gives a starting point for tuning the parameter.\n\n"
                        "num_std (float): The number of standard deviations to include when thresholding. Higher values\n"
                        "will assign more points as baseline.\n\n"
                        "diff_order (integer): The order of the differential matrix. Must be greater than 0. Typical values are 2 or 1.\n\n"
                        "min_length (integer): Any region of consecutive baseline points less than `min_length` is considered to be a false positive and all points in the region are converted to peak points. A higher `min_length` ensures less points are falsely assigned as baseline points. Default is 2, which only removes lone baseline points."
                    )
                else:
                    txt = ""

                toggle = widgets.ToggleButton(
                    value=_TB_PARAM_DETAILS_OPEN,
                    description="Parameter Details",
                    button_style="info",
                    icon="chevron-down",
                )
                details = widgets.HTML(
                    value=f"<pre style='white-space:pre-wrap;margin:0'>{txt}</pre>"
                )
                # Hide details by default
                try:
                    details.layout.display = "" if _TB_PARAM_DETAILS_OPEN else "none"
                except Exception:
                    pass

                def _on_toggle(change):
                    if change.get("name") == "value":
                        show = bool(change.get("new"))
                        global _TB_PARAM_DETAILS_OPEN
                        _TB_PARAM_DETAILS_OPEN = show
                        try:
                            details.layout.display = "" if show else "none"
                        except Exception:
                            pass
                        try:
                            toggle.icon = "chevron-up" if show else "chevron-down"
                        except Exception:
                            pass

                try:
                    toggle.observe(_on_toggle, names="value")
                except Exception:
                    pass
                return widgets.VBox([toggle, details])

            # Append the details row as its own row in the UI
            try:
                widget_rows_full.append(_make_param_details_row(baseline_function))
            except Exception:
                pass
            reset_all_btn2 = widgets.Button(
                description="Reset All",
                button_style="warning",
                # Align with per-parameter Reset button (same vertical spacing) and small left gap
                layout=widgets.Layout(width="90px", margin="0 0 6px 8px"),
            )

            def _reset_all2(_b=None):
                for kk, ww in baseline_parameter_widgets.items():
                    if kk == "scale":
                        try:
                            ww.value = int(
                                np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                            )
                        except Exception:
                            pass
                    elif kk in defaults_full:
                        ww.value = defaults_full[kk]

            reset_all_btn2.on_click(_reset_all2)
            save_file_btn2 = widgets.Button(
                description="Save for spectrum",
                button_style="success",
                layout=widgets.Layout(margin="10px 10px 0 0"),
            )
            save_material_btn2 = widgets.Button(
                description="Save for material",
                button_style="info",
                layout=widgets.Layout(margin="10px 10px 0 0"),
            )
            mark_bad_btn2, mark_good_btn2, _refresh_mark_btns2 = _make_quality_controls(
                FTIR_DataFrame, lambda: selected_row, margin="10px 10px 0 0"
            )
            close_btn2 = widgets.Button(
                description="Close",
                button_style="danger",
                layout=widgets.Layout(margin="10px 0 0 0"),
            )

            # Track quality changes
            def _log_mark_bad2(_b=None):
                try:
                    if selected_row is not None:
                        baseline_session_changes.setdefault("quality", []).append((selected_row.name, "bad"))
                        _quality_dropdown_handle(
                            "bad",
                            dropdown=spectrum_sel,
                            include_bad_flag=include_bad_cb.value,
                            idx=selected_row.name,
                            label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                            observer_fn=_on_spec_full,
                        )
                except Exception:
                    pass

            def _log_mark_good2(_b=None):
                try:
                    if selected_row is not None:
                        baseline_session_changes.setdefault("quality", []).append((selected_row.name, "good"))
                        _quality_dropdown_handle(
                            "good",
                            dropdown=spectrum_sel,
                            include_bad_flag=include_bad_cb.value,
                            idx=selected_row.name,
                            label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition',''))} | T={FTIR_DataFrame.loc[i].get('Time','')} | {FTIR_DataFrame.loc[i].get('File Name','')}",
                            observer_fn=_on_spec_full,
                        )
                except Exception:
                    pass

            try:
                mark_bad_btn2.on_click(_log_mark_bad2)
                mark_good_btn2.on_click(_log_mark_good2)
            except Exception:
                pass

            def _current_params2():
                cur = parameters_local.copy()
                for kk, ww in baseline_parameter_widgets.items():
                    cur[kk] = ww.value
                return _cast_parameter_types(baseline_function, cur)

            def _serialize2(d):
                def to_plain(v):
                    try:
                        if isinstance(v, (np.integer,)):
                            return int(v)
                        if isinstance(v, (np.floating,)):
                            return float(v)
                        if isinstance(v, np.ndarray):
                            return v.tolist()
                    except Exception:
                        pass
                    return v

                return {kk: to_plain(vv) for kk, vv in d.items()}

            def _save_file2(_b=None):
                if selected_row is None:
                    return
                pv = _serialize2(_current_params2())
                FTIR_DataFrame.at[selected_row.name, "Baseline Function"] = (
                    baseline_function.upper()
                )
                FTIR_DataFrame.at[selected_row.name, "Baseline Parameters"] = str(pv)
                # Compute and persist baseline + corrected arrays for this row
                try:
                    # Parse current spectrum arrays
                    xs = (
                        ast.literal_eval(selected_row["X-Axis"])
                        if isinstance(selected_row.get("X-Axis"), str)
                        else selected_row.get("X-Axis")
                    )
                    ys = (
                        ast.literal_eval(selected_row["Raw Data"])
                        if isinstance(selected_row.get("Raw Data"), str)
                        else selected_row.get("Raw Data")
                    )
                    xs = np.asarray(xs, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                    if xs.size and ys.size:
                        # Choose computation path by function
                        params_exec = _cast_parameter_types(
                            baseline_function, pv.copy()
                        )
                        if baseline_function.upper() == "ARPLS":
                            bres = arpls(ys, **params_exec)
                        elif baseline_function.upper() == "IRSQR":
                            bres = irsqr(ys, **params_exec, x_data=xs)
                        elif baseline_function.upper() == "FABC":
                            bres = fabc(ys, **params_exec)
                        else:
                            bres = None
                        if isinstance(bres, tuple):
                            baseline_arr = np.asarray(bres[0], dtype=float)
                        elif isinstance(bres, dict):
                            baseline_arr = np.asarray(bres.get("baseline"), dtype=float)
                        else:
                            baseline_arr = np.asarray(bres, dtype=float)
                        if baseline_arr.size == ys.size:
                            corrected_arr = ys - baseline_arr
                            FTIR_DataFrame.at[selected_row.name, "Baseline"] = (
                                baseline_arr.tolist()
                            )
                            FTIR_DataFrame.at[
                                selected_row.name, "Baseline-Corrected Data"
                            ] = corrected_arr.tolist()
                            try:
                                try:
                                    fn = FTIR_DataFrame.at[
                                        selected_row.name, "File Name"
                                    ]
                                    if not isinstance(fn, str) or not fn.strip():
                                        _loc = FTIR_DataFrame.at[
                                            selected_row.name, "File Location"
                                        ]
                                        fn = os.path.basename(_loc) if isinstance(_loc, str) else str(selected_row.name)
                                except Exception:
                                    fn = str(selected_row.name)
                                baseline_session_changes.setdefault(
                                    "baseline_corrected_file", []
                                ).append((selected_row.name, baseline_function.upper(), str(fn)))
                            except Exception:
                                pass
                except Exception:
                    pass
                with baseline_output_area:
                    print("Saved baseline (parameters + arrays) for this file.")
                try:
                    baseline_session_changes.setdefault("saved_file", []).append(
                        (selected_row.name, None)
                    )
                except Exception:
                    pass

            def _save_material2(_b=None):
                if selected_row is None:
                    return
                pv = _serialize2(_current_params2())
                mat_val = selected_row.get("Material", material)
                try:
                    mat_series2 = FTIR_DataFrame.get(
                        "Material",
                        pd.Series(index=FTIR_DataFrame.index, dtype=object),
                    ).astype(str)
                    msk = (
                        mat_series2.str.strip().str.casefold()
                        == str(mat_val).strip().casefold()
                    )
                    msk = pd.Series(msk, index=FTIR_DataFrame.index)
                except Exception:
                    msk = pd.Series(
                        [False] * len(FTIR_DataFrame), index=FTIR_DataFrame.index
                    )
                # Persist function/parameters
                FTIR_DataFrame.loc[msk, "Baseline Function"] = baseline_function.upper()
                FTIR_DataFrame.loc[msk, "Baseline Parameters"] = str(pv)
                # Ensure destination columns exist and are object dtype (lists per row)
                for _col in ("Baseline", "Baseline-Corrected Data"):
                    if _col not in FTIR_DataFrame.columns:
                        FTIR_DataFrame[_col] = None
                try:
                    FTIR_DataFrame["Baseline"] = FTIR_DataFrame["Baseline"].astype(
                        object
                    )
                    FTIR_DataFrame["Baseline-Corrected Data"] = FTIR_DataFrame[
                        "Baseline-Corrected Data"
                    ].astype(object)
                except Exception:
                    pass
                # Iterate material rows and compute baseline arrays
                matched_total = int(msk.sum())
                updated_count = 0
                skipped_count = 0
                params_exec_global = _cast_parameter_types(baseline_function, pv.copy())
                for _ridx in FTIR_DataFrame.loc[msk].index:
                    try:
                        rloc = FTIR_DataFrame.loc[_ridx]
                        xs, ys = _row_xy(rloc)
                        if xs.size == 0:
                            skipped_count += 1
                            continue
                        # Compute baseline
                        if baseline_function.upper() == "ARPLS":
                            bres = arpls(ys, **params_exec_global)
                        elif baseline_function.upper() == "IRSQR":
                            bres = irsqr(ys, **params_exec_global, x_data=xs)
                        elif baseline_function.upper() == "FABC":
                            bres = fabc(ys, **params_exec_global)
                        else:
                            bres = None
                        if isinstance(bres, tuple):
                            baseline_arr = np.asarray(bres[0], dtype=float)
                        elif isinstance(bres, dict):
                            baseline_arr = np.asarray(bres.get("baseline"), dtype=float)
                        else:
                            baseline_arr = np.asarray(bres, dtype=float)
                        if baseline_arr.size != ys.size:
                            skipped_count += 1
                            continue
                        corrected_arr = ys - baseline_arr
                        FTIR_DataFrame.at[_ridx, "Baseline"] = baseline_arr.tolist()
                        FTIR_DataFrame.at[_ridx, "Baseline-Corrected Data"] = (
                            corrected_arr.tolist()
                        )
                        updated_count += 1
                    except Exception:
                        skipped_count += 1
                        continue
                with baseline_output_area:
                    print(
                        f"Saved baseline (parameters + arrays) for material '{mat_val}' on {updated_count}/{matched_total} rows (skipped {skipped_count})."
                    )
                try:
                    baseline_session_changes["saved_filtered"] = (
                        int(baseline_session_changes.get("saved_filtered", 0))
                        + matched_total
                    )
                except Exception:
                    pass
                try:
                    baseline_session_changes.setdefault(
                        "baseline_corrected_material", []
                    ).append((mat_val, baseline_function.upper(), updated_count))
                except Exception:
                    pass

            save_file_btn2.on_click(_save_file2)
            save_material_btn2.on_click(_save_material2)

            def _close_full2(_b=None):
                # Emit summary and collapse UI to summary output
                try:
                    lines = _session_summary_lines(
                        baseline_session_changes, context="baseline_correct_spectra"
                    )
                    _emit_session_summary(
                        baseline_output_area, lines, title="Session Summary (Baseline Correction)"
                    )
                except Exception:
                    pass
                try:
                    container.children = (baseline_output_area,)
                except Exception:
                    try:
                        plt.close("all")
                    except Exception:
                        pass

            close_btn2.on_click(_close_full2)

            # Observers for filters in full mode
            def _on_mat_full(change):
                """Full-mode observer: Material changed -> rebuild Conditions & Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _rebuild_conditions_options()
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_cond_full(change):
                """Full-mode observer: Conditions changed -> rebuild Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_inc_full(change):
                """Full-mode observer: Include bad spectra toggled -> rebuild Spectrum and refresh plot."""
                if change.get("name") == "value":
                    _build_spectrum_options()
                    _plot_baseline(
                        **{
                            kk: ww.value
                            for kk, ww in baseline_parameter_widgets.items()
                        }
                    )
                    _refresh_mark_btns2()

            def _on_base_full(change):
                """Full-mode observer: Baseline method changed -> rebuild parameter UI or switch to MANUAL."""
                if change.get("name") != "value":
                    return
                new_val = str(change.get("new")).upper()
                nonlocal baseline_function, baseline_parameter_widgets
                if new_val == "MANUAL":
                    baseline_function = new_val
                    _build_manual_ui()
                    return
                # Rebuild the parameter UI inline for the selected baseline without re-entering the function
                baseline_function = new_val
                # Recreate parameter widgets for the new method
                baseline_parameter_widgets = {}
                params_local = _get_default_parameters(baseline_function)
                params_local = _cast_parameter_types(baseline_function, params_local)
                if baseline_function == "ARPLS":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e5),
                        min=1e4,
                        max=1e6,
                        step=1e4,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                        value=params_local.get("max_iter", 50),
                        min=1,
                        max=200,
                        step=1,
                        description="Max Iterations",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                        value=params_local.get("tol", 1e-3),
                        min=1e-6,
                        max=1e-1,
                        step=1e-4,
                        description="Tolerance",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                elif baseline_function == "IRSQR":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e6),
                        min=1e5,
                        max=1e7,
                        step=1e5,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["quantile"] = widgets.FloatSlider(
                        value=params_local.get("quantile", 0.05),
                        min=0.001,
                        max=0.5,
                        step=0.001,
                        description="Quantile",
                        readout_format=".3f",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["num_knots"] = widgets.IntSlider(
                        value=params_local.get("num_knots", 100),
                        min=5,
                        max=500,
                        step=5,
                        description="Knots",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["spline_degree"] = widgets.IntSlider(
                        value=params_local.get("spline_degree", 3),
                        min=1,
                        max=5,
                        step=1,
                        description="Spline Degree",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                        value=params_local.get("diff_order", 3),
                        min=1,
                        max=3,
                        step=1,
                        description="Differential Order",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["max_iter"] = widgets.IntSlider(
                        value=params_local.get("max_iter", 100),
                        min=1,
                        max=1000,
                        step=1,
                        description="Max Iterations",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["tol"] = widgets.FloatSlider(
                        value=params_local.get("tol", 1e-6),
                        min=1e-10,
                        max=1e-2,
                        step=1e-7,
                        description="Tolerance",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                elif baseline_function == "FABC":
                    baseline_parameter_widgets["lam"] = widgets.FloatSlider(
                        value=params_local.get("lam", 1e6),
                        min=1e4,
                        max=1e7,
                        step=1e5,
                        description="Smoothness (lam)",
                        readout_format=".1e",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    try:
                        scale_default3 = int(
                            np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                        )
                    except Exception:
                        scale_default3 = 50
                    scale_val3 = params_local.get("scale") or scale_default3
                    baseline_parameter_widgets["scale"] = widgets.IntSlider(
                        value=int(scale_val3),
                        min=2,
                        max=500,
                        step=1,
                        description="Scale",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["num_std"] = widgets.FloatSlider(
                        value=params_local.get("num_std", 3.0),
                        min=1.5,
                        max=4.5,
                        step=0.1,
                        description="Standard Deviations",
                        readout_format=".2f",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["diff_order"] = widgets.IntSlider(
                        value=params_local.get("diff_order", 2),
                        min=1,
                        max=3,
                        step=1,
                        description="Differential Order",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )
                    baseline_parameter_widgets["min_length"] = widgets.IntSlider(
                        value=params_local.get("min_length", 2),
                        min=1,
                        max=6,
                        step=1,
                        description="Min Baseline Span Length",
                        continuous_update=False,
                        style={"description_width": "auto"},
                    )

                defaults_new = _get_default_parameters(baseline_function)
                rows = []
                for k, w in baseline_parameter_widgets.items():
                    rb = widgets.Button(
                        description="Reset",
                        button_style="info",
                        layout=widgets.Layout(width="70px", margin="0 0 6px 8px"),
                    )
                    if k == "scale":

                        def _reset_scale3(_b=None, w=w):
                            try:
                                w.value = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                            except Exception:
                                pass

                        rb.on_click(_reset_scale3)
                    else:
                        rv = defaults_new.get(k, w.value)

                        def make_reset_f2(w2, val2):
                            return lambda _b=None: setattr(w2, "value", val2)

                        rb.on_click(make_reset_f2(w, rv))
                    rows.append(widgets.HBox([w, rb]))

                # Parameter details toggle
                try:
                    rows.append(_make_param_details_row(baseline_function))
                except Exception:
                    pass

                reset_all_btn3 = widgets.Button(
                    description="Reset All",
                    button_style="warning",
                    layout=widgets.Layout(width="90px", margin="0 0 6px 8px"),
                )

                def _reset_all3(_b=None):
                    for kk, ww in baseline_parameter_widgets.items():
                        if kk == "scale":
                            try:
                                ww.value = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                            except Exception:
                                pass
                        elif kk in defaults_new:
                            ww.value = defaults_new[kk]

                reset_all_btn3.on_click(_reset_all3)

                save_file_btn3 = widgets.Button(
                    description="Save for spectrum",
                    button_style="success",
                    layout=widgets.Layout(margin="10px 10px 0 0"),
                )
                save_material_btn3 = widgets.Button(
                    description="Save for material",
                    button_style="info",
                    layout=widgets.Layout(margin="10px 10px 0 0"),
                )
                mark_bad_btn3, mark_good_btn3, _refresh_mark_btns3 = (
                    _make_quality_controls(
                        FTIR_DataFrame, lambda: selected_row, margin="10px 10px 0 0"
                    )
                )
                close_btn3 = widgets.Button(
                    description="Close",
                    button_style="danger",
                    layout=widgets.Layout(margin="10px 0 0 0"),
                )

                # Track quality changes for session summary
                def _log_mark_bad3(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append(
                                (selected_row.name, "bad")
                            )
                        if selected_row is not None and not include_bad_cb.value:
                            try:
                                spectrum_sel.unobserve(_on_spec_full, names="value")
                                spectrum_sel.options = [opt for opt in spectrum_sel.options if opt[1] != selected_row.name]
                                spectrum_sel.value = None if spectrum_sel.options else None
                                spectrum_sel.observe(_on_spec_full, names="value")
                            except Exception:
                                pass
                    except Exception:
                        pass

                def _log_mark_good3(_b=None):
                    try:
                        if selected_row is not None:
                            baseline_session_changes.setdefault("quality", []).append(
                                (selected_row.name, "good")
                            )
                        if selected_row is not None and not include_bad_cb.value:
                            try:
                                ids = [v for (_l,v) in spectrum_sel.options]
                                if selected_row.name not in ids:
                                    spectrum_sel.unobserve(_on_spec_full, names="value")
                                    _mat = selected_row.get('Material','')
                                    _cond = selected_row.get('Conditions', selected_row.get('Condition',''))
                                    _t = selected_row.get('Time','')
                                    _fn = selected_row.get('File Name','')
                                    _label = f"{_mat} | {_cond} | T={_t} | {_fn}"
                                    spectrum_sel.options = spectrum_sel.options + [(_label, selected_row.name)] if spectrum_sel.options else [(_label, selected_row.name)]
                                    spectrum_sel.value = selected_row.name
                                    spectrum_sel.observe(_on_spec_full, names="value")
                            except Exception:
                                pass
                    except Exception:
                        pass

                try:
                    mark_bad_btn3.on_click(_log_mark_bad3)
                    mark_good_btn3.on_click(_log_mark_good3)
                except Exception:
                    pass

                # Wire save/close handlers (quality handled by helper)
                save_file_btn3.on_click(_save_file2)
                save_material_btn3.on_click(_save_material2)
                close_btn3.on_click(_close_full2)

                mark_row3 = widgets.HBox([mark_bad_btn3, mark_good_btn3])
                # Attach reset-all (rebuild path) to bottom-most parameter row
                try:
                    for _j in range(len(rows) - 1, -1, -1):
                        _row = rows[_j]
                        if isinstance(_row, widgets.HBox) and hasattr(_row, "children") and len(_row.children) == 2:
                            rows[_j] = widgets.HBox(list(_row.children) + [reset_all_btn3])
                            break
                except Exception:
                    pass
                footer3 = widgets.HBox(
                    [save_file_btn3, save_material_btn3, close_btn3]
                )
                # Bordered parameter cluster (rebuild path) for clarity
                try:
                    param_cluster_new = widgets.VBox(
                        rows,
                        layout=widgets.Layout(
                            border="1px solid #aaa",
                            padding="6px",
                            margin="6px 0",
                        ),
                    )
                except Exception:
                    param_cluster_new = widgets.VBox(rows)
                ui_full_new = widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                material_dd,
                                conditions_dd,
                                baseline_dd,
                            ]
                        ),
                        widgets.HBox([spectrum_sel, include_bad_cb]),
                        param_cluster_new,
                        mark_row3,
                        footer3,
                    ]
                )

                # Swap UI inline
                try:
                    container.children = (ui_full_new, baseline_output_area)
                except Exception:
                    display(widgets.VBox([ui_full_new, baseline_output_area]))

                # Observe parameter changes for live updates
                def _on_param_change3(change):
                    """Full-mode: parameter widget changed -> update baseline preview with current values."""
                    if change.get("name") == "value":
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )

                for _pw in baseline_parameter_widgets.values():
                    try:
                        _pw.observe(_on_param_change3, names="value")
                    except Exception:
                        pass

                # Refresh mark buttons and plot with new method
                try:
                    _refresh_mark_btns3()
                except Exception:
                    pass
                _plot_baseline(
                    **{kk: ww.value for kk, ww in baseline_parameter_widgets.items()}
                )

            material_dd.observe(_on_mat_full, names="value")
            conditions_dd.observe(_on_cond_full, names="value")
            include_bad_cb.observe(_on_inc_full, names="value")
            baseline_dd.observe(_on_base_full, names="value")

            def _on_spec_full(change):
                """Full-mode observer: Spectrum changed -> update selection, recompute scale (FABC), refresh plot."""
                if change.get("name") == "value" and change.get("new") is not None:
                    try:
                        sel_idx2 = change.get("new")
                        rsel2 = FTIR_DataFrame.loc[sel_idx2]
                        nonlocal selected_row, x_values, y_values, material, current_idx_bc
                        selected_row = rsel2
                        current_idx_bc = sel_idx2
                        material = rsel2.get("Material", material)
                        x_values = (
                            ast.literal_eval(rsel2["X-Axis"])
                            if isinstance(rsel2["X-Axis"], str)
                            else rsel2["X-Axis"]
                        )
                        y_values = (
                            ast.literal_eval(rsel2["Raw Data"])
                            if isinstance(rsel2["Raw Data"], str)
                            else rsel2["Raw Data"]
                        )
                        y_values = np.array(y_values, dtype=float)
                        # If currently using MANUAL baseline, switch into the integrated inline manual UI
                        if baseline_function.upper() == "MANUAL":
                            _build_manual_ui()
                            return
                        if (
                            baseline_function.upper() == "FABC"
                            and "scale" in baseline_parameter_widgets
                        ):
                            try:
                                new_scale3 = int(
                                    np.clip(ceil(optimize_window(y_values) / 2), 2, 500)
                                )
                                baseline_parameter_widgets["scale"].value = new_scale3
                            except Exception:
                                pass
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )
                        _refresh_mark_btns2()
                        _set_session_selection(
                            material=selected_row.get("Material"),
                            conditions=selected_row.get("Conditions"),
                            time=selected_row.get("Time"),
                        )
                    except Exception:
                        pass

            spectrum_sel.observe(_on_spec_full, names="value")
            # Build final UI inside existing container (replace children to avoid flicker/disappearance)
            if baseline_function.upper() != "MANUAL":
                mark_row2 = widgets.HBox([mark_bad_btn2, mark_good_btn2])
                # Attach reset-all button to the bottom-most parameter row (after its individual Reset)
                try:
                    for _j in range(len(widget_rows_full) - 1, -1, -1):
                        _row = widget_rows_full[_j]
                        if isinstance(_row, widgets.HBox) and hasattr(_row, "children") and len(_row.children) == 2:
                            # Append the global reset button to this HBox
                            widget_rows_full[_j] = widgets.HBox(list(_row.children) + [reset_all_btn2])
                            break
                except Exception:
                    pass
                controls_footer2 = widgets.HBox(
                    [save_file_btn2, save_material_btn2, close_btn2]
                )
                # Build UI with bordered plot+mark section below parameter controls
                # Bordered parameter cluster for clarity (individual rows + details + global reset)
                try:
                    param_cluster_full = widgets.VBox(
                        widget_rows_full,
                        layout=widgets.Layout(
                            border="1px solid #aaa",
                            padding="6px",
                            margin="6px 0",
                        ),
                    )
                except Exception:
                    param_cluster_full = widgets.VBox(widget_rows_full)
                ui_controls_full = widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                material_dd,
                                conditions_dd,
                                baseline_dd,
                            ]
                        ),
                        widgets.HBox([spectrum_sel, include_bad_cb]),
                        param_cluster_full,
                        controls_footer2,
                    ]
                )
                plot_and_mark_full = widgets.VBox(
                    [baseline_output_area, mark_row2],
                    layout=widgets.Layout(
                        border="1px solid #ccc",
                        padding="8px",
                        margin="6px 0",
                    ),
                )
                try:
                    container.children = (ui_controls_full, plot_and_mark_full)
                except Exception:
                    display(widgets.VBox([ui_controls_full, plot_and_mark_full]))
                _refresh_mark_btns2()

                # Parameter slider -> live plot updates
                def _on_param_change(change):
                    """Full-mode: parameter widget changed -> update baseline preview with current values."""
                    if change.get("name") == "value":
                        _plot_baseline(
                            **{
                                kk: ww.value
                                for kk, ww in baseline_parameter_widgets.items()
                            }
                        )

                for _pw in baseline_parameter_widgets.values():
                    try:
                        _pw.observe(_on_param_change, names="value")
                    except Exception:
                        pass
                # Initial plot
                _plot_baseline(
                    **{kk: ww.value for kk, ww in baseline_parameter_widgets.items()}
                )
            else:
                # If MANUAL, delegate to manual UI builder
                _build_manual_ui()

    def _on_close_min(_b=None):
        """Close minimal UI and emit a concise per-session summary (runs regardless of spectrum selection)."""
        try:
            lines = _session_summary_lines(
                baseline_session_changes, context="baseline_correct_spectra"
            )
            _emit_session_summary(
                baseline_output_area, lines, title="Session Summary (Baseline Correction)"
            )
        except Exception:
            pass
        try:
            container.children = (baseline_output_area,)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    # Attach minimal-mode observers immediately so dropdowns work before spectrum selection
    material_dd.observe(_on_mat_min, names="value")
    conditions_dd.observe(_on_cond_min, names="value")
    include_bad_cb.observe(_on_inc_min, names="value")
    baseline_dd.observe(_on_base_min, names="value")
    spectrum_sel.observe(_on_spec_min, names="value")
    close_btn.on_click(_on_close_min)

    return FTIR_DataFrame


def populate_material_dictionary(
    FTIR_DataFrame,
    materials_json_path=None,
):
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """
    Populate materials.json with materials, conditions, and time values from FTIR_DataFrame.

    Parameters
    ------
    FTIR_DataFrame : pd.DataFrame
        In-memory FTIR dataset containing at least the columns 'Material',
        'Conditions', and 'Time'.
    materials_json_path : str | None
        Path to the materials.json file. Defaults to Trenton_Project/materials.json
        alongside this module.

    Behavior
    --------
    - Preserves existing M000 metadata block as-is.
        - Adds or updates entries (M001, M002, ...) for each unique material in the
            DataFrame, creating a minimal structure compatible with materials_backup.json:
                {
                    "name": material,
                    "alias": material,
                    "peaks": {
                        "1": {
                            "name": "",
                            "center_wavenumber": 0,
                            "σ": 0,
                            "α": 0
                        }
                    }
                }
    """
    # Validate required columns early for clearer errors
    _require_columns(
        FTIR_DataFrame,
        ["Material", "Conditions", "Time"],
        context="FTIR_DataFrame (populate_material_dictionary)",
    )

    # Normalize DataFrame copy used throughout this helper
    df = FTIR_DataFrame.copy()
    df = df[~df["Material"].isna() & ~df["Conditions"].isna()]
    try:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    except Exception:
        pass
    material_names = sorted(df["Material"].dropna().astype(str).unique())

    # Resolve default path relative to this file (or treat input as directory)
    if materials_json_path is None:
        base_dir = os.path.dirname(__file__)
        materials_json_path = os.path.join(base_dir, "materials.json")
    else:
        materials_json_path = str(materials_json_path)
        if os.path.isdir(materials_json_path):
            materials_json_path = os.path.join(materials_json_path, "materials.json")

    target_dir = os.path.dirname(materials_json_path)
    if target_dir and not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    def _build_initial_materials_structure(material_list, *, metadata_block=None):
        metadata_block = {
            "metadata": {
                "created": "9-4-2025",
                "author": "Trenton Wells",
                "organization": "National Laboratory of the Rockies",
                "NLR_contact": "trenton.wells@nrel.gov",
                "personal_contact": "trentonwells73@gmail.com",
            },
            "terms": {
                "name": "name of the material or peak",
                "alias": "shorthand name for the material",
                "σ": "shape factor",
                "α": {
                    "definition": "fractional Gauss character of the compound peak shape",
                },
            },
            "notes": {
                "1": "All wavenumbers are in cm⁻¹",
                "2": "This file will be populated in the 'Create JSON File' cell of the Main.ipynb",
                "3": "It is suggested to rename each material to be descriptive. Later functions will look for 'alias', so manually editing 'name' is acceptable",
                "4": "Peak models based on Pseudo-Voigt functions",
            },
        } if metadata_block is None else metadata_block
        top_dict = {"M000": metadata_block}
        for idx, mat_name in enumerate(material_list, start=1):
            code = f"M{idx:03d}"
            top_dict[code] = {
                "name": mat_name,
                "alias": mat_name,
                "peaks": {
                    "1": {
                        "name": "",
                        "center_wavenumber": 0,
                        "σ": 0,
                        "α": 0,
                    }
                },
            }
        return top_dict

    if os.path.exists(materials_json_path):
        try:
            with open(materials_json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception:
            content = [_build_initial_materials_structure(material_names)]
    else:
        content = [_build_initial_materials_structure(material_names)]

    if not isinstance(content, list) or not content:
        # Normalize to expected shape
        content = [content if isinstance(content, dict) else {}]
    top = content[0]
    if not isinstance(top, dict):
        top = {}

    # Ensure metadata block present
    if "M000" not in top or not isinstance(top.get("M000"), dict):
        top["M000"] = _build_initial_materials_structure([])["M000"]

    def _next_material_code(existing_keys):
        nums = [
            int(k[1:])
            for k in existing_keys
            if isinstance(k, str) and len(k) == 4 and k.startswith("M") and k[1:].isdigit()
        ]
        nxt = max(nums) + 1 if nums else 1
        return f"M{nxt:03d}"

    existing_materials = set()
    for code_key, payload in top.items():
        if not isinstance(payload, dict) or code_key == "M000":
            continue
        alias = payload.get("alias")
        name = payload.get("name")
        if alias is not None:
            existing_materials.add(str(alias))
        if name is not None:
            existing_materials.add(str(name))

    for material in material_names:
        if material in existing_materials:
            continue
        new_code = _next_material_code(top.keys())
        top[new_code] = {
            "name": material,
            "alias": material,
            "peaks": {
                "1": {
                    "name": "",
                    "center_wavenumber": 0,
                    "σ": 0,
                    "α": 0,
                }
            },
        }
        existing_materials.add(material)

    content[0] = top

    # Write back to file with pretty formatting
    with open(materials_json_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

    print(f"materials.json updated at: {materials_json_path}")


def bring_in_DataFrame(DataFrame_path=None):
    """
    Load the CSV file into a pandas DataFrame.

    Allows for easy DataFrame manipulation in memory over the course of the analysis.

    Parameters
    ----------
    DataFrame_path : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if DataFrame_path is None:
        DataFrame_path = "FTIR_DataFrame.csv"  # Default path if none is provided (will
        # be in active directory)
    else:
        pass
    if os.path.exists(DataFrame_path):
        FTIR_DataFrame = pd.read_csv(
            DataFrame_path
        )  # Load the DataFrame from the specified path
    else:
        FTIR_DataFrame = (
            pd.DataFrame()
        )  # Create a new empty DataFrame if it doesn't exist
    return FTIR_DataFrame, DataFrame_path


def normalize_spectra(FTIR_DataFrame, filepath=None):
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """
    Interactively select and save a normalization peak range for FTIR spectra.

    Plots either a predefined specific file (via filepath) or the first time-zero
    file for a specified material. The user selects two points on the plot to define
    an x-range (wavenumber window) for normalization. The selected range is printed
    and can be saved to the DataFrame column 'Normalization Peak Wavenumber'.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing the spectral data.
    filepath : str, optional
        If provided, preselect this file (full path or just filename).

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the selected normalization peak range saved.
    """

    clear_output(wait=True)

    # Ensure destination column exists
    target_col = "Normalization Peak Wavenumber"
    if target_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[target_col] = None

    # Identify condition column name
    cond_col = _conditions_column_name(FTIR_DataFrame)

    # Build dropdown options
    try:
        df_all = FTIR_DataFrame.copy()
        # Exclude rows marked as bad quality
        try:
            df_all = df_all[_quality_good_mask(df_all)]
        except Exception:
            pass
    except Exception:
        df_all = FTIR_DataFrame

    materials, conditions = _extract_material_condition_lists(
        df_all, exclude_unexposed=True
    )

    # Preselect by filepath/material if provided
    preselect_idx = None
    preselect_material = None
    preselect_condition = None
    if filepath is not None:
        if os.path.sep in filepath:
            folder, fname = os.path.split(filepath)
            flt = df_all[
                (df_all.get("File Location", "") == folder)
                & (df_all.get("File Name", "") == fname)
            ]
        else:
            flt = df_all[df_all.get("File Name", "") == filepath]
        if not flt.empty:
            preselect_idx = flt.index[0]
            preselect_material = str(flt.iloc[0].get("Material", "any"))
            if cond_col is not None:
                preselect_condition = str(flt.iloc[0].get(cond_col, "any"))
    # No material argument; selection is made via the dropdowns

    # Helper: perform normalization for a material
    def _normalize(material_name: str):
        if FTIR_DataFrame is None:
            raise ValueError("FTIR_DataFrame is None.")
        if material_name is None or str(material_name).strip() == "":
            raise ValueError("Material must be provided for normalization.")

        source_column = "Baseline-Corrected Data"
        dest_column = "Normalized and Corrected Data"
        range_column = "Normalization Peak Wavenumber"

        subset = FTIR_DataFrame[FTIR_DataFrame["Material"] == material_name]
        if subset.empty:
            raise ValueError(f"No rows found for material '{material_name}'.")

        # Determine x-axis column
        x_axis_column = "X-Axis" if "X-Axis" in FTIR_DataFrame.columns else None
        if x_axis_column is None:
            raise KeyError(
                "Could not find an x-axis column ('X-Axis') in the DataFrame."
            )
        # Validate required columns for normalization
        _require_columns(
            FTIR_DataFrame,
            ["Material", source_column, range_column, x_axis_column],
            context="FTIR_DataFrame (_normalize)",
        )

        # Ensure source/destination columns are object dtype (hold per-row lists)
        if source_column in FTIR_DataFrame.columns:
            try:
                FTIR_DataFrame[source_column] = FTIR_DataFrame[source_column].astype(
                    object
                )
            except Exception:
                pass
        # Ensure destination column exists and is object dtype
        if dest_column not in FTIR_DataFrame.columns:
            FTIR_DataFrame[dest_column] = None
        try:
            FTIR_DataFrame[dest_column] = FTIR_DataFrame[dest_column].astype(object)
        except Exception:
            pass

        # Normalize each spectrum by its own max within the normalization window
        updated = 0
        skipped = 0
        errors = []
        for idx, row in subset.iterrows():
            try:
                x = row.get(x_axis_column)
                y = row.get(source_column)
                rng = row.get(range_column)
                # parse potential string-literals
                if isinstance(x, str):
                    x = ast.literal_eval(x)
                if isinstance(y, str):
                    y = ast.literal_eval(y)
                if isinstance(rng, str):
                    rng = ast.literal_eval(rng)
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    skipped += 1
                    continue
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                lo, hi = float(min(rng)), float(max(rng))
                if x.size == 0 or y.size == 0 or x.size != y.size:
                    skipped += 1
                    continue
                # window mask
                mask = (x >= lo) & (x <= hi)
                if not np.any(mask):
                    skipped += 1
                    continue
                local_max = np.nanmax(y[mask])
                if not np.isfinite(local_max) or local_max == 0:
                    skipped += 1
                    continue
                FTIR_DataFrame.at[idx, dest_column] = (y / local_max).tolist()
                updated += 1
            except Exception as e:
                skipped += 1
                errors.append((idx, str(e)))

        print(
            f"Normalized material '{material_name}': updated {updated} spectra; skipped "
            f"{skipped} (missing/invalid range or data). Each spectrum scaled by its own peak."
        )
        if updated == 0 and skipped > 0 and errors:
            # Provide a small hint in the console; keep UI uncluttered
            print(
                "Note: Some rows lacked valid ranges or data for normalization. "
                f"First error: {errors[0][1]}"
            )

    # Widgets: Material/Conditions/Spectrum
    material_dd = widgets.Dropdown(
        options=["any"] + materials,
        value=(preselect_material if preselect_material in materials else "any"),
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=(["any"] + conditions if cond_col else ["any"]),
        value=(preselect_condition if (preselect_condition in conditions) else "any"),
        description=("Conditions" if cond_col else "Conditions"),
        layout=widgets.Layout(width="40%"),
    )
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    # New user-facing display mode toggle: Single vs Time-series
    display_mode = widgets.ToggleButtons(
        options=[("Single spectrum", "single"), ("Time-series", "series")],
        value="single",
        description="Display",
        layout=widgets.Layout(width="40%"),
    )
    spectrum_sel = widgets.Dropdown(
        options=[], description="Spectrum", layout=widgets.Layout(width="70%")
    )

    # Apply session defaults for material/conditions dropdowns before building spectra list
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    info_out = widgets.Output()
    msg_out = widgets.Output()
    # Track session changes and messages for a shared summary
    session_lines = []
    session_changes = {}

    # State shared with callbacks
    selected_points = []  # up to two x positions
    x_data = []
    y_data = []
    # Track currently displayed spectrum independently of dropdown so a bad-marked spectrum can remain visible
    current_idx = None

    # --- Plot ---
    # Initialize figure (name will be updated per selection depending on data used)
    fig = go.FigureWidget(data=[go.Scatter(x=[], y=[], mode="lines", name="Spectrum")])
    fig.update_layout(
        title="Select Normalization Range",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _clear_selection_visuals():
        fig.layout.shapes = ()

    def _current_y_bounds():
        """Return (ymin, ymax) for selection visuals.

        In time-series mode, compute from all plotted traces; otherwise use current y_data.
        """
        try:
            if display_mode.value == "series" and len(fig.data) > 0:
                ys = []
                for tr in fig.data:
                    try:
                        ys.extend([float(v) for v in tr.y if v is not None and np.isfinite(v)])
                    except Exception:
                        pass
                if ys:
                    return float(np.nanmin(ys)), float(np.nanmax(ys))
        except Exception:
            pass
        # Fallback to current single-spectrum data bounds
        y0min = float(np.nanmin(y_data)) if len(y_data) else 0.0
        y0max = float(np.nanmax(y_data)) if len(y_data) else 1.0
        return y0min, y0max

    def _draw_first_click(x0: float):
        y0min, y0max = _current_y_bounds()
        vline = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dot"),
            name="norm_vline_first",
        )
        fig.add_shape(vline)

    def _draw_selection_visuals(x0, x1):
        y0min, y0max = _current_y_bounds()
        vline1 = dict(
            type="line",
            x0=x0,
            x1=x0,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dash"),
        )
        vline2 = dict(
            type="line",
            x0=x1,
            x1=x1,
            y0=y0min,
            y1=y0max,
            line=dict(color="red", dash="dash"),
        )
        rect = dict(
            type="rect",
            x0=min(x0, x1),
            x1=max(x0, x1),
            y0=y0min,
            y1=y0max,
            fillcolor="rgba(0,128,0,0.15)",
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(vline1)
        fig.add_shape(vline2)
        fig.add_shape(rect)

    def _on_click(trace, points, selector):
        if not points.xs:
            return
        x_val = float(points.xs[0])
        if len(selected_points) >= 2:
            selected_points.clear()
            _clear_selection_visuals()
        selected_points.append(x_val)
        if len(selected_points) == 1:
            _clear_selection_visuals()
            _draw_first_click(x_val)
            with msg_out:
                clear_output(wait=True)
                print(f"First point set at x = {x_val:.3f} cm⁻¹. Click second point…")
        elif len(selected_points) == 2:
            a, b = selected_points
            _clear_selection_visuals()
            _draw_selection_visuals(a, b)
            with msg_out:
                clear_output(wait=True)
                lo, hi = (min(a, b), max(a, b))
                print(f"Selected normalization range: [{lo:.3f}, {hi:.3f}] cm⁻¹")

    # Decide interaction mode (click vs slider) based on Colab detection.
    range_slider = None
    range_lo_text = None
    range_hi_text = None
    range_apply_btn = None
    if _IN_COLAB:
        try:
            # Determine global xmin/xmax across all spectra for slider bounds.
            low_vals, high_vals = [], []
            for idx_tmp in FTIR_DataFrame.index:
                xv_tmp = _parse_seq(FTIR_DataFrame.loc[idx_tmp].get("X-Axis"))
                if xv_tmp:
                    low_vals.append(np.nanmin(xv_tmp))
                    high_vals.append(np.nanmax(xv_tmp))
            if low_vals and high_vals:
                global_low = float(np.nanmin(low_vals))
                global_high = float(np.nanmax(high_vals))
            else:
                global_low, global_high = 0.0, 1.0
        except Exception:
            global_low, global_high = 0.0, 1.0
        step = (global_high - global_low) / 1000.0 or 1.0
        range_slider = widgets.FloatRangeSlider(
            value=[global_low, global_high],
            min=global_low,
            max=global_high,
            step=step,
            description="Range (cm⁻¹)",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="90%"),
        )
        range_lo_text = widgets.FloatText(
            value=global_low,
            description="Low (cm⁻¹)",
            layout=widgets.Layout(width="160px"),
        )
        range_hi_text = widgets.FloatText(
            value=global_high,
            description="High (cm⁻¹)",
            layout=widgets.Layout(width="160px"),
        )
        range_apply_btn = widgets.Button(
            description="Apply",
            button_style="info",
            layout=widgets.Layout(width="90px"),
        )
        range_help = widgets.HTML(
            "<span style='color:#555;font-size:12px;'>Colab: Type Low/High and click Apply, or drag the slider. Selection is clamped to data bounds.</span>"
        )

        def _on_range_slider(change):
            if change.get("name") != "value":
                return
            try:
                lo, hi = change.get("new")
                # Sync text boxes
                try:
                    range_lo_text.value = float(lo)
                    range_hi_text.value = float(hi)
                except Exception:
                    pass
                selected_points.clear()
                selected_points.extend([float(lo), float(hi)])
                _clear_selection_visuals()
                _draw_selection_visuals(float(lo), float(hi))
                with msg_out:
                    clear_output(wait=True)
                    lo2, hi2 = (min(lo, hi), max(lo, hi))
                    print(
                        f"Selected normalization range (slider): [{lo2:.3f}, {hi2:.3f}] cm⁻¹"
                    )
            except Exception:
                pass

        def _apply_text_range(_b=None):
            try:
                lo = float(range_lo_text.value)
                hi = float(range_hi_text.value)
            except Exception:
                with msg_out:
                    clear_output(wait=True)
                    print("Invalid numeric input for Low/High.")
                return
            # Clamp/order
            if lo > hi:
                lo, hi = hi, lo
            try:
                lo_clamped = max(global_low, min(global_high, lo))
                hi_clamped = max(global_low, min(global_high, hi))
            except Exception:
                lo_clamped, hi_clamped = lo, hi
            # Update slider (which triggers its observer to update visuals & messages)
            try:
                range_slider.value = [float(lo_clamped), float(hi_clamped)]
            except Exception:
                pass
            # In case observer didn't fire (same values), manually reflect
            if (
                len(selected_points) != 2
                or selected_points[0] != lo_clamped
                or selected_points[1] != hi_clamped
            ):
                selected_points.clear()
                selected_points.extend([float(lo_clamped), float(hi_clamped)])
                _clear_selection_visuals()
                _draw_selection_visuals(float(lo_clamped), float(hi_clamped))
                with msg_out:
                    clear_output(wait=True)
                    print(
                        f"Selected normalization range (typed): [{lo_clamped:.3f}, {hi_clamped:.3f}] cm⁻¹"
                    )

        try:
            range_apply_btn.on_click(_apply_text_range)
        except Exception:
            pass

        range_slider.observe(_on_range_slider, names="value")
    else:
        try:
            fig.data[0].on_click(_on_click)
        except Exception:
            pass

    def _current_range():
        if len(selected_points) != 2:
            return None
        a, b = selected_points
        return [float(min(a, b)), float(max(a, b))]

    def _get_xy(idx):
        """Return (x, y, used_baseline) for the selected row.

        Uses baseline-corrected data when available; falls back to raw data otherwise.
        """
        row = FTIR_DataFrame.loc[idx]
        x = row.get("X-Axis")
        y_bc = row.get("Baseline-Corrected Data")
        raw = row.get("Raw Data")
        # Parse x if stored as literal string
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                pass
        # Normalize baseline-corrected value
        if isinstance(y_bc, str):
            if y_bc.strip().lower() == "nan":
                y_bc = None
            else:
                try:
                    y_bc = ast.literal_eval(y_bc)
                except Exception:
                    # leave as original string if cannot eval
                    pass
        # If baseline is a pandas Series convert to list
        try:
            import pandas as _pd
            if isinstance(y_bc, _pd.Series):
                y_bc = y_bc.tolist()
        except Exception:
            pass
        # Determine validity
        y_bc_valid = False
        try:
            import numpy as _np
            if isinstance(y_bc, _np.ndarray):
                y_bc_valid = y_bc.ndim == 1 and y_bc.size > 1 and not _np.isnan(y_bc).all()
            elif isinstance(y_bc, (list, tuple)):
                y_bc_valid = len(y_bc) > 1 and not all(
                    (isinstance(v, float) and np.isnan(v)) for v in y_bc
                )
        except Exception:
            if isinstance(y_bc, (list, tuple)) and len(y_bc) > 1:
                y_bc_valid = True
        # Fallback to raw if baseline invalid
        y = y_bc if y_bc_valid else raw
        # Parse raw if needed for potential fallback
        if y is raw and isinstance(raw, str):
            try:
                y = ast.literal_eval(raw)
            except Exception:
                pass
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                pass
        # Final guard: fallback again if y still not a proper sequence
        if not isinstance(y, (list, tuple, np.ndarray)) or (isinstance(y, (list, tuple)) and len(y) <= 1):
            # Attempt to parse raw more aggressively
            raw_fallback = raw
            if isinstance(raw_fallback, str):
                try:
                    raw_fallback = ast.literal_eval(raw_fallback)
                except Exception:
                    pass
            if isinstance(raw_fallback, (list, tuple, np.ndarray)) and len(raw_fallback) > 1:
                y = raw_fallback
                y_bc_valid = False
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), bool(y_bc_valid)

    def _row_filepath(idx):
        r = FTIR_DataFrame.loc[idx]
        return os.path.join(
            str(r.get("File Location", "")), str(r.get("File Name", ""))
        )

    def _row_label(idx):
        r = FTIR_DataFrame.loc[idx]
        t = r.get("Time", "?")
        mat = r.get("Material", "?")
        cond = r.get(cond_col, "?") if cond_col else "?"
        fname = r.get("File Name", "?")
        return f"{mat} | {cond} | t={t} | {fname}"

    def _filter_by_material_condition(df):
        """Return DataFrame filtered by current Material/Conditions selections.

        Includes unexposed spectra for the selected material regardless of the
        chosen condition. When either dropdown is set to 'any', that dimension is
        not restricted."""
        if df is None or len(df) == 0:
            return df.iloc[0:0]
        mask = pd.Series(True, index=df.index, dtype=bool)
        # Normalize material values for comparison
        mat_series = df.get("Material")
        if mat_series is None:
            mat_series = pd.Series([""] * len(df), index=df.index, dtype=object)
        else:
            mat_series = mat_series.fillna("")
        mat_clean = mat_series.astype(str).str.strip()
        sel_mat_value = material_dd.value if hasattr(material_dd, "value") else "any"
        sel_mat_norm = None
        if sel_mat_value is not None and str(sel_mat_value).strip().lower() != "any":
            sel_mat_norm = str(sel_mat_value).strip().lower()
            mask &= mat_clean.str.lower() == sel_mat_norm
        # Apply condition filter
        sel_cond_value = conditions_dd.value if hasattr(conditions_dd, "value") else "any"
        cond_column = _conditions_column_name(df)
        if cond_column and sel_cond_value is not None and str(sel_cond_value).strip().lower() != "any":
            cond_series = df.get(cond_column)
            if cond_series is None:
                cond_series = pd.Series([""] * len(df), index=df.index, dtype=object)
            else:
                cond_series = cond_series.fillna("")
            cond_clean = cond_series.astype(str).str.strip()
            target_cond = str(sel_cond_value).strip().lower()
            cond_match = cond_clean.str.lower() == target_cond
            unexposed_mask = cond_clean.str.contains(r"\bunexposed\b", case=False, na=False)
            if sel_mat_norm is not None:
                mat_match = mat_clean.str.lower() == sel_mat_norm
                mask &= cond_match | (unexposed_mask & mat_match)
            else:
                mask &= cond_match
        return df[mask]

    def _rebuild_spectrum_options(*_):
        filtered = _filter_by_material_condition(FTIR_DataFrame)
        # Optionally exclude rows marked as bad quality
        try:
            if not include_bad_cb.value:
                filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass
        # sort by Time if present
        if "Time" in filtered.columns:
            try:
                filtered = filtered.sort_values(by="Time")
            except Exception:
                pass
        opts = [(_row_label(i), i) for i in filtered.index]
        if not opts:
            spectrum_sel.options = [("<no spectra>", None)]
            spectrum_sel.value = None
            with info_out:
                clear_output(wait=True)
                if display_mode.value == "series":
                    print("Select specific Material and Conditions (not 'any') to view time-series.")
                else:
                    print("Select a spectrum from the dropdown to begin normalization.")
            # If in time-series mode, also clear the plot so it's obvious
            try:
                if display_mode.value == "series":
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Time Series | No data to display")
            except Exception:
                pass
            return
        spectrum_sel.options = opts
        # Preserve existing selection if still valid; otherwise require manual user selection
        if spectrum_sel.value not in [v for _, v in opts]:
            spectrum_sel.value = None
            with info_out:
                clear_output(wait=True)
                if display_mode.value == "series":
                    print("Select specific Material and Conditions (not 'any') to view time-series.")
                else:
                    print("Select a spectrum from the dropdown to begin normalization.")
        # If time-series mode is active, refresh plot to reflect current filters
        try:
            if display_mode.value == "series":
                _update_plot_for_selection()
        except Exception:
            pass

    def _update_plot_for_selection(*_):
        idx = spectrum_sel.value
        # Time-series mode: plot all spectra currently present in dropdown options (filtered set)
        if display_mode.value == "series":
            # Always show time-series guidance message
            try:
                with info_out:
                    clear_output(wait=True)
                    print("Select specific Material and Conditions (not 'any') to view time-series.")
            except Exception:
                pass
            # Require specific Material and (if present) Conditions before plotting
            try:
                if material_dd.value == "any" or (cond_col and conditions_dd.value == "any"):
                    try:
                        mark_row.layout.display = "none"
                    except Exception:
                        pass
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Time Series | Awaiting selections")
                    return
            except Exception:
                pass
            # Build filtered DataFrame directly (dropdown hidden in series mode)
            try:
                filtered_ts = _filter_by_material_condition(FTIR_DataFrame)
                if not include_bad_cb.value and len(filtered_ts):
                    try:
                        filtered_ts = filtered_ts[_quality_good_mask(filtered_ts)]
                    except Exception:
                        pass
                if "Time" in filtered_ts.columns:
                    try:
                        filtered_ts = filtered_ts.sort_values(by="Time")
                    except Exception:
                        pass
            except Exception:
                filtered_ts = FTIR_DataFrame.head(0)
            if filtered_ts is None or len(filtered_ts) == 0:
                # Keep guidance consistent
                with info_out:
                    clear_output(wait=True)
                    print("Select specific Material and Conditions (not 'any') to view time-series.")
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
                with fig.batch_update():
                    fig.data = tuple()
                    fig.update_layout(title="Time Series | No spectra")
                return
            series_data = []  # list of dicts: {x, y, name, time}
            count_plotted = 0
            for i in filtered_ts.index:
                try:
                    r = FTIR_DataFrame.loc[i]
                    x_arr, y_arr, _used_bc = _get_xy(i)
                    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
                        continue
                    tval = r.get("Time", "?")
                    cond_val = r.get(cond_col, "?") if cond_col else None
                    parts = [f"t={tval}"]
                    if cond_col:
                        parts.append(str(cond_val))
                    name = " | ".join(parts)
                    series_data.append({
                        "x": x_arr.tolist(),
                        "y": y_arr.tolist(),
                        "name": name,
                        "time": tval,
                    })
                    count_plotted += 1
                except Exception:
                    continue
            if len(series_data) == 0:
                with info_out:
                    clear_output(wait=True)
                    print("Select specific Material and Conditions (not 'any') to view time-series.")
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
                with fig.batch_update():
                    fig.data = tuple()
                    fig.update_layout(title="Time Series | No data to display")
                # Still reveal container so user sees the state
                try:
                    bordered_plot.layout.display = ""
                except Exception:
                    pass
                try:
                    action_row.layout.display = "none"
                except Exception:
                    pass
                return
            title_mat = material_dd.value
            title_cond_sel = conditions_dd.value if (cond_col and conditions_dd.value != "any") else None
            # Build unique times list in display order for palette mapping
            times_unique = []
            try:
                for d in series_data:
                    tv = d.get("time")
                    if tv not in times_unique:
                        times_unique.append(tv)
            except Exception:
                times_unique = []

            with fig.batch_update():
                # Shrink to desired number of traces by taking a subset (allowed by FigureWidget)
                while len(fig.data) > len(series_data):
                    fig.data = fig.data[:-1]
                # Update existing traces or add new ones
                for i, d in enumerate(series_data):
                    # Assign palette color based on time position
                    try:
                        _col = _time_gradient_color(times_unique, d.get("time"))
                    except Exception:
                        _col = None
                    if i < len(fig.data):
                        fig.data[i].x = d["x"]
                        fig.data[i].y = d["y"]
                        fig.data[i].mode = "lines"
                        fig.data[i].name = d["name"]
                        try:
                            if _col:
                                fig.data[i].line.color = _col
                        except Exception:
                            pass
                    else:
                        if _col:
                            fig.add_scatter(x=d["x"], y=d["y"], mode="lines", name=d["name"], line=dict(color=_col))
                        else:
                            fig.add_scatter(x=d["x"], y=d["y"], mode="lines", name=d["name"])
                title = f"Time Series | {title_mat}"
                if title_cond_sel:
                    title += f" | Condition: {title_cond_sel} (+ unexposed)"
                fig.update_layout(title=title)
            # Attach click handlers to all traces (non-Colab) for range selection
            try:
                if not _IN_COLAB:
                    for tr in fig.data:
                        try:
                            tr.on_click(_on_click)
                        except Exception:
                            pass
            except Exception:
                pass
            _clear_selection_visuals()
            selected_points.clear()
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            # Keep the guidance message; add plot count to logs if needed
            with info_out:
                clear_output(wait=True)
                print("Select specific Material and Conditions (not 'any') to view time-series.")
            # Reveal plot container and show Save/Redo in time-series mode
            try:
                bordered_plot.layout.display = ""
            except Exception:
                pass
            try:
                action_row.layout.display = ""
            except Exception:
                pass
            return
        # Single-spectrum mode below
        if idx is None:
            # Leaving time-series mode or no spectrum chosen yet -> clear figure for clarity
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            try:
                if display_mode.value != "series":
                    with fig.batch_update():
                        fig.data = tuple()
                        fig.update_layout(title="Select a spectrum (or enable Time-Series)")
                    with info_out:
                        clear_output(wait=True)
                        print("Choose a spectrum from the dropdown to plot, or enable Time-Series.")
            except Exception:
                pass
            return
        # Update current displayed index
        nonlocal current_idx
        current_idx = idx
        nonlocal x_data, y_data
        x_arr, y_arr, used_bc = _get_xy(idx)
        x_data, y_data = x_arr, y_arr
        # update trace (create if missing)
        try:
            if len(fig.data) == 0:
                fig.add_scatter(x=x_data.tolist(), y=y_data.tolist(), mode="lines", name="Spectrum")
            else:
                fig.data[0].x = x_data.tolist()
                fig.data[0].y = y_data.tolist()
        except Exception:
            pass
        try:
            fig.data[0].name = (
                "Baseline-Corrected" if used_bc else "Raw Data (baseline not saved)"
            )
        except Exception:
            pass
        _clear_selection_visuals()
        selected_points.clear()
        # update title and info
        mat_val = FTIR_DataFrame.loc[idx].get("Material", "?")
        fig.update_layout(title=f"Select Normalization Range | Material: {mat_val}")
        with info_out:
            clear_output(wait=True)
            print(f"Plotting: {_row_filepath(idx)}")
            if used_bc:
                print("Using baseline-corrected data for range selection.")
            else:
                print(
                    "Baseline-corrected data unavailable; showing raw data (apply baseline first)."
                )
            print("Click two points to select the tip of the normalization peak.")
        # Reveal plot and action buttons for single-spectrum mode
        try:
            bordered_plot.layout.display = ""
        except Exception:
            pass
        try:
            action_row.layout.display = ""
        except Exception:
            pass
        # if this row already has a saved range, visualize it
        try:
            rng = FTIR_DataFrame.loc[idx].get(target_col, None)
            if isinstance(rng, str):
                rng = ast.literal_eval(rng)
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                a, b = float(rng[0]), float(rng[1])
                selected_points[:] = [a, b]
                _clear_selection_visuals()
                _draw_selection_visuals(a, b)
                with msg_out:
                    clear_output(wait=True)
                    lo, hi = (min(a, b), max(a, b))
                    print(
                        f"Existing normalization range: [{lo:.3f}, {hi:.3f}] cm⁻¹ (not yet re-saved)"
                    )
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        # Show mark row now that a spectrum is actively plotted
        try:
            mark_row.layout.display = ""
        except Exception:
            pass
        # Persist time selection for session
        try:
            if idx is not None and "Time" in FTIR_DataFrame.columns:
                _set_session_selection(time=FTIR_DataFrame.loc[idx].get("Time"))
        except Exception:
            pass

    # Build initial options and selection
    _rebuild_spectrum_options()
    # If filepath/material preselection points to a specific row, set it now
    if preselect_idx is not None:
        # adjust dropdowns if needed to include this row context
        try:
            r = FTIR_DataFrame.loc[preselect_idx]
            if str(r.get("Material", "any")) in material_dd.options:
                material_dd.value = str(r.get("Material", "any"))
            if cond_col and (str(r.get(cond_col, "any")) in conditions_dd.options):
                conditions_dd.value = str(r.get(cond_col, "any"))
        except Exception:
            pass
        _rebuild_spectrum_options()
        try:
            spectrum_sel.value = preselect_idx
        except Exception:
            pass

    # Delayed plotting: wait for user to manually choose a spectrum.

    # --- Buttons ---
    # Widen Save button so full description is visible (was truncated previously)
    save_mat_btn = widgets.Button(
        description="Save range and normalize material",
        button_style="info",
        layout=widgets.Layout(width="300px"),
    )
    redo_btn = widgets.Button(description="Redo", button_style="warning")
    cancel_btn = widgets.Button(description="Close", button_style="danger")
    # --- Change tracking (session summary on close) ---
    # Store structured events so we can summarize intelligently.
    _norm_changes = {
        "range_material": [],  # list[(material, count_rows, range_str)] saved per material
        "normalized_materials": [],  # list[(material, updated_count, skipped_count)]
        "quality": [],  # list[(idx, new_quality)]
    }
    # Use shared quality controls (mutually exclusive buttons)
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx]
            if current_idx is not None
            else None
        ),
    )
    # Main control row excludes mark buttons; they go on their own row as a pair
    # Separate action buttons from Close so Close is always visible
    action_row = widgets.HBox([save_mat_btn, redo_btn])
    close_row = widgets.HBox([cancel_btn])
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide Mark buttons until a spectrum is selected; place below plot inside border
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass
    bordered_plot = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    # Hide plot and action buttons until actual plotting occurs
    try:
        bordered_plot.layout.display = "none"
    except Exception:
        pass
    try:
        action_row.layout.display = "none"
    except Exception:
        pass

    # Refresh function provided by helper; keep name for local uses

    def _finalize_and_clear():
        # Detach click handler and close figure
        try:
            fig.data[0].on_click(None)
        except Exception:
            pass
        try:
            fig.close()
        except Exception:
            pass
        # Close widgets created in this UI
        widget_list = [
            save_mat_btn,
            redo_btn,
            cancel_btn,
            mark_bad_btn,
            mark_good_btn,
            spectrum_sel,
            material_dd,
            conditions_dd,
            include_bad_cb,
            display_mode,
            info_out,  # ensure info output removed after close
        ]
        if range_slider is not None:
            widget_list.append(range_slider)
        for w in widget_list:
            try:
                w.close()
            except Exception:
                pass
        # Close layout containers (including bordered plot frame) so only summary msg_out remains
        for container in (controls_row, mode_row, spectrum_row, mark_row, action_row, close_row, bordered_plot):
            try:
                container.close()
            except Exception:
                pass


    def _save_for_this_material(_b=None):
        # Determine material context and validate selection range
        rng = _current_range()
        if rng is None:
            with msg_out:
                clear_output(wait=True)
                print("Please select two points before saving.")
            return
        mat = None
        if display_mode.value == "series":
            # In time-series mode, use the chosen Material directly
            try:
                mat = material_dd.value if material_dd.value != "any" else None
            except Exception:
                mat = None
        else:
            # In single-spectrum mode, infer from the selected row
            idx = spectrum_sel.value
            if idx is None:
                with msg_out:
                    clear_output(wait=True)
                    print("No spectrum selected.")
                return
            mat = FTIR_DataFrame.loc[idx].get("Material", None)
        if mat is None:
            with msg_out:
                clear_output(wait=True)
                print("Material not determined; select a material/spectrum first.")
            return
        mask = FTIR_DataFrame["Material"] == mat
        FTIR_DataFrame.loc[mask, target_col] = str(rng)
        try:
            _norm_changes["range_material"].append((mat, int(mask.sum()), str(rng)))
        except Exception:
            pass
        # Normalize immediately after saving range for material
        try:
            _normalize(mat)
            try:
                updated_count = int((FTIR_DataFrame["Material"] == mat).sum())
            except Exception:
                updated_count = 0
            try:
                _norm_changes["normalized_materials"].append((mat, updated_count, None))
            except Exception:
                pass
        except Exception:
            pass
        with msg_out:
            clear_output(wait=True)
            print(f"Saved normalization peak range {rng} for material '{mat}' and normalized its spectra.")


    def _redo(_b=None):
        selected_points.clear()
        _clear_selection_visuals()
        with msg_out:
            clear_output(wait=True)
            print("Selection cleared. Click two points to select the tip of the normalization peak.")

    def _close(_b=None):
        # Clear all prior info/messages so only session summary remains
        try:
            with info_out:
                clear_output(wait=True)
        except Exception:
            pass
        try:
            with msg_out:
                clear_output(wait=True)
        except Exception:
            pass
        # Build and display session summary, then remove other widgets
        try:
            lines = _session_summary_lines(_norm_changes, context="normalization")
            _emit_session_summary(msg_out, lines, title="Session summary (Normalization)")
        except Exception:
            pass
        # Close remaining interactive widgets except msg_out (preserve summary)
        _finalize_and_clear()

    # Add extra UI effects after helper toggles quality
    def _post_mark_update(status_label: str):
        nonlocal current_idx
        try:
            idx = current_idx
            if idx is None:
                return
            with msg_out:
                clear_output(wait=True)
                print(f"Marked row {idx} as {status_label} quality.")
            try:
                _norm_changes["quality"].append((idx, status_label))
            except Exception:
                pass
            # Decouple dropdown from plot via centralized helper
            if status_label in ("bad", "good"):
                try:
                    # Keep current plot index stable so display doesn't flicker
                    if status_label == "bad":
                        current_idx = idx
                    _quality_dropdown_handle(
                        status_label,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=_row_label,
                        observer_fn=_update_plot_for_selection,
                    )
                except Exception:
                    pass
            else:
                try:
                    _rebuild_spectrum_options()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    # Wire events
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")
    spectrum_sel.observe(_update_plot_for_selection, names="value")
    include_bad_cb.observe(_rebuild_spectrum_options, names="value")
    include_bad_cb.observe(_update_plot_for_selection, names="value")
    # Ensure time-series re-plots immediately on filter changes
    material_dd.observe(_update_plot_for_selection, names="value")
    conditions_dd.observe(_update_plot_for_selection, names="value")
    save_mat_btn.on_click(_save_for_this_material)
    redo_btn.on_click(_redo)
    # The helper already wires core quality changes; add post-effects
    mark_bad_btn.on_click(lambda _b=None: _post_mark_update("bad"))
    mark_good_btn.on_click(lambda _b=None: _post_mark_update("good"))
    cancel_btn.on_click(_close)

    # New behaviors: mode toggle and reveal-on-plot
    def _on_display_mode(change):
        is_series = change.get("new") == "series"
        # Toggle spectrum selector visibility
        try:
            spectrum_row.layout.display = "none" if is_series else ""
        except Exception:
            pass
        # Hide plot and action buttons until an actual plot is produced
        try:
            bordered_plot.layout.display = "none"
        except Exception:
            pass
        try:
            action_row.layout.display = "none"
        except Exception:
            pass
        try:
            mark_row.layout.display = "none"
        except Exception:
            pass
        # Auto-trigger update when switching to time-series
        if is_series:
            try:
                _rebuild_spectrum_options()
                _update_plot_for_selection(None)
            except Exception:
                pass

    def _on_spectrum_change_reveal(change):
        # Reveal plot and action buttons when a single spectrum is chosen
        if display_mode.value == "single" and change.get("new") is not None:
            try:
                bordered_plot.layout.display = ""
            except Exception:
                pass
            try:
                action_row.layout.display = ""
            except Exception:
                pass
            try:
                mark_row.layout.display = ""
            except Exception:
                pass

    def _on_filter_change_reveal(_change):
        # In time-series mode, only reveal plot after both Material and Conditions chosen (if conditions exist)
        if display_mode.value == "series":
            material_ready = material_dd.value != "any"
            cond_ready = True if not cond_col else conditions_dd.value != "any"
            if material_ready and cond_ready:
                try:
                    bordered_plot.layout.display = ""
                except Exception:
                    pass
                try:
                    action_row.layout.display = ""  # show Save/Redo in series mode
                except Exception:
                    pass
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass
            else:
                # Hide until selections complete
                try:
                    bordered_plot.layout.display = "none"
                except Exception:
                    pass
                try:
                    action_row.layout.display = "none"
                except Exception:
                    pass
                try:
                    mark_row.layout.display = "none"
                except Exception:
                    pass

    display_mode.observe(_on_display_mode, names="value")
    spectrum_sel.observe(_on_spectrum_change_reveal, names="value")
    material_dd.observe(_on_filter_change_reveal, names="value")
    conditions_dd.observe(_on_filter_change_reveal, names="value")

    # Layout: controls on top, then plot, then info and messages, then buttons
    controls_row = widgets.HBox([material_dd, conditions_dd])
    mode_row = widgets.HBox([display_mode])
    spectrum_row = widgets.HBox([spectrum_sel, include_bad_cb])
    if range_slider is not None:
        # Compose Colab range selection row with texts + slider + Apply button and help
        try:
            colab_range_row = widgets.HBox(
                [range_lo_text, range_hi_text, range_apply_btn]
            )
            display(
                controls_row,
                mode_row,
                spectrum_row,
                colab_range_row,
                range_slider,
                range_help,
                bordered_plot,
                info_out,
                msg_out,
                action_row,
                close_row,
            )
        except Exception:
            display(
                controls_row,
                mode_row,
                spectrum_row,
                range_slider,
                bordered_plot,
                info_out,
                msg_out,
                action_row,
                close_row,
            )
    else:
        display(
            controls_row,
            mode_row,
            spectrum_row,
            bordered_plot,
            info_out,
            msg_out,
            action_row,
            close_row,
        )
    try:
        _refresh_mark_buttons()
    except Exception:
        pass
    return FTIR_DataFrame


def find_peak_info(FTIR_DataFrame, filepath=None):
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """
    Interactive peak finder for normalized and baseline-corrected spectra.

    - Uses scipy.signal.find_peaks on 'Normalized and Corrected Data'.
    - Checkboxes enable up to 3 independent X-range sliders; peaks are found in the
    union of enabled ranges.
    - Displays a live-updating plot with user-adjustable parameters.
    - Saves results (lists) to 'Peak Wavenumbers' and 'Peak Absorbances' columns.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    filepath : str | None
        Optional. If provided, the UI will be pre-filtered to this file (by matching
        full path or just filename). Otherwise, all spectra will be available and you
        can select Material/Conditions/Spectrum via dropdowns.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame (in-place modifications also applied).
    """
    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Start with the full DataFrame and optionally pre-filter by filepath
    if filepath is not None:
        # Match by full path (File Location + File Name) or by just filename
        if os.path.sep in str(filepath):
            file_dir, file_name = os.path.split(str(filepath))
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame.get("File Location", "") == file_dir)
                & (FTIR_DataFrame.get("File Name", "") == file_name)
            ]
        else:
            filtered = FTIR_DataFrame[
                FTIR_DataFrame.get("File Name", "") == str(filepath)
            ]
        # Do not exclude bad rows here; a UI checkbox will control inclusion below
        if filtered.empty:
            raise ValueError(
                f"No rows found for filepath '{filepath}'. Ensure 'File Location' and 'File Name' are populated."
            )
    else:
        filtered = FTIR_DataFrame.copy()

    # Ensure destination columns exist and are object dtype
    for col in ("Peak Wavenumbers", "Peak Absorbances"):
        if col not in FTIR_DataFrame.columns:
            FTIR_DataFrame[col] = None
        try:
            FTIR_DataFrame[col] = FTIR_DataFrame[col].astype(object)
        except Exception:
            pass

    # Use shared _parse_seq helper defined at module scope

    # Determine the conditions column via shared helper
    cond_col = _conditions_column_name(filtered)

    # Material and Conditions dropdowns (Conditions excludes 'unexposed')
    try:
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in filtered.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
            if "Material" in filtered.columns
            else []
        )
    except Exception:
        unique_materials = []
    cond_series = (
        filtered[cond_col]
        if cond_col and (cond_col in filtered.columns)
        else pd.Series([], dtype=object)
    )
    _all_conditions = [
        str(v) for v in cond_series.dropna().astype(str).unique().tolist()
    ]
    unique_conditions = sorted(
        [c for c in _all_conditions if c.strip().lower() != "unexposed"]
    )

    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    # Seed Material/Conditions from session defaults if available
    try:
        _sess_defaults = _get_session_defaults()
        _sess_mat = str(_sess_defaults.get("material", "any"))
        _sess_cond = str(_sess_defaults.get("conditions", "any"))
        if _sess_mat in list(material_dd.options):
            material_dd.value = _sess_mat
        if _sess_cond in list(conditions_dd.options):
            conditions_dd.value = _sess_cond
    except Exception:
        pass
    # Apply session defaults if present
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    # Apply session defaults if present before building options
    try:
        _sess = _get_session_defaults()
        if _sess.get("material") in material_dd.options:
            material_dd.value = _sess.get("material")
        if _sess.get("conditions") in conditions_dd.options:
            conditions_dd.value = _sess.get("conditions")
    except Exception:
        pass

    # Build spectrum options using current filters; include 'unexposed' spectra always
    def _current_filtered_df():
        try:
            include_bad_flag = bool(getattr(include_bad_cb, "value", True))
        except Exception:
            include_bad_flag = True
        return _filter_spectra_dataframe(
            filtered,
            material=getattr(material_dd, "value", "any"),
            condition=getattr(conditions_dd, "value", "any"),
            include_bad=include_bad_flag,
            include_unexposed=True,
            normalized_column="Normalized and Corrected Data",
        )

    def _build_options():
        df = _current_filtered_df()
        # Only include rows with normalized data available
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            # If column missing or not Series, this will result in empty options
            df = df.iloc[0:0]
        # Sort by Time ascending if present
        try:
            if "Time" in df.columns:
                df = df.copy()
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
        except Exception:
            pass
        opts = []
        for idx, r in df.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            opts.append((label, idx))
        return opts

    options = _build_options()
    if not options:
        try:
            info_html = widgets.HTML(
                "<span style='color:#a00;'>No spectra currently match the filters. "
                "Ensure 'Normalized and Corrected Data' is populated and adjust Material/Conditions.</span>"
            )
            display(info_html)
        except Exception:
            pass
        options = [("<no spectra>", None)]

    # Seed from first spectrum, prefer session 'time' if available; skip placeholder None values
    try:
        _sess = _get_session_defaults()
        saved_time = _sess.get("time", "any")

        def _matches_time(idx):
            try:
                t = FTIR_DataFrame.loc[idx].get("Time")
                if isinstance(saved_time, str) and saved_time.strip().lower() == "any":
                    return False
                try:
                    return int(t) == int(saved_time)
                except Exception:
                    return str(t) == str(saved_time)
            except Exception:
                return False

        match = next((idx for (_lab, idx) in options if idx is not None and _matches_time(idx)), None)
        valid_vals = [v for (_lab, v) in options if v is not None]
        first_idx = match if match is not None else (valid_vals[0] if valid_vals else None)
    except Exception:
        valid_vals = [v for (_lab, v) in options if v is not None]
        first_idx = valid_vals[0] if valid_vals else None

    if first_idx is not None:
        x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
        y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
        if x0 is None or y0 is None:
            # Fall back to empty arrays if selected row lacks data
            x0, y0 = [], []
            xmin, xmax = 0.0, 1.0
        else:
            x0 = np.asarray(x0, dtype=float)
            y0 = np.asarray(y0, dtype=float)
            if x0.size == 0 or y0.size == 0:
                xmin, xmax = 0.0, 1.0
            else:
                xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))
    else:
        x0, y0 = [], []
        xmin, xmax = 0.0, 1.0

    # Build spectrum options using current filters; include 'unexposed' spectra always
    def _current_filtered_df():
        try:
            include_bad_flag = bool(getattr(include_bad_cb, "value", True))
        except Exception:
            include_bad_flag = True
        return _filter_spectra_dataframe(
            filtered,
            material=getattr(material_dd, "value", "any"),
            condition=getattr(conditions_dd, "value", "any"),
            include_bad=include_bad_flag,
            include_unexposed=True,
            normalized_column="Normalized and Corrected Data",
        )

    def _build_options():
        df = _current_filtered_df()
        # Only include rows with normalized data available
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            df = df[
                (
                    df.get("Normalized and Corrected Data", None).notna()
                    if hasattr(df.get("Normalized and Corrected Data", None), "notna")
                    else []
                )
            ]
        # Sort by Time ascending if present
        try:
            if "Time" in df.columns:
                df = df.copy()
                df["_sort_time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(
                    float("inf")
                )
                df = df.sort_values(by=["_sort_time"], kind="mergesort")
        except Exception:
            pass
        opts = []
        for idx, r in df.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            opts.append((label, idx))
        return opts

    options = _build_options()
    if not options:
        options = [("<no spectra>", None)]

    # Seed from first spectrum; prefer session 'time' match if available; skip None
    try:
        _sess = _get_session_defaults()
        saved_time = _sess.get("time", "any")

        def _time_matches(idx):
            try:
                tval = FTIR_DataFrame.loc[idx].get("Time")
                if isinstance(saved_time, str) and saved_time.strip().lower() == "any":
                    return False
                try:
                    return int(tval) == int(saved_time)
                except Exception:
                    return str(tval) == str(saved_time)
            except Exception:
                return False

        _match_idx = next((idx for (_lab, idx) in options if idx is not None and _time_matches(idx)), None)
        _valid_vals = [v for (_lab, v) in options if v is not None]
        first_idx = _match_idx if _match_idx is not None else (_valid_vals[0] if _valid_vals else None)
    except Exception:
        _valid_vals = [v for (_lab, v) in options if v is not None]
        first_idx = _valid_vals[0] if _valid_vals else None

    if first_idx is not None:
        x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
        y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
        if x0 is None or y0 is None:
            x0, y0 = [], []
            xmin, xmax = 0.0, 1.0
        else:
            x0 = np.asarray(x0, dtype=float)
            y0 = np.asarray(y0, dtype=float)
            if x0.size == 0 or y0.size == 0:
                xmin, xmax = 0.0, 1.0
            else:
                xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))
    else:
        x0, y0 = [], []
        xmin, xmax = 0.0, 1.0
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="70%"),
    )

    # Persist changes in dropdowns to session state
    def _persist_material_cond(_=None):
        try:
            _set_session_selection(
                material=material_dd.value, conditions=conditions_dd.value
            )
        except Exception:
            pass

    material_dd.observe(_persist_material_cond, names="value")
    conditions_dd.observe(_persist_material_cond, names="value")
    # Up to three optional X-range selectors, each gated by a checkbox
    step_val = (xmax - xmin) / 1000 or 1.0
    use_r1 = widgets.Checkbox(value=True, description="Use range 1")
    x_range1 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 1",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r1.value,
    )
    use_r2 = widgets.Checkbox(value=False, description="Use range 2")
    x_range2 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 2",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r2.value,
    )
    use_r3 = widgets.Checkbox(value=False, description="Use range 3")
    x_range3 = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=step_val,
        description="X-range 3",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
        disabled=not use_r3.value,
    )
    prominence = widgets.FloatSlider(
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.005,
        description="Prominence",
        readout_format=".3f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    min_height = widgets.FloatSlider(
        value=0.0,
        min=0.01,
        max=1.0,
        step=0.01,
        description="Min height",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    distance = widgets.IntSlider(
        value=5,
        min=1,
        max=250,
        step=1,
        description="Min separation",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    width = widgets.IntSlider(
        value=1,
        min=1,
        max=50,
        step=1,
        description="Min width",
        continuous_update=False,
        style={"description_width": "auto"},
    )
    max_peaks = widgets.IntSlider(
        value=10,
        min=1,
        max=25,
        step=1,
        description="Max peaks",
        continuous_update=False,
        style={"description_width": "auto"},
    )

    save_file_btn = widgets.Button(description="Save for spectrum", button_style="success")
    save_all_btn = widgets.Button(description="Save for material", button_style="info")
    # --- Change tracking for session summary on Close ---
    _peak_changes = {
        "saved_file": [],  # list[(idx, n_peaks)]
        "saved_filtered": 0,  # count of spectra updated via filtered save
        "quality": [],  # list[(idx, new_quality)]
    }
    # Use shared quality controls
    current_idx_fp = None  # track displayed spectrum independently of dropdown
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx_fp]
            if current_idx_fp is not None
            else None
        ),
    )
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    close_btn = widgets.Button(description="Close", button_style="danger")
    msg_out = widgets.Output()

    # Refresh provided by helper

    # Plotly figure
    fig = go.FigureWidget()
    fig.add_scatter(
        x=x0.tolist(), y=y0.tolist(), mode="lines", name="Normalized and Corrected"
    )
    fig.add_scatter(
        x=[],
        y=[],
        mode="markers",
        name="Peaks",
        marker=dict(color="red", size=9, symbol="x"),
    )
    fig.update_layout(
        title="Peak Selection (live)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    def _get_xy(row_idx):
        if row_idx is None:
            return None, None
        try:
            r = FTIR_DataFrame.loc[row_idx]
        except Exception:
            return None, None
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if (
            x_arr.ndim != 1
            or y_arr.ndim != 1
            or x_arr.shape[0] != y_arr.shape[0]
            or x_arr.size == 0
            or y_arr.size == 0
        ):
            return None, None
        return x_arr, y_arr

    def _compute_peaks_for_ranges(x_arr, y_arr, ranges):
        # Build a combined mask for all enabled ranges
        if not ranges:
            return np.array([], dtype=int), np.array([], dtype=float)
        mask = np.zeros(x_arr.shape[0], dtype=bool)
        for x_min, x_max in ranges:
            if x_min is None or x_max is None:
                continue
            lo, hi = (float(min(x_min, x_max)), float(max(x_min, x_max)))
            mask |= (x_arr >= lo) & (x_arr <= hi)
        if not np.any(mask):
            return np.array([], dtype=int), np.array([], dtype=float)
        y_sub = y_arr[mask]
        idx_sub = np.where(mask)[0]
        kwargs = {
            "prominence": (
                float(prominence.value) if prominence.value is not None else None
            ),
            "distance": int(distance.value) if distance.value is not None else None,
            "width": int(width.value) if width.value is not None else None,
        }
        if min_height.value and float(min_height.value) > 0:
            kwargs["height"] = float(min_height.value)
        peaks_local, _props = find_peaks(
            y_sub, **{k: v for k, v in kwargs.items() if v is not None}
        )
        if peaks_local.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        peaks_global = idx_sub[peaks_local]
        # limit to top-N by height if requested
        if (
            max_peaks.value
            and int(max_peaks.value) > 0
            and peaks_global.size > int(max_peaks.value)
        ):
            heights = y_arr[peaks_global]
            order = np.argsort(heights)[::-1][: int(max_peaks.value)]
            peaks_global = peaks_global[order]
        return peaks_global, y_arr[peaks_global]

    def _current_ranges():
        rs = []
        if use_r1.value:
            rs.append((x_range1.value[0], x_range1.value[1]))
        if use_r2.value:
            rs.append((x_range2.value[0], x_range2.value[1]))
        if use_r3.value:
            rs.append((x_range3.value[0], x_range3.value[1]))
        return rs

    def _update_plot(*args):
        idx = spectrum_sel.value
        nonlocal current_idx_fp
        current_idx_fp = idx
        if idx is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum missing or invalid normalized data.")
            # Hide mark row if we have no valid selection/data
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            return
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Selected spectrum has missing or invalid normalized data.")
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            return
        # Update traces
        with fig.batch_update():
            fig.data[0].x = x_arr.tolist()
        # Update bounds for each slider and enable/disable based on checkboxes
        try:
            x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
        except Exception:
            x_min, x_max = 0.0, 1.0
        for cb, sl in ((use_r1, x_range1), (use_r2, x_range2), (use_r3, x_range3)):
            sl.min = x_min
            sl.max = x_max
            try:
                lo, hi = sl.value
            except Exception:
                lo, hi = x_min, x_max
            lo = max(x_min, min(lo, x_max))
            hi = max(lo, min(hi, x_max))
            sl.value = [lo, hi]
            sl.disabled = not cb.value
        # Peaks and shading across all enabled ranges
        ranges = _current_ranges()
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        with fig.batch_update():
            fig.data[1].x = x_arr[peaks_idx].tolist() if peaks_idx.size else []
            fig.data[1].y = peaks_y.tolist() if peaks_idx.size else []
            # Rebuild all range shapes deterministically for reliability with multiple
            # ranges
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
            shapes_list = []
            for i_r, rng in enumerate(ranges):
                if rng is None:
                    continue
                lo_i, hi_i = float(min(rng)), float(max(rng))
                shapes_list.append(
                    dict(
                        type="rect",
                        x0=lo_i,
                        x1=hi_i,
                        y0=y0_min,
                        y1=y0_max,
                        fillcolor="rgba(0,128,0,0.12)",
                        line=dict(width=0),
                        layer="below",
                        name=f"range_rect_{i_r}",
                    )
                )
            # Assign shapes in one shot to avoid inconsistent state when multiple are
            # active
            fig.layout.shapes = tuple(shapes_list)
        with msg_out:
            msg_out.clear_output()
            if not ranges:
                print("Enable at least one X-range to find peaks.")
            else:
                print(f"Peaks found: {len(peaks_idx)}")
        # Update mark buttons visibility for current selection
        _refresh_mark_buttons()
        # Show mark row now that a spectrum is actively plotted
        try:
            mark_row.layout.display = ""
        except Exception:
            pass

    def _save_for_file(b):
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            with msg_out:
                msg_out.clear_output()
                print("Cannot save: selected spectrum missing normalized data.")
            return
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
        FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
            x_arr[peaks_idx].astype(float).tolist()
        )
        FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
        try:
            _peak_changes["saved_file"].append((idx, int(peaks_idx.size)))
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print(
                f"Saved {len(peaks_idx)} peaks for file "
                f"'{FTIR_DataFrame.loc[idx, 'File Name']}'."
            )

    def _save_for_filtered(b):
        # apply to all spectra for the selected material (ignore Conditions)
        ranges = _current_ranges()
        if not ranges:
            with msg_out:
                msg_out.clear_output()
                print("Please enable at least one X-range before saving.")
            return
        # Require a specific material selection
        sel_material = getattr(material_dd, "value", "any")
        if sel_material is None or str(sel_material).strip().lower() == "any":
            with msg_out:
                msg_out.clear_output()
                print("Select a specific Material before saving for material.")
            return
        # Respect Include bad spectra; always include unexposed; require normalized data
        try:
            include_bad_flag = bool(getattr(include_bad_cb, "value", True))
        except Exception:
            include_bad_flag = True
        df_mat = _filter_spectra_dataframe(
            FTIR_DataFrame,
            material=sel_material,
            condition="any",
            include_bad=include_bad_flag,
            include_unexposed=True,
            normalized_column="Normalized and Corrected Data",
        )
        # Only rows with normalized data
        try:
            df_mat = df_mat[df_mat["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        updated, skipped = 0, 0
        for idx, _row in df_mat.iterrows():
            x_arr, y_arr = _get_xy(idx)
            if x_arr is None:
                skipped += 1
                continue
            peaks_idx, peaks_y = _compute_peaks_for_ranges(x_arr, y_arr, ranges)
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = (
                x_arr[peaks_idx].astype(float).tolist()
            )
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = peaks_y.astype(float).tolist()
            updated += 1
        try:
            _peak_changes["saved_filtered"] += int(updated)
        except Exception:
            pass
        with msg_out:
            msg_out.clear_output()
            print(
                f"Updated {updated} spectra; skipped {skipped} (missing/invalid data)."
            )

    # Additional UI updates after helper toggles quality
    def _post_mark_update_find_peaks(_b=None):
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        try:
            idx = current_idx_fp
            if idx is not None:
                # Determine current quality
                qcol = _quality_column_name(FTIR_DataFrame)
                qval = FTIR_DataFrame.loc[idx].get(qcol, None)
                if qval in ("bad", "good"):
                    # Use centralized helper to remove/reinsert without disturbing current plot
                    _quality_dropdown_handle(
                        qval,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','?')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition','?'))} | {FTIR_DataFrame.loc[i].get('File Name','?')}",
                        observer_fn=_update_plot,
                    )
                else:
                    try:
                        if not include_bad_cb.value:
                            _on_filters_change()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            idx = current_idx_fp
            if idx is not None:
                # capture current quality value
                qcol = _quality_column_name(FTIR_DataFrame)
                qval = FTIR_DataFrame.loc[idx].get(qcol, None)
                _peak_changes["quality"].append((idx, qval))
        except Exception:
            pass

    def _close_ui(b):
        # Emit session summary before closing figure (leave msg_out visible)
        try:
            lines = _session_summary_lines(_peak_changes, context="peaks")
        except Exception:
            lines = ["Peak Finder session closed."]
        try:
            _emit_function_summary(msg_out, lines, title="Session Summary (Peak Finder)")
        except Exception:
            pass
        # Close all interactive widgets and containers except msg_out
        try:
            # Primitive widgets
            try: material_dd.close()
            except Exception: pass
            try: conditions_dd.close()
            except Exception: pass
            try: spectrum_sel.close()
            except Exception: pass
            for w in (x_range1, x_range2, x_range3, use_r1, use_r2, use_r3,
                      prominence, min_height, distance, width, max_peaks,
                      save_file_btn, save_all_btn, include_bad_cb, close_btn):
                try:
                    w.close()
                except Exception:
                    pass
            try: mark_bad_btn.close()
            except Exception: pass
            try: mark_good_btn.close()
            except Exception: pass
            # Figure
            try: fig.close()
            except Exception: pass
            # Row/container widgets displayed via display(...)
            for row in (
                filters_row,
                controls_row1,
                controls_row2,
                controls_row3,
                controls_row4,
                controls_row5,
                controls_row6,
                buttons_row,
                plot_and_mark_pf,
            ):
                try:
                    row.close()
                except Exception:
                    pass
        except Exception:
            pass

    # Wire events
    def _on_filters_change(*_):
        new_opts = _build_options()
        if not new_opts:
            spectrum_sel.options = [("<no spectra>", None)]
            try:
                spectrum_sel.value = None
            except Exception:
                pass
            with msg_out:
                msg_out.clear_output()
                print("No spectra match the current filters.")
            # Hide mark row when nothing is selectable
            try:
                mark_row.layout.display = "none"
            except Exception:
                pass
            return
        spectrum_sel.options = new_opts
        if spectrum_sel.value not in [v for _, v in new_opts]:
            spectrum_sel.value = new_opts[0][1]
        _update_plot()
        _refresh_mark_buttons()

    spectrum_sel.observe(_update_plot, names="value")
    # Keep mark buttons visibility in sync with selection
    spectrum_sel.observe(lambda *_: _refresh_mark_buttons(), names="value")
    material_dd.observe(_on_filters_change, names="value")
    conditions_dd.observe(_on_filters_change, names="value")
    for w in (x_range1, x_range2, x_range3, use_r1, use_r2, use_r3):
        w.observe(_update_plot, names="value")
    prominence.observe(_update_plot, names="value")
    min_height.observe(_update_plot, names="value")
    distance.observe(_update_plot, names="value")
    width.observe(_update_plot, names="value")
    max_peaks.observe(_update_plot, names="value")
    save_file_btn.on_click(_save_for_file)
    save_all_btn.on_click(_save_for_filtered)
    # Helper wires core behavior; add post-effects
    mark_bad_btn.on_click(_post_mark_update_find_peaks)
    mark_good_btn.on_click(_post_mark_update_find_peaks)
    include_bad_cb.observe(_on_filters_change, names="value")
    close_btn.on_click(_close_ui)

    controls_row1 = widgets.HBox([spectrum_sel, include_bad_cb])
    controls_row2 = widgets.HBox([use_r1, x_range1])
    controls_row3 = widgets.HBox([use_r2, x_range2])
    controls_row4 = widgets.HBox([use_r3, x_range3])
    controls_row5 = widgets.HBox([prominence, min_height, distance])
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide mark buttons until a spectrum is selected; place below plot in border
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass
    # Keep sliders separate from buttons per requirements
    controls_row6 = widgets.HBox([width, max_peaks])
    buttons_row = widgets.HBox([save_file_btn, save_all_btn, close_btn])
    # Prepend filter controls row
    filters_row = widgets.HBox([material_dd, conditions_dd])
    # Bordered plot + mark container
    plot_and_mark_pf = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    # Display the Peak-Finding UI components
    try:
        display(
            filters_row,
            controls_row1,
            controls_row2,
            controls_row3,
            controls_row4,
            controls_row5,
            plot_and_mark_pf,
            controls_row6,
            buttons_row,
            msg_out,
        )
    except Exception:
        # Fallback: display essential parts if batch display fails
        try:
            display(filters_row, controls_row1, plot_and_mark_pf, msg_out)
        except Exception:
            pass
    return FTIR_DataFrame


def deconvolute_peaks(FTIR_DataFrame, filepath=None):
    global peak_box_cache, peak_accordion
    # Begin each session with a fresh widget cache so rebuilt UIs do not reuse
    # placeholders from a previous run (which could hide the peak list until a
    # manual interaction occurs).
    if isinstance(peak_box_cache, dict):
        peak_box_cache.clear()
    peak_box_cache = {}
    # Reset accordion reference; a new instance will be assigned when the UI is built
    peak_accordion = None
    # Remove any helper callbacks left behind by a prior session
    try:
        globals().pop('_refresh_slider_lists', None)
    except Exception:
        pass
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """
    Interactively deconvolute found peaks for area analysis.

    Takes the peak info from find_peak_info and utilizes a Pseudo-Voigt model to
    approximately model the peak components as a linear combination of Gaussian and
    Lorentzian distributions. Allows for live changing of the Gaussian-Lorentzian
    fraction parameter for each peak.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame containing FTIR spectral data.
    filepath : str | None
        Specific file path to filter by (exact match). If provided, limits the list to
        that file.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with deconvolution components stored in the
        'Deconvolution Results' column.
    """

    try:
        _lmfit_models = importlib.import_module("lmfit.models")
        PseudoVoigtModel = getattr(_lmfit_models, "PseudoVoigtModel")
    except Exception as e:
        raise ImportError(
            "lmfit is required for deconvolute_peaks. Please install it (e.g., pip "
            "install lmfit)."
        ) from e

    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    # Initial selection
    if filepath is not None:
        # Match by full path (File Location + File Name) or by just filename
        if os.path.sep in str(filepath):
            file_dir, file_name = os.path.split(str(filepath))
            filtered = FTIR_DataFrame[
                (FTIR_DataFrame.get("File Location", "") == file_dir)
                & (FTIR_DataFrame.get("File Name", "") == file_name)
            ]
        else:
            filtered = FTIR_DataFrame[
                FTIR_DataFrame.get("File Name", "") == str(filepath)
            ]
        # Exclude bad-quality rows
        try:
            filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass
        if filtered.empty:
            raise ValueError(f"No rows found for filepath '{filepath}'.")
    else:
        # Start with full DataFrame; Material/Conditions dropdowns will constrain
        filtered = FTIR_DataFrame.copy()
        try:
            filtered = filtered[_quality_good_mask(filtered)]
        except Exception:
            pass

    # Ensure destination columns exist for saving results
    results_col = "Deconvolution Results"
    x_ranges_col = "Deconvolution X-Ranges"
    if results_col not in FTIR_DataFrame.columns:
        FTIR_DataFrame[results_col] = None
    try:
        FTIR_DataFrame[results_col] = FTIR_DataFrame[results_col].astype(object)
    except Exception:
        pass

    # Insert X-ranges column adjacent to results when possible
    if x_ranges_col not in FTIR_DataFrame.columns:
        try:
            loc = int(list(FTIR_DataFrame.columns).index(results_col)) + 1
            FTIR_DataFrame.insert(loc, x_ranges_col, None)
        except Exception:
            FTIR_DataFrame[x_ranges_col] = None
    try:
        FTIR_DataFrame[x_ranges_col] = FTIR_DataFrame[x_ranges_col].astype(object)
    except Exception:
        pass

    def _persist_deconv_results(idx, components):
        """Store a deep copy of *components* in the DataFrame results column."""
        try:
            payload = copy.deepcopy(components)
        except Exception:
            try:
                payload = [dict(comp) for comp in components]
            except Exception:
                payload = components
        try:
            FTIR_DataFrame.at[idx, results_col] = payload
        except Exception:
            try:
                FTIR_DataFrame.loc[idx, results_col] = payload
            except Exception:
                pass

    def _persist_deconv_x_ranges(idx, x_ranges):
        """Store a deep copy of *x_ranges* in the DataFrame x-ranges column."""
        try:
            payload = copy.deepcopy(x_ranges)
        except Exception:
            payload = x_ranges
        try:
            FTIR_DataFrame.at[idx, x_ranges_col] = payload
        except Exception:
            try:
                FTIR_DataFrame.loc[idx, x_ranges_col] = payload
            except Exception:
                pass

    # Use shared _parse_seq helper (module-level)

    # Build dropdowns used for filtering and spectrum selection
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")
    try:
        norm_mask_init = filtered["Normalized and Corrected Data"].notna()
        filterable_df = filtered[norm_mask_init]
    except Exception:
        filterable_df = filtered
    unique_materials, unique_conditions = _extract_material_condition_lists(
        filterable_df, exclude_unexposed=True
    )
    material_dd = widgets.Dropdown(
        options=["any"] + unique_materials,
        value="any",
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + unique_conditions,
        value="any",
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )

    def _current_filtered_df():
        try:
            include_bad_flag = bool(getattr(include_bad_cb, "value", True))
        except Exception:
            include_bad_flag = True
        return _filter_spectra_dataframe(
            filtered,
            material=getattr(material_dd, "value", "any"),
            condition=getattr(conditions_dd, "value", "any"),
            include_bad=include_bad_flag,
            include_unexposed=True,
            normalized_column="Normalized and Corrected Data",
        )

    def _build_options_for_filters():
        df_filtered = _current_filtered_df()
        try:
            if "Time" in df_filtered.columns:
                df_filtered = df_filtered.copy()
                df_filtered["_sort_time"] = pd.to_numeric(
                    df_filtered["Time"], errors="coerce"
                ).fillna(float("inf"))
                df_filtered = df_filtered.sort_values(
                    by=["_sort_time"], kind="mergesort"
                )
                df_filtered = df_filtered.drop(columns=["_sort_time"], errors="ignore")
        except Exception:
            pass
        opts_local = []
        for idx, r in df_filtered.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            opts_local.append((label, idx))
        return opts_local

    options = _build_options_for_filters()
    if not options:
        raise ValueError("No spectra available after filtering.")

    # Seed from first spectrum
    first_idx = options[0][1]
    x0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("X-Axis"))
    y0 = _parse_seq(FTIR_DataFrame.loc[first_idx].get("Normalized and Corrected Data"))
    if x0 is None or y0 is None:
        raise ValueError(
            "Selected spectrum is missing 'X-Axis' or 'Normalized and Corrected Data'."
        )
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    xmin, xmax = (float(np.nanmin(x0)), float(np.nanmax(x0)))

    # Widgets (spectrum and global fit controls; full x-range is always used)
    spectrum_sel = widgets.Dropdown(
        options=options,
        value=first_idx,
        description="Spectrum",
        layout=widgets.Layout(width="60%"),
    )

    prev_spectrum_btn = widgets.Button(
        description="Previous spectrum",
        button_style="",
        tooltip="Select the previous spectrum in the Spectrum dropdown list.",
        layout=widgets.Layout(width="170px"),
    )
    next_spectrum_btn = widgets.Button(
        description="Next spectrum",
        button_style="",
        tooltip="Select the next spectrum in the Spectrum dropdown list.",
        layout=widgets.Layout(width="140px"),
    )

    spectrum_counter = widgets.HTML(
        value="",
        layout=widgets.Layout(width="70px"),
    )

    def _spectrum_dropdown_values():
        try:
            opts = list(spectrum_sel.options) if spectrum_sel.options else []
        except Exception:
            opts = []

        values = []
        for opt in opts:
            try:
                _lbl, _v = opt
            except Exception:
                _v = opt
            if _v is None:
                continue
            values.append(_v)
        return values

    def _update_spectrum_counter():
        values = _spectrum_dropdown_values()
        total = len(values)
        cur = getattr(spectrum_sel, "value", None)
        pos = 0
        if total and (cur in values):
            try:
                pos = values.index(cur) + 1
            except Exception:
                pos = 0
        try:
            spectrum_counter.value = f"<b>{pos}/{total}</b>"
        except Exception:
            pass

    def _step_spectrum_selection(delta: int):
        values = _spectrum_dropdown_values()
        if not values:
            return

        cur = getattr(spectrum_sel, "value", None)
        try:
            i = values.index(cur)
        except Exception:
            try:
                spectrum_sel.value = values[0]
            except Exception:
                pass
            return

        new_i = i + int(delta)
        if new_i < 0:
            new_i = 0
        elif new_i >= len(values):
            new_i = len(values) - 1

        if values[new_i] != cur:
            try:
                spectrum_sel.value = values[new_i]
            except Exception:
                pass
        try:
            _update_spectrum_counter()
        except Exception:
            pass

    prev_spectrum_btn.on_click(lambda _b=None: _step_spectrum_selection(-1))
    next_spectrum_btn.on_click(lambda _b=None: _step_spectrum_selection(1))

    def _refresh_spectrum_options_deconv(*_):
        opts_local = _build_options_for_filters()
        if not opts_local:
            spectrum_sel.options = [("<no spectra>", None)]
            try:
                spectrum_sel.value = None
            except Exception:
                pass
            try:
                _update_spectrum_counter()
            except Exception:
                pass
            return
        prev_val = spectrum_sel.value
        spectrum_sel.options = opts_local
        valid_values = [val for _, val in opts_local]
        if prev_val in valid_values:
            try:
                spectrum_sel.value = prev_val
            except Exception:
                pass
        else:
            try:
                spectrum_sel.value = valid_values[0]
            except Exception:
                pass
        try:
            _update_spectrum_counter()
        except Exception:
            pass

    material_dd.observe(_refresh_spectrum_options_deconv, names="value")
    conditions_dd.observe(_refresh_spectrum_options_deconv, names="value")
    include_bad_cb.observe(_refresh_spectrum_options_deconv, names="value")
    # Primary fit range (Range 1). Additional ranges can be added dynamically via a button.
    fit_range = widgets.FloatRangeSlider(
        value=[xmin, xmax],
        min=xmin,
        max=xmax,
        step=(xmax - xmin) / 1000 or 1.0,
        description="Range 1",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="90%"),
    )

    # Dynamic range slider list (initial contains Range 1). Limit to 3 total to mirror previous functionality.
    range_sliders = [fit_range]

    # Persist the most recently chosen Fit X-ranges across spectrum switches.
    # Stored as a list of (lo, hi) floats, length 1..3.
    persisted_fit_ranges = None

    add_range_btn = widgets.Button(
        description="Add another range",
        button_style="info",
        layout=widgets.Layout(width="180px"),
        tooltip="Insert an additional fit range (up to 3).",
    )

    def _make_range_slider(n: int):
        # n is 1-based index
        return widgets.FloatRangeSlider(
            value=[xmin, xmax],
            min=xmin,
            max=xmax,
            step=(xmax - xmin) / 1000 or 1.0,
            description=f"Range {n}",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="90%"),
        )

    def _clamp_ranges_to_span(ranges, x_min: float, x_max: float):
        """Clamp ranges to [x_min, x_max]; fall back to full-span if collapsed."""
        out = []
        if ranges is None:
            ranges = []
        for lo, hi in ranges:
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                continue
            lo_f, hi_f = (min(lo_f, hi_f), max(lo_f, hi_f))
            lo_f = max(float(x_min), lo_f)
            hi_f = min(float(x_max), hi_f)
            # If a range collapses after clamping, use the full span.
            if not (hi_f > lo_f):
                lo_f, hi_f = float(x_min), float(x_max)
            out.append((lo_f, hi_f))
            if len(out) >= 3:
                break
        if not out:
            out = [(float(x_min), float(x_max))]
        return out

    def _apply_ranges_to_sliders(ranges, x_min: float, x_max: float):
        """Ensure slider count matches ranges and apply limits/values."""
        nonlocal range_sliders
        safe = _clamp_ranges_to_span(ranges, x_min, x_max)

        target_n = max(1, min(3, len(safe)))
        # Remove extras
        try:
            while len(range_sliders) > target_n:
                range_sliders.pop()
        except Exception:
            pass
        # Add missing
        try:
            while len(range_sliders) < target_n:
                sl_new = _make_range_slider(len(range_sliders) + 1)
                range_sliders.append(sl_new)
                try:
                    sl_new.observe(_on_fit_range_change, names="value")
                except Exception:
                    pass
        except Exception:
            pass

        step = (x_max - x_min) / 1000 or 1.0
        for sl, (lo, hi) in zip(range_sliders, safe[:target_n]):
            try:
                sl.min = float(x_min)
                sl.max = float(x_max)
                sl.step = float(step)
                sl.value = [float(lo), float(hi)]
            except Exception:
                pass

        try:
            _renumber_range_sliders()
        except Exception:
            pass
        try:
            add_range_btn.disabled = len(range_sliders) >= 3
            add_range_btn.tooltip = (
                "Maximum of 3 ranges reached" if len(range_sliders) >= 3
                else "Insert an additional fit range (up to 3)."
            )
        except Exception:
            pass
        try:
            _refresh_range_box()
        except Exception:
            pass

    range_sliders_box = widgets.VBox([widgets.HBox([fit_range])])

    def _renumber_range_sliders():
        """Update slider descriptions to maintain contiguous numbering after add/remove."""
        try:
            for i, sl in enumerate(range_sliders, start=1):
                sl.description = f"Range {i}"
        except Exception:
            pass

    def _remove_range(slider):
        try:
            if slider in range_sliders and len(range_sliders) > 1:
                range_sliders.remove(slider)
                _renumber_range_sliders()
                # Re-enable add button if previously disabled due to max reached
                try:
                    if len(range_sliders) < 3:
                        add_range_btn.disabled = False
                        add_range_btn.tooltip = "Insert an additional fit range (up to 3)."
                except Exception:
                    pass
                _refresh_range_box()
                _on_fit_range_change()
        except Exception:
            pass

    def _refresh_range_box():
        try:
            rows = []
            multi = len(range_sliders) > 1
            for sl in range_sliders:
                if multi:
                    rm_btn = widgets.Button(
                        description="Remove Range",
                        button_style="warning",
                        layout=widgets.Layout(width="130px"),
                        tooltip="Remove this fit range",
                    )
                    def _bind(btn, s=sl):
                        try:
                            btn.on_click(lambda _b: _remove_range(s))
                        except Exception:
                            pass
                    _bind(rm_btn)
                    rows.append(widgets.HBox([sl, rm_btn]))
                else:
                    rows.append(widgets.HBox([sl]))
            range_sliders_box.children = rows
        except Exception:
            pass

    def _add_range(_b=None):
        if len(range_sliders) >= 3:
            try:
                add_range_btn.disabled = True
                add_range_btn.tooltip = "Maximum of 3 ranges reached"
            except Exception:
                pass
            return
        idx_new = len(range_sliders) + 1
        sl = _make_range_slider(idx_new)
        range_sliders.append(sl)
        try:
            sl.observe(_on_fit_range_change, names="value")
        except Exception:
            pass
        _renumber_range_sliders()
        _refresh_range_box()
        if len(range_sliders) >= 3:
            try:
                add_range_btn.disabled = True
                add_range_btn.tooltip = "Maximum of 3 ranges reached"
            except Exception:
                pass
        _on_fit_range_change()

    add_range_btn.on_click(_add_range)
    # Per-peak default seeds (no global center/sigma controls)
    PER_PEAK_DEFAULT_CENTER_WINDOW = 5.0
    PER_PEAK_DEFAULT_SIGMA = 10.0
    SESSION_WINDOW_MARGIN = 1.0  # cm⁻¹ margin used when sizing session peak window vs existing peaks
    # Defaults for reset operations
    DEFAULT_ALPHA = 0.5
    DEFAULT_INCLUDE = True
    CENTER_MODE_WINDOW = "Window"
    CENTER_MODE_EXACT = "Exact"
    CENTER_MODE_OPTIONS = [
        (CENTER_MODE_WINDOW, CENTER_MODE_WINDOW),
        (CENTER_MODE_EXACT, CENTER_MODE_EXACT),
    ]

    def _normalize_center_mode_value(val):
        mode = str(val).strip() if isinstance(val, str) else ""
        if not mode:
            return CENTER_MODE_WINDOW
        lowered = mode.lower()
        if lowered == "auto":  # legacy
            return CENTER_MODE_WINDOW
        if lowered == "manual":  # legacy
            return CENTER_MODE_EXACT
        if lowered == CENTER_MODE_WINDOW.lower():
            return CENTER_MODE_WINDOW
        if lowered == CENTER_MODE_EXACT.lower():
            return CENTER_MODE_EXACT
        return CENTER_MODE_WINDOW
    default_fit_range_value = (float(fit_range.value[0]), float(fit_range.value[1]))
    # Fit button (manual trigger for fitting)
    fit_btn = widgets.Button(
        description="Fit",
        tooltip="Run fit using current settings",
        button_style="primary",
        layout=widgets.Layout(width="80px"),
    )
    add_peaks_btn = widgets.Button(
        description="Add peaks",
        tooltip="Add new peak(s)",
        button_style="info",
        layout=widgets.Layout(width="110px"),
    )
    # Dedicated Delete peaks button (wired later)
    delete_peaks_btn = widgets.Button(
        description="Delete peaks",
        button_style="warning",
        layout=widgets.Layout(width="120px"),
    )
    accept_new_peaks_btn = widgets.Button(
        description="Accept new peak(s)",
        button_style="success",
        layout=widgets.Layout(width="150px"),
    )
    # Iterative correction button to minimize reduced chi-square via coordinate descent
    iter_btn = widgets.Button(
        description="Optimize α values",
        button_style="info",
        layout=widgets.Layout(width="185px"),
        tooltip=(
            "Iteratively adjust included peaks' α values to minimize error."
        ),
    )
    cancel_fit_btn = widgets.Button(
        description="Cancel Fit",
        button_style="danger",
        layout=widgets.Layout(width="150px"),
        tooltip="Interrupt the current fit or optimization run",
    )
    redo_new_peaks_btn = widgets.Button(
        description="Redo new peaks",
        button_style="warning",
        layout=widgets.Layout(width="140px"),
    )
    cancel_new_peaks_btn = widgets.Button(
        description="Cancel peak addition",
        button_style="danger",
        layout=widgets.Layout(width="190px"),
    )
    save_btn = widgets.Button(
        description="Save for spectrum",
        button_style="success",
        tooltip="Save deconvolution results for the current spectrum",
    )
    close_btn = widgets.Button(description="Close", button_style="danger")
    # Delete-mode action buttons (shown only in delete mode)
    accept_deletions_btn = widgets.Button(
        description="Accept deletions",
        button_style="success",
        layout=widgets.Layout(width="160px"),
    )
    redo_deletions_btn = widgets.Button(
        description="Redo deletions",
        button_style="warning",
        layout=widgets.Layout(width="140px"),
    )
    cancel_deletions_btn = widgets.Button(
        description="Cancel deletions",
        button_style="danger",
        layout=widgets.Layout(width="170px"),
    )
    # Dedicated status label to avoid Output-widget buffering issues
    status_html = widgets.HTML(value="")
    optimize_status_html = widgets.HTML(value="")
    # Replace Output-based logging with a single HTML widget to avoid renderer
    # double-echo issues in some notebook front-ends.
    log_html = widgets.HTML(value="")
    # De-duplicate log messages: avoid echoing the same text twice in quick succession
    # Make this thread-safe since some logs originate from worker threads.
    last_msg_text = ""
    last_msg_ts = 0.0
    log_lock = threading.Lock()
    # --- Change tracking for session summary (on Close) ---
    _deconv_changes = {
        "saved": [],  # list[(idx, count_components)]
        "quality": [],  # list[(idx, new_quality)]
        "iter": [],  # list[(idx, start_redchi, final_redchi, improvements)]
    }

    def _log_once(message: str, *, wait: bool = True, clear: bool = True):
        nonlocal last_msg_text, last_msg_ts
        # Normalize message to minimize false negatives due to stray whitespace
        msg_norm = str(message).rstrip()
        try:
            now = time.time()
        except Exception:
            now = 0.0
        # Use a small lock to serialize dedup checks across threads
        with log_lock:
            # Suppress duplicates of identical text within ~1.0s window
            if msg_norm == last_msg_text and (now - last_msg_ts) < 1.0:
                return
            # Update the HTML log widget directly to avoid Output buffering quirks.
            try:
                # Escape to prevent unintended HTML rendering; keep it simple text.
                safe = html.escape(msg_norm)
                log_html.value = f"<div style='font-family:monospace; white-space:pre-wrap;'>{safe}</div>"
            except Exception:
                # As a last resort, try printing to the notebook output
                try:
                    print(msg_norm)
                except Exception:
                    pass
            last_msg_text = msg_norm
            last_msg_ts = now

    def _append_status(message: str):
        """Append a status line (monospace) without clearing previous messages.

        Keeps a running log in the log_html widget so multiple actions are visible.
        """
        try:
            msg_norm = html.escape(str(message).rstrip())
        except Exception:
            msg_norm = str(message)
        try:
            prev = str(getattr(log_html, 'value', ''))
            # Ensure container div exists; append a new line
            if not prev:
                log_html.value = f"<div style='font-family:monospace; white-space:pre-wrap;'>{msg_norm}</div>"
            else:
                # Insert before closing div tag when present
                if prev.endswith("</div>"):
                    log_html.value = prev[:-6] + "\n" + msg_norm + "</div>"
                else:
                    log_html.value = prev + "\n" + msg_norm
        except Exception:
            # Fallback to print if HTML update fails
            try:
                print(message)
            except Exception:
                pass

    # Reset buttons for globals and all
    reset_all_btn = widgets.Button(
        description="Reset all peak parameters to defaults",
        button_style="warning",
        tooltip="Reset all per-peak sliders and selections to defaults",
        layout=widgets.Layout(width="340px"),
    )

    # Cancellation/interrupt support for long-running fits
    # Global cancellation token for long-running workflows (iteration).
    # Note: Do NOT replace this Event with a new instance elsewhere; it must remain
    # stable so a single set() call cancels the entire workflow.
    cancel_event = threading.Event()
    fit_thread = None
    iter_thread = None
    fit_cancel_token = None  # points to the active fit's cancel event, if any
    iterating_in_progress = False  # suppress per-fit status updates while iterating
    cancel_fit_btn_frozen = False  # suppress show/hide toggles to prevent flicker
    # Snapshot of reduced chi-square at the moment the user clicks Iterate
    iter_start_redchi = None
    # Final reduced chi-square once iteration completes and a flag to show a summary
    iter_final_redchi = None
    iter_summary_pending = False
    # Helper to marshal UI updates back onto the notebook's main IOLoop
    try:
        from tornado.ioloop import IOLoop as _IOLoop
    except Exception:
        _IOLoop = None
    # Capture the main thread's IOLoop now; using current() inside a worker thread
    # can create a new, non-running loop which would drop callbacks.
    try:
        _MAIN_IOLOOP = _IOLoop.current() if _IOLoop is not None else None
    except Exception:
        _MAIN_IOLOOP = None

    def _on_main_thread(fn, *args, **kwargs):
        """Schedule *fn* to run on the notebook's main UI thread when possible."""
        try:
            if _MAIN_IOLOOP is not None:
                _MAIN_IOLOOP.add_callback(lambda: fn(*args, **kwargs))
                return
        except Exception:
            pass
        try:  # Fall back to asyncio event loop used by ipykernel (JupyterLab/VS Code)
            import asyncio

            loop = asyncio.get_event_loop()
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(fn, *args, **kwargs)
                return
        except Exception:
            pass
        # As a last resort, execute immediately (may succeed in synchronous shells)
        try:
            fn(*args, **kwargs)
        except Exception:
            pass

    # Dynamic per-peak controls: include checkbox + per-peak parameter sliders/locks
    alpha_sliders = []  # list[widgets.FloatSlider] (fraction α)
    include_checkboxes = []  # list[widgets.Checkbox]
    center_sliders = []  # list[widgets.FloatSlider] direct center (μ)
    # Map the ordering of per-peak sliders to original peak indices. The existing logic
    # builds slider lists in ascending order of original peak indices, but later fit
    # code incorrectly indexes sliders with the original peak index (which may be
    # sparse), causing parameter tweaks to be ignored. We capture the ordered list
    # here and keep it updated on rebuild so we can translate original peak indices
    # to slider positions reliably.
    center_slider_peak_indices = []  # list[int] parallel to center_sliders
    # Added lists for peak label/header synchronization post-fit
    peak_label_widgets = []  # list[widgets.HTML] per-peak headers shown in UI
    original_peak_centers = []  # list[float] detected (pre-fit) peak centers
    last_active_ranges = []  # list[(lo, hi)] currently active fit ranges for header range status
    sigma_sliders = []  # list[widgets.FloatSlider]
    amplitude_sliders = []  # list[widgets.FloatSlider] amplitude A
    center_window_sliders = []  # list[widgets.FloatSlider] center ± window width
    # Mode toggles (Auto vs Manual) for A, μ, σ; alpha is always Manual per spec
    amplitude_mode_toggles = []  # list[widgets.ToggleButtons]
    center_mode_toggles = []  # list[widgets.ToggleButtons]
    sigma_mode_toggles = []  # list[widgets.ToggleButtons]
    # Legacy lock checkboxes retained as empty lists for backward compatibility with existing logic
    lock_alpha_checkboxes = []
    lock_center_checkboxes = []
    lock_sigma_checkboxes = []
    peak_controls_box = widgets.VBox([])
    # Container (previously Accordion) for peak sections; now simple VBox to avoid nested blank accordion
    peak_accordion = widgets.VBox([])
    globals()['peak_accordion'] = peak_accordion
    excluded_peaks_html = widgets.HTML(value="")
    # Map original peak index -> accordion position for title updates after fit
    included_index_map = {}
    # Container displayed in UI
    peak_controls_section = widgets.VBox([peak_controls_box])
    # Per-spectrum peak center bookkeeping: maps spectrum idx -> {peak_idx: {'initial': float, 'user': float, 'fit': float | None}}
    peak_center_state_by_idx = {}
    # Track which peaks are currently rendered so range changes can diff efficiently
    previous_in_range_indices: set[int] = set()

    # Guard to suppress redundant fits during bulk programmatic updates
    bulk_update_in_progress = False
    # Lightweight reentrancy/debounce guards to avoid duplicate callbacks
    on_spectrum_change_inflight = False
    last_on_spectrum_change_ts = 0.0
    fit_update_inflight = False
    last_fit_update_ts = 0.0

    # Lightweight per-button click de-bounce to avoid double-firing handlers
    last_click_ts = {}

    def _recent_click(key: str, min_interval: float = 0.35) -> bool:
        try:
            now = time.time()
        except Exception:
            now = 0.0
        last = last_click_ts.get(key, 0.0)
        if (now - last) < min_interval:
            return True
        last_click_ts[key] = now
        return False

    def _snapshot_current_controls():
        """Best-effort snapshot of current UI widget values.

        Important: this is intentionally *not* used to carry state between spectra.
        It exists to keep internal bookkeeping consistent (e.g., during range
        rebuilds or when lazy widgets materialize).
        """
        try:
            idx = spectrum_sel.value
        except Exception:
            idx = None
        if idx is None:
            return
        # Capture current center slider values into the per-spectrum state for this
        # currently-selected idx only (the dict is cleared on spectrum change).
        try:
            state = peak_center_state_by_idx.setdefault(idx, {})
            for pos, sl in enumerate(center_sliders):
                try:
                    orig_idx = (
                        center_slider_peak_indices[pos]
                        if pos < len(center_slider_peak_indices)
                        else pos
                    )
                except Exception:
                    orig_idx = pos
                try:
                    v = float(getattr(sl, 'value', np.nan))
                except Exception:
                    v = getattr(sl, 'value', None)
                entry = state.setdefault(orig_idx, {})
                if entry.get('initial') is None:
                    entry['initial'] = v
                entry['user'] = v
        except Exception:
            pass



    def _on_include_toggle(*_):
        """Handle include checkbox toggles without rebuilding the per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                # Still snapshot state so iteration worker can see updated flags
                _snapshot_current_controls()
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>Include toggled. Click Fit to update.</span>"
            )
        except Exception:
            pass

    def _on_alpha_change(*_):
        """Handle alpha slider changes without rebuilding the per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>α changed. Click Fit to update.</span>"
            )
        except Exception:
            pass

    def _on_center_sigma_change(*_):
        """Handle center window/initial sigma changes without rebuilding per-peak UI."""
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        _snapshot_current_controls()
        try:
            status_html.value = (
                "<span style='color:#555;'>Parameters changed. Click Fit to "
                "update.</span>"
            )
        except Exception:
            pass

    def _format_center_value(val):
        try:
            v = float(val)
        except Exception:
            return "--"
        try:
            if 'np' in globals():
                if bool(np.isfinite(v)):
                    return f"{v:.1f}"
                return "--"
        except Exception:
            pass
        try:
            import math

            if math.isfinite(v):
                return f"{v:.1f}"
        except Exception:
            pass
        return "--"

    def _set_peak_initial_center(spectrum_idx, peak_idx, value):
        state = peak_center_state_by_idx.setdefault(spectrum_idx, {})
        peak_state = state.setdefault(peak_idx, {})
        try:
            peak_state['initial'] = float(value)
        except Exception:
            peak_state['initial'] = value
        if peak_state.get('user') is None:
            peak_state['user'] = peak_state.get('initial')

    def _set_peak_user_center(spectrum_idx, peak_idx, value):
        state = peak_center_state_by_idx.setdefault(spectrum_idx, {})
        peak_state = state.setdefault(peak_idx, {})
        try:
            peak_state['user'] = float(value)
        except Exception:
            peak_state['user'] = value

    def _set_peak_fit_center(spectrum_idx, peak_idx, value):
        state = peak_center_state_by_idx.setdefault(spectrum_idx, {})
        peak_state = state.setdefault(peak_idx, {})
        try:
            peak_state['fit'] = float(value)
        except Exception:
            peak_state['fit'] = value

    def _update_peak_toggle_label(peak_idx: int, *, spectrum_idx=None):
        if spectrum_idx is None:
            try:
                spectrum_idx = spectrum_sel.value
            except Exception:
                spectrum_idx = current_idx_deconv
        if spectrum_idx is None:
            return
        state = peak_center_state_by_idx.get(spectrum_idx, {})
        peak_state = state.get(peak_idx, {})
        user_val = peak_state.get('user', peak_state.get('initial'))
        fit_val = peak_state.get('fit')
        if fit_val is None:
            fit_val = user_val
        entry = None
        try:
            if isinstance(peak_box_cache, dict):
                entry = peak_box_cache.get(peak_idx)
        except Exception:
            entry = None
        if not entry:
            return
        toggle = entry.get('toggle')
        if toggle is None:
            return
        toggle.description = (
            f"Peak {peak_idx + 1} @ {_format_center_value(user_val)} → "
            f"{_format_center_value(fit_val)} cm⁻¹"
        )

    def _bind_center_slider(slider, peak_idx, spectrum_idx):
        def _on_change(change):
            try:
                new_val = change.get('new')
            except Exception:
                new_val = None
            _set_peak_user_center(spectrum_idx, peak_idx, new_val)
            _update_peak_toggle_label(peak_idx, spectrum_idx=spectrum_idx)
            _on_center_sigma_change(change)

        try:
            slider.observe(_on_change, names="value")
        except Exception:
            pass

    def _on_fit_range_change(*_):
        """Only Fit X-range changes should rebuild the per-peak controls."""
        nonlocal persisted_fit_ranges
        try:
            if iterating_in_progress or bulk_update_in_progress:
                return
        except Exception:
            pass
        # Snapshot first so states persist across rebuild
        _snapshot_current_controls()
        try:
            persisted_fit_ranges = _current_fit_ranges()
        except Exception:
            pass
        try:
            _update_fit_range_indicator()
        except Exception:
            pass
        try:
            idx = spectrum_sel.value
            _refresh_peak_control_widgets(idx)
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#555;'>Fit range changed. Click Fit to "
                "update.</span>"
            )
        except Exception:
            pass

    # Track last reduced chi-square per spectrum to report refit deltas
    last_redchi_by_idx = {}
    # Store last successful fit result per spectrum for Save action
    last_result_by_idx = {}

    # Plot figure: data, fit, components (dynamic)
    fig = go.FigureWidget()
    fig.add_scatter(x=x0.tolist(), y=y0.tolist(), mode="lines", name="Data (Norm+Corr)")
    fig.add_scatter(
        x=[], y=[], mode="lines", name="Composite Fit", line=dict(color="red")
    )
    fig.update_layout(
        title="Peak Deconvolution (Pseudo-Voigt)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance (AU)",
    )

    # --- Add-peaks mode state ---
    adding_mode = False
    new_peak_xs = []  # temporary stash of user-clicked x positions (snapped to grid)
    new_peak_windows = {}  # x_new -> effective session window (min(default, min_dist_to_existing - margin))

    def _hide(w):
        try:
            w.layout.display = "none"
        except Exception:
            pass

    def _show(w):
        try:
            w.layout.display = ""
        except Exception:
            pass

    def _hide_controls_during_fit_or_opt():
        """Hide the same controls during Fit and optimization."""
        try:
            _hide(fit_btn)
        except Exception:
            pass
        try:
            _hide(add_peaks_btn)
        except Exception:
            pass
        try:
            _hide(canonize_btn)
        except Exception:
            pass
        try:
            _hide(canonize_confirm_btn)
        except Exception:
            pass
        try:
            _hide(canonize_cancel_btn)
        except Exception:
            pass
        try:
            _hide(load_canon_btn)
        except Exception:
            pass
        try:
            delete_peaks_btn.layout.display = "none"
        except Exception:
            pass
        try:
            _hide(iter_btn)
        except Exception:
            pass
        try:
            _hide(reset_all_row)
        except Exception:
            pass
        try:
            _hide(save_btn)
        except Exception:
            pass
        try:
            _hide(fit_range_row)
        except Exception:
            pass

    def _show_controls_after_fit_or_opt():
        """Restore controls hidden by _hide_controls_during_fit_or_opt."""
        try:
            _show(fit_btn)
        except Exception:
            pass
        try:
            _show(add_peaks_btn)
        except Exception:
            pass
        try:
            _show(canonize_btn)
        except Exception:
            pass
        try:
            _hide(canonize_confirm_btn)
        except Exception:
            pass
        try:
            _hide(canonize_cancel_btn)
        except Exception:
            pass
        try:
            _show(load_canon_btn)
        except Exception:
            pass
        try:
            delete_peaks_btn.layout.display = ""
        except Exception:
            pass
        try:
            _show(iter_btn)
        except Exception:
            pass
        try:
            _show(reset_all_row)
        except Exception:
            pass
        try:
            _show(save_btn)
        except Exception:
            pass
        try:
            _show(fit_range_row)
        except Exception:
            pass

    def _set_cancel_button_mode(mode: str = "fit"):
        label = "Cancel Fit"
        tip = "Interrupt the current fit."
        if str(mode).lower() == "optimize":
            label = "Cancel Optimize"
            tip = "Stop the running optimization sweep."
        try:
            cancel_fit_btn.description = label
            cancel_fit_btn.tooltip = tip
        except Exception:
            pass

    _set_cancel_button_mode("fit")

    def _show_cancel_button():
        # Try several display hints so the widget becomes visible across front-ends
        for value in ("", "flex", "inline-flex", None):
            try:
                cancel_fit_btn.layout.display = value
                break
            except Exception:
                continue
        try:
            cancel_fit_btn.layout.visibility = "visible"
        except Exception:
            pass
        # Ensure the containing row is visible
        try:
            cancel_row.layout.display = ""
        except Exception:
            pass

    def _hide_cancel_button():
        try:
            cancel_fit_btn.layout.display = "none"
        except Exception:
            pass
        try:
            cancel_fit_btn.layout.visibility = "hidden"
        except Exception:
            pass
        # Hide the containing row when not needed
        try:
            cancel_row.layout.display = "none"
        except Exception:
            pass

    # Show/Hide and enablement for the Cancel Fit button based on active work
    def _update_cancel_fit_visibility():
        # If an iterative optimization is in progress or visibility is frozen,
        # keep the button visible and enabled without toggling to avoid flicker
        # between the many short internal fits.
        try:
            if iterating_in_progress or cancel_fit_btn_frozen:
                try:
                    _set_cancel_button_mode("optimize")
                except Exception:
                    pass
                try:
                    _show_cancel_button()
                    cancel_fit_btn.disabled = False
                except Exception:
                    pass
                return  # never hide while iterating/frozen
        except Exception:
            pass
        active = False
        try:
            active = (fit_thread is not None and fit_thread.is_alive()) or (
                iter_thread is not None and iter_thread.is_alive()
            )
        except Exception:
            pass
        try:
            _set_cancel_button_mode("optimize" if active and iterating_in_progress else "fit")
        except Exception:
            pass
        if active:
            try:
                cancel_fit_btn.disabled = False
            except Exception:
                pass
            _show_cancel_button()
        else:
            try:
                cancel_fit_btn.disabled = True
            except Exception:
                pass
            _hide_cancel_button()

    # Explicit helpers to force show/hide independent of thread state.
    # These provide a deterministic UI state when fits/iterations start or end,
    # avoiding races where the thread reference may still appear alive briefly.
    def _force_cancel_fit_shown():
        try:
            _set_cancel_button_mode("optimize" if iterating_in_progress else "fit")
        except Exception:
            pass
        try:
            cancel_fit_btn.disabled = False
        except Exception:
            pass
        _show_cancel_button()

    def _force_cancel_fit_hidden():
        # Do not hide while an iteration is active or visibility frozen
        try:
            if iterating_in_progress or cancel_fit_btn_frozen:
                return
        except Exception:
            pass
        try:
            _set_cancel_button_mode("fit")
        except Exception:
            pass
        try:
            cancel_fit_btn.disabled = True
        except Exception:
            pass
        _hide_cancel_button()

    # hide action buttons initially
    _hide(accept_new_peaks_btn)
    _hide(redo_new_peaks_btn)
    _hide(cancel_new_peaks_btn)
    _hide_cancel_button()

    def _clear_add_peak_shapes():
        # Remove only our temporary marker shapes
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            shapes = [
                s for s in shapes if getattr(s, "name", None) != "add_peak_marker"
            ]
            fig.layout.shapes = tuple(shapes)
        except Exception:
            try:
                # When shapes are plain dicts
                shapes = list(getattr(fig.layout, "shapes", ()))
                new_shapes = []
                for s in shapes:
                    try:
                        if s.get("name") != "add_peak_marker":
                            new_shapes.append(s)
                    except Exception:
                        new_shapes.append(s)
                fig.layout.shapes = tuple(new_shapes)
            except Exception:
                pass

    # Click handler for adding peaks when in adding mode (desktop / non-Colab)
    def _on_data_click(trace, points, selector):
        # Only respond when adding mode is ON and a valid click occurred
        try:
            nonlocal adding_mode
            if not adding_mode:
                return
            if not points or not getattr(points, "xs", None):
                return
            x_clicked = float(points.xs[0])
        except Exception:
            return

        # Get current spectrum arrays
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            _log_once("Cannot add peak: current spectrum has no normalized data.")
            return

        # Snap to nearest x
        try:
            nearest_i = int(np.argmin(np.abs(x_arr - x_clicked)))
        except Exception:
            _log_once("Could not determine nearest x for the clicked location.")
            return
        x_new = float(x_arr[nearest_i])
        # Reject if within any existing visible peak's per-peak Center ± window
        try:
            vis_xs, _vis_ys = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        try:
            default_win = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            default_win = 0.0
        try:
            for i, cx in enumerate(vis_xs or []):
                try:
                    w_i = float(center_window_sliders[i].value)
                except Exception:
                    w_i = default_win
                # Enforce minimum separation of 2× the per-peak Center ± window
                if abs(float(cx) - float(x_new)) <= (2.0 * abs(w_i)):
                    try:
                        status_html.value = (
                            f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is too close to existing peak @ {cx:.3f}. "
                            f"Must be ≥ 2×(Center ± window) = {2.0*abs(w_i):.1f} cm⁻¹ away. "
                            f"Adjust selection or that peak’s Center ± window.</span>"
                        )
                    except Exception:
                        _log_once(
                            f"Rejected: {x_new:.3f} cm⁻¹ is too close to existing peak @ {cx:.3f}. "
                            f"Must be ≥ 2×(Center ± window) = {2.0*abs(w_i):.1f} cm⁻¹ away. "
                            f"Adjust selection or that peak’s Center ± window."
                        )
                    return
        except Exception:
            pass
        # Enforce minimum separation between newly selected peaks as well.
        # Use 2× the (± window) as the required center-to-center spacing.
        try:
            candidate_w = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            candidate_w = 0.0
        for existing_x in new_peak_xs:
            try:
                existing_w = float(new_peak_windows.get(existing_x, candidate_w))
            except Exception:
                existing_w = candidate_w
            required_sep = 2.0 * max(abs(existing_w), abs(candidate_w))
            if abs(existing_x - x_new) <= required_sep:
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Rejected: {x_new:.3f} cm⁻¹ is "
                        f"too close to another selected peak ({existing_x:.3f}). "
                        f"Must be ≥ {required_sep:.2f} cm⁻¹ away (2× window). "
                        f"Tip: reduce the per-peak Center ± slider if you need peaks closer.</span>"
                    )
                except Exception:
                    _log_once(
                        f"Rejected: {x_new:.3f} cm⁻¹ is too close to another selected peak "
                        f"({existing_x:.3f}). Must be ≥ {required_sep:.2f} cm⁻¹ away (2× window). "
                        f"Tip: reduce the per-peak Center ± slider if you need peaks closer."
                    )
                return
        # Compute an effective session window so the new peak's window won't cover existing peaks
        try:
            xs_existing, _ys_existing = _get_peaks(idx)
        except Exception:
            xs_existing = []
        try:
            if xs_existing:
                min_dist = min(abs(float(xe) - x_new) for xe in xs_existing)
            else:
                min_dist = float('inf')
        except Exception:
            min_dist = float('inf')
        try:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
            if np.isfinite(min_dist):
                eff_window = min(eff_window, max(0.0, float(min_dist) - float(SESSION_WINDOW_MARGIN)))
        except Exception:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        try:
            new_peak_windows[x_new] = float(eff_window)
        except Exception:
            pass
        # Record and draw a vertical line marker
        new_peak_xs.append(x_new)
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        try:
            fig.add_shape(
                dict(
                    type="line",
                    x0=x_new,
                    x1=x_new,
                    y0=y_min,
                    y1=y_max,
                    line=dict(color="#ff7f0e", dash="dot", width=1.5),
                    name="add_peak_marker",
                )
            )
        except Exception:
            pass
        _log_once(
            f"Selected new peak at x = {x_new:.3f} cm⁻¹. Click more points, or 'Accept "
            f"new peaks'."
        )

    # Register click handler only outside Colab; Colab uses slider-based fallback
    try:
        if not _IN_COLAB:
            fig.data[0].on_click(_on_data_click)
    except Exception:
        pass

    def _get_xy(row_idx):
        r = FTIR_DataFrame.loc[row_idx]
        x = _parse_seq(r.get("X-Axis"))
        y = _parse_seq(r.get("Normalized and Corrected Data"))
        if x is None or y is None:
            return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            return None, None
        return x_arr, y_arr

    def _get_peaks(row_idx):
        # Read peaks solely from DataFrame.
        r = FTIR_DataFrame.loc[row_idx]
        xs = _parse_seq(r.get("Peak Wavenumbers"))
        ys = _parse_seq(r.get("Peak Absorbances"))
        if xs is None or ys is None:
            return [], []
        try:
            xs = list(xs)
            ys = list(ys)
        except Exception:
            return [], []
        if len(xs) != len(ys):
            return [], []
        # Sort peaks by wavenumber (ascending)
        try:
            pairs = sorted(zip(xs, ys), key=lambda t: float(t[0]))
            xs_sorted, ys_sorted = [list(t) for t in zip(*pairs)] if pairs else ([], [])
            return xs_sorted, ys_sorted
        except Exception:
            return xs, ys

    def _current_fit_range():
        try:
            lo, hi = fit_range.value
            return float(min(lo, hi)), float(max(lo, hi))
        except Exception:
            return xmin, xmax
    # Multi-range support helpers
    def _current_fit_ranges():
        """Return list of active (lo, hi) ranges from dynamic slider list."""
        ranges = []
        for sl in range_sliders:
            try:
                lo, hi = sl.value
                ranges.append((float(min(lo, hi)), float(max(lo, hi))))
            except Exception:
                continue
        if not ranges:
            ranges.append((xmin, xmax))
        return ranges
    def _overall_fit_span():
        rs = _current_fit_ranges()
        lows = [r[0] for r in rs]
        highs = [r[1] for r in rs]
        return (float(min(lows)), float(max(highs)))

    def _get_visible_peaks(row_idx):
        xs, ys = _get_peaks(row_idx)
        if not xs:
            return [], []
        ranges = _current_fit_ranges() if ' _current_fit_ranges' in globals() or True else [(lo, hi)]
        xs_f = []
        ys_f = []
        for cx, cy in zip(xs, ys):
            try:
                cxv = float(cx)
            except Exception:
                continue
            if any(lo <= cxv <= hi for lo, hi in ranges):
                xs_f.append(cxv)
                try:
                    ys_f.append(float(cy))
                except Exception:
                    ys_f.append(float("nan"))
        return xs_f, ys_f

    # Visual indicator of the current Fit X-range on the plot
    def _update_fit_range_indicator():
        try:
            idx = spectrum_sel.value
        except Exception:
            return
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            return
        try:
            ranges = _current_fit_ranges()
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
        except Exception:
            return
        rects = []
        for ridx, (lo, hi) in enumerate(ranges):
            rects.append(dict(
                type="rect",
                x0=float(min(lo, hi)),
                x1=float(max(lo, hi)),
                y0=y0_min,
                y1=y0_max,
                fillcolor="rgba(0,120,215,0.12)",
                line=dict(color="rgba(0,120,215,0.6)", width=1),
                layer="below",
                name=f"fit_range_rect_{ridx}",
            ))
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            new_shapes = []
            for s in shapes:
                try:
                    # Skip any prior fit range shape; keep others
                    # (e.g., add_peak_marker)
                    nm = getattr(s, "name", None)
                except Exception:
                    nm = None
                if nm is None:
                    try:
                        nm = s.get("name")
                    except Exception:
                        nm = None
                if nm and nm.startswith("fit_range_rect"):
                    continue
                new_shapes.append(s)
            new_shapes.extend(rects)
            fig.layout.shapes = tuple(new_shapes)
        except Exception:
            # Best-effort; ignore if shapes unavailable
            pass

    def _refresh_peak_control_widgets(row_idx):
        """Refresh peak parameter widgets for the given spectrum, reusing cached controls when ranges change."""
        nonlocal alpha_sliders, include_checkboxes, center_sliders, sigma_sliders, amplitude_sliders, center_window_sliders
        nonlocal amplitude_mode_toggles, center_mode_toggles, sigma_mode_toggles
        nonlocal lock_alpha_checkboxes, lock_center_checkboxes, lock_sigma_checkboxes
        nonlocal peak_center_state_by_idx, previous_in_range_indices
        nonlocal center_slider_peak_indices
        # Build controls only for peaks within any active Fit X-range (lazy creation).
        # Out-of-range peak widgets (headers) are created only when the excluded toggle is enabled.
        try:
            # Show transient building message while list is (re)constructed.
            peak_controls_box.children = [
                widgets.HTML("<b>Building Peak List . . .</b>")
            ]
        except Exception:
            pass
        peaks_x_all, peaks_y_all = _get_peaks(row_idx)
        state_for_spectrum = peak_center_state_by_idx.setdefault(row_idx, {})
        valid_indices = set(range(len(peaks_x_all)))
        for _idx_key in list(state_for_spectrum.keys()):
            if _idx_key not in valid_indices:
                try:
                    del state_for_spectrum[_idx_key]
                except Exception:
                    pass
        for _idx_local, _cx_val in enumerate(peaks_x_all):
            _set_peak_initial_center(row_idx, _idx_local, _cx_val)
        try:
            original_peak_centers[:] = [float(_cx) for _cx in peaks_x_all]
        except Exception:
            original_peak_centers[:] = list(peaks_x_all) if peaks_x_all else []
        alpha_sliders = []
        include_checkboxes = []
        center_sliders = []
        sigma_sliders = []
        amplitude_sliders = []
        center_window_sliders = []
        amplitude_mode_toggles = []
        center_mode_toggles = []
        sigma_mode_toggles = []
        lock_alpha_checkboxes = []
        lock_center_checkboxes = []
        lock_sigma_checkboxes = []
        children = []
        if not peaks_x_all:
            peak_controls_box.children = [
                widgets.HTML(
                    "<b>No peaks found.</b> Run find_peak_info first."
                )
            ]
            return
        try:
            active_ranges = _current_fit_ranges()
        except Exception:
            active_ranges = [(float("-inf"), float("inf"))]
        # Do not restore per-spectrum/group-level settings; start from defaults
        saved_alphas = None
        saved_includes = None
        saved_center = None
        saved_sigma = None
        saved_amplitude = None
        saved_modes = {'amplitude': None, 'center': None, 'sigma': None}
        saved_center_window = None
        saved_locks = {'alpha': None, 'center': None, 'sigma': None}
        if saved_center:
            try:
                for _idx_local, _val in enumerate(saved_center):
                    if _idx_local in valid_indices:
                        _set_peak_user_center(row_idx, _idx_local, _val)
            except Exception:
                pass
        # No group-level fallbacks; rely on defaults each rebuild
        if not isinstance(saved_locks, dict):
            saved_locks = {'alpha': None, 'center': None, 'sigma': None}
        included_boxes = []  # will be used only for full rebuild path
        excluded_boxes = []
        # Cache for previously built in-range peak widgets so we can reuse them on subsequent rebuilds
        # (avoids re-instantiating many sliders when only the Fit X-ranges changed).
        # Persist across calls inside the closure scope.
        global peak_box_cache
        if not isinstance(peak_box_cache, dict):
            peak_box_cache = {}
        # Never reuse cached widget objects across different spectra.
        nonlocal cache_owner_idx
        try:
            if cache_owner_idx != row_idx:
                peak_box_cache.clear()
                cache_owner_idx = row_idx
                previous_in_range_indices = set()
        except Exception:
            cache_owner_idx = row_idx

        # --- Parameter explanations for the toggle ---
        param_details_text = (
            "α: Inversely relates to the steepness of the peak. Fractional composition of Gaussian (0) and Lorentzian (1) components.\n"
            "μ: Center position of the peak (wavenumber). Window mode varies μ within ± the window; Exact mode fixes μ at the slider value.\n"
            "A: Relates to the height of the peak.\n"
            "σ: Relates to the width of the peak. The FWHM is 2σ.\n"
        )

        def _make_param_details_row():
            toggle = widgets.ToggleButton(
                value=False,
                description="Parameter Details",
                button_style="info",
                icon="chevron-down",
                layout=widgets.Layout(width="180px", margin="0 0 0 10px"),
                tooltip="Show/hide parameter explanations"
            )
            details = widgets.HTML(
                value=f"<pre style='white-space:pre-wrap;margin:0'>{param_details_text}</pre>",
                layout=widgets.Layout(margin="0 0 0 10px", border="1px solid #bbb", padding="6px", display="none", background="#f8f8f8")
            )
            def _on_toggle(change):
                if change.get("name") == "value":
                    show = bool(change.get("new"))
                    details.layout.display = "block" if show else "none"
                    try:
                        toggle.icon = "chevron-up" if show else "chevron-down"
                    except Exception:
                        pass
            toggle.observe(_on_toggle, names="value")
            return widgets.HBox([toggle, details], layout=widgets.Layout(margin="4px 0 4px 0"))

        # Build per-peak controls as before, but without per-row info buttons
        # Reset tracking lists for headers/original centers and capture current active ranges
        peak_label_widgets.clear()
        original_peak_centers.clear()
        last_active_ranges = list(active_ranges)
        # Vectorized in-range computation for performance (avoids per-peak Python loops)
        try:
            import numpy as _np
            cx_array = _np.asarray(peaks_x_all, dtype=float)
            in_range_mask = _np.zeros_like(cx_array, dtype=bool)
            for _lo, _hi in active_ranges:
                try:
                    lo_v = float(min(_lo, _hi))
                    hi_v = float(max(_lo, _hi))
                    in_range_mask |= (cx_array >= lo_v) & (cx_array <= hi_v)
                except Exception:
                    pass
        except Exception:
            # Fallback: compute naïvely if numpy not available
            in_range_mask = []
            for _cx in peaks_x_all:
                try:
                    _cxv = float(_cx)
                    in_range_mask.append(any(lo <= _cxv <= hi for lo, hi in active_ranges))
                except Exception:
                    in_range_mask.append(False)

        excluded_peaks_list = []

        # --- Diff removal path -------------------------------------------------
        # If we have a cache of all previously built peak boxes and the total count of peaks
        # did not change, we can perform an in-place diff update instead of a full rebuild.
        diff_applicable = False
        if isinstance(peak_box_cache, dict) and len(peak_box_cache) == len(peaks_x_all):
            diff_applicable = True

        # Compute new in-range set early for diff logic
        new_in_range_set = {i for i, flag in enumerate(in_range_mask) if bool(flag)}

        if diff_applicable and previous_in_range_indices:
            removed = previous_in_range_indices - new_in_range_set
            added = new_in_range_set - previous_in_range_indices
            unchanged = previous_in_range_indices & new_in_range_set
            # If no changes, simply update summary and return fast
            if not removed and not added:
                try:
                    # Update excluded list text (unchanged except counts)
                    excluded_peaks_list = [
                        (i+1, float(peaks_x_all[i])) for i in range(len(peaks_x_all)) if i not in new_in_range_set
                    ]
                    if excluded_peaks_list:
                        excluded_lines = [f"Peak {num} @ {val:.1f} cm⁻¹" for num, val in excluded_peaks_list]
                        excluded_text = "<br>".join(excluded_lines)
                        excluded_peaks_html.value = (
                            f"<b>Excluded Peaks:</b><div style='color:#777; margin-top:4px; line-height:1.35;'>{excluded_text}</div>"
                        )
                    else:
                        excluded_peaks_html.value = "<b>Excluded Peaks:</b> <span style='color:#777;'>None</span>"
                    summary_html = widgets.HTML(
                        f"<span style='color:#555;'>In-range peaks: {len(new_in_range_set)} / Total: {len(peaks_x_all)}</span>"
                    )
                    # Preserve existing parameter details row (first child) and accordion
                    try:
                        existing_children = list(peak_controls_box.children)
                        if existing_children and isinstance(existing_children[0], widgets.HBox):
                            param_row = existing_children[0]
                        else:
                            param_row = _make_param_details_row()
                        # Reconstruct children minimally (param row, included header, accordion, excluded html)
                        peak_controls_box.children = [
                            param_row,
                            widgets.HTML("<b>Included Peaks:</b>"),
                            peak_accordion,
                            excluded_peaks_html,
                        ]
                        # Swap summary HTML into param row if it contains a summary placeholder
                        # (We assume param_row contains toggle + details; we append summary alongside)
                        try:
                            if len(param_row.children) == 2:
                                param_row.children = (param_row.children[0], param_row.children[1], summary_html)
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
                previous_in_range_indices = new_in_range_set
                # Rebuild tracking lists from cache for unchanged peaks
                alpha_sliders = []
                include_checkboxes = []
                center_sliders = []
                center_slider_peak_indices = []
                sigma_sliders = []
                amplitude_sliders = []
                center_window_sliders = []
                amplitude_mode_toggles = []
                center_mode_toggles = []
                sigma_mode_toggles = []
                for i in sorted(new_in_range_set):
                    try:
                        w = peak_box_cache[i]
                        include_checkboxes.append(w['include'])
                        alpha_sliders.append(w['alpha'])
                        center_sliders.append(w['center'])
                        center_slider_peak_indices.append(i)
                        sigma_sliders.append(w['sigma'])
                        amplitude_sliders.append(w['amplitude'])
                        center_window_sliders.append(w['center_window'])
                        amplitude_mode_toggles.append(w['amp_mode'])
                        center_mode_toggles.append(w['center_mode'])
                        sigma_mode_toggles.append(w['sigma_mode'])
                        # Attach canonical index to include checkbox for later mapping
                        try:
                            setattr(include_checkboxes[-1], '_original_idx', i)
                        except Exception:
                            pass
                    except Exception:
                        pass
                try:
                    for _idx_in_range in sorted(new_in_range_set):
                        _update_peak_toggle_label(_idx_in_range, spectrum_idx=row_idx)
                except Exception:
                    pass
                return  # No UI rebuild required

            # Process removals: hide or remove boxes
            try:
                current_children = list(getattr(peak_accordion, 'children', ()))
            except Exception:
                current_children = []
            # Map box -> peak index for quick removal
            inv_map = {}
            for idx in previous_in_range_indices:
                try:
                    inv_map[peak_box_cache[idx]['box']] = idx
                except Exception:
                    pass
            # Remove boxes for peaks leaving range
            if removed:
                new_children = []
                for box in current_children:
                    idx = inv_map.get(box, None)
                    if idx is not None and idx in removed:
                        # Move to excluded (we just drop from included; excluded list is text-only)
                        try:
                            box.layout.display = 'none'
                        except Exception:
                            pass
                        continue
                    new_children.append(box)
                try:
                    peak_accordion.children = tuple(new_children)
                except Exception:
                    pass
            # Helper to rebuild tracking lists (placeholders and materialized widgets alike)
            def _refresh_slider_lists():
                nonlocal alpha_sliders, include_checkboxes, center_sliders, sigma_sliders, amplitude_sliders, center_window_sliders
                nonlocal amplitude_mode_toggles, center_mode_toggles, sigma_mode_toggles
                nonlocal center_slider_peak_indices
                alpha_sliders = []
                include_checkboxes = []
                center_sliders = []
                center_slider_peak_indices = []
                sigma_sliders = []
                amplitude_sliders = []
                center_window_sliders = []
                amplitude_mode_toggles = []
                center_mode_toggles = []
                sigma_mode_toggles = []
                for _i in sorted(new_in_range_set):
                    try:
                        w = peak_box_cache[_i]
                        include_checkboxes.append(w['include'])
                        alpha_sliders.append(w['alpha'])
                        center_sliders.append(w['center'])
                        center_slider_peak_indices.append(_i)
                        sigma_sliders.append(w['sigma'])
                        amplitude_sliders.append(w['amplitude'])
                        center_window_sliders.append(w['center_window'])
                        amplitude_mode_toggles.append(w['amp_mode'])
                        center_mode_toggles.append(w['center_mode'])
                        sigma_mode_toggles.append(w['sigma_mode'])
                        try:
                            setattr(include_checkboxes[-1], '_original_idx', _i)
                        except Exception:
                            pass
                    except Exception:
                        pass
            # Expose helper so module-level utilities (e.g., accordion scans) can refresh lists.
            globals()['_refresh_slider_lists'] = _refresh_slider_lists

            # Materialization function builds full controls for a single peak on demand
            def _materialize_peak(peak_idx: int, force: bool = False):
                # Build full control set for a peak. When lazy mode has been disabled
                # (LAZY_PEAK_WIDGETS False) we allow forced materialization for eager
                # fallback by passing force=True.
                global LAZY_PEAK_WIDGETS
                if (not LAZY_PEAK_WIDGETS) and (not force):
                    return
                try:
                    cache_entry = peak_box_cache.get(peak_idx)
                    if not cache_entry or cache_entry.get('materialized'):
                        return  # Already built or missing
                    # Heavy build replicating original per-peak construction logic
                    cx = peaks_x_all[peak_idx]; cy = peaks_y_all[peak_idx]
                    original_center_val_local = float(cx)
                    default_bound = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
                    min_cwin = 0.5
                    # Recover saved values if present
                    w_saved = default_bound
                    if saved_center_window is not None and peak_idx < len(saved_center_window):
                        try:
                            candidate = float(saved_center_window[peak_idx])
                            if np.isfinite(candidate) and candidate > 0.0:
                                w_saved = candidate
                        except Exception:
                            pass
                    center_min, center_max = _overall_fit_span()
                    center_val_local = float(cx)
                    if saved_center is not None and peak_idx < len(saved_center):
                        try:
                            center_val_local = float(saved_center[peak_idx])
                        except Exception:
                            pass
                    center_slider = widgets.FloatSlider(
                        value=center_val_local, min=center_min, max=center_max, step=0.1,
                        description="μ (cm⁻¹)", continuous_update=False, style={"description_width": "auto"},
                        readout_format=".1f", layout=widgets.Layout(width="220px")
                    )
                    _set_peak_user_center(row_idx, peak_idx, center_val_local)
                    peak_state_local = peak_center_state_by_idx.get(row_idx, {}).get(peak_idx, {})
                    initial_center_val = peak_state_local.get('initial', center_val_local)
                    center_window_slider = widgets.FloatSlider(
                        value=w_saved, min=min_cwin, max=max(default_bound*2.0, 1.0), step=0.5,
                        description="Window ± (cm⁻¹)", continuous_update=False, style={"description_width": "auto"},
                        readout_format=".1f", layout=widgets.Layout(width="220px")
                    )
                    sigma_val = float(PER_PEAK_DEFAULT_SIGMA)
                    if saved_sigma is not None and peak_idx < len(saved_sigma):
                        try:
                            tmp = float(saved_sigma[peak_idx])
                            if np.isfinite(tmp):
                                sigma_val = tmp
                        except Exception:
                            pass
                    sigma_slider = widgets.FloatSlider(
                        value=sigma_val, min=1.0, max=100.0, step=0.5, description="σ (cm⁻¹)",
                        continuous_update=False, style={"description_width": "auto"}, readout_format=".1f",
                        layout=widgets.Layout(width="220px")
                    )
                    amp_default = abs(float(cy)) * max(1.0, float(PER_PEAK_DEFAULT_SIGMA))
                    amp_val_local = amp_default
                    if saved_amplitude is not None and peak_idx < len(saved_amplitude):
                        try:
                            amp_candidate = float(saved_amplitude[peak_idx])
                            if np.isfinite(amp_candidate) and amp_candidate >= 0.0:
                                amp_val_local = amp_candidate
                        except Exception:
                            pass
                    amplitude_slider = widgets.FloatSlider(
                        value=amp_val_local, min=0.0, max=max(amp_default*5.0, amp_val_local, 1.0),
                        step=max(amp_default*0.01, 0.01), description="A", continuous_update=False,
                        style={"description_width": "auto"}, readout_format=".3f", layout=widgets.Layout(width="220px")
                    )
                    alpha_val_local = DEFAULT_ALPHA
                    if saved_alphas is not None and peak_idx < len(saved_alphas):
                        try:
                            a_candidate = float(saved_alphas[peak_idx])
                            if 0.0 <= a_candidate <= 1.0:
                                alpha_val_local = a_candidate
                        except Exception:
                            pass
                    alpha_slider = widgets.FloatSlider(
                        value=alpha_val_local, min=0.0, max=1.0, step=0.01, description="α",
                        continuous_update=False, readout_format=".2f", style={"description_width": "auto"},
                        layout=widgets.Layout(width="220px")
                    )
                    include_val_local = True
                    if saved_includes is not None and peak_idx < len(saved_includes):
                        try:
                            include_val_local = bool(saved_includes[peak_idx])
                        except Exception:
                            pass
                    include_checkbox = widgets.Checkbox(value=include_val_local, description="Include peak in Fit", indent=False,
                                                       layout=widgets.Layout(width="200px"))
                    try:
                        setattr(include_checkbox, '_original_idx', peak_idx)
                    except Exception:
                        pass
                    # Restore saved modes if available
                    amp_mode_val_local = 'Auto'
                    center_mode_val_local = CENTER_MODE_WINDOW
                    sigma_mode_val_local = 'Auto'
                    if saved_modes and isinstance(saved_modes.get('amplitude'), list) and peak_idx < len(saved_modes.get('amplitude')):
                        amp_mode_val_local = saved_modes['amplitude'][peak_idx]
                    if saved_modes and isinstance(saved_modes.get('center'), list) and peak_idx < len(saved_modes.get('center')):
                        center_mode_val_local = _normalize_center_mode_value(saved_modes['center'][peak_idx])
                    if saved_modes and isinstance(saved_modes.get('sigma'), list) and peak_idx < len(saved_modes.get('sigma')):
                        sigma_mode_val_local = saved_modes['sigma'][peak_idx]
                    amp_mode_toggle = widgets.Dropdown(options=[('Auto','Auto'),('Manual','Manual')], value=amp_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')
                    center_mode_val_local = _normalize_center_mode_value(center_mode_val_local)
                    center_mode_toggle = widgets.Dropdown(options=CENTER_MODE_OPTIONS, value=center_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')
                    sigma_mode_toggle = widgets.Dropdown(options=[('Auto','Auto'),('Manual','Manual')], value=sigma_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')

                    # Attach observers (subset copied from original logic)
                    include_checkbox.observe(_on_include_toggle, names="value")
                    alpha_slider.observe(_on_alpha_change, names="value")
                    _bind_center_slider(center_slider, peak_idx, row_idx)
                    amplitude_slider.observe(_on_center_sigma_change, names="value")
                    center_window_slider.observe(_on_center_sigma_change, names="value")
                    def _on_sigma_change(change, idx_local=peak_idx):
                        # No persistence across spectra; simply snapshot (no-op) to maintain flow
                        if sigma_mode_toggle.value == 'Manual':
                            _snapshot_current_controls()
                    sigma_slider.observe(_on_sigma_change, names="value")

                    # Build layout inside details box
                    details_box = cache_entry['details']
                    try:
                        # Separate labels to keep all controls horizontal; remove slider descriptions (except alpha) to prevent vertical stacking.
                        amplitude_slider.description = ''
                        center_slider.description = ''
                        center_window_slider.description = ''  # we will render a side label instead
                        sigma_slider.description = ''
                        alpha_slider.description = ''  # remove inline description per request
                        # Short widths for horizontal fit
                        amplitude_slider.layout.width = '160px'
                        center_slider.layout.width = '160px'
                        center_window_slider.layout.width = '160px'
                        sigma_slider.layout.width = '160px'
                        alpha_slider.layout.width = '160px'
                        # Capture original values for reset functionality
                        amplitude_slider._original_value = amplitude_slider.value  # type: ignore[attr-defined]
                        center_slider._original_value = center_slider.value  # type: ignore[attr-defined]
                        center_window_slider._original_value = center_window_slider.value  # type: ignore[attr-defined]
                        sigma_slider._original_value = sigma_slider.value  # type: ignore[attr-defined]
                        alpha_slider._original_value = alpha_slider.value  # type: ignore[attr-defined]
                        # Build reset buttons
                        def _mk_reset(target, default_cb=None):
                            btn = widgets.Button(description='Reset', layout=widgets.Layout(width='58px'))
                            def _on_click(_):
                                try:
                                    if callable(default_cb):
                                        candidate = default_cb()
                                        if candidate is not None:
                                            target.value = candidate
                                            return
                                    orig = getattr(target, '_original_value', None)
                                    if orig is not None:
                                        target.value = orig
                                except Exception:
                                    pass
                            btn.on_click(_on_click)
                            return btn
                        amplitude_reset = _mk_reset(amplitude_slider)
                        def _center_reset_value():
                            try:
                                mode_val = center_mode_toggle.value
                            except Exception:
                                mode_val = CENTER_MODE_WINDOW
                            try:
                                base_val = float(initial_center_val)
                            except Exception:
                                base_val = initial_center_val
                            if mode_val == CENTER_MODE_EXACT:
                                return base_val
                            orig_local = getattr(center_slider, '_original_value', None)
                            return orig_local if orig_local is not None else base_val
                        center_reset = _mk_reset(center_slider, default_cb=_center_reset_value)
                        center_window_reset = _mk_reset(center_window_slider)
                        sigma_reset = _mk_reset(sigma_slider)
                        alpha_reset = _mk_reset(alpha_slider)
                        # Common row layout: horizontal flex, visible overflow, generous height to avoid scroll
                        _row_layout = widgets.Layout(
                            display='flex', flex_flow='row nowrap', align_items='center',
                            gap='10px', overflow='visible', min_height='42px'
                        )
                        include_row = widgets.HBox([include_checkbox], layout=_row_layout)
                        # Restore row labels for legibility
                        # Window label separate to avoid vertical stacking; mu label separated for mode visibility logic
                        mu_label = widgets.HTML('<b>μ</b>')
                        # Make window label non-bold for reduced visual weight
                        window_label = widgets.HTML('Window ± (cm⁻¹)')
                        mu_row = widgets.HBox([
                            mu_label, center_mode_toggle, center_slider, center_reset, window_label, center_window_slider, center_window_reset
                        ], layout=_row_layout)
                        amplitude_label = widgets.HTML('<b>A</b>')
                        amplitude_row = widgets.HBox([
                            amplitude_label, amp_mode_toggle, amplitude_slider, amplitude_reset
                        ], layout=_row_layout)
                        sigma_label = widgets.HTML('<b>σ</b>')
                        sigma_row = widgets.HBox([
                            sigma_label, sigma_mode_toggle, sigma_slider, sigma_reset
                        ], layout=_row_layout)
                        alpha_row = widgets.HBox([
                            widgets.HTML('<b>α</b>'), alpha_slider, alpha_reset
                        ], layout=_row_layout)
                        # Ensure container overflow does not force internal scrolling
                        try:
                            details_box.layout.overflow = 'visible'
                        except Exception:
                            pass

                        def _sync_mode_visibility(*_):
                            # Amplitude row: show label always; hide slider/reset in Auto
                            try:
                                if amp_mode_toggle.value == 'Auto':
                                    amplitude_slider.layout.display = 'none'
                                    amplitude_reset.layout.display = 'none'
                                else:
                                    amplitude_slider.layout.display = 'block'
                                    amplitude_reset.layout.display = 'block'
                                amplitude_label.layout.display = 'block'
                            except Exception:
                                pass
                            # Center/window row: label always; toggle sliders per mode
                            try:
                                center_slider.layout.display = 'block'
                                center_reset.layout.display = 'block'
                                if center_mode_toggle.value == CENTER_MODE_WINDOW:
                                    center_window_slider.layout.display = 'block'
                                    center_window_reset.layout.display = 'block'
                                    window_label.layout.display = 'block'
                                else:
                                    center_window_slider.layout.display = 'none'
                                    center_window_reset.layout.display = 'none'
                                    window_label.layout.display = 'none'
                                mu_label.layout.display = 'block'
                            except Exception:
                                pass
                            # Sigma row: show label always; hide slider/reset in Auto
                            try:
                                if sigma_mode_toggle.value == 'Auto':
                                    sigma_slider.layout.display = 'none'
                                    sigma_reset.layout.display = 'none'
                                else:
                                    sigma_slider.layout.display = 'block'
                                    sigma_reset.layout.display = 'block'
                                sigma_label.layout.display = 'block'
                            except Exception:
                                pass
                        try:
                            amp_mode_toggle.observe(_sync_mode_visibility, names='value')
                            center_mode_toggle.observe(_sync_mode_visibility, names='value')
                            sigma_mode_toggle.observe(_sync_mode_visibility, names='value')
                        except Exception:
                            pass
                        _sync_mode_visibility()
                        details_box.children = [include_row, alpha_row, mu_row, amplitude_row, sigma_row]
                    except Exception:
                        details_box.children = [alpha_slider, center_slider, center_window_slider, amplitude_slider, sigma_slider]

                    # Update cache entry with real widgets
                    cache_entry.update({
                        'include': include_checkbox,
                        'alpha': alpha_slider,
                        'center': center_slider,
                        'sigma': sigma_slider,
                        'amplitude': amplitude_slider,
                        'center_window': center_window_slider,
                        'amp_mode': amp_mode_toggle,
                        'center_mode': center_mode_toggle,
                        'sigma_mode': sigma_mode_toggle,
                        'materialized': True,
                    })
                    _refresh_slider_lists()
                except Exception as e:
                    # Capture detailed traceback for diagnostics when debug enabled
                    try:
                        import traceback
                        _DECONV_DEBUG_ERRORS.append(traceback.format_exc(limit=12))
                    except Exception:
                        pass
                    _lazy_debug(f"Materialization failed for peak {peak_idx}: {e.__class__.__name__}: {e}")
                    try:
                        status_html.value = "<span style='color:#c33;'>Lazy peak build failed; reverting to eager mode. Set DECONV_DEBUG=True for details.</span>"
                    except Exception:
                        pass
                    LAZY_PEAK_WIDGETS = False
                    return

            # Add boxes for peaks newly entering range
            if added:
                for idx in sorted(added):
                    try:
                        # Reuse existing cached widgets if previously built; make visible
                        if idx in peak_box_cache:
                            entry = peak_box_cache.get(idx, {})
                            box = entry.get('box')
                            if box is not None:
                                try:
                                    setattr(box, '_peak_idx', idx)
                                except Exception:
                                    pass
                                try:
                                    current_children = list(getattr(peak_accordion, 'children', ()))
                                except Exception:
                                    current_children = []
                                if box not in current_children:
                                    insert_pos = 0
                                    for child in current_children:
                                        try:
                                            child_idx = getattr(child, '_peak_idx', None)
                                        except Exception:
                                            child_idx = None
                                        if child_idx is None and hasattr(child, 'children') and child.children:
                                            try:
                                                desc = getattr(child.children[0], 'description', '')
                                                if desc.startswith('Peak '):
                                                    num_part = desc.split(' ')[1]
                                                    child_idx = int(num_part) - 1
                                            except Exception:
                                                child_idx = None
                                        if child_idx is not None and child_idx < idx:
                                            insert_pos += 1
                                    try:
                                        current_children.insert(insert_pos, box)
                                        peak_accordion.children = tuple(current_children)
                                    except Exception:
                                        pass
                                try:
                                    box.layout.display = ''
                                except Exception:
                                    pass
                            _update_peak_toggle_label(idx, spectrum_idx=row_idx)
                            continue
                        # Minimal lazy build: only toggle + empty details container; placeholders for controls
                        cx = peaks_x_all[idx]
                        original_center_val_local = float(cx)
                        details = widgets.VBox([]); details.layout.display = 'none'
                        toggle = widgets.ToggleButton(value=False, description=f"Peak {idx+1} @ {original_center_val_local:.1f} → {original_center_val_local:.1f} cm⁻¹", icon='chevron-down', layout=widgets.Layout(width='auto'))
                        def _on_toggle_local(change, _t=toggle, _d=details, _idx=idx):
                            if change.get('name') == 'value':
                                show = bool(change.get('new'))
                                _d.layout.display = 'block' if show else 'none'
                                try:
                                    _t.icon = 'chevron-up' if show else 'chevron-down'
                                except Exception:
                                    pass
                                # Lazy materialization trigger
                                if show:
                                    try:
                                        # Try lazy first (may raise). If lazy disabled, force eager build.
                                        if LAZY_PEAK_WIDGETS:
                                            _materialize_peak(_idx)
                                        else:
                                            _materialize_peak(_idx, force=True)
                                    except Exception:
                                        # Capture traceback for diagnostics
                                        try:
                                            import traceback
                                            global _DECONV_DEBUG_ERRORS
                                            _DECONV_DEBUG_ERRORS.append(traceback.format_exc(limit=12))
                                        except Exception:
                                            pass
                                        _lazy_debug(f"Toggle local failed for peak {_idx}")
                                        try:
                                            status_html.value = "<span style='color:#c33;'>Lazy materialization failed; switching to eager mode.</span>"
                                        except Exception:
                                            pass
                                        try:
                                            LAZY_PEAK_WIDGETS = False
                                        except Exception:
                                            pass
                                        # Attempt eager materialization once after failure
                                        try:
                                            _materialize_peak(_idx, force=True)
                                        except Exception:
                                            pass
                        toggle.observe(_on_toggle_local, names='value')
                        box = widgets.VBox([toggle, details], layout=widgets.Layout(border="2.5px solid #222", margin="12px 0", padding="16px 22px", border_radius="12px", background="#2d3748"))
                        try:
                            setattr(box, '_peak_idx', idx)
                        except Exception:
                            pass
                        # Insert at correct sorted position
                        try:
                            current_children = list(getattr(peak_accordion, 'children', ()))
                            insert_pos = 0
                            # Determine insertion point by counting existing children with peak index < idx
                            for c in current_children:
                                try:
                                    desc = getattr(c.children[0], 'description', '')
                                    if desc.startswith('Peak '):
                                        # Extract original peak number
                                        num_part = desc.split(' ')[1]
                                        peak_num = int(num_part)
                                        # peak_num is 1-based
                                        if peak_num-1 < idx:
                                            insert_pos += 1
                                except Exception:
                                    pass
                            current_children.insert(insert_pos, box)
                            peak_accordion.children = tuple(current_children)
                        except Exception:
                            pass
                        # Cache placeholders only if not already materialized (avoid overwriting real widgets)
                        if not (idx in peak_box_cache and peak_box_cache[idx].get('materialized')):
                            peak_box_cache[idx] = {
                                'box': box,
                                'toggle': toggle,
                                'details': details,
                                'include': _LazyPlaceholder(True),
                                'alpha': _LazyPlaceholder(DEFAULT_ALPHA),
                                'center': _LazyPlaceholder(float(cx)),
                                'sigma': _LazyPlaceholder(float(PER_PEAK_DEFAULT_SIGMA)),
                                'amplitude': _LazyPlaceholder(abs(float(peaks_y_all[idx])) * max(1.0, float(PER_PEAK_DEFAULT_SIGMA))),
                                'center_window': _LazyPlaceholder(float(PER_PEAK_DEFAULT_CENTER_WINDOW)),
                                'amp_mode': _LazyModePlaceholder('Auto'),
                                'center_mode': _LazyModePlaceholder(CENTER_MODE_WINDOW),
                                'sigma_mode': _LazyModePlaceholder('Auto'),
                                'materialized': False,
                            }
                        _update_peak_toggle_label(idx, spectrum_idx=row_idx)
                    except Exception:
                        pass
                # After lazy additions, refresh tracking lists (placeholders included)
                _refresh_slider_lists()
            # Update excluded list & summary after diff ops
            try:
                excluded_peaks_list = [
                    (i+1, float(peaks_x_all[i])) for i in range(len(peaks_x_all)) if i not in new_in_range_set
                ]
                if excluded_peaks_list:
                    excluded_lines = [f"Peak {num} @ {val:.1f} cm⁻¹" for num, val in excluded_peaks_list]
                    excluded_text = "<br>".join(excluded_lines)
                    excluded_peaks_html.value = (
                        f"<b>Excluded Peaks:</b><div style='color:#777; margin-top:4px; line-height:1.35;'>{excluded_text}</div>"
                    )
                else:
                    excluded_peaks_html.value = "<b>Excluded Peaks:</b> <span style='color:#777;'>None</span>"
                summary_html = widgets.HTML(
                    f"<span style='color:#555;'>In-range peaks: {len(new_in_range_set)} / Total: {len(peaks_x_all)}</span>"
                )
                # Reconstruct control box children minimally
                try:
                    existing_children = list(peak_controls_box.children)
                    if existing_children and isinstance(existing_children[0], widgets.HBox):
                        param_row = existing_children[0]
                    else:
                        param_row = _make_param_details_row()
                    peak_controls_box.children = [
                        param_row,
                        widgets.HTML("<b>Included Peaks:</b>"),
                        peak_accordion,
                        excluded_peaks_html,
                    ]
                    try:
                        if len(param_row.children) == 2:
                            param_row.children = (param_row.children[0], param_row.children[1], summary_html)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
            # Rebuild tracking lists from cache for new set
            alpha_sliders = []
            include_checkboxes = []
            center_sliders = []
            center_slider_peak_indices = []
            sigma_sliders = []
            amplitude_sliders = []
            center_window_sliders = []
            amplitude_mode_toggles = []
            center_mode_toggles = []
            sigma_mode_toggles = []
            for i in sorted(new_in_range_set):
                try:
                    w = peak_box_cache[i]
                    include_checkboxes.append(w['include'])
                    alpha_sliders.append(w['alpha'])
                    center_sliders.append(w['center'])
                    center_slider_peak_indices.append(i)
                    sigma_sliders.append(w['sigma'])
                    amplitude_sliders.append(w['amplitude'])
                    center_window_sliders.append(w['center_window'])
                    amplitude_mode_toggles.append(w['amp_mode'])
                    center_mode_toggles.append(w['center_mode'])
                    sigma_mode_toggles.append(w['sigma_mode'])
                    try:
                        setattr(include_checkboxes[-1], '_original_idx', i)
                    except Exception:
                        pass
                except Exception:
                    pass
            previous_in_range_indices = new_in_range_set
            return  # Diff path handled fully; skip full rebuild below

        # --- End diff removal path; fall back to full rebuild ------------------
        previous_in_range_indices = new_in_range_set  # initialize tracking for next diff cycle
        # Placeholder-only initial build (no heavy sliders) when diff path not available
        included_boxes = []
        excluded_peaks_list = []
        for i, cx in enumerate(peaks_x_all):
            in_range = bool(in_range_mask[i]) if i < len(in_range_mask) else False
            original_center_val = float(cx)
            if not in_range:
                excluded_peaks_list.append((i+1, original_center_val))
                continue
            # Ensure materialization function exists even when entering fallback path
            if '_materialize_peak' not in locals():
                def _materialize_peak(peak_idx: int, force: bool = False):
                    global LAZY_PEAK_WIDGETS
                    if (not LAZY_PEAK_WIDGETS) and (not force):
                        return
                    try:
                        cache_entry = peak_box_cache.get(peak_idx)
                        if not cache_entry or cache_entry.get('materialized'):
                            return
                        cx_local = peaks_x_all[peak_idx]; cy_local = peaks_y_all[peak_idx]
                        original_center_val_local = float(cx_local)
                        default_bound_local = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
                        min_cwin_local = 0.5
                        w_saved_local = default_bound_local
                        try:
                            if saved_center_window is not None and peak_idx < len(saved_center_window):
                                cand = float(saved_center_window[peak_idx])
                                if np.isfinite(cand) and cand > 0.0:
                                    w_saved_local = cand
                        except Exception:
                            pass
                        center_min_local, center_max_local = _overall_fit_span() if ' _overall_fit_span' in globals() or '_overall_fit_span' in locals() else (float(np.nanmin(peaks_x_all)), float(np.nanmax(peaks_x_all)))
                        center_val_local = float(cx_local)
                        try:
                            if saved_center is not None and peak_idx < len(saved_center):
                                center_val_local = float(saved_center[peak_idx])
                        except Exception:
                            pass
                        center_slider = widgets.FloatSlider(value=center_val_local, min=center_min_local, max=center_max_local, step=0.1, description="μ (cm⁻¹)", continuous_update=False, style={"description_width": "auto"}, readout_format=".1f", layout=widgets.Layout(width="220px"))
                        _set_peak_user_center(row_idx, peak_idx, center_val_local)
                        peak_state_local = peak_center_state_by_idx.get(row_idx, {}).get(peak_idx, {})
                        initial_center_val = peak_state_local.get('initial', center_val_local)
                        center_window_slider = widgets.FloatSlider(value=w_saved_local, min=min_cwin_local, max=max(default_bound_local*2.0, 1.0), step=0.5, description="Window ± (cm⁻¹)", continuous_update=False, style={"description_width": "auto"}, readout_format=".1f", layout=widgets.Layout(width="220px"))
                        sigma_val_local = float(PER_PEAK_DEFAULT_SIGMA)
                        try:
                            if saved_sigma is not None and peak_idx < len(saved_sigma):
                                tmp = float(saved_sigma[peak_idx])
                                if np.isfinite(tmp):
                                    sigma_val_local = tmp
                        except Exception:
                            pass
                        sigma_slider = widgets.FloatSlider(value=sigma_val_local, min=1.0, max=100.0, step=0.5, description="σ (cm⁻¹)", continuous_update=False, style={"description_width": "auto"}, readout_format=".1f", layout=widgets.Layout(width="220px"))
                        amp_default_local = abs(float(cy_local)) * max(1.0, float(PER_PEAK_DEFAULT_SIGMA))
                        amp_val_local = amp_default_local
                        try:
                            if saved_amplitude is not None and peak_idx < len(saved_amplitude):
                                cand_a = float(saved_amplitude[peak_idx])
                                if np.isfinite(cand_a) and cand_a >= 0.0:
                                    amp_val_local = cand_a
                        except Exception:
                            pass
                        amplitude_slider = widgets.FloatSlider(value=amp_val_local, min=0.0, max=max(amp_default_local*5.0, amp_val_local, 1.0), step=max(amp_default_local*0.01, 0.01), description="A", continuous_update=False, style={"description_width": "auto"}, readout_format=".3f", layout=widgets.Layout(width="220px"))
                        alpha_val_local = DEFAULT_ALPHA
                        try:
                            if saved_alphas is not None and peak_idx < len(saved_alphas):
                                cand_alpha = float(saved_alphas[peak_idx])
                                if 0.0 <= cand_alpha <= 1.0:
                                    alpha_val_local = cand_alpha
                        except Exception:
                            pass
                        alpha_slider = widgets.FloatSlider(value=alpha_val_local, min=0.0, max=1.0, step=0.01, description="α", continuous_update=False, readout_format=".2f", style={"description_width": "auto"}, layout=widgets.Layout(width="220px"))
                        include_val_local = True
                        try:
                            if saved_includes is not None and peak_idx < len(saved_includes):
                                include_val_local = bool(saved_includes[peak_idx])
                        except Exception:
                            pass
                        include_checkbox = widgets.Checkbox(value=include_val_local, description="Include peak in Fit", indent=False, layout=widgets.Layout(width="200px"))
                        try:
                            setattr(include_checkbox, '_original_idx', peak_idx)
                        except Exception:
                            pass
                        amp_mode_val_local = 'Auto'; center_mode_val_local = CENTER_MODE_WINDOW; sigma_mode_val_local = 'Auto'
                        try:
                            if saved_modes and isinstance(saved_modes.get('amplitude'), list) and peak_idx < len(saved_modes.get('amplitude')):
                                amp_mode_val_local = saved_modes['amplitude'][peak_idx]
                            if saved_modes and isinstance(saved_modes.get('center'), list) and peak_idx < len(saved_modes.get('center')):
                                center_mode_val_local = _normalize_center_mode_value(saved_modes['center'][peak_idx])
                            if saved_modes and isinstance(saved_modes.get('sigma'), list) and peak_idx < len(saved_modes.get('sigma')):
                                sigma_mode_val_local = saved_modes['sigma'][peak_idx]
                        except Exception:
                            pass
                        amp_mode_toggle = widgets.Dropdown(options=[('Auto','Auto'),('Manual','Manual')], value=amp_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')
                        center_mode_val_local = _normalize_center_mode_value(center_mode_val_local)
                        center_mode_toggle = widgets.Dropdown(options=CENTER_MODE_OPTIONS, value=center_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')
                        sigma_mode_toggle = widgets.Dropdown(options=[('Auto','Auto'),('Manual','Manual')], value=sigma_mode_val_local, layout=widgets.Layout(width='110px'), style={'description_width':'initial'}, description='')
                        try:
                            include_checkbox.observe(_on_include_toggle, names='value')
                            alpha_slider.observe(_on_alpha_change, names='value')
                            _bind_center_slider(center_slider, peak_idx, row_idx)
                            amplitude_slider.observe(_on_center_sigma_change, names='value')
                            center_window_slider.observe(_on_center_sigma_change, names='value')
                            def _on_sigma_change_stub(change, idx_local=peak_idx):
                                if sigma_mode_toggle.value == 'Manual':
                                    # No persistence; maintain callback structure
                                    _snapshot_current_controls()
                            sigma_slider.observe(_on_sigma_change_stub, names='value')
                        except Exception:
                            pass
                        details_box_local = cache_entry['details']
                        try:
                            amplitude_slider.description = ''
                            center_slider.description = ''
                            center_window_slider.description = ''  # side label used instead
                            sigma_slider.description = ''
                            alpha_slider.description = ''
                            amplitude_slider.layout.width = '160px'
                            center_slider.layout.width = '160px'
                            center_window_slider.layout.width = '160px'
                            sigma_slider.layout.width = '160px'
                            alpha_slider.layout.width = '160px'
                            amplitude_slider._original_value = amplitude_slider.value  # type: ignore[attr-defined]
                            center_slider._original_value = center_slider.value  # type: ignore[attr-defined]
                            center_window_slider._original_value = center_window_slider.value  # type: ignore[attr-defined]
                            sigma_slider._original_value = sigma_slider.value  # type: ignore[attr-defined]
                            alpha_slider._original_value = alpha_slider.value  # type: ignore[attr-defined]
                            def _mk_reset_local(target, default_cb=None):
                                btn = widgets.Button(description='Reset', layout=widgets.Layout(width='58px'))
                                def _on_click(_):
                                    try:
                                        if callable(default_cb):
                                            candidate = default_cb()
                                            if candidate is not None:
                                                target.value = candidate
                                                return
                                        orig = getattr(target, '_original_value', None)
                                        if orig is not None:
                                            target.value = orig
                                    except Exception:
                                        pass
                                btn.on_click(_on_click)
                                return btn
                            amplitude_reset_l = _mk_reset_local(amplitude_slider)
                            def _center_reset_value_local():
                                try:
                                    mode_val = center_mode_toggle.value
                                except Exception:
                                    mode_val = CENTER_MODE_WINDOW
                                try:
                                    base_val = float(initial_center_val)
                                except Exception:
                                    base_val = initial_center_val
                                if mode_val == CENTER_MODE_EXACT:
                                    return base_val
                                orig_local = getattr(center_slider, '_original_value', None)
                                return orig_local if orig_local is not None else base_val
                            center_reset_l = _mk_reset_local(center_slider, default_cb=_center_reset_value_local)
                            center_window_reset_l = _mk_reset_local(center_window_slider)
                            sigma_reset_l = _mk_reset_local(sigma_slider)
                            alpha_reset_l = _mk_reset_local(alpha_slider)
                            _row_layout_local = widgets.Layout(
                                display='flex', flex_flow='row nowrap', align_items='center',
                                gap='10px', overflow='visible', min_height='42px'
                            )
                            include_row_local = widgets.HBox([include_checkbox], layout=_row_layout_local)
                            mu_label_l = widgets.HTML('<b>μ</b>')
                            window_label_l = widgets.HTML('Window ± (cm⁻¹)')
                            mu_row_local = widgets.HBox([mu_label_l, center_mode_toggle, center_slider, center_reset_l, window_label_l, center_window_slider, center_window_reset_l], layout=_row_layout_local)
                            amplitude_label_l = widgets.HTML('<b>A</b>')
                            amplitude_row_local = widgets.HBox([amplitude_label_l, amp_mode_toggle, amplitude_slider, amplitude_reset_l], layout=_row_layout_local)
                            sigma_label_l = widgets.HTML('<b>σ</b>')
                            sigma_row_local = widgets.HBox([sigma_label_l, sigma_mode_toggle, sigma_slider, sigma_reset_l], layout=_row_layout_local)
                            alpha_row_local = widgets.HBox([widgets.HTML('<b>α</b>'), alpha_slider, alpha_reset_l], layout=_row_layout_local)
                            try:
                                details_box_local.layout.overflow = 'visible'
                            except Exception:
                                pass
                            def _sync_mode_visibility_local(*_):
                                # Amplitude row labels always visible
                                try:
                                    if amp_mode_toggle.value == 'Auto':
                                        amplitude_slider.layout.display = 'none'
                                        amplitude_reset_l.layout.display = 'none'
                                    else:
                                        amplitude_slider.layout.display = 'block'
                                        amplitude_reset_l.layout.display = 'block'
                                    amplitude_label_l.layout.display = 'block'
                                except Exception:
                                    pass
                                # Center / window handling
                                try:
                                    center_slider.layout.display = 'block'
                                    center_reset_l.layout.display = 'block'
                                    if center_mode_toggle.value == CENTER_MODE_WINDOW:
                                        center_window_slider.layout.display = 'block'
                                        center_window_reset_l.layout.display = 'block'
                                        window_label_l.layout.display = 'block'
                                    else:
                                        center_window_slider.layout.display = 'none'
                                        center_window_reset_l.layout.display = 'none'
                                        window_label_l.layout.display = 'none'
                                    mu_label_l.layout.display = 'block'
                                except Exception:
                                    pass
                                # Sigma row handling
                                try:
                                    if sigma_mode_toggle.value == 'Auto':
                                        sigma_slider.layout.display = 'none'
                                        sigma_reset_l.layout.display = 'none'
                                    else:
                                        sigma_slider.layout.display = 'block'
                                        sigma_reset_l.layout.display = 'block'
                                    sigma_label_l.layout.display = 'block'
                                except Exception:
                                    pass
                            try:
                                amp_mode_toggle.observe(_sync_mode_visibility_local, names='value')
                                center_mode_toggle.observe(_sync_mode_visibility_local, names='value')
                                sigma_mode_toggle.observe(_sync_mode_visibility_local, names='value')
                            except Exception:
                                pass
                            _sync_mode_visibility_local()
                            details_box_local.children = [include_row_local, alpha_row_local, mu_row_local, amplitude_row_local, sigma_row_local]
                        except Exception:
                            details_box_local.children = [alpha_slider, center_slider, center_window_slider, amplitude_slider, sigma_slider]
                        # Update cache with real widgets; store both 'amp_mode' and 'amplitude_mode' for consistency with downstream re-sync logic
                        cache_entry.update({
                            'include': include_checkbox,
                            'alpha': alpha_slider,
                            'center': center_slider,
                            'sigma': sigma_slider,
                            'amplitude': amplitude_slider,
                            'center_window': center_window_slider,
                            'amp_mode': amp_mode_toggle,
                            'amplitude_mode': amp_mode_toggle,
                            'center_mode': center_mode_toggle,
                            'sigma_mode': sigma_mode_toggle,
                            'materialized': True
                        })
                        try:
                            # Replace placeholder references in master lists at index i so subsequent fits read real widgets directly
                            center_sliders[i] = center_slider
                            sigma_sliders[i] = sigma_slider
                            amplitude_sliders[i] = amplitude_slider
                            alpha_sliders[i] = alpha_slider
                            center_window_sliders[i] = center_window_slider
                            amplitude_mode_toggles[i] = amp_mode_toggle
                            center_mode_toggles[i] = center_mode_toggle
                            sigma_mode_toggles[i] = sigma_mode_toggle
                            _debug_log(f"[PEAK_MATERIALIZED] peak={peak_idx+1} center={center_slider.value} sigma={sigma_slider.value} amp={amplitude_slider.value} win={center_window_slider.value} modes c={center_mode_toggle.value} s={sigma_mode_toggle.value} a={amp_mode_toggle.value}")
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            import traceback
                            _DECONV_DEBUG_ERRORS.append(traceback.format_exc(limit=12))
                        except Exception:
                            pass
                        _lazy_debug(f"Fallback materialization failed for peak {peak_idx}: {e}")
                        try:
                            status_html.value = "<span style='color:#c33;'>Fallback materialization failed; disabling lazy mode.</span>"
                        except Exception:
                            pass
                        LAZY_PEAK_WIDGETS = False
                        return
            # Toggle + empty details; materialization deferred until expand
            details = widgets.VBox([]); details.layout.display = 'none'
            toggle = widgets.ToggleButton(
                value=False,
                description=f"Peak {i+1}",
                icon='chevron-down', layout=widgets.Layout(width='auto'),
                tooltip='Show/hide peak parameter controls'
            )
            def _on_toggle_init(change, _t=toggle, _d=details, _idx=i):
                # Declare global early since we conditionally reassign LAZY_PEAK_WIDGETS
                global LAZY_PEAK_WIDGETS
                if change.get('name') == 'value':
                    show = bool(change.get('new'))
                    _d.layout.display = 'block' if show else 'none'
                    try:
                        _t.icon = 'chevron-up' if show else 'chevron-down'
                    except Exception:
                        pass
                    if show:
                        # Instrument toggle open prior to any materialization attempt
                        try:
                            pre_mat = peak_box_cache.get(_idx, {}).get('materialized', False)
                            cur_center = None
                            try:
                                cent_obj = peak_box_cache.get(_idx, {}).get('center')
                                if cent_obj is not None:
                                    cur_center = getattr(cent_obj, 'value', cent_obj)
                            except Exception:
                                cur_center = None
                            _debug_log(f"[PEAK_TOGGLE_OPEN] peak={_idx+1} pre_materialized={pre_mat} center={cur_center}")
                        except Exception:
                            pass
                        try:
                            if LAZY_PEAK_WIDGETS:
                                _materialize_peak(_idx)
                            else:
                                _materialize_peak(_idx, force=True)
                            # Post-materialization instrumentation
                            try:
                                entry = peak_box_cache.get(_idx, {})
                                post_mat = entry.get('materialized', False)
                                # Detect any remaining placeholder keys
                                placeholder_keys = []
                                try:
                                    from inspect import isclass
                                except Exception:
                                    pass
                                for k in ('center','sigma','amplitude','center_window','alpha','amp_mode','center_mode','sigma_mode'):
                                    v = entry.get(k)
                                    try:
                                        if isinstance(v, (_LazyPlaceholder, _LazyModePlaceholder)):
                                            placeholder_keys.append(k)
                                    except Exception:
                                        pass
                                post_center = None
                                try:
                                    c_obj = entry.get('center')
                                    if c_obj is not None:
                                        post_center = getattr(c_obj, 'value', c_obj)
                                except Exception:
                                    post_center = None
                                _debug_log(f"[PEAK_TOGGLE_AFTER] peak={_idx+1} materialized={post_mat} center={post_center} remaining_placeholders={placeholder_keys}")
                            except Exception:
                                pass
                        except Exception:
                            # Capture traceback for diagnostics
                            try:
                                import traceback
                                global _DECONV_DEBUG_ERRORS
                                _DECONV_DEBUG_ERRORS.append(traceback.format_exc(limit=12))
                            except Exception:
                                pass
                            _lazy_debug(f"Toggle init failed for peak {_idx}")
                            # Switch to eager mode and attempt forced build so user sees controls
                            try:
                                status_html.value = "<span style='color:#c33;'>Lazy materialization failed; forcing eager build.</span>"
                            except Exception:
                                pass
                            try:
                                LAZY_PEAK_WIDGETS = False
                            except Exception:
                                pass
                            try:
                                _materialize_peak(_idx, force=True)
                            except Exception:
                                pass
            toggle.observe(_on_toggle_init, names='value')
            box = widgets.VBox([toggle, details], layout=widgets.Layout(
                border="2.5px solid #222", margin="12px 0", padding="16px 22px", border_radius="12px", background="#2d3748"
            ))
            try:
                setattr(box, '_peak_idx', i)
            except Exception:
                pass
            included_boxes.append(box)
            # Cache placeholder entry only if not already materialized
            if not (i in peak_box_cache and peak_box_cache[i].get('materialized')):
                peak_box_cache[i] = {
                    'box': box,
                    'toggle': toggle,
                    'details': details,
                    'include': _LazyPlaceholder(True),
                    'alpha': _LazyPlaceholder(DEFAULT_ALPHA),
                    'center': _LazyPlaceholder(float(cx)),
                    'sigma': _LazyPlaceholder(float(PER_PEAK_DEFAULT_SIGMA)),
                    'amplitude': _LazyPlaceholder(abs(float(peaks_y_all[i])) * max(1.0, float(PER_PEAK_DEFAULT_SIGMA))),
                    'center_window': _LazyPlaceholder(float(PER_PEAK_DEFAULT_CENTER_WINDOW)),
                    'amp_mode': _LazyModePlaceholder('Auto'),
                    'center_mode': _LazyModePlaceholder(CENTER_MODE_WINDOW),
                    'sigma_mode': _LazyModePlaceholder('Auto'),
                    'materialized': False,
                }
            _update_peak_toggle_label(i, spectrum_idx=row_idx)
        # Populate accordion
        try:
            peak_accordion.children = tuple(included_boxes)
        except Exception:
            pass
        # Update excluded peaks HTML
        if excluded_peaks_list:
            excluded_lines = [f"Peak {num} @ {val:.1f} cm⁻¹" for num, val in excluded_peaks_list]
            excluded_peaks_html.value = (
                "<b>Excluded Peaks:</b><div style='color:#777; margin-top:4px; line-height:1.35;'>" + "<br>".join(excluded_lines) + "</div>"
            )
        else:
            excluded_peaks_html.value = "<b>Excluded Peaks:</b> <span style='color:#777;'>None</span>"
        summary_html = widgets.HTML(
            f"<span style='color:#555;'>In-range peaks: {len(included_boxes)} / Total: {len(peaks_x_all)}</span>"
        )
        param_row = _make_param_details_row()
        peak_controls_box.children = [
            widgets.HBox([param_row, summary_html]),
            widgets.HTML("<b>Included Peaks:</b>"),
            peak_accordion,
            excluded_peaks_html,
        ]
        # Rebuild tracking lists using placeholders
        alpha_sliders = []
        include_checkboxes = []
        center_sliders = []
        center_slider_peak_indices = []
        sigma_sliders = []
        amplitude_sliders = []
        center_window_sliders = []
        amplitude_mode_toggles = []
        center_mode_toggles = []
        sigma_mode_toggles = []
        for i in range(len(peaks_x_all)):
            if i in peak_box_cache and i in new_in_range_set:
                w = peak_box_cache[i]
                include_checkboxes.append(w['include'])
                alpha_sliders.append(w['alpha'])
                center_sliders.append(w['center'])
                center_slider_peak_indices.append(i)
                sigma_sliders.append(w['sigma'])
                amplitude_sliders.append(w['amplitude'])
                center_window_sliders.append(w['center_window'])
                amplitude_mode_toggles.append(w['amp_mode'])
                center_mode_toggles.append(w['center_mode'])
                sigma_mode_toggles.append(w['sigma_mode'])
                try:
                    included_index_map[i] = w['toggle']
                except Exception:
                    pass
                # Attach canonical index to include checkbox for later mapping
                try:
                    setattr(include_checkboxes[-1], '_original_idx', i)
                except Exception:
                    pass
        previous_in_range_indices = new_in_range_set
        return

    # --- Filtering helpers for material/conditions -> spectrum options ---
    def _row_condition_value(row):
        """Return the Conditions/Condition cell value for a DataFrame row."""
        try:
            return row.get("Conditions", row.get("Condition", ""))
        except Exception:
            return ""

    def _filter_spectra_by_material_condition(df):
        try:
            include_bad_flag = bool(getattr(include_bad_cb, "value", True))
        except Exception:
            include_bad_flag = True
        return _filter_spectra_dataframe(
            df,
            material=getattr(material_dd, "value", "any"),
            condition=getattr(conditions_dd, "value", "any"),
            include_bad=include_bad_flag,
            include_unexposed=True,
            normalized_column="Normalized and Corrected Data",
        )

    def _rebuild_spectrum_options(*_):
        """Recompute the spectrum dropdown options based on current filters."""
        nonlocal bulk_update_in_progress
        # Build candidate set from initial 'filtered' and drop rows without normalized
        # data
        try:
            cand = _filter_spectra_by_material_condition(filtered)
        except Exception:
            cand = filtered.copy()
        try:
            if "Time" in cand.columns:
                cand = cand.copy()
                cand["_sort_time"] = pd.to_numeric(cand["Time"], errors="coerce").fillna(float("inf"))
                cand = cand.sort_values(by=["_sort_time"], kind="mergesort")
                try:
                    cand = cand.drop(columns=["_sort_time"], errors="ignore")
                except Exception:
                    pass
        except Exception:
            pass
        # Read current filter selections
        sel_mat = material_dd.value if hasattr(material_dd, "value") else "any"
        sel_cond = conditions_dd.value if hasattr(conditions_dd, "value") else "any"

        # Build new_options: list of (label, idx) for spectra matching current filters
        new_options = []
        for idx, r in cand.iterrows():
            label = (
                f"{r.get('Material','')} | {r.get('Conditions', r.get('Condition',''))}"
                f" | T={r.get('Time','')} | {r.get('File Name','')}"
            )
            new_options.append((label, idx))

        # Save current value before updating options
        prev_value = spectrum_sel.value

        valid_values = [v for (_lbl, v) in new_options]
        value_changed = False
        # Try to select a row matching the saved session 'time' (within current filters)
        preferred_idx = None
        try:
            _sess = _get_session_defaults()
            _sess_time = _sess.get("time", "any")
            if _sess_time is not None and str(_sess_time).strip().lower() != "any":
                # Compare numerically when possible, else fallback to string compare
                for _lbl, _v in new_options:
                    try:
                        _tval = FTIR_DataFrame.loc[_v].get("Time")
                        # Numeric equality if both castable
                        try:
                            if float(_tval) == float(_sess_time):
                                preferred_idx = _v
                                break
                        except Exception:
                            if str(_tval) == str(_sess_time):
                                preferred_idx = _v
                                break
                    except Exception:
                        continue
        except Exception:
            preferred_idx = None
        if preferred_idx in valid_values:
            spectrum_sel.value = preferred_idx
            value_changed = True
        elif prev_value in valid_values:
            # Only assign if the value actually needs to change
            if spectrum_sel.value != prev_value:
                spectrum_sel.value = prev_value
                value_changed = True
        elif valid_values:
            spectrum_sel.value = valid_values[0]
            value_changed = True
        else:
            # No spectra available, clear plot and controls
            with fig.batch_update():
                fig.data[0].x = []
                fig.data[0].y = []
                fig.data[1].x = []
                fig.data[1].y = []
            peak_controls_box.children = [
                widgets.HTML("<b>No spectra available after filtering.</b>")
            ]
            _log_once("No spectra available after filtering selections.")
            return
        # Re-attach observer and trigger a single downstream update
        try:
            spectrum_sel.observe(_on_spectrum_change, names="value")
        except Exception:
            pass
        # If we did not change the value (e.g., filters adjusted but selection stable),
        # explicitly trigger one update to refresh controls/plot exactly once.
        if not value_changed:
            _on_spectrum_change()
        try:
            _update_spectrum_counter()
        except Exception:
            pass
        # If no valid values remain, hide mark row until user selects something later
        try:
            if not valid_values and ('mark_row' in locals() or 'mark_row' in globals()):
                mark_row.layout.display = "none"
        except Exception:
            pass

    def _fit_and_update_plot(*_, ignore_debounce=False, override_ranges=None, update_plot=True, update_controls=True):
        """Run a Pseudo-Voigt fit for the selected spectrum.

        When update_plot is False, the fit still runs and redchi/result are
        recorded, but Plotly traces are not updated.

        When update_controls is False, status text and show/hide control
        animations are suppressed.
        """
        nonlocal fit_thread, cancel_event, fit_cancel_token, iterating_in_progress
        nonlocal fit_update_inflight, last_fit_update_ts
        nonlocal alpha_sliders, include_checkboxes, center_sliders, sigma_sliders
        nonlocal amplitude_sliders, center_window_sliders
        nonlocal center_mode_toggles, sigma_mode_toggles, amplitude_mode_toggles
        nonlocal status_html, fig, center_slider_peak_indices
        nonlocal last_redchi_by_idx, last_result_by_idx

        # Local helper to ensure logging occurs even if DECONV_DEBUG False (to diagnose why nothing changes)
        def _log_dbg(tag: str, msg: str):
            try:
                _debug_log(f"[{tag}] {msg}")
            except Exception:
                pass

        _log_dbg('FIT_START', 'Invoked _fit_and_update_plot')

        # Guard reset helper (missing previously, causing NameError and hang)
        def _finish_fit_guard():
            nonlocal fit_update_inflight, last_fit_update_ts
            fit_update_inflight = False
            try:
                last_fit_update_ts = time.time()
            except Exception:
                last_fit_update_ts = 0.0

        def _sanitize_fit_ranges(ranges_in):
            """Return list of (lo, hi) floats with NaNs removed and ordered."""
            cleaned = []
            if not ranges_in:
                return cleaned
            for pair in ranges_in:
                try:
                    lo_val, hi_val = pair
                except Exception:
                    continue
                try:
                    lo_f = float(lo_val)
                    hi_f = float(hi_val)
                except Exception:
                    continue
                finite_ok = True
                try:
                    finite_ok = bool(np.isfinite(lo_f)) and bool(np.isfinite(hi_f))
                except Exception:
                    pass
                if not finite_ok:
                    continue
                if hi_f < lo_f:
                    lo_f, hi_f = hi_f, lo_f
                cleaned.append((lo_f, hi_f))
            return cleaned

        # Debounce
        try:
            now_ts = time.time()
        except Exception:
            now_ts = 0.0
        if fit_update_inflight:
            _log_dbg('GUARD_SKIP', f"fit_update_inflight True; last_ts={last_fit_update_ts:.4f}")
            return
        delta_since_last = now_ts - last_fit_update_ts
        if not ignore_debounce and (delta_since_last) < 0.03:
            _log_dbg('DEBOUNCE_SKIP', f"delta={delta_since_last:.4f} < 0.03")
            return
        if ignore_debounce:
            _log_dbg('DEBOUNCE_BYPASS', f"ignore_debounce=True; delta={delta_since_last:.4f}")
        fit_update_inflight = True
        _log_dbg('GUARD_SET', f"fit_update_inflight set True at ts={now_ts:.4f}")
        if update_controls:
            # Notify user immediately before any heavy per-peak inspection runs
            if not iterating_in_progress:
                try:
                    status_html.value = (
                        "<span style='color:#555;'>Fitting...</span>"
                    )
                except Exception:
                    _log_once("Fitting...")
            # Hide the same controls during Fit and optimization
            try:
                _hide_controls_during_fit_or_opt()
            except Exception:
                pass

        # Snapshot current control state
        try:
            _snapshot_current_controls()
        except Exception:
            pass
        if update_controls:
            try:
                _set_cancel_button_mode("optimize" if iterating_in_progress else "fit")
            except Exception:
                pass
            try:
                _force_cancel_fit_shown()
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_shown)
            except Exception:
                pass

        # Resolve current spectrum
        try:
            idx = spectrum_sel.value
        except Exception:
            idx = None
        _log_dbg('SPECTRUM_IDX', f"idx={idx}")
        if idx is None:
            try:
                status_html.value = "<span style='color:#a00;'>Select a spectrum before fitting.</span>"
            except Exception:
                _log_once("Select a spectrum before fitting.")
            _log_dbg('ERROR', 'No spectrum selected; aborting fit')
            _finish_fit_guard()
            return
        x_arr, y_arr = _get_xy(idx)
        _log_dbg('DATA_SHAPE', f"x_arr={getattr(x_arr,'size',None)} y_arr={getattr(y_arr,'size',None)}")
        if x_arr is None or y_arr is None or x_arr.size == 0:
            try:
                status_html.value = "<span style='color:#a00;'>No spectral data available for this row.</span>"
            except Exception:
                _log_once("No spectral data available for this row.")
            _log_dbg('ERROR', 'Spectrum has no data; aborting fit')
            _finish_fit_guard()
            return
        # Determine active fit ranges for this run (override or current slider selection)
        if override_ranges is not None:
            override_clean = _sanitize_fit_ranges(override_ranges)
        else:
            override_clean = None
        if override_clean:
            active_ranges = list(override_clean)
        else:
            try:
                active_ranges = _sanitize_fit_ranges(_current_fit_ranges())
            except Exception:
                active_ranges = []
        if not active_ranges:
            try:
                lo_full = float(np.nanmin(x_arr))
                hi_full = float(np.nanmax(x_arr))
                if hi_full < lo_full:
                    lo_full, hi_full = hi_full, lo_full
                active_ranges = [(lo_full, hi_full)]
            except Exception:
                active_ranges = []
        fit_ranges_for_this_run = list(active_ranges)
        _log_dbg('ACTIVE_RANGES', f"fit_ranges_for_this_run={fit_ranges_for_this_run}")
        peaks_x, peaks_y = _get_peaks(idx)
        _log_dbg('PEAK_COUNTS', f"peaks_x_len={len(peaks_x)} peaks_y_len={len(peaks_y)}")
        if not peaks_x:
            try:
                status_html.value = "<span style='color:#a00;'>No peaks found. Run peak-finding or add peaks.</span>"
            except Exception:
                _log_once("No peaks found. Run peak-finding or add peaks.")
            # Clear component traces
            try:
                with fig.batch_update():
                    if len(fig.data) >= 2:
                        fig.data = tuple(fig.data[:2])
            except Exception:
                pass
            _finish_fit_guard()
            return
        # Eager materialization fallback: if sliders/checkboxes not built yet (e.g., after Close),
        # construct a minimal control set for all detected peaks using the full current span.
        try:
            if (not include_checkboxes) or (len(center_sliders) == 0):
                # Clear any prior partial lists
                include_checkboxes[:] = []
                center_sliders[:] = []
                sigma_sliders[:] = []
                amplitude_sliders[:] = []
                alpha_sliders[:] = []
                center_window_sliders[:] = []
                amplitude_mode_toggles[:] = []
                center_mode_toggles[:] = []
                sigma_mode_toggles[:] = []
                center_slider_peak_indices[:] = []
                # Derive default constants (fallbacks if not defined)
                default_sigma = globals().get('PER_PEAK_DEFAULT_SIGMA', 5.0)
                default_center_window = globals().get('PER_PEAK_DEFAULT_CENTER_WINDOW', 5.0)
                # Build widgets
                for pi, (px, py) in enumerate(zip(peaks_x, peaks_y)):
                    try:
                        pxv = float(px)
                        pyv = float(py)
                    except Exception:
                        continue
                    cb = widgets.Checkbox(value=True, description=f"Include {pi+1}")
                    include_checkboxes.append(cb)
                    center_slider = widgets.FloatSlider(value=pxv, min=pxv-abs(default_center_window), max=pxv+abs(default_center_window), step=max(abs(default_center_window)/200.0, 1e-3), description=f"Center {pi+1}", layout=widgets.Layout(width="48%"))
                    _set_peak_initial_center(idx, pi, pxv)
                    _set_peak_user_center(idx, pi, pxv)
                    center_sliders.append(center_slider)
                    sigma_slider = widgets.FloatSlider(value=default_sigma, min=1e-3, max=1e3, step=default_sigma/200.0 if default_sigma>0 else 0.1, description=f"Sigma {pi+1}", layout=widgets.Layout(width="48%"))
                    sigma_sliders.append(sigma_slider)
                    amp_guess = abs(pyv) * max(1.0, float(default_sigma))
                    amplitude_slider = widgets.FloatSlider(value=amp_guess, min=0.0, max=amp_guess * 10.0 if amp_guess>0 else 1.0, step=(amp_guess/200.0) if amp_guess>0 else 0.1, description=f"Amp {pi+1}", layout=widgets.Layout(width="48%"))
                    amplitude_sliders.append(amplitude_slider)
                    alpha_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01, description=f"α {pi+1}", layout=widgets.Layout(width="48%"))
                    alpha_sliders.append(alpha_slider)
                    cwin_slider = widgets.FloatSlider(value=default_center_window, min=0.1, max=max(default_center_window*4.0, 1.0), step=max(default_center_window/200.0, 0.01), description=f"Win {pi+1}", layout=widgets.Layout(width="48%"))
                    center_window_sliders.append(cwin_slider)
                    amp_mode = widgets.Dropdown(options=['Auto','Manual'], value='Auto', description=f"A-mode {pi+1}")
                    center_mode = widgets.Dropdown(options=CENTER_MODE_OPTIONS, value=CENTER_MODE_WINDOW, description=f"C-mode {pi+1}")
                    sigma_mode = widgets.Dropdown(options=['Auto','Manual'], value='Auto', description=f"S-mode {pi+1}")
                    amplitude_mode_toggles.append(amp_mode)
                    center_mode_toggles.append(center_mode)
                    sigma_mode_toggles.append(sigma_mode)
                    center_slider_peak_indices.append(pi)
                    # Add to peak_controls_box if available
                    try:
                        peak_controls_box.children = tuple(list(peak_controls_box.children) + [widgets.HBox([cb, center_slider, sigma_slider, amplitude_slider, alpha_slider, cwin_slider, amp_mode, center_mode, sigma_mode])])
                    except Exception:
                        pass
                    try:
                        _bind_center_slider(center_slider, pi, idx)
                    except Exception:
                        pass
        except Exception:
            pass
        # Rebuild included list using ORIGINAL peak indices (preserve visible numbering)
        try:
            for _chk_idx, _cb in enumerate(include_checkboxes):
                # Attach original index if missing; center_slider_peak_indices holds original mapping
                try:
                    if not hasattr(_cb, '_original_idx'):
                        _cb._original_idx = center_slider_peak_indices[_chk_idx] if _chk_idx < len(center_slider_peak_indices) else _chk_idx
                except Exception:
                    pass
        except Exception:
            pass
        included = []
        try:
            for _cb in include_checkboxes:
                try:
                    if getattr(_cb, 'value', False):
                        included.append(getattr(_cb, '_original_idx', None))
                except Exception:
                    pass
            included = [i for i in included if i is not None]
        except Exception:
            pass
        _log_dbg('INCLUDED', f"included_original_indices={included}")
        blocked_labels = ""
        blocked_included = []
        try:
            if 'peak_box_cache' in globals() and isinstance(peak_box_cache, dict):
                filtered = []
                for peak_idx in included:
                    entry = peak_box_cache.get(peak_idx)
                    if entry and entry.get('materialized'):
                        inc_widget = entry.get('include')
                        try:
                            inc_value = getattr(inc_widget, "value", True)
                        except Exception:
                            inc_value = True
                        try:
                            include_flag = bool(inc_value)
                        except Exception:
                            include_flag = True
                        if not include_flag:
                            blocked_included.append(peak_idx)
                            continue
                    filtered.append(peak_idx)
                if blocked_included:
                    blocked_labels = ", ".join(f"Peak {pi+1}" for pi in blocked_included)
                    _log_dbg('INCLUDE_ENFORCE', f"Skipped materialized peaks without include checked: {[pi+1 for pi in blocked_included]}")
                    if filtered:
                        _log_once(f"Skipped {blocked_labels}; check 'Include peak in Fit' to include them.")
                included = filtered
        except Exception:
            pass
        # Enforce fit-range membership even when fallback controls create all peaks
        slider_centers = {}
        try:
            for pos, orig_idx in enumerate(center_slider_peak_indices):
                val = None
                if pos < len(center_sliders):
                    slider_obj = center_sliders[pos]
                    try:
                        val = float(getattr(slider_obj, "value"))
                    except Exception:
                        val = None
                needs_fallback = False
                if val is None:
                    needs_fallback = True
                else:
                    try:
                        if 'np' in globals() and hasattr(np, 'isfinite') and not np.isfinite(val):
                            needs_fallback = True
                    except Exception:
                        pass
                if needs_fallback:
                    try:
                        val = float(peaks_x[orig_idx])
                    except Exception:
                        val = None
                slider_centers[orig_idx] = val
        except Exception:
            pass
        if fit_ranges_for_this_run:
            filtered_included = []
            dropped_for_range = []
            for orig_idx in included:
                center_val = slider_centers.get(orig_idx)
                if center_val is None:
                    try:
                        center_val = float(peaks_x[orig_idx])
                    except Exception:
                        continue
                try:
                    if 'np' in globals() and hasattr(np, 'isfinite') and not np.isfinite(center_val):
                        continue
                except Exception:
                    pass
                in_any = False
                for lo_v, hi_v in fit_ranges_for_this_run:
                    try:
                        if lo_v <= center_val <= hi_v:
                            in_any = True
                            break
                    except Exception:
                        continue
                if in_any:
                    filtered_included.append(orig_idx)
                else:
                    dropped_for_range.append(orig_idx)
            if dropped_for_range:
                _log_dbg('RANGE_FILTER', f"Dropped peaks outside fit ranges: {[pi+1 for pi in dropped_for_range]}")
            included = filtered_included
        # Diagnostic: enumerate peaks whose center falls within current fit ranges
        try:
            fit_ranges_current = list(fit_ranges_for_this_run)
            if not fit_ranges_current:
                try:
                    fit_ranges_current = _sanitize_fit_ranges(_current_fit_ranges())
                except Exception:
                    fit_ranges_current = []
            in_range_lines = []
            for orig_idx, cx in enumerate(peaks_x):
                try:
                    cxf = float(cx)
                except Exception:
                    continue
                in_any = False
                for rlo, rhi in fit_ranges_current:
                    try:
                        if rlo <= cxf <= rhi:
                            in_any = True; break
                    except Exception:
                        pass
                if in_any:
                    try:
                        mat_flag = peak_box_cache.get(orig_idx, {}).get('materialized') if 'peak_box_cache' in globals() else None
                    except Exception:
                        mat_flag = None
                    in_range_lines.append(f"{orig_idx+1}:{cxf:.2f}:{'Y' if mat_flag else 'N'}")
            if in_range_lines:
                _debug_log(f"[IN_RANGE_PEAKS] {in_range_lines} (format originalIdx:center:materialized)")
        except Exception:
            pass
        # Mapping no longer needed; included already carries original indices
        if not included:
            # Fallback: determine in-range peaks still marked for inclusion using persisted state.
            available_in_range = []
            try:
                ranges_for_fallback = list(fit_ranges_for_this_run)
            except Exception:
                ranges_for_fallback = []
            if not ranges_for_fallback:
                try:
                    ranges_for_fallback = _sanitize_fit_ranges(_current_fit_ranges())
                except Exception:
                    ranges_for_fallback = []

            def _user_requested_include(orig_idx: int) -> bool:
                # Consult the live include checkbox mapping if accessible.
                try:
                    if center_slider_peak_indices and orig_idx in center_slider_peak_indices:
                        pos = center_slider_peak_indices.index(orig_idx)
                        if pos < len(include_checkboxes):
                            return bool(getattr(include_checkboxes[pos], 'value', True))
                except Exception:
                    pass
                try:
                    if orig_idx < len(include_checkboxes):
                        return bool(getattr(include_checkboxes[orig_idx], 'value', True))
                except Exception:
                    pass
                # Default to included when no explicit preference exists.
                return True

            if ranges_for_fallback:
                for orig_idx, cx in enumerate(peaks_x):
                    try:
                        cx_val = float(cx)
                    except Exception:
                        continue
                    in_any = False
                    for lo_v, hi_v in ranges_for_fallback:
                        try:
                            if lo_v <= cx_val <= hi_v:
                                in_any = True
                                break
                        except Exception:
                            continue
                    if not in_any:
                        continue
                    if not _user_requested_include(orig_idx):
                        continue
                    available_in_range.append(orig_idx)

            if available_in_range:
                included = available_in_range
            else:
                try:
                    status_html.value = "<span style='color:#a00;'>No peaks selected.</span>"
                except Exception:
                    _log_once("No peaks selected.")
                _log_dbg('ERROR', 'No included peaks after fallback; aborting')
                _finish_fit_guard()
                return

        # Re-sync slider references ONLY with fully materialized widgets.
        # Previous logic overwrote real user-modified slider widgets with lazy placeholders,
        # causing fits to ignore recent UI changes. We now preserve existing lists unless
        # the cache entry is marked materialized (real ipywidgets object, not _LazyPlaceholder).
        try:
            if 'peak_box_cache' in globals() and isinstance(peak_box_cache, dict):
                real_alpha = []
                real_center = []
                real_sigma = []
                real_amp = []
                real_center_window = []
                # Also refresh mode toggles (previously omitted, causing stale 'Auto' captures)
                real_amp_mode = []
                real_center_mode = []
                real_sigma_mode = []
                for idx_slider, cb in enumerate(include_checkboxes):
                    # Map row position directly to canonical/original peak index
                    peak_idx = None
                    try:
                        if center_slider_peak_indices and idx_slider < len(center_slider_peak_indices):
                            peak_idx = center_slider_peak_indices[idx_slider]
                    except Exception:
                        peak_idx = None
                    if peak_idx is None:
                        # Secondary fallback: legacy included_index_map (if present)
                        try:
                            for k, v in included_index_map.items():
                                if v is cb:
                                    peak_idx = k
                                    break
                        except Exception:
                            peak_idx = None
                    if peak_idx is None:
                        peak_idx = idx_slider
                    entry = peak_box_cache.get(peak_idx)
                    if entry and entry.get('materialized'):
                        # Only adopt if a real widget (heuristic: has .observe attribute AND not our placeholder class)
                        def _use_or_fallback(key, fallback_list):
                            # Support legacy 'amp_mode' key alongside 'amplitude_mode'
                            w = entry.get(key)
                            if w is None and key == 'amplitude_mode':
                                w = entry.get('amp_mode')
                            try:
                                if w is None:
                                    return fallback_list[idx_slider]
                                cls_name = w.__class__.__name__
                                if cls_name.startswith('_Lazy'):
                                    # Placeholder; abort replacement for entire list
                                    return fallback_list[idx_slider]
                                return w
                            except Exception:
                                return fallback_list[idx_slider]
                        real_alpha.append(_use_or_fallback('alpha', alpha_sliders))
                        real_center.append(_use_or_fallback('center', center_sliders))
                        real_sigma.append(_use_or_fallback('sigma', sigma_sliders))
                        real_amp.append(_use_or_fallback('amplitude', amplitude_sliders))
                        real_center_window.append(_use_or_fallback('center_window', center_window_sliders))
                        # Mode widgets: adopt if materialized; keys match those stored in cache entries
                        real_amp_mode.append(_use_or_fallback('amplitude_mode', amplitude_mode_toggles))
                        real_center_mode.append(_use_or_fallback('center_mode', center_mode_toggles))
                        real_sigma_mode.append(_use_or_fallback('sigma_mode', sigma_mode_toggles))
                    else:
                        # Keep the existing widget references
                        real_alpha.append(alpha_sliders[idx_slider])
                        real_center.append(center_sliders[idx_slider])
                        real_sigma.append(sigma_sliders[idx_slider])
                        real_amp.append(amplitude_sliders[idx_slider])
                        real_center_window.append(center_window_sliders[idx_slider])
                        try:
                            real_amp_mode.append(amplitude_mode_toggles[idx_slider])
                            real_center_mode.append(center_mode_toggles[idx_slider])
                            real_sigma_mode.append(sigma_mode_toggles[idx_slider])
                        except Exception:
                            pass
                # Replace only if sizes match; otherwise keep originals
                if (
                    len(real_alpha) == len(alpha_sliders)
                    and len(real_center) == len(center_sliders)
                    and len(real_amp_mode) == len(amplitude_mode_toggles)
                    and len(real_center_mode) == len(center_mode_toggles)
                    and len(real_sigma_mode) == len(sigma_mode_toggles)
                ):
                    alpha_sliders = real_alpha
                    center_sliders = real_center
                    sigma_sliders = real_sigma
                    amplitude_sliders = real_amp
                    center_window_sliders = real_center_window
                    amplitude_mode_toggles = real_amp_mode
                    center_mode_toggles = real_center_mode
                    sigma_mode_toggles = real_sigma_mode
                    try:
                        _log_dbg('WIDGET_RESYNC', f"Resynced {len(center_sliders)} sliders & mode toggles from cache")
                    except Exception:
                        pass
                # Final safeguard: force enumeration of actual displayed widgets in peak_controls_box
                # Some user edits may occur on newly materialized widgets not yet reflected in cached lists.
                try:
                    force_alpha = []
                    force_center = []
                    force_sigma = []
                    force_amp = []
                    force_center_window = []
                    force_amp_mode = []
                    force_center_mode = []
                    force_sigma_mode = []
                    force_include = []
                    # Each child HBox layout: [cb, center, sigma, amplitude, alpha, win, amp_mode, center_mode, sigma_mode]
                    for child in getattr(peak_controls_box, 'children', []):
                        try:
                            widgets_list = list(getattr(child, 'children', []))
                            if len(widgets_list) < 9:
                                continue
                            cb, c_sl, s_sl, a_sl, al_sl, w_sl, am_mode, c_mode, s_mode = widgets_list[:9]
                            force_include.append(cb)
                            force_center.append(c_sl)
                            force_sigma.append(s_sl)
                            force_amp.append(a_sl)
                            force_alpha.append(al_sl)
                            force_center_window.append(w_sl)
                            force_amp_mode.append(am_mode)
                            force_center_mode.append(c_mode)
                            force_sigma_mode.append(s_mode)
                        except Exception:
                            continue
                    if force_center and len(force_center) == len(center_sliders):
                        include_checkboxes = force_include
                        center_sliders = force_center
                        sigma_sliders = force_sigma
                        amplitude_sliders = force_amp
                        alpha_sliders = force_alpha
                        center_window_sliders = force_center_window
                        amplitude_mode_toggles = force_amp_mode
                        center_mode_toggles = force_center_mode
                        sigma_mode_toggles = force_sigma_mode
                        _log_dbg('WIDGET_FORCE_REFRESH', f"Forced refresh adopted {len(center_sliders)} live widget rows")
                    # Deep traversal fallback: some layouts may nest widgets more than one level.
                    try:
                        def _walk(node, acc):
                            try:
                                kids = getattr(node, 'children', [])
                            except Exception:
                                kids = []
                            for k in kids:
                                acc.append(k)
                                _walk(k, acc)
                        all_widgets = []
                        _walk(peak_controls_box, all_widgets)
                        # Maps for peak index 1 (0-based) by description pattern
                        desc_map = {
                            'center': None,
                            'sigma': None,
                            'amp': None,
                            'alpha': None,
                            'win': None,
                            'include': None,
                            'A-mode': None,
                            'C-mode': None,
                            'S-mode': None,
                        }
                        for w in all_widgets:
                            try:
                                d = getattr(w, 'description', '')
                            except Exception:
                                continue
                            if not isinstance(d, str):
                                continue
                            if d.startswith('Center 1') and desc_map['center'] is None:
                                desc_map['center'] = w
                            elif d.startswith('Sigma 1') and desc_map['sigma'] is None:
                                desc_map['sigma'] = w
                            elif d.startswith('Amp 1') and desc_map['amp'] is None:
                                desc_map['amp'] = w
                            elif d.startswith('α 1') and desc_map['alpha'] is None:
                                desc_map['alpha'] = w
                            elif d.startswith('Win 1') and desc_map['win'] is None:
                                desc_map['win'] = w
                            elif d.startswith('Include 1') and desc_map['include'] is None:
                                desc_map['include'] = w
                            elif d.startswith('A-mode 1') and desc_map['A-mode'] is None:
                                desc_map['A-mode'] = w
                            elif d.startswith('C-mode 1') and desc_map['C-mode'] is None:
                                desc_map['C-mode'] = w
                            elif d.startswith('S-mode 1') and desc_map['S-mode'] is None:
                                desc_map['S-mode'] = w
                        # If we found a center or sigma widget whose value differs from our current list, adopt full deep set for index 0 only.
                        adopt = False
                        try:
                            if desc_map['center'] is not None and center_sliders and center_sliders[0] is not desc_map['center']:
                                adopt = True
                            if desc_map['sigma'] is not None and sigma_sliders and sigma_sliders[0] is not desc_map['sigma']:
                                adopt = True
                            # Value mismatch triggers adoption (user changed value on a different widget instance)
                            if desc_map['win'] is not None and center_window_sliders and abs(float(center_window_sliders[0].value) - float(getattr(desc_map['win'],'value', center_window_sliders[0].value))) > 1e-9:
                                adopt = True
                        except Exception:
                            pass
                        if adopt:
                            try:
                                center_sliders[0] = desc_map['center'] or center_sliders[0]
                                sigma_sliders[0] = desc_map['sigma'] or sigma_sliders[0]
                                amplitude_sliders[0] = desc_map['amp'] or amplitude_sliders[0]
                                alpha_sliders[0] = desc_map['alpha'] or alpha_sliders[0]
                                center_window_sliders[0] = desc_map['win'] or center_window_sliders[0]
                                amplitude_mode_toggles[0] = desc_map['A-mode'] or amplitude_mode_toggles[0]
                                center_mode_toggles[0] = desc_map['C-mode'] or center_mode_toggles[0]
                                sigma_mode_toggles[0] = desc_map['S-mode'] or sigma_mode_toggles[0]
                                _log_dbg('WIDGET_DEEP_REFRESH', 'Adopted deep traversal widgets for Peak 1')
                            except Exception:
                                pass
                        # Emit diagnostic snapshot of Peak 1 widget values right before parameter capture
                        try:
                            _log_dbg('PEAK1_WIDGET_VALUES', 'center={c} sigma={s} amp={a} win={w} modes center={mc} sigma={ms} amp={ma}'.format(
                                c=(center_sliders[0].value if center_sliders else 'NA'),
                                s=(sigma_sliders[0].value if sigma_sliders else 'NA'),
                                a=(amplitude_sliders[0].value if amplitude_sliders else 'NA'),
                                w=(center_window_sliders[0].value if center_window_sliders else 'NA'),
                                mc=(center_mode_toggles[0].value if center_mode_toggles else 'NA'),
                                ms=(sigma_mode_toggles[0].value if sigma_mode_toggles else 'NA'),
                                ma=(amplitude_mode_toggles[0].value if amplitude_mode_toggles else 'NA'),
                            ))
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        # Re-capture current widget parameter values and modes AFTER any re-sync so Manual changes are used.
        param_seed_map = {}
        # New: ensure all INCLUDED peaks are materialized so user edits apply.
        try:
            if 'peak_box_cache' in globals():
                auto_mat = []
                for _inc_idx in included:
                    try:
                        entry = peak_box_cache.get(_inc_idx)
                        if entry and not entry.get('materialized'):
                            # Programmatically open the toggle to invoke existing materialization logic
                            tg = None
                            toggled_open = False
                            try:
                                tg = entry.get('toggle')
                                if tg is not None:
                                    try:
                                        original_state = bool(tg.value)
                                    except Exception:
                                        original_state = True
                                    if not original_state:
                                        tg.value = True  # triggers observer to materialize
                                        toggled_open = True
                            except Exception:
                                pass
                            # Briefly poll to allow materialization observers to run
                            try:
                                import time as _time
                            except Exception:
                                _time = None
                            for _wait in range(10):
                                try:
                                    entry = peak_box_cache.get(_inc_idx)
                                except Exception:
                                    entry = None
                                if entry and entry.get('materialized'):
                                    break
                                try:
                                    if _time is not None:
                                        _time.sleep(0.02)
                                except Exception:
                                    pass
                            # Check final state
                            try:
                                entry = peak_box_cache.get(_inc_idx)
                            except Exception:
                                entry = None
                            if entry and entry.get('materialized'):
                                auto_mat.append(_inc_idx+1)
                            # Restore original toggle state so UIs stay collapsed after auto materialization
                            try:
                                if tg is not None and toggled_open:
                                    tg.value = False
                            except Exception:
                                pass
                        # Ensure cache sync even if flag missing
                        try:
                            _ensure_cache_from_accordion(_inc_idx)
                        except Exception:
                            pass
                        try:
                            if peak_box_cache.get(_inc_idx, {}).get('materialized') and (_inc_idx+1) not in auto_mat:
                                auto_mat.append(_inc_idx+1)
                        except Exception:
                            pass
                    except Exception:
                        pass
                if auto_mat:
                    _debug_log(f"[PEAK_AUTOMATERIALIZE] peaks={auto_mat} (forced materialization for included peaks)")
                # Emit summary of included peaks materialization state
                try:
                    mat_states = []
                    for _inc_idx in included:
                        try:
                            ms = peak_box_cache.get(_inc_idx, {}).get('materialized')
                            mat_states.append(f"{_inc_idx+1}:{'Y' if ms else 'N'}")
                        except Exception:
                            mat_states.append(f"{_inc_idx+1}:?")
                    _debug_log(f"[MATERIALIZATION_SUMMARY] included={ [i+1 for i in included] } states={mat_states}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # Optional mapping: if center_slider_peak_indices length matches center_sliders, use positional mapping.
            for peak_idx in included:
                # Determine positional index into slider lists via center_slider_peak_indices mapping
                pos = peak_idx
                if center_slider_peak_indices:
                    try:
                        pos = center_slider_peak_indices.index(peak_idx)
                    except Exception:
                        pos = peak_idx
                original_idx_for_cache = peak_idx
                # Extract values with fallbacks
                def _val(lst, fallback):
                    try:
                        return float(lst[pos].value)
                    except Exception:
                        return float(fallback)
                c_val = _val(center_sliders, peaks_x[peak_idx] if peak_idx < len(peaks_x) else float('nan'))
                s_val = _val(sigma_sliders, globals().get('PER_PEAK_DEFAULT_SIGMA', 5.0))
                a_val = _val(amplitude_sliders, (abs(float(peaks_y[peak_idx])) * max(1.0, s_val)) if peak_idx < len(peaks_y) else 1.0)
                alpha_val = _val(alpha_sliders, 0.5)
                w_val = _val(center_window_sliders, globals().get('PER_PEAK_DEFAULT_CENTER_WINDOW', 5.0))
                def _mode(lst, key):
                    try:
                        m = lst[pos].value
                    except Exception:
                        m = 'Auto' if key in ('amplitude','sigma') else CENTER_MODE_WINDOW
                    if key == 'center':
                        return _normalize_center_mode_value(m)
                    if m not in ('Auto','Manual'):
                        m = 'Auto'
                    return m
                # Cache-first capture: if peak has been materialized, read directly from cache widgets.
                source = 'list'
                placeholder_flags = []
                cache_entry = None
                try:
                    if 'peak_box_cache' in globals():
                        cache_entry = peak_box_cache.get(original_idx_for_cache)
                except Exception:
                    cache_entry = None
                if cache_entry and cache_entry.get('materialized'):
                    try:
                        c_val = float(getattr(cache_entry.get('center'), 'value', c_val))
                        s_val = float(getattr(cache_entry.get('sigma'), 'value', s_val))
                        a_val = float(getattr(cache_entry.get('amplitude'), 'value', a_val))
                        alpha_val = float(getattr(cache_entry.get('alpha'), 'value', alpha_val))
                        w_val = float(getattr(cache_entry.get('center_window'), 'value', w_val))
                        mode_center = _normalize_center_mode_value(getattr(cache_entry.get('center_mode'), 'value', CENTER_MODE_WINDOW))
                        mode_sigma = str(getattr(cache_entry.get('sigma_mode'), 'value', 'Auto'))
                        mode_amp = str(getattr(cache_entry.get('amplitude_mode') or cache_entry.get('amp_mode'), 'value', 'Auto'))
                        source = 'cache'
                    except Exception:
                        source = 'cache_partial'
                        mode_center = _mode(center_mode_toggles, 'center')
                        mode_sigma = _mode(sigma_mode_toggles, 'sigma')
                        mode_amp = _mode(amplitude_mode_toggles, 'amplitude')
                else:
                    # Attempt to recover materialized widgets from accordion if flag missing
                    try:
                        _ensure_cache_from_accordion(original_idx_for_cache)
                        cache_entry = peak_box_cache.get(original_idx_for_cache)
                        if cache_entry and cache_entry.get('materialized'):
                            c_val = float(getattr(cache_entry.get('center'), 'value', c_val))
                            s_val = float(getattr(cache_entry.get('sigma'), 'value', s_val))
                            a_val = float(getattr(cache_entry.get('amplitude'), 'value', a_val))
                            alpha_val = float(getattr(cache_entry.get('alpha'), 'value', alpha_val))
                            w_val = float(getattr(cache_entry.get('center_window'), 'value', w_val))
                            mode_center = _normalize_center_mode_value(getattr(cache_entry.get('center_mode'), 'value', CENTER_MODE_WINDOW))
                            mode_sigma = str(getattr(cache_entry.get('sigma_mode'), 'value', 'Auto'))
                            mode_amp = str(getattr(cache_entry.get('amplitude_mode') or cache_entry.get('amp_mode'), 'value', 'Auto'))
                            source = 'cache'
                        else:
                            raise RuntimeError('Accordion recovery not materialized')
                    except Exception:
                        pass
                    # Try to read directly from live row widgets by original peak number
                    live_row_found = False
                    try:
                        for child in getattr(peak_controls_box, 'children', []):
                            try:
                                widgets_list = list(getattr(child, 'children', []))
                                if len(widgets_list) < 9:
                                    continue
                                cb, c_sl, s_sl, a_sl, al_sl, w_sl, am_mode, c_mode, s_mode = widgets_list[:9]
                                desc = getattr(cb, 'description', '')
                                if isinstance(desc, str) and desc.strip() == f"Include {original_idx_for_cache+1}":
                                    c_val = float(getattr(c_sl, 'value', c_val))
                                    s_val = float(getattr(s_sl, 'value', s_val))
                                    a_val = float(getattr(a_sl, 'value', a_val))
                                    alpha_val = float(getattr(al_sl, 'value', alpha_val))
                                    w_val = float(getattr(w_sl, 'value', w_val))
                                    mode_center = _normalize_center_mode_value(getattr(c_mode, 'value', CENTER_MODE_WINDOW))
                                    mode_sigma = str(getattr(s_mode, 'value', 'Auto'))
                                    mode_amp = str(getattr(am_mode, 'value', 'Auto'))
                                    source = 'cache'  # treat live widgets as authoritative
                                    live_row_found = True
                                    break
                            except Exception:
                                continue
                    except Exception:
                        live_row_found = False
                    if not live_row_found:
                        # Fallback to list-based modes and note placeholders
                        mode_center = _mode(center_mode_toggles, 'center')
                        mode_sigma = _mode(sigma_mode_toggles, 'sigma')
                        mode_amp = _mode(amplitude_mode_toggles, 'amplitude')
                        def _is_placeholder(lst):
                            try:
                                cls_name = lst[pos].__class__.__name__
                                return cls_name.startswith('_Lazy')
                            except Exception:
                                return False
                        for name,lst in [('center',center_sliders),('sigma',sigma_sliders),('amp',amplitude_sliders)]:
                            if _is_placeholder(lst):
                                placeholder_flags.append(name)
                param_seed_map[original_idx_for_cache] = {
                    'center': c_val,
                    'sigma': s_val,
                    'amplitude': a_val,
                    'alpha': alpha_val,
                    'center_window': w_val,
                    'mode_center': mode_center,
                    'mode_sigma': mode_sigma,
                    'mode_amp': mode_amp,
                    'placeholders': placeholder_flags,
                    'source': source,
                    'original_idx': original_idx_for_cache,
                }
                try:
                    _debug_log(f"[PARAM_CAPTURE] peak={original_idx_for_cache+1} source={source} center={c_val} sigma={s_val} amp={a_val} win={w_val} modes c={mode_center} s={mode_sigma} a={mode_amp} placeholders={placeholder_flags}")
                    # Additional per-peak detail: widget classes and materialized flag
                    try:
                        if cache_entry:
                            cls_center = cache_entry.get('center').__class__.__name__ if cache_entry.get('center') is not None else 'None'
                            cls_sigma = cache_entry.get('sigma').__class__.__name__ if cache_entry.get('sigma') is not None else 'None'
                            cls_amp = cache_entry.get('amplitude').__class__.__name__ if cache_entry.get('amplitude') is not None else 'None'
                            _debug_log(f"[PARAM_CAPTURE_DETAIL] peak={original_idx_for_cache+1} materialized={cache_entry.get('materialized')} classes center={cls_center} sigma={cls_sigma} amp={cls_amp}")
                    except Exception:
                        pass
                except Exception:
                    pass
            # Emit diagnostic for first included peak (usually Peak 1)
            first_original = included[0]
            seed_first = param_seed_map.get(first_original, {})
            _debug_log("[PARAM_SEED_CAPTURED] peak={} center={} sigma={} amp={} win={} modes c={} s={} a={} placeholders={}".format(
                first_original+1,
                seed_first.get('center','NA'),
                seed_first.get('sigma','NA'),
                seed_first.get('amplitude','NA'),
                seed_first.get('center_window','NA'),
                seed_first.get('mode_center','NA'),
                seed_first.get('mode_sigma','NA'),
                seed_first.get('mode_amp','NA'),
                seed_first.get('placeholders','NA'),
            ))
            # Materialized peaks that are currently not part of this fit (user may have edited but range excludes them)
            try:
                if 'peak_box_cache' in globals():
                    materialized_not_included = [i+1 for i,entry in peak_box_cache.items() if entry.get('materialized') and i not in included]
                    if materialized_not_included:
                        _debug_log(f"[OPEN_NOT_INCLUDED] peaks={materialized_not_included} (materialized but excluded by current fit range)")
            except Exception:
                pass
        except Exception:
            pass

        if update_plot:
            # Prepare component traces count on main thread for consistent layout
            comp_traces_needed = len(included)
            with fig.batch_update():
                current_components = max(0, len(fig.data) - 2)
                if current_components > comp_traces_needed:
                    fig.data = tuple(list(fig.data)[: 2 + comp_traces_needed])
                elif current_components < comp_traces_needed:
                    for _k in range(comp_traces_needed - current_components):
                        # Placeholder name will be updated after fit with actual peak number
                        fig.add_scatter(
                            x=[],
                            y=[],
                            mode="lines",
                            line=dict(dash="dot"),
                            name="Peak ?",
                        )

        # Cancel any running fit and start a new one in the background
        try:
            if fit_thread is not None and fit_thread.is_alive():
                # Signal the currently running worker to stop (per-fit only)
                try:
                    if fit_cancel_token is not None:
                        fit_cancel_token.set()
                except Exception:
                    pass
        except Exception:
            pass
        # Create a fresh cancel token for this new worker and capture it locally
        local_cancel = threading.Event()
        # Keep the global cancel_event stable; only update the active per-fit token
        fit_cancel_token = local_cancel

        old_redchi = last_redchi_by_idx.get(idx, None)
        _log_dbg('OLD_REDCHI', f"old_redchi={old_redchi}")
        if update_controls:
            # Update status label immediately on main thread
            if not iterating_in_progress:
                try:
                    status_html.value = (
                        "<span style='color:#555;'>Refitting...</span>"
                        if old_redchi is not None
                        else "<span style='color:#555;'>Fitting...</span>"
                    )
                except Exception:
                    _log_once("Refitting..." if old_redchi is not None else "Fitting...")

        def _worker(local_cancel_token=local_cancel, forced_ranges=fit_ranges_for_this_run):
            nonlocal fit_thread
            try:
                _log_dbg('WORKER_START', 'Background worker thread started')
                # Use only the selected Fit X-range (or override) for fitting to prevent
                # components going nearly flat when focusing on a small region.
                if forced_ranges:
                    ranges = _sanitize_fit_ranges(forced_ranges)
                else:
                    try:
                        ranges = _sanitize_fit_ranges(_current_fit_ranges())
                    except Exception:
                        ranges = []

                if not ranges:
                    try:
                        if 'np' in globals():
                            lo_full = float(np.nanmin(x_arr))
                            hi_full = float(np.nanmax(x_arr))
                        else:
                            lo_full = float(min(x_arr))
                            hi_full = float(max(x_arr))
                        if hi_full < lo_full:
                            lo_full, hi_full = hi_full, lo_full
                        ranges = [(lo_full, hi_full)]
                    except Exception:
                        ranges = []

                _log_dbg('FIT_RANGES', f"ranges={ranges}")

                try:
                    msk = np.zeros_like(x_arr, dtype=bool)
                    if ranges:
                        for lo_v, hi_v in ranges:
                            msk |= ((x_arr >= lo_v) & (x_arr <= hi_v))
                    else:
                        msk[:] = True
                except Exception:
                    msk = np.ones_like(x_arr, dtype=bool)
                x_sub = x_arr[msk]
                y_sub = y_arr[msk]
                _log_dbg('SUBSET_SHAPE', f"x_sub={x_sub.size} y_sub={y_sub.size}")
                # If too few points in range, bail gracefully
                if x_sub.size < max(10, 3 * max(1, len(included))):
                    _on_main_thread(
                        lambda: _log_once(
                            "Selected Fit X-range has too few points for a stable fit. "
                            "Expand the range or include more peaks."
                        )
                    )
                    _finish_fit_guard()
                    return

                # Pre-fit sanity: ensure included indices are within bounds
                try:
                    valid_included = []
                    dropped = []
                    px_len = int(len(peaks_x))
                    py_len = int(len(peaks_y))
                    for i in included:
                        try:
                            if isinstance(i, (int, float)):
                                ii = int(i)
                            else:
                                ii = int(i)
                        except Exception:
                            dropped.append(i)
                            continue
                        if ii < 0 or ii >= px_len or ii >= py_len:
                            dropped.append(ii)
                        else:
                            valid_included.append(ii)
                    if dropped:
                        _on_main_thread(lambda: _log_once(f"Dropped invalid peaks: {', '.join('Peak '+str(di+1) for di in dropped if isinstance(di,int))}"))
                    included[:] = valid_included
                except Exception:
                    pass
                if not included:
                    _on_main_thread(lambda: _log_once("No valid peaks remain after deletion; nothing to fit."))
                    _finish_fit_guard()
                    return

                # Build composite model using captured param_seed_map snapshot
                comp_model = None
                params = None
                param_pre_lines = []
                for i in included:
                    try:
                        seed = param_seed_map.get(i, {})
                        mu_val = float(seed.get('center', peaks_x[i]))
                        sg_val = float(seed.get('sigma', globals().get('PER_PEAK_DEFAULT_SIGMA', 5.0)))
                        amp_val = float(seed.get('amplitude', abs(float(peaks_y[i])) * max(1.0, sg_val)))
                        alpha_val = float(seed.get('alpha', 0.5))
                        w = float(seed.get('center_window', globals().get('PER_PEAK_DEFAULT_CENTER_WINDOW', 5.0)))
                        mode_center = _normalize_center_mode_value(seed.get('mode_center', CENTER_MODE_WINDOW))
                        mode_sigma = seed.get('mode_sigma', 'Auto')
                        mode_amp = seed.get('mode_amp', 'Auto')
                        if mode_sigma not in ('Auto','Manual'):
                            mode_sigma = 'Auto'
                        if mode_amp not in ('Auto','Manual'):
                            mode_amp = 'Auto'
                        m = PseudoVoigtModel(prefix=f"p{i}_")
                        p = m.make_params()
                        # Center
                        if mode_center == CENTER_MODE_EXACT:
                            p[f"p{i}_center"].set(value=mu_val, vary=False)
                        else:
                            p[f"p{i}_center"].set(value=mu_val, min=mu_val - abs(w), max=mu_val + abs(w), vary=True)
                        # Sigma
                        if mode_sigma == 'Manual':
                            p[f"p{i}_sigma"].set(value=sg_val, vary=False)
                        else:
                            p[f"p{i}_sigma"].set(value=sg_val, min=1e-3, max=1e3, vary=True)
                        # Alpha fixed
                        p[f"p{i}_fraction"].set(value=alpha_val, min=0.0, max=1.0, vary=False)
                        # Amplitude
                        if mode_amp == 'Manual':
                            p[f"p{i}_amplitude"].set(value=amp_val, vary=False)
                        else:
                            p[f"p{i}_amplitude"].set(value=amp_val, min=0.0)
                        # Snapshot line
                        param_pre_lines.append(
                            f"Peak {i+1} PRE  center={mu_val:.6g} mode={mode_center}  sigma={sg_val:.6g} mode={mode_sigma}  amp={amp_val:.6g} mode={mode_amp}  alpha={alpha_val:.6g} fixed  win={w:.6g}"
                        )
                        if comp_model is None:
                            comp_model = m
                            params = p
                        else:
                            comp_model = comp_model + m
                            params.update(p)
                    except Exception as e:
                        _log_once(f"Error setting up parameters for peak {i+1}: {e}")
                        _finish_fit_guard()
                        return

                # Emit captured parameter snapshot (PRE) for debugging
                if param_pre_lines:
                    try:
                        if DECONV_DEBUG:
                            _debug_log("[DECONV_DEBUG] PRE-fit parameter snapshot (" + str(len(param_pre_lines)) + " peaks):")
                            for line in param_pre_lines:
                                _debug_log(line)
                        else:
                            _debug_log(f"[PRE_CAPTURE] Captured {len(param_pre_lines)} peak parameter lines (debug flag off)")
                    except Exception:
                        pass
                else:
                    _log_dbg('PRE_PARAMS_EMPTY', 'No param_pre_lines captured at all')

                # iter_cb to allow cooperative cancellation
                def _iter_cb(params_, iter_, resid_, *args, **kws):
                    if local_cancel_token.is_set():
                        raise KeyboardInterrupt("Fit cancelled by user")

                _log_dbg('LMFIT_START', f"Starting lmfit with {len(included)} components")
                result = comp_model.fit(y_sub, params, x=x_sub, iter_cb=_iter_cb)
                _log_dbg('LMFIT_DONE', f"Fit result redchi={getattr(result,'redchi',None)}")
                if local_cancel_token.is_set():
                    return
                # Evaluate results for full x-array for plotting (optional)
                if update_plot:
                    y_fit = result.eval(x=x_arr)
                    comps = result.eval_components(x=x_arr)
                else:
                    y_fit = None
                    comps = None

                # Post-fit parameter snapshot & mode/vary verification
                # Post-fit snapshot (always capture; conditional detailed logging)
                if True:
                    try:
                        param_post_lines = []
                        vary_mismatches = []
                        for i in included:
                            prefix = f"p{i}_"
                            def _pv(name):
                                try:
                                    par = result.params.get(prefix + name)
                                    return float(getattr(par, 'value', par))
                                except Exception:
                                    return float('nan')
                            c_val = _pv('center')
                            s_val = _pv('sigma')
                            a_val = _pv('amplitude')
                            fr_val = _pv('fraction')
                            try:
                                raw_center_mode = center_mode_toggles[i].value if i < len(center_mode_toggles) else CENTER_MODE_WINDOW
                            except Exception:
                                raw_center_mode = CENTER_MODE_WINDOW
                            mode_center = _normalize_center_mode_value(raw_center_mode)
                            try:
                                mode_sigma = sigma_mode_toggles[i].value if i < len(sigma_mode_toggles) else 'Auto'
                            except Exception:
                                mode_sigma = 'Auto'
                            try:
                                mode_amp = amplitude_mode_toggles[i].value if i < len(amplitude_mode_toggles) else 'Auto'
                            except Exception:
                                mode_amp = 'Auto'
                            def _vary(name):
                                try:
                                    par = result.params.get(prefix + name)
                                    return bool(getattr(par, 'vary', False))
                                except Exception:
                                    return False
                            vc = _vary('center')
                            vs = _vary('sigma')
                            va = _vary('amplitude')
                            if (mode_center == CENTER_MODE_EXACT and vc) or (mode_center == CENTER_MODE_WINDOW and not vc):
                                vary_mismatches.append(f"Peak {i+1} center mode={mode_center} vary={vc}")
                            if (mode_sigma == 'Manual' and vs) or (mode_sigma == 'Auto' and not vs):
                                vary_mismatches.append(f"Peak {i+1} sigma mode={mode_sigma} vary={vs}")
                            if (mode_amp == 'Manual' and va) or (mode_amp == 'Auto' and not va):
                                vary_mismatches.append(f"Peak {i+1} amplitude mode={mode_amp} vary={va}")
                            param_post_lines.append(
                                f"Peak {i+1} POST center={c_val:.6g} vary={vc}  sigma={s_val:.6g} vary={vs}  amp={a_val:.6g} vary={va}  alpha={fr_val:.6g} fixed"
                            )
                        if DECONV_DEBUG:
                            _debug_log("[DECONV_DEBUG] POST-fit parameter snapshot (" + str(len(param_post_lines)) + " peaks):")
                            for line in param_post_lines:
                                _debug_log(line)
                            if vary_mismatches:
                                _debug_log("[DECONV_DEBUG] VARY FLAG MISMATCHES: " + "; ".join(vary_mismatches))
                            else:
                                _debug_log("[DECONV_DEBUG] VARY FLAGS OK")
                        else:
                            _debug_log(f"[POST_CAPTURE] Captured {len(param_post_lines)} peak parameter lines (debug flag off)")
                            if vary_mismatches:
                                _debug_log("[POST_CAPTURE] VARY FLAG MISMATCHES: " + "; ".join(vary_mismatches))
                            else:
                                _debug_log("[POST_CAPTURE] VARY FLAGS OK")
                    except Exception:
                        pass

                # Persist last successful result for this spectrum
                try:
                    last_result_by_idx[idx] = result
                except Exception:
                    pass

                # Persist redchi immediately (used by silent objective evaluation)
                try:
                    rc_val = getattr(result, "redchi", None)
                    if rc_val is not None:
                        try:
                            last_redchi_by_idx[idx] = float(rc_val)
                        except Exception:
                            last_redchi_by_idx[idx] = rc_val
                except Exception:
                    pass

                # Silent mode: do not touch the UI/plot; just clear guards.
                if (not update_plot) and (not update_controls):
                    try:
                        fit_thread = None
                    except Exception:
                        pass
                    try:
                        _finish_fit_guard()
                    except Exception:
                        pass
                    return

                def _apply_results_on_ui():
                    nonlocal iter_summary_pending, iter_start_redchi, iter_final_redchi
                    # Always try to update the plot, but don't let plotting failures
                    # prevent status messages from updating.
                    plot_ok = True
                    try:
                        comp_traces_needed = len(included)
                        with fig.batch_update():
                            # Ensure we have exactly 2 + comp_traces_needed traces
                            current_components = max(0, len(fig.data) - 2)
                            if current_components > comp_traces_needed:
                                fig.data = tuple(
                                    list(fig.data)[: 2 + comp_traces_needed]
                                )
                            elif current_components < comp_traces_needed:
                                for _k in range(
                                    comp_traces_needed - current_components
                                ):
                                    fig.add_scatter(
                                        x=[],
                                        y=[],
                                        mode="lines",
                                        line=dict(dash="dot"),
                                        name="Peak ?",
                                    )
                            # Update data and fit traces
                            fig.data[0].x = x_arr.tolist()
                            fig.data[0].y = y_arr.tolist()
                            fig.data[1].x = x_arr.tolist()
                            fig.data[1].y = (
                                y_fit.tolist()
                                if hasattr(y_fit, "tolist")
                                else list(y_fit)
                            )
                            # Update component traces safely
                            for comp_pos, orig_idx in enumerate(included):
                                key = f"p{orig_idx}_"
                                y_comp = comps.get(key, np.zeros_like(x_arr))
                                fig.data[2 + comp_pos].x = x_arr.tolist()
                                fig.data[2 + comp_pos].y = (
                                    y_comp.tolist()
                                    if hasattr(y_comp, "tolist")
                                    else list(y_comp)
                                )
                                # Ensure component names reflect peak number and FINAL fitted center value
                                try:
                                    mu_display = None
                                    try:
                                        par_obj = result.params.get(f"p{orig_idx}_center")
                                        if par_obj is not None:
                                            mu_display = float(getattr(par_obj, 'value', par_obj))
                                    except Exception:
                                        mu_display = None
                                    if mu_display is None:
                                        try:
                                            # orig_idx corresponds to slider index; use direct lookup
                                            mu_display = float(center_sliders[orig_idx].value) if orig_idx < len(center_sliders) else float(peaks_x[orig_idx])
                                        except Exception:
                                            mu_display = float(peaks_x[orig_idx]) if orig_idx < len(peaks_x) else float('nan')
                                    # Legend numbering uses original peak index (i+1) to match toggle titles
                                    fig.data[2 + comp_pos].name = f"Peak {orig_idx+1} @ {mu_display:.1f} cm⁻¹"
                                    try:
                                        _set_peak_fit_center(idx, orig_idx, mu_display)
                                    except Exception:
                                        pass
                                    try:
                                        _update_peak_toggle_label(orig_idx, spectrum_idx=idx)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                        # Keep slider widgets in sync with fitted parameter values so the UI
                        # reflects the latest optimization step even when it ran off-thread.
                        try:
                            for orig_idx in included:
                                pos = None
                                if center_slider_peak_indices:
                                    try:
                                        pos = center_slider_peak_indices.index(orig_idx)
                                    except ValueError:
                                        pos = None
                                if pos is None and orig_idx < len(center_sliders):
                                    pos = orig_idx
                                cache_entry = None
                                try:
                                    if isinstance(peak_box_cache, dict):
                                        cache_entry = peak_box_cache.get(orig_idx)
                                except Exception:
                                    cache_entry = None
                                def _real_widget(candidate, fallback_list):
                                    if candidate is not None and hasattr(candidate, "value"):
                                        return candidate
                                    if pos is None:
                                        return None
                                    if fallback_list is None:
                                        return None
                                    try:
                                        if pos < len(fallback_list):
                                            widget_obj = fallback_list[pos]
                                            if hasattr(widget_obj, "value"):
                                                return widget_obj
                                    except Exception:
                                        pass
                                    return None
                                sigma_widget = _real_widget(
                                    cache_entry.get("sigma") if cache_entry else None,
                                    sigma_sliders,
                                )
                                amplitude_widget = _real_widget(
                                    cache_entry.get("amplitude") if cache_entry else None,
                                    amplitude_sliders,
                                )
                                prefix = f"p{orig_idx}_"
                                try:
                                    par_center = result.params.get(prefix + "center")
                                except Exception:
                                    par_center = None
                                if par_center is not None:
                                    try:
                                        center_val = float(getattr(par_center, "value", par_center))
                                        _set_peak_fit_center(idx, orig_idx, center_val)
                                    except Exception:
                                        _set_peak_fit_center(idx, orig_idx, par_center)
                                try:
                                    _update_peak_toggle_label(orig_idx, spectrum_idx=idx)
                                except Exception:
                                    pass
                                try:
                                    par_sigma = result.params.get(prefix + "sigma")
                                except Exception:
                                    par_sigma = None
                                if (
                                    par_sigma is not None
                                    and sigma_widget is not None
                                    and hasattr(sigma_widget, "value")
                                ):
                                    try:
                                        sigma_val = float(getattr(par_sigma, "value", par_sigma))
                                        _set_quiet(sigma_widget, "value", sigma_val)
                                    except Exception:
                                        pass
                                try:
                                    par_amp = result.params.get(prefix + "amplitude")
                                except Exception:
                                    par_amp = None
                                if (
                                    par_amp is not None
                                    and amplitude_widget is not None
                                    and hasattr(amplitude_widget, "value")
                                ):
                                    try:
                                        amp_val = float(getattr(par_amp, "value", par_amp))
                                        _set_quiet(amplitude_widget, "value", amp_val)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            _snapshot_current_controls()
                        except Exception:
                            pass
                    except Exception:
                        plot_ok = False
                    # Update message regardless of plot success
                    # Update status label text (suppressed during iterative correction)
                    new_redchi = getattr(result, "redchi", np.nan)
                    try:
                        if not iterating_in_progress:
                            # If an iteration just completed, prefer the pre- vs
                            # post-iteration summary
                            if (
                                iter_summary_pending
                                and iter_start_redchi is not None
                                and iter_final_redchi is not None
                            ):
                                try:
                                    old_str = f"{float(iter_start_redchi):.4g}"
                                except Exception:
                                    old_str = str(iter_start_redchi)
                                try:
                                    new_str = f"{float(iter_final_redchi):.4g}"
                                except Exception:
                                    new_str = str(iter_final_redchi)
                                status_html.value = (
                                    f"<span style='color:#000;'>Iterative correction "
                                    f"complete. Reduced chi-square: ("
                                    f"{old_str}) ---&gt; ({new_str})</span>"
                                )
                                # Clear the pending summary after showing it once
                                iter_summary_pending = False
                            else:
                                # Format values consistently, showing change when refitting
                                try:
                                    old_str = (
                                        f"{old_redchi:.4g}"
                                        if old_redchi is not None
                                        else None
                                    )
                                except Exception:
                                    old_str = (
                                        str(old_redchi)
                                        if old_redchi is not None
                                        else None
                                    )
                                try:
                                    new_str = f"{new_redchi:.4g}"
                                except Exception:
                                    new_str = str(new_redchi)
                                if old_redchi is not None:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Refit complete. "
                                        f"Reduced chi-square: ("
                                        f"{old_str}) ---&gt; ({new_str})</span>"
                                    )
                                else:
                                    status_html.value = (
                                        f"<span style='color:#000;'>Fit complete. "
                                        f"Reduced chi-square: ("
                                        f"{new_str})</span>"
                                    )
                        # Persist the new redchi so subsequent runs are treated as
                        # refits
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                        except Exception:
                            # If np.isfinite is not available or new_redchi is not
                            # numeric, store raw
                            try:
                                last_redchi_by_idx[idx] = float(new_redchi)
                            except Exception:
                                last_redchi_by_idx[idx] = new_redchi
                        if not plot_ok:
                            # Also echo a note in the log output
                            _log_once(
                                "(Note: Plot update partially failed; re-run to refresh components.)"
                            )
                    except Exception:
                        if not iterating_in_progress:
                            status_html.value = (
                                "<span style='color:#000;'>Fit complete.</span>"
                            )
                        try:
                            if np.isfinite(new_redchi):
                                last_redchi_by_idx[idx] = float(new_redchi)
                            else:
                                last_redchi_by_idx[idx] = new_redchi
                        except Exception:
                            last_redchi_by_idx[idx] = new_redchi
                    # Update Cancel Fit button visibility after this fit completes
                    try:
                        _update_cancel_fit_visibility()
                    except Exception:
                        pass
                    if update_controls and (not iterating_in_progress):
                        try:
                            _show_controls_after_fit_or_opt()
                        except Exception:
                            pass

                _on_main_thread(_apply_results_on_ui)
                # Ensure the Cancel Fit button hides after the worker fully ends
                # by clearing the thread reference and scheduling a final UI update.
                try:
                    fit_thread = None
                except Exception:
                    pass
                # Only hide at the end if not iterating/frozen
                if update_controls and (not iterating_in_progress):
                    try:
                        _on_main_thread(_force_cancel_fit_hidden)
                    except Exception:
                        pass
                    try:
                        _on_main_thread(_force_cancel_fit_hidden)
                    except Exception:
                        pass
                # Mark fit complete (successful path)
                try:
                    _finish_fit_guard()
                except Exception:
                    pass
            except KeyboardInterrupt:
                # Cancellation requested
                def _notify_cancel():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                "<span style='color:#a00;'>Fit cancelled.</span>"
                            )
                        except Exception:
                            _log_once("Fit cancelled.")
                    # Keep visible during iteration; outer loop will hide at end
                    try:
                        if iterating_in_progress or cancel_fit_btn_frozen:
                            _force_cancel_fit_shown()
                        else:
                            _force_cancel_fit_hidden()
                    except Exception:
                        pass
                    if update_controls and (not iterating_in_progress):
                        try:
                            _show_controls_after_fit_or_opt()
                        except Exception:
                            pass
                        try:
                            delete_peaks_btn.layout.display = ""
                        except Exception:
                            pass

                _on_main_thread(_notify_cancel)
                # Clear thread reference and trigger a final visibility update
                try:
                    fit_thread = None
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
                # Mark fit completion after cancellation
                try:
                    _finish_fit_guard()
                except Exception:
                    pass
            except Exception as e:

                def _notify_error():
                    if not iterating_in_progress:
                        try:
                            status_html.value = (
                                f"<span style='color:#a00;'>Fit failed: {e}</span>"
                            )
                        except Exception:
                            _log_once(f"Fit failed: {e}")
                    # Keep visible during iteration; final cleanup handles hide.
                    try:
                        if iterating_in_progress or cancel_fit_btn_frozen:
                            _force_cancel_fit_shown()
                        else:
                            _force_cancel_fit_hidden()
                    except Exception:
                        pass
                    if update_controls and (not iterating_in_progress):
                        # Restore controls hidden during Fit
                        try:
                            _show(fit_btn)
                            _show(add_peaks_btn)
                            _show(iter_btn)
                            _show(reset_all_row)
                            _show(save_btn)
                            _show(fit_range_row)
                        except Exception:
                            pass
                        try:
                            _show(canonize_btn)
                            _show(load_canon_btn)
                        except Exception:
                            pass
                        try:
                            delete_peaks_btn.layout.display = ""
                        except Exception:
                            pass

                _on_main_thread(_notify_error)
                # Clear thread reference and trigger a final visibility update
                try:
                    fit_thread = None
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
                # Mark fit completion after error
                try:
                    _finish_fit_guard()
                except Exception:
                    pass

        fit_thread = threading.Thread(target=_worker, daemon=True)
        fit_thread.start()
        if update_controls:
            # Show the Cancel Fit button immediately when a fit starts (guard will be
            # cleared only when the worker actually finishes to prevent overlapping fits).
            try:
                _force_cancel_fit_shown()
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_shown)
            except Exception:
                try:
                    _update_cancel_fit_visibility()
                except Exception:
                    pass
        return None

    # Track displayed spectrum independently for deconvolution
    current_idx_deconv = None
    cache_owner_idx = None

    def _on_spectrum_change(*_):
        nonlocal bulk_update_in_progress
        nonlocal on_spectrum_change_inflight, last_on_spectrum_change_ts
        nonlocal cache_owner_idx
        nonlocal range_sliders
        nonlocal adding_mode, new_peak_xs, new_peak_windows
        nonlocal previous_in_range_indices
        nonlocal xmin, xmax
        nonlocal default_fit_range_value
        nonlocal persisted_fit_ranges
        # Debounce/guard: prevent rapid double invocation
        try:
            now_ts = time.time()
        except Exception:
            now_ts = 0.0
        if on_spectrum_change_inflight:
            return
        if (now_ts - last_on_spectrum_change_ts) < 0.03:
            return
        on_spectrum_change_inflight = True
        try:
            idx = spectrum_sel.value
            nonlocal current_idx_deconv
            current_idx_deconv = idx
            try:
                _update_spectrum_counter()
            except Exception:
                pass
            # Reset canonize confirmation UI on spectrum switch
            try:
                _show(canonize_btn)
                _hide(canonize_confirm_btn)
                _hide(canonize_cancel_btn)
            except Exception:
                pass
            # Persist current Fit X-ranges before switching.
            try:
                persisted_fit_ranges = _current_fit_ranges()
            except Exception:
                pass

            # Immediately clear the displayed peak list so no prior spectrum's peaks linger
            # if something fails during rebuild.
            try:
                peak_controls_box.children = [widgets.HTML("<b>Loading spectrum . . .</b>")]
            except Exception:
                pass

            # Hard reset any transient UI state so NOTHING carries over between spectra.
            # New spectrum should initialize from defaults and the DataFrame only.
            try:
                # Use a fresh dict so cached widget objects cannot be reused.
                global peak_box_cache
                peak_box_cache = {}
            except Exception:
                pass
            cache_owner_idx = None
            try:
                previous_in_range_indices = set()
            except Exception:
                pass
            try:
                peak_center_state_by_idx.clear()
            except Exception:
                pass
            try:
                last_result_by_idx.clear()
            except Exception:
                pass
            try:
                last_redchi_by_idx.clear()
            except Exception:
                pass
            try:
                included_index_map.clear()
            except Exception:
                pass

            # Force-exit delete mode (if it was active) and clear any markers/staging.
            try:
                nonlocal deleting_mode
                deleting_mode = False
            except Exception:
                pass
            try:
                staged_delete_indices.clear()
            except Exception:
                pass
            try:
                _clear_delete_peak_shapes()
            except Exception:
                pass
            # Exit add-peak mode and clear any temporary markers
            try:
                adding_mode = False
                new_peak_xs.clear()
                new_peak_windows.clear()
            except Exception:
                pass
            try:
                _clear_add_peak_shapes()
            except Exception:
                pass

            # Update x-span for this spectrum, then restore persisted Fit X-ranges (clamped).
            try:
                x_arr, y_arr = _get_xy(idx)
            except Exception:
                x_arr, y_arr = (None, None)
            if x_arr is not None and getattr(x_arr, 'size', 0) > 0:
                try:
                    xmin = float(np.nanmin(x_arr))
                    xmax = float(np.nanmax(x_arr))
                    if xmax < xmin:
                        xmin, xmax = xmax, xmin
                except Exception:
                    pass
            try:
                default_fit_range_value = (float(xmin), float(xmax))
            except Exception:
                pass

            # Avoid triggering expensive rebuilds while updating sliders.
            bulk_prev = bulk_update_in_progress
            bulk_update_in_progress = True
            try:
                _apply_ranges_to_sliders(persisted_fit_ranges, xmin, xmax)
            finally:
                bulk_update_in_progress = bulk_prev

            # Update plot data for the new spectrum and clear any prior fit
            try:
                x_arr, y_arr = _get_xy(idx)
                if x_arr is not None and y_arr is not None:
                    with fig.batch_update():
                        fig.data[0].x = x_arr.tolist()
                        fig.data[0].y = y_arr.tolist()
            except Exception:
                pass
            try:
                with fig.batch_update():
                    fig.data[1].x = []
                    fig.data[1].y = []
                    while len(fig.data) > 2:
                        fig.data = tuple(fig.data[:2])
            except Exception:
                pass

            # Rebuild per-peak controls from scratch (defaults + DataFrame)
            bulk_update_in_progress = True
            try:
                _refresh_peak_control_widgets(idx)
            finally:
                bulk_update_in_progress = False
            # Refresh the fit range overlay for the new spectrum
            try:
                _update_fit_range_indicator()
            except Exception:
                pass
            # Freeze y-axis to this spectrum's data range to reduce flicker on updates
            try:
                _fix_y_range(idx)
            except Exception:
                pass
            # Clear any prior fit from plot; wait for user to click Fit
            try:
                with fig.batch_update():
                    fig.data[1].x = []
                    fig.data[1].y = []
                    while len(fig.data) > 2:
                        fig.data = tuple(fig.data[:2])
            except Exception:
                pass
            try:
                status_html.value = "<span style='color:#555;'>Spectrum changed. Click Fit to compute.</span>"
            except Exception:
                pass
            # Persist time selection to session state
            try:
                if idx is not None and "Time" in FTIR_DataFrame.columns:
                    _set_session_selection(time=FTIR_DataFrame.loc[idx].get("Time"))
            except Exception:
                pass
            # Show mark row now that a spectrum is actively selected
            try:
                if 'mark_row' in locals() or 'mark_row' in globals():
                    mark_row.layout.display = ""
            except Exception:
                pass
        finally:
            on_spectrum_change_inflight = False
            try:
                last_on_spectrum_change_ts = time.time()
            except Exception:
                last_on_spectrum_change_ts = now_ts

    # No range checkboxes to manage

    def _save_for_file(b):
        if _recent_click("save_for_file"):
            return
        idx = spectrum_sel.value
        res = last_result_by_idx.get(idx)
        # Require a Fit for the current spectrum before saving.
        if res is None:
            _log_once("No current fit. Click 'Fit' to compute, then click 'Save' again.")
            return
        # Only save parameters for peaks included in the most recent fit
        all_peaks_x, _ = _get_peaks(idx)
        state_for_save = peak_center_state_by_idx.get(idx, {})

        def _resolve_original_peak_index(cb, pos: int):
            """Return 0-based original peak index for a checkbox row."""
            orig = getattr(cb, "_original_idx", None)
            if orig is not None:
                try:
                    return int(orig)
                except Exception:
                    pass
            # Fallback: parse from the visible label (e.g., "Include 19")
            try:
                import re

                desc = str(getattr(cb, "description", ""))
                m = re.search(r"(\d+)", desc)
                if m:
                    # UI labels are 1-based
                    return max(0, int(m.group(1)) - 1)
            except Exception:
                pass
            # Fallback: use the peak-index mapping list when available (in-range subset UIs)
            try:
                if pos < len(center_slider_peak_indices):
                    return int(center_slider_peak_indices[pos])
            except Exception:
                pass
            return int(pos)

        included_original = []
        for pos, cb in enumerate(include_checkboxes):
            try:
                if not getattr(cb, "value", False):
                    continue
            except Exception:
                continue
            try:
                orig_idx = _resolve_original_peak_index(cb, pos)
            except Exception:
                orig_idx = pos
            included_original.append(orig_idx)
        if not included_original:
            _log_once("No peaks selected. Select peaks then Save.")
            return
        # Preserve ordering by original peak index and remove duplicates
        included_original = sorted(dict.fromkeys(included_original))

        def _coerce_ranges(ranges_in):
            """Return list of [lo, hi] floats for the current Fit X-ranges."""
            out = []
            if not ranges_in:
                return out
            for pair in ranges_in:
                try:
                    lo_v, hi_v = pair
                except Exception:
                    continue
                try:
                    lo_f = float(lo_v)
                    hi_f = float(hi_v)
                except Exception:
                    continue
                try:
                    if 'np' in globals():
                        if not (bool(np.isfinite(lo_f)) and bool(np.isfinite(hi_f))):
                            continue
                except Exception:
                    pass
                if hi_f < lo_f:
                    lo_f, hi_f = hi_f, lo_f
                out.append([lo_f, hi_f])
            return out

        try:
            selected_x_ranges = _coerce_ranges(_current_fit_ranges())
        except Exception:
            selected_x_ranges = []

        def _coerce_float(val):
            try:
                v = float(val)
            except Exception:
                return None
            try:
                if 'np' in globals():
                    if bool(np.isfinite(v)):
                        return v
                    return None
            except Exception:
                pass
            try:
                import math

                if math.isfinite(v):
                    return v
            except Exception:
                pass
            return None

        def _slider_value_for(orig_idx: int, sliders, default=None):
            """Return the slider.value corresponding to orig_idx, if available."""
            try:
                pos = center_slider_peak_indices.index(orig_idx)
            except Exception:
                return default
            try:
                return sliders[pos].value
            except Exception:
                return default

        out = []
        for orig_idx in included_original:
            prefix = f"p{orig_idx}_"
            d = {}
            for name in ("amplitude", "center", "sigma", "fraction"):
                p = res.params.get(prefix + name)
                if p is not None:
                    try:
                        d[name] = float(p.value)
                    except Exception:
                        d[name] = p.value
            peak_state = state_for_save.get(orig_idx, {})
            initial_val = peak_state.get('initial')
            user_val = peak_state.get('user')
            fit_val = peak_state.get('fit')

            if initial_val is None and orig_idx < len(all_peaks_x):
                initial_val = all_peaks_x[orig_idx]

            # Do not restore user centers from per-spectrum caches
            if user_val is None:
                try:
                    pos = center_slider_peak_indices.index(orig_idx)
                    user_val = center_sliders[pos].value
                except Exception:
                    user_val = initial_val

            if 'center' in d:
                fit_val = d.get('center')
            else:
                if fit_val is None:
                    p_center = res.params.get(prefix + 'center')
                    if p_center is not None:
                        try:
                            fit_val = float(p_center.value)
                            d['center'] = fit_val
                        except Exception:
                            fit_val = p_center.value
                elif fit_val is not None:
                    d['center'] = fit_val
            if fit_val is None:
                fit_val = user_val

            coerced_initial = _coerce_float(initial_val)
            coerced_user = _coerce_float(user_val)
            coerced_fit = _coerce_float(fit_val)
            # Persist only the fit-derived center; fall back to user or initial values
            final_center = None
            if coerced_fit is not None:
                final_center = coerced_fit
            elif fit_val is not None:
                final_center = fit_val
            elif coerced_user is not None:
                final_center = coerced_user
            elif user_val is not None:
                final_center = user_val
            elif coerced_initial is not None:
                final_center = coerced_initial
            elif initial_val is not None:
                final_center = initial_val
            if final_center is None and 'center' in d:
                final_center = d['center']
            if final_center is not None:
                d['center'] = final_center

            # Ensure sigma, alpha, and A are always saved.
            # - lmfit uses 'fraction' for pseudo-Voigt Lorentz fraction; we expose it as 'alpha'
            # - lmfit uses 'amplitude'; we persist it as 'A'
            try:
                sigma_fit = _coerce_float(d.get('sigma'))
            except Exception:
                sigma_fit = None
            if sigma_fit is None:
                sigma_fit = _coerce_float(_slider_value_for(orig_idx, sigma_sliders))
            if sigma_fit is not None:
                d['sigma'] = float(sigma_fit)

            try:
                alpha_fit = _coerce_float(d.get('fraction'))
            except Exception:
                alpha_fit = None
            if alpha_fit is None:
                alpha_fit = _coerce_float(d.get('alpha'))
            if alpha_fit is None:
                alpha_fit = _coerce_float(_slider_value_for(orig_idx, alpha_sliders))
            if alpha_fit is not None:
                # Keep both keys for compatibility
                d['alpha'] = float(alpha_fit)
                d['fraction'] = float(alpha_fit)

            try:
                amp_fit = _coerce_float(d.get('amplitude'))
            except Exception:
                amp_fit = None
            if amp_fit is None:
                amp_fit = _coerce_float(_slider_value_for(orig_idx, amplitude_sliders))
            if amp_fit is not None:
                d['A'] = float(amp_fit)
            # Avoid saving redundant 'amplitude' alongside 'A'
            try:
                d.pop('amplitude', None)
            except Exception:
                pass
            # Persist the original peak number (1-based, matches UI)
            d['peak_number'] = int(orig_idx) + 1
            out.append(d)
        _persist_deconv_results(idx, out)
        try:
            _persist_deconv_x_ranges(idx, selected_x_ranges)
        except Exception:
            pass
        try:
            file_label = FTIR_DataFrame.loc[idx, "File Name"]
        except Exception:
            file_label = str(idx)
        _log_once(
            f"Saved deconvolution for file '{file_label}'. Stored results in DataFrame."
        )
        try:
            _deconv_changes["saved"].append((idx, len(out)))
        except Exception:
            pass

    def _close_ui(b):
        global peak_box_cache, peak_accordion
        nonlocal adding_mode, previous_in_range_indices
        # Emit a session summary before closing widgets; keep log_html visible
        try:
            lines = _session_summary_lines(_deconv_changes, context="deconvolution")
            _emit_session_summary(
                log_html, lines, title="Session summary (Deconvolution)"
            )
        except Exception:
            pass
        # Signal cancellation and close widgets promptly
        try:
            cancel_event.set()
        except Exception:
            pass
        # Reset add-peaks workflow state prior to tearing down widgets
        try:
            adding_mode = False
        except Exception:
            pass
        try:
            new_peak_xs.clear()
        except Exception:
            pass
        try:
            new_peak_windows.clear()
        except Exception:
            pass
        # No shared cross-spectrum state to clear
        try:
            spectrum_sel.close()
            material_dd.close()
            conditions_dd.close()
            include_bad_cb.close()
            add_peaks_btn.close()
            accept_new_peaks_btn.close()
            redo_new_peaks_btn.close()
            cancel_new_peaks_btn.close()
            # Ensure iterative/cancel controls disappear on close
            iter_btn.close()
            cancel_fit_btn.close()
            save_btn.close()
            close_btn.close()
            reset_all_btn.close()
            # Close fit-range related widgets before clearing caches
            for sl in list(range_sliders):
                if hasattr(sl, "close"):
                    try:
                        sl.close()
                    except Exception:
                        pass
            range_sliders.clear()
            try:
                range_sliders_box.close()
            except Exception:
                pass
            try:
                add_range_btn.close()
            except Exception:
                pass
            try:
                fit_range_row.close()
            except Exception:
                pass
            # Safely close any parameter widgets (placeholders lack .close)
            for lst in [alpha_sliders, center_sliders, sigma_sliders, amplitude_sliders, center_window_sliders, amplitude_mode_toggles, center_mode_toggles, sigma_mode_toggles, include_checkboxes]:
                for w in lst:
                    if hasattr(w, "close"):
                        try:
                            w.close()
                        except Exception:
                            pass
            # Close any cached peak boxes (handles lazy placeholders + materialized widgets)
            try:
                for entry in getattr(peak_box_cache, 'values', lambda: [])():
                    for k in ['box','toggle','details','include','alpha','center','sigma','amplitude','center_window','amp_mode','center_mode','sigma_mode']:
                        w = entry.get(k)
                        if hasattr(w, 'close'):
                            try:
                                w.close()
                            except Exception:
                                pass
            except Exception:
                pass
            peak_controls_box.close()
            try:
                peak_controls_section.close()
            except Exception:
                pass
            try:
                excluded_peaks_html.close()
            except Exception:
                pass
            try:
                mark_bad_btn.close()
            except Exception:
                pass
            try:
                mark_good_btn.close()
            except Exception:
                pass
            try:
                mark_row.close()
            except Exception:
                pass
            try:
                status_row.close()
            except Exception:
                pass
            try:
                optimize_status_html.close()
            except Exception:
                pass
            try:
                add_peaks_slider.close()
            except Exception:
                pass
            try:
                add_peaks_text.close()
            except Exception:
                pass
            try:
                add_peaks_add_btn.close()
            except Exception:
                pass
            try:
                colab_add_row.close()
            except Exception:
                pass
            try:
                colab_add_help.close()
            except Exception:
                pass
            # Close the main UI container and status label so no stray widgets remain
            try:
                status_html.close()
            except Exception:
                pass
            try:
                ui.close()
            except Exception:
                pass
            # Leave the log widget displayed so the last message remains visible
            fig.close()
        except Exception:
            pass
        # Reset peak-related state so a fresh invocation reconstructs everything
        try:
            alpha_sliders[:] = []
            center_sliders[:] = []
            sigma_sliders[:] = []
            amplitude_sliders[:] = []
            center_window_sliders[:] = []
            include_checkboxes[:] = []
            amplitude_mode_toggles[:] = []
            center_mode_toggles[:] = []
            sigma_mode_toggles[:] = []
            center_slider_peak_indices[:] = []
        except Exception:
            pass
        try:
            if isinstance(peak_box_cache, dict):
                peak_box_cache.clear()
        except Exception:
            pass
        try:
            peak_box_cache = {}
        except Exception:
            pass
        try:
            peak_accordion = None
        except Exception:
            pass
        try:
            peak_center_state_by_idx.clear()
        except Exception:
            pass
        try:
            last_result_by_idx.clear()
        except Exception:
            pass
        try:
            last_redchi_by_idx.clear()
        except Exception:
            pass
        try:
            included_index_map.clear()
        except Exception:
            pass
        try:
            peak_label_widgets.clear()
            original_peak_centers.clear()
            last_active_ranges.clear()
        except Exception:
            pass
        try:
            previous_in_range_indices = set()
        except Exception:
            pass
        try:
            last_click_ts.clear()
        except Exception:
            pass
        try:
            globals().pop('_refresh_slider_lists', None)
        except Exception:
            pass
        # Clear fit guard so next session can start fitting
        try:
            fit_update_inflight = False
        except Exception:
            pass

    # Wire events
    spectrum_sel.observe(_on_spectrum_change, names="value")
    material_dd.observe(_rebuild_spectrum_options, names="value")
    conditions_dd.observe(_rebuild_spectrum_options, names="value")

    # Persist material/conditions selections
    def _persist_pd_filters(_=None):
        try:
            _set_session_selection(
                material=material_dd.value, conditions=conditions_dd.value
            )
        except Exception:
            pass

    material_dd.observe(_persist_pd_filters, names="value")
    conditions_dd.observe(_persist_pd_filters, names="value")
    # Observers
    # Observe all dynamic range sliders
    try:
        fit_range.observe(_on_fit_range_change, names="value")
    except Exception:
        pass

    # Wire reset buttons
    def _reset_all(_b=None):
        nonlocal bulk_update_in_progress
        bulk_update_in_progress = True
        try:
            lo, hi = default_fit_range_value
            fit_range.value = [float(lo), float(hi)]
        except Exception:
            pass
        # Reset per-peak controls
        try:
            for cb in include_checkboxes:
                cb.value = DEFAULT_INCLUDE
        except Exception:
            pass
        try:
            for s in alpha_sliders:
                s.value = DEFAULT_ALPHA
        except Exception:
            pass
        try:
            for s in center_sliders:
                # Reset center to its detected peak position stored in attribute if available; fallback to current
                pass  # center sliders retain current value
        except Exception:
            pass
        try:
            for s in center_window_sliders:
                s.value = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            pass
        try:
            for s in sigma_sliders:
                s.value = float(PER_PEAK_DEFAULT_SIGMA)
        except Exception:
            pass
        try:
            for s in amplitude_sliders:
                # Do not recompute default; keep current amplitude as starting point
                pass
        except Exception:
            pass
        try:
            for tb in amplitude_mode_toggles:
                tb.value = 'Auto'
            for tb in center_mode_toggles:
                tb.value = CENTER_MODE_WINDOW
            for tb in sigma_mode_toggles:
                tb.value = 'Auto'
        except Exception:
            pass
        _snapshot_current_controls()
        bulk_update_in_progress = False
        try:
            status_html.value = "<span style='color:#555;'>All peak parameters reset. Click Fit to update.</span>"
        except Exception:
            pass
    reset_all_btn.on_click(_reset_all)

    def _on_fit_click(_b=None):
        # Always allow fit to start, regardless of parameter mode
        if _recent_click("fit"):
            return
        # Show Cancel Fit button immediately
        try:
            _set_cancel_button_mode("fit")
        except Exception:
            pass
        try:
            _on_main_thread(_force_cancel_fit_shown)
        except Exception:
            try:
                _update_cancel_fit_visibility()
            except Exception:
                pass
        _fit_and_update_plot()

    fit_btn.on_click(_on_fit_click)
    save_btn.on_click(_save_for_file)
    close_btn.on_click(_close_ui)

    # Helper: quietly set a widget attribute without triggering control-change side effects
    def _set_quiet(widget, attr, value):
        """Set a widget attribute while suppressing observers (thread-safe)."""
        nonlocal bulk_update_in_progress

        def _assign():
            nonlocal bulk_update_in_progress
            bulk_update_in_progress = True
            try:
                setattr(widget, attr, value)
            except Exception:
                pass
            finally:
                bulk_update_in_progress = False

        try:
            if threading.current_thread() is threading.main_thread():
                _assign()
                return
        except Exception:
            pass

        done_event = threading.Event()

        def _wrapped():
            try:
                _assign()
            finally:
                try:
                    done_event.set()
                except Exception:
                    pass

        try:
            _on_main_thread(_wrapped)
        except Exception:
            _wrapped()
            return

        try:
            done_event.wait(timeout=2.0)
        except Exception:
            pass

    # Fix y-axis range to the current data to prevent autoscale flicker during updates
    def _fix_y_range(idx):
        try:
            x_arr, y_arr = _get_xy(idx)
        except Exception:
            x_arr, y_arr = None, None
        if y_arr is None:
            return
        try:
            y0_min = float(np.nanmin(y_arr))
            y0_max = float(np.nanmax(y_arr))
        except Exception:
            return
        try:
            with fig.batch_update():
                fig.update_yaxes(autorange=False, range=[y0_min, y0_max])
        except Exception:
            pass

    def _iteratively_correct_worker():
        """Coordinate-descent style tuning of parameters to reduce reduced chi-square.

        Iterates over per-peak α sliders (for included peaks), the center window, and
        the initial sigma, trying +/- step changes one parameter at a time. Keeps a
        change only if reduced chi-square improves. Stops when a full sweep makes no
        improvements, or after a safety cap of sweeps.
        """

        nonlocal iterating_in_progress, iter_start_redchi
        nonlocal iter_final_redchi, iter_summary_pending
        nonlocal cancel_fit_btn_frozen, fit_thread, fit_update_inflight

        idx = spectrum_sel.value

        FIT_WAIT_TIMEOUT = 40.0  # seconds
        FIT_WAIT_POLL_INTERVAL = 0.1
        iteration_timed_out = False
        iteration_timeout_message = ""

        def _sanitize_ranges(ranges_in):
            cleaned = []
            if not ranges_in:
                return cleaned
            has_np = 'np' in globals()
            for pair in ranges_in:
                try:
                    lo, hi = pair
                    lo_f = float(lo)
                    hi_f = float(hi)
                except Exception:
                    continue
                finite_ok = True
                if has_np:
                    try:
                        finite_ok = bool(np.isfinite(lo_f)) and bool(np.isfinite(hi_f))
                    except Exception:
                        finite_ok = True
                if not finite_ok:
                    continue
                if hi_f < lo_f:
                    lo_f, hi_f = hi_f, lo_f
                cleaned.append((lo_f, hi_f))
            return cleaned

        try:
            fit_ranges_snapshot = _sanitize_ranges(_current_fit_ranges())
        except Exception:
            fit_ranges_snapshot = []
        try:
            _debug_log(f"[ITER_RANGE_SNAPSHOT] ranges={fit_ranges_snapshot}")
        except Exception:
            pass

        def _wait_for_fit_idle(max_wait=FIT_WAIT_TIMEOUT, *, reason: str = "fit to finish"):
            """Wait until no fit thread is active and the guard is clear.

            Returns True when idle, False on timeout, and None if cancellation is requested.
            """
            try:
                start_ts = time.time()
            except Exception:
                start_ts = 0.0
            while True:
                try:
                    if cancel_event.is_set():
                        return None
                except Exception:
                    pass
                guard_active = False
                try:
                    guard_active = bool(fit_update_inflight)
                except Exception:
                    guard_active = False
                thread_active = False
                try:
                    th = fit_thread
                    thread_active = th is not None and th.is_alive()
                except Exception:
                    thread_active = False
                if not guard_active and not thread_active:
                    return True
                if max_wait is not None:
                    try:
                        now_ts = time.time()
                    except Exception:
                        now_ts = start_ts
                    if (now_ts - start_ts) >= max_wait:
                        return False
                try:
                    time.sleep(FIT_WAIT_POLL_INTERVAL)
                except Exception:
                    break
            return False

        # Helper: run fit synchronously and return last redchi
        def _run_fit_and_wait(force_ranges=None):
            nonlocal iteration_timed_out, iteration_timeout_message
            # If cancellation is already requested, don't start a new fit
            try:
                if cancel_event.is_set():
                    return np.inf
            except Exception:
                pass
            local_ranges = force_ranges
            if not local_ranges:
                try:
                    local_ranges = _sanitize_ranges(_current_fit_ranges())
                except Exception:
                    local_ranges = []
            wait_prev = _wait_for_fit_idle(reason="previous fit to finish")
            if wait_prev is False:
                msg_prev = "Timed out waiting for previous fit to finish. Optimization stopped."
                iteration_timed_out = True
                iteration_timeout_message = msg_prev
                _log_once(msg_prev)
                try:
                    status_html.value = f"<span style='color:#a00;'>{msg_prev}</span>"
                except Exception:
                    pass
                try:
                    cancel_event.set()
                except Exception:
                    pass
                return np.inf
            if wait_prev is None:
                return np.inf
            override_ranges = list(local_ranges) if local_ranges else None
            _fit_and_update_plot(
                ignore_debounce=True,
                override_ranges=override_ranges,
                update_plot=False,
                update_controls=False,
            )
            wait_current = _wait_for_fit_idle(reason="fit to finish")
            if wait_current is False:
                msg_curr = "Timed out waiting for fit to finish. Optimization stopped."
                iteration_timed_out = True
                iteration_timeout_message = msg_curr
                _log_once(msg_curr)
                try:
                    status_html.value = f"<span style='color:#a00;'>{msg_curr}</span>"
                except Exception:
                    pass
                try:
                    cancel_event.set()
                except Exception:
                    pass
                return np.inf
            if wait_current is None:
                return np.inf
            # Prefer the most recent lmfit result (set inside the worker thread)
            try:
                res_obj = last_result_by_idx.get(idx)
            except Exception:
                res_obj = None
            if res_obj is not None:
                try:
                    rc_val = getattr(res_obj, "redchi", np.inf)
                except Exception:
                    rc_val = np.inf
            else:
                rc_val = last_redchi_by_idx.get(idx, np.inf)
            try:
                rc_val = float(rc_val)
            except Exception:
                rc_val = np.inf
            if not np.isfinite(rc_val):
                # Treat non-finite reduced chi-square as a failed fit for iteration purposes
                return np.inf
            return rc_val

        # Establish baseline reduced chi-square from snapshot taken at click time.
        # Fall back to one quick fit if no prior value exists.
        start_rc = iter_start_redchi
        try:
            start_rc = float(start_rc)
        except Exception:
            start_rc = np.inf
        if not np.isfinite(start_rc):
            # No prior redchi recorded; compute once.
            try:
                status_html.value = "<span style='color:#555;'>Iteratively correcting (pre-fit)...</span>"
            except Exception:
                pass
            start_rc = _run_fit_and_wait(fit_ranges_snapshot)
        base_rc = start_rc
        # Iteration change counter: increments whenever a parameter change is kept
        iteration_changes = 0
        # Minimal running status; will update the counter on each kept change
        try:
            status_html.value = (
                "<span style='color:#555;'>iterating... (iterations so far: 0)</span>"
            )
        except Exception:
            pass
        try:
            if cancel_event.is_set():
                # Show old -> current comparison on cancel
                try:
                    old_str = f"{start_rc:.4g}"
                except Exception:
                    old_str = str(start_rc)
                try:
                    new_str = f"{base_rc:.4g}"
                except Exception:
                    new_str = str(base_rc)
                try:
                    status_html.value = (
                        f"<span style='color:#a00;'>Iterative correction cancelled. "
                        f"Reduced chi-square: ({old_str}) ---&gt; ({new_str})</span>"
                    )
                except Exception:
                    pass
                try:
                    _on_main_thread(_update_cancel_fit_visibility)
                except Exception:
                    pass
                # Allow normal fit status updates again; unfreeze and hide now
                iterating_in_progress = False
                try:
                    cancel_fit_btn_frozen = False
                except Exception:
                    pass
                try:
                    _on_main_thread(_force_cancel_fit_hidden)
                except Exception:
                    pass
                try:
                    _on_main_thread(_show_controls_after_fit_or_opt)
                except Exception:
                    try:
                        _show_controls_after_fit_or_opt()
                    except Exception:
                        pass
                return
        except Exception:
            pass
        if not np.isfinite(base_rc):
            base_rc = _run_fit_and_wait(fit_ranges_snapshot)
        if not np.isfinite(base_rc):
            base_rc = np.inf

        def _active_alpha_targets():
            targets = []
            try:
                ensure_len = len(include_checkboxes)
            except Exception:
                ensure_len = 0
            for pos in range(ensure_len):
                cb = include_checkboxes[pos]
                include_flag = False
                try:
                    include_flag = bool(getattr(cb, "value", False))
                except Exception:
                    include_flag = False
                if not include_flag:
                    continue
                slider = None
                try:
                    if pos < len(alpha_sliders):
                        slider = alpha_sliders[pos]
                except Exception:
                    slider = None
                if slider is None or not hasattr(slider, "value"):
                    continue
                try:
                    float(slider.value)
                except Exception:
                    continue
                peak_idx = None
                try:
                    peak_idx = getattr(cb, "_original_idx")
                except Exception:
                    peak_idx = None
                if peak_idx is None and center_slider_peak_indices:
                    try:
                        if pos < len(center_slider_peak_indices):
                            peak_idx = center_slider_peak_indices[pos]
                    except Exception:
                        peak_idx = None
                targets.append({"pos": pos, "slider": slider, "peak_idx": peak_idx})
            return targets

        alpha_targets = _active_alpha_targets()
        if not alpha_targets:
            try:
                status_html.value = (
                    "<span style='color:#a00;'>Cannot iterate: select at least one in-range peak with 'Include peak in Fit'.</span>"
                )
            except Exception:
                _log_once("Cannot iterate: no included peaks within the current fit range.")
            try:
                iterating_in_progress = False
                cancel_fit_btn_frozen = False
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_hidden)
            except Exception:
                pass
            try:
                _on_main_thread(_show_controls_after_fit_or_opt)
            except Exception:
                try:
                    _show_controls_after_fit_or_opt()
                except Exception:
                    pass
            return

        def _clamp(val, lo, hi):
            try:
                return max(lo, min(hi, val))
            except Exception:
                return val

        def _ensure_real_slider(target):
            slider = target.get("slider")
            if slider is not None and not isinstance(slider, (_LazyPlaceholder, _LazyModePlaceholder)):
                return
            peak_idx = target.get("peak_idx")
            if peak_idx is None:
                return
            entry = None
            try:
                if isinstance(peak_box_cache, dict):
                    entry = peak_box_cache.get(peak_idx)
            except Exception:
                entry = None
            if not entry:
                try:
                    _ensure_cache_from_accordion(peak_idx)
                    if isinstance(peak_box_cache, dict):
                        entry = peak_box_cache.get(peak_idx)
                except Exception:
                    entry = None
            if not entry:
                return
            if not entry.get('materialized'):
                toggle = entry.get('toggle')
                if toggle is not None:
                    toggle_ready = threading.Event()

                    def _open_toggle():
                        try:
                            toggle.value = True
                        except Exception:
                            pass
                        finally:
                            toggle_ready.set()

                    _on_main_thread(_open_toggle)
                    try:
                        toggle_ready.wait(timeout=1.5)
                    except Exception:
                        pass
                try:
                    for _ in range(40):
                        try:
                            time.sleep(0.05)
                        except Exception:
                            pass
                        if isinstance(peak_box_cache, dict):
                            entry = peak_box_cache.get(peak_idx)
                        if entry and entry.get('materialized'):
                            break
                except Exception:
                    pass
            if not entry or not entry.get('materialized'):
                return
            real_slider = entry.get('alpha')
            if real_slider is not None:
                target['slider'] = real_slider
                pos = target.get('pos')
                try:
                    if pos is not None and pos < len(alpha_sliders):
                        alpha_sliders[pos] = real_slider
                except Exception:
                    pass
            real_include = entry.get('include')
            if real_include is not None:
                pos = target.get('pos')
                try:
                    if pos is not None and pos < len(include_checkboxes):
                        include_checkboxes[pos] = real_include
                except Exception:
                    pass

        for tgt in alpha_targets:
            try:
                _ensure_real_slider(tgt)
            except Exception:
                pass

        optimizable_targets = []
        for tgt in alpha_targets:
            locked = False
            try:
                pos = tgt.get("pos")
                if pos is not None and pos < len(lock_alpha_checkboxes):
                    locked = bool(getattr(lock_alpha_checkboxes[pos], "value", False))
            except Exception:
                locked = False
            if not locked:
                optimizable_targets.append(tgt)

        num_targets = len(optimizable_targets)
        if num_targets == 0:
            try:
                status_html.value = (
                    "<span style='color:#a00;'>Cannot iterate: all peaks within range are locked.</span>"
                )
            except Exception:
                pass
            try:
                iterating_in_progress = False
                cancel_fit_btn_frozen = False
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_hidden)
            except Exception:
                pass
            try:
                _on_main_thread(_show_controls_after_fit_or_opt)
            except Exception:
                try:
                    _show_controls_after_fit_or_opt()
                except Exception:
                    pass
            return

        current_rc = base_rc
        if not np.isfinite(current_rc):
            try:
                status_html.value = "<span style='color:#555;'>Iteratively correcting (pre-fit)...</span>"
            except Exception:
                pass
            current_rc = _run_fit_and_wait(fit_ranges_snapshot)
        if not np.isfinite(current_rc):
            base_rc = current_rc
            try:
                status_html.value = (
                    "<span style='color:#a00;'>Unable to start optimization: fit did not produce a finite reduced chi-square.</span>"
                )
            except Exception:
                pass
            try:
                iterating_in_progress = False
                cancel_fit_btn_frozen = False
            except Exception:
                pass
            try:
                _on_main_thread(_force_cancel_fit_hidden)
            except Exception:
                pass
            try:
                _on_main_thread(_show_controls_after_fit_or_opt)
            except Exception:
                try:
                    _show_controls_after_fit_or_opt()
                except Exception:
                    pass
            return

        improvement_threshold = 1e-6
        step_sequence = [0.1, 0.05]
        max_sweeps = 12
        total_evaluations = 0
        changed_peak_indices = set()

        def _update_iter_status(msg):
            try:
                status_html.value = msg
            except Exception:
                _log_once(html.unescape(msg) if isinstance(msg, str) else str(msg))

        def _clamp_alpha(val):
            try:
                return max(0.0, min(1.0, float(val)))
            except Exception:
                return val

        def _try_directional_alpha(slider, current_val, step, current_rc_local):
            """Try +/-step once to choose direction, then keep stepping until no improvement."""
            nonlocal total_evaluations
            if slider is None or not hasattr(slider, "value"):
                return current_val, current_rc_local, False

            best_val = current_val
            best_rc = current_rc_local
            chosen_dir = 0.0

            # First decision: compare +step vs -step
            for direction in (step, -step):
                cand = _clamp_alpha(current_val + direction)
                if abs(cand - current_val) < 1e-9:
                    continue
                _set_quiet(slider, "value", cand)
                rc = _run_fit_and_wait(fit_ranges_snapshot)
                total_evaluations += 1
                if not np.isfinite(rc):
                    rc = np.inf
                if rc + improvement_threshold < best_rc:
                    best_rc = rc
                    best_val = cand
                    chosen_dir = direction

            # No improvement at this step size
            if chosen_dir == 0.0 or abs(best_val - current_val) < 1e-12:
                _set_quiet(slider, "value", current_val)
                return current_val, current_rc_local, False

            # Accept best direction and keep stepping while improving
            current_val = best_val
            current_rc_local = best_rc
            _set_quiet(slider, "value", current_val)

            while True:
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                cand = _clamp_alpha(current_val + chosen_dir)
                if abs(cand - current_val) < 1e-9:
                    break
                _set_quiet(slider, "value", cand)
                rc = _run_fit_and_wait(fit_ranges_snapshot)
                total_evaluations += 1
                if not np.isfinite(rc):
                    rc = np.inf
                if rc + improvement_threshold < current_rc_local:
                    current_val = cand
                    current_rc_local = rc
                    continue
                # Revert and stop when no improvement
                _set_quiet(slider, "value", current_val)
                break

            return current_val, current_rc_local, True

        sweep = 0
        while True:
            try:
                if cancel_event.is_set():
                    break
            except Exception:
                pass
            sweep += 1
            sweep_changed = False
            for idx_target, target in enumerate(optimizable_targets, start=1):
                try:
                    if cancel_event.is_set():
                        break
                except Exception:
                    pass
                slider = target.get("slider")
                if slider is None or not hasattr(slider, "value"):
                    continue
                try:
                    current_val = float(slider.value)
                except Exception:
                    current_val = 0.5
                original_val = current_val
                peak_idx = target.get("peak_idx")

                for step in step_sequence:
                    try:
                        if cancel_event.is_set():
                            break
                    except Exception:
                        pass
                    new_val, new_rc, improved = _try_directional_alpha(
                        slider, current_val, step, current_rc
                    )
                    if improved and (new_rc + improvement_threshold < current_rc):
                        current_val = new_val
                        current_rc = new_rc
                        sweep_changed = True
                        idx_key = peak_idx if peak_idx is not None else target.get("pos")
                        if idx_key is not None:
                            changed_peak_indices.add(idx_key)
                        try:
                            _update_iter_status(
                                f"<span style='color:#555;'>iterating... sweep {sweep} | peak {idx_target}/{num_targets} | evaluations={total_evaluations} | α={current_val:.3f} | redχ={current_rc:.4g}</span>"
                            )
                        except Exception:
                            pass
                        continue
                    # No improvement at this step size; try next smaller step
                    _set_quiet(slider, "value", current_val)
                _set_quiet(slider, "value", current_val)
                if cancel_event.is_set():
                    break

            if cancel_event.is_set():
                break
            if not sweep_changed:
                break
            if sweep >= max_sweeps:
                _log_once("Iteration sweep limit reached; stopping to avoid infinite loop.")
                break

        changed_peaks = len(changed_peak_indices)
        iteration_changes = changed_peaks
        base_rc = current_rc
        try:
            _snapshot_current_controls()
        except Exception:
            pass

        # Final status: show old -> new comparison
        try:
            _snapshot_current_controls()
        except Exception:
            pass
        try:
            old_str = f"{start_rc:.4g}"
        except Exception:
            old_str = str(start_rc)
        try:
            new_str = f"{base_rc:.4g}"
        except Exception:
            new_str = str(base_rc)
        try:
            if cancel_event.is_set():
                prefix = "Iterative correction cancelled."
                if iteration_timed_out and iteration_timeout_message:
                    prefix = iteration_timeout_message.strip()
                status_html.value = (
                    f"<span style='color:#a00;'>{prefix} Reduced chi-square: "
                    f"({old_str}) ---&gt; ({new_str}) | peaks adjusted: {changed_peaks}/{num_targets} | evaluations: {total_evaluations}</span>"
                )
            else:
                status_html.value = (
                    f"<span style='color:#000;'>Iterative correction complete. "
                    f"Reduced chi-square: ({old_str}) ---&gt; ({new_str}) | peaks adjusted: {changed_peaks}/{num_targets} | evaluations: {total_evaluations}</span>"
                )
        except Exception:
            if cancel_event.is_set():
                prefix = "Iterative correction cancelled."
                if iteration_timed_out and iteration_timeout_message:
                    prefix = iteration_timeout_message.strip()
                _log_once(
                    f"{prefix} Reduced chi-square: ({old_str}) -> ({new_str}) | peaks adjusted: {changed_peaks}/{num_targets} | evaluations: {total_evaluations}"
                )
            else:
                _log_once(
                    f"Iterative correction complete. Reduced chi-square: ("
                    f"{old_str}) -> ({new_str}) | peaks adjusted: {changed_peaks}/{num_targets} | evaluations: {total_evaluations}"
                )
        # Persist iteration summary for session-level reporting
        try:
            idx_snapshot = spectrum_sel.value
            if idx_snapshot is not None:
                def _normalize_rc(val):
                    try:
                        if isinstance(val, (int, float)) and math.isfinite(val):
                            return float(val)
                    except Exception:
                        pass
                    try:
                        return float(val)
                    except Exception:
                        return val

                _deconv_changes["iter"].append(
                    (
                        idx_snapshot,
                        _normalize_rc(start_rc),
                        _normalize_rc(base_rc),
                        int(changed_peaks),
                    )
                )
        except Exception:
            pass
        # Remember iteration summary for the next refit message
        iter_final_redchi = base_rc
        iter_summary_pending = True

        # Allow normal fit status updates again and unfreeze button visibility.
        # Perform the hide only once here to prevent mid-loop flicker.
        iterating_in_progress = False
        cancel_fit_btn_frozen = False
        try:
            _on_main_thread(_force_cancel_fit_hidden)
        except Exception:
            pass

        # One final visible refresh (silent evaluations do not update the plot)
        try:
            _fit_and_update_plot(
                ignore_debounce=True,
                override_ranges=fit_ranges_snapshot,
                update_plot=True,
                update_controls=True,
            )
        except Exception:
            pass

    # Wire the iterative correct to run in background and support cancellation
    def _on_iteratively_correct_click(b):
        if _recent_click("iteratively_correct"):
            return
        nonlocal iter_thread, iterating_in_progress, cancel_event, iter_start_redchi, cancel_fit_btn_frozen
        try:
            if iter_thread is not None and iter_thread.is_alive():
                try:
                    status_html.value = (
                        f"<span style='color:#a60;'>Iterative correction"
                        f" is already running.</span>"
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Snapshot current redchi for this spectrum at the moment of click
        try:
            idx_snapshot = spectrum_sel.value
            iter_start_redchi = last_redchi_by_idx.get(idx_snapshot, np.inf)
        except Exception:
            iter_start_redchi = np.inf
        # Reset cancel flag and mark iteration as active
        try:
            cancel_event.clear()
        except Exception:
            pass
        iterating_in_progress = True
        cancel_fit_btn_frozen = True
        # Hide the same controls as during Fit while optimization runs
        try:
            _hide_controls_during_fit_or_opt()
        except Exception:
            pass
        try:
            _set_cancel_button_mode("optimize")
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#555;'>Starting iterative correction...</span>"
            )
        except Exception:
            pass
        try:
            _force_cancel_fit_shown()
        except Exception:
            pass
        iter_thread = threading.Thread(target=_iteratively_correct_worker, daemon=True)
        iter_thread.start()
        try:
            _force_cancel_fit_shown()
        except Exception:
            pass
        try:
            _on_main_thread(_force_cancel_fit_shown)
        except Exception:
            try:
                _update_cancel_fit_visibility()
            except Exception:
                pass

    iter_btn.on_click(_on_iteratively_correct_click)

    def _on_cancel_fit(b):
        if _recent_click("cancel_fit"):
            return
        # Signal cancellation for any active fit/iteration
        try:
            cancel_event.set()
        except Exception:
            pass
        try:
            if fit_cancel_token is not None:
                fit_cancel_token.set()
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#a00;'>Cancellation requested...</span>"
            )
        except Exception:
            pass
        # Keep visibility consistent; it will auto-hide when work stops
        try:
            _update_cancel_fit_visibility()
        except Exception:
            pass

    cancel_fit_btn.on_click(_on_cancel_fit)

    def _on_mark_bad_click(_b=None):
        try:
            idx = spectrum_sel.value
            qcol = _quality_column_name(FTIR_DataFrame)
            FTIR_DataFrame.at[idx, qcol] = "bad"
            try:
                status_html.value = (
                    f"<span style='color:#a00;'>Marked row {idx} as bad quality.</span>"
                )
            except Exception:
                pass
            # Rebuild spectrum options to drop the bad row
            try:
                _rebuild_spectrum_options()
            except Exception:
                pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass

    # Defer wiring mark_bad_btn until after it's instantiated below

    # Layout
    controls_row_filters = widgets.HBox(
        [material_dd, conditions_dd, include_bad_cb],
        layout=widgets.Layout(align_items="center"),
    )
    controls_row_spectrum = widgets.HBox(
        [spectrum_sel, prev_spectrum_btn, next_spectrum_btn, spectrum_counter],
        layout=widgets.Layout(align_items="center"),
    )
    # We'll place the Fit X-ranges section AFTER the primary action rows
    fit_range_row = widgets.VBox([
        widgets.HTML("<b>Fit X-ranges</b>"),
        range_sliders_box,
        widgets.HBox([add_range_btn]),
    ])
    # Keep other global parameters grouped below the peak controls
    reset_all_row = widgets.HBox([reset_all_btn])
    # Mark buttons (use shared helper for mutually exclusive controls)
    mark_bad_btn, mark_good_btn, _refresh_mark_buttons = _make_quality_controls(
        FTIR_DataFrame,
        lambda: (
            FTIR_DataFrame.loc[current_idx_deconv]
            if current_idx_deconv is not None
            else None
        ),
        margin="0 8px 0 0",
    )

    # Add additional status and list refresh after helper toggles quality
    def _post_mark_update_deconv(status: str):
        nonlocal current_idx_deconv
        try:
            idx = current_idx_deconv
            if status == "bad":
                try:
                    status_html.value = f"<span style='color:#a00;'>Marked row {idx} as bad quality.</span>"
                except Exception:
                    pass
            else:
                try:
                    status_html.value = f"<span style='color:#0a0;'>Marked row {idx} as good quality.</span>"
                except Exception:
                    pass
            # Removal/reinsertion logic delegated to centralized helper to avoid flicker
            if idx is not None:
                try:
                    if status == "bad":
                        current_idx_deconv = idx
                    _quality_dropdown_handle(
                        status,
                        dropdown=spectrum_sel,
                        include_bad_flag=include_bad_cb.value,
                        idx=idx,
                        label_builder=lambda i: f"{FTIR_DataFrame.loc[i].get('Material','?')} | {FTIR_DataFrame.loc[i].get('Conditions', FTIR_DataFrame.loc[i].get('Condition','?'))} | {FTIR_DataFrame.loc[i].get('File Name','?')}",
                        observer_fn=_on_spectrum_change,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _refresh_mark_buttons()
        except Exception:
            pass
        try:
            _deconv_changes["quality"].append((current_idx_deconv, status))
        except Exception:
            pass

    mark_bad_btn.on_click(lambda _b=None: _post_mark_update_deconv("bad"))
    mark_good_btn.on_click(lambda _b=None: _post_mark_update_deconv("good"))
    mark_row = widgets.HBox([mark_bad_btn, mark_good_btn])
    # Hide mark buttons until a spectrum is selected; will show on selection
    try:
        mark_row.layout.display = "none"
    except Exception:
        pass

    # Refresh function provided by helper

    # Three action rows before Fit X-ranges
    primary_actions_row = widgets.HBox(
        [fit_btn, save_btn, close_btn],
        layout=widgets.Layout(
            flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%"
        ),
    )
    canonize_btn = widgets.Button(
        description="Canonize peaks for material",
        button_style="info",
        tooltip="Save current spectrum's peak centers as canonical for this material",
        layout=widgets.Layout(width="260px"),
    )
    canonize_confirm_btn = widgets.Button(
        description="Confirm canonization",
        button_style="warning",
        tooltip="Confirm saving the current peak centers as canonical for this material",
        layout=widgets.Layout(width="200px"),
    )
    canonize_cancel_btn = widgets.Button(
        description="Cancel canonization",
        button_style="",
        tooltip="Cancel and return without changing materials.json",
        layout=widgets.Layout(width="190px"),
    )
    try:
        canonize_confirm_btn.layout.display = "none"
        canonize_cancel_btn.layout.display = "none"
    except Exception:
        pass
    load_canon_btn = widgets.Button(
        description="Load canon peaks",
        button_style="info",
        tooltip="Load canonical peak centers from materials.json for this material",
        layout=widgets.Layout(width="190px"),
    )
    edit_actions_row = widgets.HBox(
        [
            add_peaks_btn,
            delete_peaks_btn,
            canonize_btn,
            canonize_confirm_btn,
            canonize_cancel_btn,
            load_canon_btn,
        ],
        layout=widgets.Layout(
            flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%"
        ),
    )
    optimize_actions_row = widgets.HBox(
        [iter_btn, reset_all_btn],
        layout=widgets.Layout(
            flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%"
        ),
    )
    # Alias: existing code toggles reset_all_row; make it refer to the optimize row
    reset_all_row = optimize_actions_row
    # Additional rows that are shown contextually
    add_peaks_actions_row = widgets.HBox(
        [accept_new_peaks_btn, redo_new_peaks_btn, cancel_new_peaks_btn],
        layout=widgets.Layout(
            flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%"
        ),
    )
    delete_actions_row = widgets.HBox(
        [accept_deletions_btn, redo_deletions_btn, cancel_deletions_btn],
        layout=widgets.Layout(
            flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%"
        ),
    )
    # Hide contextual rows initially
    try:
        add_peaks_actions_row.layout.display = "none"
        delete_actions_row.layout.display = "none"
    except Exception:
        pass
    status_row = widgets.HBox([status_html])
    cancel_row = widgets.HBox([cancel_fit_btn])
    try:
        cancel_row.layout.display = "none"
    except Exception:
        pass
    # Colab fallback slider + typed input + Add button for peak addition
    add_peaks_slider = widgets.FloatSlider(
        value=float((xmin + xmax) / 2.0),
        min=float(xmin),
        max=float(xmax),
        step=(float(xmax) - float(xmin)) / 1000.0 or 1.0,
        description="Select (cm⁻¹)",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="70%"),
    )
    add_peaks_text = widgets.FloatText(
        value=float((xmin + xmax) / 2.0),
        description="Type (cm⁻¹)",
        layout=widgets.Layout(width="160px"),
    )
    add_peaks_add_btn = widgets.Button(
        description="Add", button_style="info", layout=widgets.Layout(width="80px")
    )

    # Keep text and slider in sync (slider drives text)
    def _sync_text_from_slider(change):
        if change.get("name") != "value":
            return
        try:
            add_peaks_text.value = float(change.get("new"))
        except Exception:
            pass

    add_peaks_slider.observe(_sync_text_from_slider, names="value")
    # Colab delete-by-number: enter peak index (1-based) to delete
    delete_peak_num_text = widgets.IntText(
        value=1,
        description="Peak #",
        layout=widgets.Layout(width="140px"),
    )
    delete_peak_num_btn = widgets.Button(
        description="Delete",
        button_style="danger",
        layout=widgets.Layout(width="100px"),
    )
    colab_delete_row = widgets.HBox([delete_peak_num_text, delete_peak_num_btn])
    colab_delete_help = widgets.HTML(
        "<span style='color:#555;font-size:12px;'>Colab: Enter the peak number (as shown in the list) to delete.</span>"
    )
    colab_add_row = widgets.HBox([add_peaks_slider, add_peaks_text, add_peaks_add_btn])
    colab_add_help = widgets.HTML(
        "<span style='color:#555;font-size:12px;'>Colab: Use the slider or type a wavenumber, then click Add. Peaks snap to nearest data point; Accept to commit.</span>"
    )
    # Hide row if not in Colab or not in add-peaks mode
    if not _IN_COLAB:
        colab_add_row.layout.display = "none"
        colab_add_help.layout.display = "none"
        colab_delete_row.layout.display = "none"
        colab_delete_help.layout.display = "none"
    # Place plot + mark row in a bordered container; keep mark below the plot
    plot_and_mark_deconv = widgets.VBox(
        [fig, mark_row],
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", margin="6px 0"),
    )
    ui = widgets.VBox(
        [
            # 1) Material and Conditions
            controls_row_filters,
            # 2) Spectrum dropdown + Include bad checkbox
            controls_row_spectrum,
            # 3) Plot container (with mark buttons below plot)
            plot_and_mark_deconv,
            # Status (reduced chi-square and operation updates) directly under plot
            status_row,
            # Cancel button row (visible only during long-running fits)
            cancel_row,
            # 4) Primary actions rows (before Fit X-ranges)
            primary_actions_row,
            edit_actions_row,
            optimize_actions_row,
            # Contextual action rows
            add_peaks_actions_row,
            delete_actions_row,
            #    Colab add controls (only shown in Colab when adding)
            colab_add_row,
            colab_add_help,
            #    Colab delete controls (only shown in Colab when deleting)
            colab_delete_row,
            colab_delete_help,
            # 6) X range selector bar
            fit_range_row,
            # 7) List of peaks and per-peak controls
            peak_controls_section,
        ]
    )

    # --- Canonize/Load canonical peaks handlers ---
    def _materials_json_path():
        try:
            base_dir_js = os.path.dirname(__file__)
        except Exception:
            base_dir_js = os.getcwd()
        return os.path.join(base_dir_js, "materials.json")

    def _lookup_material_code(top: dict, mat_name: str):
        try:
            m = str(mat_name)
        except Exception:
            m = mat_name
        if not isinstance(top, dict):
            return None
        for k, payload in top.items():
            try:
                if not isinstance(payload, dict):
                    continue
                alias = str(payload.get("alias", ""))
                namev = str(payload.get("name", ""))
                if alias == m or namev == m:
                    return k
            except Exception:
                continue
        return None

    def _current_material_name():
        try:
            mval = material_dd.value
            if mval and str(mval).lower() != "any":
                return str(mval)
        except Exception:
            pass
        # fallback to spectrum row value
        try:
            idx = spectrum_sel.value
            return str(FTIR_DataFrame.loc[idx].get("Material", ""))
        except Exception:
            return ""

    def _canonize_peaks_for_material(_b=None):
        # Get current peaks centers
        try:
            idx = spectrum_sel.value
        except Exception:
            idx = None
        if idx is None:
            _append_status("No spectrum selected to canonize peaks from.")
            return
        try:
            xs, ys = _get_peaks(idx)
        except Exception:
            xs, ys = ([], [])
        centers = [float(x) for x in (xs or [])]
        if not centers:
            _append_status("No peaks available to canonize.")
            return
        # Sort by wavenumber to maintain order
        centers = sorted(centers)
        mat_name = _current_material_name()
        if not mat_name:
            _append_status("Material not determined; cannot update materials.json.")
            return
        # Load JSON
        try:
            mpath = _materials_json_path()
            with open(mpath, "r", encoding="utf-8") as jf:
                content = json.load(jf)
        except Exception as e:
            _append_status(f"Failed to read materials.json: {e}")
            return
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            _append_status("materials.json structure unexpected; aborting.")
            return
        top = content[0]
        code_key = _lookup_material_code(top, mat_name)
        if code_key is None:
            _append_status(f"Material '{mat_name}' not found in materials.json.")
            return
        mat_payload = dict(top.get(code_key, {}))
        peaks_payload = dict(mat_payload.get("peaks", {})) if isinstance(mat_payload.get("peaks"), dict) else {}
        # Build new mapping while preserving existing per-peak σ/α when present
        new_len = len(centers)
        new_map = {}
        for i, c in enumerate(centers, start=1):
            key = str(i)
            prev = peaks_payload.get(key, {}) if isinstance(peaks_payload, dict) else {}
            try:
                sigma_prev = float(prev.get("σ", 0.0))
            except Exception:
                sigma_prev = 0.0
            try:
                alpha_prev = float(prev.get("α", 0.0))
            except Exception:
                alpha_prev = 0.0
            new_map[key] = {
                "name": str(prev.get("name", "")),
                "center_wavenumber": float(c),
                "σ": sigma_prev,
                "α": alpha_prev,
            }
        # Replace peaks, effectively adding/removing to match current set
        mat_payload["peaks"] = new_map
        top[code_key] = mat_payload
        try:
            with open(mpath, "w", encoding="utf-8") as jf:
                json.dump(content, jf, indent=4, ensure_ascii=False)
            _append_status(f"Canonized {new_len} peak(s) for material '{mat_name}'.")
        except Exception as e:
            _append_status(f"Failed to write materials.json: {e}")

        # Restore canonize UI after a canonization attempt
        try:
            _show(canonize_btn)
            _hide(canonize_confirm_btn)
            _hide(canonize_cancel_btn)
        except Exception:
            pass

    def _load_canon_peaks(_b=None):
        mat_name = _current_material_name()
        if not mat_name:
            _append_status("Material not determined; cannot load canonical peaks.")
            return
        try:
            mpath = _materials_json_path()
            with open(mpath, "r", encoding="utf-8") as jf:
                content = json.load(jf)
        except Exception as e:
            _append_status(f"Failed to read materials.json: {e}")
            return
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            _append_status("materials.json structure unexpected; aborting.")
            return
        top = content[0]
        code_key = _lookup_material_code(top, mat_name)
        if code_key is None:
            _append_status(f"Material '{mat_name}' not found in materials.json.")
            return
        peaks_dict = {}
        try:
            peaks_dict = top.get(code_key, {}).get("peaks", {}) or {}
        except Exception:
            peaks_dict = {}
        if not isinstance(peaks_dict, dict) or not peaks_dict:
            _append_status(f"No canonical peaks stored for '{mat_name}'.")
            return
        # Build centers list ordered by numeric key
        try:
            ordered_items = sorted(((int(k), v) for k, v in peaks_dict.items()), key=lambda t: t[0])
            centers = [float(v.get("center_wavenumber", 0.0)) for _k, v in ordered_items]
        except Exception:
            centers = []
        if not centers:
            _append_status(f"No valid canonical peak centers for '{mat_name}'.")
            return
        # Apply to current spectrum: set Peak Wavenumbers and infer amplitudes from data
        try:
            idx = spectrum_sel.value
            x_arr, y_arr = _get_xy(idx)
        except Exception:
            idx, x_arr, y_arr = None, None, None
        if idx is None or x_arr is None or y_arr is None or getattr(x_arr, "size", 0) == 0:
            _append_status("Cannot apply canonical peaks: current spectrum data unavailable.")
            return
        amps = []
        for c in centers:
            try:
                i = int(np.argmin(np.abs(x_arr - float(c))))
                amps.append(float(y_arr[i]))
            except Exception:
                amps.append(0.0)
        # Persist into DataFrame
        try:
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = list(centers)
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = list(amps)
            # Optional column to mark usage
            if "Using Canon Peaks" not in FTIR_DataFrame.columns:
                FTIR_DataFrame["Using Canon Peaks"] = None
            FTIR_DataFrame.at[idx, "Using Canon Peaks"] = "True"
        except Exception:
            pass
        # Clear caches to reflect new list
        try:
            if isinstance(peak_box_cache, dict):
                peak_box_cache.clear()
        except Exception:
            pass
        try:
            included_index_map.clear()
        except Exception:
            pass
        try:
            last_result_by_idx.pop(idx, None)
        except Exception:
            pass
        try:
            last_redchi_by_idx.pop(idx, None)
        except Exception:
            pass
        try:
            _refresh_peak_control_widgets(idx)
        except Exception:
            pass
        _append_status(f"Loaded {len(centers)} canonical peak(s) for '{mat_name}'.")

    def _begin_canonize_confirmation(_b=None):
        # Replace the canonize button with Confirm/Cancel
        try:
            _hide(canonize_btn)
            _show(canonize_confirm_btn)
            _show(canonize_cancel_btn)
        except Exception:
            pass

    def _cancel_canonize_confirmation(_b=None):
        try:
            _show(canonize_btn)
            _hide(canonize_confirm_btn)
            _hide(canonize_cancel_btn)
        except Exception:
            pass

    canonize_btn.on_click(_begin_canonize_confirmation)
    canonize_confirm_btn.on_click(_canonize_peaks_for_material)
    canonize_cancel_btn.on_click(_cancel_canonize_confirmation)
    load_canon_btn.on_click(_load_canon_peaks)

    # --- Add-peaks workflow callbacks ---
    def _enter_add_mode(b=None):
        if _recent_click("enter_add_mode"):
            return
        nonlocal adding_mode
        adding_mode = True
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _hide(add_peaks_btn)
        # Ensure plot click is wired for add-mode selections (desktop)
        try:
            if not _IN_COLAB:
                fig.data[0].on_click(_on_data_click)
        except Exception:
            pass
        # Hide Delete peaks if present
        try:
            delete_peaks_btn.layout.display = "none"
        except Exception:
            pass
        # Hide canon buttons during add-peaks mode
        try:
            _hide(canonize_btn)
        except Exception:
            pass
        try:
            _hide(load_canon_btn)
        except Exception:
            pass
        # Hide non-relevant controls in Peak Addition mode
        _hide(fit_btn)
        _hide(reset_all_row)
        _hide(fit_range_row)
        # Show add-peaks contextual row
        try:
            add_peaks_actions_row.layout.display = ""
        except Exception:
            pass
        _show(accept_new_peaks_btn)
        _show(redo_new_peaks_btn)
        _show(cancel_new_peaks_btn)
        # Hide parameter modifiers during add-peaks mode
        _hide(iter_btn)
        try:
            cancel_fit_btn.disabled = True
        except Exception:
            pass
        _hide_cancel_button()
        _hide(save_btn)
        _hide(close_btn)
        _hide(peak_controls_box)
        # Disable selection widgets to avoid changing spectrum/filters mid-selection
        try:
            spectrum_sel.disabled = True
            material_dd.disabled = True
            conditions_dd.disabled = True
        except Exception:
            pass
        if _IN_COLAB:
            # Show slider row/help and set bounds from current spectrum
            try:
                x_arr, _y_arr = _get_xy(spectrum_sel.value)
                if x_arr is not None and x_arr.size > 0:
                    x_min = float(np.nanmin(x_arr))
                    x_max = float(np.nanmax(x_arr))
                    add_peaks_slider.min = x_min
                    add_peaks_slider.max = x_max
                    add_peaks_slider.step = (x_max - x_min) / 1000.0 or 1.0
                    mid = float((x_min + x_max) / 2.0)
                    add_peaks_slider.value = mid
                    try:
                        add_peaks_text.value = mid
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                colab_add_row.layout.display = ""
                colab_add_help.layout.display = ""
            except Exception:
                pass
            _log_once(
                "Add-peaks mode (Colab): use the slider then click Add; Accept/Redo/Cancel when done."
            )
        else:
            _log_once(
                "Add-peaks mode: click x-locations on the plot; then Accept/Redo/Cancel."
            )

    # --- Delete-peaks workflow ---
    deleting_mode = False
    delete_markers_drawn = False

    def _clear_delete_peak_shapes():
        try:
            shapes = list(getattr(fig.layout, "shapes", ()))
            shapes = [s for s in shapes if getattr(s, "name", None) != "delete_peak_marker"]
            fig.layout.shapes = tuple(shapes)
        except Exception:
            # dict fallback
            try:
                shapes = list(getattr(fig.layout, "shapes", ()))
                new_shapes = []
                for s in shapes:
                    try:
                        if s.get("name") != "delete_peak_marker":
                            new_shapes.append(s)
                    except Exception:
                        new_shapes.append(s)
                fig.layout.shapes = tuple(new_shapes)
            except Exception:
                pass

    def _draw_current_peaks_for_delete(idx):
        nonlocal delete_markers_drawn
        _clear_delete_peak_shapes()
        xs, _ys = _get_visible_peaks(idx)
        x_arr, y_arr = _get_xy(idx)
        if not xs or x_arr is None or y_arr is None or x_arr.size == 0:
            delete_markers_drawn = False
            return
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        for cx in xs:
            try:
                fig.add_shape(
                    dict(
                        type="line",
                        x0=float(cx),
                        x1=float(cx),
                        y0=y_min,
                        y1=y_max,
                        line=dict(color="#d62728", dash="dash", width=1.5),
                        name="delete_peak_marker",
                    )
                )
            except Exception:
                pass
        delete_markers_drawn = True

    # Staging for deletions and snapshot
    staged_delete_indices = []
    _delete_snapshot = {}

    def _enter_delete_mode(b=None):
        if _recent_click("enter_delete_mode"):
            return
        nonlocal deleting_mode
        deleting_mode = True
        staged_delete_indices.clear()
        # Bind delete-mode click handler (desktop)
        try:
            if not _IN_COLAB:
                fig.data[0].on_click(_on_delete_click)
        except Exception:
            pass
        # Hide the Delete Peaks button itself while in delete mode
        try:
            delete_peaks_btn.layout.display = "none"
        except Exception:
            pass
        # Hide canon buttons during delete-peaks mode
        try:
            _hide(canonize_btn)
        except Exception:
            pass
        try:
            _hide(load_canon_btn)
        except Exception:
            pass
        # Snapshot current state
        try:
            idx = spectrum_sel.value
            _delete_snapshot = {}
        except Exception:
            _delete_snapshot = {}
        _hide(add_peaks_btn)
        try:
            # Hide Fit and unrelated controls
            _hide(fit_btn)
            _hide(reset_all_row)
            _hide(fit_range_row)
            # Show Delete button hidden state is handled by creating it below
            # Hide accept/redo/cancel for add-peaks
            _hide(accept_new_peaks_btn)
            _hide(redo_new_peaks_btn)
            _hide(cancel_new_peaks_btn)
            # Show delete-action row
            try:
                delete_actions_row.layout.display = ""
            except Exception:
                pass
            # Hide add-peaks action row if visible
            try:
                add_peaks_actions_row.layout.display = "none"
            except Exception:
                pass
            _hide(iter_btn)
            cancel_fit_btn.disabled = True
            _hide_cancel_button()
            _hide(save_btn)
            _hide(close_btn)
            _hide(peak_controls_box)
        except Exception:
            pass
        try:
            spectrum_sel.disabled = True
            material_dd.disabled = True
            conditions_dd.disabled = True
        except Exception:
            pass
        idx = spectrum_sel.value
        _draw_current_peaks_for_delete(idx)
        if _IN_COLAB:
            try:
                colab_delete_row.layout.display = ""
                colab_delete_help.layout.display = ""
            except Exception:
                pass
            _log_once("Delete-peaks mode (Colab): enter peak number and click Delete.")
        else:
            _log_once("Delete-peaks mode: click existing peak markers to delete.")

    def _exit_delete_mode():
        nonlocal deleting_mode
        deleting_mode = False
        _clear_delete_peak_shapes()
        # Remove delete-mode click handler
        try:
            if not _IN_COLAB:
                fig.data[0].on_click(None)
        except Exception:
            pass
        _show(add_peaks_btn)
        # Restore canon buttons when leaving delete mode
        try:
            _show(canonize_btn)
        except Exception:
            pass
        try:
            _show(load_canon_btn)
        except Exception:
            pass
        # Restore the Delete Peaks button visibility when leaving delete mode
        try:
            delete_peaks_btn.layout.display = ""
        except Exception:
            pass
        _show(fit_btn)
        _show(reset_all_row)
        _show(fit_range_row)
        # Hide delete action row
        try:
            delete_actions_row.layout.display = "none"
        except Exception:
            pass
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        if _IN_COLAB:
            try:
                colab_delete_row.layout.display = "none"
                colab_delete_help.layout.display = "none"
            except Exception:
                pass
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass

    # Click handler to stage deletion of nearest visible peak when in delete mode
    def _on_delete_click(trace, points, selector):
        try:
            nonlocal deleting_mode
            if not deleting_mode:
                return
            if not points or not getattr(points, "xs", None):
                return
            x_clicked = float(points.xs[0])
        except Exception:
            return
        idx = spectrum_sel.value
        try:
            vis_xs, _ = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        if not vis_xs:
            _log_once("No peaks to delete.")
            return
        # Find nearest peak center
        try:
            nearest_i = int(np.argmin(np.abs(np.asarray(vis_xs, dtype=float) - x_clicked)))
        except Exception:
            nearest_i = None
        if nearest_i is None:
            return
        # Stage deletion index
        try:
            if nearest_i not in staged_delete_indices:
                staged_delete_indices.append(nearest_i)
                _append_status(
                    f"Staged deletion: Peak {nearest_i + 1} at {vis_xs[nearest_i]:.3f} cm⁻¹."
                )
            else:
                _append_status(
                    f"Peak {nearest_i + 1} already staged for deletion."
                )
        except Exception:
            pass
        _draw_current_peaks_for_delete(idx)

    try:
        if not _IN_COLAB:
            fig.data[0].on_click(_on_delete_click)
    except Exception:
        pass

    def _colab_delete_peak(_b=None):
        idx = spectrum_sel.value
        try:
            peak_num = int(delete_peak_num_text.value)
        except Exception:
            return
        if peak_num <= 0:
            return
        # Convert to 0-based index within visible peaks ordering
        try:
            vis_xs, _ = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        if not vis_xs:
            return
        target_i = peak_num - 1
        if target_i < 0 or target_i >= len(vis_xs):
            return
        # Stage deletion by number
        try:
            if target_i not in staged_delete_indices:
                staged_delete_indices.append(target_i)
                _append_status(
                    f"Staged deletion: Peak {target_i + 1} at {vis_xs[target_i]:.3f} cm⁻¹."
                )
        except Exception:
            pass
        _draw_current_peaks_for_delete(idx)
        # Wait for Accept before applying

    delete_peak_num_btn.on_click(_colab_delete_peak)

    # Bind Delete Peaks button
    try:
        delete_peaks_btn.on_click(_enter_delete_mode)
    except Exception:
        pass

    # Wire Accept/Redo/Cancel deletions using placeholders indices 3,4,5
    def _apply_staged_deletions():
        try:
            idx = spectrum_sel.value
            xs, ys = _get_peaks(idx)
            xs = list(xs or [])
            ys = list(ys or [])
            for del_i in sorted(staged_delete_indices, reverse=True):
                try:
                    if 0 <= del_i < len(xs):
                        xs.pop(del_i)
                except Exception:
                    pass
                try:
                    if 0 <= del_i < len(ys):
                        ys.pop(del_i)
                except Exception:
                    pass
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = xs
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = ys
        except Exception:
            pass

    def _accept_deletions(_b=None):
        if _recent_click("accept_deletions"):
            return
        try:
            idx = spectrum_sel.value
            vis_xs, _ = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        _apply_staged_deletions()
        # Record in session changes for summary
        try:
            deleted_centers = [vis_xs[i] for i in staged_delete_indices if i < len(vis_xs)]
            if 'deleted' not in _deconv_changes:
                _deconv_changes['deleted'] = []
            _deconv_changes['deleted'].append((idx, [float(c) for c in deleted_centers]))
        except Exception:
            pass
        # Clear caches so main mode respects new peak list
        try:
            if isinstance(peak_box_cache, dict):
                peak_box_cache.clear()
        except Exception:
            pass
        try:
            included_index_map.clear()
        except Exception:
            pass
        try:
            last_result_by_idx.pop(idx, None)
        except Exception:
            pass
        try:
            last_redchi_by_idx.pop(idx, None)
        except Exception:
            pass
        # No staged peak additions are retained.
        # Rebuild peak-center state mapping to align with new list length
        try:
            new_centers, _new_ys = _get_peaks(idx)
            peak_center_state_by_idx[idx] = {
                i: {'initial': float(c), 'user': float(c)} for i, c in enumerate(new_centers)
            }
        except Exception:
            try:
                new_centers, _new_ys = _get_peaks(idx)
                peak_center_state_by_idx[idx] = {
                    i: {'initial': c, 'user': c} for i, c in enumerate(new_centers)
                }
            except Exception:
                pass
        _append_status("Deletions accepted and applied.")
        try:
            _refresh_peak_control_widgets(spectrum_sel.value)
        except Exception:
            pass
        try:
            _update_fit_range_indicator()
        except Exception:
            pass
        staged_delete_indices.clear()
        _exit_delete_mode()

    def _redo_deletions(_b=None):
        if _recent_click("redo_deletions"):
            return
        staged_delete_indices.clear()
        try:
            idx = spectrum_sel.value
            xs_restore = list(_delete_snapshot.get('centers', []))
            ys_restore = list(_delete_snapshot.get('amps', []))
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = xs_restore
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = ys_restore
        except Exception:
            pass
        _draw_current_peaks_for_delete(spectrum_sel.value)
        _append_status("Deletions cleared; restored original peaks.")

    def _cancel_deletions(_b=None):
        if _recent_click("cancel_deletions"):
            return
        try:
            idx = spectrum_sel.value
            xs_restore = list(_delete_snapshot.get('centers', []))
            ys_restore = list(_delete_snapshot.get('amps', []))
            FTIR_DataFrame.at[idx, "Peak Wavenumbers"] = xs_restore
            FTIR_DataFrame.at[idx, "Peak Absorbances"] = ys_restore
        except Exception:
            pass
        staged_delete_indices.clear()
        _append_status("Deletions cancelled. No changes applied.")
        _exit_delete_mode()

    try:
        accept_deletions_btn.on_click(_accept_deletions)
        redo_deletions_btn.on_click(_redo_deletions)
        cancel_deletions_btn.on_click(_cancel_deletions)
        # Ensure hidden initially
        delete_actions_row.layout.display = "none"
    except Exception:
        pass

    def _accept_new_peaks(b=None):
        if _recent_click("accept_new_peaks"):
            return
        nonlocal adding_mode
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None:
            _log_once("Cannot accept: current spectrum has no normalized data.")
            return
        if len(new_peak_xs) == 0:
            # Nothing selected; just exit mode
            _cancel_new_peaks()
            return
        # Merge with existing peaks and sort; reject new peaks too close to existing
        xs_existing, ys_existing = _get_peaks(idx)
        xs_list = [float(v) for v in (xs_existing or [])]
        ys_list = [float(v) for v in (ys_existing or [])]
        old_sorted_centers = list(xs_list)
        # Collect newly accepted (non-overlapping) peaks before integrating
        valid_new_peaks = []

        # Use per-peak windows from visible peaks for overlap checks
        rejected_close = []
        try:
            peaks_x, _ = _get_visible_peaks(idx)
        except Exception:
            peaks_x = []
        for x_new in new_peak_xs:
            # Reject if new point falls inside any existing peak's per-peak window
            overlapped = False
            try:
                for i, cx in enumerate(peaks_x or []):
                    try:
                        w_i = float(center_window_sliders[i].value)
                    except Exception:
                        w_i = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
                    # Enforce minimum separation of 2× the per-peak Center ± window
                    if abs(float(cx) - float(x_new)) <= (2.0 * abs(w_i)):
                        overlapped = True
                        break
            except Exception:
                overlapped = False
            if overlapped:
                rejected_close.append(float(x_new))
                continue
            # find nearest y
            try:
                i = int(np.argmin(np.abs(x_arr - x_new)))
                y_new = float(y_arr[i])
            except Exception:
                # If we cannot compute a nearest point, skip this new peak
                continue
            # Stash valid new peak (we'll integrate after the loop)
            try:
                valid_new_peaks.append((float(x_new), float(y_new)))
            except Exception:
                pass
        # Integrate new peaks: build combined sorted list and update DataFrame.
        # Note: we intentionally do NOT preserve parameter choices from the previous
        # spectrum or from the pre-add state; after accepting, the spectrum derives
        # from FTIR_DataFrame + defaults.
        try:
            for (c_new, a_new) in valid_new_peaks:
                xs_list.append(float(c_new))
                ys_list.append(float(a_new))
            combined = sorted(zip(xs_list, ys_list), key=lambda t: float(t[0]))
            new_centers = [float(c) for c, _a in combined] if combined else []
            new_amplitudes = [float(_a) for _c, _a in combined] if combined else []
            try:
                peak_center_state_by_idx[idx] = {
                    i: {'initial': float(c), 'user': float(c)}
                    for i, c in enumerate(new_centers)
                }
            except Exception:
                peak_center_state_by_idx[idx] = {
                    i: {'initial': c, 'user': c} for i, c in enumerate(new_centers)
                }
            # Persist the accepted peak set directly to the DataFrame so the spectrum
            # always derives its state from FTIR_DataFrame (no cross-spectrum carry-over).
            try:
                # Use .at for scalar cell assignment; .loc with list-like values
                # can be interpreted as iterable assignment and fail.
                FTIR_DataFrame.at[idx, 'Peak Wavenumbers'] = list(new_centers)
                FTIR_DataFrame.at[idx, 'Peak Absorbances'] = list(new_amplitudes)
            except Exception:
                pass
            try:
                if isinstance(peak_box_cache, dict):
                    peak_box_cache.clear()
            except Exception:
                pass
            try:
                included_index_map.clear()
            except Exception:
                pass
            try:
                last_result_by_idx.pop(idx, None)
            except Exception:
                pass
            try:
                last_redchi_by_idx.pop(idx, None)
            except Exception:
                pass
            # No shared peaks template; rebuild reflects new list
        except Exception as _e_add:
            _log_once(f"Failed to integrate new peaks: {_e_add}")
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        # Restore previously hidden/disabled controls after exiting add-peaks mode
        _show(fit_btn)
        _show(reset_all_row)
        _show(fit_range_row)
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        # Restore canon buttons when leaving add-peaks mode
        try:
            _show(canonize_btn)
        except Exception:
            pass
        try:
            _show(load_canon_btn)
        except Exception:
            pass
        # Exit add mode and restore Add peaks button
        try:
            adding_mode = False
            _show(add_peaks_btn)
            # Unbind add-mode click handler when add mode ends
            try:
                if not _IN_COLAB:
                    fig.data[0].on_click(None)
            except Exception:
                pass
            # Ensure Delete Peaks button reappears after ending add mode
            try:
                delete_peaks_btn.layout.display = ""
            except Exception:
                pass
            # Hide add-peaks actions row after exit
            try:
                add_peaks_actions_row.layout.display = "none"
            except Exception:
                pass
            _clear_add_peak_shapes()
            new_peak_xs.clear()
            new_peak_windows.clear()
        except Exception:
            pass
        # Rebuild UI to reflect newly added peaks
        try:
            _refresh_peak_control_widgets(idx)
        except Exception:
            pass
        # Hide Colab slider row when exiting add-peaks mode via Accept
        if _IN_COLAB:
            try:
                colab_add_row.layout.display = "none"
                colab_add_help.layout.display = "none"
            except Exception:
                pass
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        # Status: show any rejections in red
        if rejected_close:
            joined = ", ".join(f"{v:.3f}" for v in rejected_close)
            msg = (
                "Rejected: {} are too close to existing peaks (must be ≥ 2× Center ± window away). "
                "Tip: adjust per-peak Center ± sliders to make room."
            ).format(joined)
            try:
                status_html.value = f"<span style='color:#a00;'>{msg}</span>"
            except Exception:
                _log_once(msg)
        else:
            # Build informative message listing the newly staged peak centers.
            try:
                staged_centers = [p[0] for p in valid_new_peaks] if 'valid_new_peaks' in locals() else []
            except Exception:
                staged_centers = []
            if staged_centers:
                locs = ", ".join(f"{c:.3f}" for c in staged_centers)
                _log_once(
                    f"New peaks accepted and staged at: {locs}. (Unsaved – run Fit then Save to commit.)"
                )
            else:
                _log_once(
                    "New peaks staged (unsaved). Run Fit then click Save to commit."
                )

    def _redo_new_peaks(b=None):
        if _recent_click("redo_new_peaks"):
            return
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        if _IN_COLAB:
            _log_once("Selection cleared. Use slider to select peaks again, then Add.")
        else:
            _log_once("Selection cleared. Click on the plot to select peaks again.")

    def _cancel_new_peaks(b=None):
        if _recent_click("cancel_new_peaks"):
            return
        nonlocal adding_mode
        adding_mode = False
        new_peak_xs.clear()
        _clear_add_peak_shapes()
        _show(add_peaks_btn)
        # Unbind add-mode click handler when add mode is cancelled
        try:
            if not _IN_COLAB:
                fig.data[0].on_click(None)
        except Exception:
            pass
        # Ensure Delete Peaks button reappears when add mode is cancelled
        try:
            delete_peaks_btn.layout.display = ""
        except Exception:
            pass
        _hide(accept_new_peaks_btn)
        _hide(redo_new_peaks_btn)
        _hide(cancel_new_peaks_btn)
        try:
            add_peaks_actions_row.layout.display = "none"
        except Exception:
            pass
        # Restore controls hidden during Peak Addition mode
        _show(fit_btn)
        _show(reset_all_row)
        _show(fit_range_row)
        # Restore previously hidden/disabled controls after cancelling add-peaks mode
        _show(iter_btn)
        _update_cancel_fit_visibility()
        _show(save_btn)
        _show(close_btn)
        _show(peak_controls_box)
        # Restore canon buttons when leaving add-peaks mode
        try:
            _show(canonize_btn)
        except Exception:
            pass
        try:
            _show(load_canon_btn)
        except Exception:
            pass
        if _IN_COLAB:
            try:
                colab_add_row.layout.display = "none"
                colab_add_help.layout.display = "none"
            except Exception:
                pass
        try:
            spectrum_sel.disabled = False
            material_dd.disabled = False
            conditions_dd.disabled = False
        except Exception:
            pass
        _log_once("Peak addition cancelled. No changes were made.")

    add_peaks_btn.on_click(_enter_add_mode)

    # Colab slider-driven addition
    def _colab_add_peak(_b=None):
        try:
            # Prefer typed value when provided; fall back to slider
            x_target = float(add_peaks_text.value)
        except Exception:
            try:
                x_target = float(add_peaks_slider.value)
            except Exception:
                return
        idx = spectrum_sel.value
        x_arr, y_arr = _get_xy(idx)
        if x_arr is None or y_arr is None or x_arr.size == 0:
            _log_once("Cannot add peak: spectrum has no normalized data.")
            return
        # Snap to nearest existing x
        try:
            nearest_i = int(np.argmin(np.abs(x_arr - x_target)))
            x_new = float(x_arr[nearest_i])
        except Exception:
            _log_once("Could not determine nearest x for selected location.")
            return
        # Reject if within any existing visible peak's per-peak Center ± window
        try:
            vis_xs, _vis_ys = _get_visible_peaks(idx)
        except Exception:
            vis_xs = []
        try:
            default_win = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            default_win = 0.0
        try:
            for i, cx in enumerate(vis_xs or []):
                try:
                    w_i = float(center_window_sliders[i].value)
                except Exception:
                    w_i = default_win
                # Enforce minimum separation of 2× the per-peak Center ± window
                if abs(float(cx) - float(x_new)) <= (2.0 * abs(w_i)):
                    _log_once(
                        f"Rejected: {x_new:.3f} cm⁻¹ is too close to existing peak @ {cx:.3f}. "
                        f"Must be ≥ 2×(Center ± window) = {2.0*abs(w_i):.1f} cm⁻¹ away. "
                        f"Adjust selection or that peak’s Center ± window."
                    )
                    return
        except Exception:
            pass
        # Check against session-selected peaks using 2× window rule
        try:
            candidate_w = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        except Exception:
            candidate_w = 0.0
        for existing_x in new_peak_xs:
            try:
                existing_w = float(new_peak_windows.get(existing_x, candidate_w))
            except Exception:
                existing_w = candidate_w
            required_sep = 2.0 * max(abs(existing_w), abs(candidate_w))
            if abs(existing_x - x_new) <= required_sep:
                _log_once(
                    f"Rejected: {x_new:.3f} cm⁻¹ is too close to selected peak {existing_x:.3f}. "
                    f"Must be ≥ {required_sep:.2f} cm⁻¹ away (2× window)."
                )
                return
        # Compute session-effective window vs existing peaks and store
        try:
            xs_existing, _ys_existing = _get_peaks(idx)
        except Exception:
            xs_existing = []
        try:
            if xs_existing:
                min_dist = min(abs(float(xe) - x_new) for xe in xs_existing)
            else:
                min_dist = float('inf')
        except Exception:
            min_dist = float('inf')
        try:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
            if np.isfinite(min_dist):
                eff_window = min(eff_window, max(0.0, float(min_dist) - float(SESSION_WINDOW_MARGIN)))
        except Exception:
            eff_window = float(PER_PEAK_DEFAULT_CENTER_WINDOW)
        try:
            new_peak_windows[x_new] = float(eff_window)
        except Exception:
            pass
        # Accept
        new_peak_xs.append(x_new)
        try:
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
        except Exception:
            y_min, y_max = 0.0, 1.0
        try:
            fig.add_shape(
                dict(
                    type="line",
                    x0=x_new,
                    x1=x_new,
                    y0=y_min,
                    y1=y_max,
                    line=dict(color="#ff7f0e", dash="dot", width=1.5),
                    name="add_peak_marker",
                )
            )
        except Exception:
            pass
        _log_once(f"Selected new peak at x = {x_new:.3f} cm⁻¹. Add more or Accept.")

    try:
        if _IN_COLAB:
            add_peaks_add_btn.on_click(_colab_add_peak)
    except Exception:
        pass
    accept_new_peaks_btn.on_click(_accept_new_peaks)
    redo_new_peaks_btn.on_click(_redo_new_peaks)
    cancel_new_peaks_btn.on_click(_cancel_new_peaks)

    # Prime the peak controls so the list renders immediately on first display
    try:
        if spectrum_sel.value is not None:
            current_idx_deconv = spectrum_sel.value
            _refresh_peak_control_widgets(current_idx_deconv)
            _update_fit_range_indicator()
            _fix_y_range(current_idx_deconv)
    except Exception:
        pass

    # Only display the UI (which already contains the figure) and the log
    display(ui, log_html)
    # Seed options with current filters (defaults 'any') and trigger initial updates
    _rebuild_spectrum_options()

    return FTIR_DataFrame


def fit_material(FTIR_DataFrame):

    """
    Fit deconvoluted spectra of selected material to a unified set of peaks.

    Parameters:
    ----------
    FTIR_DataFrame : pd.DataFrame
        DataFrame containing FTIR spectral data and prior analysis results.

    Returns:
    -------
    pd.DataFrame
        Updated DataFrame with fit-results stored.
    """
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    if FTIR_DataFrame is None or len(FTIR_DataFrame) == 0:
        raise ValueError("FTIR_DataFrame must be loaded and non-empty.")

    required_cols = [
        "Material",
        "Time",
        "Deconvolution Results",
        "Deconvolution X-Ranges",
        "X-Axis",
        "Normalized and Corrected Data",
        "Material Fit Results",
    ]
    missing = [c for c in required_cols if c not in FTIR_DataFrame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Ensure your DataFrame is prepared with prior steps."
        )

    try:
        FTIR_DataFrame["Material Fit Results"] = FTIR_DataFrame[
            "Material Fit Results"
        ].astype(object)
    except Exception:
        pass

    # --- Fit across all spectra of selected material ---
    # Important: this tool uses ONLY peaks from 'Deconvolution Results' and
    # uses ONLY the union of 'Deconvolution X-Ranges' for fitting/optimization.
    import json, ast, os, numpy as np

    # Shared (per-session) canonical parameters sourced from materials.json or computed
    shared_centers_list = None  # list[float]
    shared_sigma_list = None    # list[float]
    shared_alpha_list = None    # list[float]
    shared_peak_numbers_list = None  # list[int|None] (source: Deconvolution Results)

    # Active union of X-Ranges for the current selection (material + include_bad)
    current_union_x_ranges = []  # list[tuple[float, float]]

    # Row indices actually used in the most recent Fit/Optimize (i.e., had usable
    # Deconvolution Results that were incorporated into the canonical peak set).
    last_used_fit_row_indices = set()

    # materials.json helpers
    def _load_materials_json():
        try:
            base_dir_js = os.path.dirname(__file__)
            path = os.path.join(base_dir_js, "materials.json")
            with open(path, "r", encoding="utf-8") as jf:
                content = json.load(jf)
            return path, content
        except Exception:
            return None, None

    def _lookup_material_code(top_dict, mat_name: str):
        try:
            for k, payload in top_dict.items():
                if not isinstance(payload, dict):
                    continue
                if (
                    str(payload.get("alias", "")).strip() == str(mat_name)
                    or str(payload.get("name", "")).strip() == str(mat_name)
                ):
                    return k
        except Exception:
            pass
        return None

    def _canon_from_json(mat_name: str):
        """Return (centers, sigmas, alphas) from materials.json for a material.
        Falls back to empty lists when unavailable.
        """
        path, content = _load_materials_json()
        if content is None or not isinstance(content, list) or not content:
            return [], [], []
        top = content[0] if isinstance(content[0], dict) else {}
        code = _lookup_material_code(top, mat_name)
        if code is None:
            return [], [], []
        payload = top.get(code, {})
        try:
            peaks = payload.get("peaks", {}) or {}
        except Exception:
            peaks = {}
        centers, sigmas, alphas = [], [], []
        try:
            # Sort by numeric key order "1","2",...
            for idx_key in sorted(peaks.keys(), key=lambda s: int(str(s)) if str(s).isdigit() else str(s)):
                p = peaks.get(idx_key, {}) or {}
                try:
                    centers.append(float(p.get("center_wavenumber", 0.0)))
                except Exception:
                    centers.append(0.0)
                try:
                    sigmas.append(float(p.get("σ", 0.0)))
                except Exception:
                    sigmas.append(0.0)
                try:
                    alphas.append(float(p.get("α", 0.0)))
                except Exception:
                    alphas.append(0.0)
        except Exception:
            pass
        return centers, sigmas, alphas

    # Shared parameters are computed on-demand in _compute_for_selection / _optimize_centers_for_selection
    # from the DataFrame's 'Deconvolution Results' (no dependency on materials.json for peak selection).

    # -------------------------- Backend helpers --------------------------- #
    def _parse_deconv(val):
        """Parse a Deconvolution Results cell to a list[dict] or None."""
        if val is None:
            return None
        if isinstance(val, str):
            try:
                v = ast.literal_eval(val)
            except Exception:
                return None
        else:
            v = val
        if isinstance(v, list):
            # Ensure items are dicts and sort by center if present
            try:
                items = [dict(d) for d in v if isinstance(d, dict)]
            except Exception:
                return None
            try:
                items = sorted(
                    items, key=lambda d: float(d.get("center", float("nan")))
                )
            except Exception:
                # If centers not coercible, keep original order
                pass
            return items
        return None

    def _parse_x_ranges(val):
        """Parse a Deconvolution X-Ranges cell to list[(lo, hi)] in ascending order."""
        if val is None:
            return []
        v = val
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except Exception:
                return []
        if not isinstance(v, (list, tuple)):
            return []
        out = []
        for item in v:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                a = float(item[0])
                b = float(item[1])
            except Exception:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            lo = min(a, b)
            hi = max(a, b)
            # ignore degenerate intervals
            if hi <= lo:
                continue
            out.append((lo, hi))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def _union_ranges(ranges):
        """Merge overlapping/adjacent (lo, hi) pairs."""
        if not ranges:
            return []
        rs = sorted([(float(lo), float(hi)) for lo, hi in ranges if np.isfinite(lo) and np.isfinite(hi)], key=lambda t: (t[0], t[1]))
        if not rs:
            return []
        merged = [list(rs[0])]
        for lo, hi in rs[1:]:
            cur = merged[-1]
            # merge if overlapping or touching
            if lo <= cur[1]:
                cur[1] = max(cur[1], hi)
            else:
                merged.append([lo, hi])
        return [(float(a), float(b)) for a, b in merged]

    def _collect_union_x_ranges(df_series):
        all_rs = []
        try:
            for _idx, _row in df_series.iterrows():
                all_rs.extend(_parse_x_ranges(_row.get("Deconvolution X-Ranges")))
        except Exception:
            pass
        return _union_ranges(all_rs)

    def _slice_to_union_ranges(x_arr, y_arr, union_ranges):
        if not union_ranges:
            return x_arr, y_arr
        try:
            x_np = np.asarray(x_arr, dtype=float)
            y_np = np.asarray(y_arr, dtype=float)
            n = min(x_np.size, y_np.size)
            if n <= 1:
                return x_np[:0], y_np[:0]
            x_np = x_np[:n]
            y_np = y_np[:n]
        except Exception:
            return None, None
        mask = np.zeros(x_np.shape, dtype=bool)
        for lo, hi in union_ranges:
            try:
                mask |= (x_np >= lo) & (x_np <= hi)
            except Exception:
                pass
        if not np.any(mask):
            return x_np[:0], y_np[:0]
        return x_np[mask], y_np[mask]

    def _parse_xy(row):
        """Return (x_arr, y_arr) as 1D float arrays or (None, None)."""
        x = row.get("X-Axis")
        y = row.get("Normalized and Corrected Data")
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return None, None
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                return None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
                return None, None
        except Exception:
            return None, None
        return x_arr, y_arr

    def _mode_peak_count(lists_of_peaks):
        """Return the most common positive length among lists; tie -> max length."""
        lengths = [len(p) for p in lists_of_peaks if isinstance(p, list) and len(p) > 0]
        if not lengths:
            return 0
        vals, counts = np.unique(lengths, return_counts=True)
        # Choose value with max count; if tie, the larger length wins
        max_count = np.max(counts)
        candidates = [v for v, c in zip(vals, counts) if c == max_count]
        return int(max(candidates))

    def _parse_peak_number(v):
        """Coerce a peak_number field to int when possible."""
        if v is None:
            return None
        try:
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, float) and np.isfinite(v):
                return int(v)
        except Exception:
            pass
        try:
            s = str(v).strip()
            if not s:
                return None
            # Accept "18", "Peak 18", "peak_18" etc
            m = re.search(r"(\d+)", s)
            if not m:
                return None
            return int(m.group(1))
        except Exception:
            return None

    def _compute_shared_peak_numbers_from_aligned(aligned_by_idx, k):
        """Return list length k of most-common peak_number per canonical peak."""
        nums_out = []
        for i in range(int(k)):
            candidates = []
            try:
                for _idx, aligned in (aligned_by_idx or {}).items():
                    if not isinstance(aligned, list) or i >= len(aligned):
                        continue
                    p = aligned[i]
                    if not isinstance(p, dict):
                        continue
                    pn = _parse_peak_number(p.get("peak_number"))
                    if pn is not None:
                        candidates.append(int(pn))
            except Exception:
                candidates = []
            if not candidates:
                nums_out.append(None)
                continue
            try:
                vals, counts = np.unique(np.asarray(candidates, dtype=int), return_counts=True)
                # Most frequent; tie -> smallest (stable)
                max_count = int(np.max(counts))
                best_vals = [int(v) for v, c in zip(vals.tolist(), counts.tolist()) if int(c) == max_count]
                nums_out.append(int(min(best_vals)) if best_vals else None)
            except Exception:
                try:
                    nums_out.append(int(candidates[0]))
                except Exception:
                    nums_out.append(None)
        return nums_out

    def _peak_labels_from_numbers(peak_numbers, k):
        """Return unique column labels (length k) like 'Peak 18'."""
        labels = []
        seen = {}
        for i in range(int(k)):
            pn = None
            try:
                if isinstance(peak_numbers, list) and i < len(peak_numbers):
                    pn = peak_numbers[i]
            except Exception:
                pn = None
            base = f"Peak {int(pn)}" if pn is not None else f"Peak {i+1}"
            n = seen.get(base, 0) + 1
            seen[base] = n
            labels.append(base if n == 1 else f"{base} ({n})")
        return labels

    def _is_unexposed(val):
        try:
            return str(val).strip().lower() == "unexposed"
        except Exception:
            return False

    # Compute only for the currently selected Material
    def _compute_for_selection(material, include_bad=False):
        # Build the series subset for the selected material
        try:
            series_df = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(material)].copy()
        except Exception:
            series_df = FTIR_DataFrame.copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                series_df = series_df[_quality_good_mask(series_df)]
            except Exception:
                pass
        if series_df.empty:
            print("No spectra found for the selected material.")
            return

        # Compute the union of all saved X-Ranges for this selection
        nonlocal current_union_x_ranges
        try:
            current_union_x_ranges = _collect_union_x_ranges(series_df)
        except Exception:
            current_union_x_ranges = []

        # Collect deconvolution peak lists for this series
        peak_lists = []
        peak_lists_by_idx = {}
        for idx, row in series_df.iterrows():
            peaks = _parse_deconv(row.get("Deconvolution Results"))
            if peaks is not None and len(peaks) > 0:
                peak_lists.append(peaks)
                peak_lists_by_idx[idx] = peaks

        # Build canonical centers by clustering all peak centers across spectra
        sigma_samples = []
        all_centers = []
        for peaks in peak_lists:
            for p in peaks:
                try:
                    sval = float(p.get("sigma", p.get("σ", np.nan)))
                    if np.isfinite(sval):
                        sigma_samples.append(sval)
                except Exception:
                    pass
                try:
                    cval = float(p.get("center", p.get("center_wavenumber", np.nan)))
                    if np.isfinite(cval):
                        all_centers.append(cval)
                except Exception:
                    pass
        try:
            tol = float(np.nanmedian(sigma_samples)) if np.isfinite(np.nanmedian(sigma_samples)) else 8.0
        except Exception:
            tol = 8.0
        try:
            centers_sorted = sorted(set([float(c) for c in all_centers if np.isfinite(c)]))
        except Exception:
            centers_sorted = []
        canonical_centers = []
        for c in centers_sorted:
            if not canonical_centers:
                canonical_centers.append(c)
                continue
            if abs(c - canonical_centers[-1]) <= tol:
                canonical_centers[-1] = (canonical_centers[-1] + c) / 2.0
            else:
                canonical_centers.append(c)
        k = len(canonical_centers)
        if k <= 0:
            print("Could not establish canonical peaks for the series.")
            return

        # Align each spectrum to the canonical centers; add missing peaks with amplitude 0
        aligned_by_idx = {}
        for idx, peaks in peak_lists_by_idx.items():
            aligned = [None] * k
            used = set()
            for i, ccan in enumerate(canonical_centers):
                best_j = None
                best_d = float("inf")
                for j, p in enumerate(peaks):
                    if j in used:
                        continue
                    try:
                        cval = float(p.get("center", p.get("center_wavenumber", np.nan)))
                    except Exception:
                        cval = float("nan")
                    if not np.isfinite(cval):
                        continue
                    d = abs(cval - ccan)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is not None and best_d <= tol:
                    aligned[i] = dict(peaks[best_j])
                    used.add(best_j)
                else:
                    aligned[i] = {
                        "amplitude": 0.0,
                        "A": 0.0,
                        "center": float(ccan),
                        "alpha": 0.5,
                        "fraction": 0.5,
                        "α": 0.5,
                        "sigma": float(np.nanmedian(sigma_samples)) if np.isfinite(np.nanmedian(sigma_samples)) else 10.0,
                        "σ": float(np.nanmedian(sigma_samples)) if np.isfinite(np.nanmedian(sigma_samples)) else 10.0,
                    }
            aligned_by_idx[idx] = aligned

        # Track which rows were actually used (had usable deconvolution results)
        nonlocal last_used_fit_row_indices
        try:
            last_used_fit_row_indices = set(aligned_by_idx.keys())
        except Exception:
            last_used_fit_row_indices = set()

        # Compute averages over aligned peaks
        centers, sigmas, alphas = [], [], []
        for idx, aligned in aligned_by_idx.items():
            try:
                centers.append([float(p.get("center", p.get("center_wavenumber", np.nan))) for p in aligned])
                sigmas.append([float(p.get("sigma", p.get("σ", np.nan))) for p in aligned])
                alphas.append([float(p.get("alpha", p.get("fraction", p.get("α", np.nan)))) for p in aligned])
            except Exception:
                continue
        centers = np.asarray(centers, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        alphas = np.asarray(alphas, dtype=float)
        with np.errstate(all="ignore"):
            avg_center = np.nanmean(centers, axis=0)
            avg_sigma = np.nanmean(sigmas, axis=0)
            avg_alpha = np.nanmean(alphas, axis=0)
        for i in range(k):
            if not np.isfinite(avg_center[i]):
                vals = centers[:, i]
                avg_center[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
            if not np.isfinite(avg_sigma[i]):
                vals = sigmas[:, i]
                avg_sigma[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 10.0
            if not np.isfinite(avg_alpha[i]):
                vals = alphas[:, i]
                avg_alpha[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.5

        # Update shared parameters for plotting and downstream fitting
        nonlocal shared_centers_list, shared_sigma_list, shared_alpha_list, shared_peak_numbers_list
        try:
            shared_centers_list = [float(v) for v in avg_center.tolist()]
        except Exception:
            shared_centers_list = [float(c) for c in canonical_centers]
        try:
            shared_sigma_list = [float(v) if np.isfinite(v) else 10.0 for v in avg_sigma.tolist()]
        except Exception:
            shared_sigma_list = [10.0] * k
        try:
            shared_alpha_list = [float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 0.5 for v in avg_alpha.tolist()]
        except Exception:
            shared_alpha_list = [0.5] * k
        try:
            shared_peak_numbers_list = _compute_shared_peak_numbers_from_aligned(aligned_by_idx, k)
        except Exception:
            shared_peak_numbers_list = [None] * k

        # Using averaged parameters, refit amplitudes per spectrum with lmfit.
        assigned = 0
        refit_logs = []  # collect per-spectrum status messages
        for idx, aligned in aligned_by_idx.items():
            try:
                x_arr, y_arr = _parse_xy(series_df.loc[idx])
            except Exception:
                x_arr, y_arr = None, None
            if (
                x_arr is None
                or y_arr is None
                or np.asarray(x_arr).size == 0
                or np.asarray(y_arr).size == 0
            ):
                # Fall back to preserving amplitudes when data unavailable
                try:
                    refit_logs.append(f"Row {idx}: no data available; preserved amplitudes.")
                except Exception:
                    pass
                # Persist amplitude-only results
                out_amps = []
                for i in range(k):
                    try:
                        out_amps.append(float(aligned[i].get("amplitude", 0.0)))
                    except Exception:
                        out_amps.append(0.0)
                try:
                    FTIR_DataFrame.at[idx, "Material Fit Results"] = out_amps
                    assigned += 1
                except Exception:
                    pass
                continue

            # Build composite model with fixed center/sigma/fraction and variable amplitudes
            comp_model = None
            params = None
            for i in range(k):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                p = m.make_params()
                # Use canonical centers and aggregated sigma/alpha
                p[f"p{i}_center"].set(value=float(shared_centers_list[i]), vary=False)
                p[f"p{i}_sigma"].set(value=float(shared_sigma_list[i]), min=1e-3, max=1e4, vary=False)
                p[f"p{i}_fraction"].set(value=float(shared_alpha_list[i]), min=0.0, max=1.0, vary=False)
                # Seed amplitude from aligned peak or a heuristic
                try:
                    amp0 = float(aligned[i].get("amplitude", aligned[i].get("A", 1.0)))
                    if not np.isfinite(amp0):
                        raise ValueError()
                except Exception:
                    # heuristic: local y at center times sigma scale
                    try:
                        ci = float(avg_center[i])
                        nearest = int(np.argmin(np.abs(np.asarray(x_arr, dtype=float) - ci)))
                        amp0 = max(0.0, float(np.asarray(y_arr, dtype=float)[nearest]) * max(1.0, float(avg_sigma[i])))
                    except Exception:
                        amp0 = 1.0
                p[f"p{i}_amplitude"].set(value=max(0.0, amp0), min=0.0)
                if comp_model is None:
                    comp_model = m
                    params = p
                else:
                    comp_model = comp_model + m
                    params.update(p)

            # Sanitize input arrays: drop non-finite points and ensure matching lengths
            try:
                x_np = np.asarray(x_arr, dtype=float)
                y_np = np.asarray(y_arr, dtype=float)
                n = min(x_np.size, y_np.size)
                if n <= 1:
                    raise ValueError("Insufficient data points for fitting")
                x_np = x_np[:n]
                y_np = y_np[:n]
                mask = np.isfinite(x_np) & np.isfinite(y_np)
                if not np.any(mask):
                    raise ValueError("No finite data points for fitting")
                x_np = x_np[mask]
                y_np = y_np[mask]

                # Restrict fit to the UNION of the saved Deconvolution X-Ranges
                try:
                    x_np, y_np = _slice_to_union_ranges(x_np, y_np, current_union_x_ranges)
                except Exception:
                    pass
                if x_np is None or y_np is None or x_np.size <= 1 or y_np.size <= 1:
                    raise ValueError("No data points in selected Deconvolution X-Ranges")
                # Log heavy sanitization
                try:
                    removed = int(n - x_np.size)
                except Exception:
                    pass
                # Guard: centers outside x-range -> fix amplitude to 0 and do not vary
                try:
                    x_min_loc = float(np.nanmin(x_np))
                    x_max_loc = float(np.nanmax(x_np))
                except Exception:
                    x_min_loc, x_max_loc = None, None
                if x_min_loc is not None and x_max_loc is not None and np.isfinite(x_min_loc) and np.isfinite(x_max_loc):
                    for i in range(k):
                        try:
                            ci = float(avg_center[i])
                            if not (x_min_loc <= ci <= x_max_loc):
                                # freeze amplitude to 0 for out-of-range peak
                                params[f"p{i}_amplitude"].set(value=0.0, min=0.0, vary=False)
                                refit_logs.append(
                                    f"Row {idx}: center {ci:.6g} outside x-range [{x_min_loc:.6g}, {x_max_loc:.6g}] for Peak {i+1}; amplitude fixed to 0."
                                )
                        except Exception:
                            pass
                # Optional scale guard: clip extreme magnitudes to reduce numeric issues
                y_range = float(np.nanmax(y_np) - np.nanmin(y_np)) if y_np.size else 0.0
                if np.isfinite(y_range) and y_range > 0:
                    y_np = np.clip(y_np, np.nanmin(y_np) - 10*y_range, np.nanmax(y_np) + 10*y_range)
                res = comp_model.fit(y_np, params, x=x_np)
                fit_list = []
                for i in range(k):
                    try:
                        amp = float(res.params.get(f"p{i}_amplitude").value)
                        if not np.isfinite(amp):
                            amp = 0.0
                    except Exception:
                        amp = 0.0
                    fit_list.append(amp)
                FTIR_DataFrame.at[idx, "Material Fit Results"] = fit_list
                assigned += 1
            except Exception as _fit_err:
                # If fit fails, preserve amplitudes
                try:
                    refit_logs.append(f"Row {idx}: fit failed ({_fit_err}); preserved amplitudes.")
                except Exception:
                    pass
                out_amps = []
                for i in range(k):
                    try:
                        out_amps.append(float(aligned[i].get("amplitude", aligned[i].get("A", 0.0))))
                    except Exception:
                        out_amps.append(0.0)
                try:
                    FTIR_DataFrame.at[idx, "Material Fit Results"] = out_amps
                    assigned += 1
                except Exception:
                    pass

        print(f"Amplitudes refit for Material={material} using canon centers. Updated {assigned} spectra.")
        # Surface logs in the UI if available
        try:
            if refit_logs:
                log_html = "<br/>".join([widgets.HTML.escape(str(m)) if hasattr(widgets.HTML, 'escape') else str(m) for m in refit_logs])
                optimize_status_html.value = (
                    "<div style='margin-top:6px'><b>Refit Status</b></div>"
                    + f"<div style='color:#555'>{log_html}</div>"
                )
        except Exception:
            pass

    def _optimize_centers_for_selection(material, include_bad=False):
        """Iteratively optimize shared centers and per-spectrum amplitudes to reduce SSE.

        Approach:
        - Coordinate descent on shared centers. For each peak, try +/- step shifts.
        - For every center trial, re-fit amplitudes per spectrum with centers/σ/α fixed
          against the 'Normalized and Corrected Data' to get the best amplitudes.
        - Accept moves that lower the total SSE; shrink step when no progress; stop
          after a few passes or when below tolerance.

        Returns a status message and the optimized centers list (or None on failure).
        """
        # Build series subset (Material matches; Conditions match or are 'unexposed')
        try:
            series_df = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(material)].copy()
        except Exception:
            series_df = FTIR_DataFrame.copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                series_df = series_df[_quality_good_mask(series_df)]
            except Exception:
                pass
        if series_df.empty:
            return "No spectra found for the selected material.", None

        # Compute union of X-Ranges for this selection (used throughout optimization)
        nonlocal current_union_x_ranges
        try:
            current_union_x_ranges = _collect_union_x_ranges(series_df)
        except Exception:
            current_union_x_ranges = []

        # Seed shared parameters by aligning deconvolution peaks to canonical centers
        peak_lists = []
        peak_lists_by_idx = {}
        for _idx, _row in series_df.iterrows():
            pk = _parse_deconv(_row.get("Deconvolution Results"))
            if pk is not None and len(pk) > 0:
                peak_lists.append(pk)
                peak_lists_by_idx[_idx] = pk

        if not peak_lists:
            return "Selected series has no usable deconvolution results.", None

        sigma_samples = []
        all_centers = []
        for _pk in peak_lists:
            for p in _pk:
                try:
                    sval = float(p.get("sigma", p.get("σ", np.nan)))
                    if np.isfinite(sval):
                        sigma_samples.append(sval)
                except Exception:
                    pass
                try:
                    cval = float(p.get("center", p.get("center_wavenumber", np.nan)))
                    if np.isfinite(cval):
                        all_centers.append(cval)
                except Exception:
                    pass
        try:
            tol = float(np.nanmedian(sigma_samples)) if np.isfinite(np.nanmedian(sigma_samples)) else 8.0
        except Exception:
            tol = 8.0

        try:
            centers_sorted = sorted(set([float(c) for c in all_centers if np.isfinite(c)]))
        except Exception:
            centers_sorted = []

        canonical_centers = []
        for c in centers_sorted:
            if not canonical_centers:
                canonical_centers.append(c)
                continue
            if abs(c - canonical_centers[-1]) <= tol:
                canonical_centers[-1] = (canonical_centers[-1] + c) / 2.0
            else:
                canonical_centers.append(c)

        k = len(canonical_centers)
        if k <= 0:
            return "Could not establish canonical peaks for the series.", None

        # Align each spectrum to the canonical centers; add missing peaks with amplitude 0
        aligned_by_idx = {}
        try:
            sigma_default = float(np.nanmedian(sigma_samples)) if np.isfinite(np.nanmedian(sigma_samples)) else 10.0
        except Exception:
            sigma_default = 10.0
        for _idx, peaks in peak_lists_by_idx.items():
            aligned = [None] * k
            used = set()
            for i, ccan in enumerate(canonical_centers):
                best_j = None
                best_d = float("inf")
                for j, p in enumerate(peaks):
                    if j in used:
                        continue
                    try:
                        cval = float(p.get("center", p.get("center_wavenumber", np.nan)))
                    except Exception:
                        cval = float("nan")
                    if not np.isfinite(cval):
                        continue
                    d = abs(cval - ccan)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is not None and best_d <= tol:
                    aligned[i] = dict(peaks[best_j])
                    used.add(best_j)
                else:
                    aligned[i] = {
                        "amplitude": 0.0,
                        "center": float(ccan),
                        "alpha": 0.5,
                        "fraction": 0.5,
                        "sigma": float(sigma_default),
                    }
            aligned_by_idx[_idx] = aligned

        # Track which rows are actually used (usable deconvolution results)
        nonlocal last_used_fit_row_indices
        try:
            last_used_fit_row_indices = set(aligned_by_idx.keys())
        except Exception:
            last_used_fit_row_indices = set()

        # Aggregate initial shared params (centers/sigma/alpha)
        centers_mat, sigmas_mat, fracs_mat = [], [], []
        for _idx, aligned in aligned_by_idx.items():
            try:
                centers_mat.append([float(p.get("center", np.nan)) for p in aligned])
                sigmas_mat.append([float(p.get("sigma", p.get("σ", np.nan))) for p in aligned])
                fracs_mat.append([float(p.get("fraction", p.get("alpha", np.nan))) for p in aligned])
            except Exception:
                continue
        if not centers_mat:
            return "No usable aligned peaks for optimization.", None

        centers_arr = np.asarray(centers_mat, dtype=float)
        sigmas_arr = np.asarray(sigmas_mat, dtype=float)
        fracs_arr = np.asarray(fracs_mat, dtype=float)
        with np.errstate(all="ignore"):
            cen = np.nanmean(centers_arr, axis=0)
            sig_l = np.nanmean(sigmas_arr, axis=0)
            frc = np.nanmean(fracs_arr, axis=0)
        for i in range(k):
            if not np.isfinite(cen[i]):
                vals = centers_arr[:, i]
                cen[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else float(canonical_centers[i])
            if not np.isfinite(sig_l[i]):
                vals = sigmas_arr[:, i]
                sig_l[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else float(sigma_default)
            if not np.isfinite(frc[i]):
                vals = fracs_arr[:, i]
                frc[i] = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.5

        # Determine global x-bounds across the series
        try:
            x_min = float("inf")
            x_max = float("-inf")
            for _idx, _row in series_df.iterrows():
                x_arr, y_arr = _parse_xy(_row)
                if x_arr is None or y_arr is None or x_arr.size == 0:
                    continue
                x_min = min(x_min, float(np.nanmin(x_arr)))
                x_max = max(x_max, float(np.nanmax(x_arr)))
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                x_min, x_max = 0.0, 1.0
        except Exception:
            x_min, x_max = 0.0, 1.0

        def _build_model_fixed(cen_arr):
            comp = None
            params = None
            for i in range(k):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                p = m.make_params()
                p[f"p{i}_center"].set(value=float(cen_arr[i]), vary=False)
                p[f"p{i}_sigma"].set(
                    value=float(sig_l[i]), min=1e-3, max=1e4, vary=False
                )
                p[f"p{i}_fraction"].set(
                    value=float(frc[i]), min=0.0, max=1.0, vary=False
                )
                p[f"p{i}_amplitude"].set(min=0.0, value=1.0)
                if comp is None:
                    comp = m
                    params = p
                else:
                    comp = comp + m
                    params.update(p)
            return comp, params

        def _fit_and_sse(cen_arr, assign=False, capture=None):
            comp, base = _build_model_fixed(cen_arr)
            total = 0.0
            cache = {}
            for _idx, _row in series_df.iterrows():
                # Always compute residuals against processed data
                # Explicitly read from 'X-Axis' and 'Normalized and Corrected Data'
                x_val = _row.get("X-Axis")
                y_val = _row.get("Normalized and Corrected Data")
                if isinstance(x_val, str):
                    try:
                        x_val = ast.literal_eval(x_val)
                    except Exception:
                        x_val = None
                if isinstance(y_val, str):
                    try:
                        y_val = ast.literal_eval(y_val)
                    except Exception:
                        y_val = None
                try:
                    x_arr = np.asarray(x_val, dtype=float)
                    y_arr = np.asarray(y_val, dtype=float)
                    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
                        continue
                except Exception:
                    continue
                # Restrict to union x-ranges
                try:
                    x_use, y_use = _slice_to_union_ranges(x_arr, y_arr, current_union_x_ranges)
                except Exception:
                    x_use, y_use = x_arr, y_arr
                if x_use is None or y_use is None or np.asarray(x_use).size <= 1 or np.asarray(y_use).size <= 1:
                    continue
                x_arr = np.asarray(x_use, dtype=float)
                y_arr = np.asarray(y_use, dtype=float)

                p = base.copy()
                aligned = aligned_by_idx.get(_idx)
                if isinstance(aligned, list) and len(aligned) == k:
                    for i in range(k):
                        try:
                            ai = float(aligned[i].get("amplitude", aligned[i].get("A", p[f"p{i}_amplitude"].value)))
                            p[f"p{i}_amplitude"].set(value=max(0.0, ai))
                        except Exception:
                            pass
                else:
                    for i in range(k):
                        try:
                            ci = float(cen_arr[i])
                            nearest = int(np.argmin(np.abs(x_arr - ci)))
                            ai0 = max(0.0, float(y_arr[nearest]) * max(1.0, float(sig_l[i])))
                            p[f"p{i}_amplitude"].set(value=ai0)
                        except Exception:
                            pass
                try:
                    res = comp.fit(y_arr, p, x=x_arr)
                    y_fit = comp.eval(res.params, x=x_arr)
                    err = y_arr - y_fit
                    total += float(np.nansum(err * err))
                    if assign:
                        fit_list = []
                        amps_only = []
                        for j in range(k):
                            try:
                                amp = float(res.params.get(f"p{j}_amplitude").value)
                            except Exception:
                                amp = float("nan")
                            fit_list.append({"amplitude": amp})
                            try:
                                amps_only.append(float(amp) if np.isfinite(amp) else 0.0)
                            except Exception:
                                amps_only.append(0.0)
                        cache[_idx] = fit_list
                        try:
                            FTIR_DataFrame.at[_idx, "Material Fit Results"] = amps_only
                        except Exception:
                            pass
                except Exception:
                    total += 1e12
            if assign:
                if capture is not None:
                    try:
                        capture.clear()
                        capture.update(cache)
                    except Exception:
                        pass
            return total

        # Initial evaluation: fit amplitudes with seed centers and write results
        cen_init = cen.copy()
        cen_curr = cen.copy()
        init_cache = {}
        best_sse = _fit_and_sse(cen_curr, assign=True, capture=init_cache)
        initial_sse = best_sse
        # Track the centers corresponding to the current best SSE explicitly
        best_centers = cen_curr.copy()

        # Step size based on sigma scale
        try:
            sig_med = (
                float(np.nanmedian(sig_l)) if np.isfinite(np.nanmedian(sig_l)) else 10.0
            )
        except Exception:
            sig_med = 10.0
        step = max(0.5, min(10.0, 0.2 * sig_med))
        min_step = 0.05
        max_passes = 10
        tol = 1e-6

        passes_done = 0
        moves_attempted = 0
        moves_accepted = 0
        for _pass in range(max_passes):
            improved = False
            for i in range(k):
                for direction in (-1.0, 1.0):
                    cen_try = cen_curr.copy()
                    cen_try[i] = float(
                        np.clip(cen_try[i] + direction * step, x_min, x_max)
                    )
                    sse_try = _fit_and_sse(cen_try, assign=False)
                    moves_attempted += 1
                    if sse_try + tol < best_sse:
                        cen_curr = cen_try
                        best_sse = sse_try
                        best_centers = cen_try.copy()
                        improved = True
                        moves_accepted += 1
                        break
            if not improved:
                step *= 0.5
                if step < min_step:
                    break
            passes_done += 1

        # Final amplitudes re-fit and saved using the optimized centers (best found)
        final_cache = {}
        # Ensure we evaluate and assign using the best centers discovered
        cen_curr = best_centers.copy()
        final_sse = _fit_and_sse(cen_curr, assign=True, capture=final_cache)

        # Update shared params to reflect optimized centers
        nonlocal shared_centers_list, shared_sigma_list, shared_alpha_list, shared_peak_numbers_list
        try:
            shared_centers_list = [float(v) for v in cen_curr.tolist()]
        except Exception:
            shared_centers_list = cen_curr.tolist() if hasattr(cen_curr, "tolist") else list(cen_curr)
        try:
            shared_sigma_list = [float(v) if np.isfinite(v) else 10.0 for v in sig_l.tolist()]
        except Exception:
            shared_sigma_list = [10.0] * k
        try:
            shared_alpha_list = [float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 0.5 for v in frc.tolist()]
        except Exception:
            shared_alpha_list = [0.5] * k
        try:
            shared_peak_numbers_list = _compute_shared_peak_numbers_from_aligned(aligned_by_idx, k)
        except Exception:
            shared_peak_numbers_list = [None] * k
        # Report delta: negative means decreased SSE
        delta = final_sse - initial_sse
        msg = (
            f"Optimized centers and amplitudes over {series_df.shape[0]} spectra. SSE: {initial_sse:.6g} -> {final_sse:.6g}"
            + (f" (Δ={delta:.6g})" if np.isfinite(delta) else "")
        )
        # Build detailed summary
        try:
            center_deltas = (cen_curr - cen_init).tolist()
        except Exception:
            center_deltas = None

        # Compute mean amplitude per peak across spectra (before vs after)
        def _mean_amps(cache_dict):
            means = []
            try:
                for j in range(k):
                    vals = []
                    for _v in cache_dict.values():
                        try:
                            a = float(_v[j].get("amplitude", float("nan")))
                            if np.isfinite(a):
                                vals.append(a)
                        except Exception:
                            pass
                    means.append(float(np.nanmean(vals)) if len(vals) else float("nan"))
            except Exception:
                pass
            return means

        amp_init_means = _mean_amps(init_cache)
        amp_final_means = _mean_amps(final_cache)
        try:
            amp_delta_means = [
                (
                    (amp_final_means[i] - amp_init_means[i])
                    if i < len(amp_init_means)
                    and np.isfinite(amp_init_means[i])
                    and np.isfinite(amp_final_means[i])
                    else float("nan")
                )
                for i in range(max(len(amp_init_means), len(amp_final_means)))
            ]
        except Exception:
            amp_delta_means = []
        # Build per-spectrum amplitude deltas (final - initial) for each peak
        amp_delta_by_spectrum = []
        try:
            # Sort spectra by numeric Time if available
            df_order = series_df.copy()
            try:
                df_order["_sort_time"] = pd.to_numeric(
                    df_order.get("Time", np.nan), errors="coerce"
                )
                df_order["_sort_time"] = df_order["_sort_time"].fillna(float("inf"))
                df_order = df_order.sort_values(by=["_sort_time"], kind="mergesort")
            except Exception:
                pass
            for _idx, _row in df_order.iterrows():
                init_list = init_cache.get(_idx)
                final_list = final_cache.get(_idx)
                label = f"T={_row.get('Time')}"
                try:
                    file_name = _row.get("File Name", "")
                except Exception:
                    file_name = ""
                deltas = []
                if isinstance(init_list, list) and isinstance(final_list, list):
                    for j in range(k):
                        try:
                            a0 = float(init_list[j].get("amplitude", float("nan")))
                        except Exception:
                            a0 = float("nan")
                        try:
                            a1 = float(final_list[j].get("amplitude", float("nan")))
                        except Exception:
                            a1 = float("nan")
                        try:
                            d = (
                                a1 - a0
                                if np.isfinite(a0) and np.isfinite(a1)
                                else float("nan")
                            )
                        except Exception:
                            d = float("nan")
                        deltas.append(d)
                amp_delta_by_spectrum.append({"label": label, "file_name": file_name, "deltas": deltas})
        except Exception:
            amp_delta_by_spectrum = []
        summary = {
            "initial_sse": float(initial_sse) if np.isfinite(initial_sse) else None,
            "final_sse": float(final_sse) if np.isfinite(final_sse) else None,
            "passes": int(passes_done),
            "moves_attempted": int(moves_attempted),
            "moves_accepted": int(moves_accepted),
            "initial_centers": cen_init.tolist(),
            "final_centers": cen_curr.tolist(),
            "center_deltas": center_deltas,
            "peak_numbers": shared_peak_numbers_list,
            "amp_initial_means": amp_init_means,
            "amp_final_means": amp_final_means,
            "amp_delta_means": amp_delta_means,
            "amp_delta_by_spectrum": amp_delta_by_spectrum,
        }
        return msg, cen_curr.tolist(), summary

    # ---------------------------- UI helpers ----------------------------- #
    def _series_df(material_val, include_bad=False):
        df = FTIR_DataFrame.copy()
        # Optionally exclude rows marked as bad quality
        if not include_bad:
            try:
                df = df[_quality_good_mask(df)]
            except Exception:
                pass
        try:
            df = df[df["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        try:
            df = df[df["Material"].astype(str) == str(material_val)]
        except Exception:
            df = df[df.get("Material", "").astype(str) == str(material_val)]

        def _is_unexp(v):
            try:
                return str(v).strip().lower() == "unexposed"
            except Exception:
                return False

        # No condition filtering; include all spectra of the material
        try:
            df = df.copy()
            df["_sort_time"] = pd.to_numeric(df.get("Time", np.nan), errors="coerce")
            df["_sort_time"] = df["_sort_time"].fillna(float("inf"))
            df = df.sort_values(by=["_sort_time"], kind="mergesort")
            try:
                df = df.drop(columns=["_sort_time"])
            except Exception:
                pass
        except Exception:
            pass
        return df

    def _eval_timeseries_fit(row):
        x = row.get("X-Axis")
        y = row.get("Normalized and Corrected Data")
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return None, None, None
        if isinstance(y, str):
            try:
                y = ast.literal_eval(y)
            except Exception:
                return None, None, None
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return None, None, None
        # Build fit from amplitude-only results using shared parameters
        res = row.get("Material Fit Results")
        if isinstance(res, str):
            try:
                res = ast.literal_eval(res)
            except Exception:
                res = None
        amps = None
        if isinstance(res, list) and len(res) > 0:
            try:
                # list of amplitudes
                amps = [float(v) for v in res]
            except Exception:
                amps = None
        if amps is None or not isinstance(shared_centers_list, list) or not isinstance(shared_sigma_list, list) or not isinstance(shared_alpha_list, list):
            return x_arr, y_arr, None
        try:
            k = min(len(amps), len(shared_centers_list), len(shared_sigma_list), len(shared_alpha_list))
            comp_model = None
            params = None
            for i in range(k):
                m = PseudoVoigtModel(prefix=f"p{i}_")
                pr = m.make_params()
                pr[f"p{i}_center"].set(value=float(shared_centers_list[i]), vary=False)
                pr[f"p{i}_sigma"].set(value=float(shared_sigma_list[i]), min=1e-3, max=1e4, vary=False)
                pr[f"p{i}_fraction"].set(value=float(np.clip(shared_alpha_list[i], 0.0, 1.0)), min=0.0, max=1.0, vary=False)
                pr[f"p{i}_amplitude"].set(value=max(0.0, float(amps[i])), min=0.0)
                if comp_model is None:
                    comp_model = m
                    params = pr
                else:
                    comp_model = comp_model + m
                    params.update(pr)
            y_fit = comp_model.eval(params, x=x_arr) if comp_model is not None else None
            return x_arr, y_arr, y_fit
        except Exception:
            return x_arr, y_arr, None

    # ---------------------------- Build UI ------------------------------- #
    # Checkbox to include/exclude rows marked as bad quality
    include_bad_cb = widgets.Checkbox(
        value=False,
        description="Include bad spectra",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )

    try:
        df_opts = FTIR_DataFrame.copy()
        if not include_bad_cb.value:
            try:
                df_opts = df_opts[_quality_good_mask(df_opts)]
            except Exception:
                pass
        try:
            df_opts = df_opts[df_opts["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        unique_materials = (
            sorted(
                {
                    str(v)
                    for v in df_opts.get("Material", pd.Series([], dtype=object))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                }
            )
            if "Material" in df_opts.columns
            else []
        )
    except Exception:
        unique_materials = []
    if not unique_materials:
        raise ValueError("No materials found in DataFrame.")
    # Choose initial material from session defaults when available
    default_material = unique_materials[0]
    try:
        _sess_defaults = _get_session_defaults()
        _sess_mat = str(_sess_defaults.get("material", "any"))
        if _sess_mat in unique_materials:
            default_material = _sess_mat
    except Exception:
        pass
    try:
        dfm = FTIR_DataFrame[
            FTIR_DataFrame["Material"].astype(str) == str(default_material)
        ]
    except Exception:
        dfm = FTIR_DataFrame
    # Optionally exclude bad quality rows and require normalized data for condition options
    if not include_bad_cb.value:
        try:
            dfm = dfm[_quality_good_mask(dfm)]
        except Exception:
            pass
    try:
        dfm = dfm[dfm["Normalized and Corrected Data"].notna()]
    except Exception:
        pass

    material_dd = widgets.Dropdown(
        options=unique_materials,
        value=default_material,
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    fit_btn = widgets.Button(
        description="Fit Material",
        button_style="primary",
        layout=widgets.Layout(width="180px"),
        tooltip="Find parameter averages (preserving existing amplitudes) for the selected Material",
    )
    opt_btn = widgets.Button(
        description="Optimize",
        button_style="info",
        layout=widgets.Layout(width="180px"),
        tooltip=(
            "Iteratively optimize shared peak centers and per-spectrum amplitudes to "
            "minimize SSE against Normalized and Corrected Data"
        ),
    )
    close_btn = widgets.Button(description="Close", button_style="danger")
    status_html = widgets.HTML(value="")
    x_ranges_html = widgets.HTML(value="")
    optimize_status_html = widgets.HTML(value="")
    # Tables: Peak Areas (per-time amplitudes) and Peak Wavenumbers (shared centers)
    A_table_html = widgets.HTML(value="")
    WN_table_html = widgets.HTML(value="")
    # Store the most recently built tables for optional export
    last_table_df = None  # areas
    last_centers_list = None  # wavenumbers (list[float])

    # Plotting removed per request; keep tables and controls only

    def _plot_series(
        material_val=None, condition_val=None, with_fits=False, include_bad=None
    ):
        nonlocal last_table_df, last_centers_list
        # No plotting; only build tables and status
        if material_val is None:
            material_val = material_dd.value
        if include_bad is None:
            include_bad = include_bad_cb.value
        # _series_df accepts (material_val, include_bad) — no conditions argument
        df_series = _series_df(material_val, include_bad=include_bad)
        if df_series.empty:
            try:
                status_html.value = (
                    "<span style='color:#a00;'>No spectra for this selection.</span>"
                )
            except Exception:
                pass
            try:
                A_table_html.value = ""
            except Exception:
                pass
            try:
                WN_table_html.value = ""
            except Exception:
                pass
            return
        # Build a consistent color palette (blue → purple → red) based on time values,
        # matching the gradient used in plot_spectra
        time_values = []
        try:
            time_values = [
                t for t in df_series.get("Time", pd.Series([], dtype=object)) if pd.notna(t)
            ]
        except Exception:
            time_values = []
        try:
            time_values_unique = sorted(
                {float(t) for t in time_values if str(t).strip() not in ("", "nan")}
            )
        except Exception:
            # Fallback to string sorting
            time_values_unique = sorted({str(t) for t in time_values})

        def _time_to_color(val):
            return _time_gradient_color(time_values_unique, val)

        # Only show final time-series fits (no raw data traces)
        has_any_fit = False
        for idx, row in df_series.iterrows():
            if not with_fits:
                continue
            x_f, _y_f, y_fit = _eval_timeseries_fit(row)
            if y_fit is None or x_f is None:
                continue
            has_any_fit = True
            t_val = row.get("Time")
            name = f"T={t_val}"
            # Plotting removed
        # Build/update wavenumbers (shared centers), sigmas, alphas table and amplitude table
        try:
            if with_fits and has_any_fit:
                # Use shared canonical parameters (centers/sigmas/alphas)
                centers_list = shared_centers_list or []
                sigmas_list = shared_sigma_list or []
                fracs_list = shared_alpha_list or []
                peak_numbers_list = shared_peak_numbers_list or []

                if centers_list and len(centers_list) > 0:
                    try:
                        import pandas as pd  # local import safe here

                        # Build a table with rows: Center, Sigma (σ), Alpha (Lorentz); columns: Peak 1..N
                        k = len(centers_list)
                        header = _peak_labels_from_numbers(peak_numbers_list, k)

                        def _fmt(v, fmt=".6g"):
                            try:
                                return f"{float(v):{fmt}}" if np.isfinite(v) else ""
                            except Exception:
                                return ""

                        rows = [
                            ["Center (cm⁻¹)"]
                            + [_fmt(centers_list[i]) for i in range(k)],
                            ["σ (cm⁻¹)"]
                            + [
                                _fmt(
                                    (
                                        sigmas_list[i]
                                        if sigmas_list and i < len(sigmas_list)
                                        else float("nan")
                                    )
                                )
                                for i in range(k)
                            ],
                            ["α (Lorentz frac)"]
                            + [
                                _fmt(
                                    (
                                        fracs_list[i]
                                        if fracs_list and i < len(fracs_list)
                                        else float("nan")
                                    )
                                )
                                for i in range(k)
                            ],
                        ]
                        wn_df = pd.DataFrame(rows, columns=["Parameter"] + header)
                        html_wn = wn_df.to_html(index=False, escape=False)
                        WN_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Parameters</b></div>"
                            + f"<div style='max-height:180px; overflow:auto'>{html_wn}</div>"
                        )
                        last_centers_list = [
                            (
                                float(c)
                                if isinstance(c, (int, float)) and np.isfinite(c)
                                else float("nan")
                            )
                            for c in centers_list
                        ]
                    except Exception:
                        # Fallback manual HTML
                        k = len(centers_list or [])
                        header_cells_wn = "".join(
                            [f"<th>Parameter</th>"]
                            + [f"<th>{h}</th>" for h in _peak_labels_from_numbers(peak_numbers_list, k)]
                        )

                        def _cell(v, fmt=".6g"):
                            try:
                                return (
                                    f"<td>{float(v):{fmt}}</td>"
                                    if np.isfinite(v)
                                    else "<td></td>"
                                )
                            except Exception:
                                return "<td></td>"

                        row_center = "".join(
                            ["<td>Center (cm⁻¹)</td>"]
                            + [_cell(c) for c in (centers_list or [])]
                        )
                        row_sigma = "".join(
                            ["<td>σ (cm⁻¹)</td>"]
                            + [
                                _cell(
                                    sigmas_list[i]
                                    if sigmas_list and i < len(sigmas_list)
                                    else float("nan")
                                )
                                for i in range(k)
                            ]
                        )
                        row_alpha = "".join(
                            ["<td>α (Lorentz frac)</td>"]
                            + [
                                _cell(
                                    fracs_list[i]
                                    if fracs_list and i < len(fracs_list)
                                    else float("nan")
                                )
                                for i in range(k)
                            ]
                        )
                        WN_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Parameters</b></div>"
                            + "<div style='max-height:180px; overflow:auto'>"
                            + f"<table><thead><tr>{header_cells_wn}</tr></thead><tbody>"
                            + f"<tr>{row_center}</tr><tr>{row_sigma}</tr><tr>{row_alpha}</tr>"
                            + "</tbody></table>"
                            + "</div>"
                        )
                        try:
                            last_centers_list = [float(c) for c in (centers_list or [])]
                        except Exception:
                            last_centers_list = centers_list or []
                else:
                    WN_table_html.value = "<span style='color:#555;'>No peak parameters available to display for this selection.</span>"

                # Determine max number of peaks across available fits for areas table
                k_max = 0
                # Collect per-spectrum info for areas table: (time, conditions, amplitudes | None, notes)
                series_rows = []
                for _idx, _row in df_series.iterrows():
                    t_val = _row.get("Time")
                    # Prefer 'Conditions' else fallback to 'Condition'
                    try:
                        cond_val = _row.get("Conditions")
                        if cond_val is None and "Condition" in _row.index:
                            cond_val = _row.get("Condition")
                    except Exception:
                        cond_val = None
                    try:
                        file_name_val = _row.get("File Name")
                    except Exception:
                        file_name_val = None
                    res = _row.get("Material Fit Results")
                    if isinstance(res, str):
                        try:
                            res = ast.literal_eval(res)
                        except Exception:
                            res = None
                    if isinstance(res, list) and len(res) > 0:
                        try:
                            # amplitude-only list
                            amps = [float(a) for a in res]
                            k_max = max(k_max, len(amps))
                            try:
                                finite_amps = [a for a in amps if np.isfinite(a)]
                                if len(finite_amps) == 0:
                                    notes = "no usable fit amplitudes; padded"
                                elif all(abs(a) == 0.0 for a in finite_amps):
                                    notes = "all amplitudes zero (guarded/sanitized)"
                                else:
                                    notes = ""
                            except Exception:
                                notes = ""
                        except Exception:
                            amps = None
                    else:
                        amps = None
                        # If no fit list, row will be padded; add explanatory note
                        notes = "no fit results; padded"
                    series_rows.append((t_val, cond_val, file_name_val, amps, notes))

                if k_max > 0:
                    try:
                        import pandas as pd  # local import safe here

                        table_records = []
                        peak_headers = _peak_labels_from_numbers(peak_numbers_list, k_max)
                        for t_val, cond_val, file_name_val, amps, notes in series_rows:
                            # Build record with Time, Conditions, Peaks..., Notes
                            rec = {"File Name": file_name_val if file_name_val is not None else ""}
                            rec["Time"] = t_val if t_val is not None else ""
                            rec["Conditions"] = cond_val if cond_val is not None else ""
                            # Pad missing amplitudes with zeros to keep table dense
                            padded = []
                            try:
                                if isinstance(amps, list):
                                    padded = amps[:k_max] + [0.0] * max(0, k_max - len(amps))
                                else:
                                    padded = [0.0] * k_max
                            except Exception:
                                padded = [0.0] * k_max
                            for i in range(k_max):
                                val = ""
                                try:
                                    a = float(padded[i])
                                    if np.isfinite(a):
                                        val = f"{a:.6g}"
                                except Exception:
                                    val = ""
                                rec[peak_headers[i] if i < len(peak_headers) else f"Peak {i+1}"] = val
                            # Append Notes last to explain zero rows
                            rec["Notes"] = notes if isinstance(notes, str) else ""
                            table_records.append(rec)
                        df_table = pd.DataFrame(table_records)
                        html = df_table.to_html(index=False, escape=False)
                        A_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Areas</b></div>"
                            + f"<div style='max-height:300px; overflow:auto'>{html}</div>"
                        )
                        # Keep a copy for saving
                        try:
                            last_table_df = df_table.copy()
                        except Exception:
                            last_table_df = df_table
                    except Exception:
                        # Fallback manual HTML
                        peak_headers = _peak_labels_from_numbers(peak_numbers_list, k_max)
                        header_cells = "".join(
                            ["<th>File Name</th>", "<th>Time</th>", "<th>Conditions</th>"]
                            + [f"<th>{peak_headers[i] if i < len(peak_headers) else f'Peak {i+1}'}</th>" for i in range(k_max)]
                            + ["<th>Notes</th>"]
                        )
                        body_rows = []
                        for t_val, cond_val, file_name_val, amps, notes in series_rows:
                            cells = [
                                f"<td>{file_name_val if file_name_val is not None else ''}</td>",
                                f"<td>{t_val}</td>",
                                f"<td>{cond_val if cond_val is not None else ''}</td>",
                            ]
                            for i in range(k_max):
                                try:
                                    if (
                                        amps is not None
                                        and i < len(amps)
                                        and np.isfinite(amps[i])
                                    ):
                                        cells.append(f"<td>{amps[i]:.6g}</td>")
                                    else:
                                        cells.append("<td></td>")
                                except Exception:
                                    cells.append("<td></td>")
                            # Notes cell
                            try:
                                cells.append(f"<td>{notes}</td>")
                            except Exception:
                                cells.append("<td></td>")
                            body_rows.append(f"<tr>{''.join(cells)}</tr>")
                        A_table_html.value = (
                            "<div style='margin-top:8px'><b>Peak Areas</b></div>"
                            + "<div style='max-height:300px; overflow:auto'>"
                            + f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
                            + "</div>"
                        )
                else:
                    A_table_html.value = "<span style='color:#555;'>No fitted peaks available to tabulate for this selection.</span>"
            else:
                A_table_html.value = "<span style='color:#555;'>Run 'Fit Material' to populate the peak areas table.</span>"
                WN_table_html.value = "<span style='color:#555;'>Run 'Fit Material' to populate the peak parameters table.</span>"
        except Exception:
            pass
        try:
            status_html.value = (
                "<span style='color:#555;'>Click 'Fit Material' to compute and display fits for this selection.</span>"
                if not with_fits
                else "<span style='color:#000;'>Displayed material fits (amplitudes vary; centers/σ/α shared).</span>"
            )
        except Exception:
            pass

        # X-Range message (shown directly under the 'Displayed material fits...' line)
        try:
            if not with_fits:
                x_ranges_html.value = ""
            else:
                rs = current_union_x_ranges or []
                if rs:
                    rs_txt = ", ".join([f"[{lo:.6g}, {hi:.6g}]" for lo, hi in rs])
                    x_ranges_html.value = f"<span style='color:#000;'><b>X-Ranges used:</b> {rs_txt}</span>"
                else:
                    x_ranges_html.value = "<span style='color:#a00;'><b>X-Ranges used:</b> none found in 'Deconvolution X-Ranges' for this material (using full spectrum).</span>"
        except Exception:
            pass

    def _on_material_change(*_):
        try:
            mat = material_dd.value
            dfm2 = FTIR_DataFrame[FTIR_DataFrame["Material"].astype(str) == str(mat)]
        except Exception:
            dfm2 = FTIR_DataFrame
        if not include_bad_cb.value:
            try:
                dfm2 = dfm2[_quality_good_mask(dfm2)]
            except Exception:
                pass
        try:
            dfm2 = dfm2[dfm2["Normalized and Corrected Data"].notna()]
        except Exception:
            pass
        try:
            _plot_series(mat, with_fits=False, include_bad=include_bad_cb.value)
        except Exception:
            pass
    

    def _on_fit_click(_b=None):
        try:
            status_html.value = (
                "<span style='color:#555;'>Running fit...</span>"
            )
        except Exception:
            pass
        try:
            _compute_for_selection(material_dd.value, include_bad=include_bad_cb.value)
        except Exception as e:
            try:
                status_html.value = f"<span style='color:#a00;'>Fit failed: {e}</span>"
            except Exception:
                pass
            return
        _plot_series(with_fits=True, include_bad=include_bad_cb.value)

    def _on_optimize_click(_b=None):
        try:
            status_html.value = "<span style='color:#555;'>Optimizing centers and amplitudes (this may take a moment)...</span>"
        except Exception:
            pass

        # Compute SSE for the current selection BEFORE optimization using current stored fits
        def _series_sse(material_val):
            try:
                df_sel = _series_df(material_val, include_bad=include_bad_cb.value)
            except Exception:
                return float("nan")
            total = 0.0
            any_added = False
            for _idx, _row in df_sel.iterrows():
                try:
                    x_arr, y_arr, y_fit = _eval_timeseries_fit(_row)
                    if x_arr is None or y_arr is None or y_fit is None:
                        continue
                    # ensure arrays
                    y_fit_arr = np.asarray(y_fit, dtype=float)
                    y_arr = np.asarray(y_arr, dtype=float)
                    if y_fit_arr.shape != y_arr.shape:
                        n = min(y_fit_arr.size, y_arr.size)
                        if n <= 0:
                            continue
                        y_fit_arr = y_fit_arr[:n]
                        y_arr = y_arr[:n]
                    mask = np.isfinite(y_arr) & np.isfinite(y_fit_arr)
                    if not np.any(mask):
                        continue
                    # Restrict SSE computation to current union x-ranges
                    try:
                        x_np = np.asarray(x_arr, dtype=float)
                        y_np = np.asarray(y_arr, dtype=float)
                        yfit_np = np.asarray(y_fit_arr, dtype=float)
                        n2 = min(x_np.size, y_np.size, yfit_np.size)
                        x_np = x_np[:n2]
                        y_np = y_np[:n2]
                        yfit_np = yfit_np[:n2]
                        x_s, y_s = _slice_to_union_ranges(x_np, y_np, current_union_x_ranges)
                        x_s2, yfit_s = _slice_to_union_ranges(x_np, yfit_np, current_union_x_ranges)
                        if x_s is None or y_s is None or yfit_s is None or y_s.size <= 1 or yfit_s.size <= 1:
                            continue
                        # y_s and yfit_s correspond to the same x mask because both sliced by x_np
                        err = y_s - yfit_s
                    except Exception:
                        err = y_arr[mask] - y_fit_arr[mask]
                    total += float(np.sum(err * err))
                    any_added = True
                except Exception:
                    continue
            return total if any_added else float("nan")

        try:
            sse_before_calc = _series_sse(material_dd.value)
        except Exception:
            sse_before_calc = float("nan")
        try:
            result = _optimize_centers_for_selection(
                material_dd.value, include_bad=include_bad_cb.value
            )
            # Backward compatibility if tuple size differs
            if isinstance(result, tuple) and len(result) == 3:
                msg, centers_out, summary = result
            elif isinstance(result, tuple) and len(result) == 2:
                msg, centers_out = result
                summary = None
            else:
                msg = str(result)
                centers_out = None
                summary = None
            if centers_out is None:
                status_html.value = f"<span style='color:#a00;'>{msg}</span>"
                try:
                    optimize_status_html.value = f"<div style='color:#a00'>{msg}</div>"
                except Exception:
                    pass
            else:
                status_html.value = f"<span style='color:#0a0;'>{msg}</span>"
                # Build a persistent, detailed optimization log
                try:
                    # Compute SSE AFTER optimization using updated fits stored in the DataFrame
                    try:
                        sse_after_calc = _series_sse(material_dd.value)
                    except Exception:
                        sse_after_calc = float("nan")
                    if summary is None:
                        optimize_status_html.value = (
                            f"<div><b>Optimization</b>: {msg}</div>"
                        )
                    else:
                        # Compose a compact table of per-peak changes
                        centers0 = summary.get("initial_centers", []) or []
                        centers1 = summary.get("final_centers", []) or []
                        deltas = summary.get("center_deltas", []) or []
                        peak_nums = summary.get("peak_numbers", []) or []
                        peak_labels = _peak_labels_from_numbers(peak_nums, klen)
                        amp0 = summary.get("amp_initial_means", []) or []
                        amp1 = summary.get("amp_final_means", []) or []
                        ampd = summary.get("amp_delta_means", []) or []
                        rows = []
                        klen = max(len(centers0), len(centers1), len(deltas))
                        for i in range(klen):
                            c0 = centers0[i] if i < len(centers0) else None
                            c1 = centers1[i] if i < len(centers1) else None
                            dd = deltas[i] if i < len(deltas) else None

                            def _fmt(v):
                                try:
                                    return f"{float(v):.6g}"
                                except Exception:
                                    return ""

                            rows.append(
                                f"<tr><td>{peak_labels[i] if i < len(peak_labels) else f'Peak {i+1}'}</td><td>{_fmt(c0)}</td><td>{_fmt(c1)}</td><td>{_fmt(dd)}</td></tr>"
                            )
                        table_html = (
                            "<table style='border-collapse:collapse'>"
                            "<thead><tr><th></th><th>Initial center</th><th>Final center</th><th>Δ center</th></tr></thead>"
                            f"<tbody>{''.join(rows)}</tbody></table>"
                        )
                        # Amplitude change table by spectrum (Δ per peak)
                        delta_by_spec = summary.get("amp_delta_by_spectrum", []) or []
                        # Determine max number of peaks to render
                        kmax = 0
                        for item in delta_by_spec:
                            try:
                                kmax = max(kmax, len(item.get("deltas", [])))
                            except Exception:
                                pass
                        # Build header and body rows
                        header_cells = "".join(
                            ["<th>File Name</th>", "<th>Series</th>"]
                            + [f"<th>{(peak_labels[i] if i < len(peak_labels) else f'Peak {i+1}')} Δ</th>" for i in range(kmax)]
                        )
                        body_rows = []

                        def _fmtd(v):
                            try:
                                return f"{float(v):.6g}"
                            except Exception:
                                return ""

                        for item in delta_by_spec:
                            label = item.get("label", "")
                            file_name = item.get("file_name", "")
                            deltas = item.get("deltas", []) or []
                            cells = [f"<td>{file_name}</td>", f"<td>{label}</td>"]
                            for i in range(kmax):
                                val = deltas[i] if i < len(deltas) else None
                                cells.append(f"<td>{_fmtd(val)}</td>")
                            body_rows.append(f"<tr>{''.join(cells)}</tr>")
                        amp_table_html = (
                            "<table style='border-collapse:collapse'>"
                            f"<thead><tr>{header_cells}</tr></thead>"
                            f"<tbody>{''.join(body_rows)}</tbody></table>"
                        )
                        # Prefer recomputed SSE for consistent selection and evaluation; fallback to optimizer-reported values
                        sse0 = (
                            sse_before_calc
                            if np.isfinite(sse_before_calc)
                            else summary.get("initial_sse")
                        )
                        if sse0 is None or not np.isfinite(sse0):
                            sse0 = None
                        sse1 = (
                            sse_after_calc
                            if np.isfinite(sse_after_calc)
                            else summary.get("final_sse")
                        )
                        if sse1 is None or not np.isfinite(sse1):
                            sse1 = None
                        try:
                            # Display delta as (final - initial): negative means decreased SSE
                            delta_disp = (
                                (sse1 - sse0)
                                if (sse0 is not None and sse1 is not None)
                                else None
                            )
                        except Exception:
                            delta_disp = None
                        passes_done = summary.get("passes")
                        m_att = summary.get("moves_attempted")
                        m_acc = summary.get("moves_accepted")
                        optimize_status_html.value = (
                            "<div style='margin-top:8px'><b>Optimization Summary</b></div>"
                            + (
                                f"<div>SSE: {sse0:.6g} → {sse1:.6g} (Δ={delta_disp:.6g}) | Passes: {passes_done} | Moves: {m_acc}/{m_att} accepted</div>"
                                if (sse0 is not None and sse1 is not None)
                                else f"<div>Passes: {passes_done} | Moves: {m_acc}/{m_att} accepted</div>"
                            )
                            + "<div style='color:#555; margin-top:2px'>(Amplitudes re-fit per spectrum during optimization)</div>"
                            + f"<div style='margin-top:6px'><b>Peak Centers</b></div>"
                            + f"<div style='margin-top:2px'>{table_html}</div>"
                            + f"<div style='margin-top:8px'><b>Amplitude Changes by Spectrum (Δ = final − initial)</b></div>"
                            + f"<div style='margin-top:2px'>{amp_table_html}</div>"
                        )
                except Exception:
                    pass
        except Exception as e:
            try:
                status_html.value = (
                    f"<span style='color:#a00;'>Optimize failed: {e}</span>"
                )
            except Exception:
                pass
            return

    _plot_series(with_fits=True, include_bad=include_bad_cb.value)

    def _on_save_click(_b=None):
        """Save amplitudes to DataFrame; write σ and α to materials.json for the selected material."""
        try:
            mat = str(material_dd.value)
            cond = None
        except Exception:
            mat = "material"
            cond = "condition"
        # Build a fresh table to ensure we capture latest amplitudes, independent of UI
        df_series = _series_df(mat, include_bad=include_bad_cb.value)
        if df_series.empty:
            try:
                status_html.value = "<span style='color:#a00;'>Nothing to save for this selection.</span>"
            except Exception:
                pass
            return

        # Only save for rows that were actually used in the fit/optimization.
        # (i.e., had usable Deconvolution Results that contributed to the canonical peak set)
        nonlocal last_used_fit_row_indices
        used_idx = set()
        try:
            used_idx = set(int(i) for i in (last_used_fit_row_indices or set()))
        except Exception:
            used_idx = set()

        # Fallback inference when user clicks Save without running Fit/Optimize this session
        if not used_idx:
            try:
                for _idx, _row in df_series.iterrows():
                    pk = _parse_deconv(_row.get("Deconvolution Results"))
                    if pk is not None and len(pk) > 0:
                        used_idx.add(int(_idx))
            except Exception:
                used_idx = set()

        try:
            df_series = df_series.loc[df_series.index.intersection(sorted(used_idx))]
        except Exception:
            pass

        if df_series.empty:
            try:
                status_html.value = "<span style='color:#a00;'>Nothing to save: no rows with usable Deconvolution Results were used in the fit.</span>"
            except Exception:
                pass
            return
        # Determine peak count and build records for areas
        k_max = 0
        for _idx, _row in df_series.iterrows():
            res = _row.get("Material Fit Results")
            if isinstance(res, str):
                try:
                    res = ast.literal_eval(res)
                except Exception:
                    res = None
            if isinstance(res, list) and len(res) > 0:
                try:
                    # amplitude-only
                    amps = [float(a) for a in res]
                    k_max = max(k_max, len(amps))
                except Exception:
                    amps = None
            else:
                amps = None
        if k_max <= 0:
            try:
                status_html.value = "<span style='color:#a00;'>No fitted peaks to save for this selection.</span>"
            except Exception:
                pass
            return
        # Persist amplitudes back into 'Material Fit Results'
        try:
            dest_col = "Material Fit Results"
            if dest_col not in FTIR_DataFrame.columns:
                try:
                    FTIR_DataFrame[dest_col] = None
                except Exception:
                    pass
            try:
                FTIR_DataFrame[dest_col] = FTIR_DataFrame[dest_col].astype(object)
            except Exception:
                pass

            updated = 0
            for idx, _row in df_series.iterrows():
                res = _row.get(dest_col)
                if isinstance(res, str):
                    try:
                        res = ast.literal_eval(res)
                    except Exception:
                        res = None
                if isinstance(res, list) and len(res) > 0:
                    try:
                        FTIR_DataFrame.at[idx, dest_col] = [float(a) for a in res]
                        updated += 1
                    except Exception:
                        pass
            try:
                status_html.value = f"<span style='color:#0a0;'>Saved per-row peak amplitudes to '{dest_col}' for {updated} row(s) used in the fit.</span>"
            except Exception:
                pass
            # --- JSON update: write σ and α for canon peaks of this material --- #
            try:
                path, content = _load_materials_json()
                if content is None or not isinstance(content, list) or not content:
                    raise ValueError("materials.json unavailable")
                top = content[0]
                code_key = _lookup_material_code(top, mat)
                if code_key is None:
                    print(f"[fit_material] Material '{mat}' not found in materials.json; skip JSON update.")
                else:
                    mat_payload = top.get(code_key, {}) or {}
                    peaks_payload = mat_payload.get("peaks", {}) or {}
                    kjson = len(shared_centers_list or [])
                    # Use peak numbers sourced from Deconvolution Results, falling back to 1..N
                    peak_nums = shared_peak_numbers_list or [None] * kjson
                    for i in range(kjson):
                        try:
                            pn = peak_nums[i] if i < len(peak_nums) else None
                        except Exception:
                            pn = None
                        key = str(int(pn)) if pn is not None else str(i + 1)
                        entry = peaks_payload.get(key, {})
                        # Update center/σ/α using the newest shared fit values
                        try:
                            entry["center_wavenumber"] = float(shared_centers_list[i])
                        except Exception:
                            entry["center_wavenumber"] = entry.get("center_wavenumber", 0.0)
                        try:
                            entry["σ"] = float(shared_sigma_list[i])
                        except Exception:
                            entry["σ"] = entry.get("σ", 0.0)
                        try:
                            entry["α"] = float(shared_alpha_list[i])
                        except Exception:
                            entry["α"] = entry.get("α", 0.0)
                        peaks_payload[key] = entry
                    mat_payload["peaks"] = peaks_payload
                    top[code_key] = mat_payload
                    # Write file
                    try:
                        with open(path, "w", encoding="utf-8") as jf:
                            json.dump(content, jf, indent=4, ensure_ascii=False)
                        print(f"[fit_material] materials.json updated (σ, α) for {mat}.")
                    except Exception as _je:
                        print(f"[fit_material] Failed to write materials.json: {_je}")
            except Exception as _json_err:
                print(f"[fit_material] JSON update skipped: {_json_err}")
        except Exception as e:
            try:
                status_html.value = f"<span style='color:#a00;'>Failed to save per-row results: {e}</span>"
            except Exception:
                pass

    def _on_close(_b=None):
        try:
            material_dd.close()
            fit_btn.close()
            close_btn.close()
            status_html.close()
            optimize_status_html.close()
            ui.close()
            # No figure to close
        except Exception:
            pass

    def _refresh_materials_and_conditions():
        """Recompute material options based on include_bad toggle and current data."""
        try:
            df_opts_local = FTIR_DataFrame.copy()
            if not include_bad_cb.value:
                try:
                    df_opts_local = df_opts_local[_quality_good_mask(df_opts_local)]
                except Exception:
                    pass
            try:
                df_opts_local = df_opts_local[
                    df_opts_local["Normalized and Corrected Data"].notna()
                ]
            except Exception:
                pass
            new_materials = (
                sorted(
                    {
                        str(v)
                        for v in df_opts_local.get(
                            "Material", pd.Series([], dtype=object)
                        )
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    }
                )
                if "Material" in df_opts_local.columns
                else []
            )
        except Exception:
            new_materials = []
        # Update materials dropdown
        try:
            cur_mat = (
                material_dd.value
                if material_dd.value in getattr(material_dd, "options", [])
                else None
            )
            material_dd.options = new_materials
            if not new_materials:
                return
            if cur_mat not in new_materials:
                material_dd.value = new_materials[0]
        except Exception:
            pass
        # Re-plot
        _plot_series(with_fits=False, include_bad=include_bad_cb.value)

    def _on_include_bad_toggle(*_):
        _refresh_materials_and_conditions()

    material_dd.observe(_on_material_change, names="value")

    # Persist Material/Conditions selections to session
    def _persist_ts_filters(_=None):
        try:
            _set_session_selection(material=material_dd.value)
        except Exception:
            pass

    material_dd.observe(_persist_ts_filters, names="value")
    fit_btn.on_click(_on_fit_click)
    opt_btn.on_click(_on_optimize_click)
    include_bad_cb.observe(_on_include_bad_toggle, names="value")
    # Add Save button to save per-row results (wavenumbers + areas) back into 'Material Fit Results'
    save_btn = widgets.Button(
        description="Save",
        button_style="success",
        layout=widgets.Layout(width="120px"),
        tooltip="Save per-row peak parameters to 'Material Fit Results'",
    )
    save_btn.on_click(_on_save_click)
    close_btn.on_click(_on_close)

    # Keep dropdowns/checkbox in one row; buttons in a separate row
    controls = widgets.HBox([material_dd, include_bad_cb])
    buttons = widgets.HBox([fit_btn, opt_btn, save_btn, close_btn])
    ui = widgets.VBox([controls, buttons, status_html, x_ranges_html, optimize_status_html])
    display(ui, WN_table_html, A_table_html)
    # Ensure options reflect current include_bad state
    _refresh_materials_and_conditions()
    _plot_series(with_fits=False, include_bad=include_bad_cb.value)

    return FTIR_DataFrame


def check_fit_quality(FTIR_DataFrame):
    """Interactive review of material fit quality.

    - Presents a dropdown to select a material.
    - When selected, checks if the material has entries in the
      'Material Fit Results' column.
      - If none, shows a message prompting to run 'fit_material' first.
      - If present, computes error between the stored Fit and the
        'Normalized and Corrected Data' for each spectrum, and displays
        a list sorted by ascending error to quickly identify poor fits.

    Returns the DataFrame unchanged.
    """

    _require_columns(
        FTIR_DataFrame,
        [
            "Material",
            "X-Axis",
            "Normalized and Corrected Data",
            "Material Fit Results",
            "File Name",
            "Conditions",
            "Time",
        ],
        context="check_fit_quality",
    )

    try:
        materials, _conds = _extract_material_condition_lists(FTIR_DataFrame)
    except Exception:
        materials = []

    from IPython.display import display, clear_output  # type: ignore
    if widgets is None:
        if not materials:
            print("No materials found. Populate the DataFrame first.")
            return FTIR_DataFrame
        print("Widgets not available; showing per-material fit quality summary:")
        for mat in materials:
            _check_fit_quality_material(FTIR_DataFrame, mat, _print_only=True)
        return FTIR_DataFrame

    material_options = ["Select material..."] + (materials or [])
    dropdown = widgets.Dropdown(options=material_options, description="Material:", layout=widgets.Layout(width="400px"))
    out = widgets.Output()
    # Persistent status/output area that must remain after Close
    top_status_out = widgets.Output()
        # Track per-session changes to emit a meaningful summary on Close
    session_changes = {"quality": []}
    # Show brief instructions initially (will be replaced with session summary on Close)
    try:
        _emit_function_summary(
            top_status_out,
            [
                "Choose a material to review fit quality.",
                "Click Close to end and view session summary.",
            ],
            title="Quality Check",
        )
    except Exception:
        with top_status_out:
            print("Quality Check: choose a material, or Close to hide the UI.")
    close_top_btn = widgets.Button(description="Close", button_style="danger", layout=widgets.Layout(width="100px"))
    ui = widgets.VBox([widgets.HBox([close_top_btn, dropdown]), out])

    def on_material_change(change):
        if change.get("name") != "value":
            return
        sel = change.get("new")
        # Ensure only material-level messages are shown; clear the top status
        try:
            with top_status_out:
                clear_output()
        except Exception:
            pass
        with out:
            clear_output()
            if not sel or str(sel).strip() == "Select material...":
                print("Select a material from the dropdown.")
                return
            _check_fit_quality_material(FTIR_DataFrame, sel, _print_only=False, _changes=session_changes)

    def _on_top_close(_b=None):
        try:
            dropdown.close()
        except Exception:
            pass
        try:
            out.clear_output()
            out.close()
        except Exception:
            pass
        try:
            close_top_btn.close()
        except Exception:
            pass
        try:
            ui.close()
        except Exception:
            pass
        # Leave top_status_out visible but replace its content with a session summary
        # Build and emit session summary based on collected changes
        try:
            with top_status_out:
                clear_output()
        except Exception:
            pass
        try:
            lines = _session_summary_lines(session_changes, context="Quality Check")
        except Exception:
            lines = [
                "Quality Check session closed.",
            ]
        try:
            _emit_function_summary(
                top_status_out,
                lines,
                title="Session Summary (Quality Check)",
            )
        except Exception:
            # Fallback to plain print if helpers are unavailable
            try:
                with top_status_out:
                    print("Quality Check session closed.")
            except Exception:
                pass

    close_top_btn.on_click(_on_top_close)
    dropdown.observe(on_material_change, names="value")
    display(ui, top_status_out)
    return FTIR_DataFrame


def _check_fit_quality_material(FTIR_DataFrame, material: str, _print_only: bool = False, _changes: dict | None = None):
    """Compute and display per-spectrum fit errors for a single material.

    - If no usable fit data exists for the material, prints a clear message.
    - Otherwise, displays a sorted table of spectra by RMSE between Fit and
      'Normalized and Corrected Data'.
    - When _print_only is True, prints textual summary instead of a table.
    """
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from IPython.display import display  # type: ignore

    try:
        mat_mask = (
            FTIR_DataFrame["Material"].astype(str).str.lower()
            == str(material).strip().lower()
        )
    except Exception:
        mat_mask = []
    subset = FTIR_DataFrame.loc[mat_mask] if len(getattr(FTIR_DataFrame, "index", [])) > 0 else FTIR_DataFrame
    if subset is None or len(subset) == 0:
        print(f"No spectra found for material '{material}'.")
        return

    errors = []
    any_fit_present = False

    def _fit_cell_is_empty(val) -> bool:
        """Return True when the Material Fit Results cell should be treated as empty.

        Handles None, NaN, blank strings, empty literals ("[]", "{}"), and common
        textual placeholders like "none", "null", "nan" (case-insensitive).
        """
        try:
            import pandas as _pd  # local to avoid top-level dependency changes
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return True
            if isinstance(val, str):
                s = val.strip().lower()
                if s in {"", "none", "null", "nan"}:
                    return True
                if s in {"[]", "{}"}:
                    return True
            if isinstance(val, (list, tuple, dict)) and len(val) == 0:
                return True
            if '_pd' in locals() and _pd.isna(val):
                return True
        except Exception:
            pass
        return False

    for idx, row in subset.iterrows():
        x = _parse_seq(row.get("X-Axis"))
        y = _parse_seq(row.get("Normalized and Corrected Data"))
        mfr_raw = row.get("Material Fit Results")
        mfr = _safe_literal_eval(mfr_raw, value_name="Material Fit Results")

        note = ""
        err = np.inf
        y_fit = None

        if not _fit_cell_is_empty(mfr_raw):
            any_fit_present = True

        try:
            if isinstance(mfr, dict):
                for key in ("fit_y", "y_fit", "fit", "y"):
                    if key in mfr:
                        y_fit = _parse_seq(mfr[key])
                        break
            elif isinstance(mfr, list):
                # Reconstruct composite using lmfit's PseudoVoigtModel components
                components = []
                for comp in mfr:
                    if not isinstance(comp, dict):
                        continue
                    try:
                        A = float(comp.get("amplitude", comp.get("A", 0)) or 0.0)
                        c = float(comp.get("center", comp.get("c", np.nan)) or np.nan)
                        s_val = comp.get("sigma", comp.get("sigma_g", comp.get("sigma_l", None)))
                        s = float(s_val) if s_val is not None else np.nan
                        frac_val = comp.get("alpha", comp.get("fraction", comp.get("f", None)))
                        frac = float(frac_val) if frac_val is not None else 0.5
                    except Exception:
                        continue
                    if np.isfinite(A) and np.isfinite(c) and np.isfinite(s) and s > 0 and np.isfinite(frac):
                        components.append((A, c, s, frac))

                if x is not None and y is not None and components:
                    try:
                        xx = np.array(x, dtype=float)
                        # Build composite model and parameters
                        model = None
                        params = None
                        for i, (A, c, s, frac) in enumerate(components, start=1):
                            pv = PseudoVoigtModel(prefix=f"p{i}_")
                            model = pv if model is None else model + pv
                            p = pv.make_params()
                            p[f"p{i}_amplitude"].set(value=max(A, 0.0), min=0.0)
                            p[f"p{i}_center"].set(value=c)
                            p[f"p{i}_sigma"].set(value=max(s, 1e-9), min=1e-12)
                            # alpha in [0,1]
                            alpha_val = min(max(frac, 0.0), 1.0)
                            p[f"p{i}_fraction"].set(value=alpha_val, min=0.0, max=1.0)
                            params = p if params is None else params.update(p)
                        yy = model.eval(params=params, x=xx)
                        y_fit = yy.tolist()
                    except Exception:
                        # Fall back to None so error handling notes it as unusable
                        y_fit = None
        except Exception:
            note = "fit data parse error"

        if (
            y_fit is None
            or y is None
            or x is None
            or len(y) != len(x)
            or len(y_fit) != len(x)
        ):
            if _fit_cell_is_empty(mfr_raw):
                note = note or "fit not ran"
            else:
                note = note or "unusable fit"
            err = np.inf
        else:
            arr_y = np.array(y, dtype=float)
            arr_fit = np.array(y_fit, dtype=float)
            dif = arr_y - arr_fit
            err = float(np.sqrt(np.mean(dif ** 2)))

        errors.append(
            {
                "Index": idx,
                "File Name": row.get("File Name", ""),
                "Conditions": row.get("Conditions", ""),
                "Time": row.get("Time", ""),
                "Error": err,
                "Note": note,
            }
        )

    if not any_fit_present:
        print(
            f"No Material Fit Results found for material '{material}'. Please run 'fit_material' first."
        )
        return

    df_sorted = pd.DataFrame(errors)
    # Split into usable (finite error) and excluded (infinite or missing)
    finite_mask = np.isfinite(df_sorted["Error"].to_numpy()) if not df_sorted.empty else np.array([])
    df_finite = df_sorted[finite_mask] if finite_mask.size else df_sorted.iloc[0:0]
    df_excluded = df_sorted[~finite_mask] if finite_mask.size else df_sorted
    if df_sorted.empty or df_finite.empty:
        print(
            f"Fit results exist but are unusable for material '{material}'. Please re-run 'fit_material' or inspect deconvolution parameters."
        )
        return

    # Sort the usable rows by ascending error
    df_sorted = df_finite.sort_values(by="Error", ascending=True)

    if _print_only:
        print(f"Material: {material}")
        print("Worst-fitting spectra (ascending RMSE):")
        for _, r in df_sorted.iterrows():
            print(
                f"  idx={r['Index']} | time={r['Time']} | cond={r['Conditions']} | err={r['Error']:.6f} | file='{r['File Name']}'"
            )
    else:
        # Build an interactive list with Plot buttons for rows having finite error
        try:
            import ipywidgets as widgets  # local import for notebook UI
            from IPython.display import display
        except Exception:
            display(df_sorted)
            return

        # Output areas: plotting overlays and persistent status messages
        plot_out = widgets.Output()
        status_out = widgets.Output()  # persistent session summary / status
        excluded_out = widgets.Output()  # separate area for excluded list (toggle)
        # Seed initial instruction using universal emit helper
        _emit_function_summary(
            status_out,
            [
                "Select a row and click Plot to view overlay.",
                "Use Mark buttons to set quality.",
            ],
            title="Session Summary (Quality Check)",
        )
        # Track currently plotted row index for quality marking
        current_plot_idx = {"idx": None}
        # Quality controls bound to the currently plotted row
        mark_bad_btn, mark_good_btn, refresh_quality = _make_quality_controls(
            FTIR_DataFrame,
            row_getter=lambda: (
                FTIR_DataFrame.loc[current_plot_idx["idx"]]
                if current_plot_idx["idx"] is not None
                else None
            ),
            margin="8px 10px 0 0",
            status_out=status_out,
        )
        # Record quality mark changes into session changes for summary on Close
        def _record_mark(status: str):
            try:
                idx = current_plot_idx.get("idx")
            except Exception:
                idx = None
            if _changes is not None and idx is not None:
                try:
                    lst = _changes.setdefault("quality", [])
                    lst.append((idx, status))
                except Exception:
                    pass

        mark_bad_btn.on_click(lambda _b=None: _record_mark("bad"))
        mark_good_btn.on_click(lambda _b=None: _record_mark("good"))

        def _get_xy_and_fit(row_idx):
            try:
                row = FTIR_DataFrame.loc[row_idx]
            except Exception:
                return None, None, None, None
            # Parse x and y
            x = _parse_seq(row.get("X-Axis"))
            y = _parse_seq(row.get("Normalized and Corrected Data"))
            # Parse fit cell
            mfr_raw = row.get("Material Fit Results")
            mfr = _safe_literal_eval(mfr_raw, value_name="Material Fit Results")
            y_fit = None
            try:
                if isinstance(mfr, dict):
                    for key in ("fit_y", "y_fit", "fit", "y"):
                        if key in mfr:
                            y_fit = _parse_seq(mfr[key])
                            break
                elif isinstance(mfr, list):
                    # Reconstruct composite fit using lmfit PseudoVoigtModel
                    comps = []
                    for comp in mfr:
                        if not isinstance(comp, dict):
                            continue
                        try:
                            A = float(comp.get("amplitude", comp.get("A", 0)) or 0.0)
                            c = float(comp.get("center", comp.get("c", np.nan)) or np.nan)
                            s_val = comp.get("sigma", comp.get("sigma_g", comp.get("sigma_l", None)))
                            s = float(s_val) if s_val is not None else np.nan
                            frac_val = comp.get("alpha", comp.get("fraction", comp.get("f", None)))
                            frac = float(frac_val) if frac_val is not None else 0.5
                        except Exception:
                            continue
                        if np.isfinite(A) and np.isfinite(c) and np.isfinite(s) and s > 0 and np.isfinite(frac):
                            comps.append((A, c, s, frac))
                    if x is not None and comps:
                        try:
                            xx = np.array(x, dtype=float)
                            model = None
                            params = None
                            for i, (A, c, s, frac) in enumerate(comps, start=1):
                                pv = PseudoVoigtModel(prefix=f"p{i}_")
                                model = pv if model is None else model + pv
                                p = pv.make_params()
                                p[f"p{i}_amplitude"].set(value=max(A, 0.0), min=0.0)
                                p[f"p{i}_center"].set(value=c)
                                p[f"p{i}_sigma"].set(value=max(s, 1e-9), min=1e-12)
                                alpha_val = min(max(frac, 0.0), 1.0)
                                p[f"p{i}_fraction"].set(value=alpha_val, min=0.0, max=1.0)
                                params = p if params is None else params.update(p)
                            yy = model.eval(params=params, x=xx)
                            y_fit = yy.tolist()
                        except Exception:
                            y_fit = None
            except Exception:
                y_fit = None
            return x, y, y_fit, row

        # Build interactive rows
        items = []
        for _, r in df_sorted.iterrows():
            idx = r.get("Index")
            fname = r.get("File Name", "")
            cond = r.get("Conditions", "")
            tval = r.get("Time", "")
            err_val = r.get("Error", np.inf)
            note = r.get("Note", "")

            label_text = f"time={tval} | cond={cond} | err={err_val:.6g} | file={fname}"
            # Ensure readable black text (some notebook themes style <code> tan)
            label = widgets.HTML(value=f"<div style='color:#000;font-family:mono'>{html.escape(label_text)}</div>")

            if np.isfinite(err_val):
                btn = widgets.Button(description="Plot", tooltip="Plot Fit vs Normalized data", layout=widgets.Layout(width="80px"))

                def _make_onclick(row_index):
                    def _handler(_b=None):
                        with plot_out:
                            plot_out.clear_output()
                            x, y, y_fit, row_obj = _get_xy_and_fit(row_index)
                            if x is None or y is None or y_fit is None:
                                print("Plot unavailable: missing or invalid Fit/Normalized data for this row.")
                                # Also note in status via session summary helper
                                _emit_function_summary(
                                    status_out,
                                    [f"Plot unavailable for index {row_index}."],
                                    title="Session Summary (Quality Check)",
                                )
                                return
                            try:
                                xx = np.asarray(x, dtype=float)
                                yy = np.asarray(y, dtype=float)
                                ff = np.asarray(y_fit, dtype=float)
                                # Align lengths if needed
                                n = min(xx.size, yy.size, ff.size)
                                xx, yy, ff = xx[:n], yy[:n], ff[:n]
                            except Exception:
                                print("Plot unavailable: data conversion error.")
                                _emit_function_summary(
                                    status_out,
                                    [f"Data conversion error for index {row_index}."],
                                    title="Session Summary (Quality Check)",
                                )
                                return
                            # Update the current plotted index for quality controls
                            try:
                                current_plot_idx["idx"] = row_index
                                refresh_quality()
                            except Exception:
                                pass
                            fig = go.FigureWidget()
                            fig.add_scatter(x=xx, y=yy, mode="lines", name="Normalized", line=dict(color="#1f77b4"))
                            fig.add_scatter(x=xx, y=ff, mode="lines", name="Fit", line=dict(color="#d62728"))
                            try:
                                title = f"Fit vs Normalized | {row_obj.get('File Name','')} (T={row_obj.get('Time','')}, {row_obj.get('Conditions','')})"
                            except Exception:
                                title = "Fit vs Normalized"
                            fig.update_layout(title=title, xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity (a.u.)", legend=dict(orientation="h"))
                            # Container: plot with "Mark spectrum as bad/good" buttons beneath
                            display(widgets.VBox([fig, widgets.HBox([mark_bad_btn, mark_good_btn])]))
                    return _handler

                btn.on_click(_make_onclick(idx))
            items.append(widgets.HBox([btn, label]))

        # If there are excluded spectra, add a button to display their details
        excluded_btn = None
        excluded_visible = {"on": False}
        if df_excluded is not None and not df_excluded.empty:
            excluded_btn = widgets.Button(
                description=f"Show excluded ({len(df_excluded)})",
                button_style="warning",
                tooltip="List spectra excluded due to missing/unusable fits",
                layout=widgets.Layout(width="220px")
            )

            def _show_excluded(_b=None):
                # Toggle: show list on first click, hide on second
                if not excluded_visible["on"]:
                    lines = []
                    try:
                        for _, rr in df_excluded.iterrows():
                            lines.append(
                                f"idx={rr.get('Index')} | time={rr.get('Time')} | cond={rr.get('Conditions')} | note={rr.get('Note','unusable')} | file='{rr.get('File Name','')}'"
                            )
                    except Exception:
                        lines = ["Excluded spectra list unavailable."]
                    # Emit excluded list in a separate output so session summary persists
                    _emit_function_summary(
                        excluded_out,
                        lines,
                        title="Excluded Spectra (Unusable)",
                    )
                    excluded_visible["on"] = True
                    try:
                        excluded_btn.description = f"Hide excluded ({len(df_excluded)})"
                        excluded_btn.button_style = "warning"
                    except Exception:
                        pass
                else:
                    # Hide by clearing the excluded area only (session summary persists)
                    try:
                        with excluded_out:
                            clear_output()
                    except Exception:
                        pass
                    excluded_visible["on"] = False
                    try:
                        excluded_btn.description = f"Show excluded ({len(df_excluded)})"
                        excluded_btn.button_style = "warning"
                    except Exception:
                        pass

            excluded_btn.on_click(_show_excluded)

        header = widgets.HTML(value=f"<b>Material:</b> {html.escape(str(material))} &nbsp; <span style='color:#555'>(click Plot to overlay)</span>")
        # Only keep the top-level Close button from check_fit_quality; no per-material Close here
        list_children = items
        if excluded_btn is not None:
            list_children = [widgets.HBox([excluded_btn])] + list_children
        list_box = widgets.VBox(list_children)
        display(widgets.VBox([header, list_box, plot_out, status_out, excluded_out]))


def display_DataFrame(FTIR_DataFrame, height: int = 500):
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    """Display the DataFrame in a scrollable table with dropdown filters.

    - Defaults to showing the entire DataFrame in a scrollable HTML table.
    - Provides dropdowns for Material, Conditions/Condition, and Time to filter.

    Parameters
    ----------
    FTIR_DataFrame : pd.DataFrame
        The DataFrame to display.
    height : int, optional
        Height in pixels for the scrollable container (default 500).

    Returns
    -------
    pd.DataFrame
        The (unchanged) input DataFrame.
    """
    # Basic validation
    if FTIR_DataFrame is None:
        raise ValueError("FTIR_DataFrame must be provided.")

    # Local imports for HTML display to avoid relying on module-level imports
    try:
        from IPython.display import display as _ip_display, HTML as _ip_HTML
    except Exception:
        _ip_display = None
        _ip_HTML = None

    # Determine the conditions column name (supports either 'Conditions' or 'Condition')
    cond_col = (
        "Conditions"
        if "Conditions" in FTIR_DataFrame.columns
        else ("Condition" if "Condition" in FTIR_DataFrame.columns else None)
    )

    # Helper to build dropdown options with an 'All' entry
    def _options_for(colname):
        try:
            vals = (
                FTIR_DataFrame.get(colname, None).dropna().astype(str).unique().tolist()
                if colname in FTIR_DataFrame.columns
                else []
            )
            # Unique, sorted, stringified
            vals = sorted({str(v) for v in vals})
        except Exception:
            vals = []
        return ["All"] + vals

    # Special case: Time options sorted in descending numeric order (then non-numeric)
    def _options_for_time():
        try:
            if "Time" not in FTIR_DataFrame.columns:
                return ["All"]
            ser = FTIR_DataFrame.get("Time", pd.Series([], dtype=object)).dropna()
            # Work with string forms for stable filtering downstream
            str_vals = list({str(v) for v in ser.astype(str).tolist()})

            def _sort_key(s):
                try:
                    num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
                except Exception:
                    num = float("nan")
                # Numeric first (ascending), then non-numeric (ascending lexicographic)
                if pd.isna(num):
                    return (1, s)
                return (0, float(num))

            sorted_vals = sorted(str_vals, key=_sort_key)
        except Exception:
            sorted_vals = []
        return ["All"] + sorted_vals

    # Build dropdowns
    mat_dd = widgets.Dropdown(
        options=_options_for("Material"),
        value="All",
        description="Material",
        layout=widgets.Layout(width="33%"),
        disabled=("Material" not in FTIR_DataFrame.columns),
    )
    cond_label = "Conditions" if cond_col != "Condition" else "Condition"
    cond_dd = widgets.Dropdown(
        options=_options_for(cond_col) if cond_col else ["All"],
        value="All",
        description=cond_label,
        layout=widgets.Layout(width="33%"),
        disabled=(cond_col is None),
    )
    time_dd = widgets.Dropdown(
        options=_options_for_time(),
        value="All",
        description="Time",
        layout=widgets.Layout(width="33%"),
        disabled=("Time" not in FTIR_DataFrame.columns),
    )
    # Quality filter dropdown (canonical column name)
    quality_dd = widgets.Dropdown(
        options=_options_for("Quality"),
        value="All",
        description="Quality",
        layout=widgets.Layout(width="33%"),
        disabled=("Quality" not in FTIR_DataFrame.columns),
    )

    # Output area for the HTML table
    out = widgets.Output()

    # Render helper
    def _render(*_):
        with out:
            out.clear_output()
            # Compose filter mask
            try:
                mask = pd.Series(True, index=FTIR_DataFrame.index)
            except Exception:
                # Fallback: no filtering if mask creation fails
                mask = None

            # Apply Material filter
            try:
                if not mat_dd.disabled and mat_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame["Material"].astype(str) == str(mat_dd.value)
                    )
            except Exception:
                pass

            # Apply Conditions/Condition filter
            try:
                if cond_col and not cond_dd.disabled and cond_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame[cond_col].astype(str) == str(cond_dd.value)
                    )
            except Exception:
                pass

            # Apply Time filter
            try:
                if not time_dd.disabled and time_dd.value != "All":
                    mask = mask & (
                        FTIR_DataFrame["Time"].astype(str) == str(time_dd.value)
                    )
            except Exception:
                pass

            # Apply Quality filter
            try:
                if not quality_dd.disabled and quality_dd.value != "All":
                    # Case-insensitive compare to be tolerant to stored values
                    mask = mask & (
                        FTIR_DataFrame["Quality"].astype(str).str.lower()
                        == str(quality_dd.value).lower()
                    )
            except Exception:
                pass

            try:
                df_view = (
                    FTIR_DataFrame[mask].copy()
                    if isinstance(mask, pd.Series)
                    else FTIR_DataFrame.copy()
                )
            except Exception:
                df_view = FTIR_DataFrame.copy()

            # Preserve and temporarily expand display options
            prev_rows = pd.get_option("display.max_rows")
            prev_cols = pd.get_option("display.max_columns")
            try:
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                html = df_view.to_html(max_rows=None, max_cols=None, notebook=True)
            finally:
                try:
                    pd.set_option("display.max_rows", prev_rows)
                    pd.set_option("display.max_columns", prev_cols)
                except Exception:
                    pass

            if _ip_display is not None and _ip_HTML is not None:
                _ip_display(
                    _ip_HTML(
                        f"<div style='height:{int(height)}px;overflow:auto;'>{html}</div>"
                    )
                )

    # Wire up events
    mat_dd.observe(_render, names="value")
    cond_dd.observe(_render, names="value")
    time_dd.observe(_render, names="value")
    quality_dd.observe(_render, names="value")

    # Action buttons
    reset_btn = widgets.Button(
        description="Reset Filters",
        button_style="warning",
        layout=widgets.Layout(width="150px"),
        tooltip="Reset all filters to 'All'",
    )
    close_btn = widgets.Button(
        description="Close",
        button_style="danger",
        layout=widgets.Layout(width="120px"),
        tooltip="Close this view",
    )

    def _on_reset(_b=None):
        # Reset dropdowns to 'All' when available and re-render
        try:
            if not mat_dd.disabled and "All" in mat_dd.options:
                mat_dd.value = "All"
        except Exception:
            pass
        try:
            if not cond_dd.disabled and "All" in cond_dd.options:
                cond_dd.value = "All"
        except Exception:
            pass
        try:
            if not time_dd.disabled and "All" in time_dd.options:
                time_dd.value = "All"
        except Exception:
            pass
        try:
            if not quality_dd.disabled and "All" in quality_dd.options:
                quality_dd.value = "All"
        except Exception:
            pass
        # Ensure a refresh even if values were already 'All'
        _render()

    def _on_close(_b=None):
        # Close all widgets and clear output area
        try:
            mat_dd.close()
        except Exception:
            pass
        try:
            cond_dd.close()
        except Exception:
            pass
        try:
            time_dd.close()
        except Exception:
            pass
        try:
            quality_dd.close()
        except Exception:
            pass
        try:
            reset_btn.close()
        except Exception:
            pass
        try:
            close_btn.close()
        except Exception:
            pass
        try:
            out.clear_output()
            out.close()
        except Exception:
            pass
            try:
                if 'ui' in locals() or 'ui' in globals():
                    ui.close()
            except Exception:
                pass

    reset_btn.on_click(_on_reset)
    close_btn.on_click(_on_close)

    # Layout and initial render
    controls = widgets.HBox([mat_dd, cond_dd, time_dd, quality_dd])
    buttons = widgets.HBox([reset_btn, close_btn])
    ui = widgets.VBox([controls, buttons, out])
    display(ui)
    _render()

    return FTIR_DataFrame


def export_material_output_csv(
    FTIR_DataFrame,
    material: str,
    *,
    materials_json_path: str | None = None,
    output_path: str | None = None,
):
    """Export a material-specific CSV with standardized columns and peak headers.

    Filename: material_output_<material>.csv (unless output_path is provided)

    Columns (in order):
    - Name, Alias, Condition, Sample Humidity, Sample Temperature, Time, Normalization Peak, Quality
    - Followed by one column per peak with header:
      'Peak <i>_name="<name>"_sigma=<σ>,alpha=<α>' (values from materials.json)

        Notes
        -----
        - Populates one CSV row per DataFrame row for the selected material.
        - Peak columns are populated from 'Material Fit Results' amplitudes; blanks when
            amplitudes are missing or unusable.
        - Uses the DataFrame's 'Normalization Peak Wavenumber' for the 'Normalization Peak'
            column as-is (string or numeric). Quality is taken from the detected quality column.
    """
    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("Error: FTIR_DataFrame not defined. Load or Create DataFrame first.")
    if material is None or str(material).strip() == "":
        raise ValueError("'material' must be a non-empty string.")

    import os, json

    # Resolve materials.json path (default: alongside this module)
    if materials_json_path is None:
        try:
            materials_json_path = os.path.join(os.path.dirname(__file__), "materials.json")
        except Exception:
            materials_json_path = "materials.json"

    # Load materials.json
    try:
        with open(materials_json_path, "r", encoding="utf-8") as f:
            _content = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load materials.json at {materials_json_path!r}: {e}")

    # Top-level structure is a list with one dict of material codes
    if not isinstance(_content, list) or not _content or not isinstance(_content[0], dict):
        raise ValueError("materials.json unexpected structure; expected a list with a single mapping object")
    _top = _content[0]

    # Find material entry by name or alias
    mat_code = None
    mat_payload = None
    target_name = str(material)
    try:
        for k, v in _top.items():
            if not isinstance(v, dict):
                continue
            if str(v.get("alias", "")) == target_name or str(v.get("name", "")) == target_name:
                mat_code = k
                mat_payload = v
                break
    except Exception:
        pass
    if mat_code is None or not isinstance(mat_payload, dict):
        raise KeyError(f"Material {material!r} not found in materials.json (by 'name' or 'alias').")

    mat_name = str(mat_payload.get("name", material))
    mat_alias = str(mat_payload.get("alias", material))
    peaks_def = mat_payload.get("peaks", {}) or {}

    # Build peak column headers from JSON (sorted by numeric key order)
    def _peak_header(idx_key: str):
        p = peaks_def.get(idx_key, {}) or {}
        pname = str(p.get("name", ""))
        # Extract numeric center; support both 'center_wavenumber' and 'center' keys
        pcenter = None
        for key in ("center_wavenumber", "center"):
            try:
                val = p.get(key, None)
                if val is not None:
                    pcenter = float(val)
                    break
            except Exception:
                pass
        if pcenter is None:
            pcenter = 0.0
        try:
            psigma = float(p.get("σ", 0))
        except Exception:
            psigma = 0.0
        try:
            palpha = float(p.get("α", 0))
        except Exception:
            palpha = 0.0
        # Title format: Peak 1_name=""_center=#,sigma=#,alpha=#
        return f"Peak{idx_key}_name={pname}_center={pcenter},sigma={psigma},alpha={palpha}"

    peak_keys_sorted = sorted(peaks_def.keys(), key=lambda s: int(str(s)) if str(s).isdigit() else str(s))
    peak_headers = [_peak_header(k) for k in peak_keys_sorted]

    # Determine columns from DataFrame
    cond_col = _conditions_column_name(FTIR_DataFrame)
    q_col = _quality_column_name(FTIR_DataFrame)
    norm_col = "Normalization Peak Wavenumber"

    # Filter rows for selected material (exact match on 'Material')
    try:
        df_mat = FTIR_DataFrame[FTIR_DataFrame.get("Material").astype(str) == str(material)]
    except Exception:
        df_mat = FTIR_DataFrame.iloc[0:0]

    # Sort rows by Time (if present)
    try:
        if "Time" in df_mat.columns:
            df_mat = df_mat.copy()
            df_mat["_sort_time"] = pd.to_numeric(df_mat["Time"], errors="coerce").fillna(float("inf"))
            df_mat = df_mat.sort_values(by=["_sort_time"], kind="mergesort")
            df_mat.drop(columns=["_sort_time"], inplace=True, errors="ignore")
    except Exception:
        pass

    # Base columns
    base_headers = [
        "Name",
        "Alias",
        "Condition",
        "Sample Humidity",
        "Sample Temperature",
        "Time",
        "Normalization Peak",
        "Quality",
    ]
    export_headers = base_headers + peak_headers

    # Helper: extract amplitudes list from a 'Material Fit Results' cell
    def _amps_from_fit_cell(cell):
        try:
            v = _safe_literal_eval(cell, value_name="Material Fit Results")
        except Exception:
            v = cell
        # Direct list of numbers (preferred modern format)
        if isinstance(v, (list, tuple)):
            try:
                return [float(x) for x in v]
            except Exception:
                return None
        # Dict-based legacy formats
        if isinstance(v, dict):
            for key in ("amplitudes", "amps", "areas", "A"):
                if key in v and isinstance(v[key], (list, tuple)):
                    try:
                        return [float(x) for x in v[key]]
                    except Exception:
                        return None
            # Sometimes stored as list of peak dicts under 'peaks'
            if isinstance(v.get("peaks"), (list, tuple)):
                vals = []
                for p in v.get("peaks", []):
                    if isinstance(p, dict):
                        a = p.get("amplitude", p.get("amp", p.get("area")))
                        try:
                            vals.append(float(a))
                        except Exception:
                            vals.append(None)
                return vals if vals else None
        return None

    # Assemble rows (populate peak columns from amplitudes where available)
    rows = []
    for _, r in df_mat.iterrows():
        try:
            row_vals = [
                mat_name,
                mat_alias,
                (r.get(cond_col) if cond_col else None),
                r.get("Sample Humidity"),
                r.get("Sample Temperature"),
                r.get("Time"),
                r.get(norm_col),
                r.get(q_col),
            ]
            # Extract amplitudes for peaks
            amps = _amps_from_fit_cell(r.get("Material Fit Results"))
            fname = r.get("File Name", "")
            # Only populate when counts match; otherwise leave blank and report
            if isinstance(amps, list) and len(amps) == len(peak_headers):
                row_vals.extend(amps)
            else:
                row_vals.extend([None] * len(peak_headers))
                try:
                    if isinstance(amps, list):
                        print(
                            f"export_material_output_csv: amplitude count {len(amps)} != peak count {len(peak_headers)} for file '{fname}'; leaving peak columns blank."
                        )
                    else:
                        print(
                            f"export_material_output_csv: missing/unreadable amplitudes in 'Material Fit Results' for file '{fname}'; leaving peak columns blank."
                        )
                except Exception:
                    pass
        except Exception:
            # On any failure, append a minimal-length row padded with None
            row_vals = [mat_name, mat_alias, None, None, None, None, None, None] + [None] * len(peak_headers)
        rows.append(row_vals)

    # Build DataFrame and write CSV
    out_df = pd.DataFrame(rows, columns=export_headers)
    # Default output: CWD/material_output_<material>.csv
    if output_path is None:
        safe_mat = str(material).replace("/", "-").replace("\\", "-")
        output_path = os.path.join(os.getcwd(), f"material_output_{safe_mat}.csv")
    try:
        out_df.to_csv(output_path, index=False)
        print(f"Exported material CSV: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to write CSV to {output_path!r}: {e}")
    return output_path


def trim_DataFrame(FTIR_DataFrame):
    """Interactively clear selected columns in `FTIR_DataFrame`.

    UI features:
    - Filter mode: select `Material`, `Conditions`, and `Time`, with an option to include bad spectra.
    - Index mode: toggle to target a single row index directly.
    - Column checklist: choose which columns to clear for the targeted rows.
    - Apply clears selected columns, Close hides the UI but keeps a session summary visible.

    Returns the updated DataFrame.
    """

    if FTIR_DataFrame is None or not isinstance(FTIR_DataFrame, pd.DataFrame):
        raise ValueError("trim_DataFrame: FTIR_DataFrame must be a valid pandas DataFrame")
    
    # Session tracking for summary output
    session_lines = []
    session_changes = {}

    # Ensure common columns exist gracefully
    _require_columns(
        FTIR_DataFrame,
        [
            "File Name",
            "Material",
            "Conditions",
            "Time",
        ],
        context="FTIR_DataFrame (trim_DataFrame)",
    )

    # Persisted defaults across tools
    defaults = _get_session_defaults()

    # Materials and Conditions lists
    materials, conditions = _extract_material_condition_lists(FTIR_DataFrame, exclude_unexposed=False) or ([], [])
    materials = sorted(materials)
    conditions = sorted(conditions)

    # Time list
    try:
        times_unique = sorted([t for t in FTIR_DataFrame["Time"].dropna().unique()])
    except Exception:
        times_unique = []

    # Safe import for display/clear_output (avoid re-importing widgets)
    try:
        from IPython.display import display, clear_output  # type: ignore
    except Exception:
        display = None  # type: ignore
        clear_output = None  # type: ignore

    # Widgets
    use_index_toggle = widgets.ToggleButton(
        value=False,
        description="Use index instead",
        layout=widgets.Layout(width="200px", margin="0 0 10px 0"),
        button_style="warning",
    )
    index_input = widgets.IntText(
        value=int(FTIR_DataFrame.index[0]) if len(FTIR_DataFrame.index) > 0 else 0,
        description="Index",
        layout=widgets.Layout(width="220px"),
    )

    material_dd = widgets.Dropdown(
        options=["any"] + materials,
        value=(defaults.get("material") if defaults.get("material") in materials else "any"),
        description="Material",
        layout=widgets.Layout(width="40%"),
    )
    conditions_dd = widgets.Dropdown(
        options=["any"] + conditions,
        value=(defaults.get("conditions") if defaults.get("conditions") in conditions else "any"),
        description="Conditions",
        layout=widgets.Layout(width="40%"),
    )
    time_dd = widgets.Dropdown(
        options=["any"] + times_unique,
        value=(defaults.get("time") if defaults.get("time") in times_unique else "any"),
        description="Time",
        layout=widgets.Layout(width="25%"),
    )
    include_bad_cb = widgets.Checkbox(value=False, description="Include bad spectra")

    # Column checklist: list all columns except core identifiers
    identifier_cols = {"File Location", "File Name", "Material", "Conditions", "Time"}
    col_options = [c for c in FTIR_DataFrame.columns if c not in identifier_cols]
    col_checks = [widgets.Checkbox(value=False, description=c) for c in col_options]
    # Make the list scrollable
    col_box = widgets.VBox(col_checks, layout=widgets.Layout(max_height="250px", overflow_y="auto", border="1px solid #ddd", padding="6px"))

    apply_btn = widgets.Button(description="Apply", button_style="success")
    close_btn = widgets.Button(description="Close", button_style="danger")

    msg_out = widgets.Output()
    # Persistent status area that remains visible after Close
    top_status_out = widgets.Output()
    # Session tracking for summary output
    session_lines = []
    session_changes = {}

    # Dynamic containers for filter vs index-only mode
    filter_row = widgets.HBox([material_dd, conditions_dd, time_dd, include_bad_cb])
    index_row = widgets.HBox([index_input])
    # Start in filter mode
    index_row.layout.display = "none"

    # Layout
    controls_top = widgets.HBox([use_index_toggle])
    controls = widgets.VBox([controls_top, filter_row, index_row])
    actions = widgets.HBox([apply_btn, close_btn])
    ui = widgets.VBox([controls, col_box, actions, msg_out])

    # Seed initial instruction in persistent area
    try:
        _emit_function_summary(
            top_status_out,
            [
                "Select filters or toggle to Index mode.",
                "Check columns to clear, then click Apply.",
                "Click Close to view session summary.",
            ],
            title="Trim DataFrame",
        )
    except Exception:
        pass

    if display is not None:
        display(ui, top_status_out)
    else:
        # Fallback: at least show UI; summary will print on close via msg_out
        try:
            print("Trim DataFrame UI loaded. Close to see summary.")
        except Exception:
            pass

    # Helper: build mask based on mode
    def _mask_for_selection():
        if use_index_toggle.value:
            # Index-only mode
            idx = index_input.value
            mask = FTIR_DataFrame.index == idx
            return mask
        # Filter mode
        mask = pd.Series([True] * len(FTIR_DataFrame), index=FTIR_DataFrame.index)
        mat = material_dd.value
        cond = conditions_dd.value
        tim = time_dd.value
        if isinstance(mat, str) and mat.strip().lower() != "any":
            mask &= FTIR_DataFrame["Material"].astype(str).str.lower() == mat.strip().lower()
        if isinstance(cond, str) and cond.strip().lower() != "any":
            # Use detected conditions column name
            cond_col = _conditions_column_name(FTIR_DataFrame) or "Conditions"
            mask &= FTIR_DataFrame[cond_col].astype(str).str.lower() == cond.strip().lower()
        if tim != "any":
            try:
                mask &= FTIR_DataFrame["Time"] == tim
            except Exception:
                pass
        if not include_bad_cb.value:
            try:
                mask &= _quality_good_mask(FTIR_DataFrame)
            except Exception:
                pass
        return mask

    # Toggle behavior
    def _on_toggle(change):
        if use_index_toggle.value:
            use_index_toggle.description = "Use filters instead"
            use_index_toggle.button_style = "info"
            filter_row.layout.display = "none"
            index_row.layout.display = ""
        else:
            use_index_toggle.description = "Use index instead"
            use_index_toggle.button_style = "warning"
            filter_row.layout.display = ""
            index_row.layout.display = "none"

    use_index_toggle.observe(_on_toggle, names="value")

    # Apply
    def _on_apply(_b=None):
        selected_cols = [cb.description for cb in col_checks if cb.value]
        with msg_out:
            msg_out.clear_output()
            if not selected_cols:
                print("No columns selected; nothing to clear.")
                return
            mask = _mask_for_selection()
            rows = FTIR_DataFrame[mask]
            if rows.empty:
                print("No rows match the current selection; nothing to clear.")
                return
            # Perform clearing: set cells to NaN
            FTIR_DataFrame.loc[mask, selected_cols] = np.nan

            # Persist session defaults when using filters
            if not use_index_toggle.value:
                _set_session_selection(material=material_dd.value, conditions=conditions_dd.value, time=(time_dd.value if time_dd.value != "any" else "any"))

            # Summary
            session_changes.setdefault("trim", []).append((len(rows), selected_cols))
            session_lines.append(
                f"Cleared {len(rows)} row(s) in columns: {', '.join(selected_cols)}"
            )
            try:
                _emit_function_summary(top_status_out, session_lines, title="Session Summary")
            except Exception:
                for line in session_lines:
                    print(line)

    apply_btn.on_click(_on_apply)

    # Close: hide UI but keep summary visible
    def _on_close(_b=None):
        # Hide interactive elements
        try:
            ui.layout.display = "none"
        except Exception:
            pass
        # Replace persistent area content with a consolidated session summary
        try:
            if clear_output is not None:
                with top_status_out:
                    clear_output()
        except Exception:
            pass
        with top_status_out:
            try:
                if session_lines:
                    _emit_function_summary(top_status_out, session_lines, title="Session Summary")
            except Exception:
                # Fallback to printing accumulated lines
                for line in session_lines:
                    print(line)

    close_btn.on_click(_on_close)

    return FTIR_DataFrame
