# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import os
import tempfile
import datetime
import json
import re
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# -------------------------
# Helpers
# -------------------------
def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_backups_dir(path="backups"):
    os.makedirs(path, exist_ok=True)
    return path

def save_backup_csv(df: pd.DataFrame, desc: str, backups_dir="backups"):
    """
    Save a timestamped CSV backup to disk and store bytes + metadata in session_state.
    Returns metadata dict.
    """
    ensure_backups_dir(backups_dir)
    ts = now_str()
    fname = f"cleaned_backup_{ts}.csv"
    path = os.path.join(backups_dir, fname)
    try:
        df.to_csv(path, index=False)
        # read bytes for download button (keeps UI independent of disk)
        with open(path, "rb") as f:
            data = f.read()
        meta = {"filename": fname, "path": path, "timestamp": ts, "desc": desc}
        # store in session state list
        if 'backups' not in st.session_state:
            st.session_state['backups'] = []
        st.session_state['backups'].append({"meta": meta, "bytes": data})
        return meta
    except Exception as e:
        st.warning(f"Backup save failed: {e}")
        return None

def ensure_df_state_persisted():
    """
    Ensure the current df state is properly persisted in session_state.
    This prevents state loss on reruns.
    """
    if st.session_state.df is not None:
        # Force a deep copy to ensure Streamlit detects the change
        st.session_state.df = st.session_state.df.copy(deep=True)
        # Update state hash for tracking
        try:
            st.session_state.last_df_state = hash(str(st.session_state.df.values.tobytes()))
        except:
            st.session_state.last_df_state = hash(str(st.session_state.df.to_csv()))

def push_history(action: str, desc: str, prev_df: pd.DataFrame):
    """
    Push a history entry that stores a copy of the previous df + metadata.
    """
    entry = {
        "id": now_str(),
        "time": datetime.datetime.now().isoformat(),
        "action": action,
        "desc": desc,
        # store CSV snapshot so restoring is easy and memory friendly
        "df_csv": prev_df.to_csv(index=False)
    }
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.session_state['history'].append(entry)

def restore_history_entry(index: int):
    """
    Restore session_state.df to the state from history[index], and trim history.
    """
    if 'history' not in st.session_state or index < 0 or index >= len(st.session_state['history']):
        st.warning("Invalid history index.")
        return
    entry = st.session_state['history'][index]
    csv_text = entry['df_csv']
    restored_df = pd.read_csv(io.StringIO(csv_text))
    # trim history to this index (keeps older steps but allows reverting only up to chosen)
    st.session_state['history'] = st.session_state['history'][:index+1]
    st.session_state.df = restored_df.copy()
    ensure_df_state_persisted()  # Ensure state is properly saved
    st.session_state.show_cleaned_data = True
    # add a backup for this restoration for safety
    save_backup_csv(st.session_state.df, desc=f"restore:{entry['id']}")
    st.success(f"Restored to history step: {entry['time']} - {entry['action']} - {entry['desc']}")

def build_ann(input_shape, problem_type, output_units):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    if problem_type == "Regression":
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    elif problem_type == "Binary Classification":
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(output_units, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def smart_clean_numeric(series):
    s_cleaned = series.astype(str).str.strip()
    # strip common units (case-insensitive)
    s_cleaned = s_cleaned.str.replace(r'\s*(kms|km|kg|lbs|miles)\b', '', flags=re.IGNORECASE, regex=True)
    # remove any non-digit except decimal and minus
    s_cleaned = s_cleaned.str.replace(r'[^\d.-]', '', regex=True)
    return pd.to_numeric(s_cleaned, errors='coerce')

def get_problem_details(y: pd.Series):
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() / max(1, len(y)) > 0.1:
            return "Regression", 1
        else:
            if y.nunique() == 2:
                return "Binary Classification", 1
            else:
                return "Multiclass Classification", y.nunique()
    else:
        if y.nunique() == 2:
            return "Binary Classification", 1
        else:
            return "Multiclass Classification", y.nunique()

# -------------------------
# App start
# -------------------------
st.set_page_config(layout="wide", page_title="AutoML App — backups & history")
st.title("AutoML App — backups & history")
st.write("Removed AI co-pilot. Added automatic CSV backups after every change + a full undo/history system.")

# Initialize session state keys
if 'df' not in st.session_state: st.session_state.df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'results' not in st.session_state: st.session_state.results = None
if 'trained_pipelines' not in st.session_state: st.session_state.trained_pipelines = {}
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'show_cleaned_data' not in st.session_state: st.session_state.show_cleaned_data = False
if 'history' not in st.session_state: st.session_state.history = []
if 'backups' not in st.session_state: st.session_state.backups = []
if 'ann_model' not in st.session_state: st.session_state.ann_model = None
if 'ann_preprocessor' not in st.session_state: st.session_state.ann_preprocessor = None
if 'tmp_ann_bytes' not in st.session_state: st.session_state.tmp_ann_bytes = None
if 'tmp_preproc_bytes' not in st.session_state: st.session_state.tmp_preproc_bytes = None
if '_busy_fill' not in st.session_state: st.session_state['_busy_fill'] = False
if 'uploaded_file_id' not in st.session_state: st.session_state.uploaded_file_id = None
if 'last_df_state' not in st.session_state: st.session_state.last_df_state = None

st.header("1) Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Only process upload if it's a NEW file (track by file ID to prevent reset on reruns)
if uploaded_file is not None:
    # Get unique file identifier (name + size) - more reliable than using tell()
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Only process if this is a different file than what we last processed
    if st.session_state.uploaded_file_id != current_file_id:
        try:
            # Reset file pointer to beginning before reading
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.show_cleaned_data = True
            # reset history & backups on fresh upload
            st.session_state.history = []
            st.session_state.backups = []
            # Track this file as processed
            st.session_state.uploaded_file_id = current_file_id
            # Store initial state hash for comparison
            try:
                st.session_state.last_df_state = hash(str(st.session_state.df.values.tobytes()))
            except:
                st.session_state.last_df_state = hash(str(st.session_state.df.to_csv()))
            # initial backup & history entry
            meta = save_backup_csv(st.session_state.df, desc="upload_initial")
            push_history(action="upload", desc="Initial upload", prev_df=st.session_state.df.copy())
            st.success("File uploaded and initial backup saved.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    # If same file, ensure df state is preserved (don't reset) - this prevents data loss on reruns
    elif st.session_state.df is None:
        # Edge case: file is same but df was lost somehow, reload
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.show_cleaned_data = True
        except Exception as e:
            st.error(f"Could not reload CSV: {e}")

if st.session_state.df is not None:
    st.subheader("Original Data (unchanged)")
    st.dataframe(st.session_state.original_df)

    st.header("2) Interactive Cleaning & History")
    st.write("Every cleaning action is recorded in history and saved as a CSV backup automatically. Use **Undo** or restore to any history step.")

    # Layout: left controls, right history/backups
    left, right = st.columns([3,1])

    with left:
        st.subheader("Cleaning actions")

        # 1. Remove rows by value
        st.markdown("**Remove rows where column equals value**")
        object_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()
        if object_cols:
            col_for_removal = st.selectbox("Choose text column", object_cols, key="rm_col")
            unique_vals = st.session_state.df[col_for_removal].astype(str).unique().tolist()
            vals_to_remove = st.multiselect("Values to remove", unique_vals, key="rm_vals")
            if st.button("Apply - Remove rows", key="apply_remove"):
                prev = st.session_state.df.copy(deep=True)
                if vals_to_remove:
                    st.session_state.df = st.session_state.df[~st.session_state.df[col_for_removal].astype(str).isin(vals_to_remove)]
                    push_history(action="remove_rows", desc=f"{col_for_removal} != {vals_to_remove}", prev_df=prev)
                    meta = save_backup_csv(st.session_state.df, desc=f"remove_rows:{col_for_removal}")
                    ensure_df_state_persisted()  # Ensure state is properly saved
                    st.session_state.show_cleaned_data = True
                    st.success(f"Removed rows where {col_for_removal} in {vals_to_remove}. Backup saved: {meta['filename']}")
                else:
                    st.info("No values selected.")
        else:
            st.info("No object (text) columns available for value-based removal.")

        st.markdown("---")

        # 2. Force-clean numeric
        st.markdown("**Force-clean text columns to numeric (strip $ , units, commas)**")
        text_cols_now = st.session_state.df.select_dtypes(include='object').columns.tolist()
        cols_to_force_clean = st.multiselect("Columns to force-clean", text_cols_now, key="force_cols")
        if st.button("Apply - Force-clean numeric", key="apply_force"):
            prev = st.session_state.df.copy(deep=True)
            cleaned = []
            for c in cols_to_force_clean:
                try:
                    st.session_state.df[c] = smart_clean_numeric(st.session_state.df[c])
                    cleaned.append(c)
                except Exception as e:
                    st.warning(f"Failed to clean {c}: {e}")
            if cleaned:
                push_history(action="force_clean", desc=f"force_clean:{cleaned}", prev_df=prev)
                meta = save_backup_csv(st.session_state.df, desc=f"force_clean:{','.join(cleaned)}")
                ensure_df_state_persisted()  # Ensure state is properly saved
                st.session_state.show_cleaned_data = True
                st.success(f"Force-cleaned {cleaned}. Backup saved: {meta['filename']}")
            else:
                st.info("No columns were cleaned.")

        st.markdown("---")

        # 3. Simple impute numeric
        st.markdown("**Impute numeric columns (simple)**")
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            impute_strategy = st.selectbox("Imputation strategy", ["Mean", "Median", "Most Frequent"], key="impute_choice")
            if st.button("Apply - Impute numeric", key="apply_impute"):
                prev = st.session_state.df.copy(deep=True)
                strat = "mean" if impute_strategy == "Mean" else ("median" if impute_strategy == "Median" else "most_frequent")
                try:
                    imputer = SimpleImputer(strategy=strat)
                    st.session_state.df[numeric_cols] = imputer.fit_transform(st.session_state.df[numeric_cols])
                    push_history(action="impute_numeric", desc=strat, prev_df=prev)
                    meta = save_backup_csv(st.session_state.df, desc=f"impute_numeric:{strat}")
                    ensure_df_state_persisted()  # Ensure state is properly saved
                    st.session_state.show_cleaned_data = True
                    st.success(f"Imputed numeric columns with {strat}. Backup: {meta['filename']}")
                except Exception as e:
                    st.error(f"Imputation failed: {e}")
        else:
            st.info("No numeric columns currently available.")

        st.markdown("---")

        # 4. Drop columns
        st.markdown("**Drop unwanted columns**")
        all_cols = st.session_state.df.columns.tolist()
        cols_to_drop = st.multiselect("Columns to drop", all_cols, key="drop_cols_ui")
        if st.button("Apply - Drop columns", key="apply_drop"):
            if cols_to_drop:
                prev = st.session_state.df.copy(deep=True)
                st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                push_history(action="drop_columns", desc=",".join(cols_to_drop), prev_df=prev)
                meta = save_backup_csv(st.session_state.df, desc=f"drop:{','.join(cols_to_drop)}")
                ensure_df_state_persisted()  # Ensure state is properly saved
                st.session_state.show_cleaned_data = True
                st.success(f"Dropped columns {cols_to_drop}. Backup: {meta['filename']}")
            else:
                st.info("No columns selected to drop.")

        st.markdown("---")

        # ---------- SUPER-SAFE Fill Nulls block (complete) ----------
        st.markdown("**Fill nulls (column-level or all) — SUPER SAFE**")
        # Build options from current df snapshot (store into local vars so reruns don't change mid-op)
        current_columns = list(st.session_state.df.columns)
        fill_cols_options = ["__ALL__"] + current_columns

        # Use unique widget keys to avoid collisions
        selected_for_fill = st.multiselect("Select columns (or choose '__ALL__')", fill_cols_options, default=[], key="fill_cols_select_safe")

        # classify columns (local copy)
        col_types_num = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        col_types_obj = st.session_state.df.select_dtypes(include='object').columns.tolist()

        fill_strategy_num = st.selectbox("Numeric strategy", ["Mean", "Median", "Mode", "Constant", "Forward", "Backward"], key="fill_num_safe")
        fill_strategy_obj = st.selectbox("Text strategy", ["Mode", "Constant", "Forward", "Backward"], key="fill_obj_safe")
        fill_constant = st.text_input("Constant value (for Constant strategy)", key="fill_constant_safe")

        # busy guard to prevent racey reruns (already inited above)
        # show quick null counts
        if st.button("Show null counts (quick)", key="show_nulls_quick"):
            null_counts = st.session_state.df.isnull().sum()
            st.write(null_counts[null_counts > 0].sort_values(ascending=False))

        if st.button("Apply - Fill nulls (super safe)", key="apply_fill_super_safe"):
            if st.session_state['_busy_fill']:
                st.info("Fill already running — wait a sec.")
            elif not selected_for_fill:
                st.info("Pick '__ALL__' or specific columns to fill.")
            else:
                try:
                    st.session_state['_busy_fill'] = True
                    # DEEP snapshot BEFORE mutation (so undo restores pre-action state)
                    prev_df = st.session_state.df.copy(deep=True)
                    prev_null_summary = prev_df.isnull().sum().to_dict()
                    prev_shape = prev_df.shape

                    # Compute actual list of columns to apply (guard against changed df)
                    cols_to_apply = prev_df.columns.tolist() if "__ALL__" in selected_for_fill else [c for c in selected_for_fill if c in prev_df.columns]

                    if not cols_to_apply:
                        st.warning("No selected columns exist in the current dataframe (it may have changed). Aborting.")
                    else:
                        applied = []
                        for col in cols_to_apply:
                            try:
                                if col in col_types_num:
                                    strat = fill_strategy_num
                                    if strat == "Mean":
                                        val = prev_df[col].mean()
                                        st.session_state.df[col] = st.session_state.df[col].fillna(val)
                                    elif strat == "Median":
                                        val = prev_df[col].median()
                                        st.session_state.df[col] = st.session_state.df[col].fillna(val)
                                    elif strat == "Mode":
                                        mode = prev_df[col].mode()
                                        if not mode.empty:
                                            st.session_state.df[col] = st.session_state.df[col].fillna(mode.iloc[0])
                                    elif strat == "Constant":
                                        try:
                                            cv = float(fill_constant)
                                            st.session_state.df[col] = st.session_state.df[col].fillna(cv)
                                        except Exception:
                                            st.warning(f"Invalid numeric constant for column {col}; skipping.")
                                            continue
                                    elif strat == "Forward":
                                        st.session_state.df[col] = st.session_state.df[col].fillna(method='ffill')
                                    elif strat == "Backward":
                                        st.session_state.df[col] = st.session_state.df[col].fillna(method='bfill')
                                    applied.append(col)
                                else:
                                    # object or other dtype
                                    strat = fill_strategy_obj
                                    if strat == "Mode":
                                        mode = prev_df[col].mode()
                                        if not mode.empty:
                                            st.session_state.df[col] = st.session_state.df[col].fillna(mode.iloc[0])
                                    elif strat == "Constant":
                                        st.session_state.df[col] = st.session_state.df[col].fillna(fill_constant)
                                    elif strat == "Forward":
                                        st.session_state.df[col] = st.session_state.df[col].fillna(method='ffill')
                                    elif strat == "Backward":
                                        st.session_state.df[col] = st.session_state.df[col].fillna(method='bfill')
                                    applied.append(col)
                            except Exception as e_col:
                                st.warning(f"Could not fill {col}: {e_col}")

                        # After applying all changes, compute after-summary
                        after_df = st.session_state.df
                        after_null_summary = after_df.isnull().sum().to_dict()
                        after_shape = after_df.shape

                        if applied:
                            # push history entry that stores the PREVIOUS snapshot so undo works
                            action_desc = f"fill_nulls:{','.join(applied)}"
                            push_history(action="fill_nulls", desc=action_desc, prev_df=prev_df)
                            meta = save_backup_csv(st.session_state.df, desc=action_desc)
                            # force UI refresh and ensure state persistence
                            ensure_df_state_persisted()  # Ensure state is properly saved
                            st.session_state.show_cleaned_data = True
                            st.success(f"Filled nulls for: {', '.join(applied)}. Backup: {meta['filename'] if meta else 'failed-to-save'}")
                            # show before/after quick summary
                            st.write("Null counts before (sample):")
                            st.write({k: v for k, v in prev_null_summary.items() if v > 0})
                            st.write("Null counts after (sample):")
                            st.write({k: v for k, v in after_null_summary.items() if v > 0})
                            st.write(f"Shape before: {prev_shape}, after: {after_shape}")
                        else:
                            st.info("Nothing applied for fill operation.")
                except Exception as e:
                    st.error(f"Fill operation failed: {e}")
                finally:
                    st.session_state['_busy_fill'] = False
        st.markdown("---")

        # 6. Manual edit: apply arbitrary pandas expression (advanced)
        st.markdown("**Advanced: Run a small pandas expression on df (use with care)**")
        st.write("Example: df['price'] = df['price'].str.replace(',', '').astype(float)")
        expr = st.text_area("Pandas snippet (use `df` as variable) — code will run inside try/except", height=120)
        if st.button("Run expression", key="run_expr"):
            if not expr.strip():
                st.info("Enter code to run.")
            else:
                prev = st.session_state.df.copy(deep=True)
                # run in a restricted/local namespace
                local_vars = {"df": st.session_state.df}
                try:
                    exec(expr, {"np": np, "pd": pd, "re": re}, local_vars)
                    if 'df' in local_vars:
                        st.session_state.df = local_vars['df'].copy()
                        push_history(action="exec_expr", desc=expr[:120], prev_df=prev)
                        meta = save_backup_csv(st.session_state.df, desc="exec_expr")
                        ensure_df_state_persisted()  # Ensure state is properly saved
                        st.session_state.show_cleaned_data = True
                        st.success(f"Expression executed. Backup: {meta['filename']}")
                    else:
                        st.error("Your expression must assign back to `df` variable.")
                except Exception as e:
                    st.error(f"Failed to execute expression: {e}")

        st.markdown("---")

        # Reset to original with confirmation checkbox
        st.markdown("**Reset**")
        if st.checkbox("Confirm: I want to reset to the original uploaded dataset", key="confirm_reset"):
            if st.button("Reset to original upload", key="reset_confirmed"):
                if st.session_state.original_df is not None:
                    prev = st.session_state.df.copy(deep=True)
                    st.session_state.df = st.session_state.original_df.copy()
                    push_history(action="reset_to_original", desc="reset to original upload", prev_df=prev)
                    meta = save_backup_csv(st.session_state.df, desc="reset_to_original")
                    ensure_df_state_persisted()  # Ensure state is properly saved
                    st.session_state.show_cleaned_data = True
                    st.success(f"Reset done. Backup saved: {meta['filename']}")
                else:
                    st.error("No original dataset available to reset to.")

    with right:
        st.subheader("History (undo / restore)")
        hist: List[Dict] = st.session_state.get('history', [])
        if hist:
            # show as list with restore buttons
            # newest last; present as numbered list
            for idx, entry in enumerate(reversed(hist)):
                real_idx = len(hist) - 1 - idx
                ent = hist[real_idx]
                t = ent['time']
                desc = ent['desc']
                action = ent['action']
                st.markdown(f"**Step {real_idx}** — {t}")
                st.write(f"- action: {action}")
                st.write(f"- desc: {desc}")
                cols = st.columns([1,1,2])
                if cols[0].button("Undo (last)", key=f"undo_{real_idx}"):
                    # Undo last action only if it's the last element
                    if real_idx == len(hist)-1:
                        # pop last history and restore previous df stored inside this entry (i.e., previous before action)
                        entry_to_restore = st.session_state['history'].pop()
                        prev_df = pd.read_csv(io.StringIO(entry_to_restore['df_csv']))
                        st.session_state.df = prev_df.copy()
                        ensure_df_state_persisted()  # Ensure state is properly saved
                        st.session_state.show_cleaned_data = True
                        save_backup_csv(st.session_state.df, desc=f"undo:{entry_to_restore['id']}")
                        st.success("Undone last action.")
                    else:
                        st.warning("You can only 'Undo (last)' on the most recent action. Use 'Restore' to jump to earlier steps.")
                if cols[1].button("Restore this step", key=f"restore_{real_idx}"):
                    # restore that specific step
                    restore_history_entry(real_idx)
                # spacer for readability
                st.write("---")
        else:
            st.info("No history yet. Actions will be recorded here.")

        st.subheader("Backups")
        bks = st.session_state.get('backups', [])
        if bks:
            st.write("Auto-saved CSV backups (most recent last)")
            for i, bk in enumerate(reversed(bks)):
                idx = len(bks)-1-i
                meta = bk['meta']
                name = meta['filename']
                ts = meta['timestamp']
                desc = meta.get('desc', '')
                st.write(f"- {name} ({ts}) - {desc}")
                st.download_button(label=f"Download {name}", data=bk['bytes'], file_name=name, key=f"bk_dn_{idx}")
        else:
            st.info("No backups yet.")

    # cleaned table view
    st.divider()
    if st.button("Show/Hide Current Cleaned Data", key="toggle_show"):
        st.session_state.show_cleaned_data = not st.session_state.show_cleaned_data

    if st.session_state.show_cleaned_data:
        st.subheader("Current Cleaned Data (will be used for training)")
        st.dataframe(st.session_state.df)
        st.write(f"Shape: {st.session_state.df.shape}")

    st.header("3) Configure Training & Models")
    # safety guard
    if st.session_state.df is None or st.session_state.df.shape[1] == 0:
        st.error("No data available. Upload or restore from history/backups.")
    else:
        # ----------------- NEW: explicit target selection form -----------------
        st.subheader("Choose target variable (Y)")
        st.write("Select a column below and click **Confirm Target**. This prevents accidental reruns/resets when you click a column name.")
        cols_for_target = list(st.session_state.df.columns)
        with st.form(key="target_form"):
            selected_candidate = st.selectbox("Pick a column as target", cols_for_target, index=cols_for_target.index(st.session_state.target_col) if st.session_state.target_col in cols_for_target else 0)
            confirm = st.form_submit_button(label="Confirm Target")
            clear_target = st.form_submit_button(label="Clear Target")
        if confirm:
            # set target only on explicit confirm (no other side-effects)
            st.session_state.target_col = selected_candidate
            st.success(f"Target set to: {st.session_state.target_col}")
        if clear_target:
            st.session_state.target_col = None
            st.info("Target cleared. Re-select if you want to train again.")

        if st.session_state.get('target_col'):
            st.write(f"Current target: **{st.session_state.target_col}**")
            # show detected problem type without modifying data
            y_preview = st.session_state.df[st.session_state.target_col]
            prob_type, out_units = get_problem_details(y_preview)
            st.info(f"Detected Problem Type (preview): **{prob_type}**")

            # prepare features preview (but do not mutate df)
            X_preview = st.session_state.df.drop(columns=[st.session_state.target_col])
            numeric_features = X_preview.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_preview.select_dtypes(include='object').columns.tolist()
            st.write(f"- Numeric features: {numeric_features if numeric_features else 'None'}")
            st.write(f"- Categorical features: {categorical_features if categorical_features else 'None'}")

            if not numeric_features and not categorical_features:
                st.error("No features detected for training.")
            else:
                st.header("Train models")
                # ---------- SAFE training block ----------
                if st.button("Start training (all models)"):
                    # quick sanity: require a target_col selected
                    if 'target_col' not in st.session_state or not st.session_state.target_col:
                        st.error("Select a target column (Confirm Target) before training.")
                    else:
                        try:
                            # 1) Make a deep local copy of the cleaned DF and operate only on it
                            df_train = st.session_state.df.copy(deep=True)

                            # 2) Save a pre-training backup (so nothing is lost and you can compare)
                            save_backup_csv(df_train, desc="pre_training_backup")

                            # 3) Prepare X, y from the local copy only (do NOT mutate st.session_state.df)
                            target_col_local = st.session_state.target_col
                            if target_col_local not in df_train.columns:
                                st.error(f"Target column '{target_col_local}' not found in data. Re-select target and try again.")
                            else:
                                X_local = df_train.drop(columns=[target_col_local])
                                y_local = df_train[target_col_local]

                                problem_type_local, output_units_local = get_problem_details(y_local)
                                st.info(f"Detected problem type: {problem_type_local}")

                                numeric_features_local = X_local.select_dtypes(include=np.number).columns.tolist()
                                categorical_features_local = X_local.select_dtypes(include='object').columns.tolist()

                                if not numeric_features_local and not categorical_features_local:
                                    st.error("No features found for training. You may have dropped all feature columns.")
                                else:
                                    models_to_train = {}
                                    if problem_type_local == "Regression":
                                        models_to_train["Linear Regression"] = LinearRegression()
                                        models_to_train["Random Forest"] = RandomForestRegressor(random_state=42)
                                        models_to_train["SVR"] = SVR()
                                        models_to_train["KNN Regressor"] = KNeighborsRegressor()
                                        models_to_train["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
                                    else:
                                        models_to_train["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
                                        models_to_train["Random Forest"] = RandomForestClassifier(random_state=42)
                                        models_to_train["SVC"] = SVC(random_state=42, probability=True)
                                        models_to_train["KNN Classifier"] = KNeighborsClassifier()
                                        models_to_train["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)

                                    # ANN toggle if you want to skip: set to False to save time
                                    train_ann = True

                                    with st.spinner("Training models... this may take a bit"):
                                        # Build preprocessors based on local features
                                        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
                                        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

                                        preprocessor_local = ColumnTransformer(
                                            transformers=[
                                                ('num', numeric_transformer, numeric_features_local),
                                                ('cat', categorical_transformer, categorical_features_local)
                                            ],
                                            remainder='passthrough'
                                        )

                                        # Train/test split on local df
                                        X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=42)

                                        results = {}
                                        trained_pipelines = {}

                                        # Train scikit-learn models (local objects only)
                                        for name, model in models_to_train.items():
                                            pipe = Pipeline(steps=[('preprocessor', preprocessor_local), ('model', model)])
                                            pipe.fit(X_train, y_train)
                                            y_pred = pipe.predict(X_test)

                                            if problem_type_local == "Regression":
                                                results[name] = {"R-squared": float(r2_score(y_test, y_pred)),
                                                                 "MSE": float(mean_squared_error(y_test, y_pred))}
                                            else:
                                                results[name] = {"Accuracy": float(accuracy_score(y_test, y_pred)),
                                                                 "Precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                                                                 "Recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0))}
                                            # store pipeline to session state for later download (ok to persist)
                                            trained_pipelines[name] = pipe

                                        # Train ANN using preprocessor copy (but still not touching st.session_state.df)
                                        if train_ann:
                                            preprocessor_ann = ColumnTransformer(
                                                transformers=[
                                                    ('num', numeric_transformer, numeric_features_local),
                                                    ('cat', categorical_transformer, categorical_features_local)
                                                ],
                                                remainder='passthrough'
                                            )
                                            # Fit-transform using training split
                                            X_train_processed = preprocessor_ann.fit_transform(X_train)
                                            X_test_processed = preprocessor_ann.transform(X_test)

                                            y_train_ann, y_test_ann = y_train.copy(), y_test.copy()
                                            if problem_type_local != "Regression":
                                                le = LabelEncoder()
                                                le.fit(pd.concat([y_train, y_test]))
                                                y_train_ann = le.transform(y_train)
                                                y_test_ann = le.transform(y_test)
                                                if problem_type_local == "Multiclass Classification":
                                                    y_train_ann = to_categorical(y_train_ann, num_classes=output_units_local)
                                                    y_test_ann = to_categorical(y_test_ann, num_classes=output_units_local)

                                            if X_train_processed.shape[1] == 0:
                                                results["ANN"] = "Skipped (no features)"
                                            else:
                                                ann_model_local = build_ann(X_train_processed.shape[1], problem_type_local, output_units_local)
                                                ann_model_local.fit(X_train_processed, y_train_ann, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

                                                # Evaluate ANN
                                                if problem_type_local == "Regression":
                                                    ann_preds = ann_model_local.predict(X_test_processed).flatten()
                                                    results["ANN"] = {"R-squared": float(r2_score(y_test_ann, ann_preds)),
                                                                      "MSE": float(mean_squared_error(y_test_ann, ann_preds))}
                                                else:
                                                    ann_preds_prob = ann_model_local.predict(X_test_processed)
                                                    if problem_type_local == "Binary Classification":
                                                        ann_preds = (ann_preds_prob > 0.5).astype(int).flatten()
                                                        y_test_eval = y_test_ann
                                                    else:
                                                        ann_preds = np.argmax(ann_preds_prob, axis=1)
                                                        y_test_eval = np.argmax(y_test_ann, axis=1)
                                                    results["ANN"] = {"Accuracy": float(accuracy_score(y_test_eval, ann_preds)),
                                                                      "Precision": float(precision_score(y_test_eval, ann_preds, average='weighted', zero_division=0)),
                                                                      "Recall": float(recall_score(y_test_eval, ann_preds, average='weighted', zero_division=0))}
                                                # SAFE: persist ANN model & its preprocessor (these are new objects)
                                                st.session_state.ann_model = ann_model_local
                                                st.session_state.ann_preprocessor = preprocessor_ann

                                        # Finally store training results & pipelines in session_state (safe)
                                        st.session_state.results = results
                                        st.session_state.trained_pipelines = trained_pipelines
                                        st.success("Training finished — models stored in session state. Your cleaned DataFrame was NOT changed.")
                        except Exception as e:
                            # DON'T mutate st.session_state.df here; only surface error
                            st.error(f"Training failed: {e}")
                            st.exception(e)
                # ---------- end SAFE training block ----------
        else:
            st.info("No target selected. Use the Confirm Target form above to lock a target for training.")

    # Results & downloads
    if st.session_state.get('results'):
        st.header("4) Results & Downloads")
        results_df = pd.DataFrame.from_dict(st.session_state.results, orient='index')
        st.dataframe(results_df.style.format(precision=4))

        st.subheader("Download scikit-learn pipelines")
        download_options = {k: v for k, v in st.session_state.trained_pipelines.items() if "ANN" not in k}
        if download_options:
            sel = st.selectbox("Select pipeline to download", list(download_options.keys()), key="dl_sel")
            if sel:
                buf = io.BytesIO()
                joblib.dump(download_options[sel], buf)
                buf.seek(0)
                st.download_button(label=f"Download {sel} pipeline (.pkl)", data=buf, file_name=f"{sel}_pipeline.pkl")
        else:
            st.info("No scikit-learn pipelines available to download.")

        st.subheader("Download ANN + preprocessor")
        if st.session_state.ann_model is not None and st.session_state.ann_preprocessor is not None:
            if st.button("Prepare ANN + preprocessor for download"):
                try:
                    # ANN .h5
                    tmp_h5 = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
                    tmp_h5_name = tmp_h5.name
                    tmp_h5.close()
                    st.session_state.ann_model.save(tmp_h5_name, save_format='h5')
                    with open(tmp_h5_name, "rb") as f:
                        st.session_state.tmp_ann_bytes = f.read()
                    # preprocessor .pkl
                    buf = io.BytesIO()
                    joblib.dump(st.session_state.ann_preprocessor, buf)
                    buf.seek(0)
                    st.session_state.tmp_preproc_bytes = buf.read()
                    st.success("Prepared ANN + preprocessor.")
                except Exception as e:
                    st.error(f"Failed to prepare ANN/preprocessor: {e}")

            if st.session_state.get('tmp_ann_bytes') is not None:
                st.download_button("Download ANN (.h5)", data=st.session_state.tmp_ann_bytes, file_name="ann_model.h5")
            if st.session_state.get('tmp_preproc_bytes') is not None:
                st.download_button("Download ANN Preprocessor (.pkl)", data=st.session_state.tmp_preproc_bytes, file_name="ann_preprocessor.pkl")
        else:
            st.info("Train ANN to enable ANN downloads.")

    # end if df exists
else:
    st.info("Upload a CSV to start.")

st.markdown("---")
st.caption("Notes: \n- Every cleaning action automatically creates a timestamped CSV backup in the `backups/` folder and a downloadable snapshot in the Backups panel.\n- History entries keep CSV snapshots; you can Undo the last action or Restore to any earlier step.\n- Keeping many large history/backups will consume RAM/disk; clear old backups in your local `backups/` folder if needed.")