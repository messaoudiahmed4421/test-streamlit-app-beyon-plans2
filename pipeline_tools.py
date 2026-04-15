"""
Real Tools module extracted from notebook for ADK agents - implements all callable functions.
"""
from __future__ import annotations

import json
from io import StringIO
from io import BytesIO
import traceback as _tb
from typing import Any, Callable, Dict, List

import pandas as pd
import numpy as _np
import networkx as nx
import re

import logging
logger = logging.getLogger(__name__)

# To support the ToolContext types used in notebook code
class FakeToolContext:
    pass
ToolContext = Any



MONTH_COLS = ["jan", "fev", "mar", "avr", "mai", "jun",
              "jul", "aou", "sep", "oct", "nov", "dec"]

A1_REQUIRED_COLS: Dict[str, List[str]] = {
    "budget":  ["code", "rubrique"] + MONTH_COLS,
    "actual":  ["code", "rubrique"] + MONTH_COLS,
    "mapping": ["code", "parent_code", "libelle", "classe", "categorie_analyse"],
}

# ── Lignes d'agrégats à exclure de l'analyse ligne-à-ligne ───────────────
AGGREGATE_CODES = {"TOTAL_CHARGES", "EBITDA", "RÉSULTAT_NET"}

MATERIALITY_THRESHOLD = 0.02

VARIANCE_THRESHOLD_PCT = 15.0

def _clean_numeric_value(val) -> float:
    """Nettoie une valeur numérique : symboles monétaires, séparateurs, NaN→0."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    s = re.sub(r'[€$£¥₹\s]', '', s)
    if ',' in s and '.' in s:
        if s.rindex(',') > s.rindex('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    s = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def _safe_repr(obj) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame({len(obj)} rows, cols={list(obj.columns)})"
        s = str(obj)
        return s[:200] if len(s) > 200 else s
    except Exception:
        return "<non-representable>"


def _gen_fn(fn_name: str, fn_body: Callable, log: list, **kwargs) -> Any:
    entry = {"function": fn_name, "args": {k: _safe_repr(v) for k, v in kwargs.items()}}
    try:
        result = fn_body(**kwargs)
        entry["status"] = "ok"
        entry["result_preview"] = _safe_repr(result)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        result = None
    log.append(entry)
    return result

def _safe_repr(obj) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame({len(obj)} rows, cols={list(obj.columns)})"
        s = str(obj)
        return s[:200] if len(s) > 200 else s
    except Exception:
        return "<non-representable>"


def _gen_fn(fn_name: str, fn_body: Callable, log: list, **kwargs) -> Any:
    entry = {"function": fn_name, "args": {k: _safe_repr(v) for k, v in kwargs.items()}}
    try:
        result = fn_body(**kwargs)
        entry["status"] = "ok"
        entry["result_preview"] = _safe_repr(result)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        result = None
    log.append(entry)
    return result

def _gen_fn(fn_name: str, fn_body: Callable, log: list, **kwargs) -> Any:
    entry = {"function": fn_name, "args": {k: _safe_repr(v) for k, v in kwargs.items()}}
    try:
        result = fn_body(**kwargs)
        entry["status"] = "ok"
        entry["result_preview"] = _safe_repr(result)
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        result = None
    log.append(entry)
    return result


# ════════════════════════════════════════════════════════════════════════════
#  TOOL 1 : analyze_pnl_variances
#  Phases 1 & 2 — Decorticage + Scoring (TOUTES les anomalies, pas de filtre)
# ════════════════════════════════════════════════════════════════════════════

def analyze_pnl_variances(tool_context: ToolContext) -> dict:
    """Analyse P&L robuste : decorticage + scoring 5 axes. AUCUN filtre.

    Retourne TOUTES les anomalies scorees avec une suggestion :
      fortement_recommande / a_evaluer / probablement_negligeable
    C est au LLM de decider RETENIR ou ECARTER chacune.
    """
    execution_log: list = []

    # ── 0. Precondition ───────────────────────────────────────────────
    a2_output = tool_context.state.get("a2_output")
    if not a2_output or a2_output.get("status") != "success":
        result = {
            "status": "error",
            "stage": "precondition_check",
            "errors": ["a2_output absent ou en erreur."],
        }
        tool_context.state["a3_output"] = result
        return result

    try:
        # ── 1. Charger DataFrames ─────────────────────────────────────
        budget_df = _gen_fn(
            "load_budget", lambda: pd.read_json(
                _SIO(a2_output["normalized_budget"]), orient="records"
            ), execution_log
        )
        actual_df = _gen_fn(
            "load_actual", lambda: pd.read_json(
                _SIO(a2_output["normalized_actual"]), orient="records"
            ), execution_log
        )
        mapping_df = _gen_fn(
            "load_mapping",
            lambda: pd.read_json(
                _SIO(a2_output.get("classified_actual", "[]")), orient="records"
            )[["code", "parent_code"]].drop_duplicates()
            if "classified_actual" in a2_output
            else pd.DataFrame(columns=["code", "parent_code"]),
            execution_log,
        )

        for df in [budget_df, actual_df]:
            df["code"] = df["code"].astype(str).str.strip()
        mapping_df["code"] = mapping_df["code"].astype(str).str.strip()
        if "parent_code" in mapping_df.columns:
            mapping_df["parent_code"] = mapping_df["parent_code"].astype(str).str.strip()

        # ── 2. Agreger totaux annuels ─────────────────────────────────
        month_cols = [c for c in MONTH_COLS if c in budget_df.columns]
        total_col = "total" if "total" in budget_df.columns else None

        def _aggregate_totals(df, label):
            if total_col and total_col in df.columns:
                agg = df.groupby("code", as_index=False)[total_col].sum()
                agg.rename(columns={total_col: f"total_{label}"}, inplace=True)
            else:
                df_m = df.copy()
                df_m["_sum"] = df_m[month_cols].sum(axis=1)
                agg = df_m.groupby("code", as_index=False)["_sum"].sum()
                agg.rename(columns={"_sum": f"total_{label}"}, inplace=True)
            return agg

        budget_agg = _gen_fn("aggregate_budget",
            lambda: _aggregate_totals(budget_df, "budget"), execution_log)
        actual_agg = _gen_fn("aggregate_actual",
            lambda: _aggregate_totals(actual_df, "actual"), execution_log)

        # ── 3. Fusionner ──────────────────────────────────────────────
        merged = _gen_fn("merge_budget_actual",
            lambda: budget_agg.merge(actual_agg, on="code", how="outer").fillna(0.0),
            execution_log)
        merged = _gen_fn("attach_hierarchy",
            lambda: merged.merge(
                mapping_df[["code", "parent_code"]].drop_duplicates(),
                on="code", how="left"),
            execution_log)

        # ── 3b. Parents manquants ─────────────────────────────────────
        a1_data = tool_context.state.get("a1_output")
        if a1_data and "normalized_mapping" in a1_data:
            full_hier = _gen_fn("load_full_hierarchy",
                lambda: pd.read_json(
                    _SIO(a1_data["normalized_mapping"]), orient="records"
                )[["code", "parent_code"]].drop_duplicates(),
                execution_log)
            full_hier["code"] = full_hier["code"].astype(str).str.strip()
            full_hier["parent_code"] = full_hier["parent_code"].astype(str).str.strip()
        else:
            full_hier = mapping_df[["code", "parent_code"]].copy()

        def _insert_missing_parents(df, hierarchy):
            out = df.copy()
            hier_map = dict(zip(
                hierarchy["code"].astype(str).str.strip(),
                hierarchy["parent_code"].astype(str).str.strip()))
            existing = set(out["code"].astype(str).str.strip())
            while True:
                referenced = set()
                for _, r in out.iterrows():
                    pc = str(r["parent_code"]).strip()
                    c = str(r["code"]).strip()
                    if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                        referenced.add(pc)
                missing = referenced - existing
                if not missing:
                    break
                new_rows = []
                for code in missing:
                    new_rows.append({
                        "code": code, "total_budget": 0.0, "total_actual": 0.0,
                        "parent_code": hier_map.get(code, code),
                        "variance": 0.0, "pct_variance": None,
                        "new_or_unplanned_activity": False,
                    })
                out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
                existing.update(missing)
            return out

        merged = _gen_fn("insert_missing_parents",
            lambda: _insert_missing_parents(merged, full_hier), execution_log)

        # ── 4. Variance feuille ───────────────────────────────────────
        def _compute_leaf_variance(df):
            out = df.copy()
            out["variance"] = out["total_actual"] - out["total_budget"]
            out["pct_variance"] = out.apply(
                lambda r: round(float(r["variance"] / r["total_budget"]) * 100, 4)
                if r["total_budget"] != 0 else None, axis=1)
            out["new_or_unplanned_activity"] = (
                (out["total_budget"] == 0) & (out["total_actual"] != 0))
            return out

        merged = _gen_fn("compute_leaf_variance",
            lambda: _compute_leaf_variance(merged), execution_log)

        # ── 5. Niveaux hierarchiques ──────────────────────────────────
        def _assign_levels(df):
            out = df.copy()
            code_to_parent = dict(zip(out["code"], out["parent_code"].fillna("")))
            def _level(code, visited=None):
                if visited is None:
                    visited = set()
                if code in visited:
                    return 0
                visited.add(code)
                parent = str(code_to_parent.get(code, "")).strip()
                if not parent or parent == code or parent.upper() in ("NAN", "NULL", "NONE", ""):
                    return 0
                return 1 + _level(parent, visited)
            out["level"] = out["code"].apply(_level)
            return out

        merged = _gen_fn("assign_hierarchy_levels",
            lambda: _assign_levels(merged), execution_log)

        # ── 6. Rollup ascendant multi-niveaux ─────────────────────────
        def _rollup(df):
            out = df.copy()
            max_level = int(out["level"].max()) if len(out) > 0 else 0
            for current_level in range(max_level, 0, -1):
                children = out[out["level"] == current_level]
                parent_sums = {}
                for _, row in children.iterrows():
                    pc = str(row["parent_code"]).strip()
                    c = str(row["code"]).strip()
                    if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                        if pc not in parent_sums:
                            parent_sums[pc] = {"budget": 0.0, "actual": 0.0}
                        parent_sums[pc]["budget"] += float(row["total_budget"])
                        parent_sums[pc]["actual"] += float(row["total_actual"])
                for idx, row in out.iterrows():
                    code = str(row["code"]).strip()
                    if code in parent_sums:
                        out.at[idx, "total_budget"] = parent_sums[code]["budget"]
                        out.at[idx, "total_actual"] = parent_sums[code]["actual"]
            out["variance"] = out["total_actual"] - out["total_budget"]
            out["pct_variance"] = out.apply(
                lambda r: round(float(r["variance"] / r["total_budget"]) * 100, 4)
                if r["total_budget"] != 0 else None, axis=1)
            out["new_or_unplanned_activity"] = (
                (out["total_budget"] == 0) & (out["total_actual"] != 0))
            return out

        merged = _gen_fn("rollup_ascendant",
            lambda: _rollup(merged), execution_log)

        # ── 7. Validation rollup ──────────────────────────────────────
        def _validate_rollup(df):
            roots = df[df["level"] == 0]
            parent_codes_set = set()
            for _, row in df.iterrows():
                pc = str(row["parent_code"]).strip()
                c = str(row["code"]).strip()
                if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                    parent_codes_set.add(pc)
            all_codes = set(df["code"].astype(str).str.strip())
            true_leaf_codes = all_codes - parent_codes_set
            true_leaves = df[df["code"].isin(true_leaf_codes)]
            root_budget = float(roots["total_budget"].sum())
            root_actual = float(roots["total_actual"].sum())
            leaf_budget = float(true_leaves["total_budget"].sum())
            leaf_actual = float(true_leaves["total_actual"].sum())
            tol = 0.01
            budget_ok = abs(root_budget - leaf_budget) < tol or len(true_leaves) == 0
            actual_ok = abs(root_actual - leaf_actual) < tol or len(true_leaves) == 0
            return {
                "root_budget": round(root_budget, 2),
                "root_actual": round(root_actual, 2),
                "true_leaf_budget": round(leaf_budget, 2),
                "true_leaf_actual": round(leaf_actual, 2),
                "root_count": len(roots),
                "true_leaf_count": len(true_leaves),
                "coherent": bool(budget_ok and actual_ok),
            }

        validation = _gen_fn("validate_rollup",
            lambda: _validate_rollup(merged), execution_log)

        if not validation.get("coherent", False):
            result = {
                "status": "error", "stage": "rollup_validation",
                "validation": validation, "execution_log": execution_log,
            }
            tool_context.state["a3_output"] = _to_native(result)
            return _to_native(result)

        # ── 8. Variance Drivers ───────────────────────────────────────
        def _compute_drivers(df):
            drivers = []
            parents_with_children = {}
            for _, row in df.iterrows():
                pc = str(row["parent_code"]).strip()
                c = str(row["code"]).strip()
                if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                    parents_with_children.setdefault(pc, []).append(c)
            for parent_code, children in parents_with_children.items():
                parent_row = df[df["code"] == parent_code]
                if parent_row.empty:
                    continue
                parent_variance = float(parent_row.iloc[0]["variance"])
                if parent_variance == 0:
                    continue
                for child_code in children:
                    child_row = df[df["code"] == child_code]
                    if child_row.empty:
                        continue
                    child_variance = float(child_row.iloc[0]["variance"])
                    contribution = round(
                        (child_variance / parent_variance) * 100, 4
                    ) if parent_variance != 0 else 0.0
                    abs_c = abs(contribution)
                    if abs_c >= 50:
                        role = "primary"
                    elif contribution < 0:
                        role = "compensating"
                    elif abs_c < 10:
                        role = "marginal"
                    else:
                        role = "primary"
                    drivers.append({
                        "child_code": child_code,
                        "parent_code": parent_code,
                        "child_variance": round(child_variance, 2),
                        "parent_variance": round(parent_variance, 2),
                        "variance_contribution": round(contribution, 4),
                        "driver_role": role,
                    })
            return drivers

        drivers = _gen_fn("compute_variance_drivers",
            lambda: _compute_drivers(merged), execution_log)

        # ── 9. Variance tree ──────────────────────────────────────────
        def _build_tree(df):
            tree = []
            for _, row in df.iterrows():
                tree.append({
                    "code": str(row["code"]),
                    "parent_code": str(row.get("parent_code", "")),
                    "level": int(row.get("level", 0)),
                    "total_budget": round(float(row["total_budget"]), 2),
                    "total_actual": round(float(row["total_actual"]), 2),
                    "variance": round(float(row["variance"]), 2),
                    "pct_variance": round(float(row["pct_variance"]), 4)
                    if pd.notna(row.get("pct_variance")) else None,
                    "new_or_unplanned_activity": bool(
                        row.get("new_or_unplanned_activity", False)),
                })
            return tree

        variance_tree = _gen_fn("build_variance_tree",
            lambda: _build_tree(merged), execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── 10. Ecarts mensuels ───────────────────────────────────────
        # ══════════════════════════════════════════════════════════════

        def _compute_monthly_variances(budget_raw, actual_raw, mcols):
            monthly_vars = []
            if not mcols:
                return monthly_vars
            b = budget_raw.copy()
            a = actual_raw.copy()
            b["code"] = b["code"].astype(str).str.strip()
            a["code"] = a["code"].astype(str).str.strip()
            codes = set(b["code"]) | set(a["code"])
            for code in codes:
                b_row = b[b["code"] == code]
                a_row = a[a["code"] == code]
                if b_row.empty and a_row.empty:
                    continue
                for m in mcols:
                    bv = float(b_row[m].iloc[0]) if not b_row.empty and m in b_row.columns else 0.0
                    av = float(a_row[m].iloc[0]) if not a_row.empty and m in a_row.columns else 0.0
                    v = av - bv
                    pct = round((v / bv) * 100, 2) if bv != 0 else None
                    monthly_vars.append({
                        "code": code, "month": m,
                        "budget": round(bv, 2), "actual": round(av, 2),
                        "variance": round(v, 2), "pct_variance": pct,
                    })
            return monthly_vars

        monthly_variances = _gen_fn("compute_monthly_variances",
            lambda: _compute_monthly_variances(budget_df, actual_df, month_cols),
            execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── 11. Detection anomalies (5 types) ─────────────────────────
        # ══════════════════════════════════════════════════════════════

        def _detect_anomalies(budget_raw, actual_raw, mcols):
            anomalies = []
            if not mcols:
                return anomalies
            b = budget_raw.copy()
            a = actual_raw.copy()
            b["code"] = b["code"].astype(str).str.strip()
            a["code"] = a["code"].astype(str).str.strip()
            codes = set(b["code"]) | set(a["code"])
            for code in codes:
                b_row = b[b["code"] == code]
                a_row = a[a["code"] == code]
                if b_row.empty and a_row.empty:
                    continue
                budget_vals = []
                actual_vals = []
                for m in mcols:
                    bv = float(b_row[m].iloc[0]) if not b_row.empty and m in b_row.columns else 0.0
                    av = float(a_row[m].iloc[0]) if not a_row.empty and m in a_row.columns else 0.0
                    budget_vals.append(bv)
                    actual_vals.append(av)
                budget_arr = _np.array(budget_vals)
                actual_arr = _np.array(actual_vals)
                var_arr = actual_arr - budget_arr

                # 1. Spikes mensuels (> 20%)
                for i, m in enumerate(mcols):
                    if budget_arr[i] != 0:
                        pct = (var_arr[i] / budget_arr[i]) * 100
                        if abs(pct) > 20:
                            anomalies.append({
                                "code": code, "type": "monthly_spike", "month": m,
                                "budget": round(float(budget_arr[i]), 2),
                                "actual": round(float(actual_arr[i]), 2),
                                "variance": round(float(var_arr[i]), 2),
                                "pct": round(float(pct), 2),
                                "severity": "high" if abs(pct) > 50 else "medium",
                            })

                # 2. Tendance (regression)
                if len(actual_arr) >= 6 and _np.std(actual_arr) > 0:
                    x = _np.arange(len(actual_arr))
                    slope = float(_np.polyfit(x, actual_arr, 1)[0])
                    mean_val = float(_np.mean(actual_arr))
                    if mean_val != 0 and abs(slope * 12 / mean_val) > 0.15:
                        anomalies.append({
                            "code": code, "type": "trend",
                            "direction": "increasing" if slope > 0 else "decreasing",
                            "monthly_slope": round(slope, 2),
                            "annual_drift_pct": round(float(slope * 12 / mean_val * 100), 2),
                            "severity": "high" if abs(slope * 12 / mean_val) > 0.30 else "medium",
                        })

                # 3. Volatilite (CV > 30%)
                if _np.mean(actual_arr) != 0:
                    cv = float(_np.std(actual_arr) / abs(_np.mean(actual_arr)) * 100)
                    if cv > 30:
                        anomalies.append({
                            "code": code, "type": "high_volatility",
                            "cv_pct": round(cv, 2),
                            "min_month": mcols[int(_np.argmin(actual_arr))],
                            "max_month": mcols[int(_np.argmax(actual_arr))],
                            "min_val": round(float(_np.min(actual_arr)), 2),
                            "max_val": round(float(_np.max(actual_arr)), 2),
                            "severity": "high" if cv > 50 else "medium",
                        })

                # 4. Mois non budgete
                for i, m in enumerate(mcols):
                    if budget_arr[i] == 0 and actual_arr[i] != 0:
                        anomalies.append({
                            "code": code, "type": "unbudgeted_month", "month": m,
                            "actual": round(float(actual_arr[i]), 2),
                            "severity": "medium",
                        })

                # 5. Inversion de signe
                for i, m in enumerate(mcols):
                    if budget_arr[i] != 0 and actual_arr[i] != 0:
                        if (budget_arr[i] > 0) != (actual_arr[i] > 0):
                            anomalies.append({
                                "code": code, "type": "sign_reversal", "month": m,
                                "budget": round(float(budget_arr[i]), 2),
                                "actual": round(float(actual_arr[i]), 2),
                                "severity": "high",
                            })
            return anomalies

        monthly_anomalies = _gen_fn("detect_monthly_anomalies",
            lambda: _detect_anomalies(budget_df, actual_df, month_cols),
            execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   P H A S E   1  :  D E C O R T I C A G E   S Y S T E M A T I Q U E
        # ══════════════════════════════════════════════════════════════════

        def _build_monthly_profiles(monthly_vars, mcols):
            profiles = {}
            for mv in monthly_vars:
                code = mv["code"]
                if code not in profiles:
                    profiles[code] = {
                        "months_with_variance": 0,
                        "variances": [],
                        "actuals": [],
                        "pct_variances": [],
                    }
                if mv["variance"] != 0:
                    profiles[code]["months_with_variance"] += 1
                profiles[code]["variances"].append(mv["variance"])
                profiles[code]["actuals"].append(mv["actual"])
                if mv["pct_variance"] is not None:
                    profiles[code]["pct_variances"].append(mv["pct_variance"])
            for code, p in profiles.items():
                vals = _np.array(p["actuals"])
                if len(vals) >= 6 and _np.std(vals) > 0:
                    x = _np.arange(len(vals))
                    slope = float(_np.polyfit(x, vals, 1)[0])
                    mean_val = float(_np.mean(vals))
                    p["slope"] = slope
                    p["pct_drift"] = round(slope * 12 / mean_val * 100, 2) if mean_val != 0 else 0
                else:
                    p["slope"] = 0
                    p["pct_drift"] = 0
                big_months = sum(1 for pv in p["pct_variances"] if abs(pv) > 20)
                p["months_significant"] = big_months
            return profiles

        profiles = _gen_fn("build_monthly_profiles",
            lambda: _build_monthly_profiles(monthly_variances, month_cols),
            execution_log)

        NATURE_MAP = {
            "monthly_spike":    "temporelle",
            "trend":            "comportementale",
            "high_volatility":  "comportementale",
            "unbudgeted_month": "temporelle",
            "sign_reversal":    "donnee",
            "annual_variance":  "structurelle",
        }
        ORIGINE_MAP = {
            "unbudgeted_month": "planification",
            "sign_reversal":    "erreur_potentielle",
            "trend":            "operationnel",
            "high_volatility":  "operationnel",
            "monthly_spike":    "saisonnier",
        }

        def _classify_origine(a):
            atype = a.get("type", "annual_variance")
            if atype in ORIGINE_MAP:
                return ORIGINE_MAP[atype]
            pct = abs(a.get("pct_variance", 0))
            if pct > 100:
                return "planification"
            return "operationnel"

        def _classify_frequence(code, profiles_dict, atype):
            if atype == "annual_variance":
                return "annuelle"
            p = profiles_dict.get(code, {})
            n = p.get("months_with_variance", 0)
            if n <= 1:
                return "ponctuelle"
            elif n <= 3:
                return "occasionnelle"
            else:
                return "recurrente"

        def _classify_tendance(code, profiles_dict, atype):
            if atype in ("sign_reversal", "unbudgeted_month"):
                return "indeterminee"
            p = profiles_dict.get(code, {})
            drift = p.get("pct_drift", 0)
            if abs(drift) < 5:
                return "stable"
            elif drift > 15:
                return "croissante"
            elif drift > 0:
                return "legerement_croissante"
            elif drift < -15:
                return "decroissante"
            else:
                return "legerement_decroissante"

        def _classify_portee(level):
            if level == 0:
                return "racine"
            elif level == 1:
                return "intermediaire"
            else:
                return "feuille"

        def _phase1_decorticage(vtree, monthly_anoms, profiles_dict, merged_df):
            unified = []
            for v in vtree:
                pct = v.get("pct_variance")
                if pct is None or pct == 0:
                    continue
                code = v["code"]
                level = v["level"]
                a = {
                    "source": "annual",
                    "type": "annual_variance",
                    "code": code,
                    "level": level,
                    "parent_code": v["parent_code"],
                    "total_budget": v["total_budget"],
                    "total_actual": v["total_actual"],
                    "variance": v["variance"],
                    "pct_variance": round(pct, 2),
                    "new_or_unplanned": v.get("new_or_unplanned_activity", False),
                    "nature":    NATURE_MAP["annual_variance"],
                    "origine":   _classify_origine({"type": "annual_variance", "pct_variance": pct}),
                    "frequence": _classify_frequence(code, profiles_dict, "annual_variance"),
                    "tendance":  _classify_tendance(code, profiles_dict, "annual_variance"),
                    "portee":    _classify_portee(level),
                }
                unified.append(a)
            for ma in monthly_anoms:
                code = ma.get("code", "?")
                a = {**ma, "source": "monthly"}
                match = merged_df[merged_df["code"] == code]
                if not match.empty:
                    a["level"] = int(match.iloc[0]["level"])
                    a["parent_code"] = str(match.iloc[0]["parent_code"])
                else:
                    a["level"] = 2
                    a["parent_code"] = ""
                atype = a.get("type", "")
                level = a["level"]
                a["nature"]    = NATURE_MAP.get(atype, "donnee")
                a["origine"]   = _classify_origine(a)
                a["frequence"] = _classify_frequence(code, profiles_dict, atype)
                a["tendance"]  = _classify_tendance(code, profiles_dict, atype)
                a["portee"]    = _classify_portee(level)
                unified.append(a)
            return unified

        all_anomalies = _gen_fn("phase1_decorticage",
            lambda: _phase1_decorticage(
                variance_tree, monthly_anomalies, profiles, merged),
            execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   P H A S E   2  :  S C O R I N G   5   A X E S  (1-100)
        #              PAS DE SEUIL — SUGGESTION UNIQUEMENT
        # ══════════════════════════════════════════════════════════════════

        total_budget_abs = float(abs(merged["total_budget"].sum())) if len(merged) > 0 else 1.0
        if total_budget_abs == 0:
            total_budget_abs = 1.0

        def _phase2_scoring(anomalies, total_budget_abs_val):
            TYPE_URGENCE_BASE = {
                "sign_reversal": 18, "monthly_spike": 15,
                "trend": 12, "high_volatility": 10,
                "annual_variance": 10, "unbudgeted_month": 8,
            }
            FREQ_SCORES = {
                "recurrente": 15, "occasionnelle": 10,
                "annuelle": 8, "ponctuelle": 5,
            }
            TENDANCE_SCORES = {
                "croissante": 15, "legerement_croissante": 10,
                "stable": 7, "indeterminee": 7,
                "legerement_decroissante": 4, "decroissante": 2,
            }
            PORTEE_SCORES = {"racine": 15, "intermediaire": 10, "feuille": 5}

            scored = []
            for a in anomalies:
                atype = a.get("type", "annual_variance")
                sev = a.get("severity", "medium")

                # Pilier 1 : Impact Financier (0-30)
                p1 = 0.0
                abs_var = abs(a.get("variance", 0))
                abs_pct = abs(a.get("pct_variance",
                    a.get("pct", a.get("annual_drift_pct", a.get("cv_pct", 0)))))
                if total_budget_abs_val > 0:
                    p1 += min(round(abs_var / total_budget_abs_val * 100 * 3, 1), 15)
                p1 += min(round(abs_pct / 10, 1), 10)
                if a.get("new_or_unplanned") or atype == "unbudgeted_month":
                    p1 += 5
                p1 = round(min(p1, 30), 1)

                # Pilier 2 : Urgence (0-25)
                p2 = float(TYPE_URGENCE_BASE.get(atype, 10))
                if sev == "high":
                    p2 += 5
                elif sev == "medium":
                    p2 += 2
                p2 = round(min(p2, 25), 1)

                # Pilier 3 : Frequence (0-15)
                p3 = float(FREQ_SCORES.get(a.get("frequence", "ponctuelle"), 5))

                # Pilier 4 : Tendance (0-15)
                p4 = float(TENDANCE_SCORES.get(a.get("tendance", "stable"), 7))

                # Pilier 5 : Portee (0-15)
                p5 = float(PORTEE_SCORES.get(a.get("portee", "feuille"), 5))

                total = round(min(p1 + p2 + p3 + p4 + p5, 100))

                if total >= 80:
                    niveau = "critique"
                elif total >= 60:
                    niveau = "majeur"
                else:
                    niveau = "mineur"

                # SUGGESTION (guidance pour le LLM, PAS un filtre)
                if total >= 65:
                    suggestion = "fortement_recommande"
                elif total >= 40:
                    suggestion = "a_evaluer"
                else:
                    suggestion = "probablement_negligeable"

                a["scoring"] = {
                    "impact_financier": p1,
                    "urgence": p2,
                    "frequence": p3,
                    "tendance": p4,
                    "portee": p5,
                }
                a["score"] = total
                a["niveau"] = niveau
                a["suggestion"] = suggestion
                scored.append(a)

            return sorted(scored, key=lambda x: x["score"], reverse=True)

        all_scored = _gen_fn("phase2_scoring",
            lambda: _phase2_scoring(all_anomalies, total_budget_abs),
            execution_log)

        # Stats de scoring (pas de filtre — juste des stats)
        scoring_stats = _gen_fn("compute_scoring_stats", lambda: {
            "total_scored": len(all_scored),
            "fortement_recommande": sum(
                1 for a in all_scored if a["suggestion"] == "fortement_recommande"),
            "a_evaluer": sum(
                1 for a in all_scored if a["suggestion"] == "a_evaluer"),
            "probablement_negligeable": sum(
                1 for a in all_scored if a["suggestion"] == "probablement_negligeable"),
            "score_max": max((a["score"] for a in all_scored), default=0),
            "score_min": min((a["score"] for a in all_scored), default=0),
            "score_mean": round(
                sum(a["score"] for a in all_scored) / max(len(all_scored), 1), 1),
            "critiques": sum(1 for a in all_scored if a["niveau"] == "critique"),
            "majeurs": sum(1 for a in all_scored if a["niveau"] == "majeur"),
            "mineurs": sum(1 for a in all_scored if a["niveau"] == "mineur"),
        }, execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   FORMAT TOUTES les anomalies en cartes (le LLM triera)
        # ══════════════════════════════════════════════════════════════════

        def _format_all_cards(scored_list):
            cards = []
            for idx, a in enumerate(scored_list, 1):
                anomalie_id = f"ANM-{idx:03d}"
                code = a.get("code", "?")
                source = a.get("source", "?")
                atype = a.get("type", "annual_variance")
                score = a["score"]
                niveau = a["niveau"]
                scoring = a.get("scoring", {})
                suggestion = a.get("suggestion", "a_evaluer")

                if source == "annual":
                    pct = a.get("pct_variance", 0)
                    var = a.get("variance", 0)
                    resume = (
                        f"Compte {code} : ecart annuel de "
                        f"{var:+,.0f} ({pct:+.1f}%) vs budget"
                    )
                elif atype == "monthly_spike":
                    resume = (
                        f"Compte {code} : spike {a.get('month','')} "
                        f"({a.get('pct', 0):+.1f}%)"
                    )
                elif atype == "trend":
                    resume = (
                        f"Compte {code} : tendance {a.get('direction','')} "
                        f"(drift {a.get('annual_drift_pct', 0):+.1f}%/an)"
                    )
                elif atype == "high_volatility":
                    resume = (
                        f"Compte {code} : volatilite excessive "
                        f"(CV={a.get('cv_pct', 0):.0f}%)"
                    )
                elif atype == "sign_reversal":
                    resume = f"Compte {code} : inversion de signe {a.get('month','')}"
                elif atype == "unbudgeted_month":
                    resume = (
                        f"Compte {code} : activite non budgetee "
                        f"{a.get('month','')} ({a.get('actual', 0):,.0f})"
                    )
                else:
                    resume = f"Compte {code} : anomalie {atype}"

                top_pilier = max(
                    scoring.items(), key=lambda x: x[1]
                ) if scoring else ("?", 0)

                cards.append({
                    "anomalie_id": anomalie_id,
                    "code": code,
                    "source": source,
                    "type": atype,
                    "resume": resume,
                    "score": score,
                    "niveau": niveau,
                    "suggestion": suggestion,
                    "nature": a.get("nature", "?"),
                    "origine": a.get("origine", "?"),
                    "frequence": a.get("frequence", "?"),
                    "tendance": a.get("tendance", "?"),
                    "portee": a.get("portee", "?"),
                    "scoring_detail": scoring,
                    "pilier_dominant": f"{top_pilier[0]} ({top_pilier[1]:.0f}pts)",
                    "donnees": {
                        k: v for k, v in a.items()
                        if k not in (
                            "score", "scoring", "niveau", "source", "suggestion",
                            "nature", "origine", "frequence", "tendance", "portee"
                        )
                    },
                })
            return cards

        all_cards = _gen_fn("format_all_cards",
            lambda: _format_all_cards(all_scored), execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── RESULTAT (TOUTES les cartes, pas de filtre) ───────────────
        # ══════════════════════════════════════════════════════════════
        stats = {
            "total_accounts": len(variance_tree),
            "root_accounts": sum(1 for v in variance_tree if v["level"] == 0),
            "leaf_accounts": sum(1 for v in variance_tree if v["level"] > 0),
            "drivers_count": len(drivers),
            "unplanned_count": sum(
                1 for v in variance_tree if v["new_or_unplanned_activity"]),
            "total_anomalies_scored": len(all_scored),
            "functions_executed": len(execution_log),
        }

        result = {
            "status": "success",
            "stage": "anomaly_analysis",
            "all_anomaly_cards": all_cards,
            "scoring_stats": scoring_stats,
            "variance_tree": variance_tree,
            "drivers": drivers,
            "monthly_variances": monthly_variances,
            "validation": validation,
            "execution_log": execution_log,
            "stats": stats,
            "triage_status": "pending",  # sera mis a jour par save_triage_decisions
        }

        tool_context.state["a3_output"] = _to_native(result)
        logger.info(
            "A3 Phase 1+2 OK : %d comptes, %d anomalies scorees "
            "(%d recomm, %d a_eval, %d negli), %d fns — TRIAGE LLM EN ATTENTE",
            len(variance_tree), len(all_scored),
            scoring_stats.get("fortement_recommande", 0),
            scoring_stats.get("a_evaluer", 0),
            scoring_stats.get("probablement_negligeable", 0),
            len(execution_log),
        )

        # Retourner un resume compact pour le LLM (pas toutes les cartes
        # detaillees — seulement les fortement_recommande + a_evaluer
        # en detail, et un resume des negligeables)
        cards_for_review = [
            c for c in all_cards if c["suggestion"] != "probablement_negligeable"
        ]
        negligeable_summary = {
            "count": sum(1 for c in all_cards if c["suggestion"] == "probablement_negligeable"),
            "score_range": f"0-39",
            "examples": [
                {"id": c["anomalie_id"], "code": c["code"], "score": c["score"],
                 "resume": c["resume"][:50]}
                for c in all_cards if c["suggestion"] == "probablement_negligeable"
            ][:5],  # max 5 exemples
        }

        return _to_native({
            "status": "success",
            "stage": "anomaly_analysis_phase1_2",
            "stats": stats,
            "scoring_stats": scoring_stats,
            "validation": validation,
            "cards_to_review": cards_for_review,
            "negligeable_summary": negligeable_summary,
            "message": (
                f"Phase 1+2 terminee. {len(all_scored)} anomalies scorees. "
                f"{scoring_stats.get('fortement_recommande', 0)} fortement recommandees, "
                f"{scoring_stats.get('a_evaluer', 0)} a evaluer, "
                f"{scoring_stats.get('probablement_negligeable', 0)} negligeables. "
                f"APPELLE save_triage_decisions avec tes verdicts."
            ),
        })

    except Exception as exc:
        logger.error("A3 exception : %s\n%s", exc, _tb.format_exc())
        result = {
            "status": "error",
            "stage": "anomaly_analysis",
            "errors": [f"Exception inattendue : {exc}"],
            "execution_log": execution_log,
        }
        tool_context.state["a3_output"] = _to_native(result)
        return _to_native(result)

def normalize_pnl_files(tool_context: ToolContext) -> dict:
    """Charge, valide et nettoie les 3 fichiers CSV P&L.

    Fichiers attendus :
    - budget_previsionnel.csv  : Code, Rubrique, Jan..Dec, Total
    - compte_resultat_reel.csv : Code, Rubrique, Jan..Dec, Total
    - chart_of_accounts.csv    : Code, Parent_Code, Libelle, Classe, Categorie_Analyse

    Tâches :
    1. Valider la présence des colonnes requises dans chaque CSV
    2. Forcer les colonnes Code en str
    3. Nettoyer les 12 colonnes mensuelles (symboles, séparateurs, float, NaN→0)
    4. Exclure les lignes d'agrégats (TOTAL_CHARGES, EBITDA, RÉSULTAT_NET)
    5. Valider la hiérarchie Mapping : duplicats, Parent_Code orphelins, cycles

    Returns:
        dict avec status='success' ou 'error' + détails
    """
    errors: List[str] = []

    # ── 1. Charger les CSV ────────────────────────────────────────────
    files = {
        "budget":  None,
        "actual":  None,
        "mapping": None,
    }
    
    uploaded_files = tool_context.state.get("briefing", {}).get("uploaded_files", [])
    
    # Classify uploads
    for f in uploaded_files:
        lower = str(f.name).lower()
        if any(k in lower for k in ["budget", "prevision", "forecast"]):
            files["budget"] = f
        elif any(k in lower for k in ["actual", "real", "resultat", "reel"]):
            files["actual"] = f
        elif any(k in lower for k in ["mapping", "chart", "accounts", "coa"]):
            files["mapping"] = f

    dfs: Dict[str, Any] = {}

    for name, f in files.items():
        if f is None:
            if name != "mapping":
                errors.append(f"Fichier manquant (upload) : {name}")
            continue
            
        try:
            content = f.getvalue()
            fname_lower = str(f.name).lower()
            if fname_lower.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif fname_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            else:
                df = pd.read_csv(BytesIO(content))
                
            df.columns = [str(c).strip().lower() for c in df.columns]
            dfs[name] = df
        except Exception as exc:
            errors.append(f"Erreur lecture {name} ({f.name}) : {exc}")

    if "budget" not in dfs and "actual" in dfs:
        dfs["budget"] = dfs["actual"].copy()
    if "actual" not in dfs and "budget" in dfs:
        dfs["actual"] = dfs["budget"].copy()
    if "mapping" not in dfs:
        dfs["mapping"] = pd.DataFrame(columns=A1_REQUIRED_COLS["mapping"])

    budget_df = dfs["budget"]
    actual_df = dfs["actual"]
    mapping_df = dfs["mapping"]

    # ── 2. Valider les colonnes requises ──────────────────────────────
    for file_name, df in dfs.items():
        required = A1_REQUIRED_COLS[file_name]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(
                f"Colonnes manquantes dans {file_name} : {missing}. "
                f"Colonnes trouvées : {list(df.columns)}"
            )

    if errors:
        result = {"status": "error", "stage": "structural_validation", "errors": errors}
        tool_context.state["a1_output"] = result
        return result

    # ── 3. Forcer Code en str ─────────────────────────────────────────
    for name in dfs:
        dfs[name] = dfs[name].copy()
        dfs[name]["code"] = dfs[name]["code"].astype(str).str.strip()

    budget_df = dfs["budget"]
    actual_df = dfs["actual"]
    mapping_df = dfs["mapping"]

    # ── 4. Exclure les lignes d'agrégats ──────────────────────────────
    budget_agg = budget_df[budget_df["code"].isin(AGGREGATE_CODES)]
    actual_agg = actual_df[actual_df["code"].isin(AGGREGATE_CODES)]
    budget_df = budget_df[~budget_df["code"].isin(AGGREGATE_CODES)].copy()
    actual_df = actual_df[~actual_df["code"].isin(AGGREGATE_CODES)].copy()

    logger.info("A1 — agrégats exclus : %d budget, %d actual",
                 len(budget_agg), len(actual_agg))

    # ── 5. Nettoyer les 12 colonnes mensuelles + Total ────────────────
    numeric_cols = MONTH_COLS + (["total"] if "total" in budget_df.columns else [])

    for col in numeric_cols:
        if col in budget_df.columns:
            budget_df[col] = budget_df[col].apply(_clean_numeric_value)
            budget_df[col] = budget_df[col].fillna(0.0)
        if col in actual_df.columns:
            actual_df[col] = actual_df[col].apply(_clean_numeric_value)
            actual_df[col] = actual_df[col].fillna(0.0)

    # ── 6. Valider la hiérarchie Mapping ──────────────────────────────
    #  6a. Codes dupliqués
    if "code" in mapping_df.columns:
        dupes = mapping_df[mapping_df["code"].duplicated(keep=False)]
        if not dupes.empty:
            dupe_codes = dupes["code"].unique().tolist()
            errors.append(f"Codes dupliqués dans chart_of_accounts : {dupe_codes}")

    #  6b-6d. Hiérarchie Parent_Code
    if "parent_code" in mapping_df.columns:
        all_codes = set(mapping_df["code"].astype(str).values)
        parent_col = mapping_df["parent_code"].copy()
        # Traiter NULL/nan comme absence de parent
        parent_col = parent_col.fillna("").astype(str).str.strip().str.upper()
        parent_codes = [pc for pc in parent_col if pc and pc != "" and pc != "NULL" and pc != "NAN"]

        orphans = [pc for pc in parent_codes if pc.lower() not in {c.lower() for c in all_codes}]
        if orphans:
            errors.append(f"Parent_Code orphelins (parent inexistant) : {list(set(orphans))}")

        G = nx.DiGraph()
        for _, row in mapping_df.iterrows():
            code = str(row["code"]).strip()
            raw_parent = str(row["parent_code"]).strip().upper() if pd.notna(row.get("parent_code")) else ""
            G.add_node(code)
            # Ignorer les self-loops (comptes racines où Parent_Code == Code)
            if raw_parent and raw_parent not in ("", "NULL", "NAN", "NONE") and raw_parent.lower() != code.lower():
                G.add_edge(raw_parent, code)

        cycles = list(nx.simple_cycles(G))
        if cycles:
            errors.append(f"Cycles détectés dans la hiérarchie : {cycles}")

        logger.info("A1 — graphe hiérarchique : %d nœuds, %d arêtes, %d cycles",
                     G.number_of_nodes(), G.number_of_edges(), len(cycles))

    if errors:
        result = {"status": "error", "stage": "structural_validation", "errors": errors}
        tool_context.state["a1_output"] = result
        return result

    # ── 7. Construire le résultat success ─────────────────────────────
    result = {
        "status": "success",
        "stage": "structural_validation",
        "data_summary": {
            "budget_rows": len(budget_df),
            "budget_columns": list(budget_df.columns),
            "actual_rows": len(actual_df),
            "actual_columns": list(actual_df.columns),
            "mapping_rows": len(mapping_df),
            "mapping_columns": list(mapping_df.columns),
            "excluded_aggregates": list(AGGREGATE_CODES),
            "month_columns": MONTH_COLS,
        },
        "normalized_budget": budget_df.to_json(orient="records", force_ascii=False),
        "normalized_actual": actual_df.to_json(orient="records", force_ascii=False),
        "normalized_mapping": mapping_df.to_json(orient="records", force_ascii=False),
    }

    tool_context.state["a1_output"] = result
    logger.info("A1 — validation OK (%d budget, %d actual, %d mapping)",
                 len(budget_df), len(actual_df), len(mapping_df))
    return {
        "status": "success",
        "stage": "structural_validation",
        "budget_rows": len(budget_df),
        "actual_rows": len(actual_df),
        "mapping_rows": len(mapping_df),
        "excluded_aggregates": list(AGGREGATE_CODES),
        "hierarchy_nodes": G.number_of_nodes() if "parent_code" in mapping_df.columns else 0,
        "hierarchy_edges": G.number_of_edges() if "parent_code" in mapping_df.columns else 0,
        "message": "Les 3 fichiers CSV sont valides et nettoyés. Agrégats exclus. Hiérarchie validée (aucun cycle)."
    }

def classify_pnl_accounts(tool_context: ToolContext) -> dict:
    """Classifie les comptes P&L et contrôle la matérialité financière.

    Précondition : a1_output.status == 'success' dans le session state.

    Tâches :
    1. Vérifier précondition A1
    2. Merge Actual + Mapping (left join sur code)
    3. Identifier transactions non-mappées (categorie_analyse null)
    4. Calcul matérialité : unmapped_amount / total_amount (sur Total annuel)
    5. Si matérialité > 2%  → error
    6. Résumé des comptes classifiés (sans profils comportementaux — réservé à un agent dédié)

    Returns:
        dict avec status='success' ou 'error'
    """
    # ── 0. Précondition : A1 doit avoir réussi ───────────────────────
    a1_output = tool_context.state.get("a1_output")

    if not a1_output:
        result = {
            "status": "error",
            "stage": "precondition_check",
            "errors": ["a1_output absent du session state. A1_Normalizer doit s'exécuter en premier."]
        }
        tool_context.state["a2_output"] = result
        return result

    if a1_output.get("status") != "success":
        result = {
            "status": "error",
            "stage": "precondition_check",
            "errors": [
                f"a1_output.status = '{a1_output.get('status')}'. "
                f"Erreurs A1 : {a1_output.get('errors', [])}"
            ]
        }
        tool_context.state["a2_output"] = result
        return result

    try:
        # ── 1. Reconstruire les DataFrames depuis a1_output ───────────
        from io import StringIO
        budget_df = pd.read_json(StringIO(a1_output["normalized_budget"]), orient="records")
        actual_df = pd.read_json(StringIO(a1_output["normalized_actual"]), orient="records")
        mapping_df = pd.read_json(StringIO(a1_output["normalized_mapping"]), orient="records")

        for df in [budget_df, actual_df, mapping_df]:
            df["code"] = df["code"].astype(str).str.strip()

        # ── 2. Merge Actual + Mapping (left join sur code) ────────────
        classified_df = actual_df.merge(
            mapping_df, on="code", how="left", suffixes=("", "_map")
        )

        # ── 3. Identifier transactions non-mappées ────────────────────
        #    categorie_analyse est la colonne clé du mapping
        unmapped_mask = classified_df["categorie_analyse"].isna()
        unmapped_df = classified_df[unmapped_mask]
        mapped_df = classified_df[~unmapped_mask]

        logger.info("A2 — %d/%d mappées, %d non-mappées",
                     len(mapped_df), len(classified_df), len(unmapped_df))

        # ── 4. Calcul de matérialité financière ───────────────────────
        #    Utiliser la colonne 'total' annuel pour le calcul de matérialité
        total_col = "total" if "total" in classified_df.columns else None
        if total_col:
            unmapped_amount = unmapped_df[total_col].abs().sum() if not unmapped_df.empty else 0.0
            total_amount = classified_df[total_col].abs().sum()
        else:
            # Fallback : somme des colonnes mensuelles
            month_cols_lower = [m for m in MONTH_COLS if m in classified_df.columns]
            unmapped_amount = unmapped_df[month_cols_lower].abs().sum().sum() if not unmapped_df.empty else 0.0
            total_amount = classified_df[month_cols_lower].abs().sum().sum()

        materiality_ratio = (unmapped_amount / total_amount) if total_amount > 0 else 0.0

        logger.info("A2 — matérialité : %.4f (seuil=%.2f)", materiality_ratio, MATERIALITY_THRESHOLD)

        # ── 5. Décision : matérialité > 2% → STOP ────────────────────
        if materiality_ratio > MATERIALITY_THRESHOLD:
            unmapped_codes = unmapped_df["code"].unique().tolist()
            result = {
                "status": "error",
                "stage": "accounting_mapping_validation",
                "reason": "Accounting classification coverage insufficient for reliable P&L analysis",
                "materiality_ratio": round(materiality_ratio, 6),
                "unmapped_amount": round(unmapped_amount, 2),
                "total_amount": round(total_amount, 2),
                "unmapped_codes": unmapped_codes,
            }
            tool_context.state["a2_output"] = result
            logger.error("A2 — STOP matérialité %.2f%% > %.2f%%",
                          materiality_ratio * 100, MATERIALITY_THRESHOLD * 100)
            return result

        # ── 6. Résumé des comptes classifiés ──────────────────────────
        account_summary = []
        for code in classified_df["code"].unique():
            row = classified_df[classified_df["code"] == code].iloc[0]
            categorie = str(row.get("categorie_analyse", "")).strip() if pd.notna(row.get("categorie_analyse")) else "non-mappé"
            rubrique = str(row.get("rubrique", "")) if pd.notna(row.get("rubrique")) else ""
            libelle = str(row.get("libelle", "")) if pd.notna(row.get("libelle")) else rubrique
            classe = str(row.get("classe", "")) if pd.notna(row.get("classe")) else ""

            account_summary.append({
                "code": code,
                "rubrique": rubrique,
                "libelle": libelle,
                "classe": classe,
                "categorie_analyse": categorie,
            })

        # ── 7. Résultat success ───────────────────────────────────────
        result = {
            "status": "success",
            "stage": "accounting_classification",
            "normalized_budget": budget_df.to_json(orient="records", force_ascii=False),
            "normalized_actual": actual_df.to_json(orient="records", force_ascii=False),
            "classified_actual": classified_df.to_json(orient="records", force_ascii=False),
            "materiality_ratio": round(materiality_ratio, 6),
            "unmapped_count": len(unmapped_df),
            "mapped_count": len(mapped_df),
            "account_summary": account_summary,
        }

        tool_context.state["a2_output"] = result
        logger.info("A2 — classification OK (matérialité=%.4f, %d comptes)",
                     materiality_ratio, len(account_summary))

        return {
            "status": "success",
            "stage": "accounting_classification",
            "materiality_ratio": round(materiality_ratio, 6),
            "mapped_count": len(mapped_df),
            "unmapped_count": len(unmapped_df),
            "total_accounts": len(classified_df),
            "account_summary": account_summary,
            "message": "Classification comptable terminée avec succès."
        }

    except Exception as exc:
        logger.error("A2 — exception : %s", exc)
        result = {
            "status": "error",
            "stage": "accounting_classification",
            "errors": [f"Exception inattendue : {exc}"]
        }
        tool_context.state["a2_output"] = result
        return result

def analyze_pnl_variances(tool_context: ToolContext) -> dict:
    """Analyse P&L robuste : decorticage + scoring 5 axes. AUCUN filtre.

    Retourne TOUTES les anomalies scorees avec une suggestion :
      fortement_recommande / a_evaluer / probablement_negligeable
    C est au LLM de decider RETENIR ou ECARTER chacune.
    """
    execution_log: list = []

    # ── 0. Precondition ───────────────────────────────────────────────
    a2_output = tool_context.state.get("a2_output")
    if not a2_output or a2_output.get("status") != "success":
        result = {
            "status": "error",
            "stage": "precondition_check",
            "errors": ["a2_output absent ou en erreur."],
        }
        tool_context.state["a3_output"] = result
        return result

    try:
        # ── 1. Charger DataFrames ─────────────────────────────────────
        budget_df = _gen_fn(
            "load_budget", lambda: pd.read_json(
                _SIO(a2_output["normalized_budget"]), orient="records"
            ), execution_log
        )
        actual_df = _gen_fn(
            "load_actual", lambda: pd.read_json(
                _SIO(a2_output["normalized_actual"]), orient="records"
            ), execution_log
        )
        mapping_df = _gen_fn(
            "load_mapping",
            lambda: pd.read_json(
                _SIO(a2_output.get("classified_actual", "[]")), orient="records"
            )[["code", "parent_code"]].drop_duplicates()
            if "classified_actual" in a2_output
            else pd.DataFrame(columns=["code", "parent_code"]),
            execution_log,
        )

        for df in [budget_df, actual_df]:
            df["code"] = df["code"].astype(str).str.strip()
        mapping_df["code"] = mapping_df["code"].astype(str).str.strip()
        if "parent_code" in mapping_df.columns:
            mapping_df["parent_code"] = mapping_df["parent_code"].astype(str).str.strip()

        # ── 2. Agreger totaux annuels ─────────────────────────────────
        month_cols = [c for c in MONTH_COLS if c in budget_df.columns]
        total_col = "total" if "total" in budget_df.columns else None

        def _aggregate_totals(df, label):
            if total_col and total_col in df.columns:
                agg = df.groupby("code", as_index=False)[total_col].sum()
                agg.rename(columns={total_col: f"total_{label}"}, inplace=True)
            else:
                df_m = df.copy()
                df_m["_sum"] = df_m[month_cols].sum(axis=1)
                agg = df_m.groupby("code", as_index=False)["_sum"].sum()
                agg.rename(columns={"_sum": f"total_{label}"}, inplace=True)
            return agg

        budget_agg = _gen_fn("aggregate_budget",
            lambda: _aggregate_totals(budget_df, "budget"), execution_log)
        actual_agg = _gen_fn("aggregate_actual",
            lambda: _aggregate_totals(actual_df, "actual"), execution_log)

        # ── 3. Fusionner ──────────────────────────────────────────────
        merged = _gen_fn("merge_budget_actual",
            lambda: budget_agg.merge(actual_agg, on="code", how="outer").fillna(0.0),
            execution_log)
        merged = _gen_fn("attach_hierarchy",
            lambda: merged.merge(
                mapping_df[["code", "parent_code"]].drop_duplicates(),
                on="code", how="left"),
            execution_log)

        # ── 3b. Parents manquants ─────────────────────────────────────
        a1_data = tool_context.state.get("a1_output")
        if a1_data and "normalized_mapping" in a1_data:
            full_hier = _gen_fn("load_full_hierarchy",
                lambda: pd.read_json(
                    _SIO(a1_data["normalized_mapping"]), orient="records"
                )[["code", "parent_code"]].drop_duplicates(),
                execution_log)
            full_hier["code"] = full_hier["code"].astype(str).str.strip()
            full_hier["parent_code"] = full_hier["parent_code"].astype(str).str.strip()
        else:
            full_hier = mapping_df[["code", "parent_code"]].copy()

        def _insert_missing_parents(df, hierarchy):
            out = df.copy()
            hier_map = dict(zip(
                hierarchy["code"].astype(str).str.strip(),
                hierarchy["parent_code"].astype(str).str.strip()))
            existing = set(out["code"].astype(str).str.strip())
            while True:
                referenced = set()
                for _, r in out.iterrows():
                    pc = str(r["parent_code"]).strip()
                    c = str(r["code"]).strip()
                    if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                        referenced.add(pc)
                missing = referenced - existing
                if not missing:
                    break
                new_rows = []
                for code in missing:
                    new_rows.append({
                        "code": code, "total_budget": 0.0, "total_actual": 0.0,
                        "parent_code": hier_map.get(code, code),
                        "variance": 0.0, "pct_variance": None,
                        "new_or_unplanned_activity": False,
                    })
                out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
                existing.update(missing)
            return out

        merged = _gen_fn("insert_missing_parents",
            lambda: _insert_missing_parents(merged, full_hier), execution_log)

        # ── 4. Variance feuille ───────────────────────────────────────
        def _compute_leaf_variance(df):
            out = df.copy()
            out["variance"] = out["total_actual"] - out["total_budget"]
            out["pct_variance"] = out.apply(
                lambda r: round(float(r["variance"] / r["total_budget"]) * 100, 4)
                if r["total_budget"] != 0 else None, axis=1)
            out["new_or_unplanned_activity"] = (
                (out["total_budget"] == 0) & (out["total_actual"] != 0))
            return out

        merged = _gen_fn("compute_leaf_variance",
            lambda: _compute_leaf_variance(merged), execution_log)

        # ── 5. Niveaux hierarchiques ──────────────────────────────────
        def _assign_levels(df):
            out = df.copy()
            code_to_parent = dict(zip(out["code"], out["parent_code"].fillna("")))
            def _level(code, visited=None):
                if visited is None:
                    visited = set()
                if code in visited:
                    return 0
                visited.add(code)
                parent = str(code_to_parent.get(code, "")).strip()
                if not parent or parent == code or parent.upper() in ("NAN", "NULL", "NONE", ""):
                    return 0
                return 1 + _level(parent, visited)
            out["level"] = out["code"].apply(_level)
            return out

        merged = _gen_fn("assign_hierarchy_levels",
            lambda: _assign_levels(merged), execution_log)

        # ── 6. Rollup ascendant multi-niveaux ─────────────────────────
        def _rollup(df):
            out = df.copy()
            max_level = int(out["level"].max()) if len(out) > 0 else 0
            for current_level in range(max_level, 0, -1):
                children = out[out["level"] == current_level]
                parent_sums = {}
                for _, row in children.iterrows():
                    pc = str(row["parent_code"]).strip()
                    c = str(row["code"]).strip()
                    if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                        if pc not in parent_sums:
                            parent_sums[pc] = {"budget": 0.0, "actual": 0.0}
                        parent_sums[pc]["budget"] += float(row["total_budget"])
                        parent_sums[pc]["actual"] += float(row["total_actual"])
                for idx, row in out.iterrows():
                    code = str(row["code"]).strip()
                    if code in parent_sums:
                        out.at[idx, "total_budget"] = parent_sums[code]["budget"]
                        out.at[idx, "total_actual"] = parent_sums[code]["actual"]
            out["variance"] = out["total_actual"] - out["total_budget"]
            out["pct_variance"] = out.apply(
                lambda r: round(float(r["variance"] / r["total_budget"]) * 100, 4)
                if r["total_budget"] != 0 else None, axis=1)
            out["new_or_unplanned_activity"] = (
                (out["total_budget"] == 0) & (out["total_actual"] != 0))
            return out

        merged = _gen_fn("rollup_ascendant",
            lambda: _rollup(merged), execution_log)

        # ── 7. Validation rollup ──────────────────────────────────────
        def _validate_rollup(df):
            roots = df[df["level"] == 0]
            parent_codes_set = set()
            for _, row in df.iterrows():
                pc = str(row["parent_code"]).strip()
                c = str(row["code"]).strip()
                if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                    parent_codes_set.add(pc)
            all_codes = set(df["code"].astype(str).str.strip())
            true_leaf_codes = all_codes - parent_codes_set
            true_leaves = df[df["code"].isin(true_leaf_codes)]
            root_budget = float(roots["total_budget"].sum())
            root_actual = float(roots["total_actual"].sum())
            leaf_budget = float(true_leaves["total_budget"].sum())
            leaf_actual = float(true_leaves["total_actual"].sum())
            tol = 0.01
            budget_ok = abs(root_budget - leaf_budget) < tol or len(true_leaves) == 0
            actual_ok = abs(root_actual - leaf_actual) < tol or len(true_leaves) == 0
            return {
                "root_budget": round(root_budget, 2),
                "root_actual": round(root_actual, 2),
                "true_leaf_budget": round(leaf_budget, 2),
                "true_leaf_actual": round(leaf_actual, 2),
                "root_count": len(roots),
                "true_leaf_count": len(true_leaves),
                "coherent": bool(budget_ok and actual_ok),
            }

        validation = _gen_fn("validate_rollup",
            lambda: _validate_rollup(merged), execution_log)

        if not validation.get("coherent", False):
            result = {
                "status": "error", "stage": "rollup_validation",
                "validation": validation, "execution_log": execution_log,
            }
            tool_context.state["a3_output"] = _to_native(result)
            return _to_native(result)

        # ── 8. Variance Drivers ───────────────────────────────────────
        def _compute_drivers(df):
            drivers = []
            parents_with_children = {}
            for _, row in df.iterrows():
                pc = str(row["parent_code"]).strip()
                c = str(row["code"]).strip()
                if pc and pc != c and pc.upper() not in ("NAN", "NULL", "NONE", ""):
                    parents_with_children.setdefault(pc, []).append(c)
            for parent_code, children in parents_with_children.items():
                parent_row = df[df["code"] == parent_code]
                if parent_row.empty:
                    continue
                parent_variance = float(parent_row.iloc[0]["variance"])
                if parent_variance == 0:
                    continue
                for child_code in children:
                    child_row = df[df["code"] == child_code]
                    if child_row.empty:
                        continue
                    child_variance = float(child_row.iloc[0]["variance"])
                    contribution = round(
                        (child_variance / parent_variance) * 100, 4
                    ) if parent_variance != 0 else 0.0
                    abs_c = abs(contribution)
                    if abs_c >= 50:
                        role = "primary"
                    elif contribution < 0:
                        role = "compensating"
                    elif abs_c < 10:
                        role = "marginal"
                    else:
                        role = "primary"
                    drivers.append({
                        "child_code": child_code,
                        "parent_code": parent_code,
                        "child_variance": round(child_variance, 2),
                        "parent_variance": round(parent_variance, 2),
                        "variance_contribution": round(contribution, 4),
                        "driver_role": role,
                    })
            return drivers

        drivers = _gen_fn("compute_variance_drivers",
            lambda: _compute_drivers(merged), execution_log)

        # ── 9. Variance tree ──────────────────────────────────────────
        def _build_tree(df):
            tree = []
            for _, row in df.iterrows():
                tree.append({
                    "code": str(row["code"]),
                    "parent_code": str(row.get("parent_code", "")),
                    "level": int(row.get("level", 0)),
                    "total_budget": round(float(row["total_budget"]), 2),
                    "total_actual": round(float(row["total_actual"]), 2),
                    "variance": round(float(row["variance"]), 2),
                    "pct_variance": round(float(row["pct_variance"]), 4)
                    if pd.notna(row.get("pct_variance")) else None,
                    "new_or_unplanned_activity": bool(
                        row.get("new_or_unplanned_activity", False)),
                })
            return tree

        variance_tree = _gen_fn("build_variance_tree",
            lambda: _build_tree(merged), execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── 10. Ecarts mensuels ───────────────────────────────────────
        # ══════════════════════════════════════════════════════════════

        def _compute_monthly_variances(budget_raw, actual_raw, mcols):
            monthly_vars = []
            if not mcols:
                return monthly_vars
            b = budget_raw.copy()
            a = actual_raw.copy()
            b["code"] = b["code"].astype(str).str.strip()
            a["code"] = a["code"].astype(str).str.strip()
            codes = set(b["code"]) | set(a["code"])
            for code in codes:
                b_row = b[b["code"] == code]
                a_row = a[a["code"] == code]
                if b_row.empty and a_row.empty:
                    continue
                for m in mcols:
                    bv = float(b_row[m].iloc[0]) if not b_row.empty and m in b_row.columns else 0.0
                    av = float(a_row[m].iloc[0]) if not a_row.empty and m in a_row.columns else 0.0
                    v = av - bv
                    pct = round((v / bv) * 100, 2) if bv != 0 else None
                    monthly_vars.append({
                        "code": code, "month": m,
                        "budget": round(bv, 2), "actual": round(av, 2),
                        "variance": round(v, 2), "pct_variance": pct,
                    })
            return monthly_vars

        monthly_variances = _gen_fn("compute_monthly_variances",
            lambda: _compute_monthly_variances(budget_df, actual_df, month_cols),
            execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── 11. Detection anomalies (5 types) ─────────────────────────
        # ══════════════════════════════════════════════════════════════

        def _detect_anomalies(budget_raw, actual_raw, mcols):
            anomalies = []
            if not mcols:
                return anomalies
            b = budget_raw.copy()
            a = actual_raw.copy()
            b["code"] = b["code"].astype(str).str.strip()
            a["code"] = a["code"].astype(str).str.strip()
            codes = set(b["code"]) | set(a["code"])
            for code in codes:
                b_row = b[b["code"] == code]
                a_row = a[a["code"] == code]
                if b_row.empty and a_row.empty:
                    continue
                budget_vals = []
                actual_vals = []
                for m in mcols:
                    bv = float(b_row[m].iloc[0]) if not b_row.empty and m in b_row.columns else 0.0
                    av = float(a_row[m].iloc[0]) if not a_row.empty and m in a_row.columns else 0.0
                    budget_vals.append(bv)
                    actual_vals.append(av)
                budget_arr = _np.array(budget_vals)
                actual_arr = _np.array(actual_vals)
                var_arr = actual_arr - budget_arr

                # 1. Spikes mensuels (> 20%)
                for i, m in enumerate(mcols):
                    if budget_arr[i] != 0:
                        pct = (var_arr[i] / budget_arr[i]) * 100
                        if abs(pct) > 20:
                            anomalies.append({
                                "code": code, "type": "monthly_spike", "month": m,
                                "budget": round(float(budget_arr[i]), 2),
                                "actual": round(float(actual_arr[i]), 2),
                                "variance": round(float(var_arr[i]), 2),
                                "pct": round(float(pct), 2),
                                "severity": "high" if abs(pct) > 50 else "medium",
                            })

                # 2. Tendance (regression)
                if len(actual_arr) >= 6 and _np.std(actual_arr) > 0:
                    x = _np.arange(len(actual_arr))
                    slope = float(_np.polyfit(x, actual_arr, 1)[0])
                    mean_val = float(_np.mean(actual_arr))
                    if mean_val != 0 and abs(slope * 12 / mean_val) > 0.15:
                        anomalies.append({
                            "code": code, "type": "trend",
                            "direction": "increasing" if slope > 0 else "decreasing",
                            "monthly_slope": round(slope, 2),
                            "annual_drift_pct": round(float(slope * 12 / mean_val * 100), 2),
                            "severity": "high" if abs(slope * 12 / mean_val) > 0.30 else "medium",
                        })

                # 3. Volatilite (CV > 30%)
                if _np.mean(actual_arr) != 0:
                    cv = float(_np.std(actual_arr) / abs(_np.mean(actual_arr)) * 100)
                    if cv > 30:
                        anomalies.append({
                            "code": code, "type": "high_volatility",
                            "cv_pct": round(cv, 2),
                            "min_month": mcols[int(_np.argmin(actual_arr))],
                            "max_month": mcols[int(_np.argmax(actual_arr))],
                            "min_val": round(float(_np.min(actual_arr)), 2),
                            "max_val": round(float(_np.max(actual_arr)), 2),
                            "severity": "high" if cv > 50 else "medium",
                        })

                # 4. Mois non budgete
                for i, m in enumerate(mcols):
                    if budget_arr[i] == 0 and actual_arr[i] != 0:
                        anomalies.append({
                            "code": code, "type": "unbudgeted_month", "month": m,
                            "actual": round(float(actual_arr[i]), 2),
                            "severity": "medium",
                        })

                # 5. Inversion de signe
                for i, m in enumerate(mcols):
                    if budget_arr[i] != 0 and actual_arr[i] != 0:
                        if (budget_arr[i] > 0) != (actual_arr[i] > 0):
                            anomalies.append({
                                "code": code, "type": "sign_reversal", "month": m,
                                "budget": round(float(budget_arr[i]), 2),
                                "actual": round(float(actual_arr[i]), 2),
                                "severity": "high",
                            })
            return anomalies

        monthly_anomalies = _gen_fn("detect_monthly_anomalies",
            lambda: _detect_anomalies(budget_df, actual_df, month_cols),
            execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   P H A S E   1  :  D E C O R T I C A G E   S Y S T E M A T I Q U E
        # ══════════════════════════════════════════════════════════════════

        def _build_monthly_profiles(monthly_vars, mcols):
            profiles = {}
            for mv in monthly_vars:
                code = mv["code"]
                if code not in profiles:
                    profiles[code] = {
                        "months_with_variance": 0,
                        "variances": [],
                        "actuals": [],
                        "pct_variances": [],
                    }
                if mv["variance"] != 0:
                    profiles[code]["months_with_variance"] += 1
                profiles[code]["variances"].append(mv["variance"])
                profiles[code]["actuals"].append(mv["actual"])
                if mv["pct_variance"] is not None:
                    profiles[code]["pct_variances"].append(mv["pct_variance"])
            for code, p in profiles.items():
                vals = _np.array(p["actuals"])
                if len(vals) >= 6 and _np.std(vals) > 0:
                    x = _np.arange(len(vals))
                    slope = float(_np.polyfit(x, vals, 1)[0])
                    mean_val = float(_np.mean(vals))
                    p["slope"] = slope
                    p["pct_drift"] = round(slope * 12 / mean_val * 100, 2) if mean_val != 0 else 0
                else:
                    p["slope"] = 0
                    p["pct_drift"] = 0
                big_months = sum(1 for pv in p["pct_variances"] if abs(pv) > 20)
                p["months_significant"] = big_months
            return profiles

        profiles = _gen_fn("build_monthly_profiles",
            lambda: _build_monthly_profiles(monthly_variances, month_cols),
            execution_log)

        NATURE_MAP = {
            "monthly_spike":    "temporelle",
            "trend":            "comportementale",
            "high_volatility":  "comportementale",
            "unbudgeted_month": "temporelle",
            "sign_reversal":    "donnee",
            "annual_variance":  "structurelle",
        }
        ORIGINE_MAP = {
            "unbudgeted_month": "planification",
            "sign_reversal":    "erreur_potentielle",
            "trend":            "operationnel",
            "high_volatility":  "operationnel",
            "monthly_spike":    "saisonnier",
        }

        def _classify_origine(a):
            atype = a.get("type", "annual_variance")
            if atype in ORIGINE_MAP:
                return ORIGINE_MAP[atype]
            pct = abs(a.get("pct_variance", 0))
            if pct > 100:
                return "planification"
            return "operationnel"

        def _classify_frequence(code, profiles_dict, atype):
            if atype == "annual_variance":
                return "annuelle"
            p = profiles_dict.get(code, {})
            n = p.get("months_with_variance", 0)
            if n <= 1:
                return "ponctuelle"
            elif n <= 3:
                return "occasionnelle"
            else:
                return "recurrente"

        def _classify_tendance(code, profiles_dict, atype):
            if atype in ("sign_reversal", "unbudgeted_month"):
                return "indeterminee"
            p = profiles_dict.get(code, {})
            drift = p.get("pct_drift", 0)
            if abs(drift) < 5:
                return "stable"
            elif drift > 15:
                return "croissante"
            elif drift > 0:
                return "legerement_croissante"
            elif drift < -15:
                return "decroissante"
            else:
                return "legerement_decroissante"

        def _classify_portee(level):
            if level == 0:
                return "racine"
            elif level == 1:
                return "intermediaire"
            else:
                return "feuille"

        def _phase1_decorticage(vtree, monthly_anoms, profiles_dict, merged_df):
            unified = []
            for v in vtree:
                pct = v.get("pct_variance")
                if pct is None or pct == 0:
                    continue
                code = v["code"]
                level = v["level"]
                a = {
                    "source": "annual",
                    "type": "annual_variance",
                    "code": code,
                    "level": level,
                    "parent_code": v["parent_code"],
                    "total_budget": v["total_budget"],
                    "total_actual": v["total_actual"],
                    "variance": v["variance"],
                    "pct_variance": round(pct, 2),
                    "new_or_unplanned": v.get("new_or_unplanned_activity", False),
                    "nature":    NATURE_MAP["annual_variance"],
                    "origine":   _classify_origine({"type": "annual_variance", "pct_variance": pct}),
                    "frequence": _classify_frequence(code, profiles_dict, "annual_variance"),
                    "tendance":  _classify_tendance(code, profiles_dict, "annual_variance"),
                    "portee":    _classify_portee(level),
                }
                unified.append(a)
            for ma in monthly_anoms:
                code = ma.get("code", "?")
                a = {**ma, "source": "monthly"}
                match = merged_df[merged_df["code"] == code]
                if not match.empty:
                    a["level"] = int(match.iloc[0]["level"])
                    a["parent_code"] = str(match.iloc[0]["parent_code"])
                else:
                    a["level"] = 2
                    a["parent_code"] = ""
                atype = a.get("type", "")
                level = a["level"]
                a["nature"]    = NATURE_MAP.get(atype, "donnee")
                a["origine"]   = _classify_origine(a)
                a["frequence"] = _classify_frequence(code, profiles_dict, atype)
                a["tendance"]  = _classify_tendance(code, profiles_dict, atype)
                a["portee"]    = _classify_portee(level)
                unified.append(a)
            return unified

        all_anomalies = _gen_fn("phase1_decorticage",
            lambda: _phase1_decorticage(
                variance_tree, monthly_anomalies, profiles, merged),
            execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   P H A S E   2  :  S C O R I N G   5   A X E S  (1-100)
        #              PAS DE SEUIL — SUGGESTION UNIQUEMENT
        # ══════════════════════════════════════════════════════════════════

        total_budget_abs = float(abs(merged["total_budget"].sum())) if len(merged) > 0 else 1.0
        if total_budget_abs == 0:
            total_budget_abs = 1.0

        def _phase2_scoring(anomalies, total_budget_abs_val):
            TYPE_URGENCE_BASE = {
                "sign_reversal": 18, "monthly_spike": 15,
                "trend": 12, "high_volatility": 10,
                "annual_variance": 10, "unbudgeted_month": 8,
            }
            FREQ_SCORES = {
                "recurrente": 15, "occasionnelle": 10,
                "annuelle": 8, "ponctuelle": 5,
            }
            TENDANCE_SCORES = {
                "croissante": 15, "legerement_croissante": 10,
                "stable": 7, "indeterminee": 7,
                "legerement_decroissante": 4, "decroissante": 2,
            }
            PORTEE_SCORES = {"racine": 15, "intermediaire": 10, "feuille": 5}

            scored = []
            for a in anomalies:
                atype = a.get("type", "annual_variance")
                sev = a.get("severity", "medium")

                # Pilier 1 : Impact Financier (0-30)
                p1 = 0.0
                abs_var = abs(a.get("variance", 0))
                abs_pct = abs(a.get("pct_variance",
                    a.get("pct", a.get("annual_drift_pct", a.get("cv_pct", 0)))))
                if total_budget_abs_val > 0:
                    p1 += min(round(abs_var / total_budget_abs_val * 100 * 3, 1), 15)
                p1 += min(round(abs_pct / 10, 1), 10)
                if a.get("new_or_unplanned") or atype == "unbudgeted_month":
                    p1 += 5
                p1 = round(min(p1, 30), 1)

                # Pilier 2 : Urgence (0-25)
                p2 = float(TYPE_URGENCE_BASE.get(atype, 10))
                if sev == "high":
                    p2 += 5
                elif sev == "medium":
                    p2 += 2
                p2 = round(min(p2, 25), 1)

                # Pilier 3 : Frequence (0-15)
                p3 = float(FREQ_SCORES.get(a.get("frequence", "ponctuelle"), 5))

                # Pilier 4 : Tendance (0-15)
                p4 = float(TENDANCE_SCORES.get(a.get("tendance", "stable"), 7))

                # Pilier 5 : Portee (0-15)
                p5 = float(PORTEE_SCORES.get(a.get("portee", "feuille"), 5))

                total = round(min(p1 + p2 + p3 + p4 + p5, 100))

                if total >= 80:
                    niveau = "critique"
                elif total >= 60:
                    niveau = "majeur"
                else:
                    niveau = "mineur"

                # SUGGESTION (guidance pour le LLM, PAS un filtre)
                if total >= 65:
                    suggestion = "fortement_recommande"
                elif total >= 40:
                    suggestion = "a_evaluer"
                else:
                    suggestion = "probablement_negligeable"

                a["scoring"] = {
                    "impact_financier": p1,
                    "urgence": p2,
                    "frequence": p3,
                    "tendance": p4,
                    "portee": p5,
                }
                a["score"] = total
                a["niveau"] = niveau
                a["suggestion"] = suggestion
                scored.append(a)

            return sorted(scored, key=lambda x: x["score"], reverse=True)

        all_scored = _gen_fn("phase2_scoring",
            lambda: _phase2_scoring(all_anomalies, total_budget_abs),
            execution_log)

        # Stats de scoring (pas de filtre — juste des stats)
        scoring_stats = _gen_fn("compute_scoring_stats", lambda: {
            "total_scored": len(all_scored),
            "fortement_recommande": sum(
                1 for a in all_scored if a["suggestion"] == "fortement_recommande"),
            "a_evaluer": sum(
                1 for a in all_scored if a["suggestion"] == "a_evaluer"),
            "probablement_negligeable": sum(
                1 for a in all_scored if a["suggestion"] == "probablement_negligeable"),
            "score_max": max((a["score"] for a in all_scored), default=0),
            "score_min": min((a["score"] for a in all_scored), default=0),
            "score_mean": round(
                sum(a["score"] for a in all_scored) / max(len(all_scored), 1), 1),
            "critiques": sum(1 for a in all_scored if a["niveau"] == "critique"),
            "majeurs": sum(1 for a in all_scored if a["niveau"] == "majeur"),
            "mineurs": sum(1 for a in all_scored if a["niveau"] == "mineur"),
        }, execution_log)

        # ══════════════════════════════════════════════════════════════════
        #   FORMAT TOUTES les anomalies en cartes (le LLM triera)
        # ══════════════════════════════════════════════════════════════════

        def _format_all_cards(scored_list):
            cards = []
            for idx, a in enumerate(scored_list, 1):
                anomalie_id = f"ANM-{idx:03d}"
                code = a.get("code", "?")
                source = a.get("source", "?")
                atype = a.get("type", "annual_variance")
                score = a["score"]
                niveau = a["niveau"]
                scoring = a.get("scoring", {})
                suggestion = a.get("suggestion", "a_evaluer")

                if source == "annual":
                    pct = a.get("pct_variance", 0)
                    var = a.get("variance", 0)
                    resume = (
                        f"Compte {code} : ecart annuel de "
                        f"{var:+,.0f} ({pct:+.1f}%) vs budget"
                    )
                elif atype == "monthly_spike":
                    resume = (
                        f"Compte {code} : spike {a.get('month','')} "
                        f"({a.get('pct', 0):+.1f}%)"
                    )
                elif atype == "trend":
                    resume = (
                        f"Compte {code} : tendance {a.get('direction','')} "
                        f"(drift {a.get('annual_drift_pct', 0):+.1f}%/an)"
                    )
                elif atype == "high_volatility":
                    resume = (
                        f"Compte {code} : volatilite excessive "
                        f"(CV={a.get('cv_pct', 0):.0f}%)"
                    )
                elif atype == "sign_reversal":
                    resume = f"Compte {code} : inversion de signe {a.get('month','')}"
                elif atype == "unbudgeted_month":
                    resume = (
                        f"Compte {code} : activite non budgetee "
                        f"{a.get('month','')} ({a.get('actual', 0):,.0f})"
                    )
                else:
                    resume = f"Compte {code} : anomalie {atype}"

                top_pilier = max(
                    scoring.items(), key=lambda x: x[1]
                ) if scoring else ("?", 0)

                cards.append({
                    "anomalie_id": anomalie_id,
                    "code": code,
                    "source": source,
                    "type": atype,
                    "resume": resume,
                    "score": score,
                    "niveau": niveau,
                    "suggestion": suggestion,
                    "nature": a.get("nature", "?"),
                    "origine": a.get("origine", "?"),
                    "frequence": a.get("frequence", "?"),
                    "tendance": a.get("tendance", "?"),
                    "portee": a.get("portee", "?"),
                    "scoring_detail": scoring,
                    "pilier_dominant": f"{top_pilier[0]} ({top_pilier[1]:.0f}pts)",
                    "donnees": {
                        k: v for k, v in a.items()
                        if k not in (
                            "score", "scoring", "niveau", "source", "suggestion",
                            "nature", "origine", "frequence", "tendance", "portee"
                        )
                    },
                })
            return cards

        all_cards = _gen_fn("format_all_cards",
            lambda: _format_all_cards(all_scored), execution_log)

        # ══════════════════════════════════════════════════════════════
        # ── RESULTAT (TOUTES les cartes, pas de filtre) ───────────────
        # ══════════════════════════════════════════════════════════════
        stats = {
            "total_accounts": len(variance_tree),
            "root_accounts": sum(1 for v in variance_tree if v["level"] == 0),
            "leaf_accounts": sum(1 for v in variance_tree if v["level"] > 0),
            "drivers_count": len(drivers),
            "unplanned_count": sum(
                1 for v in variance_tree if v["new_or_unplanned_activity"]),
            "total_anomalies_scored": len(all_scored),
            "functions_executed": len(execution_log),
        }

        result = {
            "status": "success",
            "stage": "anomaly_analysis",
            "all_anomaly_cards": all_cards,
            "scoring_stats": scoring_stats,
            "variance_tree": variance_tree,
            "drivers": drivers,
            "monthly_variances": monthly_variances,
            "validation": validation,
            "execution_log": execution_log,
            "stats": stats,
            "triage_status": "pending",  # sera mis a jour par save_triage_decisions
        }

        tool_context.state["a3_output"] = _to_native(result)
        logger.info(
            "A3 Phase 1+2 OK : %d comptes, %d anomalies scorees "
            "(%d recomm, %d a_eval, %d negli), %d fns — TRIAGE LLM EN ATTENTE",
            len(variance_tree), len(all_scored),
            scoring_stats.get("fortement_recommande", 0),
            scoring_stats.get("a_evaluer", 0),
            scoring_stats.get("probablement_negligeable", 0),
            len(execution_log),
        )

        # Retourner un resume compact pour le LLM (pas toutes les cartes
        # detaillees — seulement les fortement_recommande + a_evaluer
        # en detail, et un resume des negligeables)
        cards_for_review = [
            c for c in all_cards if c["suggestion"] != "probablement_negligeable"
        ]
        negligeable_summary = {
            "count": sum(1 for c in all_cards if c["suggestion"] == "probablement_negligeable"),
            "score_range": f"0-39",
            "examples": [
                {"id": c["anomalie_id"], "code": c["code"], "score": c["score"],
                 "resume": c["resume"][:50]}
                for c in all_cards if c["suggestion"] == "probablement_negligeable"
            ][:5],  # max 5 exemples
        }

        return _to_native({
            "status": "success",
            "stage": "anomaly_analysis_phase1_2",
            "stats": stats,
            "scoring_stats": scoring_stats,
            "validation": validation,
            "cards_to_review": cards_for_review,
            "negligeable_summary": negligeable_summary,
            "message": (
                f"Phase 1+2 terminee. {len(all_scored)} anomalies scorees. "
                f"{scoring_stats.get('fortement_recommande', 0)} fortement recommandees, "
                f"{scoring_stats.get('a_evaluer', 0)} a evaluer, "
                f"{scoring_stats.get('probablement_negligeable', 0)} negligeables. "
                f"APPELLE save_triage_decisions avec tes verdicts."
            ),
        })

    except Exception as exc:
        logger.error("A3 exception : %s\n%s", exc, _tb.format_exc())
        result = {
            "status": "error",
            "stage": "anomaly_analysis",
            "errors": [f"Exception inattendue : {exc}"],
            "execution_log": execution_log,
        }
        tool_context.state["a3_output"] = _to_native(result)
        return _to_native(result)

def save_triage_decisions(
    tool_context: ToolContext,
    decisions: list[dict],
) -> dict:
    """Persiste les decisions de triage du LLM dans le state.

    Args:
        decisions: liste de dicts, chacun avec :
            - anomalie_id : str (ex: "ANM-001")
            - verdict : str ("RETENIR" ou "ECARTER")
            - justification : str (1 phrase max)

    Retourne un resume du triage.
    """
    a3_output = tool_context.state.get("a3_output", {})
    if not a3_output or a3_output.get("status") != "success":
        return {"status": "error", "message": "a3_output absent. Appelle analyze_pnl_variances d abord."}

    all_cards = a3_output.get("all_anomaly_cards", [])
    if not all_cards:
        return {"status": "error", "message": "Aucune anomaly card trouvee dans a3_output."}

    # Indexer les decisions par ID
    decision_map = {}
    for d in decisions:
        aid = d.get("anomalie_id", "")
        verdict = d.get("verdict", "ECARTER").upper()
        justif = d.get("justification", "")
        if verdict not in ("RETENIR", "ECARTER"):
            verdict = "ECARTER"
        decision_map[aid] = {"verdict": verdict, "justification": justif}

    # Appliquer les decisions aux cartes
    retained_cards = []
    excluded_cards = []
    for card in all_cards:
        aid = card["anomalie_id"]
        dec = decision_map.get(aid)
        if dec:
            card["verdict_llm"] = dec["verdict"]
            card["justification_llm"] = dec["justification"]
        else:
            # Pas de decision explicite : suivre la suggestion
            if card.get("suggestion") == "fortement_recommande":
                card["verdict_llm"] = "RETENIR"
                card["justification_llm"] = "Retenu par defaut (fortement recommande, pas de verdict explicite)"
            else:
                card["verdict_llm"] = "ECARTER"
                card["justification_llm"] = "Ecarte par defaut (pas de verdict explicite)"

        if card["verdict_llm"] == "RETENIR":
            retained_cards.append(card)
        else:
            excluded_cards.append(card)

    # Mettre a jour a3_output
    a3_output["retained_cards"] = _to_native(retained_cards)
    a3_output["excluded_cards"] = _to_native(excluded_cards)
    a3_output["triage_status"] = "completed"
    a3_output["triage_summary"] = {
        "total_reviewed": len(all_cards),
        "retained": len(retained_cards),
        "excluded": len(excluded_cards),
        "retention_rate_pct": round(
            len(retained_cards) / max(len(all_cards), 1) * 100, 1),
        "decisions_received": len(decisions),
        "decisions_defaulted": len(all_cards) - len(decisions),
    }

    # Recomputer les stats
    a3_output["stats"]["anomalies_retained"] = len(retained_cards)
    a3_output["stats"]["anomalies_excluded"] = len(excluded_cards)
    a3_output["stats"]["critiques"] = sum(
        1 for c in retained_cards if c["niveau"] == "critique")
    a3_output["stats"]["majeurs"] = sum(
        1 for c in retained_cards if c["niveau"] == "majeur")
    a3_output["stats"]["mineurs"] = sum(
        1 for c in retained_cards if c["niveau"] == "mineur")

    tool_context.state["a3_output"] = _to_native(a3_output)

    logger.info(
        "A3 Triage LLM OK : %d/%d retenues (%d critiques, %d majeurs, %d mineurs)",
        len(retained_cards), len(all_cards),
        a3_output["stats"]["critiques"],
        a3_output["stats"]["majeurs"],
        a3_output["stats"]["mineurs"],
    )

    return _to_native({
        "status": "success",
        "stage": "triage_completed",
        "triage_summary": a3_output["triage_summary"],
        "retained_cards": retained_cards,
        "message": (
            f"Triage termine. {len(retained_cards)}/{len(all_cards)} retenues "
            f"({a3_output['stats']['critiques']} critiques, "
            f"{a3_output['stats']['majeurs']} majeurs, "
            f"{a3_output['stats']['mineurs']} mineurs). "
            f"Donnees prete pour le Reporter."
        ),
    })

def load_analysis_results(tool_context: ToolContext) -> dict:
    """Charge et structure les résultats des agents A1, A2 et A3 pour le rapport.

    Lit le session state partagé et construit un briefing package complet
    pour que le sous-agent suivant puisse rédiger un rapport stratégique.

    Ne fait AUCUN calcul — ne fait que lire et formater les données existantes.
    """

    # ── A1 : Validation structurelle ──────────────────────────────────
    a1 = tool_context.state.get("a1_output", {})
    a1_brief = {
        "status": a1.get("status", "absent"),
        "data_summary": a1.get("data_summary", {}),
    }

    # ── A2 : Classification comptable ─────────────────────────────────
    a2 = tool_context.state.get("a2_output", {})
    a2_brief = {
        "status": a2.get("status", "absent"),
        "materiality_ratio": a2.get("materiality_ratio", None),
        "mapped_count": a2.get("mapped_count", 0),
        "unmapped_count": a2.get("unmapped_count", 0),
    }

    # ── A3 : Analyse des anomalies ────────────────────────────────────
    a3 = tool_context.state.get("a3_output", {})
    a3_ok = a3.get("status") == "success"

    if not a3_ok:
        return {
            "status": "error",
            "message": "A3 n'a pas produit de résultats. Impossible de générer le rapport.",
            "a1": a1_brief,
            "a2": a2_brief,
        }

    # Stats globales
    a3_stats = a3.get("stats", {})
    scoring_stats = a3.get("scoring_stats", {})
    triage = a3.get("triage_summary", {})
    validation = a3.get("validation", {})

    # Variance tree (structure hiérarchique complète)
    vtree = a3.get("variance_tree", [])

    # Drivers (contribution de chaque enfant à l'écart parent)
    drivers = a3.get("drivers", [])

    # Anomalies retenues par le LLM (après triage)
    retained = a3.get("retained_cards", [])

    # Anomalies écartées (résumé seulement)
    excluded = a3.get("excluded_cards", [])
    excluded_summary = [
        {"id": c["anomalie_id"], "code": c["code"], "score": c["score"],
         "reason": c.get("justification_llm", "?")}
        for c in excluded[:10]
    ]

    # ── Enrichir les anomalies retenues avec les données clés ─────────
    enriched_anomalies = []
    for c in retained:
        donnees = c.get("donnees", {})
        enriched_anomalies.append({
            "anomalie_id": c["anomalie_id"],
            "code": c["code"],
            "type": c.get("type", "?"),
            "score": c["score"],
            "niveau": c["niveau"],
            "resume": c["resume"],
            # Décorticage
            "nature": c.get("nature", "?"),
            "origine": c.get("origine", "?"),
            "frequence": c.get("frequence", "?"),
            "tendance": c.get("tendance", "?"),
            "portee": c.get("portee", "?"),
            # Scoring
            "scoring_detail": c.get("scoring_detail", {}),
            "pilier_dominant": c.get("pilier_dominant", "?"),
            # Données brutes utiles
            "variance": donnees.get("variance"),
            "pct_variance": donnees.get("pct_variance"),
            "total_budget": donnees.get("total_budget"),
            "total_actual": donnees.get("total_actual"),
            "month": donnees.get("month"),
            "pct": donnees.get("pct"),
            "direction": donnees.get("direction"),
        })

    # ── Calcul des totaux depuis le variance tree ─────────────────────
    root_nodes = [v for v in vtree if v.get("level") == 0]
    total_budget = sum(v.get("total_budget", 0) for v in root_nodes)
    total_actual = sum(v.get("total_actual", 0) for v in root_nodes)
    total_variance = total_actual - total_budget

    # Revenus vs Charges
    revenus_nodes = [v for v in root_nodes if str(v.get("code", "")).startswith("7")]
    charges_nodes = [v for v in root_nodes if not str(v.get("code", "")).startswith("7")]

    revenus_budget = sum(v.get("total_budget", 0) for v in revenus_nodes)
    revenus_actual = sum(v.get("total_actual", 0) for v in revenus_nodes)
    charges_budget = sum(v.get("total_budget", 0) for v in charges_nodes)
    charges_actual = sum(v.get("total_actual", 0) for v in charges_nodes)

    briefing = {
        "status": "success",
        "message": (
            f"Briefing package prêt. {len(enriched_anomalies)} anomalies retenues, "
            f"{len(vtree)} noeuds dans le variance tree, {len(drivers)} drivers."
        ),

        # Contexte pipeline
        "a1_validation": a1_brief,
        "a2_classification": a2_brief,

        # Vue d'ensemble financière
        "performance_globale": {
            "total_budget": total_budget,
            "total_actual": total_actual,
            "ecart_global": total_variance,
            "ecart_pct": round(total_variance / max(abs(total_budget), 1) * 100, 2),
            "revenus": {
                "budget": revenus_budget,
                "actual": revenus_actual,
                "ecart": revenus_actual - revenus_budget,
            },
            "charges": {
                "budget": charges_budget,
                "actual": charges_actual,
                "ecart": charges_actual - charges_budget,
            },
        },

        # Structure hiérarchique
        "variance_tree": vtree,
        "drivers": drivers,

        # Anomalies (après triage LLM)
        "triage_stats": {
            "total_scored": scoring_stats.get("total_scored", 0),
            "retained": triage.get("retained", len(retained)),
            "excluded": triage.get("excluded", len(excluded)),
            "retention_rate": triage.get("retention_rate_pct", 0),
        },
        "anomalies_retenues": enriched_anomalies,
        "anomalies_ecartees_resume": excluded_summary,

        # Validation technique
        "rollup_validation": {
            "coherent": validation.get("coherent", False),
            "root_budget": validation.get("root_budget"),
            "root_actual": validation.get("root_actual"),
        },
    }

    # ── Feedback du A5_Quality_Judge (runs précédents) ────────────────
    # Charge les constats des évaluations passées pour que A4 puisse
    # corriger ses faiblesses et éviter les erreurs récurrentes.
    judge_feedback = []
    try:
        raw_history = db.get_judge_history(limit=3)
        for h in raw_history:
            judge_feedback.append({
                "run_date": str(h.get("run_timestamp", "?")),
                "global_score": h.get("global_score", 0),
                "weaknesses": h.get("weaknesses", []),
                "improvements": h.get("improvements", []),
                "redundancies_count": len(h.get("redundancies", [])),
                "scores": h.get("scores", {}),
            })
    except Exception as e:
        logger.warning("Impossible de charger le feedback judge : %s", e)

    # Résumé des faiblesses récurrentes (apparues >=2 fois)
    from collections import Counter as _Ctr
    _all_weaknesses = []
    for jf in judge_feedback:
        _all_weaknesses.extend(jf.get("weaknesses", []))
    _wk_counter = _Ctr(w.lower().strip()[:80] for w in _all_weaknesses)
    recurring_weaknesses = [
        {"issue": issue, "occurrences": cnt}
        for issue, cnt in _wk_counter.most_common(5)
        if cnt >= 2
    ]

    briefing["judge_feedback"] = {
        "has_feedback": len(judge_feedback) > 0,
        "past_runs": len(judge_feedback),
        "latest_score": judge_feedback[0]["global_score"] if judge_feedback else None,
        "history": judge_feedback,
        "recurring_weaknesses": recurring_weaknesses,
    }

    # Persister dans le state pour le sous-agent suivant et la cellule d'inspection
    tool_context.state["a4_briefing"] = briefing

    logger.info(
        "A4 briefing package OK : %d anomalies, %d drivers, budget=%.0f, actual=%.0f, "
        "judge_feedback=%d runs",
        len(enriched_anomalies), len(drivers), total_budget, total_actual,
        len(judge_feedback),
    )

    return briefing


# ════════════════════════════════════════════════════════════════════════════
#  SUB-AGENT 1 : a4_data_loader
#  Utilise UNIQUEMENT le custom tool load_analysis_results.
# ════════════════════════════════════════════════════════════════════════════

A4_LOADER_INSTRUCTION = """Tu es un assistant de préparation de données.
Ta seule mission : appeler l'outil load_analysis_results pour récupérer
et structurer les résultats des agents A1, A2 et A3, ainsi que le feedback
des évaluations qualité passées (A5_Quality_Judge).

Étapes :
1. Appelle load_analysis_results immédiatement.
2. Si le status retourné est "success", confirme brièvement :
   nombre d'anomalies retenues, budget total, réalisé total.
3. Si le briefing contient un champ "judge_feedback" avec has_feedback=true,
   SIGNALE les faiblesses récurrentes et le dernier score qualité.
   Résume-les clairement pour que le sous-agent suivant les prenne en compte.
4. Si le status est "error", rapporte l'erreur telle quelle.

Tu ne fais RIEN d'autre : pas d'analyse, pas de rapport, pas de recherche.
"""


def _detect_report_redundancies(report_text: str, anomalies: list) -> dict:
    """Détecte les anomalies mentionnées de façon redondante dans le rapport.

    Analyse :
    1. Codes comptables cités plus d'une fois dans des sections distinctes
    2. Mêmes constats/chiffres répétés textuellement
    3. Anomalies traitées dans plusieurs axes stratégiques

    Returns:
        dict avec les redondances détectées et un score de redondance.
    """
    # Découper le rapport en sections (## ou ###)
    sections = _re.split(r'^#{2,3}\s+', report_text, flags=_re.MULTILINE)
    section_headers = _re.findall(r'^(#{2,3}\s+.+)$', report_text, flags=_re.MULTILINE)

    # Extraire les codes comptables par section
    code_pattern = _re.compile(r'\b(\d{3,5})\b')
    codes_by_section = {}
    for i, section in enumerate(sections):
        header = section_headers[i - 1] if 0 < i <= len(section_headers) else f"Section_{i}"
        header = header.strip("# ").strip()
        codes_found = code_pattern.findall(section)
        # Ne garder que les codes qui correspondent à des anomalies connues
        known_codes = {str(a.get("code", "")) for a in anomalies}
        relevant_codes = [c for c in codes_found if c in known_codes]
        if relevant_codes:
            codes_by_section[header] = relevant_codes

    # Identifier les codes présents dans plusieurs sections
    all_code_occurrences = []
    for header, codes in codes_by_section.items():
        for code in codes:
            all_code_occurrences.append((code, header))

    code_counts = _Counter(code for code, _ in all_code_occurrences)
    redundant_codes = {code: count for code, count in code_counts.items() if count > 2}

    # Détail des redondances
    redundancy_details = []
    for code, count in sorted(redundant_codes.items(), key=lambda x: x[1], reverse=True):
        sections_with_code = [h for c, h in all_code_occurrences if c == code]
        # Trouver le résumé de l'anomalie
        anomaly_resume = ""
        for a in anomalies:
            if str(a.get("code", "")) == code:
                anomaly_resume = a.get("resume", "")[:80]
                break
        redundancy_details.append({
            "code": code,
            "mentions": count,
            "sections": list(dict.fromkeys(sections_with_code)),  # ordre préservé, pas de dups
            "resume": anomaly_resume,
        })

    # Score de redondance (0 = aucune, 10 = très redondant)
    if not redundancy_details:
        redundancy_score = 0
    else:
        avg_mentions = sum(r["mentions"] for r in redundancy_details) / len(redundancy_details)
        redundancy_score = min(10, round(avg_mentions * len(redundancy_details) / 2, 1))

    return {
        "redundancy_score": redundancy_score,
        "total_redundant_codes": len(redundancy_details),
        "details": redundancy_details,
        "total_sections_analyzed": len(sections),
    }

def load_report_for_judging(tool_context: ToolContext) -> dict:
    """Charge le rapport A4 et les données source pour évaluation qualité.

    Lit le session state et retourne :
    - Le rapport Markdown complet (a4_report)
    - Le briefing package structuré (a4_briefing)
    - Les anomalies retenues par A3 (pour cross-check)
    - L'analyse de redondance intra-rapport
    - L'historique des évaluations passées (mémoire long-terme)
    - Les métadonnées du pipeline (A1/A2 status)
    """

    a4_report = tool_context.state.get("a4_report", "")
    a4_briefing = tool_context.state.get("a4_briefing", {})
    a3_output = tool_context.state.get("a3_output", {})

    if not a4_report:
        return {
            "status": "error",
            "message": "A4 n'a pas produit de rapport. Impossible d'évaluer.",
        }

    if not a4_briefing:
        return {
            "status": "error",
            "message": "Le briefing package A4 est absent. Impossible de cross-checker.",
        }

    # Anomalies retenues pour vérifier la couverture
    retained_cards = a3_output.get("retained_cards", [])
    critical_anomalies = [
        {
            "anomalie_id": c["anomalie_id"],
            "code": c["code"],
            "score": c["score"],
            "niveau": c["niveau"],
            "resume": c["resume"],
        }
        for c in sorted(retained_cards, key=lambda x: x["score"], reverse=True)
    ]

    # Performance globale (données de référence pour vérifier l'exactitude)
    perf = a4_briefing.get("performance_globale", {})
    triage = a4_briefing.get("triage_stats", {})
    enriched = a4_briefing.get("anomalies_retenues", [])

    # ── Détection de redondances ──────────────────────────────────
    redundancy_analysis = _detect_report_redundancies(a4_report, enriched)

    # ── Historique des évaluations passées (mémoire long-terme) ───
    past_judgments = []
    try:
        raw_history = db.get_judge_history(limit=5)
        for h in raw_history:
            past_judgments.append({
                "run_date": str(h.get("run_timestamp", "?")),
                "global_score": h.get("global_score", 0),
                "scores": h.get("scores", {}),
                "weaknesses": h.get("weaknesses", []),
                "improvements": h.get("improvements", []),
                "redundancies_count": len(h.get("redundancies", [])),
            })
    except Exception as e:
        logger.warning("Impossible de charger l'historique judge : %s", e)

    # ── Résumé des patterns récurrents ────────────────────────────
    recurring_issues = []
    if len(past_judgments) >= 2:
        # Collecter toutes les faiblesses passées
        all_past_weaknesses = []
        for pj in past_judgments:
            all_past_weaknesses.extend(pj.get("weaknesses", []))
        # Identifier les faiblesses qui reviennent (mêmes mots-clés)
        weakness_counter = _Counter(w.lower().strip()[:60] for w in all_past_weaknesses)
        recurring_issues = [
            {"issue": issue, "occurrences": count}
            for issue, count in weakness_counter.most_common(5)
            if count >= 2
        ]

    judging_package = {
        "status": "success",
        "message": (
            f"Package de jugement prêt. Rapport de {len(a4_report)} caractères, "
            f"{len(critical_anomalies)} anomalies à cross-checker, "
            f"{len(past_judgments)} évaluations passées chargées."
        ),

        # Le rapport à évaluer
        "rapport_markdown": a4_report,

        # Données de référence pour cross-check
        "reference": {
            "performance_globale": perf,
            "triage_stats": triage,
            "anomalies_retenues": enriched,
            "anomalies_critiques_top": critical_anomalies[:10],
            "total_anomalies_retenues": len(retained_cards),
            "drivers": a4_briefing.get("drivers", [])[:20],
            "rollup_validation": a4_briefing.get("rollup_validation", {}),
        },

        # Analyse de redondance intra-rapport
        "redundancy_analysis": redundancy_analysis,

        # Mémoire long-terme : historique des évaluations passées
        "past_evaluations": {
            "count": len(past_judgments),
            "history": past_judgments,
            "recurring_issues": recurring_issues,
        },

        # Métadonnées pipeline
        "pipeline_metadata": {
            "a1_status": a4_briefing.get("a1_validation", {}).get("status", "?"),
            "a2_status": a4_briefing.get("a2_classification", {}).get("status", "?"),
            "a3_status": a3_output.get("status", "?"),
        },
    }

    # Persister dans le state pour le sous-agent suivant et la cellule d'inspection
    tool_context.state["a5_judging_package"] = judging_package

    logger.info(
        "A5 judging package OK : rapport=%d chars, anomalies=%d, "
        "redondances=%d, historique=%d runs",
        len(a4_report), len(critical_anomalies),
        redundancy_analysis["total_redundant_codes"], len(past_judgments),
    )

    return judging_package


