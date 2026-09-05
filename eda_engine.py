import json
import io
import pandas as pd
import plotly.express as px
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import re
import google.generativeai as genai
def get_plottable_columns(df: pd.DataFrame) -> list:

    exact_id_names = {
        'id', 'uuid', 'guid', 'index', 'code', 'row_id', 
        'passengerid', 'customerid', 'userid', 'listingid', 'hostid',
        'unnamed: 0'
    }

    plottable_cols = []
    
    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()
        
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1_\2', col_str)
        tokens = set(re.split(r'[\_\s\-]+', camel_split.lower()))
        
        is_exact_id = col_lower in exact_id_names
        is_ending_with_id = col_lower.endswith('_id') or col_lower.endswith('-id') or col_str.endswith('Id')
        is_starting_with_id = col_lower.startswith('id_') or col_lower.startswith('id-') or col_str.startswith('Id')
        has_standalone_id = 'id' in tokens or 'uuid' in tokens or 'guid' in tokens
        
        
        is_unique_id_type = (
            len(df) > 10 and 
            (df[col].nunique() == len(df)) and 
            (df[col].dtype in ['int64', 'object', 'int32'])
        )
        
        if is_exact_id or is_ending_with_id or is_starting_with_id or has_standalone_id or is_unique_id_type:
            continue
            
        plottable_cols.append(col)
            
    return plottable_cols


class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

def _safe_json_dumps(obj, **kwargs):
    return json.dumps(obj, cls=_NumpySafeEncoder, **kwargs)

# ==========================================
# Phase 1: Data Loading & Validation (Error Handling)
# ==========================================



def load_data(uploaded_file):
    """
    Reads an uploaded file with error handling.
    Validates file extension, empty status, and parser integrity.
    Supports CSV, Excel (.xlsx, .xls), and Parquet formats.
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    filename = uploaded_file.name.lower()

    try:
        uploaded_file.seek(0)

        if filename.endswith(".csv"):
            encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None
            last_err = None

            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if df is None:
                if last_err:
                    raise last_err
                else:
                    return None, "Error: Unable to decode CSV file with supported encodings."

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        elif filename.endswith(".parquet"):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            
            if not file_bytes.startswith(b'PAR1'):
                
                first_few_bytes = file_bytes[:100].decode('utf-8', errors='ignore')
                return None, f"Error: Invalid Parquet file structure. Header content: '{first_few_bytes}'"

            parquet_buffer = io.BytesIO(file_bytes)
            try:
                df = pd.read_parquet(parquet_buffer, engine="pyarrow")
            except Exception:
                parquet_buffer.seek(0)
                df = pd.read_parquet(parquet_buffer, engine="fastparquet")

        else:
            return None, "Error: Unsupported file format. Please upload CSV, Excel, or Parquet."

        if df.empty:
            return None, "Error: The uploaded file is empty."

        return df, None

    except pd.errors.EmptyDataError:
        return None, "Error: File is completely empty (EmptyDataError)."
    except pd.errors.ParserError:
        return None, "Error: Failed to parse file. The file might be corrupted or malformed."
    except Exception as e:
        return None, f"An unexpected error occurred while loading the file: {str(e)}"

# ==========================================
# Task 5: Interactive Charts (Plotly)
# ==========================================

def plot_column_distribution_interactive(df, col):
    """
    Interactive Plotly version of the distribution plot. For numeric
    columns, renders a histogram with a marginal mini-boxplot on top so
    hovering shows the exact bin range/count AND the Q1/median/Q3/outlier
    values in one view. For categorical columns, renders a horizontal bar
    chart of the top 10 categories where hovering shows the exact count
    and percentage of the dataset.
    """
    ignore_cols = [c for c in df.columns if 'id' in c.lower() or 'index' in c.lower() or df[c].nunique() == len(df)]
    drawable_cols = [c for c in df.columns if c not in ignore_cols]
    
    if pd.api.types.is_numeric_dtype(df[col]):
        plot_df = df[[col]].dropna()
        fig = px.histogram(
            plot_df,
            x=col,
            nbins=30,
            marginal="box",
            opacity=0.85,
            title=f"Distribution of Numeric Column: {col}",
            color_discrete_sequence=["#4C78A8"],
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frequency",
            bargap=0.02,
            showlegend=False,
            hovermode="x",
        )
    else:
        total_rows = len(df)
        top_categories = df[col].value_counts().head(10)
        percentages = (top_categories / total_rows * 100).round(2)

        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index.astype(str),
            orientation="h",
            title=f"Top Categories in: {col}",
            labels={"x": "Count", "y": col},
            color=top_categories.values,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(
            customdata=percentages.values.reshape(-1, 1),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Share of dataset: %{customdata[0]}%<extra></extra>",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            showlegend=False,
        )

    return fig


def plot_boxplot_interactive(df, col):
    """
    Standalone interactive Plotly boxplot for a numeric column. Plotly's
    native box hover already reports the exact min/Q1/median/Q3/max and
    lets you hover each individual point beyond the IQR fences to read its
    exact outlier value - so we deliberately leave the default hover
    behavior untouched (a custom hovertemplate would replace those quartile
    numbers with a plain value, which is the opposite of what's wanted).
    """
    plot_df = df[[col]].dropna()
    fig = px.box(
        plot_df,
        y=col,
        points="outliers",
        title=f"Boxplot of {col} — IQR Outlier View",
    )
    fig.update_traces(marker_color="#E45756", boxmean=True)
    fig.update_layout(yaxis_title=col, showlegend=False)
    return fig


def plot_correlation_heatmap_interactive(df):
    """
    Interactive Plotly version of the correlation heatmap: hovering over
    any cell reveals the exact Pearson correlation coefficient between
    that feature pair, and the color scale gives an immediate visual read
    of strength/direction. Uses the same column selection as
    detect_high_correlation (identifiers excluded, low-cardinality numeric
    columns kept).
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    correlation_cols = get_correlation_cols(df, numeric_cols)

    if len(correlation_cols) < 2:
        return None

    corr_matrix = df[correlation_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Heatmap",
    )
    fig.update_traces(
        hovertemplate="%{x} \u2194 %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    )
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    return fig


# ==========================================
# Phase 2: Engine Logic & Issue Detection
# ==========================================

def filter_continuous_numeric_cols(df, numeric_cols, unique_threshold=10):
    """
    Excludes identifiers and discrete numerical columns with low cardinality
    from IQR outlier detection.
    """
    continuous_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= unique_threshold:
            continue
        col_lower = col.lower()
        if col_lower == "id" or col_lower.endswith("_id") or col_lower.startswith("id_"):
            continue
        continuous_cols.append(col)
    return continuous_cols


def check_data_health(df, top_n=5):
    """
    Evaluates duplicate rows and missing value statistics.
    Empty or whitespace-only strings ("", "   ") in text columns are
    treated as missing too, since pandas' isnull() only catches
    NaN/None/NaT and silently misses these — a common source of
    undercounted missing values in real-world CSV exports.
    """
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0

    df_check = df.copy()
    object_cols = df_check.select_dtypes(include=['object']).columns

    missing_sentinels = ["-", "N/A", "n/a", "NA", "na", "null", "None", "", "nan", "NaN", "NULL"]
    for col in object_cols:
        df_check[col] = df_check[col].replace(r'^\s*$', np.nan, regex=True)
        df_check[col] = df_check[col].replace(missing_sentinels, np.nan)

    null_counts = df_check.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    missing_summary = {}
    for col, count in null_cols.items():
        missing_summary[col] = {
            "count": int(count),
            "percentage": round((count / total_rows) * 100, 2)
        }

    sorted_missing = sorted(missing_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]

    return {
        "duplicates": {
            "count": duplicate_rows,
            "percentage": duplicate_pct
        },
        "total_cols_with_missing": len(missing_summary),
        "top_missing_columns": dict(sorted_missing)
    }


def detect_outliers(df, continuous_numeric_cols, top_n=5):
    """
    Identifies numerical outliers using the Interquartile Range (IQR) rule.
    """
    outliers_summary = {}
    total_rows = len(df)

    for col in continuous_numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            outliers_summary[col] = {
                "count": outlier_count,
                "percentage": round((outlier_count / total_rows) * 100, 2)
            }

    sorted_outliers = sorted(outliers_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]
    return {
        "total_numeric_cols_with_outliers": len(outliers_summary),
        "top_outlier_columns": dict(sorted_outliers)
    }


def detect_high_cardinality(df, categorical_cols, threshold=0.5, top_n=5):
    """
    Detects categorical features with an excessively high ratio of unique values.
    """
    total_rows = len(df)
    high_card_cols = {}

    for col in categorical_cols:
        unique_count = df[col].nunique()
        ratio = unique_count / total_rows if total_rows > 0 else 0

        if ratio >= threshold:
            high_card_cols[col] = {
                "unique_count": unique_count,
                "cardinality_ratio": round(ratio, 2)
            }

    sorted_cols = sorted(
        high_card_cols.items(),
        key=lambda x: x[1]['cardinality_ratio'],
        reverse=True
    )[:top_n]

    return {
        "high_cardinality_cols_count": len(high_card_cols),
        "columns": dict(sorted_cols)
    }


def detect_high_correlation(df, numeric_cols, threshold=0.7, top_n=5):
    """
    Identifies feature pairs with Pearson correlation exceeding the threshold.
    """
    if len(numeric_cols) < 2:
        return {"total_high_correlations": 0, "top_correlations": []}

    corr_matrix = df[numeric_cols].corr().abs()
    high_corr_list = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            corr_val = corr_matrix.loc[col1, col2]

            if pd.notnull(corr_val) and corr_val >= threshold:
                high_corr_list.append({
                    "pair": f"{col1} <-> {col2}",
                    "correlation": round(corr_val, 2)
                })

    sorted_corr = sorted(high_corr_list, key=lambda x: x['correlation'], reverse=True)[:top_n]
    return {
        "total_high_correlations": len(high_corr_list),
        "top_correlations": sorted_corr
    }


def detect_target_leakage(df, correlation_cols, threshold=0.98):
    """
    ML-Readiness check: flags numeric feature pairs with a near-perfect
    Pearson correlation (>= threshold). This usually means one column is a
    duplicate, a derived/leaked version of the other, or effectively the
    target itself — training on both (or on the leaked one) tends to
    produce over-optimistic validation metrics that collapse in production.
    """
    if len(correlation_cols) < 2:
        return {"total_leakage_alerts": 0, "alerts": []}

    corr_matrix = df[correlation_cols].corr().abs()
    alerts = []

    for i in range(len(correlation_cols)):
        for j in range(i + 1, len(correlation_cols)):
            col1, col2 = correlation_cols[i], correlation_cols[j]
            corr_val = corr_matrix.loc[col1, col2]
            if pd.notnull(corr_val) and corr_val >= threshold:
                alerts.append({
                    "pair": f"{col1} <-> {col2}",
                    "correlation": round(float(corr_val), 4)
                })

    alerts = sorted(alerts, key=lambda a: a["correlation"], reverse=True)
    return {"total_leakage_alerts": len(alerts), "alerts": alerts}


def detect_constant_features(df):
    """
    ML-Readiness check: flags zero-variance columns - columns with at most
    one distinct non-null value across every row. These carry no signal for
    a model (every row looks the same to it) and are safe to drop.
    """
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    return {"count": len(constant_cols), "columns": constant_cols}


def remove_constant_features(df, constant_cols):
    """
    Returns a copy of df with the given zero-variance columns dropped.
    Used by the 'Remove Constant Features' button once the user confirms
    the columns flagged by detect_constant_features.
    """
    cols_to_drop = [c for c in constant_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ==========================================
# Master Orchestrator Function
# ==========================================

def get_correlation_cols(df, numeric_cols):
    """
    Selects numeric columns eligible for correlation analysis.
    Unlike the outlier filter, this only excludes identifier-like columns
    (e.g. PassengerId) — it deliberately KEEPS low-cardinality numeric
    columns (e.g. Rooms, Bathroom, Bedroom2), since strong correlations
    often occur precisely between these kinds of discrete count columns.
    """
    correlation_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        if col_lower == "id" or col_lower.endswith("_id") or col_lower.startswith("id_"):
            continue
        correlation_cols.append(col)
    return correlation_cols


def detect_issues(df, top_n=5):
    """
    Master orchestrator executing full statistical health checks
    and compiling findings into a unified context dictionary.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    continuous_numeric_cols = filter_continuous_numeric_cols(df, numeric_cols)
    correlation_cols = get_correlation_cols(df, numeric_cols)

    return {
        "dataset_summary": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "numeric_cols_count": len(numeric_cols),
            "categorical_cols_count": len(categorical_cols)
        },
        "data_health": check_data_health(df, top_n=top_n),
        "outliers": detect_outliers(df, continuous_numeric_cols, top_n=top_n),
        "high_cardinality": detect_high_cardinality(df, categorical_cols, top_n=top_n),
        "high_correlations": detect_high_correlation(df, correlation_cols, top_n=top_n),
        "leakage_alerts": detect_target_leakage(df, correlation_cols),
        "constant_features": detect_constant_features(df),
    }


# ==========================================
# Phase 4: AI Layer & LLM Integration (Tasks 13 & 14)
# ==========================================

GEMINI_MODEL = "gemini-3.6-flash"


def _call_gemini(client, prompt, temperature=0.1, max_output_tokens=2048, want_json=False):

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }

    if want_json:
        config_kwargs["response_mime_type"] = "application/json"

    generation_config = genai.GenerationConfig(**config_kwargs)

    try:
        response = client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response  

    except Exception as e:
        if want_json:
            try:
                fallback_config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
                response = client.generate_content(prompt, generation_config=fallback_config)
                return response  
            except Exception:
                pass
        raise Exception(f"Gemini API Error: {str(e)}")


def _response_was_truncated(response):
    """
    Checks whether a Gemini response was cut off because it hit
    max_output_tokens, so callers can warn the user / attempt recovery
    instead of silently returning incomplete text or failing to parse JSON.
    """
    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return False
    return finish_reason is not None and "MAX_TOKENS" in str(finish_reason).upper()


def _parse_ai_json(raw_text):
    """
    Robustly parses a JSON value (object or array) out of raw LLM text.
    Tolerates markdown code fences and - importantly - output that got
    CUT OFF mid-way through generation: instead of failing outright, it
    trims back to the last fully-closed element and closes the JSON
    container itself, so a truncated response still yields whatever
    suggestions/fields were fully generated before the cutoff.

    Returns the parsed Python value, or None if nothing usable could be
    recovered at all.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = [(text.find("["), "[", "]"), (text.find("{"), "{", "}")]
    candidates = [c for c in candidates if c[0] != -1]
    if not candidates:
        return None
    start, open_char, close_char = min(candidates, key=lambda c: c[0])

    end = text.rfind(close_char)
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    remainder = text[start:]
    for cut_marker in ("},", "},\n", "\",", "\",\n"):
        last_cut = remainder.rfind(cut_marker)
        if last_cut != -1:
            repaired = remainder[:last_cut + len(cut_marker.rstrip("\n,"))] + close_char
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    last_close = remainder.rfind("}" if open_char == "[" else "\"")
    if last_close != -1:
        repaired = remainder[:last_close + 1] + close_char
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return None


def build_prompt(issues_report):
    """
    Task 13: Converts the statistical JSON report into a structured LLM prompt.
    Uses json.dumps to guarantee valid JSON syntax (double quotes) inside the
    prompt, instead of relying on Python's dict repr (single quotes), which
    is not valid JSON and could confuse the model or break downstream parsing.
    """
    formatted_report = _safe_json_dumps(issues_report, indent=2)

    prompt = f"""
    You are an expert Data Scientist and Lead Machine Learning Engineer.
    Below is an automated Exploratory Data Analysis (EDA) health report generated for a dataset.

    ### Dataset Health Report (JSON Payload):
    ```json
    {formatted_report}
    ```
    Instructions:
    Provide a crisp, professional Executive Data Health Summary in clean Markdown syntax with the following sections:

    1. Executive Summary: A concise overview of the dataset's structural integrity and overall data quality.

    2. Critical Data Issues Found: Bulleted points of key concerns (Missing values, Duplicates, Outliers, High Cardinality, High Correlation). If leakage_alerts or constant_features contain any entries, call them out explicitly as high-priority ML-readiness risks.

    3. Actionable Remediation Strategy: Clear, step-by-step recommendation steps for the Machine Learning/Data Engineering pipeline to clean and prepare this dataset.

    Keep the ENTIRE response under 500 words total and finish every section
    you start - a short complete answer is much more useful than a long one
    that gets cut off. Do not leave the last sentence or bullet unfinished.
    """
    return prompt


def generate_ai_insights(issues_report, api_key):
    if not api_key:
        return "Error: Gemini API Key is missing. Please provide a valid key in the sidebar."
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-3.6-flash')  
        prompt = build_prompt(issues_report)

        response = _call_gemini(client, prompt, temperature=0.1, max_output_tokens=4096)

        if not response.text:
            return "Error: The AI model returned an empty response."

        output = response.text
        if _response_was_truncated(response):
            output += (
                "\n\n> ⚠️ **Note:** This summary hit the model's output length "
                "limit and was cut off. Click **Generate AI Insights** again."
            )

        return output

    except Exception as e:
        return f"Failed to generate AI insights: {str(e)}"


# ==========================================
# Phase 5: Full Report Export (Task 16) - Markdown & PDF
# ==========================================

def build_markdown_report(report, ai_summary=None):
    """
    Assembles the full statistical report (all detection sections) plus
    the optional AI executive summary into a single Markdown document.
    """
    summary = report["dataset_summary"]
    health = report["data_health"]
    outliers = report["outliers"]
    correlations = report["high_correlations"]
    cardinality = report["high_cardinality"]
    leakage = report.get("leakage_alerts", {"total_leakage_alerts": 0, "alerts": []})
    constants = report.get("constant_features", {"count": 0, "columns": []})

    lines = []
    lines.append("# Automated EDA & Data Health Report")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append(f"- Total Rows: {summary['total_rows']:,}")
    lines.append(f"- Total Columns: {summary['total_cols']}")
    lines.append(f"- Numeric Features: {summary['numeric_cols_count']}")
    lines.append(f"- Categorical Features: {summary['categorical_cols_count']}")
    lines.append("")

    if leakage["total_leakage_alerts"] > 0 or constants["count"] > 0:
        lines.append("## 🚨 ML-Readiness Alerts")
        if leakage["total_leakage_alerts"] > 0:
            lines.append(f"- ⚠️ **Potential Data Leakage:** {leakage['total_leakage_alerts']} feature pair(s) with correlation >= 0.98")
            for a in leakage["alerts"]:
                lines.append(f"  - {a['pair']} (r = {a['correlation']})")
        if constants["count"] > 0:
            lines.append(f"- 🧊 **Constant (Zero-Variance) Columns:** {constants['count']} — {', '.join(constants['columns'])}")
        lines.append("")

    lines.append("## Data Health Overview")
    lines.append(f"- Duplicate Rows: {health['duplicates']['count']} ({health['duplicates']['percentage']}%)")
    lines.append(f"- Columns with Missing Values: {health['total_cols_with_missing']}")
    if health["top_missing_columns"]:
        lines.append("")
        lines.append("| Column | Missing Count | Percentage |")
        lines.append("|---|---|---|")
        for col, v in health["top_missing_columns"].items():
            lines.append(f"| {col} | {v['count']} | {v['percentage']}% |")
    lines.append("")

    lines.append("## Outliers Analysis (IQR Rule)")
    lines.append(f"- Total Numeric Columns with Outliers: {outliers['total_numeric_cols_with_outliers']}")
    if outliers["top_outlier_columns"]:
        lines.append("")
        lines.append("| Column | Outlier Count | Percentage |")
        lines.append("|---|---|---|")
        for col, v in outliers["top_outlier_columns"].items():
            lines.append(f"| {col} | {v['count']} | {v['percentage']}% |")
    lines.append("")

    lines.append("## Highly Correlated Feature Pairs")
    lines.append(f"- Total High Correlation Pairs: {correlations['total_high_correlations']}")
    if correlations["top_correlations"]:
        lines.append("")
        lines.append("| Feature Pair | Correlation |")
        lines.append("|---|---|")
        for c in correlations["top_correlations"]:
            lines.append(f"| {c['pair']} | {c['correlation']} |")
    lines.append("")

    lines.append("## High Cardinality Categorical Features")
    lines.append(f"- High Cardinality Columns Count: {cardinality['high_cardinality_cols_count']}")
    if cardinality["columns"]:
        lines.append("")
        lines.append("| Column | Unique Values | Cardinality Ratio |")
        lines.append("|---|---|---|")
        for col, v in cardinality["columns"].items():
            lines.append(f"| {col} | {v['unique_count']} | {v['cardinality_ratio']} |")
    lines.append("")

    if ai_summary:
        lines.append("## AI Executive Summary")
        lines.append("")
        lines.append(ai_summary)

    return "\n".join(lines)


def _build_pdf_table(rows, header):
    """Helper: builds a styled reportlab Table for the PDF report."""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    data = [header] + rows
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def generate_pdf_report(report, df=None, ai_summary=None):
    """
    Task 16: Builds a formatted PDF version of the full EDA report
    (all statistical sections + visual heatmap + AI executive summary) 
    and returns it as raw bytes, ready for st.download_button.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch
    )
    styles = getSampleStyleSheet()
    alert_style = ParagraphStyle("Alert", parent=styles["Normal"], textColor=colors.HexColor("#b00020"))
    story = []

    summary = report["dataset_summary"]
    health = report["data_health"]
    outliers = report["outliers"]
    correlations = report["high_correlations"]
    cardinality = report["high_cardinality"]
    leakage = report.get("leakage_alerts", {"total_leakage_alerts": 0, "alerts": []})
    constants = report.get("constant_features", {"count": 0, "columns": []})

    # Header
    story.append(Paragraph("Automated EDA & Data Health Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # 1. Dataset Summary
    story.append(Paragraph("Dataset Summary", styles["Heading2"]))
    story.append(_build_pdf_table(
        [
            ["Total Rows", f"{summary['total_rows']:,}"],
            ["Total Columns", str(summary["total_cols"])],
            ["Numeric Features", str(summary["numeric_cols_count"])],
            ["Categorical Features", str(summary["categorical_cols_count"])],
        ],
        ["Metric", "Value"]
    ))
    story.append(Spacer(1, 14))

    # 2. ML-Readiness Alerts
    if leakage["total_leakage_alerts"] > 0 or constants["count"] > 0:
        story.append(Paragraph("ML-Readiness Alerts", styles["Heading2"]))
        if leakage["total_leakage_alerts"] > 0:
            story.append(Paragraph(
                f"⚠ Potential Data Leakage: {leakage['total_leakage_alerts']} feature pair(s) with correlation >= 0.98",
                alert_style
            ))
            story.append(_build_pdf_table(
                [[a["pair"], str(a["correlation"])] for a in leakage["alerts"]],
                ["Feature Pair", "Correlation"]
            ))
            story.append(Spacer(1, 6))
        if constants["count"] > 0:
            story.append(Paragraph(
                f"Constant (Zero-Variance) Columns: {constants['count']} — {', '.join(constants['columns'])}",
                alert_style
            ))
        story.append(Spacer(1, 14))

    # 3. Data Health Overview
    story.append(Paragraph("Data Health Overview", styles["Heading2"]))
    story.append(Paragraph(
        f"Duplicate Rows: {health['duplicates']['count']} ({health['duplicates']['percentage']}%)",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"Columns with Missing Values: {health['total_cols_with_missing']}",
        styles["Normal"]
    ))
    if health["top_missing_columns"]:
        story.append(Spacer(1, 6))
        story.append(_build_pdf_table(
            [[col, str(v["count"]), f"{v['percentage']}%"] for col, v in health["top_missing_columns"].items()],
            ["Column", "Missing Count", "Percentage"]
        ))
    story.append(Spacer(1, 14))

    # 4. Outliers Analysis
    story.append(Paragraph("Outliers Analysis (IQR Rule)", styles["Heading2"]))
    story.append(Paragraph(
        f"Total Numeric Columns with Outliers: {outliers['total_numeric_cols_with_outliers']}",
        styles["Normal"]
    ))
    if outliers["top_outlier_columns"]:
        story.append(Spacer(1, 6))
        story.append(_build_pdf_table(
            [[col, str(v["count"]), f"{v['percentage']}%"] for col, v in outliers["top_outlier_columns"].items()],
            ["Column", "Outlier Count", "Percentage"]
        ))
    story.append(Spacer(1, 14))

    # 5. Correlations & Heatmap
    story.append(Paragraph("Highly Correlated Feature Pairs", styles["Heading2"]))
    story.append(Paragraph(
        f"Total High Correlation Pairs: {correlations['total_high_correlations']}",
        styles["Normal"]
    ))
    if correlations["top_correlations"]:
        story.append(Spacer(1, 6))
        story.append(_build_pdf_table(
            [[c["pair"], str(c["correlation"])] for c in correlations["top_correlations"]],
            ["Feature Pair", "Correlation"]
        ))
    
    # 🎨 Correlation Heatmap
    temp_img_path = None
    if df is not None:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(6, 3.5))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, annot_kws={"size": 8})
            plt.title("Correlation Heatmap", fontsize=10)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name, dpi=150)
                temp_img_path = tmpfile.name
            plt.close()
            
            story.append(Spacer(1, 10))
            story.append(Image(temp_img_path, width=400, height=230))
            
    story.append(Spacer(1, 14))

    # 6. High Cardinality
    story.append(Paragraph("High Cardinality Categorical Features", styles["Heading2"]))
    story.append(Paragraph(
        f"High Cardinality Columns Count: {cardinality['high_cardinality_cols_count']}",
        styles["Normal"]
    ))
    if cardinality["columns"]:
        story.append(Spacer(1, 6))
        story.append(_build_pdf_table(
            [[col, str(v["unique_count"]), str(v["cardinality_ratio"])] for col, v in cardinality["columns"].items()],
            ["Column", "Unique Values", "Cardinality Ratio"]
        ))

    # 7. AI Executive Summary
    if ai_summary:
        story.append(PageBreak())
        story.append(Paragraph("AI Executive Summary", styles["Heading2"]))
        story.append(Spacer(1, 8))
        for line in ai_summary.split("\n"):
            line = line.strip()
            if line:
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe_line, styles["Normal"]))
                story.append(Spacer(1, 4))

    
    doc.build(story)
    
    
    if temp_img_path and os.path.exists(temp_img_path):
        try:
            os.remove(temp_img_path)
        except Exception:
            pass

    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# Phase 6: Auto-Clean & One-Click Pipeline Generation
# ==========================================

def _compute_outlier_bounds(df, continuous_numeric_cols):
    """
    Computes IQR-based lower/upper capping bounds for each continuous
    numeric column. Returns a dict: {col: (lower_bound, upper_bound)}.
    Columns with zero IQR (near-constant distributions) are skipped, since
    capping would collapse them to a single value instead of fixing anything.
    """
    bounds = {}
    for col in continuous_numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return bounds


def apply_auto_fix(df, numeric_impute_strategy='median', categorical_impute_strategy='most_frequent', 
                    outlier_strategy='cap', encoding_strategy='auto', high_cardinality_onehot_limit=10):
    
    df_clean = df.copy()
    fix_summary = {
        "missing_imputed": {},
        "outliers_handled": {},
        "categoricals_encoded": {},
        "constant_cols_dropped": []
    }

    # --- Step 0: Drop Constant Columns (Zero Variance) ---
    constant_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=True) <= 1]
    if constant_cols:
        df_clean.drop(columns=constant_cols, inplace=True)
        fix_summary["constant_cols_dropped"] = constant_cols


    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # --- Step 1: Missing Values Handling ---
    missing_sentinels = ["-", "N/A", "n/a", "NA", "na", "null", "None", "", "nan", "NaN", "NULL"]
    
    # Clean Categorical Missing
    for col in categorical_cols:
        df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True)
        df_clean[col] = df_clean[col].replace(missing_sentinels, np.nan)
        
        if df_clean[col].isnull().sum() > 0:
            if categorical_impute_strategy == "most_frequent":
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Missing"
                df_clean[col] = df_clean[col].fillna(mode_val)
            elif categorical_impute_strategy == "constant":
                df_clean[col] = df_clean[col].fillna("Missing")
            elif categorical_impute_strategy == "drop":
                df_clean.dropna(subset=[col], inplace=True)
            fix_summary["missing_imputed"][col] = categorical_impute_strategy

    # Clean Numeric Missing
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if numeric_impute_strategy == "median":
                fill_val = df_clean[col].median()
            elif numeric_impute_strategy == "mean":
                fill_val = df_clean[col].mean()
            elif numeric_impute_strategy == "drop":
                df_clean.dropna(subset=[col], inplace=True)
                fill_val = None
                
            if fill_val is not None:
                df_clean[col] = df_clean[col].fillna(fill_val)
            fix_summary["missing_imputed"][col] = numeric_impute_strategy

    # --- Step 2: Outliers Handling ---
    if outlier_strategy != "none":
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if outlier_strategy == "cap":
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            elif outlier_strategy == "remove":
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            fix_summary["outliers_handled"][col] = outlier_strategy


    for col in categorical_cols:
        if col not in df_clean.columns:
            continue
            
        n_unique = df_clean[col].nunique()
        
        
        if encoding_strategy == "auto":
            actual_strategy = "onehot" if n_unique <= high_cardinality_onehot_limit else "label"
        else:
            actual_strategy = encoding_strategy

        if actual_strategy == "onehot":
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean.drop(columns=[col]), dummies], axis=1)
            fix_summary["categoricals_encoded"][col] = {"strategy": "onehot"}
            
        elif actual_strategy == "label":
            
            df_clean[col] = df_clean[col].astype('category').cat.codes
            fix_summary["categoricals_encoded"][col] = {"strategy": "label"}
            
        elif actual_strategy == "frequency":
            freq = df_clean[col].value_counts(normalize=True)
            df_clean[col] = df_clean[col].map(freq).fillna(0)
            fix_summary["categoricals_encoded"][col] = {"strategy": "frequency"}

    return df_clean, fix_summary


def _sklearn_imputer_strategy(strategy, fallback):
    """
    Maps our app-level "drop" strategy (which removes rows and can't be
    expressed inside a SimpleImputer) to a safe SimpleImputer strategy for
    the exported pipeline - acting as a no-op safety net for any row that
    slips through the manual pre-pipeline dropna() the generated script
    instructs the user to run first.
    """
    return strategy if strategy != "drop" else fallback


def generate_sklearn_pipeline_code(
    df,
    numeric_impute_strategy="median",
    categorical_impute_strategy="most_frequent",
    outlier_strategy="cap",
    encoding_strategy="auto",
    high_cardinality_onehot_limit=15,
):
    """
    'Download Python (Sklearn Pipeline)' button. Builds a ready-to-run,
    self-contained scikit-learn cleaning pipeline script (as a string) that
    reproduces the EXACT same choices made in apply_auto_fix (imputation
    strategy, outlier handling, encoding scheme) - using the real column
    names and IQR bounds computed from THIS dataset, so an ML engineer can
    drop it straight into a project.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    continuous_numeric_cols = filter_continuous_numeric_cols(df, numeric_cols)
    bounds = _compute_outlier_bounds(df, continuous_numeric_cols) if outlier_strategy == "cap" else {}
    include_capper = outlier_strategy == "cap" and bool(bounds)

    if encoding_strategy == "onehot":
        low_card_cats, high_card_cats = categorical_cols, []
    elif encoding_strategy in ("label", "frequency"):
        low_card_cats, high_card_cats = [], categorical_cols
    else:  # auto
        low_card_cats = [c for c in categorical_cols if 1 < df[c].nunique() <= high_cardinality_onehot_limit]
        high_card_cats = [c for c in categorical_cols if df[c].nunique() > high_cardinality_onehot_limit]

    other_encoder_name = "OrdinalEncoder()" if encoding_strategy == "label" else "FrequencyEncoder()"
    other_encoder_step_name = "ordinal" if encoding_strategy == "label" else "freq_encoder"

    bounds_literal = json.dumps(
        {k: [round(v[0], 4), round(v[1], 4)] for k, v in bounds.items()},
        indent=4
    )

    effective_numeric_strategy = _sklearn_imputer_strategy(numeric_impute_strategy, "median")
    effective_categorical_strategy = _sklearn_imputer_strategy(categorical_impute_strategy, "most_frequent")

    numeric_steps = [f'("imputer", SimpleImputer(strategy={effective_numeric_strategy!r}))']
    if include_capper:
        numeric_steps.append('("outlier_capper", OutlierCapper())')
    numeric_steps_code = ",\n    ".join(numeric_steps)

    manual_notes = []
    if numeric_impute_strategy == "drop" or categorical_impute_strategy == "drop":
        drop_cols = []
        if numeric_impute_strategy == "drop":
            drop_cols += numeric_cols
        if categorical_impute_strategy == "drop":
            drop_cols += categorical_cols
        manual_notes.append(
            '# NOTE: a "drop rows with missing values" strategy was selected for at\n'
            "# least one column type. Row removal can't be expressed inside a\n"
            "# ColumnTransformer (it only transforms columns, not rows), so drop\n"
            "# those rows on your raw dataframe BEFORE calling pipeline.fit_transform(df):\n"
            f"#   cols_to_check = {drop_cols!r}\n"
            "#   df = df.dropna(subset=cols_to_check)\n"
        )
    if outlier_strategy == "remove":
        manual_notes.append(
            '# NOTE: outlier_strategy="remove" was selected. Row removal cannot be\n'
            "# expressed inside a ColumnTransformer either, so apply the same IQR\n"
            "# filter on your raw dataframe BEFORE calling pipeline.fit_transform(df):\n"
            f"#   bounds = {bounds_literal if bounds else '{}'}\n"
            "#   for col, (lo, hi) in bounds.items():\n"
            "#       df = df[(df[col] >= lo) & (df[col] <= hi)]\n"
        )
    manual_steps_note = ("\n" + "\n".join(manual_notes)) if manual_notes else ""

    code = f'''"""
Auto-generated scikit-learn cleaning pipeline.
Generated by the Automated EDA & Data Health Engine.

Config used: numeric_impute_strategy={numeric_impute_strategy!r},
categorical_impute_strategy={categorical_impute_strategy!r},
outlier_strategy={outlier_strategy!r}, encoding_strategy={encoding_strategy!r}.
{manual_steps_note}
Usage:
    import pandas as pd
    from cleaning_pipeline import build_pipeline

    df = pd.read_csv("your_dataset.csv")
    pipeline = build_pipeline()
    X_clean = pipeline.fit_transform(df)
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


NUMERIC_COLS = {numeric_cols!r}
CATEGORICAL_LOW_CARD_COLS = {low_card_cats!r}
CATEGORICAL_HIGH_CARD_COLS = {high_card_cats!r}

# IQR-based capping bounds computed from the dataset used to generate this
# pipeline: {{column: [lower_bound, upper_bound]}}
OUTLIER_BOUNDS = {bounds_literal}


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numeric columns to precomputed IQR bounds (winsorizing)."""

    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else OUTLIER_BOUNDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=NUMERIC_COLS).copy()
        for col, (lower, upper) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].astype(float).clip(lower=lower, upper=upper)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns as their observed frequency."""

    def __init__(self):
        self.freq_maps_ = {{}}

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=CATEGORICAL_HIGH_CARD_COLS)
        for col in X.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=CATEGORICAL_HIGH_CARD_COLS).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps_.get(col, {{}})).fillna(0)
        return X


numeric_pipeline = Pipeline(steps=[
    {numeric_steps_code},
])

low_card_categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy={effective_categorical_strategy!r})),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

high_card_categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy={effective_categorical_strategy!r})),
    ("{other_encoder_step_name}", {other_encoder_name}),
])


def build_pipeline():
    """Builds and returns the full ColumnTransformer-based cleaning pipeline."""
    transformers = []
    if NUMERIC_COLS:
        transformers.append(("numeric", numeric_pipeline, NUMERIC_COLS))
    if CATEGORICAL_LOW_CARD_COLS:
        transformers.append(("cat_low_card", low_card_categorical_pipeline, CATEGORICAL_LOW_CARD_COLS))
    if CATEGORICAL_HIGH_CARD_COLS:
        transformers.append(("cat_high_card", high_card_categorical_pipeline, CATEGORICAL_HIGH_CARD_COLS))

    return ColumnTransformer(transformers=transformers, remainder="drop")


if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("your_dataset.csv")
    # pipeline = build_pipeline()
    # X_clean = pipeline.fit_transform(df)
    pass
'''
    return code


# ==========================================
# Phase 7: AI-Powered Feature Engineering Suggestions
# ==========================================

def build_feature_engineering_prompt(report):
    """
    Builds a prompt asking the LLM to propose new engineered features based
    on the statistical relationships already surfaced in the EDA report
    (mainly the highly-correlated feature pairs and dataset structure), and
    to return the suggestions as strict JSON so the app can render them and
    auto-generate matching pandas code.
    """
    formatted_report = _safe_json_dumps(report, indent=2)

    prompt = f"""
    You are an expert Machine Learning Feature Engineer.
    Below is an automated EDA health report (JSON) for a dataset, including
    the highly correlated numeric feature pairs already detected.

    ### Dataset Health Report (JSON Payload):
    ```json
    {formatted_report}
    ```

    Task:
    Propose up to 5 NEW engineered features that could plausibly improve a
    downstream ML model's predictive accuracy, based only on the columns and
    relationships visible in this report (e.g. ratios or products of
    correlated numeric columns, interaction terms, binned versions of
    skewed/outlier-heavy columns). Do NOT propose features built from any
    column listed under leakage_alerts or constant_features.

    Respond with ONLY a valid JSON array (no markdown fences, no prose
    before or after) where each element has EXACTLY this shape:
    {{
      "feature_name": "string, snake_case",
      "formula": "a pandas expression using df['col_name'] syntax, e.g. df['colA'] / df['colB']",
      "columns_used": ["colA", "colB"],
      "rationale": "one or two sentences explaining WHY this could help a model",
      "estimated_impact_pct": number (a plausible rough estimated percentage
        improvement in model accuracy/R2, e.g. 12)
    }}

    Only reference column names that literally appear in the report above.

    Keep "rationale" to one short sentence per feature and propose AT MOST 5
    features so the full JSON array fits comfortably and is never cut off
    mid-way. Every object in the array must be complete and well-formed.
    """
    return prompt


def generate_feature_suggestions(report, api_key):
    """
    Calls Gemini with a prompt built from the EDA report and parses the
    response into a list of structured feature suggestions. Returns
    (suggestions, error, warning):
      - error is set (and suggestions == []) only on a hard failure
        (missing key, empty response, totally unparseable text).
      - warning is set when the response was cut off mid-generation but
        enough valid JSON was recovered to still show partial suggestions.
    Uses Gemini's native JSON output mode (response_mime_type) instead of
    only asking for JSON in the prompt text, which is the main source of
    "invalid JSON returned" parse failures.
    """
    if not api_key:
        return [], "Error: Gemini API Key is missing. Please provide a valid key in the sidebar.", None

    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-3.6-flash')
        prompt = build_feature_engineering_prompt(report)

        response = _call_gemini(client, prompt, temperature=0.2, max_output_tokens=4096, want_json=True)

        if not response.text:
            return [], "Error: The AI model returned an empty response. Please try again.", None

        suggestions = _parse_ai_json(response.text)

        if suggestions is None:
            return [], (
                "Error: Failed to parse the AI's feature suggestions (invalid JSON returned). "
                "Click 'Suggest Features' again - this is usually a one-off model hiccup."
            ), None
        if not isinstance(suggestions, list):
            return [], "Error: AI response was not a JSON array as expected.", None

        warning = None
        if _response_was_truncated(response):
            warning = (
                f"⚠️ The AI's response was cut off before finishing — showing the "
                f"{len(suggestions)} suggestion(s) that were fully recovered. "
                "Click 'Suggest Features' again for the complete set."
            )

        return suggestions, None, warning

    except Exception as e:
        return [], f"Failed to generate feature suggestions: {str(e)}", None


def build_strategy_recommendation_prompt(report):
    """
    Builds a prompt asking the LLM to recommend Auto-Fix pipeline settings
    (numeric/categorical imputation, outlier handling, encoding) based on
    the patterns already detected in the EDA report, instead of the user
    having to guess between mean/median/mode/drop/etc. themselves.
    """
    formatted_report = _safe_json_dumps(report, indent=2)

    prompt = f"""
    You are an expert Data Scientist choosing data-cleaning settings for an
    automated Auto-Fix pipeline.

    ### Dataset Health Report (JSON Payload):
    ```json
    {formatted_report}
    ```

    Based on the missing-value, outlier, and cardinality patterns in this
    report, recommend the best cleaning configuration for THIS dataset.

    Respond with ONLY a valid JSON object (no markdown fences, no prose
    before or after) with EXACTLY these keys:
    {{
      "numeric_impute_strategy": "median" or "mean" or "drop",
      "categorical_impute_strategy": "most_frequent" or "constant" or "drop",
      "outlier_strategy": "cap" or "remove" or "none",
      "encoding_strategy": "auto" or "onehot" or "label" or "frequency",
      "reasoning": "2-4 sentences explaining why these choices fit this dataset"
    }}

    Guidance:
    - Prefer "median" whenever outliers or skew are present; use "mean" only
      for roughly symmetric numeric columns with few/no outliers.
    - Only recommend "drop" when the missing percentage for the affected
      columns is small enough that dropping those rows won't meaningfully
      shrink the dataset.
    - Prefer "cap" (IQR winsorizing) over "remove" unless the outlier
      percentage is tiny.
    - Prefer "auto" or "onehot" encoding when categorical columns are
      low-cardinality; avoid "onehot" if high-cardinality columns are
      present, since it would explode the column count.
    """
    return prompt


def recommend_cleaning_strategy(report, api_key):
    """
    Asks the AI to recommend Auto-Fix pipeline settings based on the
    patterns in the EDA report. Returns (recommendation, error) where
    recommendation is a dict with numeric_impute_strategy,
    categorical_impute_strategy, outlier_strategy, encoding_strategy, and
    reasoning - any missing/invalid field is silently replaced with a safe
    default so the UI never breaks on a slightly-off AI response.
    """
    if not api_key:
        return None, "API Key is missing."

    try:
        
        prompt = f"""
        You are an expert Data Scientist. Based on this dataset health report, recommend the optimal data cleaning strategy.

        Dataset Report Summary:
        {report}

        Respond ONLY with a valid JSON object. Do NOT include markdown code blocks (like ```json), commentary, or extra text.
        Use this exact JSON structure:
        {{
            "reasoning": "Brief explanation of why these strategies were chosen",
            "numeric_impute_strategy": "median",   # options: "median", "mean", "drop"
            "categorical_impute_strategy": "most_frequent", # options: "most_frequent", "constant", "drop"
            "outlier_strategy": "cap",             # options: "cap", "remove", "none"
            "encoding_strategy": "auto"             # options: "auto", "onehot", "label", "frequency"
        }}
        """


        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        
        clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

        
        strategy_json = json.loads(clean_text)
        return strategy_json, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse the AI's strategy recommendation (invalid JSON returned): {e}"
    except Exception as e:
        return None, str(e)


def generate_feature_engineering_code(suggestions):
    """
    Converts the AI's structured feature suggestions into a ready-to-run
    pandas snippet an engineer can paste directly after loading their
    dataframe as `df`.
    """
    if not suggestions:
        return ""

    lines = [
        "# Auto-generated feature engineering snippet",
        "# Based on AI-suggested feature interactions from the EDA report.",
        "",
    ]
    for s in suggestions:
        name = s.get("feature_name", "new_feature")
        formula = s.get("formula", "")
        rationale = s.get("rationale", "")
        impact = s.get("estimated_impact_pct", "N/A")
        lines.append(f"# {name}: {rationale} (est. impact: {impact}%)")
        lines.append(f"df['{name}'] = {formula}")
        lines.append("")

    return "\n".join(lines)
def generate_eda_report(df):
    """
    Generates a full structured EDA report dictionary containing dataset health,
    outliers, leakage alerts, and cardinality statistics required by app.py and PDF export.
    """
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # 1. Dataset Summary
    dataset_summary = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols)
    }

    # 2. Data Health & Missing Values
    missing_series = df.isnull().sum()
    cols_with_missing = missing_series[missing_series > 0]
    top_missing = {
        col: {
            "count": int(count),
            "percentage": round((count / total_rows) * 100, 2)
        }
        for col, count in cols_with_missing.items()
    }

    duplicates_count = int(df.duplicated().sum())
    duplicates_pct = round((duplicates_count / total_rows) * 100, 2) if total_rows > 0 else 0

    data_health = {
        "duplicates": {"count": duplicates_count, "percentage": duplicates_pct},
        "total_cols_with_missing": len(cols_with_missing),
        "top_missing_columns": top_missing
    }

    # 3. Outliers (IQR Rule)
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = int(((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum())
        if outliers_count > 0:
            outlier_info[col] = {
                "count": outliers_count,
                "percentage": round((outliers_count / total_rows) * 100, 2)
            }

    outliers = {
        "total_numeric_cols_with_outliers": len(outlier_info),
        "top_outlier_columns": outlier_info
    }

    # 4. High Correlations & Leakage Alerts (> 0.98)
    high_corrs = []
    leakage_alerts = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for row in upper_tri.index:
            for col in upper_tri.columns:
                val = upper_tri.loc[row, col]
                if not np.isnan(val) and val > 0.7:
                    pair_str = f"{row} <-> {col}"
                    corr_val = round(float(val), 4)
                    high_corrs.append({"pair": pair_str, "correlation": corr_val})
                    if val > 0.98:
                        leakage_alerts.append({"pair": pair_str, "correlation": corr_val})

    correlations = {
        "total_high_correlations": len(high_corrs),
        "top_correlations": high_corrs
    }

    leakage = {
        "total_leakage_alerts": len(leakage_alerts),
        "alerts": leakage_alerts
    }

    # 5. High Cardinality Check
    high_card_cols = {}
    for col in categorical_cols:
        u_count = int(df[col].nunique(dropna=True))
        ratio = round(u_count / total_rows, 4) if total_rows > 0 else 0
        if u_count > 20:
            high_card_cols[col] = {"unique_count": u_count, "cardinality_ratio": ratio}

    cardinality = {
        "high_cardinality_cols_count": len(high_card_cols),
        "columns": high_card_cols
    }

    # 6. Constant Columns Check
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    constants = {
        "count": len(constant_cols),
        "columns": constant_cols
    }

    return {
        "dataset_summary": dataset_summary,
        "data_health": data_health,
        "outliers": outliers,
        "high_correlations": correlations,
        "leakage_alerts": leakage,
        "high_cardinality": cardinality,
        "constant_features": constants
    }