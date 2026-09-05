import streamlit as st
import pandas as pd
import os
from eda_engine import (
    load_data,
    detect_issues,
    plot_correlation_heatmap_interactive,
    generate_ai_insights,
    build_markdown_report,
    generate_pdf_report,
    recommend_cleaning_strategy,
    generate_sklearn_pipeline_code,
    apply_auto_fix,
    generate_pdf_report,
    generate_eda_report,
    get_plottable_columns
)
import plotly.express as px
from dotenv import load_dotenv
import numpy as np
import requests
import json
import re
from pdf_generator import generate_executive_pdf

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") 

if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")

@st.cache_data
def load_data_cached(uploaded_file):
    return load_data(uploaded_file)

@st.cache_data
def run_analysis(df, top_n, _file_id):   
    return detect_issues(df, top_n=top_n)


# 1. Page Configuration
st.set_page_config(
    page_title="Automated EDA Engine",
    page_icon="📊",
    layout="wide"
)

# Application Header
st.title("📊 Automated EDA & Data Health Engine")
st.markdown("Upload any CSV, Excel, or Parquet dataset to perform automated statistical analysis, data health checks, and distributions visualization.")

# 2. Sidebar Configuration
st.sidebar.header("⚙️ Engine Controls")
top_n = st.sidebar.slider("Top Results Limit (Top-N)", min_value=3, max_value=15, value=5)

st.sidebar.divider()
st.sidebar.header("🔑 AI Integration Settings")
api_key = st.sidebar.text_input("Gemini API Key", 
                                value=st.session_state.api_key,
                                type="password", 
                                help="Enter your Google Gemini API Key to enable AI Executive Summary.")
st.session_state.api_key = api_key 

# 3. File Uploading & Validation (Multi-Format Support)
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls", "parquet"])


if uploaded_file is not None:
    df, error = load_data_cached(uploaded_file)
    st.session_state.df = df

    if error:
        st.error(f"❌ {error}")
    else:
        st.success("✅ Dataset loaded and validated successfully!")

        
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        report = run_analysis(df, top_n=top_n, _file_id=file_id) 
        if report not in st.session_state:
            with st.spinner("Analyzing dataset health and detecting underlying issues..."):
                st.session_state[report] = run_analysis(df, top_n=top_n, _file_id=file_id)
                report = st.session_state[report]

        summary = report["dataset_summary"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{summary['total_rows']:,}")
        col2.metric("Total Columns", summary["total_cols"])
        col3.metric("Numeric Features", summary["numeric_cols_count"])
        col4.metric("Categorical Features", summary["categorical_cols_count"])

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📋 Missing & Duplicates",
            "⚠️ Outliers",
            "🔗 High Correlation",
            "🔤 High Cardinality",
            "📈 Feature Distributions",
            "🤖 AI Executive Summary",
            "⚡ Auto-Fix & Cleaning",
            "💡 Feature Engineering",
            "📄 PDF Report Export"
        ])

        # ==========================================
        # TAB 1: Data Health
        # ==========================================
        with tab1:
            st.subheader("Data Health Overview")
            health = report["data_health"]

            c1, c2 = st.columns(2)
            c1.info(f"**Duplicate Rows:** {health['duplicates']['count']} ({health['duplicates']['percentage']}%)")
            c2.info(f"**Columns with Missing Values:** {health['total_cols_with_missing']}")

            if health["top_missing_columns"]:
                st.markdown("##### Top Columns by Missing Values Count")
                missing_df = pd.DataFrame.from_dict(health["top_missing_columns"], orient="index")
                missing_df.columns = ["Missing Count", "Percentage (%)"]
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("🎉 No missing values detected in the dataset!")

            st.divider()
            st.subheader("🧹 Zero-Variance Check")

            constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
            if constant_cols:
                st.error(f"⚠️ **Constant Features Found ({len(constant_cols)}):** `{', '.join(constant_cols)}` (Zero Variance)")
                if st.button("🗑️ Remove Constant Features Now", type="primary"):
                    df_cleaned = df.drop(columns=constant_cols)
                    st.session_state.df = df_cleaned
                    st.success(f"Removed {len(constant_cols)} constant columns!")
                    st.rerun()
            else:
                st.success("✅ No constant (zero-variance) features detected.")

        # ==========================================
        # TAB 2: Outliers Detection
        # ==========================================
        with tab2:
            st.subheader("Outliers Analysis (IQR Rule)")
            outliers = report["outliers"]
            st.write(f"**Total Numeric Columns with Outliers:** {outliers['total_numeric_cols_with_outliers']}")

            if outliers["top_outlier_columns"]:
                outliers_df = pd.DataFrame.from_dict(outliers["top_outlier_columns"], orient="index")
                outliers_df.columns = ["Outlier Count", "Percentage (%)"]
                st.dataframe(outliers_df, use_container_width=True)
            else:
                st.success("🎉 No numerical outliers detected!")

        # ==========================================
        # TAB 3: High Correlation Analysis
        # ==========================================
        with tab3:
            st.subheader("🚨 Leakage Radar (Extreme Correlation)")

            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                                   for row in upper_tri.index
                                   for col in upper_tri.columns
                                   if upper_tri.loc[row, col] > 0.98]

                if high_corr_pairs:
                    for col1, col2, val in high_corr_pairs:
                        st.warning(f"⚠️ **Potential Data Leakage detected** between **{col1}** and **{col2}** (Correlation: {val:.4f})!")
                else:
                    st.success("✅ No extreme feature leakage detected (Correlation < 0.98).")
            else:
                st.info("Not enough numeric features to perform leakage scan.")

            st.divider()
            correlations = report["high_correlations"]
            st.write(f"**Total High Correlation Pairs:** {correlations['total_high_correlations']}")

            if correlations["top_correlations"]:
                corr_df = pd.DataFrame(correlations["top_correlations"])
                corr_df.columns = ["Feature Pair", "Pearson Correlation coefficient"]
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info("No feature pairs exceeded the correlation threshold.")

            st.markdown("##### Correlation Heatmap")
            st.caption("🖱️ Hover over any cell to see the exact correlation coefficient.")
            heatmap_fig = plot_correlation_heatmap_interactive(df)
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns to build a correlation heatmap.")

        # ==========================================
        # TAB 4: High Cardinality Analysis
        # ==========================================
        with tab4:
            st.subheader("High Cardinality Categorical Features")
            cardinality = report["high_cardinality"]
            st.write(f"**High Cardinality Columns Count:** {cardinality['high_cardinality_cols_count']}")

            if cardinality["columns"]:
                card_df = pd.DataFrame.from_dict(cardinality["columns"], orient="index")
                card_df.columns = ["Unique Values Count", "Cardinality Ratio"]
                st.dataframe(card_df, use_container_width=True)
            else:
                st.success("🎉 No high cardinality issues detected in categorical features.")

        # ==========================================
        # TAB 5: Feature Visualizer
        # ==========================================
        with tab5:
            st.subheader("📊 Feature Distributions & Outliers")

            if "df" in st.session_state and st.session_state.df is not None:
                df = st.session_state.df

                drawable_cols = get_plottable_columns(df)
                if not drawable_cols:
                    st.warning("There are no columns for visualization (All columns are IDs)")
                else:
                    selected_col = st.selectbox("Select a column to visualize:", drawable_cols)

                if selected_col:
                    import matplotlib.pyplot as plt

                    
                    st.session_state.generated_figs = [] 

                    if pd.api.types.is_numeric_dtype(df[selected_col]):
                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown("##### 📦 Boxplot — Quartiles & Outliers")
                            fig_box = px.box(df, y=selected_col, points="outliers", title=f"Boxplot of {selected_col}")
                            st.plotly_chart(fig_box, use_container_width=True)

                        with col_right:
                            st.markdown("##### 📈 Histogram — Distribution")
                            fig_hist = px.histogram(df, x=selected_col, nbins=30, title=f"Histogram of {selected_col}", marginal="rug")
                            st.plotly_chart(fig_hist, use_container_width=True)

                        
                        fig_pdf, ax = plt.subplots(figsize=(6, 3))
                        df[selected_col].dropna().hist(bins=30, ax=ax, color='#0D6EFD', edgecolor='white')
                        ax.set_title(f"Distribution of {selected_col}")
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel("Frequency")
                        plt.tight_layout()
                        
                        st.session_state.generated_figs.append(fig_pdf)

                    else:
                        st.markdown(f"##### 📊 Category Frequencies for {selected_col}")
                        n_unique = df[selected_col].nunique()

                        if n_unique > 30:
                            st.warning(f"⚠️ Column '{selected_col}' has {n_unique} unique values. Showing Top 20 most frequent.")
                            top_data = df[selected_col].value_counts().head(20).reset_index()
                            top_data.columns = [selected_col, 'Count']
                            fig_bar = px.bar(top_data, x=selected_col, y='Count', title=f"Top 20 Frequencies in {selected_col}")
                        else:
                            cat_data = df[selected_col].value_counts().reset_index()
                            cat_data.columns = [selected_col, 'Count']
                            fig_bar = px.bar(cat_data, x=selected_col, y='Count', title=f"Distribution of {selected_col}")

                        st.plotly_chart(fig_bar, use_container_width=True)

                    
                        fig_pdf, ax = plt.subplots(figsize=(6, 3))
                        top_series = df[selected_col].value_counts().head(10)
                        top_series.plot(kind='bar', ax=ax, color='#0D6EFD')
                        ax.set_title(f"Top Categories in {selected_col}")
                        ax.set_ylabel("Count")
                        plt.tight_layout()

                        st.session_state.generated_figs.append(fig_pdf)

        # ==========================================
        # TAB 6: AI Executive Summary & Export
        # ==========================================
        with tab6:
            st.subheader("🤖 AI Executive Summary & Remediation Strategy")

            report_key = f"ai_report_{uploaded_file.name}_{uploaded_file.size}_{top_n}"

            if not api_key:
                st.warning("⚠️ Please provide a Gemini API Key in the sidebar to generate the AI summary.")
            else:
                if st.button("✨ Generate AI Insights", type="primary"):
                    with st.spinner("Connecting to LLM and synthesizing data report..."):
                        st.session_state[report_key] = generate_ai_insights(report, api_key)

                if report_key in st.session_state:
                    st.markdown("---")
                    st.markdown(st.session_state[report_key])

                    st.divider()
                    st.markdown("##### 📥 Download Full Report")
                    export_format = st.radio(
                        "Choose report format:",
                        options=["Markdown (.md)", "PDF (.pdf)"],
                        horizontal=True,
                        key=f"export_format_{report_key}"
                    )

                    if export_format == "Markdown (.md)":
                        full_md_report = build_markdown_report(report, ai_summary=st.session_state[report_key])
                        st.download_button(
                            label="📥 Download Report (.md)",
                            data=full_md_report,
                            file_name="EDA_Full_Report.md",
                            mime="text/markdown"
                        )
                    else:
                        pdf_bytes = generate_pdf_report(report, ai_summary=st.session_state[report_key])
                        st.download_button(
                            label="📥 Download Report (.pdf)",
                            data=pdf_bytes,
                            file_name="EDA_Full_Report.pdf",
                            mime="application/pdf"
                        )

        # ==========================================
        # TAB 7: Auto-Fix & Cleaning
        # ==========================================
        with tab7:
            st.subheader("⚡ Auto-Fix & One-Click Pipeline Generation")
            st.write("Clean your dataset automatically based on AI suggestions or custom options.")

            if "df" in st.session_state and st.session_state.df is not None:
                df = st.session_state.df

                with st.spinner("🤖 AI is analyzing missing values, outliers, and data types..."):
                    try:
                        ai_rec, ai_err = recommend_cleaning_strategy(report, api_key=api_key)

                        if ai_err:
                            raise Exception(ai_err)
                        ai_reasoning = ai_rec.get('reasoning', 'No specific reasoning provided.')
                        rec_num      = ai_rec.get('numeric_impute_strategy', 'median')
                        rec_cat      = ai_rec.get('categorical_impute_strategy', 'most_frequent')
                        rec_outlier  = ai_rec.get('outlier_strategy', 'cap')
                        rec_enc      = ai_rec.get('encoding_strategy', 'auto')
                    except Exception as e:
                        st.error(f"AI Recommendation Error: {e}")
                        ai_reasoning = "Using rule-based defaults due to API/Report match issue."
                        rec_num, rec_cat, rec_outlier, rec_enc = 'median', 'most_frequent', 'cap', 'auto'

                st.markdown("### 🤖 AI Recommended Strategy")
                st.info(f"💡 **AI Insight & Reasoning:**\n\n{ai_reasoning}")

                if st.button("Apply AI Recommended Strategy Directly", type="primary", use_container_width=True):
                    with st.spinner("Applying AI recommended cleaning strategy..."):
                        cleaned_df, summary = apply_auto_fix(
                            df,
                            numeric_impute_strategy=rec_num,
                            categorical_impute_strategy=rec_cat,
                            outlier_strategy=rec_outlier,
                            encoding_strategy=rec_enc
                        )
                        st.session_state.df_clean = cleaned_df
                        st.session_state.fix_summary = summary

                        st.session_state.pipeline_code = generate_sklearn_pipeline_code(
                            df,
                            numeric_impute_strategy=rec_num,
                            categorical_impute_strategy=rec_cat,
                            encoding_strategy=rec_enc
                        )

                    st.success("✅ Applied AI Recommendation successfully!")
                    st.rerun()

                st.divider()
                st.subheader("🛠️ Or Customize Cleaning Parameters")

                num_options     = ["median", "mean", "drop"]
                cat_options     = ["most_frequent", "constant", "drop"]
                outlier_options = ["cap", "remove", "none"]
                enc_options     = ["auto", "onehot", "label", "frequency"]

                col1, col2 = st.columns(2)

                with col1:
                    num_strat = st.selectbox(
                        "Numeric Imputation Strategy",
                        num_options,
                        index=num_options.index(rec_num) if rec_num in num_options else 0,
                        help="Recommended by AI: " + str(rec_num)
                    )
                    cat_strat = st.selectbox(
                        "Categorical Imputation Strategy",
                        cat_options,
                        index=cat_options.index(rec_cat) if rec_cat in cat_options else 0,
                        help="Recommended by AI: " + str(rec_cat)
                    )

                with col2:
                    outlier_strat = st.selectbox(
                        "Outlier Handling Strategy",
                        outlier_options,
                        index=outlier_options.index(rec_outlier) if rec_outlier in outlier_options else 0,
                        help="Recommended by AI: " + str(rec_outlier)
                    )
                    enc_strat = st.selectbox(
                        "Categorical Encoding Strategy",
                        enc_options,
                        index=enc_options.index(rec_enc) if rec_enc in enc_options else 0,
                        help="Recommended by AI: " + str(rec_enc)
                    )

                if st.button("⚙️ Apply Custom Fix"):
                    with st.spinner("Processing data with custom parameters..."):
                        res = apply_auto_fix(
                            df,
                            numeric_impute_strategy=num_strat,
                            categorical_impute_strategy=cat_strat,
                            outlier_strategy=outlier_strat,
                            encoding_strategy=enc_strat
                        )

                        cleaned_df = res[0] if isinstance(res, tuple) else res
                        st.session_state.df_clean = cleaned_df.copy()

                        st.session_state.pipeline_code = generate_sklearn_pipeline_code(
                            df,
                            numeric_impute_strategy=num_strat,
                            categorical_impute_strategy=cat_strat,
                            encoding_strategy=enc_strat
                        )
                    st.success("✅ Dataset cleaned using custom settings!")
                    st.rerun()

                if "df_clean" in st.session_state and st.session_state.df_clean is not None:
                    df_c = st.session_state.df_clean

                    st.divider()
                    st.subheader("👀 Cleaned Dataset Preview")

                    col_info1, col_info2 = st.columns(2)
                    col_info1.metric("Original Shape", f"{df.shape[0]} rows, {df.shape[1]} cols")
                    col_info2.metric("Cleaned Shape",  f"{df_c.shape[0]} rows, {df_c.shape[1]} cols")

                    st.dataframe(df_c.head(50), use_container_width=True)

                    col_dl1, col_dl2 = st.columns(2)

                    csv_data = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
                    col_dl1.download_button(
                        label="📥 Download Cleaned CSV",
                        data=csv_data,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    if "pipeline_code" in st.session_state:
                        col_dl2.download_button(
                            label="🐍 Download Sklearn Pipeline (.py)",
                            data=st.session_state.pipeline_code,
                            file_name="cleaning_pipeline.py",
                            mime="text/x-python",
                            use_container_width=True
                        )
            else:
                st.warning("⚠️ Please upload a dataset first to enable auto-cleaning.")

        # ==========================================
        # TAB 8: AI Feature Engineering & Sklearn Pipeline Export
        # ==========================================
        with tab8:
            st.markdown("### 💡 AI Feature Engineering & Pipeline Generator")
            st.write("Generate high-impact feature interactions and export them as a production-ready Scikit-Learn Pipeline.")

            df_fe = st.session_state.get('cleaned_df', df).copy()

            col_fe1, col_fe2 = st.columns([1, 1])

            with col_fe1:
                st.markdown("#### 📅 1. Extract Date Features")
                datetime_cols = df_fe.select_dtypes(include=['datetime', 'datetime64', 'object']).columns.tolist()
                possible_date_cols = [c for c in datetime_cols if 'date' in c.lower() or 'time' in c.lower()]
                if not possible_date_cols:
                    st.warning("⚠️ No column containing 'date' or 'time' was found. Please select carefully — all displayed columns are plain text and may not represent actual dates.")
                selected_date_col = st.selectbox("Select Date Column:", possible_date_cols if possible_date_cols else datetime_cols)

                if st.button("Extract Date Parts"):
                    try:
                        temp_date = pd.to_datetime(df_fe[selected_date_col], errors='coerce')
                        df_fe[f"{selected_date_col}_year"]      = temp_date.dt.year
                        df_fe[f"{selected_date_col}_month"]     = temp_date.dt.month
                        df_fe[f"{selected_date_col}_day"]       = temp_date.dt.day
                        df_fe[f"{selected_date_col}_dayofweek"] = temp_date.dt.dayofweek
                        st.session_state['cleaned_df'] = df_fe
                        st.success(f"✅ Extracted Year, Month, Day, and DayOfWeek from '{selected_date_col}'!")
                        st.dataframe(df_fe.head(3))
                    except Exception as e:
                        st.error(f"Could not parse dates: {e}")

            with col_fe2:
                st.markdown("#### ➕ 2. Quick Math Interaction")
                num_cols = df_fe.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

                if len(num_cols) >= 2:
                    col_a     = st.selectbox("Column A (Numerator):",   num_cols, index=0)
                    col_b     = st.selectbox("Column B (Denominator):", num_cols, index=min(1, len(num_cols)-1))
                    operation = st.selectbox("Operation:", ["Multiply", "Divide (Ratio)", "Add", "Subtract"])

                    if st.button("Create Feature"):
                        new_col_name = f"{col_a}_{operation.split()[0].lower()}_{col_b}"
                        if   "Multiply"  in operation: df_fe[new_col_name] = df_fe[col_a] * df_fe[col_b]
                        elif "Divide"    in operation: df_fe[new_col_name] = df_fe[col_a] / (df_fe[col_b].replace(0, np.nan))
                        elif "Add"       in operation: df_fe[new_col_name] = df_fe[col_a] + df_fe[col_b]
                        elif "Subtract"  in operation: df_fe[new_col_name] = df_fe[col_a] - df_fe[col_b]

                        st.session_state['cleaned_df'] = df_fe
                        st.success(f"✅ Created: `{new_col_name}`")
                        st.dataframe(df_fe[[col_a, col_b, new_col_name]].head(3))
                else:
                    st.info("Need at least 2 numeric columns.")

            st.markdown("---")

            st.markdown("#### 🤖 3. AI Predictive Feature Engine & Sklearn Export")
            st.write("Let AI analyze column relationships to predict ML accuracy gains and generate an executable Pipeline.")

            if "ai_pipeline_code"       not in st.session_state: st.session_state["ai_pipeline_code"]       = None
            if "ai_feature_suggestions" not in st.session_state: st.session_state["ai_feature_suggestions"] = None

            if st.button("✨ Analyze & Generate Sklearn Pipeline", type="primary"):
                if not api_key:
                    st.warning("Please enter your Gemini API Key in the sidebar.")
                else:
                    with st.spinner("⚡ AI is analyzing feature interactions and building Sklearn Pipeline..."):
                        try:
                            import google.generativeai as genai
                            import json, re

                            genai.configure(api_key=api_key)

                            generation_config = genai.GenerationConfig(max_output_tokens=600, temperature=0.1)
                            model = genai.GenerativeModel('gemini-3.6-flash', generation_config=generation_config)

                            num_cols_for_ai = df_fe.select_dtypes(include=[np.number]).columns.tolist()[:12]

                            fe_prompt = f"""
You are a Principal Data Scientist. Given these numeric columns:
{num_cols_for_ai}

Return ONLY a valid JSON array (no markdown, no explanation) with exactly 2 objects.
Each object must have these keys:
  "feature_name"   : string  – snake_case name for the new feature
  "col_a"          : string  – must be one of the columns listed above
  "col_b"          : string  – must be one of the columns listed above (different from col_a)
  "operation"      : string  – one of: "multiply", "divide", "add", "subtract"
  "accuracy_boost" : string  – estimated boost e.g. "~3%"
  "reason"         : string  – one sentence explanation

Example format:
[
  {{"feature_name":"revenue_per_unit","col_a":"revenue","col_b":"units","operation":"divide","accuracy_boost":"~4%","reason":"Captures efficiency ratio."}},
  {{"feature_name":"age_x_income","col_a":"age","col_b":"income","operation":"multiply","accuracy_boost":"~2%","reason":"Interaction term boosts tree splits."}}
]
"""
                            raw_response = model.generate_content(fe_prompt).text.strip()

                            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                            if not json_match:
                                raise ValueError(f"AI did not return valid JSON.\nRaw response:\n{raw_response}")

                            suggestions = json.loads(json_match.group())

                            valid_suggestions = []
                            for s in suggestions:
                                if s["col_a"] in df_fe.columns and s["col_b"] in df_fe.columns:
                                    valid_suggestions.append(s)
                                else:
                                    st.warning(f"⚠️ Skipped suggestion '{s.get('feature_name')}' — column not found in dataframe.")

                            if not valid_suggestions:
                                raise ValueError("No valid suggestions after column validation.")

                            st.session_state["ai_feature_suggestions"] = valid_suggestions

                            st.markdown("### 🚀 Recommended Interactions:")
                            ops_symbol = {"multiply": "×", "divide": "÷", "add": "+", "subtract": "−"}
                            for i, s in enumerate(valid_suggestions, 1):
                                symbol = ops_symbol.get(s["operation"], s["operation"])
                                st.markdown(
                                    f"**{i}. `{s['feature_name']}`** → "
                                    f"`{s['col_a']}` {symbol} `{s['col_b']}` | "
                                    f"Expected Boost: **{s['accuracy_boost']}** | "
                                    f"_{s['reason']}_"
                                )

                            transform_lines = []
                            for s in valid_suggestions:
                                col_a, col_b, op, fname = s["col_a"], s["col_b"], s["operation"], s["feature_name"]
                                if   op == "multiply": expr = f"X_out['{col_a}'] * X_out['{col_b}']"
                                elif op == "divide":   expr = f"X_out['{col_a}'] / X_out['{col_b}'].replace(0, np.nan)"
                                elif op == "add":      expr = f"X_out['{col_a}'] + X_out['{col_b}']"
                                elif op == "subtract": expr = f"X_out['{col_a}'] - X_out['{col_b}']"
                                else:                  expr = f"X_out['{col_a}'] + X_out['{col_b}']"
                                transform_lines.append(f"        X_out['{fname}'] = {expr}")

                            transform_block = "\n".join(transform_lines)

                            pipeline_code = f"""import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class ARGOEngineFeatureBuilder(BaseEstimator, TransformerMixin):
    \"\"\"
    Custom Scikit-Learn Transformer generated by ARGO Engine ML Copilot.
    Features engineered by AI based on actual column analysis.
    Suggested interactions:
{chr(10).join(f"      - {s['feature_name']}: {s['col_a']} {s['operation']} {s['col_b']} ({s['reason']})" for s in valid_suggestions)}
    \"\"\"
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # ── AI-Selected Feature Interactions (Real Columns) ──────────────────
{transform_block}

        return X_out

# ── Production-Ready Sklearn Pipeline ────────────────────────────────────────
ARGO_pipeline = Pipeline([
    ('feature_builder', ARGOEngineFeatureBuilder()),
    ('imputer',         SimpleImputer(strategy='median')),
    ('scaler',          StandardScaler()),
])

if __name__ == "__main__":
    print("✅ Engine Sklearn Pipeline loaded successfully!")
    # Example:
    # X_transformed = ARGO_pipeline.fit_transform(raw_df)
"""
                            st.session_state["ai_pipeline_code"] = pipeline_code

                        except json.JSONDecodeError as e:
                            st.error(f"❌ Failed to parse AI response as JSON: {e}\n\nRaw response:\n{raw_response}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            if st.session_state.get("ai_feature_suggestions"):
                st.markdown("---")
                if st.button("⚡ Apply AI Features to DataFrame"):
                    ops = {
                        "multiply": lambda a, b: a * b,
                        "divide":   lambda a, b: a / b.replace(0, np.nan),
                        "add":      lambda a, b: a + b,
                        "subtract": lambda a, b: a - b,
                    }
                    applied = []
                    for s in st.session_state["ai_feature_suggestions"]:
                        op_fn = ops.get(s["operation"])
                        if op_fn:
                            df_fe[s["feature_name"]] = op_fn(df_fe[s["col_a"]], df_fe[s["col_b"]])
                            applied.append(s["feature_name"])
                    st.session_state['cleaned_df'] = df_fe
                    st.success(f"✅ Applied {len(applied)} AI features: {applied}")
                    st.dataframe(df_fe[applied].head(4))

            if st.session_state["ai_pipeline_code"]:
                st.markdown("##### 📦 Export Production-Ready Scikit-Learn Pipeline")
                with st.expander("👀 View Generated `sklearn_pipeline.py` Code"):
                    st.code(st.session_state["ai_pipeline_code"], language="python")

                st.download_button(
                    label="📥 Download Sklearn Pipeline (.py)",
                    data=st.session_state["ai_pipeline_code"],
                    file_name="ARGO_sklearn_pipeline.py",
                    mime="text/x-python"
                )

# ==========================================
        # TAB 9: PDF Report Export
        # ==========================================
        with tab9:
            st.subheader("📄 Executive PDF Report Export")
            st.write("Generate and download a comprehensive executive PDF report based on your current data and AI findings.")

            if st.button("📥 Generate PDF Report"):
                with st.spinner("Building Executive PDF Report with charts..."):
                    try:
                        active_df = st.session_state.get('df_clean', df)

                        df_summary = {
                            "rows": len(active_df),
                            "cols": len(active_df.columns),
                            "duplicates": int(active_df.duplicated().sum()),
                            "missing_total": int(active_df.isnull().sum().sum())
                        }

                        
                        issues_list = []
                        
                        
                        dup_count = df_summary["duplicates"]
                        if dup_count > 0:
                            issues_list.append({
                                "type": "Duplicate Rows",
                                "severity": "Medium",
                                "desc": f"Found {dup_count} duplicated rows in dataset."
                            })

                        
                        missing_by_col = active_df.isnull().sum()
                        cols_with_missing = missing_by_col[missing_by_col > 0]
                        if not cols_with_missing.empty:
                            for col_name, m_cnt in cols_with_missing.items():
                                pct = (m_cnt / len(active_df)) * 100
                                issues_list.append({
                                    "type": "Missing Values",
                                    "severity": "High" if pct > 20 else "Low",
                                    "desc": f"Column '{col_name}' has {m_cnt} missing values ({pct:.1f}%)."
                                })

                        
                        for col in active_df.columns:
                            if active_df[col].nunique() == 1:
                                issues_list.append({
                                    "type": "Constant Column",
                                    "severity": "Low",
                                    "desc": f"Column '{col}' has only 1 unique value."
                                })

                        # 4. جلب ملخص AI المخزن في session_state (أو نص افتراضي إذا لم يُولّد بعد)
                        ai_insights_text = (
                            st.session_state.get('ai_summary_text') or 
                            st.session_state.get('ai_insights') or 
                            st.session_state.get('ai_report') or 
                            "No AI insights generated yet. You can generate them from the AI Copilot tab."
                        )

                        
                        fig_list = st.session_state.get('generated_figs', [])
                        
                        
                        if not fig_list and active_df.isnull().sum().sum() > 0:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(6, 3))
                            active_df.isnull().sum().plot(kind='bar', ax=ax, color='#0D6EFD')
                            ax.set_title("Missing Values per Column")
                            ax.set_ylabel("Count")
                            plt.tight_layout()
                            fig_list = [fig]

                        
                        pdf_buf = generate_executive_pdf(
                            df_summary=df_summary,
                            issues_list=issues_list,
                            ai_insights_text=ai_insights_text,
                            fig_list=fig_list
                        )

                        st.download_button(
                            label="⬇️ Download Executive PDF Now",
                            data=pdf_buf.getvalue(),
                            file_name="ARGO_engine_executive_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF generated successfully!")

                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

            st.divider()
            st.subheader("💬 Your Feedback Matters")

            with st.form("feedback_form_final", clear_on_submit=True):
                q1 = st.text_area("1. What feature did you like the most?", placeholder="Type your answer here...")
                q2 = st.text_area("2. What feature would you like us to add right away?", placeholder="Type your suggestions...")
                q3 = st.text_area("3. Did you encounter any issues or challenges?", placeholder="Describe any issue you faced...")

                user_contact = st.text_input("Email or Phone (Optional, if you'd like us to reply):", placeholder="name@example.com")

                submit_button = st.form_submit_button("Submit Feedback 🚀")

            if submit_button:
                if q1 or q2 or q3:
                    YOUR_EMAIL = "mahmoud.eladly.ai@gmail.com"
                    try:
                        payload = {
                            "Best Feature":      q1 if q1 else "N/A",
                            "Requested Feature": q2 if q2 else "N/A",
                            "Issues Faced":      q3 if q3 else "N/A",
                            "User Contact":      user_contact if user_contact else "Anonymous",
                            "_subject":          "New Feedback from Streamlit App (Report Tab)!"
                        }
                        headers = {
                            "Content-Type": "application/json",
                            "Accept":       "application/json",
                            "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                            "Referer":      "http://localhost:8501"
                        }
                        response = requests.post(f"https://formsubmit.co/ajax/{YOUR_EMAIL}", json=payload, headers=headers)
                        res_data = response.json()
                        if response.status_code == 200 and res_data.get("success") == "true":
                            st.success("Thank you for your feedback! It has been sent successfully. ❤️")
                        else:
                            st.error(f"Failed to send feedback: {res_data.get('message', response.text)}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please answer at least one question before submitting.")

# ==========================================
# No file uploaded
# ==========================================
if uploaded_file is None:
    st.info("👈 Please upload a dataset (CSV, Excel, or Parquet) from the sidebar to begin analysis.")
    st.markdown(
        """
        <div style="
            background-color: rgba(28, 131, 225, 0.05);
            border: 1px solid rgba(28, 131, 225, 0.2);
            border-left: 5px solid #0068c9;
            padding: 16px 20px;
            border-radius: 8px;
            margin-top: 15px;
            line-height: 1.6;
        ">
            <h4 style="margin: 0 0 10px 0; color: #0068c9; font-size: 1.05rem; font-weight: 600;">
                🔒 100% Data Privacy & Security Guaranteed:
            </h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.92rem; color: inherit;">
                <li style="margin-bottom: 6px;">
                    <b>Zero Raw Data Stored:</b> Your files are processed entirely in memory and deleted instantly after your session.
                </li>
                <li style="margin-bottom: 6px;">
                    <b>BYOK Privacy:</b> You use your own Gemini API Key. Your data is encrypted and never used to train AI models <i>(per Google Cloud API Enterprise Privacy Policy)</i>.
                </li>
                <li>
                    <b>Metadata-Only Processing:</b> Our AI Engine only reads statistical metrics (null counts, column types, correlations)—your actual rows and sensitive data <b>NEVER</b> leave your browser/session.
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader("💬 Your Feedback Matters")

    with st.form("feedback_form_upload", clear_on_submit=True):
        q1 = st.text_area("1. What feature did you like the most?", placeholder="Type your answer here...")
        q2 = st.text_area("2. What feature would you like us to add right away?", placeholder="Type your suggestions...")
        q3 = st.text_area("3. Did you encounter any issues or challenges?", placeholder="Describe any issue you faced...")

        user_contact = st.text_input("Email or Phone (Optional, if you'd like us to reply):", placeholder="name@example.com")

        submit_button = st.form_submit_button("Submit Feedback 🚀")

    if submit_button:
        if q1 or q2 or q3:
            YOUR_EMAIL = "mahmoud.eladly.ai@gmail.com"
            try:
                payload = {
                    "Best Feature":      q1 if q1 else "N/A",
                    "Requested Feature": q2 if q2 else "N/A",
                    "Issues Faced":      q3 if q3 else "N/A",
                    "User Contact":      user_contact if user_contact else "Anonymous",
                    "_subject":          "New Feedback from Streamlit App (Upload Page)!"
                }
                headers = {
                    "Content-Type": "application/json",
                    "Accept":       "application/json",
                    "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer":      "http://localhost:8501"
                }
                response = requests.post(f"https://formsubmit.co/ajax/{YOUR_EMAIL}", json=payload, headers=headers)
                res_data = response.json()
                if response.status_code == 200 and res_data.get("success") == "true":
                    st.success("Thank you for your feedback! It has been sent successfully. ❤️")
                else:
                    st.error(f"Failed to send feedback: {res_data.get('message', response.text)}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please answer at least one question before submitting.")