<div align="center">

  # ⚡ ARGO Engine
  ### *Automated Reported & Generative oprtimizer*

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
  [![Gemini AI](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlecloud&logoColor=white)](https://deepmind.google/technologies/gemini/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

  <p align="center">
    <b>ARGO Engine</b> is an end-to-end, high-performance Data Analytics & Quality Assessment Engine built to bridge the gap between heavy, manual Exploratory Data Analysis (EDA) and executive decision-making. Powered by <code>PyArrow</code> vectorization and <code>Google Gemini AI</code>, it processes large-scale datasets instantly and outputs polished executive PDF reports.
  </p>

  <a href="#-quick-start--local-installation"><b>Explore Installation</b></a> •
  <a href="#-key-features"><b>Core Features</b></a> •
  <a href="#-architecture--tech-stack"><b>Tech Stack</b></a>

  ---
</div>

## 📖 Table of Contents
- [The Problem](#-the-problem)
- [Key Features](#-key-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Quick Start / Local Installation](#-quick-start--local-installation)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Author](#-author)

---

## 🎯 The Problem
Data Scientists and Engineers spend up to **80% of their time** performing repetitive manual EDA, diagnosing missing data patterns, detecting outliers, and re-formatting summary charts for business stakeholders.

**ARGO Engine solves this by:**
1. Automating full statistical breakdown and data health audits.
2. Optimizing high-volume parquet/CSV ingestion up to **200MB+** seamlessly.
3. Generating automated executive PDF summaries dynamically tied to live application states.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| ⚡ **High-Speed Ingestion** | Optimized memory handling using `PyArrow` for ultra-fast CSV and Parquet file parsing. |
| 🧹 **Automated Quality Audit** | Instant detection of duplicated rows, missing value ratios, and zero-variance columns. |
| 📊 **Interactive Distributions** | Dynamic Boxplots and Marginal Histograms built via `Plotly` and synced to `Matplotlib` for report exports. |
| 🤖 **AI Copilot** | Deep natural language dataset diagnosis powered by Google's `Gemini AI` LLM API. |
| 📄 **Executive PDF Generation** | Dynamic, vectorized report exporting using `ReportLab`, pulling live session findings directly into production-grade PDFs. |

---

## 🛠️ Architecture & Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Engine:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/docs/python/)
- **Visualizations:** [Plotly Express](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Document Engine:** [ReportLab](https://www.reportlab.com/)
- **Generative AI:** [Google Generative AI (Gemini)](https://ai.google.dev/)

---

## 💻 Quick Start / Local Installation

Follow these steps to run **ARGO Engine** on your local machine:

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed. Check your version with:
```bash
python --version
```
### 2. Clone the Repository
```bash
git clone https://github.com/Mahmoud4265/ARGO-Engine.git
```
### 3. Create & Activate a Virtual Environment
 - On Windows:
```Bash
python -m venv venv
venv\Scripts\activate
```
 - On macOS / Linux:
```Bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```Bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 5. Configure API Secrets
Create a .streamlit/secrets.toml file in the root directory to store your Google Gemini API key securely:
```Ini, TOML
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```
### 6. Run the Application
```Bash
streamlit run app.py
```
The application should now automatically open in your browser at http://localhost:8501! 🚀

### 📁 Project Structure
```Plaintext
ARGO-Engine/
│
├── Datasets                  # Datasets for fixing errors and adjusting
├── eda_engine.py             # Data processing, cleaning algorithms & Gemini API calls
├── pdf_generator.py          # ReportLab flowables & dynamic PDF compilation
├── app.py                    # Main Streamlit UI layout & multi-tab navigation
├── requirements.txt          # Python dependencies
├── .gitignore                # Environment exclusions
├── LICENSE                   # Open-source MIT License
└── README.md                 # Project Documentation
```
### 📜 License
Distributed under the MIT License. See LICENSE for more information.

### 👤 Author
Mahmoud Abdelrahman Ali

GitHub: @Mahmoud4265

Role: AI & Machine Learning Developer / Data Engineer