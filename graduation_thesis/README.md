# BorsaNeuron Graduation Thesis & Platform Directory

Welcome to the **BorsaNeuron Graduation Thesis** project directory. This folder contains the complete, academically rigorous graduation thesis and guides for the **BorsaNeuron** algorithmic stock forecasting and technical pattern scanner platform.

## Directory Structure

```
graduation_thesis/
├── README.md                # This project guide & conversion instructions
└── Tez_Metni_Final.md       # Complete Graduation Thesis in academic English (with Turkish ÖZ)
```

## How to Convert the Thesis to PDF or MS Word (.docx)

The graduation thesis is written in highly structured, professional Markdown (`Tez_Metni_Final.md`). To convert it into standard academic formats, you can use any of the following methods:

### Method 1: Using VS Code Markdown PDF Extension (Recommended)
1. Open Visual Studio Code.
2. Search for and install the **Markdown PDF** extension.
3. Open `Tez_Metni_Final.md`.
4. Right-click anywhere in the file editor and select **Markdown PDF: Export (pdf)**.

### Method 2: Using Pandoc (For MS Word .docx or LaTeX)
If you have `pandoc` installed, run the following command in your terminal:
```bash
# Convert to MS Word Document (.docx)
pandoc Tez_Metni_Final.md -o BorsaNeuron_Graduation_Thesis.docx --reference-doc=template.docx

# Convert to PDF via LaTeX
pandoc Tez_Metni_Final.md -o BorsaNeuron_Graduation_Thesis.pdf
```

### Method 3: Online Markdown Editors (Quickest)
1. Copy the full content of `Tez_Metni_Final.md`.
2. Paste it into an online Markdown editor like [Dillinger.io](https://dillinger.io/) or [StackEdit.io](https://stackedit.io/).
3. Export directly as a PDF or HTML file.

---

## How to Run BorsaNeuron Web Application

To run the **BorsaNeuron** Streamlit web dashboard locally:

1. Navigate to the `ipo_analyzer` directory:
   ```bash
   cd C:\Users\ibrah\.gemini\antigravity\scratch\ipo_analyzer
   ```
2. Activate your Python environment (e.g. `tupras_env` or custom conda env).
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```
5. Open your web browser and navigate to `http://localhost:8501`.
