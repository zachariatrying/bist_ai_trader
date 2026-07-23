import os
import sys
import subprocess

def run_pip(args):
    print(f"Running pip with args: {args}")
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

try:
    import markdown
    print("markdown package is installed.")
except ImportError:
    print("markdown is not installed. Installing...")
    run_pip(["install", "markdown"])

try:
    import xhtml2pdf
    print("xhtml2pdf package is installed.")
except ImportError:
    print("xhtml2pdf is not installed. Installing...")
    run_pip(["install", "xhtml2pdf"])

import markdown
from xhtml2pdf import pisa

thesis_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
md_path = os.path.join(thesis_dir, "Tez_Metni_Final.md")
pdf_path = os.path.join(thesis_dir, "BorsaNeuron_Graduation_Thesis.pdf")

# Read Markdown text
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split cover page from body
parts = text.split("\n---", 1)
if len(parts) == 2:
    cover_md, body_md = parts[0], parts[1]
    cover_html = markdown.markdown(cover_md, extensions=['tables', 'fenced_code'])
    body_html = markdown.markdown(body_md, extensions=['tables', 'fenced_code'])
    html_body = f'<div class="cover-page">{cover_html}</div><div class="page-break"></div>{body_html}'
else:
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# Let's adjust image source paths in HTML. In markdown, we have ![alt](images/filename.png).
# xhtml2pdf needs absolute paths to find images on Windows!
# Let's replace 'images/' with the full absolute path of the images directory.
abs_images_dir = os.path.join(thesis_dir, "images").replace("\\", "/")
html_body = html_body.replace('src="images/', f'src="{abs_images_dir}/')

# CSS Stylesheet for professional academic Yeditepe University Thesis formatting
css = """
@page {
    size: A4;
    margin: 2.5cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 1cm;
        left: 2.5cm;
        right: 2.5cm;
        height: 1cm;
    }
}
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
    line-height: 1.5;
    color: black;
}
h1, h2, h3, h4, h5, h6 {
    font-family: "Arial", Helvetica, sans-serif;
    color: #111111;
    font-weight: bold;
    page-break-after: avoid;
}
h1 {
    font-size: 18pt;
    margin-top: 1cm;
    margin-bottom: 0.5cm;
    page-break-before: always;
}
h2 {
    font-size: 14pt;
    margin-top: 0.8cm;
    margin-bottom: 0.4cm;
}
h3 {
    font-size: 12pt;
    margin-top: 0.6cm;
    margin-bottom: 0.3cm;
}
p {
    margin-bottom: 0.4cm;
    text-align: justify;
}
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 10pt;
    background-color: #f4f4f4;
}
pre {
    background-color: #f4f4f4;
    padding: 10px;
    border: 1px solid #cccccc;
    page-break-inside: avoid;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 0.6cm;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #aaaaaa;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.5cm auto;
    page-break-inside: avoid;
}
.cover-page {
    text-align: center;
}
.cover-page h1 {
    page-break-before: avoid !important;
    font-size: 20pt;
    margin-top: 1.5cm;
    margin-bottom: 0.8cm;
}
.cover-page h2 {
    font-size: 16pt;
    margin-top: 1cm;
}
.cover-page h3 {
    font-size: 12pt;
    margin-top: 0.5cm;
}
.page-break {
    page-break-after: always;
}
"""

# Let's wrap the HTML body into a complete HTML page
html_document = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
<div id="footerContent" style="text-align: center; font-size: 9pt; color: #555555;">
    Page <pdf:pagenumber> of <pdf:pagecount>
</div>
{html_body}
</body>
</html>
"""

# Convert HTML to PDF
try:
    with open(pdf_path, "wb") as f_pdf:
        pisa_status = pisa.CreatePDF(html_document, dest=f_pdf)
    if pisa_status.err:
        print(f"Error occurred during PDF generation: {pisa_status.err}")
        sys.exit(1)
    else:
        print(f"PDF successfully generated: {pdf_path}")
except PermissionError:
    alt_pdf_path = pdf_path.replace(".pdf", "_v2.pdf")
    print(f"Permission denied on {pdf_path} (it is likely open in a viewer). Saving to alternative path: {alt_pdf_path}")
    with open(alt_pdf_path, "wb") as f_pdf:
        pisa_status = pisa.CreatePDF(html_document, dest=f_pdf)
    if pisa_status.err:
        print(f"Error occurred during alternative PDF generation: {pisa_status.err}")
        sys.exit(1)
    else:
        print(f"Alternative PDF successfully generated: {alt_pdf_path}")
