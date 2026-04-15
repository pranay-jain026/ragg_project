from PyPDF2 import PdfReader

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n".join(text).strip()