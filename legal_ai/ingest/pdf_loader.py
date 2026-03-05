from pathlib import Path
from PyPDF2 import PdfReader

DATA_DIR = Path(__file__).parent.parent / "data/pdfs"

def load_pdfs():
    texts = []
    for pdf_file in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        texts.append({"file": pdf_file.name, "text": text})
    return texts

if __name__ == "__main__":
    pdf_texts = load_pdfs()
    for doc in pdf_texts:
        print(f"Loaded {doc['file']}, length: {len(doc['text'])}")
