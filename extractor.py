import fitz
import docx

def extract_pages_from_pdf(path: str):
    try:
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": i, "text": text})
        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF: {e}")

def extract_text_from_docx(path: str):
    try:
        doc = docx.Document(path)
        full_text = [p.text for p in doc.paragraphs if p.text.strip()]
        if full_text:
            return [{"page": 1, "text": "\n".join(full_text)}]
        return []
    except Exception as e:
        raise RuntimeError(f"Failed to extract DOCX: {e}")

def extract_pages_from_file(path: str):
    try:
        lower = path.lower()
        if lower.endswith('.pdf'):
            pages = extract_pages_from_pdf(path)
        elif lower.endswith('.docx'):
            pages = extract_text_from_docx(path)
        elif lower.endswith('.txt'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                pages = [{"page": 1, "text": content}] if content else []
        else:
            raise ValueError(f"Unsupported file format: {path}")

        if not pages or all(not p.get("text", "").strip() for p in pages):
            raise ValueError("Document content is empty.")

        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to process file {path}: {str(e)}")
