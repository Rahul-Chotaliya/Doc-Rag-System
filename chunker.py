import nltk
from nltk.tokenize import sent_tokenize

def chunk_text_with_overlap(text, target_words=450, overlap_sentences=2):
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = text.split(".")
    chunks, current_chunk = [], []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > target_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sent)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
