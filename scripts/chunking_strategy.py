import nltk

def fixed_size_chunking(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def semantic_chunking(text):
    sentences = nltk.sent_tokenize(text)
    return [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

def overlapping_chunking(text, chunk_size=200, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += (chunk_size - overlap)
    return chunks
