from sentence_transformers import SentenceTransformer
import re
import numpy as np

class KamradtModifiedChunker:
    def __init__(self, avg_chunk_size=300, min_chunk_size=50, embedding_function=None):
        self.avg_chunk_size = avg_chunk_size
        self.min_chunk_size = min_chunk_size
        self.embedding_function = embedding_function
        
        # If no embedding function provided, use SentenceTransformer
        if self.embedding_function is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_function = lambda texts: [model.encode(text).tolist() for text in texts]
    
    def split_text(self, text):
        # Initial text splitting based on paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If average paragraph size is too large, split further
        current_avg_size = sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs))
        
        if current_avg_size > self.avg_chunk_size:
            # Split larger paragraphs into sentences
            split_paragraphs = []
            for p in paragraphs:
                if len(p.split()) > self.avg_chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    split_paragraphs.extend(sentences)
                else:
                    split_paragraphs.append(p)
            paragraphs = split_paragraphs
        
        # Group paragraphs into chunks based on semantic similarity
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed target size and we already have content
            if current_chunk_size + len(paragraph.split()) > self.avg_chunk_size and current_chunk_size > self.min_chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_chunk_size = len(paragraph.split())
            else:
                current_chunk.append(paragraph)
                current_chunk_size += len(paragraph.split())
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks