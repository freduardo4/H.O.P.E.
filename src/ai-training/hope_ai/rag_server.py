import os
import sys
import json
import glob
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_server.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("RAGServer")

class RAGService:
    def __init__(self, doc_dirs, file_extensions=['.md', '.txt']):
        self.doc_dirs = doc_dirs
        self.file_extensions = file_extensions
        self.documents = []
        self.filenames = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_documents(self):
        """Loads and indexes documents from specified directories."""
        self.documents = []
        self.filenames = []
        
        for doc_dir in self.doc_dirs:
            if not os.path.exists(doc_dir):
                logger.warning(f"Directory not found: {doc_dir}")
                continue
                
            logger.info(f"Scanning directory: {doc_dir}")
            for ext in self.file_extensions:
                # Recursive search
                search_pattern = os.path.join(doc_dir, "**", f"*{ext}")
                files = glob.glob(search_pattern, recursive=True)
                
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if len(content.strip()) > 50:  # Skip empty/tiny files
                                self.documents.append(content)
                                self.filenames.append(os.path.abspath(file_path))
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        
        logger.info(f"Loaded {len(self.documents)} documents.")
        
        if self.documents:
            logger.info("Building TF-IDF Index...")
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            logger.info("Index built successfully.")
            
    def query(self, query_text, top_k=3):
        """Retrieves top_k relevant documents for the query."""
        if not self.vectorizer or self.tfidf_matrix is None:
            return {"error": "Index not initialized"}
            
        try:
            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.05:  # Relevance threshold
                    # Extract a relevant snippet (simple heuristic: first 500 chars)
                    # Future improvement: Extract snippet around the matching keywords
                    content = self.documents[idx]
                    snippet = content[:500] + "..." if len(content) > 500 else content
                    
                    results.append({
                        "file": self.filenames[idx],
                        "score": score,
                        "content": snippet
                    })
            
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": str(e)}

def main():
    logger.info("Starting RAG Server...")
    
    # Define directories to index
    # Adjust paths relative to where the script is expected to run (project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    docs_path = os.path.join(project_root, "docs")
    src_desktop_path = os.path.join(project_root, "src", "desktop", "HOPE.Core") # Index core definitions
    
    rag = RAGService([docs_path, src_desktop_path], file_extensions=['.md', '.cs'])
    rag.load_documents()
    
    print("READY", flush=True)
    
    # Input Loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        if line == "EXIT":
            break
            
        logger.info(f"Received query: {line}")
        response = rag.query(line)
        
        # Serialize to JSON and print to stdout
        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()
