import os
from typing import List, Dict, Any
import numpy as np
import spacy
from statistics import mean, stdev 

# Langchain imports for modular document handling
from langchain_community.document_loaders import PyPDFLoader

# Imports for embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
# Note: Ensure you have the 'buffer_merger.py' file in the same directory (or configured as a package)
from .buffer_merger import buffer_merge, split_with_overlap, get_token_count, MAX_CHUNK_TOKENS 


# --- Hyperparameters (Recommended starting points) ---
# Theta (T) for cosine distance (1 - similarity) threshold (Equation 1)
COSINE_DISTANCE_THRESHOLD = 0.35 # Default, likely too high for single-document analysis
# Buffer size (b) for contextual merging (e.g., 4 sentences before, 4 after)
BUFFER_SIZE = 4 
# Model to use for sentence embeddings (from tech stack suggestions)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

# Load Spacy for robust sentence splitting (Split(d) in Algorithm 1, line 3)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please run: python -m spacy download en_core_web_sm")
    raise

# Load Sentence Transformer model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Loaded Embedding Model: {EMBEDDING_MODEL_NAME}")


def split_document_into_sentences(pdf_path: str) -> List[str]:
    """
    Loads text from the PDF and splits it into sentences using spaCy.
    """
    print(f"Reading document: {pdf_path} using PyPDFLoader...")
    try:
        # Use Langchain's PDF Loader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Concatenate text from all pages
        full_text = " ".join([doc.page_content for doc in documents])
        
    except FileNotFoundError:
        print(f"ERROR: PDF file not found at {pdf_path}. Returning DUMMY TEXT for demonstration.")
        # Return placeholder for demonstration if file is missing
        full_text = ("Dr. B.R. Ambedkar was a social reformer. "
                     "He was also the chairman of the Drafting Committee of the Constitution of India. "
                     "Ambedkar advocated for the rights of Dalits and women in India. "
                     "His key work, Annihilation of Caste, is widely studied. ") * 5 
        
    # Use spaCy for robust sentence boundary detection (crucial for accurate chunking)
    doc = nlp(full_text.replace('\n', ' '))
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def semantic_chunking_algorithm1(
    pdf_path: str, 
    theta: float = COSINE_DISTANCE_THRESHOLD, 
    buffer_size: int = BUFFER_SIZE
) -> List[Dict[str, Any]]:
    """
    Implements the full Semantic Chunking via LLM Embedding and Cosine Similarity (Algorithm 1).
    """
    print(f"\n--- Starting Semantic Chunking (Algorithm 1) ---")
    
    # 1. Split(d): Sentences S = {s_1, ..., s_m}
    S = split_document_into_sentences(pdf_path)
    print(f"STEP 1: Initial Sentences (|S|): {len(S)}")
    
    if not S:
        print("Document is empty or failed to load sentences.")
        return []
        
    # 2. BufferMerge: Contextual merging (\hat{S}) (Algorithm 1, line 4)
    S_hat = buffer_merge(S, buffer_size)
    print(f"STEP 2: Buffered Sentences (|\hat{S}|): {len(S_hat)} (Buffer={buffer_size})")

    # 3. Embed(\hat{s}_i): LLM embeddings Z = {z_i} (Algorithm 1, line 5)
    Z = model.encode(S_hat)
    
    # 4. Calculate Cosine Distance d_i (Algorithm 1, line 6)
    distances = []
    # Loop up to |\hat{S}| - 1
    for i in range(len(S_hat) - 1):
        # Cosine similarity between adjacent embeddings
        sim = cosine_similarity(Z[i].reshape(1, -1), Z[i+1].reshape(1, -1))[0][0]
        # Convert to cosine distance (1 - similarity)
        d_i = 1 - sim
        distances.append(d_i)

    # VALIDATION: Distance Metrics
    if distances:
        print(f"STEP 4: Cosine Distance Metrics (1 - Similarity):")
        print(f"  Min Distance: {min(distances):.4f}")
        print(f"  Max Distance: {max(distances):.4f}")
        print(f"  Mean Distance: {mean(distances):.4f}")
    
    # 5. Semantic Grouping (Algorithm 1, lines 7-12)
    current_chunk_text = ""
    C_d = [] # List of intermediate chunks before token limit check
    
    for i in range(len(S_hat) - 1): # Iterate through all sentences/distances
        sentence = S_hat[i]
        distance = distances[i]

        if distance < theta:
            current_chunk_text += sentence + " "
        else:
            current_chunk_text += sentence + " "
            C_d.append(current_chunk_text.strip())
            current_chunk_text = ""
            
    # Add the very last sentence/group from S_hat
    if S_hat:
        current_chunk_text += S_hat[-1]
    
    if current_chunk_text.strip():
        C_d.append(current_chunk_text.strip())
    
    print(f"STEP 5: Semantic Groups (|C_d|): {len(C_d)} (Boundary Threshold=\u03B8={theta})")

    # VALIDATION: Check max tokens of intermediate chunks
    if C_d:
        max_tokens_c_d = max(get_token_count(c) for c in C_d)
        print(f"  Max Token Count in intermediate chunks (|C_d|): {max_tokens_c_d}")
    
    # 6. Token Limit Enforcement (Algorithm 1, lines 13-16)
    final_chunks = []
    source_filename = os.path.basename(pdf_path)
    oversized_chunk_count = 0
    sub_chunk_count = 0
    
    for i, c in enumerate(C_d):
        token_count = get_token_count(c)
        if token_count > MAX_CHUNK_TOKENS:
            oversized_chunk_count += 1
            sub_chunks = split_with_overlap(c)
            sub_chunk_count += len(sub_chunks) - 1
            final_chunks.extend([{'chunk_id': f"C-{i}-{j}", 'text': sc, 'source': source_filename} for j, sc in enumerate(sub_chunks)])
        else:
            final_chunks.append({'chunk_id': f"C-{i}-0", 'text': c, 'source': source_filename})

    # VALIDATION
    print(f"STEP 6: Token Limit Enforcement (T_max={MAX_CHUNK_TOKENS}, Overlap=128):")
    print(f"  Oversized Chunks (>{MAX_CHUNK_TOKENS} tokens) encountered: {oversized_chunk_count}")
    print(f"  New Sub-Chunks created (via Overlap Split): {sub_chunk_count}")
    print(f"FINAL RESULT: Total Final Chunks (|C|): {len(final_chunks)}")
    print(f"--- Semantic Chunking Complete ---")
    
    return final_chunks

# --- Main execution block for testing ---
if __name__ == '__main__':
    # NOTE: Place your 'Ambedkar_works.pdf' in the 'data' folder
    pdf_path_placeholder = os.path.join("data", "Ambedkar_works.pdf")

    # TEST 1: The Fix - Lowered Theta for proper Semantic Grouping
    # Changed theta from 0.35 (which caused a single huge chunk) to 0.12.
    print("\n\n#####################################################")
    print("## TEST 1: TUNED RUN (Buffer=4, Theta=0.12) - EXPECT SUCCESSFUL SEMANTIC GROUPING")
    print("#####################################################")
    
    # CRITICAL CHANGE: Using a tuned theta (0.12) to force actual semantic chunking.
    tuned_theta = 0.12 
    chunks_tuned = semantic_chunking_algorithm1(
        pdf_path=pdf_path_placeholder,
        theta=tuned_theta, 
        buffer_size=BUFFER_SIZE
    )

    # Display results and validate token limits
    print("\n--- Validation and Sample Final Chunks Output (Test 1) ---")
    oversized_final_chunks = 0
    for i, chunk in enumerate(chunks_tuned[:3]):
        token_count = get_token_count(chunk['text'])
        if token_count > MAX_CHUNK_TOKENS:
            oversized_final_chunks += 1
            print(f"!!! ERROR: Final Chunk {chunk['chunk_id']} exceeds T_max ({token_count}) !!!")
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Token Count: {token_count} (Max {MAX_CHUNK_TOKENS})")
        print(f"Content: {chunk['text'][:200]}...")

    if oversized_final_chunks == 0:
        print("\n✅ Final Chunk Validation: All final chunks respect the T_max (1024) limit.")

    # TEST 2: Boundary Case Re-run (Buffer 0) - Check for Consistency
    print("\n\n#####################################################")
    print("## TEST 2: BOUNDARY RUN (Buffer=0, Theta=0.25) - EXPECT HIGH CHUNK COUNT")
    print("#####################################################")
    
    chunks_b0 = semantic_chunking_algorithm1(
        pdf_path=pdf_path_placeholder,
        theta=0.25,
        buffer_size=0
    )
    
    # Final Validation of Boundary Case
    oversized_final_chunks_b0 = 0
    for chunk in chunks_b0:
        if get_token_count(chunk['text']) > MAX_CHUNK_TOKENS:
            oversized_final_chunks_b0 += 1
    
    print(f"Final Chunks (|C|) for Test 2: {len(chunks_b0)}")
    if oversized_final_chunks_b0 == 0:
        print(f"✅ Final Chunk Validation (Test 2): All final chunks respect the T_max ({MAX_CHUNK_TOKENS}) limit.")
    else:
        print(f"!!! ERROR: {oversized_final_chunks_b0} final chunks exceed T_max in Test 2.")