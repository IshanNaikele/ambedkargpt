import tiktoken
from typing import List

# --- Hyperparameters from SEMRAG Paper ---
# Max tokens for a single chunk (Token limit T_max)
MAX_CHUNK_TOKENS = 1024 
# Max tokens for sub-chunks overlap (intersection size)
OVERLAP_TOKENS = 128

def get_token_count(text: str) -> int:
    """Uses a common LLM tokenizer (like gpt-4) to count tokens."""
    # Assuming 'cl100k_base' encoder is a good proxy for general LLM token counting
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def buffer_merge(sentences: List[str], buffer_size: int) -> List[str]:
    """
    Applies Contextual Merging (BufferMerge) from Algorithm 1, line 4.
    This step combines adjacent sentences around a central sentence to preserve contextual coherence.
    
    Args:
        sentences: List of original sentences.
        buffer_size: Number of adjacent sentences combined around a central sentence (b).
    
    Returns:
        List of buffered sentences/sentence groups (\hat{S}).
    """
    if buffer_size <= 0:
        return sentences
        
    buffered_sentences = []
    num_sentences = len(sentences)

    for i in range(num_sentences):
        # Determine the start and end indices for merging
        start = max(0, i - buffer_size)
        end = min(num_sentences, i + buffer_size + 1)
        
        # Combine the window of sentences
        merged_sentence = " ".join(sentences[start:end])
        buffered_sentences.append(merged_sentence)
        
    return buffered_sentences


def split_with_overlap(chunk: str) -> List[str]:
    """
    Splits a chunk that exceeds the token limit (T_max=1024) into smaller, 
    overlapping sub-chunks, as defined by Equation 2.
    
    Equation 2: g = \bigcup_{j=1}^{m} g_j, where g_j \cap g_{j+1} \neq \emptyset, 
    |g_j| \le 1024, and |g_j \cap g_{j+1}| = 128 .
    
    Args:
        chunk: The large text chunk (c) to be split.
        
    Returns:
        A list of smaller, overlapping sub-chunks (g_j).
    """
    if get_token_count(chunk) <= MAX_CHUNK_TOKENS:
        return [chunk]

    # Use a simpler text split for overlap based on character count 
    # as an approximation of token count for robust splitting
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(chunk)
    
    sub_chunks = []
    current_index = 0
    
    while current_index < len(tokens):
        # Determine the end of the current sub-chunk
        end_index = min(current_index + MAX_CHUNK_TOKENS, len(tokens))
        
        # Decode the tokens back to text for the sub-chunk
        sub_chunk_tokens = tokens[current_index:end_index]
        sub_chunk = encoder.decode(sub_chunk_tokens)
        sub_chunks.append(sub_chunk)
        
        # Calculate the start index for the next sub-chunk with overlap
        # The next chunk starts at (end_index - OVERLAP_TOKENS)
        overlap_tokens_end = min(current_index + MAX_CHUNK_TOKENS - OVERLAP_TOKENS, len(tokens))

        if overlap_tokens_end <= current_index:
            # If the overlap doesn't move the index forward (or is too small), 
            # we must still advance the index to prevent an infinite loop.
            current_index = end_index 
        else:
            current_index = overlap_tokens_end
            
    return sub_chunks
    

if __name__ == '__main__':
    # --- Example Usage for Testing ---
    print("--- Testing Buffer Merging ---")
    sentences = [f"Sentence {i}." for i in range(10)]
    buffered = buffer_merge(sentences, buffer_size=1)
    print(f"Original: {sentences}")
    print(f"Buffer Size 1: {buffered}")

    print("\n--- Testing Overlap Splitting (Equation 2) ---")
    # Create a very long string that will certainly exceed 1024 tokens
    long_text = "This is a sentence. " * 300 
    print(f"Original Token Count: {get_token_count(long_text)}")
    
    sub_chunks = split_with_overlap(long_text)
    print(f"Number of Sub-Chunks: {len(sub_chunks)}")
    for i, sc in enumerate(sub_chunks):
        token_count = get_token_count(sc)
        print(f"Sub-Chunk {i} Token Count: {token_count} (Max {MAX_CHUNK_TOKENS})")

        # Verify overlap for adjacent chunks
        if i > 0:
            last_chunk = sub_chunks[i-1]
            overlap_text = last_chunk[-(len(sc) - sc.find(last_chunk.split()[-1].split('.')[0])):] # Rough overlap verification
            # In a real scenario, you'd calculate token overlap explicitly, 
            # but for a basic test, we'll confirm the token count is respected.
            pass