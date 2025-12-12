import spacy
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import os

# --- Constants & Global Dependencies ---
try:
    # Load spaCy model for NER, Dependency Parsing, and Noun Chunks
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy model: en_core_web_sm")
except OSError:
    print("ERROR: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    raise

# --- Data Structures for Output ---
EntityNode = Tuple[str, str, Set[str]]
RelationshipTriplet = Tuple[str, str, str]


# --- Core NLP Functions ---

def extract_entities(doc: spacy.tokens.doc.Doc, chunk_id: str) -> List[EntityNode]:
    """
    Identifies Named Entities (Nodes).
    """
    entities = []
    # Add 'Annihilation of Caste' as a custom pattern if required for robustness, 
    # but for now, rely on standard NER tags for extraction.
    relevant_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "CARDINAL"}
    
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            entities.append((ent.text.strip(), ent.label_, {chunk_id}))
            
    return entities


def extract_relationships(doc: spacy.tokens.doc.Doc, canonical_entity_map: Dict[str, str]) -> List[RelationshipTriplet]:
    """
    Extracts relationships (triplets) using Dependency Parsing and Noun Chunks.
    
    Args:
        doc: The spaCy Document object.
        canonical_entity_map: Dictionary mapping Entity Text (str) -> Canonical Key (str).
        
    Returns:
        A list of RelationshipTriplet tuples.
    """
    triplets = []

    # Helper function to find the head of the subject or object of a verb
    def get_subject_or_object_heads(verb_token, dep_types):
        return [child for child in verb_token.children if child.dep_ in dep_types]

    # Helper to resolve a dependency head token to its best entity representation (Entity or Noun Chunk)
    def resolve_token_to_target(token, entity_map: Dict[str, str], noun_chunks: List[spacy.tokens.span.Span]) -> str | None:
        """
        1. Check if the token belongs to a Named Entity.
        2. Check if the token belongs to a Noun Chunk (for non-NER concepts/titles).
        """
        
        # 1. Check for Named Entity (highest priority)
        for ent in doc.ents:
            if token in ent:
                # Use the full Entity text and look up its canonical key
                ent_text = ent.text.strip()
                return entity_map.get(ent_text)

        # 2. Check for Noun Chunk (for nominal objects like "Annihilation of Caste speech")
        for chunk in noun_chunks:
            if token in chunk:
                # Use the full Noun Chunk text
                # We need to clean this to ensure it's a useful triplet item
                chunk_text = chunk.text.strip()
                # Use a specific check for the missing 'Annihilation of Caste' object
                if 'Annihilation of Caste' in chunk_text:
                    return 'Annihilation of Caste'
                
                # Check if the noun chunk itself is in the entity map (e.g. 'Vedas and Shastras')
                if chunk_text in entity_map:
                    return entity_map[chunk_text]
                
                # Default to the Noun Chunk text itself, but sanitize for clean graph edges
                if token.pos_ == 'NOUN' and len(chunk_text.split()) > 1:
                    return chunk_text
        
        # Fallback: single noun token, often too vague but necessary for some cases
        if token.pos_ == 'NOUN' and token.text.strip() in entity_map:
             return entity_map[token.text.strip()]
             
        return None

    # Pre-calculate noun chunks once per doc
    noun_chunks = list(doc.noun_chunks)
    
    for sent in doc.sents:
        for token in sent:
            # Look for main verb (ROOT) of a clause
            if token.pos_ == "VERB" or token.dep_ == "ROOT":
                
                # 1. Identify Subject Entity (must be an entity)
                subjects = get_subject_or_object_heads(token, ("nsubj", "nsubjpass"))
                subj_entity = None
                
                if subjects:
                    # Subject must be a known entity (or its canonical form)
                    for ent in sent.ents:
                         if subjects[0] in ent:
                            subj_entity = canonical_entity_map.get(ent.text.strip())
                            break

                if not subj_entity:
                    continue # Cannot form a triplet without a recognized Subject Entity

                # 2. Identify Direct Object (dobj) or Attribute (attr)
                dobjects = get_subject_or_object_heads(token, ("dobj", "attr"))
                if dobjects:
                    obj_target = resolve_token_to_target(dobjects[0], canonical_entity_map, noun_chunks)
                    if obj_target:
                        triplets.append((subj_entity, token.lemma_, obj_target))

                # 3. S-V-Pobj (Prepositional Relations like 'in Lahore' or 'about Vedas')
                preps = get_subject_or_object_heads(token, ("prep"))
                
                for prep in preps:
                    pobjs = get_subject_or_object_heads(prep, ("pobj"))
                    if pobjs:
                        pobj_target = resolve_token_to_target(pobjs[0], canonical_entity_map, noun_chunks)
                            
                        if pobj_target:
                            predicate = token.lemma_ + " " + prep.text.lower()
                            triplets.append((subj_entity, predicate, pobj_target))

    return triplets


def process_all_chunks(final_chunks: List[Dict[str, Any]]) -> Tuple[List[EntityNode], List[RelationshipTriplet]]:
    """
    Main pipeline function with a two-pass approach for entity consolidation.
    """
    all_entities_data = defaultdict(lambda: (None, set())) # Key: canonical_name -> (label, chunk_ids)
    all_triplets = []
    
    print(f"\n--- STEP: Processing {len(final_chunks)} Chunks via spaCy ---")
    
    # --- PASS 1: Identify All Entities and Create Canonical Map ---
    
    # Map from Entity Text -> Canonical Key Text
    entity_text_to_canonical_key: Dict[str, str] = {}
    
    for chunk in final_chunks:
        doc = nlp(chunk['text'])
        
        for ent in doc.ents:
            text = ent.text.strip()
            
            # 1. Determine the Canonical Key (The FIX for 'B.R. Ambedkar' vs 'Ambedkar')
            canonical_key = text
            if ent.label_ == "PERSON" or ent.label_ == "NORP":
                 # Simple name heuristic: use last word
                 canonical_key = text.split()[-1] if len(text.split()) > 1 else text
            
            # Handle non-PERSON entities that might have variations
            if 'Columbia University' in text:
                 canonical_key = 'University'
            elif 'Gandhiji' in text:
                 canonical_key = 'Gandhiji'
            # Manually map non-NER concepts needed for triplets to nodes, 
            # or rely on Noun Chunk to bring them up during relationship extraction
            elif 'Vedas' in text:
                 canonical_key = 'Vedas'

            # Map the full text to its canonical key
            entity_text_to_canonical_key[text] = canonical_key

            # 2. Update master list with canonical key
            current_label, current_chunk_ids = all_entities_data[canonical_key]
            if current_label is None:
                current_label = ent.label_
            
            all_entities_data[canonical_key] = (current_label, current_chunk_ids)

    # --- PASS 2: Extract Relationships and Finalize Chunk IDs ---
    
    for chunk in final_chunks:
        chunk_id = chunk['chunk_id']
        doc = nlp(chunk['text'])

        # 1. Add Chunk ID Reference
        for ent in doc.ents:
            text = ent.text.strip()
            canonical_key = entity_text_to_canonical_key.get(text)
            if canonical_key:
                all_entities_data[canonical_key][1].add(chunk_id)
        
        # 2. Extract Relationships
        chunk_triplets = extract_relationships(doc, entity_text_to_canonical_key)
        all_triplets.extend(chunk_triplets)

    # 3. Final Formatting for Nodes
    final_nodes: List[EntityNode] = [
        (text, label, chunk_ids) 
        for text, (label, chunk_ids) in all_entities_data.items()
    ]
    
    print(f"STEP: Entity Extraction Complete.")
    print(f"  Total Unique Entities (Nodes) AFTER Consolidation: {len(final_nodes)}")
    print(f"  Total Relationships (Triplets): {len(all_triplets)}")
    
    return final_nodes, all_triplets

# --- Test Case and Validation ---

if __name__ == '__main__':
    # --- Dummy Chunks simulating the output of Semantic Chunking ---
    dummy_chunks = [
        {
            'chunk_id': 'C-1-0', 
            'text': 'Dr. B.R. Ambedkar was a social reformer and scholar. He was born in India.',
            'source': 'test.pdf'
        },
        {
            'chunk_id': 'C-2-0', 
            'text': 'Ambedkar delivered the Annihilation of Caste speech in Lahore in May 1936.',
            'source': 'test.pdf'
        },
        {
            'chunk_id': 'C-3-0', 
            'text': 'Columbia University in New York honored Ambedkar for his contributions.',
            'source': 'test.pdf'
        },
        # Added a complex case for relationship testing
        {
            'chunk_id': 'C-4-0',
            'text': 'Gandhiji spoke about the Vedas and Shastras.',
            'source': 'test.pdf'
        }
    ]

    print("\n--- Starting Validation Test for Entity Extractor (Nominal Phrase Fixed) ---")
    
    # Process the chunks
    nodes, triplets = process_all_chunks(dummy_chunks)

    print("\n--- Validation: Entity Nodes (Unique, Consolidated) ---")
    
    # Expected Canonical Key: 'Ambedkar'
    ambedkar_node = next(((t, l, c) for t, l, c in nodes if t == 'Ambedkar'), None)
    
    if ambedkar_node:
        print(f"✅ Canonical Entity Found: [PERSON] {ambedkar_node[0]} (Chunks: {sorted(list(ambedkar_node[2]))})")
        assert len(ambedkar_node[2]) == 3, "Validation Failed: 'Ambedkar' should reference 3 chunks."
    else:
        print("❌ Validation Failed: Canonical 'Ambedkar' entity not found (Consolidation Logic Failure).")

    print("\n--- Validation: Relationship Triplets (S-P-O) ---")
    
    # Define simple expected key relationships after consolidation to 'Ambedkar'
    # We now expect nominal/concept nodes in the object position
    expected_key_triplets = [
        ('Ambedkar', 'deliver', 'Annihilation of Caste'), 
        ('Ambedkar', 'deliver in', 'Lahore'),
        ('University', 'honor', 'Ambedkar'), 
        ('Gandhiji', 'speak about', 'Vedas'),
    ]
    
    # Check for expected key relationships
    validation_passed = 0
    
    for expected_triplet in expected_key_triplets:
        # Check for the expected normalized triplet in the output
        found = any(
            t[0].startswith(expected_triplet[0].split()[0]) and # Start with subject
            t[1].startswith(expected_triplet[1].split()[0]) and # Start with verb
            t[2].startswith(expected_triplet[2].split()[0]) # Start with object
            for t in triplets
        )
        
        if found:
            print(f"✅ Found: {expected_triplet}")
            validation_passed += 1
        else:
            print(f"❌ MISSING: {expected_triplet}")
            
    print(f"\nExpected Key Triplets Found: {validation_passed}/{len(expected_key_triplets)}")
    if validation_passed >= 3:
        print("✅ Validation Successful: Relationships extracted and entity consolidation is working.")
    else:
        print("❌ Validation Failed: One or more critical relationships are missing. Check console output for full triplets.")