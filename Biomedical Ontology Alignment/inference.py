# -------------------------------------------------------------------------
# Embedding Inference and Term Alignment
# -------------------------------------------------------------------------

# Load tokenizer and model from Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')  # Biomedical tokenizer
model = AutoModel.from_pretrained('microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')  # Biomedical language model

def encode_texts_in_batches(texts, batch_size=10):
    """
    Encode a list of texts into dense vector embeddings using a pretrained model.
    The function processes texts in batches to optimize memory usage.

    Parameters
    ----------
    texts : list of str
        List of preprocessed input texts to be encoded.
    batch_size : int, optional
        Number of texts to process at a time (default is 10).

    Returns
    -------
    torch.Tensor
        A tensor containing the vector embeddings of all input texts.
    """
    model.eval()  # Set model to evaluation mode
    embeddings_list = []  # Initialize list to hold batch embeddings

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]  # Slice batch
        with torch.no_grad():  # Disable gradient tracking
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            output = model(**encoded_input)  # Forward pass
        batch_embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
        embeddings_list.append(batch_embeddings)

    embeddings = torch.cat(embeddings_list, dim=0)  # Concatenate all batches
    return embeddings

# Encode lexicon and MRCONSO text data into embeddings
lex_embeddings = encode_texts_in_batches(combined_lex)  # Vector representations for lexicon terms
mrconso_embeddings = encode_texts_in_batches(cleaned_mrconso)  # Vector representations for MRCONSO terms


# -------------------------------------------------------------------------
# Cosine Similarity and Matching
# -------------------------------------------------------------------------

# Compute cosine similarity matrix between lexicon and MRCONSO embeddings
cosine_sim = cosine_similarity(lex_embeddings.numpy(), mrconso_embeddings.numpy())

# Identify best match (highest cosine similarity) for each lexicon term
top_matches = np.argmax(cosine_sim, axis=1)  # Index of most similar MRCONSO entry for each lexicon term

# Display alignment results with cosine similarity scores
for idx, match_idx in enumerate(top_matches):
    lex_term = lex_df.iloc[idx]['Biomedical Term']
    lex_explication = lex_df.iloc[idx]['Public Explanation']
    matched_term = mrconso_df.iloc[match_idx]['STR (String)']
    similarity_score = cosine_sim[idx, match_idx]

    print(f"{lex_term} ({lex_explication}) aligns with {matched_term} with a cosine similarity of {similarity_score:.2f}")