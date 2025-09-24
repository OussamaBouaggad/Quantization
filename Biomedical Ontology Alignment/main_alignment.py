# -------------------------------------------------------------------------
# Main Alignment Construction and Export
# -------------------------------------------------------------------------

# Initialize a list to store aligned term data
data = []

for idx, match_idx in enumerate(top_matches):
    """
    For each biomedical term in the lexicon, retrieve the most similar MRCONSO entry,
    then extract the relevant metadata for alignment output.
    """
    lex_term = lex_df.iloc[idx]['Biomedical Term']  # Original biomedical term from lexicon
    lex_explication = lex_df.iloc[idx]['Public Explanation']  # Public explanation of the term
    matched_term = mrconso_df.iloc[match_idx]['STR (String)']  # Closest matching term from MRCONSO
    cui_id = mrconso_df.iloc[match_idx]['CUI (Concept Unique Identifier)']  # Unique concept identifier (CUI)
    similarity_score = cosine_sim[idx, match_idx]  # Cosine similarity score of the match

    # Append extracted information as a row to the data list
    data.append([
        lex_term,
        lex_explication,
        matched_term,
        cui_id,
        similarity_score
    ])

# Define the column names for the resulting alignment DataFrame
df_main_alignment_columns = [
    'Biomedical Term',
    'Public Explanation',
    'STR (String)',
    'CUI (Concept Unique Identifier)',
    'Cosine Similarity'
]

# Create the final alignment DataFrame
df_main_alignment = pd.DataFrame(data, columns=df_main_alignment_columns)

# Format the similarity scores to display with two decimal places
df_main_alignment['Cosine Similarity'] = df_main_alignment['Cosine Similarity'].apply(lambda x: f"{x:.2f}")

# Display the first few rows for verification
print(df_main_alignment.head())

# Define output file path
output_file = 'Main_Alignment.csv'

# Export the alignment DataFrame to CSV
df_main_alignment.to_csv(output_file, index=False)

# (Optional) Trigger file download if running in Google Colab
files.download(output_file)