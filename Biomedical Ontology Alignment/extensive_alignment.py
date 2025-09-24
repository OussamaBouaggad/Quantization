# -------------------------------------------------------------------------
# Extensive Alignment Construction and Export
# -------------------------------------------------------------------------

# Initialize a list to collect aligned term data with extended metadata
data = []

for idx, match_idx in enumerate(top_matches):
    """
    For each biomedical term in the lexicon, retrieve the best matching MRCONSO concept,
    and extract extended metadata including source details, term identifiers, and alignment score.

    Parameters
    ----------
    idx : int
        Index of the current row in the lexicon DataFrame.
    match_idx : int
        Index of the best matching row in the MRCONSO DataFrame.
    """

    # Extract core lexicon fields
    lex_term = lex_df.iloc[idx]['Biomedical Term']  # Term from lexicon
    lex_explication = lex_df.iloc[idx]['Public Explanation']  # Explanation for the general public
    lex_sources = lex_df.iloc[idx]['Sources']  # Source references for the term

    # Define the MRCONSO columns to include in the output
    columns_mrconso = [
        'LAT (Language)',
        'TS (Term Status)',
        'LUI (Lexical Unique Identifier)',
        'STT (String Type)',
        'SUI (String Unique Identifier)',
        'ISPREF (Is Preferred)',
        'AUI (Atom Unique Identifier)',
        'MetaUI (Metadata Unique Identifier)',
        'DescriptorUI (Descriptor Unique Identifier)',
        'SAB (Source Abbreviation)',
        'TTY (Term Type)',
        'Associated DescriptorUI (Descriptor Unique Identifier)',
        'STR (String)',
        'Frequency',
        'SUPPRESS (Suppressible Flag)'
    ]

    # Extract corresponding values from MRCONSO for the best match
    mrconso_values = [mrconso_df.iloc[match_idx][col] for col in columns_mrconso]

    # Extract CUI and similarity score
    cui_id = mrconso_df.iloc[match_idx]['CUI (Concept Unique Identifier)']  # Unique concept identifier
    similarity_score = cosine_sim[idx, match_idx]  # Cosine similarity between aligned terms

    # Compose a complete row for the alignment DataFrame
    row = [lex_term, lex_explication, lex_sources] + mrconso_values + [cui_id, similarity_score]
    data.append(row)

# Define final DataFrame column structure
df_extensive_alignment_columns = [
    'Biomedical Term',
    'Public Explanation',
    'Sources'
] + columns_mrconso + ['CUI (Concept Unique Identifier)', 'Cosine Similarity']

# Create the extensive alignment DataFrame
df_extensive_alignment = pd.DataFrame(data, columns=df_extensive_alignment_columns)

# Format similarity scores to two decimal places for readability
df_extensive_alignment['Cosine Similarity'] = df_extensive_alignment['Cosine Similarity'].apply(lambda x: f"{x:.2f}")

# Preview the first rows of the resulting DataFrame
print(df_extensive_alignment.head())

# Save the extended alignment results to CSV
output_file = 'Extensive_Alignment.csv'
df_extensive_alignment.to_csv(output_file, index=False)

# (Optional) Download the file if using Google Colab
files.download(output_file)