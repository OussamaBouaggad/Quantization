# -------------------------------------------------------------------------
# Final Alignment Merge and Export
# -------------------------------------------------------------------------

# Load the additional UMLS metadata table (left-side merge input)
df_umls = pd.read_csv('UMLS_LeftMerge.csv')  # UMLS supplementary data with extra metadata

# Merge the extensive alignment with the new UMLS metadata using the CUI as key
df_final_alignment = pd.merge(
    df_extensive_alignment,
    df_umls,
    on='CUI (Concept Unique Identifier)',
    how='left'  # Perform a left join to preserve all aligned lexicon terms
)
"""
Merge df_extensive_alignment with additional UMLS metadata.

Parameters
----------
df_extensive_alignment : pandas.DataFrame
    The DataFrame containing lexicon terms aligned with MRCONSO entries.
df_umls : pandas.DataFrame
    A DataFrame containing extended metadata related to CUIs.

Returns
-------
df_final_alignment : pandas.DataFrame
    A DataFrame combining aligned terms with additional UMLS context.
"""

# Preview the merged result for verification
print(df_final_alignment.head())

# Save the final merged alignment to a CSV file
output_file = 'Final_Alignment.csv'
df_final_alignment.to_csv(output_file, index=False)

# (Optional) Download the file if using Google Colab
files.download(output_file)