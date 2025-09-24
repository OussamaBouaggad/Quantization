# -------------------------------------------------------------------------
# Main Function: Load and Process Lexicon Data
# -------------------------------------------------------------------------

def main():
    """
    Load lexicon data from a CSV file, rename its columns from French to English,
    and print the updated column names for verification.

    The CSV file 'LEX.csv' must be located in the same directory as this script and
    should include the following columns:
        - 'Terme'
        - 'Taux de complexité'
        - 'Indice de comparaison'
        - 'Explication au grand public'
        - 'Taux de simplification'
        - 'Sources'
    """


    ############################################
    # Step 1: Read CSV File into a DataFrame
    ############################################

    lexique = pd.read_csv('LEX.csv')  # Load the CSV file 'LEX.csv' into a DataFrame


    ############################################
    # Step 2: Rename DataFrame Columns
    ############################################

    # Map French column names to English column names.
    lexique.rename(columns={
        'Terme': 'Biomedical Term',
        'Taux de complexité': 'Complexity Rate',
        'Indice de comparaison': 'Comparison Index',
        'Explication au grand public': 'Public Explanation',
        'Taux de simplification': 'Simplification Rate',
        'Sources': 'Sources'
    }, inplace=True)


    ############################################
    # Step 3: Verify the Renaming
    ############################################

    print(lexique.columns)  # Print the updated column names for verification