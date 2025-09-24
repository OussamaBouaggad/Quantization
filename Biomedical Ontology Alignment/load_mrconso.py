# -------------------------------------------------------------------------
# Main Function: Load and Process MRCONSO Data
# -------------------------------------------------------------------------

def load_and_process_mrconso_data():
    """
    Load and process MRCONSO data from a CSV file, assign descriptive column names,
    and print the updated column names for verification.

    The CSV file 'MRCONSO.csv' must be located in the same directory as this script.
    """


    ############################################
    # Step 1: Read CSV File into a DataFrame
    ############################################
    
    mrconso = pd.read_csv('MRCONSO.csv')  # Load the CSV file 'MRCONSO.csv' into a DataFrame


    ############################################
    # Step 2: Assign New Column Names
    ############################################
    # Replace the default column names with descriptive names.

    mrconso.columns = [
        'CUI (Concept Unique Identifier)',
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


    ############################################
    # Step 3: Verify the Column Assignment
    ############################################

    print(mrconso.columns)  # Print the updated column names for verification