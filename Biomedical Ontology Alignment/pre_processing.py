# -------------------------------------------------------------------------
# Preprocessing Pipeline for Textual Data
# -------------------------------------------------------------------------

# Load lexicon and MRCONSO data from CSV files
lex_df = pd.read_csv('LEX.csv')  # Biomedical lexicon entries
mrconso_df = pd.read_csv('MRCONSO.csv')  # UMLS concepts (MRCONSO data)

def clean_text(text):
    """
    Normalize text by converting to lowercase, removing non-alphanumeric characters,
    and reducing whitespace.

    Parameters
    ----------
    text : str
        Raw input string to be cleaned.

    Returns
    -------
    str
        Normalized and cleaned version of the input string.
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Replace all non-word characters with space
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def remove_stopwords(text, language='english'):
    """
    Remove stopwords from text using the specified language.

    Parameters
    ----------
    text : str
        The input text from which to remove stopwords.
    language : str, optional
        Language of the stopwords to use (default is 'english').

    Returns
    -------
    str
        Text with stopwords removed.
    """
    stop_words = set(stopwords.words(language))
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

# Load spaCy model for English biomedical text
nlp = spacy.load('en_core_web_lg')  # Ensure 'en_core_web_lg' is installed

def lemmatize_text(text):
    """
    Lemmatize input text using spaCy, reducing words to their base forms.

    Parameters
    ----------
    text : str
        Input text to lemmatize.

    Returns
    -------
    str
        Lemmatized version of the input text.
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def preprocess_text(text):
    """
    Apply a full text preprocessing pipeline: cleaning, stopword removal, and lemmatization.

    Parameters
    ----------
    text : str
        The raw text to preprocess.

    Returns
    -------
    str
        Fully preprocessed version of the input text.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Apply preprocessing to specific columns
lex_df['Cleaned'] = lex_df['Source'].apply(preprocess_text)  # Preprocess 'Source' column in lexicon
mrconso_df['Cleaned'] = mrconso_df['Target'].apply(preprocess_text)  # Preprocess 'Target' column in MRCONSO


# -------------------------------------------------------------------------
# Prepare Data for Similarity Computation
# -------------------------------------------------------------------------

# Convert cleaned text columns to lists for later processing
combined_lex = lex_df['Cleaned'].tolist()  # Preprocessed lexicon entries
cleaned_mrconso = mrconso_df['Cleaned'].tolist()  # Preprocessed UMLS concepts
