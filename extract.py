import os
import bibtexparser
import pandas as pd

def parse_bibtex(file_path):
    """Parse a single .bib file and extract relevant fields."""
    with open(file_path, 'r') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    records = []
    for entry in bib_database.entries:
        records.append({
            'title': entry.get('title', ''),
            'abstract': entry.get('abstract', ''),
            'keywords': entry.get('keywords', '').split(', '),  # Split keywords into a list
            'journal': entry.get('journal', '')
        })
    return records

def load_data_from_directory(directory_path):
    """Load data from all .bib files in a directory."""
    all_records = []
    for journal_folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, journal_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.bib'):
                    file_path = os.path.join(folder_path, file)
                    all_records.extend(parse_bibtex(file_path))
    return pd.DataFrame(all_records)

# Load data from your main directory
data_path = "data/"
df = load_data_from_directory(data_path)

# Save for reuse
df.to_csv("processed_data.csv", index=False)

print(df.head())
