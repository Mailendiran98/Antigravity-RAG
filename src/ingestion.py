import os
import glob
import json
import csv
import io
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

PROCESSED_DATA_DIR = "data/processed"
RAW_DATA_DIR = "data/raw"

def load_tsv_to_dict(tsv_content, key_col):
    """
    Parses a TSV string into a dictionary keyed by key_col.
    Returns the whole row as the value.
    """
    reader = csv.DictReader(io.StringIO(tsv_content), delimiter='\t')
    data = {}
    for row in reader:
        data[row[key_col]] = row
    return data

def process_json_file(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if required keys exist
    if 'sub.txt' not in data or 'num.txt' not in data or 'tag.txt' not in data:
        print(f"Skipping {file_path}: Missing required sub.txt, num.txt, or tag.txt")
        return

    print("Parsing sub.txt...")
    # adsh -> company info
    subs = load_tsv_to_dict(data['sub.txt'], 'adsh')
    
    print("Parsing tag.txt...")
    # tag -> tag info (we need to handle version if necessary, but tag is usually unique enough or we use composite key)
    # Actually, tag.txt has 'tag' and 'version'. num.txt links via 'tag' and 'version'.
    # Let's create a lookup: (tag, version) -> info
    tags = {}
    reader = csv.DictReader(io.StringIO(data['tag.txt']), delimiter='\t')
    for row in reader:
        tags[(row['tag'], row['version'])] = row

    print("Processing num.txt and generating text...")
    reader = csv.DictReader(io.StringIO(data['num.txt']), delimiter='\t')
    
    sentences = []
    for row in reader:
        adsh = row['adsh']
        tag = row['tag']
        version = row['version']
        value = row['value']
        uom = row['uom']
        ddate = row['ddate']
        
        # Lookup Company
        company_name = subs.get(adsh, {}).get('name', 'Unknown Company')
        
        # Lookup Tag Label
        tag_info = tags.get((tag, version), {})
        label = tag_info.get('tlabel', tag) # Fallback to tag code if label missing
        if not label: label = tag
        
        # Construct Sentence
        # "Apple Inc. reported Accounts Payable Current of 123456 USD on 2015-12-31."
        sentence = f"{company_name} reported {label} of {value} {uom} on {ddate}."
        sentences.append(sentence)
        
    output_filename = os.path.basename(file_path) + ".txt"
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    
    print(f"Saving {len(sentences)} sentences to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    json_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
    for json_file in json_files:
        try:
            process_json_file(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    main()
