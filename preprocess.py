import spacy
import datefinder
from collections import defaultdict
from tqdm import tqdm
import json
from dateutil.parser import parse
import os
import concurrent.futures

nlp = spacy.load("en_core_web_sm")

def format_date(date_str):
    try:
        # Try to parse the date string
        dt = parse(date_str)
        # Return a standardized format, e.g., YYYY
        return dt.strftime('%Y')
    except (ValueError, OverflowError):
        # If parsing fails, return None or original
        return None

def process_article(article):
    summary = article['summ']
    document = article['document'][:8]  # Select the first 8 entries

    # Sentence splitting function
    def split_sentences(text):
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    # Extract and format date expressions
    def extract_and_format_dates(sentences_with_index):
        date_sentences = []
        for index, sent in enumerate(sentences_with_index):
            doc = nlp(sent)
            dates = [format_date(ent.text) for ent in doc.ents if ent.label_ == 'DATE' and format_date(ent.text)]
            if dates:
                date_sentences.append((index, sent, dates))
        return date_sentences

    # Process summary sentences
    summary_sentences = []
    for sum_sen in summary:
        summary_sentences.extend(split_sentences(sum_sen))

    summary_dates = []
    for sentence in summary_sentences:
        dates = extract_and_format_dates([sentence])
        summary_dates.extend(dates)
        if len(summary_dates) >= 4:  # Stop if 4 or more date-related sentences found
            break

    summary_dates = extract_and_format_dates(summary_sentences)

    # Process document
    doc_sentences = []
    summary_selected = []

    for para in document:
        para_sentences = split_sentences(para)
        selected_sentences = 0
        for sentence in para_sentences:
            if selected_sentences == 3:
                break
            if len(sentence.split()) < 4:  # Skip very short sentences
                continue
            doc_sentences.append(sentence)
            selected_sentences += 1

    # Extract and format dates from document sentences
    doc_dates = extract_and_format_dates(doc_sentences)

    def inter(a, b):
        # Return intersection of two flat string lists
        return list(set(a) & set(b))

    # Match date expressions
    matched_indexes = []
    for (_, sum_sen, summary_date_list) in summary_dates:
        for (index, _, doc_date_list) in doc_dates:
            # Check for any overlapping date tokens
            if inter(summary_date_list, doc_date_list):
                matched_indexes.append(index)
                summary_selected.append(sum_sen)
                break  # Stop after finding the first match

    # Limit number of matched indexes
    matched_indexes = sorted(matched_indexes[:4])

    # Final result
    result = {
        "summ": summary_selected,
        "document": document,
        "sentence": doc_sentences,
        "extract_label": matched_indexes
    }
    return result

# Process a single JSON file line-by-line
def preprocess_json_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        # Count total lines for tqdm progress bar
        lines = infile.readlines()
        total_lines = len(lines)

        # Process each line with multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(json.loads, line) for line in lines]
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_lines):
                data = future.result()
                processed_data = process_article(data)
                json.dump(processed_data, outfile)
                outfile.write('\n')

# Example usage for processing a single JSON file
input_folder = 'wiki/train.json'
output_folder = 'data/train.json'
preprocess_json_file(input_folder, output_folder)

# Batch processing of multiple files (commented)
# def preprocess_multiple_json_files(input_folder, output_folder):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         print("Output folder does not exist")
# 
#     # Iterate over all JSON files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.json'):
#             input_file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename)
#             preprocess_json_file(input_file_path, output_file_path)
