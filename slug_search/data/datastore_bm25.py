import argparse
import pandas as pd
from datasets import load_dataset
from haystack import Document
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)
from typing import List, Dict, Any
import math

# Import all utility functions from datastore.py
from datastore import (
    make_unique_doc_id,
    extract_document_id_from_query_metadata,
    check_if_ground_truth_retrieved,
    preprocess_and_chunk_text,
    create_haystack_documents,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load HF dataset, chunk text, and store in Elasticsearch for BM25 retrieval."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lucadiliello/hotpotqa",
        help="Name or path of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="context",
        help="Column in the dataset containing the text to chunk.",
    )
    parser.add_argument(
        "--elasticsearch_host",
        type=str,
        default="http://localhost:9200",
        help="Hostname or URL for the Elasticsearch instance.",
    )
    parser.add_argument(
        "--elasticsearch_index_name",
        type=str,
        default="haystack_bm25_documents",
        help="Name of the Elasticsearch index to use.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional: Maximum number of documents from the dataset to process.",
    )
    parser.add_argument(
        "--drop_old_index",
        action="store_true",
        help="If set, delete all documents from the Elasticsearch index before writing new ones.",
    )
    parser.add_argument(
        "--metadata_columns",
        nargs="*",
        default=[],
        help="List of column names from the dataset to include as metadata.",
    )
    parser.add_argument(
        "--preprocess_function",
        type=str,
        default="preprocess_and_chunk_text",
        help="Name of the preprocessing function to use for text.",
    )
    return parser.parse_args()


def load_hf_dataset_to_dataframe(
    dataset_name: str, split_name: str, max_docs: int = None
) -> pd.DataFrame:
    print(f"Loading dataset: {dataset_name}, split: {split_name}")
    dataset = load_dataset(dataset_name, split=split_name)
    df = dataset.to_pandas()
    if max_docs and max_docs < len(df):
        print(f"Using a subset of {max_docs} documents.")
        df = df.head(max_docs)
    print(f"Loaded {len(df)} documents into DataFrame.")
    return df


if __name__ == "__main__":
    args = parse_args()
    print("Configuration:")
    print(f"  Dataset Name: {args.dataset_name}, Split: {args.split_name}")
    print(f"  Text Column: {args.text_column}")
    print(f"  Metadata Columns: {args.metadata_columns}")
    print(f"  Elasticsearch Host: {args.elasticsearch_host}")
    print(f"  Elasticsearch Index: {args.elasticsearch_index_name}")
    print(f"  Max Documents: {args.max_docs if args.max_docs else 'All'}")
    print(f"  Preprocessing Function: {args.preprocess_function}")

    dataframe = load_hf_dataset_to_dataframe(
        args.dataset_name, args.split_name, args.max_docs
    )
    if dataframe.empty:
        print("Loaded DataFrame is empty. Exiting.")
        exit()
    if args.text_column not in dataframe.columns:
        print(
            f"Error: Text column '{args.text_column}' not found. Available: {dataframe.columns.tolist()}"
        )
        exit()
    for mc in args.metadata_columns:
        if mc not in dataframe.columns:
            print(
                f"Warning: Specified metadata column '{mc}' not found in DataFrame. It will be skipped."
            )
    preprocess_function_to_use = globals().get(args.preprocess_function)
    if not callable(preprocess_function_to_use):
        print(
            f"Warning: Preprocessing function '{args.preprocess_function}' not found or not callable. "
            f"Defaulting to 'preprocess_and_chunk_text'."
        )
        preprocess_function_to_use = preprocess_and_chunk_text
    documents_to_process = create_haystack_documents(
        df=dataframe,
        text_column=args.text_column,
        metadata_columns=args.metadata_columns,
        preprocess_fn=preprocess_function_to_use,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
    )
    if not documents_to_process:
        print("No documents were created to store. Exiting.")
        exit()
    print("Initializing ElasticsearchDocumentStore...")
    document_store = ElasticsearchDocumentStore(
        hosts=[args.elasticsearch_host], index=args.elasticsearch_index_name
    )
    print(
        f"ElasticsearchDocumentStore initialized for index: {args.elasticsearch_index_name} at {args.elasticsearch_host}"
    )
    if args.drop_old_index:
        print(
            f"Deleting all documents from index '{args.elasticsearch_index_name}' due to --drop_old_index flag..."
        )
        document_store.delete_documents(index=args.elasticsearch_index_name)
        print(
            f"Successfully deleted documents from index '{args.elasticsearch_index_name}'."
        )
    print(f"Writing {len(documents_to_process)} documents to Elasticsearch...")
    document_store.write_documents(documents_to_process)
    print(f"Successfully wrote documents to the store.")
    print(f"Total documents in store: {document_store.count_documents()}")
    print("Pipeline execution finished.")
