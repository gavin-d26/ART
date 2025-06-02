import argparse
import pandas as pd
from datasets import load_dataset
from haystack import Document
from milvus_haystack.document_store import MilvusDocumentStore
from typing import List, Dict, Any
import math
import hashlib

# VLLM Imports
from vllm import LLM, EngineArgs


# --- Query ID and Ground Truth Utilities ---
def make_unique_doc_id(
    row: dict, dataset_name: str, split_name: str, index: int
) -> str:
    """
    Generate a unique document ID for a row, matching the logic used when writing to Milvus.
    """
    derived_dataset_identifier = (
        dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    )
    if "id" in row and row["id"] is not None and str(row["id"]).strip():
        original_doc_id_base = str(row["id"])
        if not original_doc_id_base.startswith(
            f"{derived_dataset_identifier}_{split_name}"
        ):
            return f"{derived_dataset_identifier}_{split_name}_{original_doc_id_base}"
        else:
            return original_doc_id_base
    else:
        return f"{derived_dataset_identifier}_{split_name}_{index}"


def extract_document_id_from_query_metadata(
    query_row: dict, dataset_name: str, split_name: str, index: int
) -> str:
    """
    Extract the document ID that corresponds to a query for ground-truth verification.
    Uses the same logic as make_unique_doc_id.
    """
    return make_unique_doc_id(query_row, dataset_name, split_name, index)


def check_if_ground_truth_retrieved(
    retrieved_chunks: List[Dict], query_document_id: str
) -> Dict[str, Any]:
    """
    Check if any ground-truth chunks were retrieved for a given query.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries with metadata
        query_document_id: Document ID corresponding to the query

    Returns:
        Dictionary with retrieval analysis results
    """
    ground_truth_chunks = []
    other_chunks = []

    for chunk in retrieved_chunks:
        # Try to get meta dict, fallback to top-level if not present
        meta = chunk.get("meta")
        if isinstance(meta, dict):
            retrieved_original_doc_id = meta.get("original_doc_id")
        else:
            retrieved_original_doc_id = chunk.get("original_doc_id")
        if retrieved_original_doc_id == query_document_id:
            ground_truth_chunks.append(chunk)
        else:
            other_chunks.append(chunk)

    return {
        "ground_truth_retrieved": len(ground_truth_chunks) > 0,
        "num_ground_truth_chunks": len(ground_truth_chunks),
        "num_other_chunks": len(other_chunks),
        "ground_truth_chunks": ground_truth_chunks,
        "other_chunks": other_chunks,
        "total_retrieved": len(retrieved_chunks),
    }


# --- Configuration & Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Load HF dataset, embed text using VLLM, and store in Milvus."
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
        help="Column in the dataset containing the text to embed.",
    )
    parser.add_argument(
        "--milvus_db_path",
        type=str,
        default="./milvus_pipeline.db",
        help="Path for the Milvus Lite DB file.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional: Maximum number of documents from the dataset to process.",
    )
    parser.add_argument(
        "--drop_old_db",
        action="store_true",
        help="If set, drop the old Milvus database before writing new documents.",
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

    # Add VLLM engine arguments to the parser
    parser = EngineArgs.add_cli_args(parser)

    # Set VLLM specific defaults
    # The --model argument is added by EngineArgs.add_cli_args
    parser.set_defaults(
        model="BAAI/bge-large-en-v1.5",  # Default VLLM model for embeddings
        task="embed",  # Crucial for telling VLLM to do embedding
        # enforce_eager=True,  # As per the example provided
        # Note: Other VLLM arguments (e.g., tensor_parallel_size, dtype) can be set via CLI
    )

    return parser.parse_args()


# --- Data Loading ---
def load_hf_dataset_to_dataframe(
    dataset_name: str, split_name: str, max_docs: int = None
) -> pd.DataFrame:
    """Loads a Hugging Face dataset and converts the specified split to a Pandas DataFrame."""
    print(f"Loading dataset: {dataset_name}, split: {split_name}")
    # Consider adding try-except for dataset loading
    dataset = load_dataset(dataset_name, split=split_name)
    df = dataset.to_pandas()
    if max_docs and max_docs < len(df):
        print(f"Using a subset of {max_docs} documents.")
        df = df.head(max_docs)
    print(f"Loaded {len(df)} documents into DataFrame.")
    return df


# --- Text Preprocessing/Chunking ---
def preprocess_and_chunk_text(
    text: str, max_chunk_length: int = 256, overlap_ratio: float = 0.25
) -> List[str]:
    """
    Creates chunks from text with a specified overlap, disregarding sentence boundaries.

    Args:
        text: Input text to chunk.
        max_chunk_length: Maximum characters per chunk.
        overlap_ratio: The ratio of overlap between consecutive chunks.
                       For example, 0.1 means 10% overlap.
    """
    if not text or not isinstance(text, str) or max_chunk_length <= 0:
        return []

    text = text.strip()
    if not text:
        return []

    if overlap_ratio < 0 or overlap_ratio >= 1:
        raise ValueError(
            "overlap_ratio must be between 0 (inclusive) and 1 (exclusive)."
        )

    overlap_in_chars = math.floor(max_chunk_length * overlap_ratio)
    step_size = max(1, max_chunk_length - overlap_in_chars)

    chunks = []
    text_len = len(text)
    current_pos = 0
    while current_pos < text_len:
        chunk_end = min(current_pos + max_chunk_length, text_len)
        chunks.append(text[current_pos:chunk_end])
        if chunk_end == text_len:
            break
        current_pos += step_size

    if len(chunks) > 1 and len(chunks[-1]) < (0.3 * max_chunk_length):
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop(-1)

    return chunks


# --- Row-level Paragraph Preprocessing ---
def row_preprocess_text_by_separator(text: str, separator: str = "[PAR]") -> list:
    """
    Splits a text string into paragraphs using the given separator, cleans up markers, and removes empty entries.
    For each paragraph, if a [SEP] marker is present, only keep the text to the right of [SEP].
    Args:
        text: The input string containing paragraphs separated by [PAR].
        separator: The separator string (default: '[PAR]').
    Returns:
        List of cleaned paragraph strings.
    """
    if not text or not isinstance(text, str):
        return []
    paragraphs = text.split(separator)[1:]
    cleaned = []
    for para in paragraphs:
        # Remove [TLE] marker
        para = para.replace("[TLE]", "").strip()
        # If [SEP] is present, keep only the text to the right of [SEP]
        if "[SEP]" in para:
            para = para.split("[SEP]", 1)[1].strip()
        else:
            para = para.strip()
        if para:
            cleaned.append(para)
    return cleaned


# --- Main Processing Logic ---
def create_haystack_documents(
    df: pd.DataFrame,
    text_column: str,
    metadata_columns: List[str],
    preprocess_fn: callable,
    dataset_name: str = None,  # New parameter for unique ID generation
    split_name: str = None,  # New parameter for unique ID generation
) -> List[Document]:
    """
    Processes text from the DataFrame and creates Haystack Documents with unique IDs across splits.
    Deduplicates paragraphs by content hash.

    Args:
        df: DataFrame containing the data
        text_column: Column name containing text to embed
        metadata_columns: List of metadata column names to include
        preprocess_fn: Function to preprocess/chunk the text
        dataset_name: Name of the dataset (e.g., "lucadiliello/hotpotqa")
        split_name: Split name (e.g., "train", "validation")

    Returns:
        List of Haystack Document objects with unique chunk IDs
    """
    print("Creating Haystack Document objects...")
    haystack_documents: List[Document] = []

    # Derive dataset_identifier from dataset_name for unique ID generation
    if dataset_name:
        derived_dataset_identifier = (
            dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        )
    else:
        derived_dataset_identifier = "unknown_dataset"
        print(
            "Warning: dataset_name not provided, using 'unknown_dataset' as identifier"
        )

    if not split_name:
        split_name = "unknown_split"
        print("Warning: split_name not provided, using 'unknown_split' as identifier")

    # Ensure index is 0-based and sequential for ID consistency
    df = df.reset_index(drop=True)

    processed_paragraph_content_hashes = set()
    for index, row in df.iterrows():
        text_content = row[text_column]
        if not text_content or not isinstance(text_content, str):
            print(f"Skipping row {index} due to empty or invalid text content in column '{text_column}'.") # fmt: skip
            print(f"Row: {row}")
            continue

        original_doc_id = make_unique_doc_id(row, dataset_name, split_name, index)

        paragraphs = row_preprocess_text_by_separator(text_content)
        for para in paragraphs:
            paragraph_content_hash = hashlib.sha256(para.encode("utf-8")).hexdigest()
            if paragraph_content_hash in processed_paragraph_content_hashes:
                continue  # Deduplicate: only process first occurrence
            processed_paragraph_content_hashes.add(paragraph_content_hash)
            chunks = preprocess_fn(para)
            for i, chunk in enumerate(chunks):
                meta_data: Dict[str, Any] = {
                    "original_doc_id": original_doc_id,
                    "split_name": split_name,
                    "dataset_name": dataset_name,
                    "derived_dataset_identifier": derived_dataset_identifier,
                    "original_index_in_split": index,
                    "paragraph_content_hash": paragraph_content_hash,
                }

                for meta_col in metadata_columns:
                    if meta_col in row:
                        meta_data[meta_col] = row[meta_col]

                meta_data["chunk_id"] = f"{paragraph_content_hash}_chunk_{i}"

                doc = Document(content=chunk, meta=meta_data)
                haystack_documents.append(doc)

    print(f"Created {len(haystack_documents)} Haystack Document objects.")
    return haystack_documents


# --- Script Execution ---
if __name__ == "__main__":
    args = parse_args()

    print("Configuration:")
    print(f"  Dataset Name: {args.dataset_name}, Split: {args.split_name}")
    print(f"  Text Column: {args.text_column}")
    print(f"  Metadata Columns: {args.metadata_columns}")
    print(f"  Milvus DB Path: {args.milvus_db_path}, Drop Old: {args.drop_old_db}")
    print(f"  VLLM Model: {args.model}")
    print(f"  Max Documents: {args.max_docs if args.max_docs else 'All'}")
    print(f"  VLLM Task: {args.task}")
    print(f"  Preprocessing Function: {args.preprocess_function}")

    # 1. Load data
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

    # Check if all specified metadata_columns exist in the DataFrame
    for mc in args.metadata_columns:
        if mc not in dataframe.columns:
            print(
                f"Warning: Specified metadata column '{mc}' not found in DataFrame. It will be skipped."
            )

    # Resolve the preprocessing function
    preprocess_function_to_use = globals().get(args.preprocess_function)
    if not callable(preprocess_function_to_use):
        print(
            f"Warning: Preprocessing function '{args.preprocess_function}' not found or not callable. "
            f"Defaulting to 'preprocess_and_chunk_text'."
        )
        preprocess_function_to_use = preprocess_and_chunk_text

    # 2. Create Haystack Documents with enhanced metadata
    #    Pass existing args.dataset_name and args.split_name
    documents_to_embed = create_haystack_documents(
        df=dataframe,
        text_column=args.text_column,
        metadata_columns=args.metadata_columns,
        preprocess_fn=preprocess_function_to_use,
        dataset_name=args.dataset_name,  # Pass the original dataset name
        split_name=args.split_name,  # Pass the split name
    )

    if not documents_to_embed:
        print("No documents were created to embed. Exiting.")
        exit()

    # 3. Initialize VLLM and Embed Documents
    print("Initializing VLLM for embeddings...")
    try:
        all_parsed_args = vars(args)

        # Arguments specific to this script and not meant for VLLM EngineArgs.
        # 'embedding_dim' is defined by this script for Milvus, not for VLLM engine initialization.
        script_specific_arg_names = {
            "dataset_name",
            "split_name",
            "text_column",
            "milvus_db_path",
            "max_docs",
            "drop_old_db",
            "metadata_columns",
            "preprocess_function",
        }

        # Filter arguments to pass only VLLM-relevant ones.
        # The LLM constructor (e.g., LLM(model="...", **other_engine_args))
        # will correctly pick 'model' and other VLLM parameters from this dictionary.
        vllm_init_kwargs = {
            key: value
            for key, value in all_parsed_args.items()
            if key not in script_specific_arg_names
        }

        # LLM constructor will pick up 'model', 'task', and other engine args from vllm_init_kwargs
        vllm_model_instance = LLM(**vllm_init_kwargs)
        print(f"VLLM initialized with model: {args.model}, task: {args.task}.")
    except Exception as e:
        print(f"Error initializing VLLM: {e}")
        exit()

    texts_to_embed = [
        doc.content
        for doc in documents_to_embed
        if doc.content and isinstance(doc.content, str)
    ]
    if not texts_to_embed:
        print("No valid text content found in documents to embed. Exiting.")
        exit()

    # Map original documents to texts to handle cases where some docs might be filtered out
    # This assumes documents_to_embed maps directly to texts_to_embed if all are valid
    # A more robust way would be to filter documents_to_embed along with texts_to_embed
    valid_documents_to_embed = [
        doc
        for doc in documents_to_embed
        if doc.content and isinstance(doc.content, str)
    ]
    if len(texts_to_embed) != len(valid_documents_to_embed):
        print(
            "Mismatch between texts and valid documents, something went wrong."
        )  # Should not happen with current logic
        exit()

    print(f"Embedding {len(texts_to_embed)} document contents using VLLM...")
    documents_with_embeddings = []

    try:
        embedding_outputs = vllm_model_instance.embed(texts_to_embed)

        if len(embedding_outputs) != len(valid_documents_to_embed):
            print(
                f"Error: Number of embeddings ({len(embedding_outputs)}) "
                f"does not match number of documents ({len(valid_documents_to_embed)})."
            )
            exit()

        for i, doc_to_embed in enumerate(valid_documents_to_embed):
            embedding_data = embedding_outputs[i].outputs.embedding
            if embedding_data is None or not isinstance(embedding_data, list):
                print(
                    f"Error: Embedding for document index {i} "
                    f"(original_doc_id: {doc_to_embed.meta.get('original_doc_id')}, chunk_id: {doc_to_embed.meta.get('chunk_id')}) "
                    f"is invalid or None. Skipping this document."
                )
                # Optionally, one could choose to exit() here or collect problematic docs
                continue  # Skip this document

            doc_to_embed.embedding = (
                embedding_data  # Assign embedding to the Haystack Document
            )
            documents_with_embeddings.append(doc_to_embed)

        if not documents_with_embeddings:
            print(
                "Embedding result is empty after processing. No documents with embeddings. Exiting."
            )
            exit()

    except Exception as e:
        import traceback

        print(f"Error during VLLM embedding process: {e}")
        traceback.print_exc()
        exit()

    if not documents_with_embeddings:
        print("No documents were successfully embedded. Exiting.")
        exit()

    # 4. Initialize DocumentStore
    print("Initializing MilvusDocumentStore...")

    milvus_init_kwargs = {
        "connection_args": {"uri": args.milvus_db_path},
        "drop_old": args.drop_old_db,
    }

    document_store = MilvusDocumentStore(**milvus_init_kwargs)
    print("MilvusDocumentStore initialized.")

    # 5. Write Documents with Embeddings to Store
    print(
        f"Writing {len(documents_with_embeddings)} documents (with embeddings) to DocumentStore..."
    )
    document_store.write_documents(documents_with_embeddings)
    print(f"Successfully wrote documents to the store.")
    print(f"Total documents in store: {document_store.count_documents()}")

    print("Pipeline execution finished.")
