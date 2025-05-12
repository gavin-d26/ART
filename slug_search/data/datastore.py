import argparse
import pandas as pd
from datasets import load_dataset
from haystack import Document
from milvus_haystack.document_store import MilvusDocumentStore
from typing import List, Dict, Any

# VLLM Imports
from vllm import LLM, EngineArgs


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
        "--embedding_dim",
        type=int,
        default=None,
        help="Dimension of the embeddings. Required by Milvus if creating a new collection and cannot infer. If not provided, it will be inferred from the first embedding.",
    )

    # Add VLLM engine arguments to the parser
    parser = EngineArgs.add_cli_args(parser)

    # Set VLLM specific defaults
    # The --model argument is added by EngineArgs.add_cli_args
    parser.set_defaults(
        model="BAAI/bge-large-en-v1.5",  # Default VLLM model for embeddings
        task="embed",  # Crucial for telling VLLM to do embedding
        enforce_eager=True,  # As per the example provided
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
def preprocess_and_chunk_text(text: str, **kwargs) -> List[str]:
    """
    Preprocesses text from a single document and optionally chunks it.
    Implement your custom chunking logic here.
    """
    # Placeholder: returns the text as a single chunk
    return [text]


# --- Main Processing Logic ---
def create_haystack_documents(
    df: pd.DataFrame,
    text_column: str,
    metadata_columns: List[str],
    preprocess_fn: callable,
) -> List[Document]:
    """
    Processes text from the DataFrame and creates Haystack Documents (without embeddings initially).
    """
    print("Creating Haystack Document objects...")
    haystack_documents: List[Document] = []
    for index, row in df.iterrows():
        text_content = row[text_column]
        if not text_content or not isinstance(text_content, str):
            print(
                f"Skipping row {index} due to empty or invalid text content in column '{text_column}'."
            )
            continue

        # Check if all metadata columns to embed exist in the row, if specified
        # This check is more relevant if meta_fields_to_embed is used extensively.
        # For now, we just ensure the main text_column is present.

        chunks = preprocess_fn(text_content)
        for i, chunk in enumerate(chunks):
            meta_data: Dict[str, Any] = {"original_doc_id": str(row.get("id", index))}
            for meta_col in metadata_columns:
                if meta_col in row:
                    meta_data[meta_col] = row[meta_col]
                # else:
                #     print(f"Warning: Metadata column '{meta_col}' not found in row {index}.")
            meta_data["chunk_id"] = f"{meta_data['original_doc_id']}_{i}"

            # Ensure all meta_fields_to_embed are present in meta_data if they are expected by the embedder
            # The embedder might raise an error if a field listed in meta_fields_to_embed is missing.
            # For simplicity, this example assumes they will be present if specified.

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
    print(
        f"  Embedding Dimension: {args.embedding_dim if args.embedding_dim else 'Will be inferred'}"
    )
    print(f"  Max Documents: {args.max_docs if args.max_docs else 'All'}")
    print(f"  VLLM Task: {args.task}")
    print(f"  VLLM Enforce Eager: {args.enforce_eager}")

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

    # 2. Create Haystack Documents (without embeddings yet)
    documents_to_embed = create_haystack_documents(
        df=dataframe,
        text_column=args.text_column,
        metadata_columns=args.metadata_columns,
        preprocess_fn=preprocess_and_chunk_text,
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
            "embedding_dim",
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
    actual_embedding_dim = None

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

        # Determine actual_embedding_dim from the first successful embedding
        if documents_with_embeddings[0].embedding:
            actual_embedding_dim = len(documents_with_embeddings[0].embedding)
            print(
                f"Embedding successful. Dimension of first embedding: {actual_embedding_dim}"
            )
        else:  # Should not happen if the list is not empty and embeddings are valid
            print(
                "Error: Could not determine embedding dimension. First document has no embedding."
            )
            exit()

        # If embedding_dim was not provided by user, use the actual one for Milvus
        # If it was provided, verify it matches actual_embedding_dim
        final_embedding_dim = actual_embedding_dim
        if args.embedding_dim:
            if args.embedding_dim != actual_embedding_dim:
                print(
                    f"Warning: User-provided --embedding_dim ({args.embedding_dim}) does not match "
                    f"actual embedding dimension ({actual_embedding_dim}). Using actual dimension: {actual_embedding_dim}."
                )
            else:
                final_embedding_dim = (
                    args.embedding_dim
                )  # Use user-provided if it matches

        if not final_embedding_dim:
            print(
                "Error: Embedding dimension could not be determined. This is required for Milvus."
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

    # final_embedding_dim is guaranteed to be set here if the script hasn't exited.
    # It incorporates args.embedding_dim logic already.
    # milvus_init_kwargs["embedding_dim"] = final_embedding_dim

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
