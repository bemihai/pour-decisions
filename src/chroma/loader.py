"""ChromaDB Collection Data Loader with Chunking and Embedding Support."""
from pathlib import Path
from tqdm import tqdm
import time

from .utils import get_or_create_collection, validate_chunks, create_batches
from .chunks import split_file
from .index_tracker import IndexTracker

from src.utils import logger, initialize_chroma_client, get_embedder


class CollectionDataLoader:
    """
    Data loader class for processing and loading documents into a ChromaDB collection.
    If the provided collection does not exist, a new one will be created.

    Args:
        collection_name: Name of the ChromaDB collection.
        collection_metadata: Optional metadata for the collection.
        chroma_host: Host for ChromaDB.
        chroma_port: Port for ChromaDB.
        embedding_model: HuggingFace model name for embeddings.
        batch_size: Number of documents to process in each batch.
    """
    def __init__(
        self,
        collection_name: str,
        collection_metadata: dict,
        chroma_host: str,
        chroma_port: int,
        embedding_model: str,
        batch_size: int = 2500,
    ):
        self.collection_name = collection_name
        self.collection_metadata = dict(collection_metadata)
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.embedder = get_embedder(model_name=embedding_model)

        # Initialize or get ChromaDB collection
        self.client = initialize_chroma_client(chroma_host, chroma_port)
        self.collection = get_or_create_collection(self.client, collection_name, collection_metadata)

    def reset_collection(self) -> None:
        """Delete and recreate the configured collection for a true forced reindex."""
        logger.warning("Resetting Chroma collection '%s' for forced reindex", self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.collection = get_or_create_collection(
            self.client,
            self.collection_name,
            dict(self.collection_metadata),
        )


    def _check_duplicate(self, content_hash: str) -> bool:
        """Check if content already exists in the collection."""
        try:
            results = self.collection.get(where={"content_hash": content_hash}, limit=1)
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return False


    def process_file(
        self,
        file_path: str | Path,
        strategy: str = "basic",
        chunk_size: int = 512,
        overlap_size: int = 128,
        skip_duplicates: bool = True,
        extract_metadata: bool = True,
    ) -> dict:
        """
        Process a single file and return a dict with stats.

        Args:
            file_path: Path to the file to process.
            strategy: Chunking strategy ("basic", "by_title", "semantic").
            chunk_size: Size of each chunk.
            overlap_size: Overlap size between chunks.
            skip_duplicates: Whether to skip duplicate chunks based on content hash.
            extract_metadata: Whether to extract wine-specific metadata from chunks.
        """
        file_path = Path(file_path)
        start_time = time.time()
        stats = {
            "filename": file_path.name,
            "chunks_generated": 0,
            "chunks_added": 0,
            "chunks_skipped": 0,
            "processing_time": 0,
            "errors": [],
        }

        try:
            logger.info(f"Processing file: {file_path.name}")

            chunks = split_file(
                filepath=file_path,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                embedding_model=self.embedding_model,
                extract_metadata=extract_metadata,
            )

            if not chunks:
                stats["errors"].append("No chunks generated")
                return stats

            stats["chunks_generated"] = len(chunks)

            valid_chunks = validate_chunks(chunks)

            docs = []
            metadata_list = []
            ids = []

            for chunk in valid_chunks:
                if skip_duplicates and self._check_duplicate(chunk["metadata"]["content_hash"]):
                    stats["chunks_skipped"] += 1
                    continue

                docs.append(chunk["text"])
                ids.append(chunk["id"])
                metadata_list.append(chunk["metadata"])

            if not docs:
                logger.info(f"No new chunks to add from {file_path.name}")
                stats["processing_time"] = time.time() - start_time
                return stats

            logger.info(f"Generating embeddings for {len(docs)} chunks...")
            embeddings = self.embedder.embed_documents(docs)

            batches = create_batches(
                batch_size=self.batch_size,
                documents=docs,
                embeddings=embeddings,
                metadata=metadata_list,
                ids=ids,
            )

            for i, batch in enumerate(batches):
                try:
                    self.collection.add(
                        ids=batch[0],
                        embeddings=batch[1],
                        metadatas=batch[2],
                        documents=batch[3],
                    )
                    logger.debug(f"Added batch {i+1}/{len(batches)} for {file_path.name}")
                except Exception as e:
                    error_msg = f"Error adding batch {i+1}: {e}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)

            stats["chunks_added"] = len(docs)
            logger.info(f"Successfully added {len(docs)} chunks from {file_path.name}")

        except Exception as e:
            error_msg = f"Error processing file {file_path.name}: {e}"
            stats["errors"].append(error_msg)
            logger.error(error_msg)

        stats["processing_time"] = time.time() - start_time
        return stats


    def load_directory(
        self,
        data_path: str | Path,
        file_extensions: list[str] = None,
        strategy: str = "basic",
        chunk_size: int = 512,
        overlap_size: int = 128,
        skip_duplicates: bool = True,
        extract_metadata: bool = True,
        incremental: bool = True,
        force_reindex: bool = False,
    ) -> dict:
        """
        Load all files from directory with progress tracking. Returns a summary dict.

        Args:
            data_path: Path to the directory containing files.
            file_extensions: List of file extensions to process, ex: [".epub", ".pdf"].
            strategy: Chunking strategy ("basic", "by_title", "semantic").
            chunk_size: Size of each chunk.
            overlap_size: Overlap size between chunks.
            skip_duplicates: Whether to skip duplicate chunks based on content hash.
            extract_metadata: Whether to extract wine-specific metadata from chunks.
            incremental: If True, only process new or modified files (default: True).
            force_reindex: If True, ignore index tracking and reprocess all files.
        """
        if file_extensions is None:
            file_extensions = [".epub", ".pdf"]

        data_dir = Path(data_path)
        if not data_dir.exists():
            raise ValueError(f"Data directory {data_path} does not exist")

        all_files = []
        for ext in file_extensions:
            all_files.extend(data_dir.glob(f"**/*{ext}"))

        if not all_files:
            logger.warning(f"No files found with extensions {file_extensions} in {data_path}")
            return {"total_files": 0, "files_processed": 0, "total_chunks": 0}

        # Initialize index tracker for incremental processing
        tracker = None
        files_to_process = all_files
        skipped_files = 0

        if incremental and not force_reindex:
            tracker = IndexTracker(collection_name=self.collection_name)
            files_to_process = tracker.get_files_to_index(all_files)
            skipped_files = len(all_files) - len(files_to_process)

            if not files_to_process:
                logger.info("All files already indexed, nothing to process")
                stats = tracker.get_stats()
                return {
                    "total_files": len(all_files),
                    "files_processed": 0,
                    "files_skipped": skipped_files,
                    "total_chunks": stats.get("total_chunks", 0),
                    "message": "All files already indexed",
                }

        logger.info(f"Found {len(all_files)} files, processing {len(files_to_process)} (skipping {skipped_files} already indexed)")

        # Process files with progress bar
        total_stats = {
            "total_files": len(all_files),
            "files_processed": 0,
            "files_skipped": skipped_files,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks_generated": 0,
            "total_chunks_added": 0,
            "total_chunks_skipped": 0,
            "total_processing_time": 0,
            "file_results": [],
            "errors": [],
        }

        with tqdm(files_to_process, desc="Processing files") as pbar:
            for file_path in pbar:
                pbar.set_description(f"Processing {file_path.name}")

                file_stats = self.process_file(
                    file_path=file_path,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    skip_duplicates=skip_duplicates,
                    extract_metadata=extract_metadata,
                )

                total_stats["files_processed"] += 1
                total_stats["total_chunks_generated"] += file_stats["chunks_generated"]
                total_stats["total_chunks_added"] += file_stats["chunks_added"]
                total_stats["total_chunks_skipped"] += file_stats["chunks_skipped"]
                total_stats["total_processing_time"] += file_stats["processing_time"]
                total_stats["file_results"].append(file_stats)

                if file_stats["errors"]:
                    total_stats["failed_files"] += 1
                    total_stats["errors"].extend(file_stats["errors"])
                    logger.warning(f"File '{file_path.name}' failed and will be retried on next run")
                else:
                    total_stats["successful_files"] += 1
                    if tracker is not None:
                        tracker.mark_indexed(file_path, file_stats["chunks_added"])
                        tracker.save()

                pbar.set_postfix(
                    {
                        "chunks": total_stats["total_chunks_added"],
                        "errors": len(total_stats["errors"]),
                    }
                )

        if tracker is not None:
            tracker.save()
            logger.info(f"Updated index manifest: {tracker.get_stats()}")

        failed_msg = ""
        if total_stats['failed_files'] > 0:
            failed_msg = f" (will retry on next run)"

        logger.info(
            f"""
            Processing Complete:
            - Files processed: {total_stats['files_processed']}/{total_stats['total_files']}
            - Files skipped (already indexed): {total_stats['files_skipped']}
            - Successful: {total_stats['successful_files']}
            - Failed: {total_stats['failed_files']}{failed_msg}
            - Total chunks added: {total_stats['total_chunks_added']}
            - Total chunks skipped: {total_stats['total_chunks_skipped']}
            - Total processing time: {total_stats['total_processing_time']:.2f}s
            """
        )

        return total_stats


