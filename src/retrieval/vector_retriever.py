"""Retriever component for querying ChromaDB collections."""
from typing import List, Dict, Any
from collections import OrderedDict
import hashlib

import chromadb as cdb

from .query_utils import normalize_query, expand_query
from src.utils import get_embedder, logger


class ChromaRetriever:
    """
    Retriever for querying ChromaDB collections and retrieving relevant documents.

    Args:
        client: ChromaDB client instance.
        collection_name: Name of the collection to query.
        embedding_model: HuggingFace model name for embeddings.
        n_results: Number of results to retrieve (default: 5).
        similarity_threshold: Minimum similarity score to filter results (default: None).
        enable_query_expansion: Whether to expand queries with related wine terminology (default: True).
        enable_cache: Whether to cache query results (default: True).
        cache_size: Maximum number of queries to cache (default: 100).
    """

    def __init__(
        self,
        client: cdb.ClientAPI,
        collection_name: str,
        embedding_model: str,
        n_results: int = 5,
        similarity_threshold: float | None = None,
        enable_query_expansion: bool = True,
        enable_cache: bool = True,
        cache_size: int = 100,
    ):
        self.client = client
        self.collection_name = collection_name
        self.n_results = n_results
        self.similarity_threshold = similarity_threshold
        self.enable_query_expansion = enable_query_expansion
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        try:
            # Use shared embedder cache to avoid repeated heavyweight model init.
            self.embedder = get_embedder(model_name=embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model '{embedding_model}': {e}") from e

        try:
            self.collection = client.get_collection(collection_name)
            logger.info(f"Retrieved collection '{collection_name}' for querying")
        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}': {e}")
            raise

    def _get_cache_key(
        self,
        query: str,
        n_results: int,
        where: Dict[str, Any] | None,
        where_document: Dict[str, Any] | None
    ) -> str:
        """Generate cache key for query parameters."""
        key_parts = f"{query}:{n_results}:{str(where)}:{str(where_document)}"
        return hashlib.md5(key_parts.encode()).hexdigest()

    def _cache_get(self, key: str) -> List[Dict[str, Any]] | None:
        """Get from cache with LRU update."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_set(self, key: str, value: List[Dict[str, Any]]) -> None:
        """Set cache with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query with wine terminology normalization and optional expansion.

        Args:
            query: Raw user query

        Returns:
            Preprocessed query string
        """
        # Normalize wine terminology (fix misspellings, canonicalize synonyms)
        processed = normalize_query(query)

        # Optionally expand with related terminology
        if self.enable_query_expansion:
            processed = expand_query(processed)

        if processed != query.lower():
            logger.debug(f"Query preprocessed: '{query}' -> '{processed}'")

        return processed

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        where: Dict[str, Any] | None = None,
        where_document: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query with optional filtering.

        Args:
            query: The user's query string.
            n_results: Number of results (uses instance default if None).
            where: Optional metadata filter (e.g., {"source": "wine_atlas.pdf"}).
            where_document: Optional document content filter.

        Returns:
            List of dicts with keys: 'id', 'document', 'metadata', 'distance', 'similarity'.
        """
        n_results = n_results or self.n_results

        # Check cache first
        cache_key = None
        if self.enable_cache:
            cache_key = self._get_cache_key(query, n_results, where, where_document)
            cached_results = self._cache_get(cache_key)
            if cached_results is not None:
                logger.debug(f"Cache hit for query: '{query[:50]}...'")
                return cached_results

        try:
            # Preprocess query with wine terminology normalization
            processed_query = self._preprocess_query(query)
            query_embedding = self.embedder.embed_query(processed_query)

            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            if where is not None:
                query_params["where"] = where
            if where_document is not None:
                query_params["where_document"] = where_document

            results = self.collection.query(**query_params)
            retrieved_docs = self._format_results(results)

            # Update cache
            if self.enable_cache and cache_key and retrieved_docs:
                self._cache_set(cache_key, retrieved_docs)

            if retrieved_docs:
                # Log retrieval statistics
                avg_similarity = sum(d['similarity'] for d in retrieved_docs if d['similarity']) / len(retrieved_docs)
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'")
                logger.debug(f"Avg similarity: {avg_similarity:.3f}, Range: [{retrieved_docs[-1]['similarity']:.3f}, {retrieved_docs[0]['similarity']:.3f}]")
            else:
                logger.warning(f"No results found for query: '{query[:50]}...'")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB query results into standardized document dicts."""
        retrieved_docs = []

        if not results or not results['ids'] or not results['ids'][0]:
            return retrieved_docs

        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i] if results['distances'] else None
            similarity = 1 - distance if distance is not None else None

            # Apply similarity threshold filter
            if self.similarity_threshold is not None and similarity is not None:
                if similarity < self.similarity_threshold:
                    continue

            retrieved_docs.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': distance,
                'similarity': similarity,
            })

        return retrieved_docs

    def retrieve_with_filter(
        self,
        query: str,
        where: Dict[str, Any] | None = None,
        where_document: Dict[str, Any] | None = None,
        n_results: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with metadata filtering.

        Deprecated: Use retrieve() with where/where_document parameters instead.
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

