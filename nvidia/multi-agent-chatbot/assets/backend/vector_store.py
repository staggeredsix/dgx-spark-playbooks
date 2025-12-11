#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import glob
import os
import time
from typing import Callable, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_unstructured import UnstructuredLoader
from logger import logger
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

load_dotenv()


class EmbeddingServiceUnavailable(Exception):
    """Raised when the embedding backend cannot be reached after retries."""


class VectorStore:
    """Vector store for document embedding and retrieval using Neo4j."""

    def __init__(
        self,
        embeddings=None,
        uri: str = "neo4j://neo4j:7687",
        username: str = "neo4j",
        password: str = "chatbot_neo4j",
        database: str = "neo4j",
        index_name: str = "context",
        node_label: str = "DocumentChunk",
        text_node_property: str = "text",
        embedding_node_property: str = "embedding",
        on_source_deleted: Optional[Callable[[str], None]] = None
    ):
        """Initialize the vector store.

        Args:
            embeddings: Embedding model to use (defaults to OllamaEmbeddings)
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            index_name: Name of the vector index
            node_label: Label used for stored chunks
            text_node_property: Node property containing the chunk text
            embedding_node_property: Node property containing embedding vectors
            on_source_deleted: Optional callback when a source is deleted
        """
        try:
            self.uri = uri
            self.username = username
            self.password = password
            self.database = database
            self.index_name = index_name
            self.node_label = node_label
            self.text_node_property = text_node_property
            self.embedding_node_property = embedding_node_property
            self.embedding_model = self._get_embedding_model()
            self.embedding_base_url = self._get_embedding_base_url()
            self.embeddings = embeddings or OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.embedding_base_url,
            )
            self.on_source_deleted = on_source_deleted
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self._initialize_store()

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            logger.debug({
                "message": "VectorStore initialized successfully",
                "uri": self.uri,
                "index_name": self.index_name
            })
        except Exception as e:
            logger.error({
                "message": "Error initializing VectorStore",
                "error": str(e)
            }, exc_info=True)
            raise

    def _initialize_store(self):
        max_attempts = int(os.getenv("NEO4J_INIT_RETRIES", "10"))
        backoff = float(os.getenv("NEO4J_INIT_BACKOFF", "3"))
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                self._ensure_vector_index()
                self._store = Neo4jVector.from_existing_index(
                    embedding=self.embeddings,
                    url=self.uri,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    index_name=self.index_name,
                    text_node_property=self.text_node_property,
                    embedding_node_property=self.embedding_node_property,
                    node_label=self.node_label,
                )
                logger.debug({
                    "message": "Neo4j vector store initialized",
                    "uri": self.uri,
                    "index_name": self.index_name,
                    "attempt": attempt
                })
                return
            except ServiceUnavailable as exc:
                last_error = exc
                logger.warning({
                    "message": "Neo4j not ready, will retry",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "backoff_seconds": backoff
                })
            except EmbeddingServiceUnavailable as exc:
                last_error = exc
                logger.warning({
                    "message": "Embedding service not ready, will retry",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "backoff_seconds": backoff,
                    "error": str(exc)
                })
            except Exception as exc:  # pragma: no cover - unexpected initialization errors
                logger.error({
                    "message": "Unexpected error initializing Neo4j vector store",
                    "error": str(exc)
                }, exc_info=True)
                raise

            time.sleep(backoff)

        logger.error({
            "message": "Failed to initialize Neo4j vector store after retries",
            "attempts": max_attempts,
            "error": str(last_error) if last_error else None
        })
        raise last_error

    def _ensure_vector_index(self):
        dimensions = self._get_embedding_dimensions()
        index_query = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (n:{self.node_label})
        ON (n.{self.embedding_node_property})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: $dimensions,
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        with self.driver.session(database=self.database) as session:
            session.run(index_query, dimensions=dimensions)
            logger.info({
                "message": "Ensured Neo4j vector index exists",
                "index_name": self.index_name,
                "node_label": self.node_label,
                "dimensions": dimensions
            })

    def _get_embedding_dimensions(self) -> int:
        max_attempts = int(os.getenv("EMBEDDING_INIT_RETRIES", "15"))
        backoff = float(os.getenv("EMBEDDING_INIT_BACKOFF", "2.5"))
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                dimensions = len(self.embeddings.embed_query("test"))
                logger.debug({
                    "message": "Embedding model responded successfully",
                    "dimensions": dimensions,
                    "attempt": attempt
                })
                return dimensions
            except Exception as exc:
                last_error = exc
                logger.warning({
                    "message": "Embedding service not ready, will retry",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "backoff_seconds": backoff,
                    "error": str(exc)
                })
                time.sleep(backoff)

        logger.error({
            "message": "Failed to contact embedding service after retries",
            "attempts": max_attempts,
            "error": str(last_error) if last_error else None,
            "base_url": self.embedding_base_url,
            "model": self.embedding_model,
        })
        raise EmbeddingServiceUnavailable(
            last_error
            or f"Embedding service unavailable at {self.embedding_base_url}. "
            f"Verify the service is running and model '{self.embedding_model}' is available."
        ) from last_error

    def _load_documents(self, file_paths: List[str] = None, input_dir: str = None) -> List[str]:
        try:
            documents = []
            source_name = None

            if input_dir:
                source_name = os.path.basename(os.path.normpath(input_dir))
                logger.debug({
                    "message": "Loading files from directory",
                    "directory": input_dir,
                    "source": source_name
                })
                file_paths = glob.glob(os.path.join(input_dir, "**"), recursive=True)
                file_paths = [f for f in file_paths if os.path.isfile(f)]

            logger.info(f"Processing {len(file_paths)} files: {file_paths}")

            for file_path in file_paths:
                try:
                    if not source_name:
                        source_name = os.path.basename(file_path)
                        logger.info(f"Using filename as source: {source_name}")

                    logger.info(f"Loading file: {file_path}")

                    file_ext = os.path.splitext(file_path)[1].lower()
                    logger.info(f"File extension: {file_ext}")

                    try:
                        loader = UnstructuredLoader(file_path)
                        docs = loader.load()
                        logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
                    except Exception as pdf_error:
                        logger.error(f'error with unstructured loader, trying to load from scratch')
                        file_text = None
                        if file_ext == ".pdf":
                            logger.info("Attempting PyPDF text extraction fallback")
                            try:
                                from pypdf import PdfReader
                                reader = PdfReader(file_path)
                                extracted_pages = []
                                for page in reader.pages:
                                    try:
                                        extracted_pages.append(page.extract_text() or "")
                                    except Exception as per_page_err:
                                        logger.info(f"Warning: failed to extract a page: {per_page_err}")
                                        extracted_pages.append("")
                                file_text = "\n\n".join(extracted_pages).strip()
                            except Exception as pypdf_error:
                                logger.info(f"PyPDF fallback failed: {pypdf_error}")
                                file_text = None

                        if not file_text:
                            logger.info("Falling back to raw text read of file contents")
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    file_text = f.read()
                            except Exception as read_error:
                                logger.info(f"Fallback read failed: {read_error}")
                                file_text = ""

                        if file_text and file_text.strip():
                            docs = [Document(
                                page_content=file_text,
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]
                        else:
                            logger.info("Creating a simple document as fallback (no text extracted)")
                            docs = [Document(
                                page_content=f"Document: {os.path.basename(file_path)}",
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]

                    for doc in docs:
                        if not doc.metadata:
                            doc.metadata = {}

                        cleaned_metadata = {}
                        cleaned_metadata["source"] = source_name
                        cleaned_metadata["file_path"] = file_path
                        cleaned_metadata["filename"] = os.path.basename(file_path)

                        for key, value in doc.metadata.items():
                            if key not in ["source", "file_path"]:
                                if isinstance(value, (list, dict, set)):
                                    cleaned_metadata[key] = str(value)
                                elif value is not None:
                                    cleaned_metadata[key] = str(value)

                        doc.metadata = cleaned_metadata
                    documents.extend(docs)
                    logger.debug({
                        "message": "Loaded documents from file",
                        "file_path": file_path,
                        "document_count": len(docs)
                    })
                except Exception as e:
                    logger.error({
                        "message": "Error loading file",
                        "file_path": file_path,
                        "error": str(e)
                    }, exc_info=True)
                    continue

            logger.info(f"Total documents loaded: {len(documents)}")
            return documents

        except Exception as e:
            logger.error({
                "message": "Error loading documents",
                "error": str(e)
            }, exc_info=True)
            raise

    def _get_embedding_base_url(self) -> str:
        base_url = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        sanitized_url = base_url.removesuffix("/v1").removesuffix("/api")
        return sanitized_url

    def _get_embedding_model(self) -> str:
        configured_model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not configured_model:
            return "qwen3-embedding:8b"
        return configured_model

    def index_documents(self, documents: List[Document]) -> List[Document]:
        try:
            logger.debug({
                "message": "Starting document indexing",
                "document_count": len(documents)
            })

            splits = self.text_splitter.split_documents(documents)
            logger.debug({
                "message": "Split documents into chunks",
                "chunk_count": len(splits)
            })

            self._store.add_documents(splits)

            logger.debug({
                "message": "Document indexing completed"
            })
        except Exception as e:
            logger.error({
                "message": "Error during document indexing",
                "error": str(e)
            }, exc_info=True)
            raise

    def flush_store(self):
        """Neo4j writes are persisted automatically."""
        logger.debug({
            "message": "Neo4j automatically persists inserts; no manual flush required"
        })

    def get_documents(self, query: str, k: int = 8, sources: List[str] = None) -> List[Document]:
        """Get relevant documents using the retriever's invoke method."""
        try:
            search_kwargs = {"k": k}

            if sources:
                search_kwargs["filter"] = {"source": {"$in": sources}}
                logger.debug({
                    "message": "Retrieving with filter",
                    "filter": sources
                })

            retriever = self._store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )

            docs = retriever.invoke(query)
            logger.debug({
                "message": "Retrieved documents",
                "query": query,
                "document_count": len(docs)
            })

            return docs
        except Exception as e:
            logger.error({
                "message": "Error retrieving documents",
                "error": str(e)
            }, exc_info=True)
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete all chunks associated with a source from Neo4j."""
        try:
            delete_query = f"""
            MATCH (n:{self.node_label})
            WHERE n.source = $collection_name
            WITH collect(n) AS nodes
            UNWIND nodes AS node
            DETACH DELETE node
            RETURN count(node) AS deleted_count
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(delete_query, collection_name=collection_name).single()
                deleted_count = result["deleted_count"] if result else 0

            if deleted_count > 0:
                if self.on_source_deleted:
                    self.on_source_deleted(collection_name)

                logger.debug({
                    "message": "Collection deleted successfully",
                    "collection_name": collection_name,
                    "deleted_count": deleted_count
                })
                return True

            logger.warning({
                "message": "Collection not found",
                "collection_name": collection_name
            })
            return False
        except Exception as e:
            logger.error({
                "message": "Error deleting collection",
                "collection_name": collection_name,
                "error": str(e)
            }, exc_info=True)
            return False


def create_vector_store_with_config(
    config_manager,
    uri: str = "neo4j://neo4j:7687",
    username: str = "neo4j",
    password: str = "chatbot_neo4j",
    database: str = "neo4j"
) -> VectorStore:
    """Factory function to create a VectorStore with ConfigManager integration."""

    def handle_source_deleted(source_name: str):
        """Handle source deletion by updating config."""
        config = config_manager.read_config()
        if hasattr(config, 'sources') and source_name in config.sources:
            config.sources.remove(source_name)
            config_manager.write_config(config)

    return VectorStore(
        uri=uri,
        username=username,
        password=password,
        database=database,
        on_source_deleted=handle_source_deleted
    )
