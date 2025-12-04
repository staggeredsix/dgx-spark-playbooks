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
from typing import List, Optional, Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_unstructured import UnstructuredLoader
from logger import logger

load_dotenv()


class VectorStore:
    """Vector store for document embedding and retrieval using Qdrant."""

    def __init__(
        self,
        embeddings=None,
        uri: str = "http://qdrant:6333",
        collection_name: str = "context",
        on_source_deleted: Optional[Callable[[str], None]] = None
    ):
        """Initialize the vector store.

        Args:
            embeddings: Embedding model to use (defaults to OllamaEmbeddings)
            uri: Qdrant connection URI
            collection_name: Name of the collection for stored chunks
            on_source_deleted: Optional callback when a source is deleted
        """
        try:
            self.uri = uri
            self.collection_name = collection_name
            self.embeddings = embeddings or OllamaEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b"),
                base_url=os.getenv("LLM_API_BASE_URL", "http://ollama:11434/v1"),
            )
            self.on_source_deleted = on_source_deleted
            self._initialize_store()

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            logger.debug({
                "message": "VectorStore initialized successfully",
                "uri": self.uri,
                "collection": self.collection_name
            })
        except Exception as e:
            logger.error({
                "message": "Error initializing VectorStore",
                "error": str(e)
            }, exc_info=True)
            raise

    def _initialize_store(self):
        self.client = QdrantClient(url=self.uri)
        if not self.client.collection_exists(self.collection_name):
            logger.info({
                "message": "Creating Qdrant collection",
                "collection": self.collection_name
            })
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=len(self.embeddings.embed_query("test")),
                    distance=qmodels.Distance.COSINE,
                )
            )
        self._store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        logger.debug({
            "message": "Qdrant vector store initialized",
            "uri": self.uri,
            "collection": self.collection_name
        })

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
        """No-op for Qdrant; kept for API compatibility."""
        logger.debug({
            "message": "Qdrant automatically persists inserts; no manual flush required"
        })

    def _build_source_filter(self, sources: List[str]) -> qmodels.Filter:
        return qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=source)
                )
                for source in sources
            ]
        )

    def get_documents(self, query: str, k: int = 8, sources: List[str] = None) -> List[Document]:
        """Get relevant documents using the retriever's invoke method."""
        try:
            search_kwargs = {"k": k}

            if sources:
                filter_expr = self._build_source_filter(sources)
                search_kwargs["filter"] = filter_expr
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
        """Delete a collection from Qdrant."""
        try:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)

                if self.on_source_deleted:
                    self.on_source_deleted(collection_name)

                logger.debug({
                    "message": "Collection deleted successfully",
                    "collection_name": collection_name
                })
                return True
            else:
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


def create_vector_store_with_config(config_manager, uri: str = "http://qdrant:6333") -> VectorStore:
    """Factory function to create a VectorStore with ConfigManager integration."""

    def handle_source_deleted(source_name: str):
        """Handle source deletion by updating config."""
        config = config_manager.read_config()
        if hasattr(config, 'sources') and source_name in config.sources:
            config.sources.remove(source_name)
            config_manager.write_config(config)

    return VectorStore(
        uri=uri,
        on_source_deleted=handle_source_deleted
    )
