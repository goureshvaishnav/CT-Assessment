"""
rag_engine.py — Core RAG logic for the Chatbot.
"""

import hashlib
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    ollama_base_url: str
    persist_dir: str
    collection_name: str
    embed_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_docs: int
    max_history: int
    min_relevance_score: float

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        return cls(
            ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            persist_dir=os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
            collection_name=os.getenv('CHROMA_COLLECTION', 'rag_docs'),
            embed_model=os.getenv('EMBED_MODEL', 'nomic-embed-text'),
            llm_model=os.getenv('LLM_MODEL', 'mistral'),
            chunk_size=int(os.getenv('CHUNK_SIZE', '800')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '150')),
            top_k_docs=int(os.getenv('TOP_K_DOCS', '5')),
            max_history=int(os.getenv('MAX_HISTORY', '8')),
            min_relevance_score=float(os.getenv('MIN_RELEVANCE_SCORE', '0.0')),
        )


CONFIG = RAGConfig.from_env()
PERSIST_DIR = CONFIG.persist_dir
EMBED_MODEL = CONFIG.embed_model
LLM_MODEL = CONFIG.llm_model
CHUNK_SIZE = CONFIG.chunk_size
CHUNK_OVERLAP = CONFIG.chunk_overlap
TOP_K_DOCS = CONFIG.top_k_docs
MAX_HISTORY = CONFIG.max_history

SYSTEM_PROMPT = (
    'You are a precise, knowledgeable AI research assistant. '
    'Answer questions strictly using the provided document context. '
    'Structure answers clearly with bullets or numbered steps where useful. '
    'If context is insufficient, say exactly: '
    "'I don't have enough information in the uploaded documents to answer that.' "
    'Never hallucinate facts outside the provided context.\n\n'
    'Context:\n{context}'
)


class RAGEngine:
    def __init__(self):
        self.config = CONFIG
        self.embeddings = OllamaEmbeddings(
            model=self.config.embed_model,
            base_url=self.config.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=self.config.llm_model,
            temperature=0.2,
            keep_alive='10m',
            base_url=self.config.ollama_base_url,
        )
        self.persist_dir = self._resolve_writable_persist_dir(self.config.persist_dir)
        self.vectorstore = self._create_vectorstore(self.persist_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=['\n\n', '\n', '.', ' ', ''],
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ])
        logger.info('RAGEngine initialised.')
        logger.info('Using Chroma persist directory: %s', self.persist_dir)

    def _resolve_writable_persist_dir(self, preferred_dir: str) -> str:
        os.makedirs(preferred_dir, exist_ok=True)
        if self._can_write_to_dir(preferred_dir):
            return preferred_dir

        fallback_dir = os.path.join(
            tempfile.gettempdir(),
            'rag_chroma_db',
        )
        os.makedirs(fallback_dir, exist_ok=True)
        logger.warning(
            "Persist directory '%s' is not writable. Falling back to '%s'.",
            preferred_dir,
            fallback_dir,
        )
        return fallback_dir

    def _can_write_to_dir(self, target_dir: str) -> bool:
        try:
            probe_path = os.path.join(target_dir, '.write_test')
            with open(probe_path, 'w', encoding='utf-8') as handle:
                handle.write('ok')
            os.remove(probe_path)
            return True
        except Exception:
            return False

    def _create_vectorstore(self, persist_dir: str) -> Chroma:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )

    def _recover_from_readonly_db(self):
        recovered_dir = os.path.join(
            tempfile.gettempdir(),
            f'rag_chroma_db_recovered_{uuid.uuid4().hex[:8]}',
        )
        os.makedirs(recovered_dir, exist_ok=True)
        self.persist_dir = recovered_dir
        self.vectorstore = self._create_vectorstore(self.persist_dir)
        logger.warning('Recovered from readonly DB. Using: %s', self.persist_dir)

    def _tags_url(self) -> str:
        return f'{self.config.ollama_base_url}/api/tags'

    def is_ollama_available(self) -> bool:
        try:
            res = requests.get(self._tags_url(), timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False

    def list_ollama_models(self) -> List[str]:
        try:
            res = requests.get(self._tags_url(), timeout=2.0)
            res.raise_for_status()
            data = res.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    def is_embed_model_available(self) -> bool:
        models = self.list_ollama_models()
        return any(self.config.embed_model in model for model in models)

    def is_llm_model_available(self) -> bool:
        models = self.list_ollama_models()
        return any(self.config.llm_model in model for model in models)

    def get_health_report(self) -> Dict[str, Any]:
        ollama_ok = self.is_ollama_available()
        embed_ok = self.is_embed_model_available() if ollama_ok else False
        llm_ok = self.is_llm_model_available() if ollama_ok else False
        return {
            'ollama': ollama_ok,
            'embed_model': embed_ok,
            'llm_model': llm_ok,
            'doc_chunks': self.get_doc_count(),
            'config': {
                'ollama_base_url': self.config.ollama_base_url,
                'persist_dir': self.persist_dir,
                'configured_persist_dir': self.config.persist_dir,
                'embed_model': self.config.embed_model,
                'llm_model': self.config.llm_model,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'top_k_docs': self.config.top_k_docs,
                'max_history': self.config.max_history,
                'min_relevance_score': self.config.min_relevance_score,
            },
        }

    def _fingerprint_docs(self, docs: List[Any]) -> str:
        joined = '\n'.join(doc.page_content for doc in docs)
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()

    def _is_already_indexed(self, file_id: str) -> bool:
        try:
            existing = self.vectorstore._collection.get(where={'file_id': file_id}, limit=1)
            ids = existing.get('ids', [])
            return bool(ids)
        except Exception:
            return False

    def load_pdfs(self, file_paths: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'files_received': len(file_paths),
            'files_indexed': 0,
            'files_skipped': 0,
            'chunks': 0,
            'errors': [],
            'indexed_files': [],
            'skipped_files': [],
        }

        for path in file_paths:
            file_name = os.path.basename(path)
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                if not docs:
                    result['errors'].append({'file': file_name, 'error': 'No text extracted'})
                    continue

                file_id = self._fingerprint_docs(docs)
                if self._is_already_indexed(file_id):
                    result['files_skipped'] += 1
                    result['skipped_files'].append(file_name)
                    continue

                indexed_at = datetime.now(timezone.utc).isoformat()
                chunks = self.splitter.split_documents(docs)
                ids: List[str] = []
                for idx, chunk in enumerate(chunks):
                    chunk.metadata['file_id'] = file_id
                    chunk.metadata['file_name'] = file_name
                    chunk.metadata['chunk_index'] = idx
                    chunk.metadata['indexed_at'] = indexed_at
                    ids.append(f'{file_id}:{idx}')

                try:
                    self.vectorstore.add_documents(chunks, ids=ids)
                except Exception as err:
                    err_text = str(err).lower()
                    if 'readonly' in err_text or 'attempt to write a readonly database' in err_text:
                        self._recover_from_readonly_db()
                        try:
                            self.vectorstore.add_documents(chunks, ids=ids)
                        except Exception as retry_err:
                            raise RuntimeError(
                                'Chroma write failed even after readonly recovery '
                                f"(active_dir='{self.persist_dir}')"
                            ) from retry_err
                    else:
                        raise
                result['files_indexed'] += 1
                result['chunks'] += len(chunks)
                result['indexed_files'].append(file_name)
                logger.info("Indexed '%s' -> %s chunks", file_name, len(chunks))
            except Exception as err:
                logger.exception('Failed to index file: %s', file_name)
                result['errors'].append({'file': file_name, 'error': str(err)})

        return result

    def get_doc_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def reset_vectorstore(self):
        import shutil

        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.persist_dir = self._resolve_writable_persist_dir(self.config.persist_dir)
        self.vectorstore = self._create_vectorstore(self.persist_dir)
        logger.info('Vector store reset.')

    def _format_docs(self, docs: List[Any]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get('file_name', os.path.basename(doc.metadata.get('source', 'unknown')))
            page = doc.metadata.get('page', '?')
            parts.append(f'[{i}] (source: {src}, page {page})\n{doc.page_content}')
        return '\n\n---\n\n'.join(parts)

    def _convert_history(self, history: List[Dict[str, str]], max_turns: int) -> List[Any]:
        msgs = []
        for item in history[-max_turns:]:
            role = item.get('role', '')
            content = str(item.get('content', ''))
            if role == 'user':
                msgs.append(HumanMessage(content=content))
            elif role == 'assistant':
                msgs.append(AIMessage(content=content))
        return msgs

    def _retrieve_with_scores(self, question: str) -> List[Dict[str, Any]]:
        raw = self.vectorstore.similarity_search_with_relevance_scores(
            question,
            k=self.config.top_k_docs,
        )
        items: List[Dict[str, Any]] = []
        for doc, score in raw:
            if score < self.config.min_relevance_score:
                continue
            items.append({'doc': doc, 'score': float(score)})
        return items

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_history: int = MAX_HISTORY,
    ) -> str:
        return self.query_with_sources(question, chat_history, max_history)['answer']

    def query_with_sources(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_history: int = MAX_HISTORY,
    ) -> Dict[str, Any]:
        if not self.is_ollama_available():
            return {
                'answer': (
                    "⚠️ **Ollama is not running.**\n\n"
                    "Start it in a terminal:\n```\nollama serve\n```\n"
                    'Then refresh this page.'
                ),
                'sources': [],
                'retrieved_count': 0,
            }
        if not self.is_llm_model_available():
            return {
                'answer': (
                    f"⚠️ **Missing LLM model `{self.config.llm_model}`.**\n\n"
                    f"Run:\n```\nollama pull {self.config.llm_model}\n```"
                ),
                'sources': [],
                'retrieved_count': 0,
            }
        if not self.is_embed_model_available():
            return {
                'answer': (
                    f"⚠️ **Missing embedding model `{self.config.embed_model}`.**\n\n"
                    f"Run:\n```\nollama pull {self.config.embed_model}\n```"
                ),
                'sources': [],
                'retrieved_count': 0,
            }
        if self.get_doc_count() == 0:
            return {
                'answer': (
                    "📂 **No documents indexed yet.**\n\n"
                    'Upload one or more PDFs using the sidebar, then ask your question.'
                ),
                'sources': [],
                'retrieved_count': 0,
            }

        try:
            retrieved = self._retrieve_with_scores(question)
            docs = [item['doc'] for item in retrieved]
            if not docs:
                return {
                    'answer': (
                        "I don't have enough information in the uploaded documents "
                        'to answer that.'
                    ),
                    'sources': [],
                    'retrieved_count': 0,
                }

            chain = self.prompt | self.llm | StrOutputParser()
            history = self._convert_history(chat_history or [], max_turns=max_history)
            response = chain.invoke({
                'context': self._format_docs(docs),
                'question': question,
                'chat_history': history,
            })
            sources = []
            for item in retrieved:
                doc = item['doc']
                sources.append({
                    'file': doc.metadata.get(
                        'file_name',
                        os.path.basename(doc.metadata.get('source', 'unknown')),
                    ),
                    'page': doc.metadata.get('page', '?'),
                    'chunk_index': doc.metadata.get('chunk_index', '?'),
                    'score': round(item['score'], 4),
                    'snippet': doc.page_content[:240].strip(),
                })
            return {
                'answer': response.strip(),
                'sources': sources,
                'retrieved_count': len(sources),
            }
        except Exception as err:
            logger.exception('Query failed')
            return {
                'answer': (
                    '⚠️ Something failed while generating the response.\n\n'
                    f'Technical detail: `{err}`'
                ),
                'sources': [],
                'retrieved_count': 0,
            }
