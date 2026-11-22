"""Lazy model imports keep artifact preparation independent of PyTorch."""
import json
from dataclasses import dataclass
from pathlib import Path

from .storage import Corpus, digest


@dataclass(frozen=True)
class RetrievedPassage:
    doc_id: str
    title: str
    text: str
    score: float


class QueryEncoder:
    def __init__(self, model='facebook/contriever-msmarco', revision=None, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device).eval()

    def encode(self, queries):
        import torch
        if any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('Questions must be nonempty strings')
        with torch.inference_mode():
            tokens = self.tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1)
            return pooled.float().cpu().numpy()


class ContrieverRetriever:
    def __init__(self, artifact, device=None, encoder=None):
        from utils.src.index import Indexer
        artifact = Path(artifact)
        self.fingerprint = digest(artifact / 'manifest.json')
        manifest = json.loads((artifact / 'manifest.json').read_text())
        if not manifest.get('complete') or manifest.get('dimension') != 768:
            raise ValueError('Incomplete or incompatible artifact')
        self.index = Indexer(768)
        self.index.deserialize_from(str(artifact))
        if self.index.index.ntotal != manifest['vectors']:
            raise ValueError('Artifact vector count mismatch')
        if self.index.index.d != manifest['dimension']:
            raise ValueError('Artifact dimension mismatch')
        self.corpus = Corpus(artifact / 'corpus.sqlite')
        self.encoder = encoder or QueryEncoder(manifest['model'], manifest['model_revision'], device)

    def retrieve(self, query, top_k=20):
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(self, queries, top_k=20):
        if top_k <= 0 or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('Invalid query or top_k')
        if not queries:
            return []
        results = self.index.search_knn(self.encoder.encode(queries), top_k)
        return [[RetrievedPassage(*row, float(score)) for row, score in zip(self.corpus.get_many(ids), scores)]
                for ids, scores in results]

    def close(self):
        self.corpus.close()
