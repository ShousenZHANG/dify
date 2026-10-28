import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import override

import dify_vdb_lindorm.lindorm_vector as lindorm_module
import pytest
from dify_vdb_lindorm.lindorm_vector import (
    ROUTING_FIELD,
    LindormVectorStore,
    LindormVectorStoreConfig,
    LindormVectorStoreFactory,
)
from opensearchpy import NotFoundError
from opensearchpy.helpers import BulkIndexError
from pydantic import ValidationError

from core.rag.datasource.vdb.field import Field
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from models.dataset import Dataset


@dataclass(frozen=True)
class SearchRequest:
    """The arguments of one ``search`` call."""

    index: str
    body: dict[str, object]
    params: dict[str, object] | None = None


class FakeIndices:
    """The index API of the fake client: tracks which indices exist and records each call."""

    def __init__(self) -> None:
        self.existing: set[str] = set()
        self.created: list[tuple[str, dict[str, object]]] = []
        self.deleted: list[tuple[str, dict[str, object]]] = []
        self.refreshed: list[str] = []
        self.exists_calls = 0

    def exists(self, *, index: str) -> bool:
        self.exists_calls += 1
        return index in self.existing

    def create(self, *, index: str, body: dict[str, object]) -> None:
        self.created.append((index, body))
        self.existing.add(index)

    def delete(self, *, index: str, params: dict[str, object]) -> None:
        self.deleted.append((index, params))

    def refresh(self, *, index: str) -> None:
        self.refreshed.append(index)


class FakeOpenSearch:
    """Stands in for the OpenSearch client: records each request and answers from in-memory state."""

    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.stored_ids: set[str] = set()
        self.bulk_bodies: list[list[dict[str, object]]] = []
        self.bulk_response: dict[str, object] = {"errors": False, "items": []}
        self.bulk_error: Exception | None = None
        self.searches: list[SearchRequest] = []
        self.search_hits: list[dict[str, object]] = []
        self.search_error: Exception | None = None
        self.deleted_by_query: list[tuple[str, dict[str, object]]] = []

    def bulk(self, *, body: list[dict[str, object]], timeout: int) -> dict[str, object]:
        self.bulk_bodies.append(body)
        if self.bulk_error:
            raise self.bulk_error
        return self.bulk_response

    def search(
        self, *, index: str, body: dict[str, object], params: dict[str, object] | None = None
    ) -> dict[str, dict[str, list[dict[str, object]]]]:
        self.searches.append(SearchRequest(index, body, params))
        if self.search_error:
            raise self.search_error
        return {"hits": {"hits": self.search_hits}}

    def exists(self, *, index: str, id: str, params: dict[str, object]) -> bool:
        return id in self.stored_ids

    def get(self, *, index: str, id: str, params: dict[str, object]) -> dict[str, str]:
        if id not in self.stored_ids:
            raise NotFoundError(404, "document_missing_exception", {"_id": id, "found": False})
        return {"_id": id}

    def delete_by_query(self, *, index: str, body: dict[str, object]) -> None:
        self.deleted_by_query.append((index, body))


class FakeRedis:
    """The parts of ``redis_client`` that collection creation uses, returning one cached value for every key."""

    def __init__(self, cached: object = None) -> None:
        self.cached = cached
        self.set_calls: list[tuple[str, object, int | None]] = []

    @contextmanager
    def lock(self, name: str, timeout: int) -> Generator[None]:
        yield

    def get(self, key: str) -> object:
        return self.cached

    def set(self, key: str, value: object, ex: int | None = None) -> None:
        self.set_calls.append((key, value, ex))


class QueryOnlyEmbeddings(Embeddings):
    """Embeddings whose query vector the factory measures to size a new index."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @override
    def embed_multimodal_documents(self, multimodel_documents: list[dict[str, object]]) -> list[list[float]]:
        raise NotImplementedError

    @override
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    @override
    def embed_multimodal_query(self, multimodel_document: dict[str, object]) -> list[float]:
        raise NotImplementedError


def _config() -> LindormVectorStoreConfig:
    return LindormVectorStoreConfig(
        hosts="http://localhost:9200",
        username="user",
        password="pass",
        using_ugc=False,
        request_timeout=3.0,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeOpenSearch:
    client = FakeOpenSearch()
    monkeypatch.setattr(lindorm_module, "OpenSearch", lambda **_params: client)
    return client


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    redis = FakeRedis()
    monkeypatch.setattr(lindorm_module, "redis_client", redis)
    return redis


@pytest.fixture
def bulk_deletes(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, object]]]:
    recorded: list[list[dict[str, object]]] = []

    def bulk(_client: FakeOpenSearch, actions: list[dict[str, object]]) -> None:
        recorded.append(actions)

    monkeypatch.setattr(lindorm_module.helpers, "bulk", bulk)
    return recorded


def _ugc_store(client: FakeOpenSearch) -> LindormVectorStore:
    store = LindormVectorStore("collection", _config(), using_ugc=True, routing_value="route")
    assert store._client is client
    return store


def _plain_store(client: FakeOpenSearch) -> LindormVectorStore:
    store = LindormVectorStore("collection", _config(), using_ugc=False)
    assert store._client is client
    return store


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hosts", None, "config URL is required"),
        ("username", None, "config USERNAME is required"),
        ("password", None, "config PASSWORD is required"),
    ],
)
def test_lindorm_config_validation(field: str, value: None, message: str) -> None:
    values = _config().model_dump()
    values[field] = value

    with pytest.raises(ValidationError, match=message):
        LindormVectorStoreConfig.model_validate(values)


def test_to_opensearch_params_and_init(client: FakeOpenSearch) -> None:
    cfg = _config()
    params = cfg.to_opensearch_params()

    assert params["hosts"] == "http://localhost:9200"
    assert params["http_auth"] == ("user", "pass")

    vector = LindormVectorStore("Collection", cfg, using_ugc=False)
    assert vector._collection_name == "collection"
    assert vector.get_type() == VectorType.LINDORM

    with pytest.raises(ValueError, match="routing_value"):
        LindormVectorStore("c", cfg, using_ugc=True)

    vector_ugc = LindormVectorStore("c", cfg, using_ugc=True, routing_value="ROUTE")
    assert vector_ugc._routing == "route"


def test_create_refresh_and_add_texts_success(
    client: FakeOpenSearch, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(lindorm_module.time, "sleep", sleeps.append)
    vector = _ugc_store(client)

    docs = [Document(page_content="a", metadata={"doc_id": "id-1"})]
    vector.create(docs, [[0.1]])
    (created_index, created_body) = client.indices.created[0]
    assert created_index == "collection"
    assert created_body["settings"] == {"index": {"knn": True, "knn_routing": True}}
    assert client.bulk_bodies == [
        [
            {"index": {"_index": "collection", "_id": "id-1", "routing": "route"}},
            {
                Field.CONTENT_KEY: "a",
                Field.VECTOR: [0.1],
                Field.METADATA_KEY: {"doc_id": "id-1"},
                ROUTING_FIELD: "route",
            },
        ]
    ]

    client.bulk_bodies.clear()
    docs = [
        Document(page_content="a", metadata={"doc_id": "id-1"}),
        Document(page_content="b", metadata={"doc_id": "id-2"}),
        Document(page_content="c", metadata={"doc_id": "id-3"}),
    ]
    embeddings = [[0.1], [0.2], [0.3]]

    vector.add_texts(docs, embeddings, batch_size=2, timeout=9)

    assert [len(body) for body in client.bulk_bodies] == [4, 2]
    header, values = client.bulk_bodies[0][:2]
    assert header == {"index": {"_index": "collection", "_id": "id-1", "routing": "route"}}
    assert values == {
        Field.CONTENT_KEY: "a",
        Field.VECTOR: [0.1],
        Field.METADATA_KEY: {"doc_id": "id-1"},
        ROUTING_FIELD: "route",
    }
    # Only the gap between the two batches is slept.
    assert sleeps == [0.5]
    vector.refresh()
    assert client.indices.refreshed == ["collection"]


def test_add_texts_error_paths(client: FakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    # Keep the production retry/stop/backoff rules, but do not wait in real time.
    monkeypatch.setattr(lindorm_module, "retry", partial(lindorm_module.retry, sleep=sleeps.append))
    vector = _plain_store(client)
    client.bulk_response = {"errors": True, "items": [{"index": {"error": "boom"}}]}

    docs = [Document(page_content="a", metadata={"doc_id": "id-1"})]
    with pytest.raises(Exception, match="RetryError"):
        vector.add_texts(docs, [[0.1]], batch_size=1)

    assert len(client.bulk_bodies) == 3
    assert sleeps == [4, 4]
    client.bulk_bodies.clear()
    sleeps.clear()

    client.bulk_error = RuntimeError("bulk failed")
    with pytest.raises(Exception, match="RetryError"):
        vector.add_texts(docs, [[0.1]], batch_size=1)

    assert len(client.bulk_bodies) == 3
    assert sleeps == [4, 4]


def test_metadata_lookup_and_delete_by_metadata(
    client: FakeOpenSearch, bulk_deletes: list[list[dict[str, object]]]
) -> None:
    vector = _ugc_store(client)
    client.indices.existing.add("collection")
    client.stored_ids.update({"id-1", "id-2"})
    client.search_hits = [{"_id": "id-1"}, {"_id": "id-2"}]

    ids = vector.get_ids_by_metadata_field("document_id", "doc-1")
    assert ids == ["id-1", "id-2"]
    assert client.searches[-1] == SearchRequest(
        "collection",
        {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"metadata.document_id.keyword": "doc-1"}},
                        {"term": {"routing_field.keyword": "route"}},
                    ]
                }
            }
        },
    )

    vector.delete_by_metadata_field("document_id", "doc-1")
    assert [[action["_id"] for action in actions] for actions in bulk_deletes] == [["id-1", "id-2"]]

    client.search_hits = []
    vector.delete_by_metadata_field("document_id", "doc-2")
    assert len(bulk_deletes) == 1


def test_delete_by_ids_paths(
    client: FakeOpenSearch, bulk_deletes: list[list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    vector = _ugc_store(client)

    vector.delete_by_ids([])
    assert client.indices.exists_calls == 0

    vector.delete_by_ids(["id-1"])
    assert client.indices.exists_calls == 1
    assert bulk_deletes == []

    client.indices.existing.add("collection")
    client.stored_ids.add("id-1")
    vector.delete_by_ids(["id-1", "id-2"])
    assert bulk_deletes == [[{"_op_type": "delete", "_index": "collection", "_id": "id-1", "routing": "route"}]]

    def failing_bulk(_client: FakeOpenSearch, actions: list[dict[str, object]]) -> None:
        raise BulkIndexError(
            "2 document(s) failed to index.",
            [
                {"delete": {"status": 404, "_id": "id-404"}},
                {"delete": {"status": 500, "_id": "id-500"}},
            ],
        )

    monkeypatch.setattr(lindorm_module.helpers, "bulk", failing_bulk)
    # Bulk delete failures are logged rather than raised.
    vector.delete_by_ids(["id-1"])


def test_delete_and_text_exists(client: FakeOpenSearch) -> None:
    vector = _ugc_store(client)
    vector.delete()
    assert client.deleted_by_query == [
        ("collection", {"query": {"bool": {"must": [{"term": {"routing_field.keyword": "route"}}]}}})
    ]
    assert client.indices.refreshed == ["collection"]

    vector = _plain_store(client)
    client.indices.existing.add("collection")
    vector.delete()
    assert client.indices.deleted == [("collection", {"timeout": 60})]

    client.indices.existing.clear()
    vector.delete()
    assert len(client.indices.deleted) == 1

    client.stored_ids.add("id-1")
    assert vector.text_exists("id-1") is True
    assert vector.text_exists("missing") is False


def test_search_by_vector_validation_and_success(client: FakeOpenSearch) -> None:
    vector = _ugc_store(client)

    # Vectors decoded from JSON are not type checked, so the store validates them at runtime.
    with pytest.raises(ValueError, match="should be a list"):
        vector.search_by_vector(json.loads('"bad"'))

    with pytest.raises(ValueError, match="should be floats"):
        vector.search_by_vector(json.loads('[0.1, "bad"]'))

    client.search_hits = [
        {
            "_score": 0.9,
            "_source": {
                Field.CONTENT_KEY: "doc-a",
                Field.VECTOR: [0.1],
                Field.METADATA_KEY: {"doc_id": "1", "document_id": "d-1"},
            },
        },
        {
            "_score": 0.2,
            "_source": {
                Field.CONTENT_KEY: "doc-b",
                Field.VECTOR: [0.2],
                Field.METADATA_KEY: {"doc_id": "2", "document_id": "d-2"},
            },
        },
    ]
    docs = vector.search_by_vector([0.1, 0.2], top_k=2, score_threshold=0.5, document_ids_filter=["d-1"])
    assert len(docs) == 1
    assert docs[0].metadata["score"] == pytest.approx(0.9)

    assert client.searches[-1] == SearchRequest(
        "collection",
        {
            "size": 2,
            "_source": True,
            "query": {
                "knn": {
                    Field.VECTOR: {
                        "vector": [0.1, 0.2],
                        "k": 2,
                        "filter": {
                            "bool": {
                                "must": [
                                    {"terms": {"metadata.document_id.keyword": ["d-1"]}},
                                    {"term": {"routing_field.keyword": "route"}},
                                ]
                            }
                        },
                    }
                }
            },
            "ext": {"lvector": {"filter_type": "pre_filter"}},
        },
        {"timeout": 3.0, "routing": "route"},
    )

    client.search_error = RuntimeError("search failed")
    with pytest.raises(RuntimeError, match="search failed"):
        vector.search_by_vector([0.1])


def test_search_by_full_text_success_and_error(client: FakeOpenSearch) -> None:
    vector = _ugc_store(client)
    client.search_hits = [
        {
            "_source": {
                Field.CONTENT_KEY: "doc-a",
                Field.VECTOR: [0.1],
                Field.METADATA_KEY: {"doc_id": "1"},
            }
        }
    ]

    docs = vector.search_by_full_text("hello", top_k=2, document_ids_filter=["d-1"])
    assert len(docs) == 1
    assert docs[0].page_content == "doc-a"

    assert client.searches[-1] == SearchRequest(
        "collection",
        {
            "query": {
                "bool": {
                    "must": [{"match": {Field.CONTENT_KEY.value: "hello"}}],
                    "filter": [
                        {"terms": {"metadata.document_id.keyword": ["d-1"]}},
                        {"term": {"routing_field.keyword": "route"}},
                    ],
                }
            }
        },
        {"timeout": 3.0, "routing": "route"},
    )

    client.search_error = RuntimeError("full text failed")
    with pytest.raises(RuntimeError, match="full text failed"):
        vector.search_by_full_text("hello")


def test_create_collection_paths(client: FakeOpenSearch, fake_redis: FakeRedis) -> None:
    vector = _plain_store(client)

    with pytest.raises(ValueError, match="cannot be empty"):
        vector.create_collection([])

    fake_redis.cached = 1
    vector.create_collection([[0.1, 0.2]])
    assert client.indices.created == []

    fake_redis.cached = None
    vector.create_collection([[0.1, 0.2]], index_params={"index_type": "ivf", "space_type": "cosine"})
    (index, body) = client.indices.created[0]
    assert index == "collection"
    assert body["mappings"] == {
        "properties": {
            Field.CONTENT_KEY: {"type": "text"},
            Field.VECTOR: {
                "type": "knn_vector",
                "dimension": 2,
                "method": {"name": "ivf", "space_type": "cosine", "engine": "lvector"},
            },
        }
    }
    assert fake_redis.set_calls == [("vector_indexing_collection", 1, 3600)]

    # The index now exists, so only the cache marker is written again.
    vector.create_collection([[0.1, 0.2]])
    assert len(client.indices.created) == 1


def test_lindorm_factory_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = LindormVectorStoreFactory()

    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_URL", "http://localhost:9200")
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_USERNAME", "user")
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_PASSWORD", "pass")
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_QUERY_TIMEOUT", 3.0)
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_INDEX_TYPE", "hnsw")
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_DISTANCE_TYPE", "l2")
    monkeypatch.setattr(lindorm_module.Dataset, "gen_collection_name_by_id", lambda _id: "AUTO_COLLECTION")

    dataset = Dataset(id="dataset-1")
    embeddings = QueryOnlyEmbeddings()
    built: list[tuple[str, bool, str | None]] = []

    def build_store(
        index_name: str, config: LindormVectorStoreConfig, *, routing_value: str | None, using_ugc: bool
    ) -> str:
        built.append((index_name, using_ugc, routing_value))
        return "vector"

    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_USING_UGC", None)
    with pytest.raises(ValueError, match="LINDORM_USING_UGC is not set"):
        factory.init_vector(dataset, attributes=[], embeddings=embeddings, session=None)

    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_USING_UGC", False)
    monkeypatch.setattr(lindorm_module, "LindormVectorStore", build_store)

    dataset_existing_plain = Dataset(
        id="dataset-1", index_struct=json.dumps({"vector_store": {"class_prefix": "EXISTING"}, "using_ugc": False})
    )
    result = factory.init_vector(dataset_existing_plain, attributes=[], embeddings=embeddings, session=None)
    assert result == "vector"
    assert built[-1] == ("existing", False, None)

    dataset_existing_ugc = Dataset(
        id="dataset-1",
        index_struct=json.dumps(
            {
                "vector_store": {"class_prefix": "ROUTING"},
                "using_ugc": True,
                "dimension": 1536,
                "index_type": "hnsw",
                "distance_type": "l2",
            }
        ),
    )
    factory.init_vector(dataset_existing_ugc, attributes=[], embeddings=embeddings, session=None)
    assert built[-1] == ("ugc_index_1536_hnsw_l2", True, "ROUTING")

    dataset_new = Dataset(id="dataset-2")

    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_USING_UGC", True)
    factory.init_vector(dataset_new, attributes=[], embeddings=embeddings, session=None)
    assert built[-1] == ("ugc_index_3_hnsw_l2", True, "auto_collection")
    assert dataset_new.index_struct is not None

    dataset_new_plain = Dataset(id="dataset-3")
    monkeypatch.setattr(lindorm_module.dify_config, "LINDORM_USING_UGC", False)
    factory.init_vector(dataset_new_plain, attributes=[], embeddings=embeddings, session=None)
    assert built[-1] == ("auto_collection", False, None)
