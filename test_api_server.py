import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_index_success(monkeypatch):
    # 模拟 manager.get_tenant_config 和 rag.index
    class DummyRag:
        def index(self, docs):
            assert docs == ["文档1", "文档2"]
    class DummyManager:
        def get_tenant_config(self, tenant_id):
            assert tenant_id == "tenant1"
            return DummyRag()
    monkeypatch.setattr("api_server.manager", DummyManager())
    
    response = client.post("/index", json={
        "tenant_id": "tenant1",
        "docs": ["文档1", "文档2"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["msg"] == "索引成功"

def test_index_fail(monkeypatch):
    class DummyRag:
        def index(self, docs):
            raise Exception("模拟索引失败")
    class DummyManager:
        def get_tenant_config(self, tenant_id):
            return DummyRag()
    monkeypatch.setattr("api_server.manager", DummyManager())
    response = client.post("/index", json={
        "tenant_id": "tenant1",
        "docs": ["文档1"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 1001
    assert "索引失败" in data["msg"]

def test_retrieve_success(monkeypatch):
    class DummyResult:
        def __init__(self, docs):
            self.docs = docs
    class DummyRag:
        def retrieve(self, querys):
            assert querys == ["问题1", "问题2"]
            return [DummyResult(["答案1"]), DummyResult(["答案2"])]
    class DummyManager:
        def get_tenant_config(self, tenant_id):
            assert tenant_id == "tenant1"
            return DummyRag()
    monkeypatch.setattr("api_server.manager", DummyManager())
    response = client.post("/retrieve", json={
        "tenant_id": "tenant1",
        "querys": ["问题1", "问题2"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["msg"] == "检索成功"
    assert data["data"]["docs"] == [["答案1"], ["答案2"]]

def test_retrieve_fail(monkeypatch):
    class DummyRag:
        def retrieve(self, querys):
            raise Exception("模拟检索失败")
    class DummyManager:
        def get_tenant_config(self, tenant_id):
            return DummyRag()
    monkeypatch.setattr("api_server.manager", DummyManager())
    response = client.post("/retrieve", json={
        "tenant_id": "tenant1",
        "querys": ["问题1"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 1002
    assert "检索失败" in data["msg"]
    assert data["data"]["docs"] == [] 