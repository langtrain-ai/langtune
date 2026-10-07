"""
The remote clients must only call routes the Langtrain API server has, with
X-API-Key and the server's field names. See fake_langtrain_api.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fake_langtrain_api import API_KEY, FakeAPI  # noqa: E402

from langtune.fast_model import FastLanguageModel, LangtrainServerClient, RemoteJob  # noqa: E402
from langtune.client import LangtuneClient  # noqa: E402


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()
    assert server.unknown == [], f"called routes the server doesn't have: {server.unknown}"


def test_remote_job_lifecycle(api):
    client = LangtrainServerClient(API_KEY, api.url)
    job = client.create_job({"base_model": "m", "dataset_id": "ds", "training_method": "qlora"})
    assert job["id"] == "job-1"
    remote = RemoteJob(job["id"], client)
    assert remote.status()["status"] == "completed"
    steps = list(remote.stream(interval_s=0))
    assert steps[0].step == 10 and steps[0].loss == 0.5
    assert remote.cancel()
    assert remote.export("you/model")["export_id"] == "exp-1"
    assert all(call[4] == API_KEY for call in api.calls)


def test_upload_with_api_key_explains_what_to_do(api, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    with pytest.raises(PermissionError, match="dashboard"):
        LangtrainServerClient(API_KEY, api.url).upload_dataset(str(data))


def test_bad_key_is_reported(api):
    with pytest.raises(PermissionError):
        LangtrainServerClient("sk-lt-wrong", api.url).get_job("job-1")


def test_langtune_client(api):
    client = LangtuneClient(api_key=API_KEY, base_url=api.url + "/api/v1")
    assert client.validate()["valid"]
    job = client.get_finetune_job("job-1")
    assert job.id == "job-1" and job.model == "meta-llama/Llama-3.1-8B-Instruct"


def test_env_key_alone_does_not_switch_to_remote(monkeypatch):
    monkeypatch.delenv("LANGTRAIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="remote=True"):
        FastLanguageModel.from_pretrained("m", remote=True)
