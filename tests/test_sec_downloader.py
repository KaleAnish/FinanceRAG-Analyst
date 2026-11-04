from finance_rag.ingestion.sec_downloader import SECFilingDownloader


class FakeResponse:
    headers = {"Content-Type": "text/html"}

    def __init__(self, payload=b"SEC filing contents"):
        self.payload = payload
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.payload[:5]
        yield self.payload[5:]

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self):
        self.response = FakeResponse()
        self.url = None

    def get_archive_response(self, url, stream=True):
        self.url = url
        return self.response


class FakeStorage:
    def __init__(self):
        self.objects = {}

    def put_bytes(self, key, data, content_type=None):
        self.objects[key] = data

    def exists(self, key):
        return key in self.objects


def sample_row():
    return {
        "cik": "0000005272",
        "ticker": "AIG",
        "form_type": "8-K",
        "accession_number": "0000005272-26-000101",
        "filing_date": "2026-11-03",
        "primary_document": "aig-20261103.htm",
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/"
            "5272/000000527226000101/aig-20261103.htm"
        ),
    }


def test_download_and_checksum():
    client = FakeClient()
    storage = FakeStorage()
    downloader = SECFilingDownloader(client, storage)

    result = downloader.download_one(sample_row())

    assert result["status"] == "downloaded"
    assert result["size_bytes"] == len(b"SEC filing contents")
    assert len(result["sha256"]) == 64
    assert storage.objects[result["object_key"]] == b"SEC filing contents"
    assert client.response.closed


def test_deterministic_accession_key():
    downloader = SECFilingDownloader(FakeClient(), FakeStorage())

    assert downloader.object_key(sample_row()) == (
        "raw/sec/5272/000000527226000101/aig-20261103.htm"
    )


def test_rejects_non_sec_url():
    row = sample_row()
    row["source_url"] = "https://example.com/filing.htm"

    result = SECFilingDownloader(
        FakeClient(), FakeStorage()
    ).download_one(row)

    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"


def test_skip_only_when_previous_object_exists():
    storage = FakeStorage()
    downloader = SECFilingDownloader(FakeClient(), storage)
    row = sample_row()
    key = downloader.object_key(row)

    assert not downloader.should_skip(row, {
        "status": "downloaded",
        "object_key": key,
    })

    storage.objects[key] = b"existing filing"

    assert downloader.should_skip(row, {
        "status": "downloaded",
        "object_key": key,
    })
