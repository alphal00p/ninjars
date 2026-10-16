"""Lossless deterministic storage for large validation JSON references."""
import gzip
import hashlib
import io
import json
from pathlib import Path


def read_bytes(path):
    path=Path(path)
    data=path.read_bytes()
    return gzip.decompress(data) if path.suffix=='.gz' else data


def read_json(path):
    return json.loads(read_bytes(path))


def archive_json(source,destination):
    raw=read_bytes(source)
    json.loads(raw)  # refuse to archive invalid input
    output=io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as stream:
        stream.write(raw)
    compressed=output.getvalue()
    assert gzip.decompress(compressed)==raw
    Path(destination).write_bytes(compressed)
    return {'compression':'gzip; mtime=0; no filename metadata',
            'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
            'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
            'uncompressed_bytes':len(raw),'compressed_bytes':len(compressed)}
