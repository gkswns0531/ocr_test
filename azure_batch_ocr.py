"""Batch Azure Layout OCR for remaining KB documents.

Calls Azure Document Intelligence API sequentially (1 at a time)
to avoid 429 rate limiting. Stores results as ocr.zip in MinIO.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from minio import Minio

# ── Azure DI config ─────────────────────────────────────────────────
AZURE_ENDPOINT = "https://allganize-document-inteligence-for-dev.cognitiveservices.azure.com/"
AZURE_API_KEY = "040069b99ceb415a87343be1ef58cdbc"

# ── MinIO config ────────────────────────────────────────────────────
MINIO_ENDPOINT = "localhost:29000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "passw0rD!"
MINIO_BUCKET = "alli-files"

PROJECT_ID = "678f65c06e31e6ec0341ba81"

# ── MongoDB config (via docker compose exec) ────────────────────────
import subprocess

COMPOSE_DIR = "/Users/hanjuncho/code_base/CCW/mally"


def _mongo_query(script: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mongodb", "mongosh", "mally_dev", "--quiet", "--eval", script],
        capture_output=True, text=True, cwd=COMPOSE_DIR, timeout=30,
    )
    return result.stdout.strip()


def get_incomplete_kbs(folder_id: str) -> list[dict]:
    """Get KB IDs that are still in 'parsing' state."""
    out = _mongo_query(f'''
    var kbs = db.knowledge_base.find({{
        target_folder_id: "{folder_id}",
        process_state: "parsing"
    }});
    var results = [];
    kbs.forEach(function(doc) {{
        results.push({{kb_id: doc._id.toString(), file_name: doc.file_name, saved_file_name: doc.saved_file_name}});
    }});
    print(JSON.stringify(results));
    ''')
    return json.loads(out)


def mark_kb_completed(kb_id: str) -> None:
    """Mark KB as completed in MongoDB."""
    _mongo_query(f'''
    db.knowledge_base.updateOne(
        {{_id: ObjectId("{kb_id}")}},
        {{$set: {{process_state: "completed", modified_at: new Date()}}}}
    );
    ''')


def get_completed_count(folder_id: str) -> int:
    out = _mongo_query(f'''
    print(db.knowledge_base.countDocuments({{target_folder_id: "{folder_id}", process_state: "completed"}}));
    ''')
    return int(out)


def run_batch(folder_id: str, folder_name: str, total_expected: int) -> None:
    """Process all incomplete KBs in a folder."""
    # Init clients
    az_client = DocumentIntelligenceClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_API_KEY))
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

    incomplete = get_incomplete_kbs(folder_id)
    done_before = get_completed_count(folder_id)
    print(f"\n[{folder_name}] {done_before}/{total_expected} already done, {len(incomplete)} remaining")

    for i, kb in enumerate(incomplete):
        kb_id = kb["kb_id"]
        file_name = kb["file_name"]
        saved = kb["saved_file_name"]

        # Extract MinIO path: minio://PROJECT_ID/KB_ID/timestamp_filename.jpg
        # → PROJECT_ID/KB_ID/timestamp_filename.jpg
        minio_path = saved.replace("minio://", "")

        try:
            # 1. Download image from MinIO
            t0 = time.time()
            response = minio_client.get_object(MINIO_BUCKET, minio_path)
            image_bytes = response.read()
            response.close()
            response.release_conn()

            # 2. Call Azure Layout API
            t1 = time.time()
            import base64
            b64_data = base64.b64encode(image_bytes).decode()
            poller = az_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=b64_data),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
            result = poller.result()
            t2 = time.time()
            api_latency = t2 - t1

            # 3. Create ocr.zip with ocr-1.json and meta.json
            result_dict = result.as_dict()
            ocr_json = json.dumps(result_dict, ensure_ascii=False)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("ocr-1.json", ocr_json)
                zf.writestr("meta.json", json.dumps({"pages": 1, "latency_ms": api_latency * 1000}))
            zip_buffer.seek(0)

            # 4. Upload ocr.zip to MinIO
            # Need a "binary_file_id" — use kb_id as prefix
            ocr_key = f"{PROJECT_ID}/{kb_id}/direct_ocr/ocr.zip"
            minio_client.put_object(
                MINIO_BUCKET, ocr_key, zip_buffer, len(zip_buffer.getvalue()),
                content_type="application/zip",
            )

            # 5. Mark as completed
            mark_kb_completed(kb_id)

            elapsed = time.time() - t0
            current_done = done_before + i + 1
            print(f"  [{current_done}/{total_expected}] {file_name}: "
                  f"API={api_latency:.1f}s, total={elapsed:.1f}s, "
                  f"content={len(result_dict.get('content', ''))} chars")

            # Rate limit safety: small delay between requests
            time.sleep(1)

        except Exception as e:
            print(f"  [{done_before + i + 1}/{total_expected}] {file_name}: ERROR - {e}")
            # On 429, wait longer
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("    Rate limited! Waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(2)

    final_done = get_completed_count(folder_id)
    print(f"\n[{folder_name}] Final: {final_done}/{total_expected} completed")


if __name__ == "__main__":
    OMNI_FOLDER_ID = "69ab6d2e401ab959ed091c35"
    DP_FOLDER_ID = "69ab6d9ab7aebc7907c53222"

    print("=" * 60)
    print("Azure Layout Batch OCR — Sequential (1 at a time)")
    print("=" * 60)

    run_batch(OMNI_FOLDER_ID, "OmniDocBench", 1355)
    run_batch(DP_FOLDER_ID, "DP-Bench", 200)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
