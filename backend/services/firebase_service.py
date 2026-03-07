"""
Firebase service — Firestore for prices, scenarios, training history, checkpoints.
Firebase Storage for model binary files (.zip).

Falls back to a local JSON file + disk store when Firebase credentials are not configured.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional
from loguru import logger
from pathlib import Path

from backend.config import settings

# Try Firebase
_fb_available = False
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    _fb_available = True
except ImportError:
    logger.warning("firebase-admin not installed — using local file store")


# ── Local fallback store ────────────────────────────────────────────────────
LOCAL_STORE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"
LOCAL_STORE_DIR.mkdir(exist_ok=True)


def _local_path(collection: str) -> Path:
    return LOCAL_STORE_DIR / f"{collection}.json"


def _load_local(collection: str) -> dict:
    p = _local_path(collection)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_local(collection: str, data: dict) -> None:
    p = _local_path(collection)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Firebase wrapper ────────────────────────────────────────────────────────

class FirebaseService:
    """Thin wrapper that works with Firestore + Storage or falls back to local JSON + disk."""

    def __init__(self):
        self.db = None
        self.bucket = None
        self.use_firebase = False

        if _fb_available and settings.FIREBASE_CREDENTIALS_PATH:
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                    firebase_admin.initialize_app(cred, {
                        "projectId": settings.FIREBASE_PROJECT_ID,
                        "storageBucket": f"{settings.FIREBASE_PROJECT_ID}.firebasestorage.app",
                    })
                self.db = firestore.client()
                try:
                    self.bucket = storage.bucket()
                    logger.info("Firebase Storage bucket connected")
                except Exception as exc:
                    logger.warning(f"Firebase Storage unavailable (models stay on disk): {exc}")
                    self.bucket = None
                self.use_firebase = True
                logger.info("Firebase Firestore connected")
            except Exception as exc:
                logger.warning(f"Firebase init failed, using local store: {exc}")
        else:
            logger.info("Using local JSON file store (no Firebase credentials)")

    # ─────────────────────────────────────────────────────────────────────
    # Prices
    # ─────────────────────────────────────────────────────────────────────

    async def save_prices(self, scenario_name: str, prices: dict) -> str:
        """Save product prices for a scenario."""
        doc_id = f"prices_{scenario_name}"
        data = {
            "scenario_name": scenario_name,
            "prices": prices,
            "updated_at": datetime.utcnow().isoformat(),
        }

        if self.use_firebase:
            self.db.collection("prices").document(doc_id).set(data)
        else:
            store = _load_local("prices")
            store[doc_id] = data
            _save_local("prices", store)

        logger.info(f"Prices saved for scenario '{scenario_name}'")
        return doc_id

    async def get_prices(self, scenario_name: str = "default") -> Optional[dict]:
        """Get prices for a scenario."""
        doc_id = f"prices_{scenario_name}"

        if self.use_firebase:
            doc = self.db.collection("prices").document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        else:
            store = _load_local("prices")
            return store.get(doc_id)

    async def list_scenarios(self) -> list[dict]:
        """List all saved price scenarios."""
        if self.use_firebase:
            docs = self.db.collection("prices").stream()
            return [doc.to_dict() for doc in docs]
        else:
            store = _load_local("prices")
            return list(store.values())

    # ─────────────────────────────────────────────────────────────────────
    # Training runs & metrics
    # ─────────────────────────────────────────────────────────────────────

    async def save_training_run(self, run_data: dict) -> str:
        """Save a training run record with full metrics."""
        run_id = run_data.get("run_id", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_data["run_id"] = run_id
        run_data["created_at"] = datetime.utcnow().isoformat()

        if self.use_firebase:
            self.db.collection("training_runs").document(run_id).set(run_data)
        else:
            store = _load_local("training_runs")
            store[run_id] = run_data
            _save_local("training_runs", store)

        logger.info(f"Training run saved: {run_id}")
        return run_id

    async def save_training_metrics(
        self, run_id: str, final_metrics: dict, metrics_history: list[dict]
    ) -> str:
        """Save detailed training metrics for a run."""
        doc_id = f"metrics_{run_id}"
        data = {
            "run_id": run_id,
            "final_metrics": final_metrics,
            "metrics_count": len(metrics_history),
            "created_at": datetime.utcnow().isoformat(),
        }
        # Firestore has a 1MB document limit, so we store a summary
        # plus the last 200 history points
        data["metrics_history"] = metrics_history[-200:]

        if self.use_firebase:
            self.db.collection("training_metrics").document(doc_id).set(data)
        else:
            store = _load_local("training_metrics")
            store[doc_id] = data
            _save_local("training_metrics", store)

        logger.info(f"Training metrics saved for run '{run_id}'")
        return doc_id

    async def get_training_metrics(self, run_id: str) -> Optional[dict]:
        """Get detailed training metrics for a run."""
        doc_id = f"metrics_{run_id}"

        if self.use_firebase:
            doc = self.db.collection("training_metrics").document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        else:
            store = _load_local("training_metrics")
            return store.get(doc_id)

    async def get_training_history(self, limit_n: int = 20) -> list[dict]:
        """Get recent training runs."""
        if self.use_firebase:
            docs = (
                self.db.collection("training_runs")
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(limit_n)
                .stream()
            )
            return [doc.to_dict() for doc in docs]
        else:
            store = _load_local("training_runs")
            runs = sorted(store.values(), key=lambda x: x.get("created_at", ""), reverse=True)
            return runs[:limit_n]

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoints (model binary + metadata)
    # ─────────────────────────────────────────────────────────────────────

    async def save_checkpoint(
        self,
        run_id: str,
        local_zip_path: str,
        metadata: dict,
    ) -> str:
        """
        Persist a model checkpoint:
        - Upload the .zip to Firebase Storage under  checkpoints/<run_id>.zip
        - Store metadata (algorithm, reward, config …) in Firestore  checkpoints/<run_id>
        - Always keeps local copy as well for fast inference
        """
        doc_id = run_id
        meta = {
            **metadata,
            "run_id": run_id,
            "local_path": local_zip_path,
            "created_at": datetime.utcnow().isoformat(),
        }

        if self.use_firebase:
            # Upload binary to Storage
            if self.bucket is not None:
                blob_name = f"checkpoints/{run_id}.zip"
                try:
                    blob = self.bucket.blob(blob_name)
                    blob.upload_from_filename(local_zip_path)
                    meta["storage_path"] = blob_name
                    try:
                        blob.make_public()
                        meta["storage_url"] = blob.public_url
                    except Exception:
                        pass  # public access may be disabled
                    logger.info(f"Checkpoint uploaded to Storage: {blob_name}")
                except Exception as exc:
                    logger.warning(f"Storage upload failed (keeping local only): {exc}")

            # Store metadata in Firestore
            self.db.collection("checkpoints").document(doc_id).set(meta)
        else:
            store = _load_local("checkpoints")
            store[doc_id] = meta
            _save_local("checkpoints", store)

        logger.info(f"Checkpoint metadata saved: {run_id}")
        return doc_id

    async def list_checkpoints(self) -> list[dict]:
        """List all saved checkpoints with their metadata."""
        if self.use_firebase:
            docs = self.db.collection("checkpoints").order_by(
                "created_at", direction=firestore.Query.DESCENDING
            ).stream()
            return [doc.to_dict() for doc in docs]
        else:
            store = _load_local("checkpoints")
            items = sorted(store.values(), key=lambda x: x.get("created_at", ""), reverse=True)
            return items

    async def get_checkpoint(self, run_id: str) -> Optional[dict]:
        """Get checkpoint metadata by run_id."""
        if self.use_firebase:
            doc = self.db.collection("checkpoints").document(run_id).get()
            return doc.to_dict() if doc.exists else None
        else:
            store = _load_local("checkpoints")
            return store.get(run_id)

    async def download_checkpoint(self, run_id: str, dest_path: str) -> str:
        """
        Download a checkpoint .zip from Firebase Storage to a local path.
        Returns the local path of the downloaded file.
        If the file already exists locally, skips download.
        """
        # Check local first
        if os.path.exists(dest_path):
            logger.info(f"Checkpoint already local: {dest_path}")
            return dest_path

        if self.use_firebase and self.bucket is not None:
            blob_name = f"checkpoints/{run_id}.zip"
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                blob.download_to_filename(dest_path)
                logger.info(f"Checkpoint downloaded: {blob_name} → {dest_path}")
                return dest_path
            else:
                raise FileNotFoundError(f"Checkpoint not in Storage: {blob_name}")
        else:
            raise FileNotFoundError(f"Checkpoint not found locally: {dest_path}")

    async def delete_checkpoint(self, run_id: str) -> bool:
        """Delete a checkpoint from Firestore, Storage, and local disk."""
        if self.use_firebase:
            # Delete from Firestore
            self.db.collection("checkpoints").document(run_id).delete()

            # Delete from Storage
            if self.bucket is not None:
                blob_name = f"checkpoints/{run_id}.zip"
                blob = self.bucket.blob(blob_name)
                if blob.exists():
                    blob.delete()

        else:
            store = _load_local("checkpoints")
            store.pop(run_id, None)
            _save_local("checkpoints", store)

        # Delete local files
        cp_dir = settings.RL_CHECKPOINT_DIR
        for ext in (".zip", "_metrics.json"):
            fpath = os.path.join(cp_dir, f"{run_id}{ext}")
            if os.path.exists(fpath):
                os.remove(fpath)

        logger.info(f"Checkpoint deleted: {run_id}")
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Optimization results
    # ─────────────────────────────────────────────────────────────────────

    async def save_optimization_result(self, result: dict) -> str:
        """Save an optimization result."""
        result_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result["result_id"] = result_id
        result["created_at"] = datetime.utcnow().isoformat()

        if self.use_firebase:
            self.db.collection("optimization_results").document(result_id).set(result)
        else:
            store = _load_local("optimization_results")
            store[result_id] = result
            _save_local("optimization_results", store)

        return result_id

    async def get_optimization_history(self, limit_n: int = 20) -> list[dict]:
        """Get recent optimization results."""
        if self.use_firebase:
            docs = (
                self.db.collection("optimization_results")
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(limit_n)
                .stream()
            )
            return [doc.to_dict() for doc in docs]
        else:
            store = _load_local("optimization_results")
            results = sorted(store.values(), key=lambda x: x.get("created_at", ""), reverse=True)
            return results[:limit_n]
