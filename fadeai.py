"""
fadeai.py
---------
Flask application entrypoint for FadeAI - Intelligent Object Removal & AI Photo Studio.

Routes:
    GET  /                      -> main UI (upload, canvas editor, results, dashboard)
    POST /api/upload            -> upload + validate source image
    POST /api/remove-object     -> brush/lasso AI inpainting
    POST /api/detect-objects    -> smart object detection (COCO 80 categories)
    POST /api/remove-background -> background remover (transparent PNG)
    POST /api/enhance-image     -> 2x/4x super-resolution, sharpen & denoise
    GET  /api/stats             -> usage dashboard metrics
    GET  /api/download/<f>      -> download result image
    GET  /api/health            -> readiness & device health check
    GET  /favicon.ico           -> clean 204 response
"""

import os
import time
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import config_map
from services.background_service import remove_background
from services.detection_service import detect_objects
from services.enhancement_service import enhance_image
from services.stats_service import get_stats, record_action
from services.inpainting_service import InpaintingService
from utils.file_utils import cleanup_old_files, save_upload
from utils.logger import get_logger
from utils.validators import (
    ValidationError,
    decode_base64_mask,
    validate_image_file,
    validate_mask_bytes,
)

logger = get_logger(__name__)


def create_app(env: str = None) -> Flask:
    """Application factory: builds and configures the Flask app instance."""
    env = env or os.environ.get("FLASK_ENV", "default")
    app_config = config_map.get(env, config_map["default"])

    app = Flask(__name__)
    app.config.from_object(app_config)
    app_config.init_app(app)

    inpainting_service = InpaintingService(app_config)

    def _resolve_image_path(filename: str) -> Optional[Path]:
        """Resolves filename within upload or result folders, blocking traversal."""
        if not filename:
            return None
        upload_root = Path(app.config["UPLOAD_FOLDER"]).resolve()
        result_root = Path(app.config["RESULT_FOLDER"]).resolve()

        source_path = (upload_root / filename).resolve()
        if source_path.exists() and (upload_root in source_path.parents or source_path == upload_root):
            return source_path

        result_path = (result_root / filename).resolve()
        if result_path.exists() and (result_root in result_path.parents or result_path == result_root):
            return result_path

        return None

    @app.before_request
    def before_request():
        try:
            cleanup_old_files(app.config["UPLOAD_FOLDER"], max_age_hours=24)
            cleanup_old_files(app.config["RESULT_FOLDER"], max_age_hours=24)
        except Exception as exc:
            logger.warning("Cleanup old files warning: %s", exc)

    # ------------------------------------------------------------------
    # Frontend Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Render the single-page FadeAI studio."""
        return render_template(
            "index.html",
            app_name=app.config.get("APP_NAME", "FadeAI"),
            max_upload_size_mb=app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024) // (1024 * 1024),
            supported_extensions=sorted(list(app.config.get("ALLOWED_EXTENSIONS", {"jpg", "jpeg", "png", "webp"}))),
            backend=app.config.get("INPAINT_BACKEND", "lama"),
        )

    # ------------------------------------------------------------------
    # API: Image Upload & Validation
    # ------------------------------------------------------------------

    @app.route("/api/upload", methods=["POST"])
    def upload_image():
        """Accepts a single photo upload, validates format/dimensions, and stores it."""
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided in request."}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        try:
            allowed = app.config.get("ALLOWED_EXTENSIONS", {"jpg", "jpeg", "png", "webp"})
            max_mb = app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024) / (1024 * 1024)
            validate_image_file(file, allowed, max_size_mb=max_mb)

            dest_folder = Path(app.config["UPLOAD_FOLDER"])
            saved_path = save_upload(file, dest_folder)

            return jsonify(
                {
                    "success": True,
                    "filename": saved_path.name,
                    "url": f"/static/uploads/{saved_path.name}",
                }
            )
        except ValidationError as val_err:
            return jsonify({"success": False, "error": str(val_err)}), 400
        except Exception as exc:
            logger.error("Upload error:\n%s", traceback.format_exc())
            return jsonify({"success": False, "error": "An error occurred while saving the upload."}), 500

    # ------------------------------------------------------------------
    # Feature: Object Removal (Inpainting)
    # ------------------------------------------------------------------

    @app.route("/api/remove-object", methods=["POST"])
    def remove_object_route():
        """Core AI inpainting engine applying mask to remove objects."""
        start_time = time.time()
        try:
            data = request.get_json(silent=True) or {}
            filename = data.get("filename")
            mask_data_url = data.get("mask")

            if not filename or not mask_data_url:
                return jsonify({"success": False, "error": "Both 'filename' and 'mask' data are required."}), 400

            resolved_path = _resolve_image_path(filename)
            if not resolved_path:
                return jsonify({"success": False, "error": f"Image file '{filename}' not found."}), 404

            mask_bytes = decode_base64_mask(mask_data_url)
            validate_mask_bytes(mask_bytes)

            results_dir = Path(app.config["RESULT_FOLDER"]).resolve()
            result_path, result_filename = inpainting_service.remove_object(
                image_path=resolved_path,
                mask_bytes=mask_bytes,
                result_folder=results_dir,
            )

            elapsed = round(time.time() - start_time, 2)
            logger.info("Inpainting request completed in %.2fs", elapsed)
            record_action("remove_object", filename, result_filename, elapsed)

            return jsonify(
                {
                    "success": True,
                    "result_url": f"/static/results/{result_filename}",
                    "result_filename": result_filename,
                    "processing_time_seconds": elapsed,
                    "backend": app.config["INPAINT_BACKEND"],
                }
            )
        except ValidationError as val_err:
            return jsonify({"success": False, "error": str(val_err)}), 400
        except Exception as exc:
            logger.error("Inpainting failure:\n%s", traceback.format_exc())
            return jsonify({"success": False, "error": f"Inpainting failed: {str(exc)}"}), 500

    # ------------------------------------------------------------------
    # Feature: Smart Object Detection
    # ------------------------------------------------------------------

    @app.route("/api/detect-objects", methods=["POST"])
    def detect_objects_route():
        """Runs FasterRCNN / COCO detector to locate objects."""
        start_time = time.time()
        try:
            payload = request.get_json(silent=True) or {}
            filename = payload.get("filename")

            resolved = _resolve_image_path(filename)
            if not resolved:
                return jsonify({"success": False, "error": "Image not found."}), 404

            objects = detect_objects(
                image_path=resolved,
                confidence_threshold=0.35,
                max_dimension=app.config.get("MAX_IMAGE_DIMENSION", 2048),
            )

            elapsed = round(time.time() - start_time, 2)
            record_action("detect_objects", filename, None, elapsed, {"count": len(objects)})

            return jsonify(
                {
                    "success": True,
                    "objects": objects,
                    "count": len(objects),
                    "processing_time_seconds": elapsed,
                }
            )
        except Exception as exc:
            logger.error("Object detection failed:\n%s", traceback.format_exc())
            return jsonify({"success": False, "error": f"Object detection failed: {str(exc)}"}), 500

    # ------------------------------------------------------------------
    # Feature: Background Remover
    # ------------------------------------------------------------------

    @app.route("/api/remove-background", methods=["POST"])
    def remove_background_route():
        """Automatically isolate subject and export transparent PNG."""
        start_time = time.time()
        try:
            payload = request.get_json(silent=True) or {}
            filename = payload.get("filename")

            resolved = _resolve_image_path(filename)
            if not resolved:
                return jsonify({"success": False, "error": "Image not found."}), 404

            results_dir = Path(app.config["RESULT_FOLDER"]).resolve()
            result_path, result_filename = remove_background(
                source_image_path=resolved,
                result_folder=results_dir,
                max_dimension=app.config.get("MAX_IMAGE_DIMENSION", 2048),
            )

            elapsed = round(time.time() - start_time, 2)
            record_action("remove_background", filename, result_filename, elapsed)

            return jsonify(
                {
                    "success": True,
                    "result_url": f"/static/results/{result_filename}",
                    "result_filename": result_filename,
                    "processing_time_seconds": elapsed,
                }
            )
        except Exception as exc:
            logger.error("Background removal failed:\n%s", traceback.format_exc())
            return jsonify({"success": False, "error": "Background removal failed. Please try again."}), 500

    # ------------------------------------------------------------------
    # Feature: AI Image Enhancement
    # ------------------------------------------------------------------

    @app.route("/api/enhance-image", methods=["POST"])
    def enhance_image_route():
        """Upscale (2x/4x), sharpen, and denoise image."""
        start_time = time.time()
        try:
            payload = request.get_json(silent=True) or {}
            filename = payload.get("filename")
            scale = int(payload.get("scale", 2))
            denoise = bool(payload.get("denoise", True))
            sharpen = bool(payload.get("sharpen", True))
            enhance_contrast = bool(payload.get("enhance_contrast", True))

            resolved = _resolve_image_path(filename)
            if not resolved:
                return jsonify({"success": False, "error": "Image not found."}), 404

            results_dir = Path(app.config["RESULT_FOLDER"]).resolve()
            result_path, result_filename = enhance_image(
                source_image_path=resolved,
                result_folder=results_dir,
                scale=scale,
                denoise=denoise,
                sharpen=sharpen,
                enhance_contrast=enhance_contrast,
            )

            elapsed = round(time.time() - start_time, 2)
            record_action("enhance_image", filename, result_filename, elapsed, {"scale": scale})

            return jsonify(
                {
                    "success": True,
                    "result_url": f"/static/results/{result_filename}",
                    "result_filename": result_filename,
                    "scale": scale,
                    "processing_time_seconds": elapsed,
                }
            )
        except Exception as exc:
            logger.error("Image enhancement failed:\n%s", traceback.format_exc())
            return jsonify({"success": False, "error": "Image enhancement failed."}), 500

    # ------------------------------------------------------------------
    # Feature: Usage Dashboard
    # ------------------------------------------------------------------

    @app.route("/api/stats", methods=["GET"])
    def get_stats_route():
        """Retrieve usage statistics and metrics."""
        stats = get_stats()
        return jsonify({"success": True, "stats": stats})

    # ------------------------------------------------------------------
    # Utility & Downloads
    # ------------------------------------------------------------------

    @app.route("/api/download/<path:filename>")
    def download_result(filename):
        """Serve a generated result image as a downloadable attachment."""
        results_dir = Path(app.config["RESULT_FOLDER"]).resolve()
        target = (results_dir / filename).resolve()

        if results_dir not in target.parents and target != results_dir:
            return jsonify({"success": False, "error": "Invalid filename."}), 400
        if not target.exists():
            return jsonify({"success": False, "error": "File not found."}), 404

        return send_from_directory(results_dir, filename, as_attachment=True)

    @app.route("/api/health")
    def health():
        """Health check reporting configured backend, features, and device."""
        import torch

        return jsonify(
            {
                "status": "ok",
                "backend": app.config["INPAINT_BACKEND"],
                "cuda_available": torch.cuda.is_available(),
                "device": app.config["DEVICE"],
                "features": [
                    "object_removal",
                    "smart_object_detection",
                    "background_remover",
                    "image_enhancement",
                    "usage_dashboard",
                ],
            }
        )

    @app.route("/favicon.ico")
    def favicon():
        """Handle browser favicon request cleanly without 404."""
        return ("", 204)

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(413)
    def handle_file_too_large(_error):
        return jsonify({"success": False, "error": "Uploaded file exceeds the size limit."}), 413

    @app.errorhandler(404)
    def handle_not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Resource not found."}), 404
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def handle_server_error(_error):
        logger.error("Internal server error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Internal server error."}), 500

    return app


# WSGI entrypoint
app = create_app()


if __name__ == "__main__":
    debug_mode = app.config.get("DEBUG", False)
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting FadeAI app on port %d (debug=%s)", port, debug_mode)
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
