"""
Simple IoU-based multi-object tracker.

Maintains active tracks across frames using Hungarian (linear-sum)
assignment on IoU costs.  Each track stores detection history, ReID
features, and face embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A single object track across frames."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float = 0.0

    # Lifecycle
    age: int = 0             # frames since creation
    hits: int = 1            # number of successful associations
    time_since_update: int = 0  # frames since last matched detection

    # Features (updated on match)
    reid_feature: Optional[np.ndarray] = None    # (dim,) L2-normalized
    face_embedding: Optional[np.ndarray] = None  # (512,) L2-normalized

    # State flags
    is_confirmed: bool = False
    is_deleted: bool = False


class Tracker:
    """IoU-based multi-object tracker with Hungarian assignment.

    Usage::

        tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
        tracks = tracker.update(detections)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._next_id: int = 1
        self._tracks: List[Track] = []

    # ── Public API ───────────────────────────────────────────────────

    @property
    def active_tracks(self) -> List[Track]:
        """Return currently active (non-deleted) tracks."""
        return [t for t in self._tracks if not t.is_deleted]

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Return confirmed tracks (hit threshold met)."""
        return [t for t in self._tracks if t.is_confirmed and not t.is_deleted]

    def update(
        self,
        detections: List["Detection"],
        reid_features: Optional[np.ndarray] = None,
        face_embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[Track]:
        """Update tracks with new detections.

        Args:
            detections: List of person detections in this frame.
            reid_features: Optional (N, dim) array of ReID features per detection.
            face_embeddings: Optional list of face embeddings per detection.

        Returns:
            List of currently active tracks after the update.
        """
        # Increment age and time_since_update for all tracks
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        if not detections:
            self._cleanup()
            return self.active_tracks

        det_boxes = np.array([d.as_xyxy() for d in detections])

        if not self._tracks:
            # No existing tracks — create new ones for all detections
            for i, det in enumerate(detections):
                self._create_track(det, i, reid_features, face_embeddings)
            return self.active_tracks

        # Build IoU cost matrix
        track_boxes = np.array([t.bbox for t in self._tracks if not t.is_deleted])
        active_tracks = [t for t in self._tracks if not t.is_deleted]

        if len(active_tracks) == 0:
            for i, det in enumerate(detections):
                self._create_track(det, i, reid_features, face_embeddings)
            return self.active_tracks

        iou_matrix = self._compute_iou_matrix(
            np.array([t.bbox for t in active_tracks]), det_boxes
        )
        cost_matrix = 1.0 - iou_matrix

        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks: set = set()
        matched_dets: set = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self._iou_threshold:
                track = active_tracks[row]
                det = detections[col]
                self._update_track(track, det, col, reid_features, face_embeddings)
                matched_tracks.add(row)
                matched_dets.add(col)

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det, i, reid_features, face_embeddings)

        self._cleanup()
        return self.active_tracks

    # ── Private ──────────────────────────────────────────────────────

    def _create_track(
        self,
        det: "Detection",
        det_idx: int,
        reid_features: Optional[np.ndarray],
        face_embeddings: Optional[List[Optional[np.ndarray]]],
    ) -> Track:
        """Create a new track from an unmatched detection."""
        track = Track(
            track_id=self._next_id,
            bbox=det.as_xyxy(),
            confidence=det.confidence,
        )
        self._next_id += 1

        if reid_features is not None and det_idx < len(reid_features):
            track.reid_feature = reid_features[det_idx]
        if face_embeddings is not None and det_idx < len(face_embeddings):
            track.face_embedding = face_embeddings[det_idx]

        self._tracks.append(track)
        logger.debug("New track #%d", track.track_id)
        return track

    def _update_track(
        self,
        track: Track,
        det: "Detection",
        det_idx: int,
        reid_features: Optional[np.ndarray],
        face_embeddings: Optional[List[Optional[np.ndarray]]],
    ) -> None:
        """Update an existing track with a matched detection."""
        track.bbox = det.as_xyxy()
        track.confidence = det.confidence
        track.hits += 1
        track.time_since_update = 0

        if reid_features is not None and det_idx < len(reid_features):
            track.reid_feature = reid_features[det_idx]
        if face_embeddings is not None and det_idx < len(face_embeddings):
            if face_embeddings[det_idx] is not None:
                track.face_embedding = face_embeddings[det_idx]

        if not track.is_confirmed and track.hits >= self._min_hits:
            track.is_confirmed = True
            logger.info("Track #%d confirmed.", track.track_id)

    def _cleanup(self) -> None:
        """Mark stale tracks as deleted."""
        for track in self._tracks:
            if track.time_since_update > self._max_age:
                track.is_deleted = True
                logger.debug("Track #%d deleted (stale).", track.track_id)

        # Prune deleted tracks to avoid unbounded growth
        self._tracks = [t for t in self._tracks if not t.is_deleted]

    # ── IoU computation ──────────────────────────────────────────────

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes.

        Args:
            boxes_a: (M, 4) array of [x1, y1, x2, y2].
            boxes_b: (N, 4) array of [x1, y1, x2, y2].

        Returns:
            (M, N) IoU matrix.
        """
        m = boxes_a.shape[0]
        n = boxes_b.shape[0]
        iou = np.zeros((m, n), dtype=np.float64)

        for i in range(m):
            for j in range(n):
                x1 = max(boxes_a[i, 0], boxes_b[j, 0])
                y1 = max(boxes_a[i, 1], boxes_b[j, 1])
                x2 = min(boxes_a[i, 2], boxes_b[j, 2])
                y2 = min(boxes_a[i, 3], boxes_b[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (
                    boxes_a[i, 3] - boxes_a[i, 1]
                )
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (
                    boxes_b[j, 3] - boxes_b[j, 1]
                )
                union = area_a + area_b - inter

                iou[i, j] = inter / union if union > 0 else 0.0

        return iou
