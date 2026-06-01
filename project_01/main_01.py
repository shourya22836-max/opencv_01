"""
Face Particle Mask — Overlay Edition
=====================================
Requirements:
    pip install opencv-python mediapipe numpy

What's new:
    • scipy removed — particle origins now use NumPy-only rejection sampling
      (sample random points inside the face bounding box, keep only those
       that fall within the filled face-oval polygon mask)
    • Everything else identical to the original

Controls:
    - Open hand   → particles BURST (float away from face surface)
    - Closed fist → particles COMPRESS (snap flat onto face)
    - Press 'Q'   → quit
"""

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands_det = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# Face oval landmark indices (ordered boundary)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
N_TOTAL    = 600
PARTICLE_R = 1
GLOW_R     = 4
CORE_R     = 1

SPRING_K     = 0.22
DAMPING_C    = 0.80
REPEL_STR    = 3.5
RELEASE_DAMP = 0.88
RETURN_PULL  = 0.06

BLUE_PALETTE = np.array([
    [255, 160,  40],
    [255, 100,   5],
    [220, 200, 110],
    [190,  70,   0],
    [255, 240, 180],
    [255, 130,  60],
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Build face oval polygon mask  (shared by origins + clipping)
# ─────────────────────────────────────────────────────────────
def build_face_mask(face_lm, W: int, H: int) -> np.ndarray:
    """Returns a binary mask (H, W) uint8 = 255 inside face oval."""
    lm  = face_lm.landmark
    pts = np.array(
        [[int(lm[i].x * W), int(lm[i].y * H)] for i in FACE_OVAL_IDX],
        dtype=np.int32,
    )
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ─────────────────────────────────────────────────────────────
# Build origins — NumPy rejection sampling (no scipy)
# ─────────────────────────────────────────────────────────────
def build_origins(face_lm, W: int, H: int, face_mask: np.ndarray) -> np.ndarray:
    """
    Sample N_TOTAL 2-D points that lie inside the face oval.

    Strategy
    --------
    1. Find the bounding box of the face oval.
    2. Draw random (x, y) pairs inside that box.
    3. Keep only those where face_mask == 255.
    4. Repeat until we have N_TOTAL accepted points.

    This is O(N) with a small constant because the oval typically
    covers ~60-70 % of its own bounding box, so acceptance is high.
    """
    lm  = face_lm.landmark
    pts = np.array(
        [[lm[i].x * W, lm[i].y * H] for i in FACE_OVAL_IDX],
        dtype=np.float32,
    )

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    bw = max(int(x_max - x_min) + 1, 1)
    bh = max(int(y_max - y_min) + 1, 1)

    accepted = []
    # Oversample to avoid too many loop iterations
    batch = max(N_TOTAL * 4, 2000)

    while len(accepted) < N_TOTAL:
        rx = (np.random.rand(batch) * bw + x_min).astype(np.float32)
        ry = (np.random.rand(batch) * bh + y_min).astype(np.float32)

        # Clamp to frame
        xi = np.clip(rx.astype(np.int32), 0, W - 1)
        yi = np.clip(ry.astype(np.int32), 0, H - 1)

        # Keep points inside the oval mask
        inside = face_mask[yi, xi] == 255
        rx, ry = rx[inside], ry[inside]

        if len(rx):
            accepted.append(np.stack([rx, ry], axis=1))

    origins = np.concatenate(accepted, axis=0)[:N_TOTAL]
    origins += np.random.uniform(-0.8, 0.8, origins.shape).astype(np.float32)
    return origins


# ─────────────────────────────────────────────────────────────
# Vectorised physics
# ─────────────────────────────────────────────────────────────
def update_particles(pos, vel, origins, compressed: bool):
    dx   = origins[:, 0] - pos[:, 0]
    dy   = origins[:, 1] - pos[:, 1]
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    if compressed:
        vel[:, 0] = (vel[:, 0] + dx * SPRING_K) * DAMPING_C
        vel[:, 1] = (vel[:, 1] + dy * SPRING_K) * DAMPING_C
    else:
        repel = np.where(dist < 80, REPEL_STR / (dist * 0.25 + 1), 0.0)
        nx = np.random.uniform(-0.1, 0.1, len(pos)).astype(np.float32)
        ny = np.random.uniform(-0.1, 0.1, len(pos)).astype(np.float32)
        vel[:, 0] = (vel[:,0] + (-dx/dist)*repel + nx + (dx/dist)*RETURN_PULL) * RELEASE_DAMP
        vel[:, 1] = (vel[:,1] + (-dy/dist)*repel + ny + (dy/dist)*RETURN_PULL) * RELEASE_DAMP

    pos += vel


# ─────────────────────────────────────────────────────────────
# Draw particles onto transparent overlay, then composite
# ─────────────────────────────────────────────────────────────
def draw_particles_on_face(frame, pos, colors, face_mask_img, W, H):
    particle_layer = np.zeros((H, W, 3), dtype=np.uint8)

    xs    = pos[:, 0].astype(np.int32)
    ys    = pos[:, 1].astype(np.int32)
    valid = np.where((xs >= 0) & (xs < W) & (ys >= 0) & (ys < H))[0]

    glow = np.zeros((H, W, 3), dtype=np.uint8)
    for i in valid:
        cv2.circle(glow, (xs[i], ys[i]), GLOW_R, colors[i].tolist(), -1)
    for i in valid:
        cv2.circle(particle_layer, (xs[i], ys[i]), PARTICLE_R, colors[i].tolist(), -1)
    for i in valid:
        cv2.circle(particle_layer, (xs[i], ys[i]), CORE_R, (255, 255, 255), -1)

    face_3ch       = cv2.merge([face_mask_img, face_mask_img, face_mask_img])
    glow           = cv2.bitwise_and(glow,           face_3ch)
    particle_layer = cv2.bitwise_and(particle_layer, face_3ch)

    out = frame.copy()
    cv2.addWeighted(glow, 0.55, out, 1.0, 0, out)
    mask_px = (particle_layer.sum(axis=2) > 0)
    out[mask_px] = (
        out[mask_px].astype(np.float32) * 0.3 +
        particle_layer[mask_px].astype(np.float32) * 0.7
    ).astype(np.uint8)

    return out


# ─────────────────────────────────────────────────────────────
# Hand gesture
# ─────────────────────────────────────────────────────────────
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIP  = [6, 10, 14, 18]

def is_hand_closed(hand_lm) -> bool:
    lm     = hand_lm.landmark
    closed = sum(1 for t, p in zip(FINGER_TIPS, FINGER_PIP) if lm[t].y > lm[p].y)
    return (closed + int(abs(lm[4].x - lm[3].x) < 0.07)) >= 4


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    W, H = 640, 480

    pos         = np.zeros((N_TOTAL, 2), dtype=np.float32) # detects the position starting from the origin.
    vel         = np.zeros((N_TOTAL, 2), dtype=np.float32)
    origins     = np.zeros((N_TOTAL, 2), dtype=np.float32)
    face_mask   = np.zeros((H, W), dtype=np.uint8)
    initialized = False

    color_idx = np.random.randint(0, len(BLUE_PALETTE), N_TOTAL)
    colors    = BLUE_PALETTE[color_idx]

    hand_closed   = False
    hand_detected = False
    face_detected = False
    frame_count   = 0 #frame_count increments every frame
    HAND_SKIP     = 2 #HAND_SKIP = 2 means hand detection only runs every 2nd frame (checked via frame_count % HAND_SKIP == 0), which saves CPU since hand detection is expensive

    fps_tm  = cv2.TickMeter()
    fps_tm.start()
    fps_val = 0.0

    print("Face Particle Mask — Overlay Edition")
    print("  Open hand = BURST  |  Fist = COMPRESS  |  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        face_res = face_mesh.process(rgb)
        hand_res = hands_det.process(rgb) if frame_count % HAND_SKIP == 0 else None

        rgb.flags.writeable = True

        # ── Face ─────────────────────────────────────────────
        face_detected = bool(face_res.multi_face_landmarks)
        if face_detected:
            fl        = face_res.multi_face_landmarks[0]
            face_mask = build_face_mask(fl, W, H)          # build mask first
            new_origins = build_origins(fl, W, H, face_mask)  # pass mask in

            if not initialized:
                origins[:]  = new_origins
                pos[:]      = new_origins.copy()
                initialized = True
            else:
                origins[:] = origins * 0.60 + new_origins * 0.40

        # ── Hand ─────────────────────────────────────────────
        if hand_res is not None:
            hand_detected = bool(hand_res.multi_hand_landmarks)
            if hand_detected:
                hl = hand_res.multi_hand_landmarks[0]
                hand_closed = is_hand_closed(hl)

        # ── Physics ───────────────────────────────────────────
        if initialized:
            update_particles(pos, vel, origins, hand_closed)

        # ── Render ────────────────────────────────────────────
        if initialized and face_detected:
            out = draw_particles_on_face(frame, pos, colors, face_mask, W, H)
        else:
            out = frame.copy()

        if hand_res is not None and hand_detected and hand_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                out, hand_res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(100, 100, 200), thickness=1),
            )

        # ── FPS ───────────────────────────────────────────────
        fps_tm.stop()
        if frame_count % 10 == 0:
            fps_val = fps_tm.getFPS()
        fps_tm.reset(); fps_tm.start()

        # ── HUD ───────────────────────────────────────────────
        state_txt   = "COMPRESS" if hand_closed else "BURST"
        state_color = (80, 80, 255) if hand_closed else (60, 240, 255)

        cv2.putText(out, state_txt,            (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"FPS {fps_val:.0f}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        if not face_detected:
            cv2.putText(out, "No face detected", (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)
        if not hand_detected:
            cv2.putText(out, "Show hand to control", (12, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(out, "Q quit", (W - 72, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        frame_count += 1
        cv2.imshow("Face Particle Mask", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands_det.close()


if __name__ == "__main__":
    main()