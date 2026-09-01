import time
from collections import deque


class AttentionEngine:

    def __init__(self):

        # ---------------------------------------------------------
        # SETTINGS
        # ---------------------------------------------------------

        self.GAZE_TOLERANCE_X = 0.20
        self.GAZE_TOLERANCE_Y = 0.20

        # Smoothing
        self.history_size = 15
        self.score_history = deque(maxlen=self.history_size)

        # Session timing
        self.session_start = time.time()
        self.last_update = time.time()

        # Current state
        self.score = 100.0
        self.status = "FOCUSED"

        # Distraction tracking
        self.distraction_start = None
        self.current_distraction_time = 0.0
        self.total_distraction_time = 0.0

        # Focus tracking
        self.total_focused_time = 0.0
        self.total_time = 0.0

        # Frames
        self.frames = 0
        self.focused_frames = 0
        self.partial_frames = 0
        self.distracted_frames = 0

        # Start with a perfect score
        for _ in range(5):
            self.score_history.append(100.0)

    # =============================================================
    # UPDATE
    # =============================================================

    def update(
        self,
        gaze_x=0.0,
        gaze_y=0.0,
        eyes_open=True,
        face_detected=True
    ):

        now = time.time()

        dt = now - self.last_update
        self.last_update = now

        # Protect against abnormal timing
        if dt < 0:
            dt = 0

        if dt > 0.25:
            dt = 0.25

        self.total_time += dt
        self.frames += 1

        # =========================================================
        # NO FACE
        # =========================================================

        if not face_detected:

            # Don't immediately punish a single missed frame.
            target_score = 65.0

            self.score_history.append(target_score)
            self.score = self._smooth()

            self._update_distraction("NO FACE", now, dt)

            return self.get_result()

        # =========================================================
        # GAZE CALCULATION
        # =========================================================

        x = abs(float(gaze_x))
        y = abs(float(gaze_y))

        # Remove the small natural movement around the center.
        x = max(0.0, x - self.GAZE_TOLERANCE_X)
        y = max(0.0, y - self.GAZE_TOLERANCE_Y)

        # Normalize
        x = min(x / 0.80, 1.0)
        y = min(y / 0.80, 1.0)

        # Horizontal movement is slightly more important.
        gaze_distance = (
            x * 0.65 +
            y * 0.35
        )

        # Convert to score
        gaze_score = 100.0 - (gaze_distance * 100.0)

        gaze_score = max(
            0.0,
            min(100.0, gaze_score)
        )

        # =========================================================
        # EYE STATE
        # =========================================================

        if eyes_open:

            eye_score = 100.0

        else:

            # Closed eyes don't instantly mean distracted.
            # They are only a moderate penalty.
            eye_score = 65.0

        # =========================================================
        # FINAL SCORE
        # =========================================================

        target_score = (
            gaze_score * 0.85 +
            eye_score * 0.15
        )

        target_score = max(
            0.0,
            min(100.0, target_score)
        )

        # =========================================================
        # SMOOTHING
        # =========================================================

        self.score_history.append(target_score)

        self.score = self._smooth()

        # =========================================================
        # STATUS
        # =========================================================

        if self.score >= 75:

            status = "FOCUSED"

        elif self.score >= 50:

            status = "PARTIALLY FOCUSED"

        else:

            status = "DISTRACTED"

        # =========================================================
        # TIME TRACKING
        # =========================================================

        self._update_distraction(
            status,
            now,
            dt
        )

        # Focus statistics

        if status == "FOCUSED":

            self.focused_frames += 1
            self.total_focused_time += dt

        elif status == "PARTIALLY FOCUSED":

            self.partial_frames += 1

        else:

            self.distracted_frames += 1

        self.status = status

        return self.get_result()

    # =============================================================
    # DISTRACTION TRACKING
    # =============================================================

    def _update_distraction(
        self,
        status,
        now,
        dt
    ):

        if status in (
            "DISTRACTED",
            "NO FACE"
        ):

            if self.distraction_start is None:

                self.distraction_start = now

            self.current_distraction_time = (
                now -
                self.distraction_start
            )

        else:

            if self.distraction_start is not None:

                duration = (
                    now -
                    self.distraction_start
                )

                self.total_distraction_time += duration

            self.distraction_start = None
            self.current_distraction_time = 0.0

    # =============================================================
    # SMOOTH SCORE
    # =============================================================

    def _smooth(self):

        if not self.score_history:
            return 100.0

        values = list(self.score_history)

        # Recent frames get more weight.
        weights = range(
            1,
            len(values) + 1
        )

        weighted_sum = sum(
            value * weight
            for value, weight
            in zip(values, weights)
        )

        total_weight = sum(weights)

        return weighted_sum / total_weight

    # =============================================================
    # ATTENTION %
    # =============================================================

    def get_attention_percentage(self):

        if self.total_time <= 0:
            return 100.0

        # Partial focus counts as half focus.
        effective_focus = (
            self.focused_frames +
            self.partial_frames * 0.5
        )

        total_count = (
            self.focused_frames +
            self.partial_frames +
            self.distracted_frames
        )

        if total_count == 0:
            return 100.0

        percentage = (
            effective_focus /
            total_count
        ) * 100.0

        return max(
            0.0,
            min(100.0, percentage)
        )

    # =============================================================
    # SESSION TIME
    # =============================================================

    def get_session_duration(self):

        return time.time() - self.session_start

    # =============================================================
    # RESULT
    # =============================================================

    def get_result(self):

        return {

            "score": round(
                self.score,
                1
            ),

            "status": self.status,

            "distraction_duration": round(
                self.current_distraction_time,
                1
            ),

            "total_distraction_time": round(
                self.total_distraction_time,
                1
            ),

            "focused_time": round(
                self.total_focused_time,
                1
            ),

            "session_time": round(
                self.get_session_duration(),
                1
            ),

            "attention_percentage": round(
                self.get_attention_percentage(),
                1
            ),

            "focused_frames":
                self.focused_frames,

            "partial_frames":
                self.partial_frames,

            "distracted_frames":
                self.distracted_frames
        }

    # =============================================================
    # RESET
    # =============================================================

    def reset(self):

        self.session_start = time.time()
        self.last_update = time.time()

        self.score = 100.0
        self.status = "FOCUSED"

        self.distraction_start = None
        self.current_distraction_time = 0.0
        self.total_distraction_time = 0.0

        self.total_focused_time = 0.0
        self.total_time = 0.0

        self.frames = 0
        self.focused_frames = 0
        self.partial_frames = 0
        self.distracted_frames = 0

        self.score_history.clear()

        for _ in range(5):
            self.score_history.append(100.0)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("        ATTENTION ENGINE TEST")
    print("=" * 55)

    engine = AttentionEngine()

    # ------------------------------------------------------------
    # TEST 1: LOOKING STRAIGHT
    # ------------------------------------------------------------

    print("\n1. LOOKING STRAIGHT")

    for _ in range(20):

        result = engine.update(
            gaze_x=0.0,
            gaze_y=0.0,
            eyes_open=True,
            face_detected=True
        )

        time.sleep(0.05)

    print(
        f"Score  : {result['score']}%"
    )

    print(
        f"Status : {result['status']}"
    )

    # ------------------------------------------------------------
    # TEST 2: SLIGHTLY AWAY
    # ------------------------------------------------------------

    print("\n2. SLIGHTLY LOOKING AWAY")

    for _ in range(20):

        result = engine.update(
            gaze_x=0.30,
            gaze_y=0.10,
            eyes_open=True,
            face_detected=True
        )

        time.sleep(0.05)

    print(
        f"Score  : {result['score']}%"
    )

    print(
        f"Status : {result['status']}"
    )

    # ------------------------------------------------------------
    # TEST 3: CLEARLY LOOKING AWAY
    # ------------------------------------------------------------

    print("\n3. CLEARLY LOOKING AWAY")

    for _ in range(20):

        result = engine.update(
            gaze_x=0.80,
            gaze_y=0.60,
            eyes_open=True,
            face_detected=True
        )

        time.sleep(0.05)

    print(
        f"Score  : {result['score']}%"
    )

    print(
        f"Status : {result['status']}"
    )

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print("\n" + "=" * 55)
    print("SESSION SUMMARY")
    print("=" * 55)

    print(
        f"Attention       : "
        f"{result['attention_percentage']}%"
    )

    print(
        f"Focused frames  : "
        f"{result['focused_frames']}"
    )

    print(
        f"Partial frames  : "
        f"{result['partial_frames']}"
    )

    print(
        f"Distracted      : "
        f"{result['distracted_frames']}"
    )

    print(
        f"Distraction     : "
        f"{result['total_distraction_time']} sec"
    )

    print("=" * 55)