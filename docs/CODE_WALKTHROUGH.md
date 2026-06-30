# Code Walkthrough — Gesture Recognition

This document is a complete, file-by-file, class-by-class, method-by-method walkthrough of every Python file in the `src/` directory of this repository. It exists as **thesis defense preparation material**: a reference you can use to answer "how does X work," "why did you do Y this way," and "where is Z implemented" with confidence and precision, citing exact line numbers.

**How to use this document:** Read the Module/Package Overview and Architectural Design sections first — they cover the *why* behind the system as a whole, which is what most "why did you choose..." defense questions target. Then use the per-file sections as a reference to refresh your memory on a specific file, or to look up exactly what a given line/method does if asked to point to code. Line numbers cited throughout refer to the current state of each file in this repository.

---

## Table of Contents

1. [Module/Package Overview](#modulepackage-overview)
2. [Architectural Design & Decisions](#architectural-design--decisions)
   - [2.1 Layered separation of concerns](#21-layered-separation-of-concerns)
   - [2.2 Dependency injection via `AppConfig`](#22-dependency-injection-via-appconfig)
   - [2.3 Two operational modes sharing one core](#23-two-operational-modes-sharing-one-core)
   - [2.4 Hybrid scale-normalization strategy](#24-hybrid-scale-normalization-strategy)
   - [2.5 Temporal smoothing](#25-temporal-smoothing-majority-vote)
   - [2.6 Dwell-based action triggering](#26-dwell-based-action-triggering)
   - [2.7 Model choice: Random Forest as production model](#27-model-choice-random-forest-as-production-model)
   - [2.8 Discrete vs. continuous action execution](#28-discrete-vs-continuous-action-execution-design)
   - [2.9 Persistent settings](#29-persistent-settings)
   - [2.10 Legacy code retained in the repo](#210-legacy-code-retained-in-the-repo)
3. Per-file sections:
   - [`src/app.py`](#srcapppy)
   - [`src/core/config.py`](#srccoreconfigpy)
   - [`src/core/gesture_detector.py`](#srccoregesture_detectorpy)
   - [`src/core/gesture_recognizer.py`](#srccoregesture_recognizerpy)
   - [`src/core/action_executor.py`](#srccoreaction_executorpy)
   - [`src/utils/normalize.py`](#srcutilsnormalizepy)
   - [`src/utils/fps_counter.py`](#srcutilsfps_counterpy)
   - [`src/ui/ui_utils.py`](#srcuiui_utilspy)
   - [`src/ui/menu_window.py`](#srcuimenu_windowpy)
   - [`src/ui/testing_mode_window.py`](#srcuitesting_mode_windowpy)
   - [`src/ui/compact_mode_window.py`](#srcuicompact_mode_windowpy)
   - [`src/collect_data.py`](#srccollect_datapy)
   - [`src/train_model.py`](#srctrain_modelpy)
   - [`src/analyze_model.py`](#srcanalyze_modelpy)
   - [`src/main.py`](#srcmainpy-legacy) (legacy)
   - [`src/realtime_recognition.py`](#srcrealtime_recognitionpy-legacy) (legacy)
   - [The `__init__.py` files](#the-__init__py-files)
4. [Appendix: `temp_scripts/`](#appendix-temp_scripts)

---

## Module/Package Overview

The application is organized as a small Python package rooted at `src/`, split into three sub-packages plus a handful of top-level scripts:

- **`src/core/`** — the domain logic. Contains everything that turns a camera frame into a recognized gesture and a recognized gesture into an OS-level action: `GestureDetector` (MediaPipe hand-landmark extraction), `GestureRecognizer` (ML prediction + temporal smoothing), `ActionExecutor` (the dwell-trigger state machine and every concrete action handler), and `AppConfig` (centralized configuration). Nothing in `core/` knows anything about windows, buttons, or OpenCV drawing — it is pure logic that could in principle be reused headlessly or tested without any UI.

- **`src/ui/`** — presentation. Two interchangeable "mode windows" (`TestingModeWindow`, `CompactModeWindow`) that each wire the same `core/` components into a runnable OpenCV-based loop, a `MenuWindow` that launches one or the other, and `ui_utils.py`, a library of stateless drawing helpers shared by both modes.

- **`src/utils/`** — small, stateless, pure-function helpers with no dependency on the rest of the app: landmark normalization math (`normalize.py`) and an FPS counter (`fps_counter.py`). These are intentionally framework-agnostic (no MediaPipe/OpenCV/tkinter imports) so they're trivially unit-testable.

- **Top-level scripts in `src/`** — `app.py` (the real entry point), `collect_data.py` (training-data collection tool), `train_model.py` (training/evaluation pipeline), `analyze_model.py` (post-hoc model analysis/plots), and two **legacy** scripts, `main.py` and `realtime_recognition.py`, kept in the repo as historical artifacts of the project's evolution but not part of the live application (see [2.10](#210-legacy-code-retained-in-the-repo)).

---

## Architectural Design & Decisions

This section explains the *why* behind the major design choices in the codebase — the kind of question a thesis committee is likely to ask beyond "what does this code do."

### 2.1 Layered separation of concerns

The codebase is split into `core/` (domain logic), `ui/` (presentation), and `utils/` (stateless helpers) rather than being one large script. The motivating reason is that there are **two operational modes** (Testing Mode and Compact Mode) that must run the *exact same* detection → normalization → recognition pipeline but present and act on it differently. If that pipeline lived inside a single monolithic script, supporting two modes would mean either duplicating the recognition logic (as `realtime_recognition.py` originally did — see [2.10](#210-legacy-code-retained-in-the-repo)) or threading mode-specific branches through the same function. Instead, `core/` exposes `GestureDetector`, `GestureRecognizer`, and `ActionExecutor` as independent, UI-agnostic classes; `TestingModeWindow` and `CompactModeWindow` each instantiate and drive these classes from their own `run()` loop. A bug fix or accuracy improvement in normalization or prediction logic in `core/` automatically benefits both modes with zero duplication. `utils/` is split out further because `normalize.py` and `fps_counter.py` have no dependency on MediaPipe, OpenCV, or tkinter at all — they're pure math/timing utilities, which keeps them trivially testable in isolation and reusable from `collect_data.py` and `train_model.py` without pulling in UI dependencies.

### 2.2 Dependency injection via `AppConfig`

Every tunable constant in the system (camera index, confidence thresholds, dwell-frame counts, hold durations, colors, window sizes, the gesture→action mapping) lives in one `@dataclass`, `AppConfig` (`src/core/config.py`), which is constructed once in `app.py` and passed by reference into every component that needs it (`MenuWindow`, `TestingModeWindow`, `CompactModeWindow`, and transitively `GestureDetector`/`GestureRecognizer`/`ActionExecutor`). The alternative — module-level constants or magic numbers scattered through each file — would make tuning the system (e.g., adjusting cursor sensitivity or dwell time during testing) require hunting through multiple files, and would make it impossible to ever run two differently-configured instances side by side. A single injected config object means every component reads from the same source of truth, and reviewers/testers can answer "what does X equal" by looking in exactly one place.

`@dataclass` was chosen over a plain class or a dict because it auto-generates `__init__`, `__repr__`, and `__eq__` from type-annotated field declarations, which is both less boilerplate and self-documenting (the field list *is* the constructor signature and the type contract). `field(default_factory=...)` is required — not optional style — for the three mutable defaults (`PROJECT_ROOT`, `SCREENSHOT_SAVE_DIR`, `GESTURE_ACTIONS`): Python dataclasses (like all Python functions) only evaluate default *expressions* once, at class-definition time, so a literal mutable default (e.g. `GESTURE_ACTIONS: dict = {}`) would be the same dict object shared and mutated across every `AppConfig` instance — `field(default_factory=lambda: {...})` calls the factory function fresh on each instantiation, sidestepping that bug entirely. `__post_init__` is a dataclass-provided hook that automatically runs immediately after the generated `__init__` finishes; it's used here to *validate* the config the moment it's constructed (project root exists, gesture map file exists, model file exists), failing fast with a clear `FileNotFoundError`/`ValueError` instead of letting the app crash confusingly deep inside `GestureRecognizer` later.

### 2.3 Two operational modes sharing the same core

Testing Mode (`TestingModeWindow`) and Compact Mode (`CompactModeWindow`) both wrap the same `GestureDetector` + `GestureRecognizer` core, but Testing Mode never calls `ActionExecutor` at all — it is **safe by default**: a full-size diagnostic window that shows raw landmarks, normalization scale values, confidence, and predictions, but never moves the mouse, presses a key, or changes system volume. This exists specifically so that during data collection, model iteration, and live demoing/debugging you can freely wave your hand in front of the camera without accidentally triggering screenshots, volume changes, or window minimizing. Compact Mode is the "real" mode — small, always-on-top, and wired to `ActionExecutor` for live OS control. Because both modes are built from the same `core/` classes, the recognition behavior you validate in Testing Mode is guaranteed to be the same recognition behavior driving actions in Compact Mode.

The menu/withdraw-deiconify pattern in `menu_window.py` (`self.root.withdraw()` before launching a mode, `self.root.deiconify()` after it returns — see lines 125/135 and 141/151) exists because tkinter's `Tk()` root window is expensive to tear down and recreate, and a process can only cleanly have one `Tk()` root at a time. `withdraw()` hides the menu window without destroying its underlying tkinter interpreter state, so when the user exits a mode and returns to the menu, `deiconify()` simply re-shows the existing window rather than re-initializing tkinter from scratch.

### 2.4 Hybrid scale-normalization strategy

For a classifier to recognize a gesture regardless of how close the hand is to the camera or where in the frame it sits, the raw `(x, y)` landmark coordinates need to be made translation- and scale-invariant before being fed to the model. Translation invariance is solved simply, by subtracting the wrist landmark (index 0) from every other landmark (`src/utils/normalize.py` lines 82-83) so the wrist becomes the local origin.

Scale invariance is more subtle. A single scale measure — e.g. just the wrist-to-middle-finger-MCP distance — is fragile: it shrinks toward zero whenever the hand is angled away from the camera or partially closed, which would inflate the normalized coordinates unpredictably. The codebase instead computes **two** independent scale measures — wrist→middle-MCP distance (landmarks 0↔9) and palm width, index-MCP→pinky-MCP distance (landmarks 5↔17) — and uses `scale_used = max(scale_wrist_mcp, scale_palm_width)` (line 90). Taking the *max* of the two measures means that whichever distance is currently more "open"/visible dominates, which makes the scale figure noticeably more stable across hand rotation and partial occlusion than either measure alone.

A **minimum-scale safety guard** (`threshold=0.05`, line 93) exists because if the hand is very far from the camera or MediaPipe's landmark estimate is degenerate, `scale_used` can collapse toward zero, and dividing by a near-zero scale would blow up the normalized coordinates into huge/unstable values that would poison both training data and live predictions. Below the threshold, normalization is rejected outright (`valid: False`) rather than returning garbage. A second, independent safety net — checking `np.isfinite` on the final flattened vector (lines 109-116) — catches any residual NaN/Inf that could still arise from edge-case landmark geometry, even when the scale guard alone would have passed.

### 2.5 Temporal smoothing (majority vote)

Raw per-frame predictions flicker: even a stable hand pose produces an occasional single-frame misclassification due to landmark jitter. `GestureRecognizer._smooth_prediction` (`gesture_recognizer.py` lines 110-136) keeps a sliding window of the last `history_size` (default 5) raw predictions and returns the majority label in that window, which absorbs isolated misfires without meaningfully delaying genuine gesture changes. This is exposed as a toggle in Testing Mode (the `s` key, `testing_mode_window.py`) specifically so the *effect* of smoothing can be demonstrated and compared live — useful for a defense demo to show "this is what raw per-frame prediction looks like vs. smoothed." Compact Mode always uses smoothing (`predict_smooth`, no raw-mode toggle) because in the action-execution context flicker isn't just a visual annoyance, it's a correctness problem: an unsmoothed flicker could reset the dwell counter (see 2.6) or interrupt a continuous control mid-gesture, so there's no use case in Compact Mode where disabling it would be desirable.

### 2.6 Dwell-based action triggering

If every recognized gesture fired its action on the very first frame it was detected, the system would be extremely prone to false positives: transitional hand poses while moving between two intended gestures would be classified as *something* for a frame or two and could trigger unwanted actions. `ActionExecutor.update()` (`action_executor.py` lines 163-273) instead requires a gesture to be held for `min_dwell_frames` (config default 5) *consecutive* frames before it fires, and fires it exactly once per dwell period via a `frame_count == min_dwell_frames` equality check (not `>=`) combined with a `triggered_this_gesture` flag — so holding a gesture for 30 frames triggers the action once, not 26 times.

This dwell requirement is bypassed, by design, for **continuous controls** once they've been activated: cursor movement, drag, volume, and scroll don't need to re-dwell every frame to keep moving — dwell is the *activation* gate (you must hold the gesture for 5 frames to start cursor control), but once active, the corresponding `update_continuous_control`/`update_drag_control`/`update_volume_control`/`update_scroll_control` methods run every single frame for as long as the gesture is sustained, because re-requiring a 5-frame dwell on every frame would make continuous interaction (e.g. smoothly moving the mouse) impossible.

### 2.7 Model choice: Random Forest as production model

`train_model.py` trains and rigorously evaluates three classifiers — Random Forest, k-NN, and SVM (RBF kernel) — with 5-fold cross-validation, paired t-tests for statistical significance, learning curves, and per-classifier inference-time benchmarking. Despite this full comparison, `save_models()` (lines 698-728) **hardcodes** Random Forest as the model written to `gesture_classifier_latest.pkl`, regardless of which classifier scores highest in a given training run. This is a deliberate, defensible choice, not an oversight: RF is highly interpretable (feature importances map directly back to which landmark coordinates matter, which is exactly what `analyze_model.py`'s `plot_feature_importance` visualizes — see [analyze_model.py](#srcanalyze_modelpy)), it is robust to the kind of noisy, moderately-sized, tabular landmark data this project produces without heavy hyperparameter tuning, and ensemble-of-trees inference is fast and consistent at runtime (no per-call kernel computation like SVM, no full-training-set distance scan like k-NN at predict time). The 3-way comparison in `train_model.py` exists to *demonstrate and justify* this choice empirically (showing RF is competitive or better, with statistical backing), not to dynamically pick the best model — for a real-time interactive system, a stable, well-understood, fast-at-inference model that's easy to explain in a thesis defense was prioritized over squeezing out a possibly marginal accuracy gain from a less interpretable model.

### 2.8 Discrete vs. continuous action execution design

`ActionExecutor` handles two structurally different categories of action, and the code is split accordingly. **Discrete one-shot actions** (click, screenshot, media keys, volume increment) are triggered once per dwell period and execute instantaneously via `pyautogui` (clicks, screenshots, scroll ticks) or `pynput` (media keys) — these go through `execute_action()`'s single dispatch table (lines 275-311). **Continuous per-frame actions** (cursor movement, drag, scroll repeat, volume repeat, swipe tracking) need fresh per-frame state (current hand position vs. an activation-time origin) and must run every frame for as long as the gesture is sustained — these live in separate `update_*` methods (`update_continuous_control`, `update_drag_control`, `update_volume_control`, `update_scroll_control`, `update_swipe_detection`, etc.) that the UI layer calls unconditionally every frame, each of which internally no-ops if its subsystem isn't currently active. Splitting these into different code paths (rather than trying to unify "trigger once" and "repeat every frame" behind one interface) keeps each path's logic simple: the discrete path is a flat dispatch table, the continuous path is a set of independent per-frame state machines.

Three different libraries are used for different responsibilities because no single library covers all of them well: **`pyautogui`** handles cross-platform mouse movement, clicks, scrolling, and screenshots (simple, high-level API, but has a default 0.1s inter-call pause that is explicitly disabled via `pyautogui.PAUSE = 0` in `compact_mode_window.py` line 9 for responsive continuous cursor movement). **`pynput`** is used specifically for press/release-style control that `pyautogui` doesn't model as cleanly — holding the mouse button down across multiple frames for drag, and sending OS-level media keys (play/pause, next/previous track) that work regardless of which window has focus. **`pycaw`** is a Windows-specific COM wrapper around the WASAPI audio endpoint volume interface — there's no cross-platform equivalent with this level of control, so it's used directly to read/set the actual system master volume (`GetMasterVolumeLevelScalar`/`SetMasterVolumeLevelScalar`).

### 2.9 Persistent settings

`data/app_settings.json` stores one piece of cross-session state — `black_screen_enabled` — plus a `version` field. A small JSON file was chosen over re-deriving this value at every startup because it's genuinely user-chosen, session-spanning state (whether the camera feed should render as a black rectangle instead of the live image, e.g., for privacy) with no other natural source of truth — it isn't derived from anything else in the system, so it has to be persisted somewhere if it's meant to survive an app restart. JSON was chosen over a database or more complex format because the state is a single small flat object; `_load_black_screen_setting`/`_save_black_screen_setting` (`compact_mode_window.py` lines 544-605) include a graceful fallback to defaults if the file is missing or corrupted, auto-recreating it, so a deleted or malformed settings file degrades to default behavior instead of crashing the app.

### 2.10 Legacy code retained in the repo

`src/main.py` and `src/realtime_recognition.py` are earlier prototypes that predate the current `core/`/`ui/`/`utils/` package structure. `main.py` is a minimal raw MediaPipe script (prints landmark coordinates, no classification at all) — essentially the first proof-of-concept that the camera + MediaPipe pipeline worked. `realtime_recognition.py` is a more developed but self-contained prototype that duplicates what `GestureDetector` + `GestureRecognizer` + `ActionExecutor` + the UI layer now do, all in one file, and still imports from an old flat module path (`from utils.normalize import ...`, no `src.` prefix) and references a gesture, `"L_shape"`, that no longer exists in `data/gesture_map.json`. Both are kept in the repository rather than deleted because they document the project's actual development trajectory — from "does MediaPipe even detect my hand" to "one monolithic script doing everything" to the current layered, configurable, dual-mode architecture — which is a reasonable and fairly common narrative arc to be able to describe in a defense if asked "did the design evolve, and how." They are explicitly **not** part of the live application; `src/app.py` is the only real entry point.

---

## `src/app.py`

**Purpose:** The real, single entry point of the application. Constructs the shared `AppConfig`, builds the `MenuWindow`, and runs it, with top-level error handling for the two realistic startup failure modes (missing model/data files, and anything else unexpected).

- **Lines 1-7**: module docstring — states this launches either Testing or Compact mode via the menu.
- **Lines 9-10**: imports `AppConfig` from `core.config` and `MenuWindow` from `ui.menu_window`.
- **`main()` (lines 13-42)**:
  - Lines 14-17: prints a startup banner.
  - Lines 19-21 (`try` block): `config = AppConfig()` — this single line is where config validation actually happens, since `AppConfig.__post_init__` runs automatically as part of construction (see [2.2](#22-dependency-injection-via-appconfig)) and will raise before `MenuWindow` is ever touched if the model or gesture-map files are missing. `menu = MenuWindow(config)` then `menu.run()` hands control to the tkinter event loop for the rest of the program's life.
  - Lines 23-27 (`except FileNotFoundError`): catches the validation failure from `__post_init__` specifically, prints the error plus an actionable hint to run `train_model.py` if the model file is what's missing.
  - Lines 29-33 (`except Exception`): a catch-all for anything unexpected, printing the error and a full traceback via `traceback.print_exc()` so a crash is still debuggable rather than silently swallowed.
  - Lines 35-36: prints a goodbye banner (runs after `menu.run()` returns, i.e., after the user quits from the menu).
- **Lines 45-46**: standard `if __name__ == "__main__": main()` guard.

**Design rationale:** This file is deliberately thin — it exists only to compose `AppConfig` + `MenuWindow` and handle startup errors gracefully (see [2.2](#22-dependency-injection-via-appconfig) for why config validation happens centrally and fails fast here rather than deep inside a UI component).

---

## `src/core/config.py`

**Purpose:** Defines `AppConfig`, the single dependency-injected source of truth for every tunable constant, file path, and the gesture→action mapping used throughout the app.

**Class `AppConfig`** (`@dataclass`, lines 7+):

- **Path fields (`field(default_factory=...)`)**: `PROJECT_ROOT` (computed from `__file__`'s grandparent directory) and `SCREENSHOT_SAVE_DIR` (`~/Pictures/GestureScreenshots`). Both use `default_factory` rather than a plain default because `Path` objects, like all objects, would otherwise be evaluated once at class-definition time and could in principle be shared/mutated across instances — see [2.2](#22-dependency-injection-via-appconfig) for the general mutable-default rationale (most directly relevant to the dict field below).
- **`MODEL_PATH` / `GESTURE_MAP_PATH` (`@property`)**: computed from `PROJECT_ROOT` on every access rather than stored as plain fields, so they automatically stay in sync if `PROJECT_ROOT` is ever overridden — there's no way for them to drift out of sync with the root the way two independently-set fields could.
- **Camera/detection constants**: `CAMERA_INDEX=1` (note: not `0` — the app expects an external/secondary camera by default), `CAMERA_WIDTH=1280`, `CAMERA_HEIGHT=720`, `CAMERA_FPS=30`, `MIN_DETECTION_CONFIDENCE=0.7`, `MIN_TRACKING_CONFIDENCE=0.7` (passed straight through to MediaPipe's `Hands(...)` constructor in `GestureDetector`).
- **Recognition tuning**: `SMOOTHING_HISTORY_SIZE=5` (the majority-vote window — see [2.5](#25-temporal-smoothing-majority-vote)), `NORMALIZATION_THRESHOLD=0.05` (the minimum-scale guard — see [2.4](#24-hybrid-scale-normalization-strategy)).
- **Action tuning**: `ACTION_DWELL_FRAMES=5` (see [2.6](#26-dwell-based-action-triggering)), `CLOSE_APP_HOLD_DURATION=5.0`, `MINIMIZE_APP_HOLD_DURATION=1.5`, `CURSOR_SENSITIVITY=1.5`, `CURSOR_SMOOTHING=0` (despite the "0=none, 1=full" intent documented in the field, a value of `0` currently means cursor smoothing is fully **disabled** — the cursor follows raw per-frame input with no lag-smoothing, per the formula used in `ActionExecutor.update_continuous_control`), `VOLUME_INCREMENT_PERCENT=5.0`, `VOLUME_INCREMENT_INTERVAL=0.5`, `VOLUME_SMOOTHING_FRAMES=3`.
- **Display/perf**: `PROCESSING_FPS_LIMIT=30`, `TESTING_MODE_SIZE=(1280, 720)`, `COMPACT_MODE_SIZE=(426, 240)`.
- **`GESTURE_ACTIONS` (`dict`, `default_factory`)**: maps recognized gesture names to action identifiers — `thumbs_up_sideways`→`volume_up`, `thumbs_down_sideways`→`volume_down`, `thumbs_up`→`scroll_up`, `thumbs_down`→`scroll_down`, `fist`→`minimize_app`, `index_middle`→`drag_control`, `open_palm`→`close_app`, `ok`→`toggle_pause_play`, `thumb_left`→`previous_track`, `thumb_right`→`next_track`, `pinch`→`screenshot`. Note that `point` and `null` have **no entry** here — `point` is special-cased directly inside `ActionExecutor.update()` (checked by name, not looked up in this dict) to activate continuous cursor control, and `null` represents the idle/no-gesture state.
- **Color constants**: `COLOR_GREEN`/`GRAY`/`LIGHT_GRAY`/`RED`/`YELLOW`/`WHITE`/`BLACK`, all BGR tuples (OpenCV's native channel order) used by `ui_utils.py`'s drawing functions.
- **`__post_init__(self)`**: dataclass-provided hook that runs automatically right after the generated `__init__` finishes (see [2.2](#22-dependency-injection-via-appconfig)). Validates, in order: `PROJECT_ROOT` exists (`ValueError` if not), `GESTURE_MAP_PATH` exists (`FileNotFoundError` if not), `MODEL_PATH` exists (`FileNotFoundError` if not) — this is what `app.py`'s `except FileNotFoundError` block is built to catch.

**Design rationale:** See [2.2](#22-dependency-injection-via-appconfig) for the full rationale on centralizing config as an injected dataclass.

**Known discrepancy worth being able to explain:** `MINIMIZE_APP_HOLD_DURATION` here is `1.5`, but `ActionExecutor.__init__`'s own constructor default (`action_executor.py` line 112) is `2.5`. This isn't a bug in practice — `CompactModeWindow.__init__` (lines 65-77) explicitly overrides `action_executor.minimize_app_hold_duration = config.MINIMIZE_APP_HOLD_DURATION` immediately after construction, so the effective runtime value is always `1.5`; the `2.5` in `ActionExecutor`'s constructor is a stale default that's never actually used by the live app (only relevant if `ActionExecutor` were ever instantiated without a config override).

---

## `src/core/gesture_detector.py`

**Purpose:** Thin wrapper around MediaPipe Hands — owns the MediaPipe pipeline object, converts BGR frames to RGB and runs detection, and draws landmarks back onto a frame.

**Class `GestureDetector`** (docstring lines 10-19):

- **`__init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7)` (lines 21-38)**: stores references to `mp.solutions.hands` and `mp.solutions.drawing_utils`. Constructs `self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=..., min_tracking_confidence=...)`. `static_image_mode=False` enables MediaPipe's internal tracking-between-frames optimization (faster, temporally consistent, appropriate for video rather than independent images); `max_num_hands=1` is a deliberate single-hand-only constraint matching the rest of the pipeline (normalization, the trained model's feature vector, and the gesture vocabulary are all defined for exactly one hand).
- **`detect_hand(self, frame)` (lines 40-60)**: converts the incoming BGR frame (OpenCV's native format) to RGB via `cv2.cvtColor` (MediaPipe expects RGB), calls `self.hands.process(rgb_frame)`, and returns `results.multi_hand_landmarks[0]` if any hand was found, else `None`. Since `max_num_hands=1`, index `[0]` is always the (only) detected hand.
- **`draw_landmarks(self, frame, hand_landmarks)` (lines 62-87)**: guards against `hand_landmarks is None` (no-op if so), otherwise calls `mp_drawing.draw_landmarks(...)` with custom `DrawingSpec`s — green circles (radius 3, thickness 2) for landmark points, white connecting lines (thickness 2) for the hand skeleton.
- **`cleanup(self)` (lines 89-91)**: calls `self.hands.close()` to release MediaPipe's internal resources.

**Note:** `numpy` is imported (line 7) but never actually used anywhere in this file's body (confirmed: zero `np.` calls) — likely leftover from an earlier version or copy-paste scaffolding.

---

## `src/core/gesture_recognizer.py`

**Purpose:** Loads the trained classifier and gesture-name map, converts a normalized landmark feature vector into a predicted gesture name + confidence, and applies majority-vote temporal smoothing.

**Class `GestureRecognizer`** (docstring lines 11-21):

- **`__init__(self, model_path, gesture_map_path, history_size=5)` (lines 23-45)**: opens and `json.load`s the gesture map; opens and `pickle.load`s the trained model (prints "Loading model from: ..." then "OK Model loaded successfully!" as console feedback); initializes `self.prediction_history = []`, `self.history_size = history_size`, `self.last_confidence = 0.0`.
- **`predict(self, hand_landmarks, normalize_func)` (lines 47-86)**: takes a **dependency-injected normalization function** (decoupling this class from a specific normalization implementation — in practice always `normalize_landmarks` from `utils/normalize.py`) and calls it with `threshold=0.05`. If the result is invalid (scale too small / NaN), returns `("Invalid (scale too small)", 0.0, False)`. Otherwise reshapes the flat 42-element normalized vector to `(1, -1)` (scikit-learn's expected 2D input shape for a single sample), predicts via `self.model.predict(features)`, casts to `int`. Computes confidence via `predict_proba` when available: `class_index = list(self.model.classes_).index(prediction_id)` — this index lookup is necessary (not just a stylistic choice) because `predict_proba`'s output columns are ordered according to `model.classes_`, which is not guaranteed to match the raw integer class IDs in numeric order. Falls back to a flat `1.0` confidence for models without `predict_proba` (e.g. plain k-NN). Maps the integer prediction to a gesture name via `self.gesture_map.get(str(prediction_id), "Unknown")`, updates `self.last_confidence`, returns `(name, confidence, True)`.
- **`predict_smooth(self, hand_landmarks, normalize_func)` (lines 88-108)**: thin wrapper — calls `predict()`, then passes the resulting name through `_smooth_prediction()` before returning.
- **`_smooth_prediction(self, gesture_name)` (lines 110-136)**: appends `gesture_name` to `self.prediction_history`; if the list exceeds `history_size`, removes the oldest entry via `pop(0)`; builds a `dict` counting occurrences of each label in the window; returns the label with `max(counts, key=counts.get)` — i.e., the current majority vote.
- **`get_confidence(self)` / `reset_history(self)` (lines 138-144)**: trivial accessor and history-clearing methods (the latter called whenever the hand disappears, so a stale history doesn't bias the next gesture's first few smoothed predictions).

**Design rationale:** See [2.5](#25-temporal-smoothing-majority-vote) for why smoothing exists and why it's a sliding-window majority vote rather than, e.g., exponential averaging of probabilities.

**Note:** `numpy` is imported (line 7) but never used in this file's body, and `Path` (line 8) is also unused — both paths arrive already constructed from `AppConfig`'s `@property` accessors, so this file never needs to build a `Path` itself.

---

## `src/core/action_executor.py`

**Purpose:** The largest and most complex file in the codebase (1186 lines). Owns the dwell-based gesture-to-action state machine and every concrete action handler — discrete one-shot actions (clicks, screenshots, media keys) and continuous per-frame actions (cursor movement, drag, volume, scroll, swipe-driven black-screen toggle, close/minimize hold-timers).

**Class `ActionExecutor`** (docstring lines 22-31):

### Imports and setup (lines 1-19)
Standard library (`time`, `os`, `datetime`, `pathlib.Path`), `pycaw.pycaw.AudioUtilities` (Windows audio COM interface), `pyautogui`. `pynput` imports are wrapped in a `try/except` that sets a `PYNPUT_AVAILABLE` flag — a graceful-degradation pattern so the app doesn't hard-crash on import if `pynput` isn't installed; methods that need it (drag, media keys) check this flag before using it.

### `__init__(self, gesture_actions, min_dwell_frames=5)` (lines 33-139)
Initializes state in clearly labeled blocks, mirroring the action categories described in [2.8](#28-discrete-vs-continuous-action-execution-design):
- **Discrete trigger state** (44-47): `current_gesture`, `frame_count`, `triggered_this_gesture` — the core dwell bookkeeping.
- **UI feedback** (49-51): `last_action`, `action_display_frames` (a countdown used by `ui_utils.draw_action_feedback` to show a fading "action fired" message).
- **Continuous cursor-control state** (53-61): origin cursor position, origin hand position, smoothed-cursor accumulator.
- **Proximity click state** (63-69, `proximity_threshold=0.1`) and **proximity double-click state** (71-77, `proximity_double_click_threshold=0.08`): dwell-then-release tracking for the thumb-ring and pinky-ring gestures (see `update_continuous_control` below).
- **Volume control state** (79-87): `volume_increment_interval=0.5`, `volume_increment_percent=5.0`, `volume_smoothing_frames=3`.
- **Screenshot directory placeholder** (90), later set by `CompactModeWindow` from config.
- **Drag control state** (92-101); **close-app hold state** (103-107, `close_app_hold_duration=5.0` default, plus the `should_close` flag the UI loop polls); **minimize-app hold state** (109-112, `minimize_app_hold_duration=2.5` default — overridden at runtime, see the config discrepancy note above); **scroll control state** (114-120, `scroll_increment_interval=0.1`, faster than volume's 0.5s by design — scrolling feels better with shorter repeat intervals); **swipe detection state** (122-128, `swipe_threshold=0.5` i.e. 50% of screen width, `swipe_vertical_tolerance=0.1`).
- Lines 130-136: `pynput` mouse/keyboard controller instances (only if available).
- Lines 138-139: calls `_initialize_volume_control()`.

### `_initialize_volume_control(self)` (lines 141-161)
Wraps `AudioUtilities.GetSpeakers().EndpointVolume` acquisition in a `try/except`; on failure sets `self.volume_interface = None` rather than crashing — the same graceful-degradation philosophy as the `pynput` import guard, so a system without a usable audio endpoint doesn't prevent the rest of the app from working.

### `update(self, detected_gesture, hand_landmarks=None)` (lines 163-273) — the core dwell state machine
This is the method called once per frame from the UI layer with the current smoothed prediction.
- **Lines 175-194**: if the gesture is non-actionable (`null`, no hand, or an invalid/unrecognized state), every piece of active state is reset (continuous control deactivated, dwell counters cleared) — this is the "nothing is happening" branch.
- **Lines 196-237 (same gesture as last frame)**: `frame_count` increments. The instant `frame_count == min_dwell_frames` (equality, not `>=`, so this branch runs exactly once per dwell period — see [2.6](#26-dwell-based-action-triggering)) and the gesture hasn't already triggered, it dispatches by gesture/action type:
  - `'point'` → `_activate_continuous_control` (cursor mode).
  - `'index_middle'` → `_activate_drag_control`.
  - a gesture mapped to `volume_up`/`volume_down` → `_activate_volume_control`.
  - a gesture mapped to `scroll_up`/`scroll_down` → `_activate_scroll_control`.
  - a gesture mapped to `close_app` (i.e. `open_palm`) → activates **both** `_activate_swipe_detection` and `_activate_close_app` simultaneously (lines 223-229) — a deliberate race between "swipe to toggle black screen" and "hold 5 seconds to close," whichever completes first wins and deactivates the other.
  - a gesture mapped to `minimize_app` (i.e. `fist`) → `_activate_minimize_app`.
  - anything else → returns the gesture name itself, signaling the caller to fire a one-shot discrete action via `execute_action`.
- **Lines 238-273 (gesture changed from last frame)**: deactivates whatever continuous subsystem was previously active and resets dwell state, since the new gesture needs its own fresh dwell period.

### `execute_action(self, gesture)` (lines 275-311)
Looks up `gesture_actions.get(gesture)` then dispatches via `if`/`elif` to one of ten handlers: `left_click`, `right_click`, `scroll_up`, `scroll_down`, `volume_up`, `volume_down`, `close_app`, `toggle_pause_play`, `previous_track`, `next_track`, `screenshot`. **Note**: `_execute_double_click` (defined line 320) and `_execute_minimize_app` (defined line 978) are deliberately *not* in this dispatch table — they're invoked from other internal code paths instead (double-click from the pinky-ring proximity logic inside `update_continuous_control`, line 714; minimize from the hold-timer check inside `update_minimize_app`, line 976), since both are triggered by their own independent mechanisms rather than a single-frame discrete dwell trigger.

### Discrete action handlers (lines 313-504)
- `_execute_left_click` / `_execute_double_click` / `_execute_right_click` (313-332): thin wrappers around `pyautogui.click()` / `doubleClick()` / `click(button='right')`, each setting `last_action` and a 30-frame UI display timer.
- `_execute_screenshot` (334-357): ensures `self.screenshot_save_dir` exists via `Path(...).mkdir(parents=True, exist_ok=True)`, builds a timestamped filename (`screenshot_YYYYMMDD_HHMMSS.png`), saves via `pyautogui.screenshot().save(...)`.
- `_execute_scroll_up` / `_execute_scroll_down` (359-371): `pyautogui.scroll(±100)`.
- `_execute_volume_up` / `_execute_volume_down` (373-431): reads current volume via `GetMasterVolumeLevelScalar()`, adds/subtracts `volume_increment_percent / 100`, clamps to `[0.0, 1.0]` via `min`/`max`, writes back via `SetMasterVolumeLevelScalar(new_volume, None)`; the COM calls are wrapped in `try/except` so a transient audio-API failure doesn't crash the frame loop.
- `_execute_toggle_pause_play` / `_execute_previous_track` / `_execute_next_track` (433-504): `pynput.keyboard` press+release of `Key.media_play_pause` / `media_previous` / `media_next` — OS-level media keys that work regardless of which window currently has focus.
- `get_current_volume_percent` (506-521): public accessor used by `CompactModeWindow`'s cached volume display.

### Continuous control activation/update/deactivation
- `_activate_continuous_control` (523-562): captures the current `pyautogui.position()` as the cursor's origin and the index-fingertip landmark (8) as the hand's origin; resets proximity click/double-click debounce state so a stale dwell from a previous activation can't carry over.
- `_activate_drag_control` (564-596): guarded by `PYNPUT_AVAILABLE`; immediately presses the left mouse button down (`mouse_controller.press(PynputButton.left)`); captures drag origins from landmark 8.
- **`update_continuous_control(self, hand_landmarks, sensitivity=1.5, smoothing=0.3, screen_width=1920, screen_height=1080)` (lines 598-730)**: the cursor-movement math — computes the delta of landmark 8 from its captured origin, scales by `screen_dimension × sensitivity` to get a pixel delta, adds that to the origin cursor position, applies exponential smoothing (`smoothed = last*smoothing + new*(1-smoothing)`), clamps to screen bounds, and calls `pyautogui.moveTo(int(x), int(y), duration=0)`. Lines 644-686 then layer in **proximity-click** logic: computes the thumb-tip(4)-to-ring-DIP(15) distance via `_calculate_thumb_ring_distance`; entering the `0.1` threshold starts a dwell counter; the click only fires on *exiting* the proximity zone after the dwell requirement was met (not on entry) — this prevents a fast pass-through pinch from accidentally registering a click, since the user must deliberately hold the pinch before releasing it. Lines 688-730 mirror this exact pattern for pinky-tip(20)-to-ring-tip(16) at a tighter `0.08` threshold, firing `_execute_double_click` on qualifying exit.
- `update_drag_control` (732-767): the same delta/sensitivity/smoothing cursor math as above, but tracked against a separate drag origin; intentionally has no proximity-click logic layered on top (drag and point/click are mutually exclusive activation states).
- `_deactivate_continuous_control` (769-795) / `_deactivate_drag_control` (797-821): teardown of the respective state. Drag deactivation explicitly calls `mouse_controller.release(PynputButton.left)` — important, since omitting this would leave the OS believing the mouse button is still held down after the gesture ends.

### Volume / scroll / swipe / close / minimize
- `_activate_volume_control` / `_deactivate_volume_control` (823-856) and `_activate_scroll_control` / `_deactivate_scroll_control` (858-887): each fires one immediate action on activation; subsequent repeats are handled by the corresponding `update_*` method.
- `_activate_swipe_detection` (889-903): records the middle-finger-MCP landmark (9) starting x/y.
- `_activate_close_app` / `_deactivate_close_app` (905-919), `_deactivate_swipe_detection` (920-928).
- `update_close_app` (930-940): per-frame poll; once `elapsed >= close_app_hold_duration`, fires `_execute_close_app`.
- `_execute_close_app` (942-949): deactivates the swipe detector (it lost the activation race) and sets `self.should_close = True`. This flag is read by `CompactModeWindow.run()`'s loop to break out and return to the menu — despite the "close app" naming, this **exits Compact Mode back to the menu, not the whole process**.
- `_activate_minimize_app` / `_deactivate_minimize_app` / `update_minimize_app` (951-976): same hold-timer pattern as close-app, at `minimize_app_hold_duration`.
- `_execute_minimize_app` (978-987): uses raw `ctypes.windll.user32.GetForegroundWindow()` then `ShowWindow(hwnd, 6)` (`SW_MINIMIZE`) — minimizes whichever window currently has OS focus (not necessarily the gesture app's own window), and is inherently Windows-only.
- `update_volume_control` (989-1019) / `update_scroll_control` (1020-1049): per-frame repeat logic — a `smoothing_counter` delays the very first repeat by `smoothing_frames` frames (avoids an immediate double-fire right after activation), then subsequent repeats are throttled by wall-clock `time.time()` interval checks (0.5s for volume, 0.1s for scroll).
- **`update_swipe_detection(self, hand_landmarks, screen_width)` (lines 1051-1094)**: tracks the horizontal/vertical delta of landmark 9 from its captured start position. If vertical drift exceeds `swipe_vertical_tolerance` (0.1), the start position is reset (a course-correction, not a cancellation — the user can angle their hand slightly without losing swipe progress). Once horizontal delta reaches `swipe_threshold` (50% of screen width), the swipe fires: deactivates both itself and the close-app timer (winning the race), and returns `True`. The caller, `CompactModeWindow.run()`, uses that `True` return to invoke `_toggle_black_screen()` — i.e., the actual end-to-end behavior of a successful swipe-while-holding-open-palm is a black-screen toggle, not closing the app.

### Helpers and reset
- `_calculate_thumb_ring_distance` / `_calculate_pinky_ring_distance` (1096-1144): plain Euclidean distance, using a locally-scoped `import math` inside each method rather than a module-level import (a minor style inconsistency, not a functional issue).
- `reset(self)` (1146-1176): the "panic button" — tears down every active subsystem (continuous control, drag, volume, scroll, swipe, close, minimize) in one call. Invoked whenever the gesture becomes invalid or the hand disappears, guaranteeing no subsystem is left silently active with stale state.
- `get_last_action()` / `decrement_display_frames()` (1178-1186): trivial UI-feedback accessors.

**Design rationale:** See [2.6](#26-dwell-based-action-triggering) for the dwell-trigger rationale and [2.8](#28-discrete-vs-continuous-action-execution-design) for why discrete and continuous actions are split into separate code paths and why three different libraries (`pyautogui`/`pynput`/`pycaw`) are each used for their respective responsibility.

---

## `src/utils/normalize.py`

**Purpose:** Pure-function landmark normalization math — no MediaPipe/OpenCV/tkinter dependency, just NumPy. Converts raw MediaPipe landmarks into a translation- and scale-invariant feature vector suitable for both training data collection and live inference.

- **`get_landmark_array(hand_landmarks) -> np.ndarray` (lines 10-23)**: iterates `hand_landmarks.landmark`, builds a list of `[x, y]` pairs, returns as a `(21, 2)` NumPy array. Only `x`/`y` are extracted — **MediaPipe's `z` depth estimate is never used anywhere in this pipeline**, making the entire recognition system effectively 2D.
- **`compute_scale_wrist_mcp(landmarks) -> float` (lines 26-39)**: `np.linalg.norm(landmarks[9] - landmarks[0])` — Euclidean distance between wrist (0) and middle-finger MCP (9).
- **`compute_scale_palm_width(landmarks) -> float` (lines 42-55)**: `np.linalg.norm(landmarks[17] - landmarks[5])` — Euclidean distance between index MCP (5) and pinky MCP (17).
- **`normalize_landmarks(hand_landmarks, threshold=0.05) -> Optional[Dict]` (lines 58-124)**:
  - Lines 82-83: centers all landmarks by subtracting the wrist (translation invariance).
  - Lines 86-87: computes both scale measures.
  - Line 90: `scale_used = max(scale_wrist_mcp, scale_palm_width)` — the hybrid scale, see [2.4](#24-hybrid-scale-normalization-strategy).
  - Lines 93-100: if `scale_used < threshold`, returns immediately with `normalized=None, valid=False` (plus both raw scale values, useful for diagnostics even on failure).
  - Line 103: `normalized = centered / scale_used` — the actual scale-invariance division.
  - Lines 105-106: `normalized.flatten()` — produces the interleaved feature order `[x0, y0, x1, y1, ..., x20, y20]`. This exact ordering is an **implicit contract**: `collect_data.py` and `train_model.py` both build their CSV columns as `x0, y0, x1, y1, ...` to match, but nothing in the type system enforces this — if the flatten order here ever changed, every downstream CSV column and the trained model's feature ordering would silently desynchronize.
  - Lines 109-116: a second safety net — explicit `np.isfinite` check on the flattened vector, catching any residual NaN/Inf the scale guard alone wouldn't (e.g., from degenerate landmark geometry even when the scale itself is above threshold).
  - Lines 118-124: success path, returns the full dict with `valid=True`.
- **`get_average_confidence(hand_landmarks) -> float` (lines 127-141)**: an explicit placeholder — every per-landmark "confidence" value is hardcoded to `1.0`, and `np.mean` of an all-`1.0` list is just `1.0`. It's imported into `collect_data.py` but, based on a full read of that file, is never actually invoked there — it appears to be unused scaffolding for a feature (per-landmark confidence weighting) that was never completed.

**Design rationale:** See [2.4](#24-hybrid-scale-normalization-strategy) for the full reasoning behind the translation/hybrid-scale/threshold/finiteness design.

---

## `src/utils/fps_counter.py`

**Purpose:** A minimal frames-per-second counter for on-screen performance display.

**Class `FPSCounter`**:
- `__init__(self)` (lines 20-23): `self.last_time = time.time()`, `self.fps = 0.0`.
- `update(self)` (lines 25-34): computes `delta_time = current_time - self.last_time`; only recomputes `self.fps = 1.0 / delta_time` if `delta_time > 0` (guards division by zero on an impossibly-fast or duplicate-timestamp call); always updates `self.last_time` regardless.
- `get_fps(self)` (lines 36-38): returns the last computed value.

No design-rationale callout needed — this is a standard, self-contained utility with no architectural tradeoffs.

---

## `src/ui/ui_utils.py`

**Purpose:** A library of stateless drawing functions shared by both `TestingModeWindow` and `CompactModeWindow`, each taking a `frame` (and usually `config`) and drawing onto it in place via OpenCV calls. Centralizing these avoids duplicating drawing code between the two mode windows.

Eight module-level functions:
- **`get_gesture_color(status, config)` (lines 9-27)**: maps a status string (e.g. valid/invalid/no-hand) to a BGR color from `config`'s color constants, used to color-code UI elements by recognition state.
- **`draw_semi_transparent_panel(frame, ...)` (lines 30-43)**: draws an alpha-blended rectangle overlay via `cv2.addWeighted`, used as the background behind text panels in both modes.
- **`draw_gesture_label(frame, gesture, config, ...)` (lines 46-69)**: renders the current gesture name as large text.
- **`draw_confidence(frame, confidence, config, ...)` (lines 72-93)**: renders the confidence percentage.
- **`draw_fps_counter(frame, fps, config, ...)` (lines 96-120)**: right-aligns FPS text using `cv2.getTextSize` to compute the text width first.
- **`draw_action_feedback(frame, action_executor, config, ...)` (lines 123-147)**: renders a pulsing/fading "action fired" message, driven by `action_executor.action_display_frames` counting down.
- **`draw_scale_diagnostics(frame, normalization_result, config, ...)` (lines 150-175)**: renders the raw scale values and valid/invalid status — primarily a Testing Mode diagnostic.
- **`draw_instructions(frame, config, ...)` (lines 178-197)**: renders static on-screen key/usage hints.

No single design-rationale callout — this module is a straightforward presentation-layer utility library; its existence as a *separate, shared* module (rather than duplicated per-mode) is covered by [2.1](#21-layered-separation-of-concerns).

---

## `src/ui/menu_window.py`

**Purpose:** The application's main menu — a small tkinter window offering Testing Mode, Compact Mode, and Quit.

**Class `MenuWindow`**:
- `__init__(self, config)` (lines 19-32): stores `config`, creates `tk.Tk()`, sets title, fixes geometry to `450x350` and disables resizing, calls `_create_widgets()`.
- `_create_widgets(self)` (lines 34-120): builds a title label, a subtitle label, and a button frame containing three buttons — **Testing Mode** (green, calls `launch_testing_mode`) and **Compact Mode** (blue, calls `launch_compact_mode`), each with a small descriptive sub-label underneath explaining what the mode does, plus a **Quit** button (red, calls `quit_application`).
- **`launch_testing_mode(self)` (lines 122-136)**: prints a status line, calls `self.root.withdraw()` to hide (not destroy) the menu, locally imports `TestingModeWindow` (a local import specifically to avoid a circular import between `menu_window.py` and `testing_mode_window.py`), instantiates it with `self.config`, calls `.run()` (blocks until the user exits that mode), then calls `self.root.deiconify()` to re-show the menu.
- **`launch_compact_mode(self)` (lines 138-152)**: identical pattern, importing and running `CompactModeWindow` instead.
- `quit_application(self)` (lines 154-157): `self.root.destroy()`.
- `run(self)` (lines 159-161): `self.root.mainloop()`.

**Design rationale:** See [2.3](#23-two-operational-modes-sharing-one-core) for why `withdraw()`/`deiconify()` is used instead of destroying and recreating the tkinter root each time a mode is entered/exited.

---

## `src/ui/testing_mode_window.py`

**Purpose:** A full-size diagnostic window — shows live landmarks, normalization diagnostics, predictions, and confidence, but never calls `ActionExecutor`. The safe-by-default mode (see [2.3](#23-two-operational-modes-sharing-one-core)).

**Class `TestingModeWindow`**:
- `__init__(self, config)`: builds `GestureDetector`, `GestureRecognizer`, `FPSCounter`; opens `cv2.VideoCapture(config.CAMERA_INDEX)` with width/height/fps set from config; initializes display state — `current_gesture`, `current_confidence`, `smoothing_enabled=True`, `last_normalization_result=None`; sets the window title "Testing Mode - Gesture Recognition (No Actions)".
- **`run(self)` (lines 67-147)**: creates a resizable `cv2.namedWindow` sized to `config.TESTING_MODE_SIZE`. Main loop: capture frame → horizontal flip (mirror view) → `detector.detect_hand(frame)`. If a hand is found: computes `normalize_landmarks` purely for the diagnostic display (scale values, valid/invalid), then predicts via `predict_smooth` or plain `predict` depending on the `smoothing_enabled` toggle, updating `current_gesture`/`current_confidence`. If no hand: resets display state and calls `recognizer.reset_history()` (so a stale smoothing window doesn't bias the next detection). Updates the FPS counter, calls `_render_frame`, shows via `cv2.imshow`. Key handling uses an FPS-limited `cv2.waitKey` (`wait_time = max(1, int(1000 / config.PROCESSING_FPS_LIMIT))`); `q`/ESC breaks the loop, `s` toggles `smoothing_enabled` and resets history (so toggling mid-gesture doesn't mix raw and smoothed history).
- **`_render_frame(self, ...)` (lines 149-225)**: draws landmarks, a semi-transparent info panel (height 180), the gesture label (font scale 1.2 — large, since this window has room), confidence, scale diagnostics with valid/skipped status, FPS, a smoothing ON/OFF indicator, and instruction text.
- **`cleanup(self)` (lines 227-235)**: releases the camera, destroys OpenCV windows, calls `detector.cleanup()`.

**Design rationale:** See [2.3](#23-two-operational-modes-sharing-one-core) for why this mode exists separately from Compact Mode and never executes actions, and [2.5](#25-temporal-smoothing-majority-vote) for why the smoothing toggle exists here specifically (and not in Compact Mode).

---

## `src/ui/compact_mode_window.py`

**Purpose:** The "real" operational mode — a small, always-on-top window that drives live OS-level actions from recognized gestures via `ActionExecutor`. The largest UI file (635 lines).

**Module-level setup (line 9)**: `pyautogui.PAUSE = 0` — disables `pyautogui`'s default 0.1s pause after every call, which is essential for responsive continuous cursor movement (at the default pause, moving the cursor every frame would be capped to ~10 moves/second regardless of camera FPS).

**Class `CompactModeWindow`**:
- **`__init__(self, config)` (lines 33-104)**: builds `GestureDetector`, `GestureRecognizer`, and `ActionExecutor(gesture_actions=config.GESTURE_ACTIONS, min_dwell_frames=config.ACTION_DWELL_FRAMES)`. Lines 65-77 then explicitly override several `action_executor` attributes from `config` post-construction — `volume_increment_interval`, `volume_increment_percent`, `volume_smoothing_frames`, `screenshot_save_dir`, `close_app_hold_duration`, `minimize_app_hold_duration` (this last line is what resolves the `1.5` vs `2.5` minimize-duration discrepancy noted in the `config.py` section). Also sets up an `FPSCounter`, volume-display caching state (`_cached_volume=None`, `_frame_index=0`), `self.screen_width, self.screen_height = pyautogui.size()`, camera capture, and loads the persisted black-screen setting via `_load_black_screen_setting()`.
- **`_on_mouse_event(self, event, x, y, ...)` (lines 106-134)**: tracks hover/click state for two on-screen buttons — Settings ("S") and open-screenshot-folder ("F") — via rectangle hit-testing against stored button bounds; `cv2.EVENT_LBUTTONDOWN` inside a button's bounds calls `_open_display_settings()` or `_open_screenshot_folder()`.
- **`run(self) (lines 136-282)`**: creates a `cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO` window sized to `config.COMPACT_MODE_SIZE`, registers the mouse callback, and attempts (in a bare `try/except`, since this can fail on some platforms/window managers) `cv2.setWindowProperty(..., cv2.WND_PROP_TOPMOST, 1)` for always-on-top behavior. Main loop, per frame:
  - `frame_start = time.perf_counter()` for frame-budget timing.
  - Capture → flip → **downsample to 640×360 for detection only** (landmarks are normalized 0-1, so a prediction made on the downsampled frame still maps correctly onto the full-resolution display — this is explicitly noted in a comment at lines 176-177, and exists purely as a performance optimization, trading detection-input resolution for speed since landmark detection doesn't need full camera resolution to be accurate).
  - If `black_screen_enabled`, the frame is replaced with `np.zeros_like(frame)` **after** detection but **before** rendering — gesture recognition keeps working even though nothing is visually displayed.
  - Predicts via `predict_smooth`. If valid: calls `action_executor.update(gesture, hand_landmarks)`; if a discrete trigger name was returned, calls `execute_action`; then calls **every** relevant per-frame `update_*` method unconditionally every frame (`update_continuous_control`, `update_drag_control`, `update_volume_control`, `update_scroll_control`, `update_close_app`, `update_minimize_app`, `update_swipe_detection`) — each is a no-op internally if its subsystem isn't currently active, which keeps this call site simple (no need to track "which subsystem is active" here, since `ActionExecutor` already tracks that internally). If the swipe update returns `True`, toggles the black screen. If invalid or no hand: resets local state and calls `action_executor.reset()`.
  - Updates FPS; refreshes the cached volume display only every 12 frames (`self._frame_index % 12 == 0`) rather than every frame, since polling the COM audio interface every single frame would be wasteful.
  - Checks `cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)` and **only renders/shows the frame when the window is actually visible** — skipping rendering overhead while minimized — but `cv2.waitKey` still runs every iteration regardless, so gesture recognition and action execution continue working even while the window is minimized.
  - Computes a frame-budget-aware wait time (`frame_budget_ms - elapsed_ms`, not a flat constant), so the loop self-paces toward the configured FPS limit accounting for how long the frame's actual processing took.
  - Breaks on `q`/ESC or when `action_executor.should_close` is set (the close-app hold-timer firing).
- **`_render_frame(self, ...)` (lines 284-518)**: the compact UI — landmarks, a small semi-transparent panel (height 80), gesture label (font scale 0.6, smaller than Testing Mode's 1.2 to fit the compact window), confidence, action feedback, **countdown progress bars** for close-app (red) and minimize-app (orange) hold timers drawn as growing rectangles along the bottom edge, a drag indicator, thumb-ring/pinky-ring proximity readouts with READY/EXTENDED color-coded status text, a volume indicator color-coded by level (orange >75%, green >25%, yellow ≤25%), an always-visible general volume readout in the bottom-right, and the circular Settings ("S") / Screenshot-folder ("F") buttons with hover-state coloring.
- **`_open_display_settings(self)` (lines 520-528)**: `subprocess.Popen(['start', 'ms-settings:display'], shell=True)` — Windows-only, opens the OS display settings panel.
- **`_open_screenshot_folder(self)` (lines 530-542)**: creates the screenshot folder if it doesn't exist, then `subprocess.Popen(['explorer', str(folder_path)])`.
- **`_load_black_screen_setting(self)` / `_save_black_screen_setting(self)` (lines 544-605)**: JSON read/write against `data/app_settings.json`; both gracefully fall back to (and auto-recreate the file with) the default `False` if the file is missing or corrupted, rather than crashing.
- **`_toggle_black_screen(self)` (lines 607-624)**: flips the state, persists it, and sets `action_executor.last_action`/`action_display_frames` (45 frames, roughly 1.5 seconds at 30fps) so the toggle gets the same on-screen feedback treatment as a regular discrete action.
- **`cleanup(self)` (lines 626-634)**: releases the camera, destroys OpenCV windows, calls `detector.cleanup()`.

**Design rationale:** See [2.3](#23-two-operational-modes-sharing-one-core) for why this mode exists as a separate, action-executing counterpart to Testing Mode, and [2.9](#29-persistent-settings) for why the black-screen setting is persisted to JSON.

---

## `src/collect_data.py`

**Purpose:** A standalone tkinter + OpenCV tool for collecting labeled training samples — captures normalized landmark vectors and appends them to `data/gestures_data.csv`.

**Note on import style**: line 18 uses a **relative import** — `from .utils.normalize import normalize_landmarks, get_average_confidence` — which means this script can only be correctly executed as a module, `python -m src.collect_data`, **not** as `python src/collect_data.py` (a relative import fails with "attempted relative import with no known parent package" if the file is run as a top-level script). This is worth being aware of if the README's literal run instructions for this script don't match how it actually needs to be invoked.

**Class `GestureDataCollector`**:
- **`__init__(self)` (lines 22-65)**: loads the gesture map, builds a `gesture_names` list sorted by numeric ID, initializes a `sample_counts` dict, loads `existing_counts` from the main CSV via `load_existing_counts()`, sets up auto-capture state (`auto_capture_interval=0.2`, i.e. up to 5 samples/second when auto-capture is active) and status-feedback state. Builds its **own** raw MediaPipe `Hands` instance directly here, rather than reusing `core.GestureDetector` — this script predates or deliberately bypasses the `core/` abstraction, since it's a standalone offline tool, not part of the live recognition pipeline. Opens the webcam at 960×540, then calls `setup_gui()`.
- `load_gesture_map(self)` (lines 67-71): loads `data/gesture_map.json`.
- `load_existing_counts(self)` (lines 73-94): reads `data/gestures_data.csv` via pandas (if it exists) and counts existing samples per gesture, so the collector can show cumulative totals, not just the current session's count.
- **`setup_gui(self)` (lines 96-180)**: builds a 450×400 tkinter control panel — title, a `ttk.Combobox` gesture dropdown bound to `on_gesture_change`, a counts-display label, an instructions panel (SPACE = toggle auto-capture, ENTER = single capture, Q/ESC = quit), a status label, and a "Save & Quit" button.
- `on_gesture_change(self, event)` (lines 182-191): updates `current_gesture_id`/`current_gesture_name` from the dropdown selection.
- `get_counts_text(self)` / `update_counts_display(self)` (lines 193-210): formats existing + newly-collected sample counts per gesture for display.
- `update_status(self, message)` (lines 212-215): updates the status label text.
- **`capture_sample(self, hand_landmarks) -> bool` (lines 217-253)**: calls `normalize_landmarks(hand_landmarks, threshold=0.05)`; returns `False` immediately if invalid. Otherwise builds a sample `dict` containing an ISO `timestamp`, `gesture_id`, `gesture_name`, the two raw scale values, `scale_used`, plus the 42 `x{i}`/`y{i}` columns extracted from the flattened normalized array (matching the implicit ordering contract from `normalize.py`). Appends to `self.samples` and increments `sample_counts`.
- **`draw_overlay(self, frame)` (lines 255-363)**: draws MediaPipe landmarks, a color-coded frame border (white=idle, yellow=auto-capture active, green=last capture succeeded, red=last capture failed, fading out after `status_duration=0.5s`), an info panel with the current gesture, per-gesture/session/overall totals, live scale diagnostics with valid/skipped status, and an auto-capture/manual mode indicator.
- **`save_data(self)` (lines 365-399)**: builds a `pd.DataFrame` from `self.samples`, appends it to `data/gestures_data.csv` without a header if the file already exists, or creates it with a header if not; prints a per-gesture breakdown and the new total row count in the file.
- `quit_application(self)` (lines 401-406): stops the capture loop, saves data, destroys the tkinter root.
- **`run(self)` (lines 408-421)**: starts `video_loop` on a daemon thread, runs `self.root.mainloop()` (blocking on the GUI thread), then joins the video thread and releases resources once the GUI closes.
- **`video_loop(self)` (lines 423-495)**: per-frame capture → flip → RGB convert → MediaPipe `process` → auto-capture logic (calls `capture_sample` at the `auto_capture_interval` while active and a hand is present) → `draw_overlay` → `cv2.imshow` (the window is explicitly moved via `cv2.moveWindow('Gesture Data Collector', 450, 50)` so it sits beside the tkinter control panel rather than overlapping it) → key handling: `q`/ESC quits and saves, SPACE toggles auto-capture, ENTER performs one manual capture with status feedback.
- `main()` (lines 498-513): prints usage instructions, instantiates and runs `GestureDataCollector`.

No dedicated design-rationale callout beyond what's already covered for normalization in [2.4](#24-hybrid-scale-normalization-strategy) — this script is primarily a data-collection UI built on top of that shared logic.

---

## `src/train_model.py`

**Purpose:** The full training and evaluation pipeline (821 lines) — loads collected data, trains and rigorously compares Random Forest, k-NN, and SVM, and saves the production model.

**Class `GestureModelTrainer`**:
- **`__init__(self) (lines 37-66)`**: sets up `project_root`, `data_dir`, `raw_dir`, `processed_dir`, and `models_dir` path attributes (note: `raw_dir`/`processed_dir` are defined here but the actual data-loading method below only reads the single consolidated `gestures_data.csv`, not files under `raw/` — `analyze_model.py`, by contrast, *does* read from `data/raw/*.csv`, a known inconsistency between the two scripts' data-source assumptions, noted again in the `analyze_model.py` section below). Loads the gesture map. Initializes data containers (`X`, `y`, train/test splits) and model slots (`rf_model`, `knn_model`, `svm_model`) to `None`, plus `cv_scores={}` and `train_times={}` dicts used later for the statistical comparison.
- `load_and_merge_data(self)` (lines 68-95): reads `data/gestures_data.csv` (raises `FileNotFoundError` if missing — the file `AppConfig.__post_init__` and this script both ultimately depend on existing), prints per-gesture sample counts.
- `prepare_features_and_labels(self)` (lines 97-139): builds the `coordinate_columns` list as the interleaved `x0, y0, x1, y1, ..., x20, y20` order (matching `normalize.py`'s flatten contract), extracts `X`/`y` arrays, checks for and removes any non-finite rows via an `np.isfinite` mask.
- **`split_train_test(self, test_size=0.2, random_state=42)` (lines 141-184)**: `train_test_split(..., stratify=self.y)` — stratification ensures each gesture class is proportionally represented in both splits, important given a moderate, possibly class-imbalanced dataset; prints class-distribution breakdowns for both the train and test sets.
- **`train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2)` (lines 186-268)**: includes an extensive docstring explaining the RF intuition for a thesis audience (ensemble of decision trees, majority vote, robustness to noise). `RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)`. Times training, computes train/test accuracy and the gap between them with a qualitative interpretation message ("Good fit" / "Acceptable fit. Slight overfitting" / "Overfitting detected!" depending on the gap size), runs 5-fold CV via `cross_val_score` (stored in `self.cv_scores['Random Forest']`), and prints the top-10 feature importances mapped back to specific landmark+coordinate names.
- **`train_knn(self, n_neighbors=5)` (lines 270-337)**: similarly documented ("you are the average of your k closest friends" intuition); `KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)`; same train/test/CV reporting pattern; notes in its training-time print that k-NN is "lazy" — the real computational cost happens at inference time (distance computation against the full training set), not at "training" time (which is just storing the data).
- **`train_svm(self, C=10, gamma='scale')` (lines 339-416)**: documents the margin-maximization/RBF-kernel intuition and the `C`/`gamma` tradeoffs; `SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)` — `probability=True` enables Platt-scaled probability estimates (needed for the confidence score `GestureRecognizer.predict` reads via `predict_proba`) at a modest extra training/inference cost; same reporting pattern as the other two.
- `plot_confusion_matrix(self, y_true, y_pred, model_name, save_path, normalize=False)` (lines 418-459): builds a raw or row-normalized confusion matrix via `ConfusionMatrixDisplay`, saves as a PNG.
- `analyze_confusion_matrix(self, cm, model_name)` (lines 461-497): prints per-class accuracy and the top-5 most-confused gesture pairs (sorted by off-diagonal count) — useful for identifying which gestures are visually/geometrically similar enough to be commonly mistaken for each other.
- **`evaluate_all_classifiers(self, predictions)` (lines 499-568)**: for each of the three trained classifiers, prints a full `classification_report` (per-class precision/recall/F1), saves a normalized confusion-matrix PNG, benchmarks inference time (100 passes over the full test set, averaged to a per-sample millisecond figure), and collects `accuracy`, `f1_weighted`, `train_time_s`, `inf_time_ms` into a results dict; prints a timing-summary table across all three.
- `plot_comparison_bar(self, results)` (lines 570-603): a grouped bar chart (accuracy vs. weighted F1 per classifier, with percentage labels via `ax.bar_label`), saved as `classifier_comparison_bar.png`.
- **`plot_learning_curves(self, models)` (lines 605-661)**: for each classifier, `learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring='accuracy', n_jobs=-1)`, plotting train vs. validation accuracy with shaded standard-deviation bands — used to visually argue whether more training data would likely help (still-diverging curves) or whether the model has plateaued (converged curves), saved as `learning_curves.png`.
- **`statistical_significance_tests(self)` (lines 663-696)**: runs a paired t-test (`scipy.stats.ttest_rel`) over every pairwise combination of the three classifiers' 5-fold CV score arrays, at α=0.05, printing a formatted results table and a plain-English interpretation of whether any pair's performance difference is statistically significant — this is what lets a "is RF actually better, or just lucky on this split" question be answered rigorously rather than anecdotally.
- **`save_models(self)` (lines 698-728)**: pickles each of the three trained models with a timestamped filename (e.g. `random_forest_TIMESTAMP.pkl`), then **always** overwrites `gesture_classifier_latest.pkl` specifically with the Random Forest model — hardcoded, with a comment noting RF as the production model, **regardless of which classifier scored highest** in `evaluate_all_classifiers`. See [2.7](#27-model-choice-random-forest-as-production-model) for why.
- **`run_full_pipeline(self)` (lines 730-810)**: orchestrates all the above steps in numbered order end-to-end, prints a final summary table comparing all three classifiers (test accuracy, weighted F1, CV mean±std) and reports which classifier scored highest by test accuracy — purely informational, since (per `save_models`) this does **not** change which model actually gets deployed as "latest." Lists the generated artifact filenames at the end.
- `main()` (lines 813-820): instantiates `GestureModelTrainer` and calls `run_full_pipeline()`.

**Design rationale:** See [2.7](#27-model-choice-random-forest-as-production-model) — this file is the empirical justification for that hardcoded choice, and is the file to point to if asked "how do you know RF was the right call?"

---

## `src/analyze_model.py`

**Purpose:** Post-hoc analysis of the trained production model — feature-importance visualization and a look at the training-data class distribution.

- `load_model(self, model_path)` (lines 17-20): plain `pickle.load`.
- **`plot_feature_importance(self, model, save_path)` (lines 23-78)**: reads `model.feature_importances_` (only meaningful for the RF model, consistent with RF being the production model — see [2.7](#27-model-choice-random-forest-as-production-model)), groups the 42 raw feature importances by landmark (combining each landmark's `x` and `y` importance), sorts descending, plots the top 15 landmarks as a grouped bar chart (x-importance vs. y-importance per landmark), and prints a top-10 textual ranking using a `landmark_names` list (Wrist, Thumb CMC/MCP/IP/Tip, Index MCP/PIP/DIP/Tip, etc.) so the output is human-readable rather than just numeric indices. This is the concrete artifact that supports the "RF is interpretable" argument in [2.7](#27-model-choice-random-forest-as-production-model) — it directly shows *which landmarks the model actually relies on*.
- **`analyze_data_distribution(self)` (lines 80-119)**: reads `data/raw/gestures_*.csv` via `glob` and concatenates them — **this reads from a `data/raw/` directory of per-session files, which is a different data source than the single `data/gestures_data.csv` used by both `collect_data.py` and `train_model.py`**. This is a real structural inconsistency in the codebase worth being able to name if asked: if `data/raw/` isn't kept populated/in sync with the consolidated CSV, this particular analysis could reflect a different (likely smaller or stale) dataset than what the model was actually trained on. Produces a bar chart and a pie chart of per-gesture sample counts, saved as `data_distribution.png`.
- `main()` (lines 122-153): loads the RF model from `models/gesture_classifier_latest.pkl`, runs both analyses, prints a summary of generated files.

---

## `src/main.py` (legacy)

**Purpose:** An early, minimal proof-of-concept script — **not the application's entry point**, despite the filename. Raw MediaPipe usage with no classification at all.

Builds `mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)` (note: `max_num_hands=2` here, unlike the current pipeline's single-hand constraint), captures from the webcam, and for each detected hand simply **prints the raw `x`/`y`/`z` landmark coordinates to the console** — no normalization, no model, no classification, no action execution. Exits on ESC.

This file predates the entire `core/`/`ui/`/`utils/` structure and exists purely as a record of the project's first working step ("can I get MediaPipe to detect a hand and read its landmarks at all"). See [2.10](#210-legacy-code-retained-in-the-repo) for why it's retained rather than deleted. **The real entry point is `src/app.py`.**

---

## `src/realtime_recognition.py` (legacy)

**Purpose:** A more developed, but ultimately superseded, self-contained prototype that duplicates the functionality now split across `core/GestureDetector`, `core/GestureRecognizer`, `core/ActionExecutor`, and the `ui/` mode windows — all in one 435-line file. **Not used by the live application.**

- **Line 21**: `from utils.normalize import normalize_landmarks` — an outdated import path missing the `src.` package prefix the rest of the codebase now uses, confirming this file predates the current package layout and would not import correctly if run from the project root the way the rest of the app is.
- **`GestureActionTrigger` (lines 24-73)**: an earlier version of the dwell-trigger logic that `ActionExecutor.update()` now implements — same conceptual mechanism, kept separate here rather than reusing the current implementation.
- **`RealtimeGestureRecognizer` (lines 76-425)**: a monolithic class combining detection, normalization, prediction, smoothing, action handling, and OpenCV UI rendering all in one place — exactly the kind of duplication that the `core/`/`ui/` split (see [2.1](#21-layered-separation-of-concerns)) was introduced to eliminate. Key methods: `predict_gesture` (129-161), `smooth_prediction` (163-190, a duplicate of `GestureRecognizer._smooth_prediction`'s majority-vote logic), `handle_action` (192-213 — notably, this method's action mapping only handles a gesture called `"L_shape"`, mapping it to a left-click; `"L_shape"` **does not exist** in the current `data/gesture_map.json`'s 13-gesture vocabulary, confirming this logic is stale relative to the current gesture set), `draw_ui` (215-319), `calculate_fps` (321-325), `run` (327-424).
- `main()` (lines 427-434): instantiates and runs `RealtimeGestureRecognizer`.

See [2.10](#210-legacy-code-retained-in-the-repo) for why this file is kept in the repository despite being dead code from the live application's perspective — it's a useful artifact for narrating the project's architectural evolution if asked.

---

## The `__init__.py` files

`src/core/__init__.py`, `src/ui/__init__.py`, and `src/utils/__init__.py` are each a single-line module docstring (`"Core gesture recognition components."`, `"User interface components."`, `"Utility functions."` respectively) marking each directory as a Python package. They contain no other code — no re-exports, no package-level initialization logic.

---

## Appendix: `temp_scripts/`

Two small, ad-hoc scripts at the repository root, outside `src/` and outside the application's actual architecture — not part of the per-file walkthrough above, but worth being aware of if asked about them:

- **`temp_scripts/script.py`**: a 6-line one-off script that reads `data/gestures_data.csv`, drops every row where `gesture_id == 8` (the `thumbs_up` gesture, per `data/gesture_map.json`), and overwrites the CSV. Note: the final `to_csv` call omits `index=False`, so running this script would reintroduce a stray unnamed index column into the CSV on save — a minor bug, but inconsequential since this is a one-off scratch script, not part of the live pipeline.
- **`temp_scripts/fix_csv.py`**: a more instrumented 20-line version of the same filtering operation (prints before/after shape and per-gesture counts), and correctly includes `index=False` on save. Its line 9 comment ("Remove rows with gesture_id = 4") doesn't match what the code on line 10 actually does (filters out `gesture_id != 8`, i.e. removes ID 8, not 4) — a stale/copy-pasted comment, not a functional bug.

Both appear to be one-time data-cleanup utilities used during dataset preparation, not scripts that are part of the regular collect → train → run workflow.
