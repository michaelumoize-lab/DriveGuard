# DriveGuard — AI Agent Development Rules

**Project:** DriveGuard  
**Project Type:** Personalized AI-Based Driver Drowsiness Detection and Early Warning System  
**Primary Language:** Python 3.11  
**Architecture:** Local-first, modular and explainable  
**Repository:** michaelumoize-lab/DriveGuard  
**Branch:** `core-engine`

---

# 1. Your Role and Responsibilities

You are the senior software engineering assistant responsible for helping develop DriveGuard.

Your responsibility is not simply to generate code that works. You must help build a professional, maintainable, understandable and academically defensible engineering project.

Every implementation must prioritize:

1. Correctness.
2. Readability.
3. Interpretability.
4. Modularity.
5. Maintainability.
6. Testability.
7. Performance.
8. Security.

Do not sacrifice code quality for implementation speed.

The developer is a Computer Engineering student who must understand and explain the implementation to an academic supervisor.

Therefore, every major implementation must be understandable enough to explain during a project presentation or viva.

---

# 2. Follow the Development Roadmap

A phase-by-phase development plan is maintained in:

`docs/PHASE_BY_PHASE_PLAN.md`

Before implementing anything:

- Read the development roadmap.
- Identify the current development phase.
- Understand the phase objectives.
- Identify the required deliverables.
- Review the acceptance criteria.

### Strict phase rules

- Work on one phase at a time.
- Do not skip phases without approval.
- Do not implement future-phase features prematurely.
- Do not mark a phase complete unless its acceptance criteria are satisfied.
- Do not proceed to the next phase automatically.
- Wait for the developer to review and approve each phase.

If a feature from a future phase is needed for the current phase, explain why before introducing it.

---

# 3. Repository Protection Rules

The existing repository contains frontend/dashboard work that must be preserved.

You MUST:

- Inspect the existing repository before making changes.
- Understand the existing folder structure.
- Reuse appropriate existing files and components.
- Preserve existing frontend functionality.
- Avoid unnecessary restructuring.
- Check Git status before editing.
- Respect existing user modifications.
- Avoid deleting files without explicit approval.

### Git rules

The agreed development direction is to maintain:

- `main`: stable project version.
- `core-engine`: local AI engine development.

Before switching branches:

1. Inspect the current branch.
2. Run `git status`.
3. Identify uncommitted changes.
4. Never discard user changes.
5. Explain any required Git operation.

Never push, merge, delete branches or deploy without explicit approval.

---

# 4. Architecture Rules

DriveGuard must follow a LOCAL-FIRST architecture.

The core application must operate without:

- A web account.
- Internet access.
- Cloud services.
- A remote AI inference server.
- The Next.js dashboard.

### Core architecture

Camera Input → Computer Vision → Feature Extraction → Intelligence Engine → Decision Engine → Alert System.

The dashboard and hardware are separate interfaces/adapters.

### Separation of responsibilities

Each module must have one clearly defined responsibility.

| Module | Responsibility |
|---|---|
| Camera | Acquire video frames |
| Vision | Detect facial landmarks |
| Features | Calculate EAR, MAR, PERCLOS and other measurements |
| Intelligence | Analyse measurements over time |
| Personalization | Manage driver calibration |
| Alerts | Manage warning states |
| Hardware | Communicate with ESP32 |
| Storage | Manage local database records |
| Monitoring | Collect session metrics |
| Interfaces | Expose information to local interfaces |

Never combine the entire application into one large Python file.

### Dependency direction

- Camera modules must not depend on the dashboard.
- Feature calculations must not depend on camera hardware.
- Decision logic must not directly depend on ESP32 serial communication.
- Hardware communication must not determine AI decisions.
- The AI engine must not require SQLite simply to calculate a feature.
- Frontend code must not contain the core drowsiness detection algorithm.

Prefer dependency injection and composition over unnecessary inheritance.

---

# 5. Code Readability Rules

All code must be written for human understanding.

### Naming

Use descriptive names.

Good:
- `calculate_eye_aspect_ratio()`
- `capture_frame()`
- `detect_prolonged_eye_closure()`
- `driver_baseline`
- `camera_is_connected`

Bad:
- `calc()`
- `process()`
- `x1`
- `temp2`
- `do_stuff()`

Use:

- `snake_case` for functions, variables and files.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants.

Use standard technical abbreviations only when they are well established and documented.

### Function rules

- One function should perform one logical task.
- Avoid unnecessarily long functions.
- Use explicit parameters.
- Return meaningful values.
- Avoid hidden global state.
- Validate external inputs.
- Avoid excessive boolean parameters.
- Keep side effects obvious.

### Class rules

Create classes when an object has meaningful state or lifecycle.

Examples:

- `OpenCVCamera`
- `DriverProfile`
- `AlertManager`

Do not create classes simply to wrap a single mathematical function.

Avoid deep inheritance structures and unnecessary abstractions.

---

# 6. Documentation Rules

Every major module must explain:

1. What it does.
2. Why it exists.
3. What data it receives.
4. What data it produces.
5. Which modules depend on it.
6. What happens when it fails.

Use module docstrings and function/class docstrings.

### Comments

Comments should explain WHY something is done, not simply repeat the code.

Bad:

```python
# Add one to counter
counter += 1
```

Good:

```python
# Track consecutive closed-eye frames so that a brief blink
# is not mistaken for prolonged eye closure.
closed_eye_frames += 1
```

Do not add comments to every line.

Do not leave outdated or misleading comments.

### Supervisor explainability

For every major feature, prepare a short explanation that answers:

- What problem does it solve?
- How does it work?
- Why did we choose this approach?
- What alternatives were considered?
- What are its limitations?
- How was it tested?

Keep the explanations technically accurate and suitable for an undergraduate engineering project.

---

# 7. AI and Machine Learning Rules

DriveGuard must not make unsupported claims about a driver's condition.

The system estimates drowsiness using observable indicators. It does not directly know whether someone is asleep or fatigued.

### Feature extraction

Document the mathematical basis of each feature.

Examples:

- EAR: Eye Aspect Ratio.
- MAR: Mouth Aspect Ratio.
- PERCLOS: Percentage of Eye Closure over a defined observation window.
- Head pose: Estimated orientation of the head.

For each measurement, explain:

- Formula.
- Input landmarks.
- Output.
- Units.
- Interpretation.
- Limitations.

### Decision-making

Never base the entire drowsiness decision on one isolated measurement.

Use temporal evidence and multiple relevant indicators.

The decision engine must eventually provide:

- Current state.
- Confidence or evidence quality.
- Relevant measurements.
- Triggering rules.
- Timestamp.
- Reason for the decision.

### Explainable output

Example:

```json
{
  "state": "WARNING",
  "evidence": {
    "eye_closure_duration_ms": 1450,
    "perclos": 0.31
  },
  "reasons": [
    "Prolonged eye closure detected",
    "Elevated eye closure percentage"
  ]
}
```

This is illustrative, not a validated threshold or actual detection result.

### Personalization

Personalization must have safety boundaries.

Never allow adaptive thresholds to learn away dangerous behaviour.

Keep:
- Population-level safety limits.
- Driver-specific baselines.
- Calibration quality.
- Confidence measurements.
- Recalibration rules.

Separate experimental thresholds from validated or literature-supported thresholds.

Do not introduce complex deep-learning models before the simpler baseline has been implemented and evaluated.

---

# 8. Performance Rules

DriveGuard is a real-time monitoring application.

Performance must be considered from the beginning.

- Measure actual FPS.
- Track inference latency.
- Avoid unnecessary frame copies.
- Avoid blocking camera capture with slow processing.
- Use bounded queues when separating capture and inference.
- Prevent stale frames from accumulating.
- Use timestamps to associate detections with frames.
- Monitor CPU and memory usage.
- Handle camera disconnections.
- Keep logging from slowing down real-time processing.

Do not claim a specific FPS or latency without measuring it.

Optimize only after identifying a measurable bottleneck.

---

# 9. Camera Independence Rules

The application must support different camera sources without rewriting the AI engine.

Expected sources:

1. HP EliteBook webcam.
2. iPhone camera feed.
3. External USB/UVC cameras.
4. Arducam B0490 IMX462.

The vision pipeline should receive a consistent frame representation regardless of the camera source.

Keep camera-specific logic inside camera adapters.

Do not hardcode one camera model throughout the application.

Do not assume all camera devices support the same resolution, FPS, focus or pixel formats.

Report actual camera capabilities where possible.

---

# 10. Error Handling and Reliability

Never silently ignore important errors.

Handle:

- Camera unavailable.
- Invalid camera configuration.
- Failed frame capture.
- Face not detected.
- Low detection confidence.
- Missing calibration.
- Database unavailable.
- ESP32 disconnected.
- Serial communication failure.
- Unexpected application shutdown.

Use specific exceptions and meaningful error messages.

Always release camera and hardware resources safely.

A disconnected ESP32 must not crash the AI engine.

A missing face should be represented as an unavailable detection state, not automatically interpreted as drowsiness.

---

# 11. Database Rules

SQLite is the initial local database.

SQLAlchemy may be used as the database abstraction layer.

Rules:

- Keep database operations inside the storage layer.
- Do not place SQL queries throughout unrelated modules.
- Use parameterised queries or ORM operations.
- Define clear relationships.
- Validate data before persistence.
- Use database migrations when the schema evolves.
- Avoid storing every camera frame.
- Store useful session summaries and meaningful events.
- Keep local calibration and session data private.
- Never commit real user profiles or database files to Git.

The database must not be required for real-time feature calculations.

---

# 12. Security and Privacy Rules

Privacy is a core architectural requirement.

- Process video locally by default.
- Do not upload camera footage without explicit user consent.
- Do not collect unnecessary personal information.
- Do not store raw recordings unless the feature explicitly requires them.
- Keep sensitive local data out of logs.
- Protect any optional online account functionality.
- Validate API inputs.
- Keep secrets out of source code.
- Document any data that leaves the user's computer.

Optional cloud features must never become a hidden dependency of the local monitoring engine.

---

# 13. Testing Rules

Every important feature must be tested.

Use:

- `pytest` for automated tests.
- Mocks for hardware-dependent unit tests.
- Integration tests for module interactions.
- Manual tests for real cameras and ESP32 hardware.

Tests should cover normal operation and failure conditions.

Never claim tests passed unless they were actually executed.

Do not remove or weaken tests simply to make a build pass.

When fixing a bug, add a regression test where practical.

---

# 14. Dependency Management Rules

Use Python 3.11 for the backend environment.

Before adding a dependency:

1. Explain why it is necessary.
2. Check whether an existing dependency already provides the functionality.
3. Consider package maintenance and compatibility.
4. Avoid unnecessary large dependencies.
5. Update dependency documentation.
6. Ensure installation instructions remain reproducible.

Do not add a framework simply because it is popular.

---

# 15. Logging Rules

Use Python's standard `logging` module.

Use appropriate levels:

- DEBUG: detailed development diagnostics.
- INFO: meaningful application events.
- WARNING: recoverable unexpected conditions.
- ERROR: operations that failed.
- CRITICAL: serious application failures.

Avoid logging every video frame at INFO level.

Log important state changes and errors with enough context to diagnose problems.

Never expose sensitive personal information unnecessarily.

---

# 16. Frontend and Dashboard Rules

The existing Next.js dashboard must be preserved.

Do not:

- Delete dashboard pages.
- Replace the frontend unnecessarily.
- Move existing components without approval.
- Make browser streaming mandatory.
- Require browser authentication for local camera access.

The eventual architecture should allow the dashboard to communicate with the local Python engine through a local API and WebSocket connection where appropriate.

The dashboard is an optional interface, not the AI engine itself.

---

# 17. Implementation Workflow

For every requested feature, follow this workflow:

### Step 1 — Understand
Read the relevant source files and project documentation. Do not assume how existing code works.

### Step 2 — Plan
Explain:
- What needs to change.
- Which files will be created.
- Which files will be modified.
- Why changes are necessary.
- What dependencies are needed.

### Step 3 — Implement
Write the smallest clean implementation that satisfies the requirement. Avoid unrelated changes.

### Step 4 — Test
Run relevant automated tests and manual checks where appropriate. Report actual results.

### Step 5 — Review
Check:
- Naming.
- Type hints.
- Docstrings.
- Error handling.
- Dependencies.
- Unnecessary complexity.
- Regression risks.

### Step 6 — Explain
Summarise the implementation in plain language suitable for academic review.

### Step 7 — Stop for approval
Do not automatically move to the next development phase.

---

# 18. Git Commit Rules

Use clear, descriptive commit messages.

Examples:

```text
chore: establish Python backend structure
feat(camera): implement OpenCV capture
feat(vision): integrate face landmarks
feat(features): implement eye aspect ratio
fix(camera): handle failed frame capture
test(features): add EAR calculation tests
docs: document calibration workflow
```

Do not make huge commits containing unrelated changes.

Never force-push, delete branches, merge or deploy without approval.

---

# 19. Prohibited Development Behaviour

The following are not acceptable:

- Generating the entire project in one response.
- Rewriting working code without justification.
- Creating unnecessary files and folders.
- Using unexplained magic numbers.
- Writing extremely long functions.
- Duplicating logic unnecessarily.
- Hiding errors with broad exception handling.
- Introducing unnecessary design patterns.
- Adding unrequested cloud dependencies.
- Storing raw camera video without a reason.
- Claiming untested functionality works.
- Inventing benchmark results.
- Treating experimental thresholds as validated science.
- Skipping project documentation.
- Proceeding to another phase without approval.

---

# 20. Required Completion Report

At the end of every implementation task, provide:

1. **What was implemented**
2. **Files changed**
3. **Architecture explanation**
4. **Technical explanation**
5. **Testing**
6. **Known limitations**
7. **Next step**

---

# FINAL DEVELOPMENT PRINCIPLE

DriveGuard must be developed as a professional engineering project.

The code must not only work; it must be understandable, explainable, testable and maintainable.

Every module must have a clear purpose.  
Every important AI measurement must be traceable.  
Every warning must have an understandable explanation.  
Every major architectural decision must have a reason.

**Never prioritise generating more code over producing better-engineered code.**
