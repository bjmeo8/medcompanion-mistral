MedCompanion Architecture
=====================

High-Level Overview
-------------------
```
┌────────────────────────┐        HTTPS        ┌──────────────────────────────┐
│  MedCompanion Frontend     │  ─────────────────▶ │   FastAPI Backend (uvicorn)  │
│  (Single Page App)     │  ◀───────────────── │   main.py                    │
│  - Tailwind UI         │    JSON responses   │   • /analyze                 │
│  - Service Worker &    │                    │   • /checkin                 │
│    Manifest (PWA)      │                    │   • /analyses                │
│  - LocalStorage profile│                    │   • /verify-medication       │
└──────────┬─────────────┘                    └──────────┬───────────────────┘
           │ fetch / data                                  │
           │                                               │ reads/writes
           ▼                                               ▼
    Browser APIs (MediaRecorder,                 DATA/<patient>/visits/<id>/
    Web Audio, File inputs)                     ├─ analysis.json
                                                ├─ state.json
                                                ├─ audio files (*.webm/mp3…)
                                                └─ prescription images (.jpg/.png)
                                                            │
                                                            ▼
                                              Mistral Stack (mistralai SDK)
                                              • Voxtral Mini Transcribe (audio → text)
                                              • Mistral Large 3 (analysis, check-ins, verification)
```

Frontend Components
-------------------
- **Navigation Shell**: Responsive sidebar/bottom nav + sticky mobile header. Handles tab switching without page reloads.
- **Views**:
  * *Overview*: Greeting, medication verification CTA, dynamic treatment timeline, Safety & Alerts, and check-in workflow.
  * *Capture*: Recording/upload UI with live spectrogram, prescription upload, doctor selection, and Mistral analysis results.
  * *History*: Searchable visit list, analysis detail modal with audio player, doctor badges, and prescription preview.
  * *Profile*: Patient metrics, allergies, and settings modal (mirrors onboarding fields).
- **Local State**:
  * `patientProfile` from `localStorage` (name, auto-generated ID, biometric data, allergies, primary doctor).
  * `overviewData`, `currentTreatmentPlan`, `analysesData` fetched from backend.
  * Recording state managed via `MediaRecorder` and Web Audio analyser nodes.
- **PWA Layer**:
  * `manifest.json` references icons from `static/icons/`.
  * `sw.js` caches shell assets, enabling install prompts and offline support on HTTPS origins.

Backend Components
------------------
- **FastAPI Application (`main.py`)**
  * Serves Jinja template (`/`) plus static assets (`/static`, `/data`).
  * REST endpoints:
    - `POST /analyze`: Accepts audio, optional prescription photo, notes, patient/doctor info; uploads to Mistral; stores analysis + state.
    - `POST /checkin`: Persists daily answers, invokes Mistral for updated treatment statuses and alerts.
    - `GET /analyses`: Returns visit list with computed status and search filtering.
    - `POST /verify-medication`: Sends user-uploaded box photo + prescription references to Mistral for conformity checks.
  * Helper modules manage patient directories, visit IDs, slugification, and legacy data migration.

- **Storage Layout (`DATA/`)**
  * Per patient slug directory (generated from onboarding ID or default).
  * Each visit has its own folder containing audio, image assets, `analysis.json`, and `state.json`.
  * State files cache Mistral outputs (treatment plan, alerts, current day index, symptom questions) for quick reloads.

- **Mistral Integration**
  * `mistralai` client instantiated with API key from `.env` (chat completions + audio transcriptions).
  * Voxtral Mini Transcribe handles raw audio → text, and Mistral Large 3 handles reasoning prompts (analysis, check-ins, verification) with strict JSON.
  * Responses are validated/logged before persisting so malformed payloads are surfaced quickly.

Data Flow
---------
1. **Onboarding**: Patient fills form → frontend saves profile locally → subsequent API calls include `patient_id` + `patient_name`.
2. **Capture Flow**:
   - User records or uploads audio (`MediaRecorder` / file input) and optional prescription photo; chooses doctor.
   - `startAnalysis()` builds `FormData` and POSTs to `/analyze`.
   - Backend stores media, calls Mistral, generates `analysis.json` + `state.json`, and returns updated overview snapshot.
   - Frontend refreshes Overview & History views without reload.
3. **Check-in Flow**:
   - Frontend fetches current symptom questions from overview data.
   - Sequential UI collects answers, then POSTs to `/checkin`.
   - Backend stores answers, prompts Mistral to update plan statuses + alerts, and responds with refreshed overview data.
4. **History View**:
   - On load or search, frontend calls `/analyses`.
   - Backend reads all visit folders, ensuring states exist, attaches doctor names and tags, and returns JSON list.
5. **Medication Verification**:
   - User opens modal, uploads photo. Frontend POSTs to `/verify-medication`.
   - Backend gathers relevant prescription images, creates prompt for Mistral, and returns match status + recommendation.

Deployment Considerations
-------------------------
- Dockerfile builds the FastAPI app, includes `static/` assets and the `DATA/` mount for persistence.
- Ensure `MISTRAL_API_KEY`, `MISTRAL_MODEL`, `MISTRAL_TRANSCRIBE_MODEL`, and optional defaults are present in `.env`.

Future Enhancements
-------------------
- Centralized auth/session if MedCompanion evolves beyond single-patient local profiles.
- Streaming Mistral responses for faster perceived latency.
- Automated tests (FastAPI + Playwright) to cover capture flow, check-ins, history modal, and verification pipeline.
