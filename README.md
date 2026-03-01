# MedCompanion

MedCompanion is a Mistral-powered companion web app/PWA that helps patients capture their consultations, follow treatments, verify medications, and maintain a structured visit history. The UI is a Tailwind-based SPA rendered by FastAPI/Jinja, while the backend orchestrates **Mistral Large 3 (`mistral-large-2512`)** for reasoning plus **Voxtral Mini Transcribe (`voxtral-mini-latest`)** for accurate audio transcription.

---

## Highlights

- **Patient onboarding + profile management** stored locally (name, biometrics, allergies, primary doctor, generated ID).
- **Visit capture** via MediaRecorder (with Web Audio spectrogram) or audio uploads, plus prescription photo attachment.
- **Mistral visit analysis** returns transcripts, summaries, treatment plans, and safety alerts that feed the Overview & History tabs.
- **Daily check-ins** with sequential questions; Mistral updates medication statuses (Taken/Missed/Pending) and refreshes alerts in real time.
- **Medication verification** modal compares uploaded box photos with prescription data using Mistral’s multimodal vision (default `MEDICATION_VERIFICATION_MODE=lite`, switch to `heavy` to include prescription photos).
- **History explorer** with powerful search, doctor badges, detailed modal (transcript sections, tags, playable audio).
- **Installable PWA** with manifest + service worker, optimized for touch devices.

---

## Repository Layout

```
medcompanion/
  main.py                # FastAPI entrypoint
  templates/medcompanion.html# SPA template w/ Tailwind + JS
  static/                # Icons, manifest, service worker, styles
  DATA/                  # Runtime storage (per-patient visit folders)
  Dockerfile             # Container build
  docs/                  # Specs + architecture + Mistral notes
  requirements.txt
```

---

### System Architecture

![MedCompanion Architecture](https://raw.githubusercontent.com/bjmeo8/medvisit/main/docs/medvisit-architecture.png)

MedCompanion is built around Mistral multimodal AI architecture combining audio transcription, image analysis, and longitudinal reasoning...

---

## Prerequisites

- Python **3.10+**
- Mistral API access + key (`MISTRAL_API_KEY`)
- Optional: Docker / Docker Compose for containerized deployments

---

## Local Setup

1. **Clone & enter repo**
   ```bash
   git clone <repo-url>
   cd medcompanion
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   Create `.env` with:
   ```env
   MISTRAL_API_KEY=your-key
   MISTRAL_MODEL=mistral-large-2512        # optional override
   MISTRAL_TRANSCRIBE_MODEL=voxtral-mini-latest
   MEDICATION_VERIFICATION_MODE=lite       # switch to "heavy" to embed prescription photos
   DEFAULT_DOCTOR=Dr. Example
   ```

5. **Run FastAPI server**
   ```bash
   uvicorn main:app --reload
   ```
   Visit `http://localhost:8000` to load MedCompanion.

---

## Using MedCompanion

1. **Onboarding** – Provide patient info (name, weight, height, blood type, allergies, primary doctor). ID is auto-generated and cached in `localStorage`.
2. **Capture a visit** – Record audio or upload `.mp3/.m4a/.wav/.webm`, attach prescription photo, note the specialist, then click **Analyze Visit**. Mistral transcribes with Voxtral Mini and summarizes with Mistral Large. Photos above 150 KB are automatically compressed before upload.
3. **Overview** – View “Your Treatment” (per-day plan), dynamic Safety & Alerts, verification CTA (“Verify your medication” sends box photos to Mistral vision). Medication photos are likewise compressed to stay under 150 KB before verification.
4. **History** – Search past visits, open detailed modal (sections, tags, playable recording). Audio auto-pauses when closing the modal.
5. **Check-ins** – Step through personalized questions; responses are POSTed to `/checkin`, Mistral updates statuses/alerts without jumping ahead a day.

---

## Data Storage

- `DATA/<patient-slug>/visits/<visit-id>/`
  - `analysis.json` – canonical visit record (summary, sections, doctor, media URLs).
  - `state.json` – cached overview state (treatment plan, safety alerts, symptom questions, current day index).
  - Uploaded media (audio + prescription photos).
- `/data` static mount serves stored files for playback/preview. Persist this directory in production (volume mount, shared storage, etc.).

---

## Testing & Validation

Manual checklist (automated tests TBD):
- Onboard a new patient and ensure data persists after reloads.
- Capture both recorded and uploaded audio; confirm Overview & History refresh using new Mistral outputs.
- Run multiple check-ins per day to ensure the same calendar day stays in focus.
- Use medication verification with valid/invalid photos to review Mistral vision responses.

---

## Deployment

### Docker
```bash
cd medcompanion
docker build -t medcompanion .
docker run -p 8000:8000 --env-file .env -v $(pwd)/DATA:/app/DATA medcompanion
```
(Mount `DATA` to persist visits.)

### Scripts
`setup-aws-2023.sh` and `deploy-aws-2023.sh` show how to provision a host, pull the container, and run it. Adjust to match your environment.

### PWA Requirements
- Serve over HTTPS for install prompts.
- Keep `/static/manifest.json` and `/static/sw.js` accessible.
- Icons are generated from `static/medcompanion-logo-hd.png` (reuse or update as needed).

---

## Troubleshooting

- **“Could not import module `app`”** – start uvicorn from the repo root (`uvicorn main:app --reload`) or update Docker CMD to `python -m uvicorn main:app`.
- **Mistral errors / malformed JSON** – backend logs the raw response before parsing. Inspect server logs for details.
- **Audio recorder unavailable** – browser lacks `MediaRecorder`. Use the upload button instead.
- **Mobile file picker can’t find audio** – the uploader accepts `audio/*` and explicit extensions (`.mp3`, `.m4a`, `.aac`, `.wav`, `.ogg`, `.webm`). Make sure files are stored locally or in iCloud/Drive.

---

## References

- [docs/medcompanion-specs.txt](docs/medcompanion-specs.txt) – behavior & UX spec (now powered by Mistral).
- [docs/architecture.md](docs/architecture.md) – system diagram.
- [docs/mistral-audio-transcript.md](docs/mistral-audio-transcript.md) – Voxtral transcription guide.
- [docs/mistral-with-vision-capabilities.md](docs/mistral-with-vision-capabilities.md) – multimodal / vision examples.

Issues and PRs are welcome! Please include reproduction steps plus expected vs actual behavior.***
