# VoiceMatch AI

VoiceMatch AI is a full-stack speaker verification application. It compares two voice recordings and estimates whether they belong to the same speaker.

The application combines a React interface, a FastAPI inference service, and a locally stored fine-tuned ECAPA-TDNN model.

## Overview

| Area | Description |
| --- | --- |
| Input | Audio uploads or browser microphone recordings |
| Processing | Mono, 16 kHz normalization and speaker embedding extraction |
| Model | Fine-tuned ECAPA-TDNN |
| Decision | Cosine similarity with recording-level consistency checks |
| Output | Match decision, confidence, similarity score, and playback controls |

## Highlights

- Supports WAV, MP3, M4A, OGG, and FLAC uploads
- Supports browser microphone recording on `localhost` or HTTPS
- Handles multilingual comparisons
- Reuses the existing trained model without retraining
- Provides a simple result for non-technical users: **Match detected** or **No match detected**

## Demo



![VoiceMatch AI demo](demo/Screenshot%202026-09-18%20102449.png)



## Project Architecture

```text
+-------------------+      HTTP multipart     +----------------------+
| React + Vite UI   | ----------------------> | FastAPI backend      |
| Upload / record   |                         | /api/verify          |
+-------------------+                         +-----------+----------+
                                                       |
                                                       |
                                                       v
                                            +----------------------+
                                            | ECAPA-TDNN inference |
                                            | Embeddings + scoring |
                                            +----------------------+
```

## File Structure

```text
VoiceMatchAI/
├── backend/
│   ├── app.py
│   ├── model_inference.py
│   ├── requirements.txt
│   └── models/
│       └── ecapa_librispeech_finetuned/
│           ├── embedding_model.ckpt
│           ├── hyperparams.yaml
│           ├── model_info.json
│           └── version
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── Header.tsx
│       ├── index.css
│       ├── types.ts
│       └── components/
│           ├── SimilarityResult.tsx
│           └── VoiceCard.tsx
├── notebook/
│   └── ecapa-tdnn-librispeech-fine-tuning-evaluation.ipynb
├── .gitignore
├── README.md
└── demo/
    └── Screenshot 2026-09-18 102449.png
```

## Evaluation

The accompanying notebook reports **87.42% validation accuracy** on **16,965 speaker pairs**.

This result is a project evaluation reference.

## Training Dataset

The notebook trains and evaluates on the **LibriSpeech ASR corpus**, specifically the `train-clean-360` split available through Kaggle. The audio is organized by speaker and chapter, which allows the notebook to create same-speaker and different-speaker pairs.

The notebook creates an 85/15 train-validation split, converts audio to mono 16 kHz, and uses three-second samples for model training and pair evaluation.

## Quick Start

### Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- Chrome or Edge for microphone recording

### Start the backend

Run these commands from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8001
```

Leave this terminal running.

### Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, normally:

```text
http://localhost:5174/
```

Allow microphone access when the browser asks for permission.

## How to Use

1. Upload or record the first voice sample.
2. Upload or record the second voice sample.
3. Select **Compare Voices**.
4. Review the result and similarity details.



## Model

The backend loads the existing fine-tuned model from:

```text
backend/models/ecapa_librispeech_finetuned/
```

The notebook contains the original fine-tuning and evaluation workflow. The frontend and backend use the saved model directly and do not retrain it during startup.


## Validation Commands

Frontend production build:

```powershell
cd frontend
npm run build
```

Backend syntax check:

```powershell
python -m py_compile backend\model_inference.py
```
