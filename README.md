# Thesis Chatbot

A Chainlit-based tutoring chat application for research and experimentation. The app provides two tutor profiles:

- **SRL Tutor**: a self-regulated learning tutor that routes each learner message through a multi-step orchestration pipeline.
- **Basic Tutor**: a simpler tutor that sends the current conversation context to a single LLM call.

The codebase includes mock password login, Chainlit chat profiles, SQLite persistence through Chainlit's SQLAlchemy data layer, local file storage for uploaded elements, transcript JSON files, and resume logic that rebuilds tutor context from saved transcripts or Chainlit database steps.

## Key features

- **Two tutor modes** exposed as Chainlit chat profiles:
  - `SRL Tutor`
  - `Basic Tutor`
- **Mock credential login** for research, usability, pilot, and experiment participants.
- **SQLite-backed Chainlit persistence** via `SQLAlchemyDataLayer` and `chainlit.db`.
- **Local upload storage** through a custom Chainlit `BaseStorageClient` implementation.
- **Uploaded file context extraction** for text-like files, JSON, CSV, notebooks, and PDFs.
- **Conversation transcript saving** to JSON files in `transcripts/`.
- **Conversation resume support** using saved transcript metadata first, then falling back to Chainlit database steps.
- **Per-turn metadata capture** in SRL mode, including route, diagnosis/checkpoint, decision, safety check, draft reply, timestamps, and uploaded file metadata.
- **Evaluator assets** for testing/rubric workflows under `src/evaluator/`.

## Architecture overview

At runtime, Chainlit loads `src/app.py`. The app authenticates the participant, lets the participant select a tutor profile, maintains in-memory `llm_history` in the Chainlit user session, and persists chat artifacts to both Chainlit's SQLite data layer and JSON transcript files.

```text
Browser / Chainlit UI
        |
        v
src/app.py
  - password auth
  - chat profiles
  - data layer + local upload storage
  - chat start/resume/message/end hooks
        |
        +--> Basic Tutor
        |      services/tutor.py
        |      single chat completion using the basic system prompt
        |
        +--> SRL Tutor
               services/orchestrator.py
               guard -> route -> classify/checkpoint -> policy -> generate -> safety/rewrite
```

### SRL tutor pipeline

The SRL tutor uses `Orchestrator.handle_turn()` to run these steps:

1. **Guard hint**: `GuardService` can provide a prompt hint for certain user requests.
2. **Route**: `RouterService` classifies the current SRL phase and related signals.
3. **Policy phase resolution**: `PolicyEngine` decides whether to keep or switch phase.
4. **Classify/checkpoint and decide**: `ClassifyService` diagnoses the learner state and selects a support decision.
5. **Policy decision enforcement**: `PolicyEngine` constrains support levels, depth, and code visibility.
6. **Generate**: `GenerateService` creates a draft tutor response.
7. **Safety check and rewrite**: `SafetyService` checks for unsafe/direct-solution leakage behavior and rewrites when needed.
8. **Metadata persistence**: `src/app.py` stores route, diagnosis, decision, safety, draft reply, timestamps, and file metadata in `llm_history` before saving the transcript.

### Basic tutor pipeline

The Basic tutor uses `_run_basic_tutor()` in `src/services/tutor.py`. It builds a normal OpenAI chat-completion message list from the saved history and the current learner message, loads `src/prompts/base/ai_base_control.txt` when available, and returns one assistant response. It does not run the SRL router, learner-state classifier, policy engine, or safety rewrite chain.

## Main files and modules

| Path | Purpose |
| --- | --- |
| `src/app.py` | Main Chainlit application: auth, profiles, data layer, local storage, upload processing, chat lifecycle hooks, transcript save/resume, and tutor mode dispatch. |
| `src/services/orchestrator.py` | Coordinates the SRL tutor pipeline. |
| `src/services/router.py` and `src/services/router_service.py` | Prompt and service wrapper for SRL phase routing. |
| `src/services/classify_service.py` | Converts checkpoint/decision LLM output into typed control objects. |
| `src/services/generate_service.py` | Converts typed SRL control state into generation inputs and calls the response generator. |
| `src/services/safety_service.py` | Runs reply safety/leak checks and rewrites unsafe replies. |
| `src/services/srl_chain.py` | Lower-level SRL prompt orchestration, model constants, JSON parsing, generation, safety check, and rewrite calls. |
| `src/services/tutor.py` | Basic tutor implementation. |
| `src/services/llm_client.py` | Creates the async OpenAI client from `OPENAI_API_KEY`. |
| `src/services/history_adapter.py` | Extracts recent SRL state and learning trajectory from stored history. |
| `src/services/policy/` | Phase transition and support-decision policy logic. |
| `src/lib/enums.py` | Tutor mode, SRL phase, support level, and learner-state enums. |
| `src/lib/contracts.py` | Dataclass contracts for route, checkpoint, decision, safety, and turn results. |
| `src/utils/file.py` | Reads uploaded files and converts supported formats into text context. |
| `src/utils/logger.py` | Saves JSON transcripts to `transcripts/`. |
| `src/db/init_db.py` | Helper script that initializes Chainlit-style SQLite tables. |
| `download_db.py` | Exports Chainlit thread/step history from `chainlit.db` to CSV and JSON. Requires `pandas`, which is not listed in `requirements.txt` at the time of writing. |
| `src/prompts/` | Prompt files used by SRL routing, classification, generation, safety, and the Basic tutor. |
| `src/evaluator/` | Evaluation scripts, prompts, rubrics, and saved reports. |
| `docker-compose.yml` | Docker Compose setup for the Chainlit app and optional evaluator profile. |
| `src/Dockerfile` | Multi-stage Python image for the app and evaluator runner. |
| `.chainlit/config.toml` and `src/.chainlit/config.toml` | Chainlit UI/configuration files. The Docker/local `src` working directory uses the `src/.chainlit` config. |

## Suggested project structure

This reflects the current repository layout:

```text
.
├── README.md
├── requirements.txt
├── docker-compose.yml
├── download_db.py
├── chainlit.md
├── .chainlit/
│   └── config.toml
└── src/
    ├── app.py
    ├── chainlit.md
    ├── Dockerfile
    ├── .chainlit/
    │   └── config.toml
    ├── db/
    │   └── init_db.py
    ├── evaluator/
    │   ├── evaluator_v1.py
    │   ├── evaluator_v2.py
    │   ├── judge*.py
    │   ├── prompts/
    │   ├── reports/
    │   └── rubrics/
    ├── lib/
    │   ├── contracts.py
    │   └── enums.py
    ├── prompts/
    │   ├── base/
    │   ├── chains/
    │   ├── phases/
    │   └── responses/
    ├── services/
    │   ├── orchestrator.py
    │   ├── tutor.py
    │   ├── srl_chain.py
    │   ├── *_service.py
    │   ├── generation/
    │   └── policy/
    └── utils/
        ├── file.py
        └── logger.py
```

Runtime-generated files and directories are expected at the repository root or app working directory:

```text
chainlit.db          # SQLite database used by Chainlit data layer
transcripts/         # JSON transcript files
uploaded_files/      # local file storage for uploaded Chainlit elements
chainlit_history.csv # optional export generated by download_db.py
chainlit_history.json
```

## Setup instructions

### Prerequisites

- Python 3.11 recommended. The Docker image uses `python:3.11-slim`.
- An OpenAI API key.
- Optional: Docker and Docker Compose.

### Local Python setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set:

```env
OPENAI_API_KEY=your_api_key_here
```

Run the app from `src/` so imports and prompt paths resolve as written:

```bash
cd src
chainlit run app.py --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

### Docker setup

From the repository root:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up --build chainlit
```

Open <http://localhost:8000>.

The Compose service mounts:

- `./src` to `/app`
- `./transcripts` to `/app/transcripts`
- `./chainlit.db` to `/app/chainlit.db`

## Environment variables and configuration

### Required

| Variable | Used by | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | `src/services/llm_client.py` | Required to create the async OpenAI client. The app raises a runtime error if it is missing. |

### Present in `.env.example`, but not clearly wired in current code

| Variable | Status |
| --- | --- |
| `SYSTEM_PROMPT_PATH` | TODO: present in `.env.example`, but the current Basic tutor loads `src/prompts/base/ai_base_control.txt` directly. |
| `TRANSCRIPT_DIR` | TODO: present in `.env.example`, but transcript code currently uses the hard-coded `transcripts` directory. |

### Chainlit configuration

The repository contains both root-level and `src/` Chainlit config/welcome files. Because the documented local command and Docker service run with `src` as the app directory, `src/.chainlit/config.toml` and `src/chainlit.md` are the effective app config/welcome files in that workflow.

Important visible settings include:

- Spontaneous file upload enabled.
- Accepted file types set to `*/*`.
- Maximum uploaded files: `20`.
- Maximum upload size: `500 MB`.
- Chain-of-thought display hidden in `src/.chainlit/config.toml`.

## How to run the app locally

```bash
source .venv/bin/activate
cd src
chainlit run app.py --host 0.0.0.0 --port 8000
```

Then:

1. Sign in with one of the mock credentials from `MOCK_USERS` in `src/app.py`.
2. Select either `SRL Tutor` or `Basic Tutor` in the Chainlit profile selector/settings.
3. Send a message and optionally attach files.

## Authentication

Authentication is implemented with Chainlit's `@cl.password_auth_callback` in `src/app.py`.

- Credentials are stored in the in-memory `MOCK_USERS` dictionary.
- A successful login returns a `cl.User` whose identifier is the submitted username.
- The display name is derived from the part before `@`.
- There is no database-backed user registration or password hashing visible in the current code.

**Important:** this is suitable for mock research/experiment access only. Do not use the hard-coded credential pattern for production authentication.

## Conversation persistence and resume behavior

The app persists conversation data in two layers:

1. **Chainlit SQLite data layer**
   - Configured in `get_data_layer()` with `sqlite+aiosqlite:///./chainlit.db`.
   - Stores Chainlit users, threads, steps, elements, feedback, and related UI state.
   - Uploaded Chainlit elements use the local storage provider rooted at `uploaded_files/`.

2. **JSON transcript files**
   - Saved by `save_conversation()` to `transcripts/user_<user_id>_session_<session_id>.json`.
   - Each transcript contains metadata and the app's `llm_history`.
   - `working_mode` sessions are intentionally not saved by `maybe_save()`.

On chat resume, `on_chat_resume()`:

1. Reads thread metadata for `user_id`, `tutor_type`, and `current_phase`.
2. Attempts to load the matching JSON transcript.
3. If the transcript exists and has a `history` list, restores that as `llm_history` and derives `current_phase` from metadata or recent assistant route metadata.
4. If no transcript is found, rebuilds a simpler history from Chainlit database steps using `output`/`input` text.

## Uploaded file handling

Uploads are handled in two related ways:

- Chainlit element storage is backed by `LocalStorageClient`, which writes uploaded element data under `uploaded_files/` and returns local `file://` URLs.
- Tutor context extraction is handled by `_build_combined_user_content()` in `src/app.py`, which calls `read_uploaded_file()` for each uploaded Chainlit `File` element and appends extracted text to the learner message inside `--- FILE: ... ---` blocks.

Supported file-reading behavior in `src/utils/file.py`:

| Extension | Behavior |
| --- | --- |
| `.txt`, `.md`, `.py`, `.js`, `.ts`, `.tex` | Reads text directly. |
| `.json` | Pretty-prints parsed JSON; falls back to raw text if parsing fails. |
| `.csv` | Creates a CSV summary with columns, inferred simple types, first five rows, and full content when below `CSV_FULL_TEXT_LIMIT`. |
| `.ipynb` | Extracts notebook cell type and source text. |
| `.pdf` | Extracts text from each page with `pypdf`. |
| Other extensions | Inserts an unsupported-file message. |

The app truncates extracted file content to `MAX_CHARS = 80_000` before adding it to the current user turn.

## SRL mode vs. Basic mode

| Area | SRL Tutor | Basic Tutor |
| --- | --- | --- |
| Chainlit profile name | `SRL Tutor` | `Basic Tutor` |
| Main entry point | `Orchestrator.handle_turn()` | `_run_basic_tutor()` |
| LLM calls | Multiple calls for guard/routing/classification/generation/safety as needed | One chat completion call |
| Learner state | Tracks phase, request kind, task stage, progress, attempt, context gap, expertise, frustration, SRL focus, support level, support depth, and related rationale | Does not classify learner state |
| Safety/leak rewrite | Runs in SRL pipeline for relevant support decisions | Not present in Basic pipeline |
| Metadata saved on assistant turn | `route`, `diagnosis`, `decision`, `check`, `draft_reply`, timestamp, persisted files | `draft_reply`, timestamp, persisted files |
| Intended style | Phase-aware SRL support with controlled scaffolding and pushback | Direct, solution-oriented tutoring/code support |

## Model usage visible in code

Current model constants are hard-coded in service modules:

- Router: `gpt-4.1-mini`
- SRL checkpoint/support decision: `gpt-4.1-mini`
- SRL reply planning: `gpt-4o-mini`
- SRL reply generation: `gpt-4.1-mini`
- SRL safety check and rewrite: `gpt-4o-mini`
- Basic tutor: `gpt-4.1-mini`
- Guard service: `gpt-4o-mini`

TODO: consider moving model names into environment variables or a typed configuration file if different experiments need reproducible model/version settings.

## Development notes

- Run local commands from `src/` when launching Chainlit directly, because the imports use paths such as `from lib.enums import ...` and prompt paths are resolved relative to `src`.
- The app uses async OpenAI calls through `AsyncOpenAI`.
- Transcript JSON is separate from Chainlit's SQLite persistence. Both may be needed to fully reconstruct SRL metadata and UI history.
- The `download_db.py` helper imports `pandas`, but `pandas` is not currently listed in `requirements.txt`.
- `src/db/init_db.py` can create Chainlit-style SQLite tables manually, but normal Chainlit SQLAlchemy data-layer usage may create/manage schema as part of app operation.
- The root `chainlit.md` is the default Chainlit welcome text; `src/chainlit.md` contains the SRL tutor welcome content used when running from `src`.
- Do not commit local runtime artifacts such as `.env`, `chainlit.db`, `transcripts/`, `uploaded_files/`, or exported history files unless intentionally needed for a research artifact.

## Troubleshooting

### `OPENAI_API_KEY not found`

Create a `.env` file and set `OPENAI_API_KEY`, or export it in your shell before starting Chainlit.

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Import errors when starting Chainlit

Run the app from `src/`:

```bash
cd src
chainlit run app.py --host 0.0.0.0 --port 8000
```

If running from another working directory, set `PYTHONPATH` appropriately or adjust imports.

### Login fails

Use a username/password pair defined in `MOCK_USERS` in `src/app.py`. Usernames and passwords are exact string matches.

### Uploaded files do not appear in model context

Check that:

- File upload is enabled in the active Chainlit config.
- The uploaded element is a Chainlit `File` element.
- The file extension is supported by `src/utils/file.py`.
- Very large extracted content is expected to be truncated to 80,000 characters in the app.

### Resume does not restore SRL metadata

The resume path restores full SRL metadata only when the matching JSON transcript exists and includes `history`. If only Chainlit database steps are available, the fallback history contains simpler `role`/`content` entries and cannot recreate all per-turn SRL metadata.

### `download_db.py` fails with `ModuleNotFoundError: pandas`

Install `pandas` manually or add it to `requirements.txt` before using the export helper:

```bash
pip install pandas
python download_db.py
```

## Assumptions and TODOs

- **Assumption:** this project is intended primarily for Python tutoring/research workflows, based on the Chainlit welcome text and prompt names. The application can technically receive arbitrary learner messages and supported uploaded files.
- **TODO:** replace hard-coded mock credentials with a secure participant-management approach if this is deployed outside a controlled local/research environment.
- **TODO:** decide whether `.env.example` variables other than `OPENAI_API_KEY` should be wired into the code or removed.
- **TODO:** add `pandas` to `requirements.txt` if `download_db.py` is part of the supported workflow.
- **TODO:** consider adding automated tests or documented smoke-test scripts for the SRL pipeline, Basic tutor, file parsing, and resume logic.
