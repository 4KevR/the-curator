# Project Module Overview

This document outlines the structure and functionality of the key modules in the repository. It provides a high-level
summary of their purpose and the services they offer.

---

## Module Structure Overview

```

src/
├── backend/
│   └── modules/
│       ├── ai_assistant/
│       ├── anki/
│       ├── asr/
│       ├── llm/
│       └── pdf_to_cards/
```

### 1. **`ai_assistant/`**

**Purpose**:  
Handles AI-driven task execution, and the orchestration of AI commands. Does **not** include LLM management.

- **Key Components**:
    - `action.py`: Defines executable actions for AI tasks.
    - `llm_task_executor.py`: Executes tasks using LLM-driven logic.
    - `llm_cmd_registration.py`: Manages LLM command registration.
    - `prompts.py`: Contains predefined prompt templates for LLMs.
    - `llm_controller_for_anki.py`: LLM controller tailored to interact with the `anki` module.

---

### 2. **`anki/`**

**Purpose**:  
Integrates Anki's API to manage spaced repetition decks, cards, and notes.

- **Key Components**:
    - `AbstractAnki.py`: Interface for abstracting Anki functionalities.
    - `anki_module.py`: Implementation for managing decks, cards, and notes.
    - `srs.py`: Services for managing spaced repetition logic.

---

### 3. **`asr/`**

**Purpose**:  
Handles Automatic Speech Recognition (ASR) and audio transcription services.

- **Key Components**:
    - `cloud_lecture_translator.py`: Translates audio into text using external ASR providers (cloud-based).
    - `speech.py`: API endpoints for handling transcription requests.
    - `AbstractASR.py`: Defines an interface for ASR services.

---

### 4. **`llm/`**

**Purpose**:  
Provides integrations for large language models (LLMs), managing multiple providers.

- **Key Components**:
    - `AbstractLLM.py`: Base class abstracting LLM behavior.
    - `kit_llm.py`: Manages LLM operations on "kit" (a specific LLM API).
    - `lm_studio_llm.py`: Integration logic for "LM Studio."

---

### 5. **`pdf_to_cards/`**

**Purpose**:  
Extracts text content from PDF files and generates flashcards.

- **Key Components**:
    - `AbstractPDFReader.py`: Generic interface for PDF-reading functionalities.
    - `pypdf2_reader.py`: Reads and extracts text content from PDFs using `PyPDF2`.
    - `card_generator/`: A submodule focused on generating cards from extracted text.

---

## Usage Guidelines

TODO
