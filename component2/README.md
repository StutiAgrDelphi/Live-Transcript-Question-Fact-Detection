# Component 2: Live Question/Fact Detector

This component implements a live meeting transcript detector that identifies relevant questions and facts in real-time, using Microsoft Agent Framework (MAF).

## Setup

1. Ensure your `.env` file in the main project folder is configured.
2. You must set the `OPENAI_API_KEY` environment variable for the `DetectorAgent` to function properly.

## Running the Pipeline

You can run the end-to-end test harness against a transcript file to simulate live captioning.

```bash
python -m component2.test_harness path/to/transcript.docx --speed 20
```

The `--speed` argument controls playback speed:
- `1.0` = real-time gaps between chunks
- `20.0` = 20x real-time (useful for testing)

The harness will stream chunks through the `ContextBuffer`, `DetectorAgent`, and `DedupEngine`, outputting unique relevant flags to the console via `ConsoleDispatcher`.
