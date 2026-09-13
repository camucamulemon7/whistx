# Recording, retention, and telemetry

The deployment operator is responsible for obtaining participants' recording consent before capture, identifying the purpose and permitted audience, and respecting applicable organizational requirements. Browser microphone permission is not participant consent.

## Processing and third parties

Audio is sent to the configured ASR provider. Transcripts, questions, and selected glossary/context are sent to the configured summary/proofreading provider when those functions are used. Shared screenshots are stored as meeting artifacts. Operators must review provider access, region, retention, and contractual terms before enabling these integrations.

Langfuse is disabled by default (`LANGFUSE_ENABLED=0`). Enabling it alone sends metadata only: counts, durations, numeric model parameters, and approved language/source categories. Arbitrary strings, model/provider details, transcript text, prompts, and glossary strings are redacted. Updates to an existing observation use the same filter.

`LANGFUSE_CAPTURE_CONTENT=1` is a separate administrator opt-in. It permits up to 8,000 characters of text per trace field, including transcription, model inputs/outputs, custom prompts, and glossary text. Email addresses, URLs, credential-looking strings, and fields named as credentials remain redacted, but automated redaction is not a complete PII detector. Obtain permission for this additional recipient and set Langfuse retention and access controls before opting in. These settings take effect after restart.

## Storage and deletion

Default application retention is seven days for saved history and 24 hours for runtime/unsaved transcripts and debug chunks. Configure `HISTORY_RETENTION_DAYS`, `RUNTIME_TRANSCRIPT_RETENTION_HOURS`, `UNSAVED_RUNTIME_RETENTION_HOURS`, and `DEBUG_CHUNKS_RETENTION_HOURS` to the approved policy. Access is restricted to the meeting owner. Deleting a history entry requests deletion of its associated media and transcript.

Application deletion cannot remove independently retained provider data or backups. Operators must define backup expiry, maintain deletion records, and replay deletions after a restore before opening the restored service to users. Keep backups encrypted and accessible only to operators. Do not copy production recordings into issues or test fixtures.

## Errors and diagnostics

HTTP provider failures and streamed errors expose stable codes and an opaque correlation ID, never raw provider exceptions. HTTP responses include `X-Correlation-ID`; SSE/WebSocket failures include `correlationId`. The same ID appears with the detailed exception in server logs. Server logs are operator-only diagnostics: restrict filesystem/container-log access, retain them only for the approved incident window, and redact attachments before sharing externally.

The repository maintainers handle security reports through the repository's private security-advisory reporting channel. Deployment-specific privacy requests belong to the deployment operator.
