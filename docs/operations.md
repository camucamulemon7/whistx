# Release, migration, rollback, and restore

The repository maintainer approves release commits and reviews CI. The deployment operator executes the following procedure and records commit/image digest, schema revision, backup location, checks, and outcome in the deployment's restricted change log. This document is a runbook; it does not claim a production restore has been exercised.

## Release

1. Select an immutable commit with all PR checks green. Run `make check` and build the container at that commit; retain the image digest and previous working image.
2. In an isolated staging deployment, restore a recent sanitized backup and run the migration and restore exercise below. Review migration scripts for destructive operations and compatibility with the old application.
3. Announce maintenance, stop new recordings, let finalization finish, and stop all application processes and cleanup workers. Keep traffic disabled throughout backup and migration.
4. Back up the database and all configured artifact roots as one consistent set. Record the current schema with `make migration-status`. Secure a separate copy of configuration and encryption/session secrets; exclude secrets from the change log.
5. Point the release environment at the intended database and run `make migrate` once as a deployment step. Application startup only verifies schema compatibility. Do not launch competing migration processes.
6. Start one application worker/replica. Check `/api/health/live` and `/api/health/ready`, log in, record synthetic audio, finalize it, open history, and download its artifacts. Confirm summary/assistant behavior if configured. Reopen traffic only after these checks pass.

## Consistent backup

Keep the application stopped during both database and artifact backup. Include every configured history, transcript, debug/audio, and screenshot directory, including paths outside `data/`. Back up permissions/ownership too. For SQLite, use its backup API or `sqlite3 PATH '.backup BACKUP_PATH'`; do not copy only the main file while a writer or WAL is active. For PostgreSQL, use `pg_dump --format=custom --file=BACKUP_PATH` with database credentials supplied by the operator's secure environment. Preserve roles/extension requirements separately.

Encrypt backups, restrict readers, verify checksums, and set an explicit expiry consistent with recording policy. Test decryption and restore access before relying on a backup. Provider data and Langfuse retention are managed independently. Keep deletion records separately so restored recordings that were deleted after backup can be removed before traffic resumes.

## Rollback

Stop traffic and application/cleanup processes. If the previous application supports the current schema, redeploy its recorded image. Otherwise restore the matching pre-migration database **and artifacts** into clean, isolated destinations, update configuration, then start the previous image. Do not overwrite the only failed-deployment copy; retain it for restricted diagnosis. Restoring the pre-release set loses post-backup writes, so assess and record that loss before switching.

Do not assume `alembic downgrade` is lossless. Inspect and rehearse the exact revision path on a disposable database first; prefer restoring the matching backup for destructive migrations. Confirm readiness, ownership, recording finalization, downloads, and deletion before reopening traffic.

## Restore exercise

Run before a release that changes storage/schema and on the operator's scheduled backup drill:

1. Create a disposable deployment with two test users, a saved synthetic recording, and a deleted recording. Capture a consistent backup and its schema/commit identifiers.
2. Restore into a fresh database and empty artifact directories. For SQLite use the backup database file while the server is stopped. For PostgreSQL create a fresh database and use `pg_restore --exit-on-error --no-owner --dbname=RESTORE_DB BACKUP_PATH`; configure required extensions/roles first. Never target production during a drill.
3. Restore artifact permissions for the container runtime user. Start the matching application commit and verify that the owner can open/download the saved recording and the other user cannot. Replay subsequent deletion records and verify deleted material is inaccessible.
4. Stop the disposable server, run `make migrate` at the candidate commit, start it, and perform the release smoke checks. Exercise rollback to the matched backup/image as described above.
5. Record backup age, restore duration, checksums, missing artifacts, test results, and whether the deployment's recovery-time and recovery-point targets were met. Remove the disposable data after review according to retention policy.

## Artifact deletion and reconciliation

Migration `20260913_0008` adds a deletion outbox. User deletion and retention expiry remove the history row and enqueue cleanup in the same transaction. Files are removed only after that transaction commits. A filesystem failure leaves the request with an attempt count and error class; the periodic cleanup worker retries batches of 100. Requests with fewer attempts run first so permanently invalid paths do not starve new requests. Missing files are treated as already deleted. PostgreSQL workers lock queue rows, and repeated execution is safe. Do not downgrade this migration with pending requests: first resolve/drain the queue or retain a backup of it.

`python -m server.storage_cli` is a read-only audit. It detects missing transcript files, unreferenced history directories, staging directories older than 24 hours, and abandoned ZIP exports in the application-owned `_exports` directory. Scheduled cleanup runs the same read-only scan and logs counts, along with pending deletion count and oldest pending age in seconds. These are structured log metrics for the operator's collector; paths and history IDs are restricted diagnostic metadata.

To retry durable deletions immediately, run `python -m server.storage_cli --retry-deletions`. To quarantine old unreferenced data, stop all application and cleanup processes, then run `python -m server.storage_cli --apply --offline`. This command rechecks references, moves only recognized paths under the configured history root, and writes a manifest under `_quarantine/TIMESTAMP/`. It never deletes a database row for missing files and never moves fresh or referenced data. Restore missing artifacts from the matching backup while offline; if irrecoverable, have the owner delete the history normally. Paths outside the root, ownership mismatches, and symlinks require manual investigation, not automatic deletion.

Retain quarantine for at least seven days for review. The operator then checks the manifest against backups and deletion policy, records approval in the restricted change log, and removes only that reviewed quarantine directory. There is no automatic quarantine purge. A mistaken quarantine can be restored offline to its manifest path after confirming that no newer data occupies it. Legacy ZIP files created in the OS-wide temp directory before this migration are excluded from automatic cleanup; review them separately with process ownership evidence.
