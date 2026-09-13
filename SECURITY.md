# Security reporting

Report vulnerabilities privately to the repository owner, **camucamulemon7**, using GitHub's “Report a vulnerability” button under this repository's Security tab when available. If private reporting is unavailable, use the owner's GitHub profile contact channel to request a private conversation; do not publish exploit details, credentials, customer data, or recordings in a public issue.

Include the affected commit/version, deployment topology, impact, minimal reproduction using synthetic data, and a suggested fix if known. Maintainers will confirm the report, investigate supported deployments, coordinate a patch and disclosure with the reporter, and publish an advisory when appropriate. Response times are best effort; no guaranteed SLA is offered.

Security fixes target current `main`. Older deployments should upgrade to a reviewed release commit after testing migrations and backups. Deployment operators are responsible for TLS, access control, updates, provider configuration, and incident response. Rotate exposed credentials immediately through the affected provider, and preserve restricted diagnostic evidence. See [privacy](docs/privacy.md) for recording and telemetry policy.
