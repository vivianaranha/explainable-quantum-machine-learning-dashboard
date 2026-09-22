# Security policy

## Supported version

Security fixes target the latest `1.x` release.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability. Contact the repository maintainers with a
minimal reproduction, affected version, impact assessment, and suggested mitigation. Avoid sending
secrets or personal data.

## Threat model

This local educational application accepts CSV uploads and reads saved model artifacts. Its primary
risks are resource exhaustion, malformed data, unsafe deserialization, dependency compromise, and
misinterpretation of explanations.

Controls include:

- Exact required-column validation, numeric coercion, finite-value checks, and a 10,000-row limit
- No shell execution, remote URL ingestion, API tokens, or dynamic code evaluation in the app
- JSON storage for trainable quantum parameters
- SHA-256 manifests for generated evidence
- Pinned dependency ranges and CI across supported Python versions
- Explicit warnings that joblib is pickle-based and must only load trusted local artifacts

## Known boundary

`joblib.load` can execute code embedded in a malicious pickle. Never point `--run` or the dashboard
at a model bundle from an untrusted source. The project does not sandbox model loading.

Created by School of AI and School of QC.

