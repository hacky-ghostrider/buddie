# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes |
| < 1.0   | No |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Email the maintainers with:

- Description of the issue
- Steps to reproduce
- Potential impact
- Any suggested fix

We will acknowledge receipt within a reasonable time and coordinate disclosure.

## Hardening notes for operators

- Never commit `.env` or API keys
- Run production containers as the non-root `rag` user (default in `Dockerfile`)
- Keep `APP_DEBUG=false` in production
- Restrict network egress if DeepEval / OpenAI / LangSmith are unused
- Treat evaluation reports as potentially sensitive (they may contain document snippets)
