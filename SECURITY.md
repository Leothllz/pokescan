# Security: Secrets and VS Code settings

Do not store API keys or other secrets in `settings.json` or repository files.

Recommended actions:

- Remove exposed keys immediately and rotate them with the provider (e.g., Anthropic).
- Store secrets in OS environment variables, a local `.env` (not committed), or a secrets manager.
- For Windows PowerShell, set env vars for the current session:

```powershell
$env:ANTHROPIC_API_KEY = "<your-new-key>"
$env:ANTHROPIC_AUTH_TOKEN = "<your-new-token>"
```

- For persistent user environment variables, use Windows System Properties → Advanced → Environment Variables, or your preferred secret manager.

Audit note: I redacted the API keys from your VS Code user settings and replaced them with placeholders. Rotate the exposed keys now and confirm any services that used them are secured.
