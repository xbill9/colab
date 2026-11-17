#!/usr/bin/env python3
"""Print a Google access token carrying the `colaboratory` scope, for the Colab API and MCP endpoint.

The token comes from the Colab CLI's own login (`~/.config/colab-cli/token.json`), which is the
credential on this machine that holds the scope. Plain `gcloud auth application-default login`
grants `cloud-platform` only, and every Colab API method refuses that with
ACCESS_TOKEN_SCOPE_INSUFFICIENT. Sign in once with:

    colab --auth oauth2 sessions

The endpoint also needs a quota project: a Google Cloud project with colaboratory.googleapis.com
enabled, sent as `x-goog-user-project`. It is COLAB_QUOTA_PROJECT when set, else the active gcloud
project (`gcloud config get-value project`):

    gcloud services enable colaboratory.googleapis.com --project <your-project>
    export COLAB_QUOTA_PROJECT=<your-project>

Usage:
    python3 tools/colab_token.py            # the bare token
    python3 tools/colab_token.py --header   # the request headers as JSON, for Claude Code's headersHelper

Register the hosted MCP endpoint in Claude Code with this as its header source:

    claude mcp add-json colab --scope user '{"type": "http",
      "url": "https://colaboratory.googleapis.com/mcp",
      "headersHelper": "COLAB_QUOTA_PROJECT=<your-project> python3 /path/to/colab/tools/colab_token.py --header"}'

The refreshed token is held in memory only; the CLI's token file is never rewritten.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

SCOPE = "https://www.googleapis.com/auth/colaboratory"


def quota_project() -> str:
    """COLAB_QUOTA_PROJECT, else the active gcloud project, else none."""
    if os.environ.get("COLAB_QUOTA_PROJECT"):
        return os.environ["COLAB_QUOTA_PROJECT"]
    try:
        out = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


TOKEN_FILE = pathlib.Path(os.environ.get("COLAB_CLI_TOKEN", "~/.config/colab-cli/token.json")).expanduser()


def access_token() -> str:
    import google.auth.transport.requests
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials

    if not TOKEN_FILE.is_file():
        sys.exit(f"no Colab CLI login at {TOKEN_FILE}; run: colab --auth oauth2 sessions")
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if SCOPE not in (creds.scopes or []):
        sys.exit(f"{TOKEN_FILE} lacks the {SCOPE} scope; run: colab --auth oauth2 sessions")
    try:
        creds.refresh(google.auth.transport.requests.Request())
    except RefreshError as e:
        sys.exit(f"Colab CLI login expired ({e}); run: colab --auth oauth2 sessions")
    return creds.token


def headers() -> dict:
    """Authorization plus the quota project the endpoint requires."""
    h = {"Authorization": f"Bearer {access_token()}"}
    if project := quota_project():
        h["x-goog-user-project"] = project
    return h


def main() -> int:
    if "--header" in sys.argv[1:]:
        print(json.dumps(headers()))
        return 0
    token = access_token()
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
