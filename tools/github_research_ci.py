"""Narrow GitHub repository/CI operations; credentials never logged or persisted."""
import argparse
import json
import os
import subprocess
from urllib.request import Request, urlopen

REPO = "damraka/CleoLOB"


def request(path, method="GET", payload=None):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
    credential = subprocess.run(["git", "credential", "fill"],
                                input="protocol=https\nhost=github.com\n\n",
                                text=True, capture_output=True, env=env, timeout=30)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("No existing GitHub credential available through Git Credential Manager")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "CleoLOB-research-ci"}
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request("https://api.github.com/repos/" + REPO + path,
                         headers=headers, data=data, method=method), timeout=30) as response:
        raw = response.read()
        return json.loads(raw) if raw else {"status": response.status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "description", "dispatch", "runs", "jobs"])
    parser.add_argument("--ref", default="main")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.action == "status":
        result = request("")
        print(json.dumps({key: result.get(key) for key in
                          ("full_name", "description", "default_branch", "permissions", "visibility")}))
    elif args.action == "description":
        result = request("", "PATCH", {"description": "Reproducible PPO vs Almgren-Chriss execution research: L2 calibration, independent seeds, uncertainty and failure reporting."})
        print(json.dumps({"description": result["description"], "html_url": result["html_url"]}))
    elif args.action == "dispatch":
        print(json.dumps(request("/actions/workflows/research.yml/dispatches", "POST", {"ref": args.ref})))
    elif args.action == "runs":
        result = request("/actions/runs?per_page=10")
        print(json.dumps([{key: run.get(key) for key in
                           ("id", "head_branch", "head_sha", "status", "conclusion", "html_url", "created_at")}
                          for run in result["workflow_runs"]], indent=2))
    elif args.action == "jobs":
        if not args.run_id or not args.run_id.isdecimal():
            parser.error("--run-id numeric is required for jobs")
        print(json.dumps(request("/actions/runs/" + args.run_id + "/jobs"), indent=2))


if __name__ == "__main__":
    main()
