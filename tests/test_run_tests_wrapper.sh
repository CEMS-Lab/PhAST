#!/usr/bin/env bash
# tests/test_run_tests_wrapper.sh -- bash test for scripts/agents/run_tests.sh
#
# Strategy: stub `python` on PATH with a script that records its argv, then
# verify the wrapper passes the right flags depending on $PWD detection.
#
# Run: bash tests/test_run_tests_wrapper.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
WRAPPER="$REPO_ROOT/scripts/agents/run_tests.sh"

if [[ ! -x "$WRAPPER" ]]; then
  echo "FAIL: wrapper not executable at $WRAPPER" >&2
  exit 1
fi

TMP="$(mktemp -d -t run_tests_wrapper_test.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# ---- Build a fake `python` on PATH that records its argv to a file. -------
STUB_BIN="$TMP/stubbin"
mkdir -p "$STUB_BIN"
ARGV_FILE="$TMP/argv.txt"
cat >"$STUB_BIN/python" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$@" >"$ARGV_FILE"
exit 0
EOF
chmod +x "$STUB_BIN/python"

run_wrapper_at() {
  # $1: pretend $PWD ; $2..: args to wrapper
  local fake_pwd="$1"; shift
  mkdir -p "$fake_pwd"
  : >"$ARGV_FILE"
  (
    cd "$fake_pwd"
    PATH="$STUB_BIN:$PATH" bash "$WRAPPER" "$@"
  )
}

# ---- Test 1: worktree path -- expect --rootdir / --import-mode flags. -----
WORKTREE_FAKE="$TMP/repo/.claude/worktrees/agent-fake1234"
run_wrapper_at "$WORKTREE_FAKE" tests/foo.py --collect-only

if ! grep -q -- "-m" "$ARGV_FILE"; then
  echo "FAIL [worktree]: expected python -m in argv, got:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "pytest" "$ARGV_FILE"; then
  echo "FAIL [worktree]: expected pytest in argv, got:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "--rootdir=$WORKTREE_FAKE/tests" "$ARGV_FILE"; then
  echo "FAIL [worktree]: --rootdir=<worktree>/tests missing from argv:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "--import-mode=importlib" "$ARGV_FILE"; then
  echo "FAIL [worktree]: --import-mode=importlib missing from argv:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "tests/foo.py" "$ARGV_FILE"; then
  echo "FAIL [worktree]: user pytest args not forwarded:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "--collect-only" "$ARGV_FILE"; then
  echo "FAIL [worktree]: --collect-only not forwarded:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
echo "PASS [worktree]: wrapper applied --rootdir=<worktree>/tests + --import-mode=importlib"

# ---- Test 2: non-worktree path -- expect clean pass-through. --------------
NORMAL_FAKE="$TMP/normal_repo"
run_wrapper_at "$NORMAL_FAKE" tests/bar.py -k smoke

if grep -q -- "--rootdir=" "$ARGV_FILE"; then
  echo "FAIL [non-worktree]: --rootdir leaked into argv:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if grep -q -- "--import-mode=importlib" "$ARGV_FILE"; then
  echo "FAIL [non-worktree]: --import-mode leaked into argv:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "tests/bar.py" "$ARGV_FILE"; then
  echo "FAIL [non-worktree]: user pytest args not forwarded:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
if ! grep -q -- "-k" "$ARGV_FILE"; then
  echo "FAIL [non-worktree]: -k flag not forwarded:" >&2
  cat "$ARGV_FILE" >&2
  exit 1
fi
echo "PASS [non-worktree]: wrapper passed args through cleanly"

echo "ALL TESTS PASSED"
