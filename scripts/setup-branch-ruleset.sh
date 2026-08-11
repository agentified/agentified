#!/usr/bin/env bash
# Provision the `main` branch ruleset on ratel-ai/ratel — the enforcement behind the
# single required `ci-gate` check (see .github/workflows/ci.yml, ADR-0016). Mirrors the
# ratel-cloud ruleset: exactly one required status check (`ci-gate`) with a strict policy,
# linear history, PR review + thread resolution, and a SINGLE bypass actor — the
# `core-maintainers` org team (self-maintaining; not named users).
#
# Idempotent: PUTs (full-replaces) the ruleset named `main` if it already exists, else
# POSTs to create it. Re-running converges the live ruleset onto this payload, so this file
# is the source of truth for the gate's enforcement.
#
# Usage:
#   scripts/setup-branch-ruleset.sh            # create or update the live ruleset
#   scripts/setup-branch-ruleset.sh --print    # dry-run: print the JSON payload, apply nothing
#   scripts/setup-branch-ruleset.sh --help
#
# Requires: `gh` authenticated with admin on ratel-ai/ratel, and the `core-maintainers`
# team must already exist (roster: faustoq, kayraucklnc, rstagi, bercaakbayir). Apply fails
# with a clear message if the team is absent; --print tolerates it with a placeholder id.
#
# Verify after applying:
#   gh api repos/ratel-ai/ratel/rulesets --jq '.[] | select(.name=="main")'
set -euo pipefail

ORG="ratel-ai"
REPO="ratel"
RULESET_NAME="main"
TEAM_SLUG="core-maintainers"
REQUIRED_CHECK="ci-gate"
# The global GitHub Actions app integration id — scopes the required check to a report
# from GitHub Actions specifically (same value ratel-cloud's ruleset pins).
ACTIONS_APP_ID=15368

PRINT=false

warn() { printf '%s\n' "$*" >&2; }
die() { warn "error: $*"; exit 1; }

usage() {
  # Print the leading comment block (the help text above), minus the shebang.
  awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --print) PRINT=true ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
  done
}

# Emit the ruleset payload with the resolved team id substituted into bypass_actors.
build_payload() {
  local team_id="$1"
  cat <<JSON
{
  "name": "$RULESET_NAME",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": ["merge", "squash", "rebase"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": true,
        "required_status_checks": [
          { "context": "$REQUIRED_CHECK", "integration_id": $ACTIONS_APP_ID }
        ]
      }
    }
  ],
  "bypass_actors": [
    { "actor_id": $team_id, "actor_type": "Team", "bypass_mode": "always" }
  ]
}
JSON
}

# Resolve the core-maintainers team id, or return non-zero if it can't be read.
resolve_team_id() {
  gh api "orgs/$ORG/teams/$TEAM_SLUG" --jq '.id' 2>/dev/null
}

# Print the id of the existing ruleset named "$RULESET_NAME", or nothing. `[...][0] // empty`
# yields a single id (or blank) without a `head` pipe that pipefail would trip on.
existing_ruleset_id() {
  gh api "repos/$ORG/$REPO/rulesets" \
    --jq "[.[] | select(.name == \"$RULESET_NAME\")][0].id // empty"
}

main() {
  parse_args "$@"

  local team_id
  if team_id="$(resolve_team_id)" && [ -n "$team_id" ]; then
    :
  elif [ "$PRINT" = true ]; then
    warn "note: team '$ORG/$TEAM_SLUG' not found (or not authorized); using a placeholder id for --print."
    team_id='"<CORE_MAINTAINERS_TEAM_ID>"'
  else
    die "org team '$ORG/$TEAM_SLUG' does not exist. Create it (roster: faustoq, kayraucklnc, rstagi, bercaakbayir) and add the maintainers, then re-run."
  fi

  local payload
  payload="$(build_payload "$team_id")"

  if [ "$PRINT" = true ]; then
    printf '%s\n' "$payload"
    return 0
  fi

  local id
  id="$(existing_ruleset_id)"
  if [ -n "$id" ]; then
    warn "updating existing '$RULESET_NAME' ruleset (id $id) on $ORG/$REPO..."
    printf '%s' "$payload" | gh api --method PUT "repos/$ORG/$REPO/rulesets/$id" --input - >/dev/null
  else
    warn "creating '$RULESET_NAME' ruleset on $ORG/$REPO..."
    printf '%s' "$payload" | gh api --method POST "repos/$ORG/$REPO/rulesets" --input - >/dev/null
  fi
  warn "done. Verify: gh api repos/$ORG/$REPO/rulesets --jq '.[] | select(.name==\"$RULESET_NAME\")'"
}

main "$@"
