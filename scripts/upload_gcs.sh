#!/usr/bin/env bash
# Upload the large artifacts to Google Cloud Storage.
#
# The git repository carries code, metrics and figures. Everything here is either too large
# for git (12 GB of weights) or is personal data that must not go to a public host.
#
# WHAT MAKES THIS SAFE TO RE-RUN: `gcloud storage rsync` skips files already present and
# identical, so an interrupted upload resumes rather than restarting. Every path is uploaded
# under a versioned prefix so a re-run cannot silently overwrite a previous study's artifacts.
#
# PRIVATE BY DEFAULT. The dataset contains photographs of an identifiable person. The bucket
# is created with uniform bucket-level access and NO public read. Making it public is a
# separate, deliberate act -- there is no flag here that does it.
#
#   ./scripts/upload_gcs.sh --bucket gs://my-bucket --dry-run    # see what would go
#   ./scripts/upload_gcs.sh --bucket gs://my-bucket              # do it
#
set -euo pipefail

BUCKET=""
PREFIX="midi-gesture-v2"
DRY=""
INCLUDE_PHOTOS="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)  BUCKET="$2"; shift 2 ;;
    --prefix)  PREFIX="$2"; shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    --include-photos) INCLUDE_PHOTOS="yes"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "usage: $0 --bucket gs://your-bucket [--prefix NAME] [--dry-run] [--include-photos]" >&2
  exit 2
fi
command -v gcloud >/dev/null || { echo "gcloud not found. Install the Google Cloud CLI first." >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEST="${BUCKET%/}/${PREFIX}"

echo "source      : $ROOT"
echo "destination : $DEST"
echo "photos      : $INCLUDE_PHOTOS"
[[ -n "$DRY" ]] && echo "MODE        : dry run, nothing will be written"
echo

# name:path pairs. Kept explicit rather than globbed so that adding a directory to the
# project does not silently start uploading it.
UPLOADS=(
  "yolo-checkpoints:midi_results/weights"
  "deimv2-checkpoints:deim_results/weights"
  "deimv2-pretrained:DEIMv2checkpoints"
  "coreml-exports:models"
  "results:results"
  "figures:figures"
  "logs-deimv2:deim_results/logs"
  "dataset-labels:data"
)

for entry in "${UPLOADS[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  if [[ ! -e "$path" ]]; then
    echo "  skip $name  ($path not present)"
    continue
  fi
  size=$(du -sh "$path" 2>/dev/null | cut -f1)
  echo "==> $name  ($size)  $path"
  # shellcheck disable=SC2086
  gcloud storage rsync --recursive $DRY \
      --exclude='.*\.DS_Store$' --exclude='.*__pycache__.*' --exclude='.*\.cache$' \
      "$path" "$DEST/$name"
done

if [[ "$INCLUDE_PHOTOS" == "yes" ]]; then
  echo
  echo "==> photographs  (IDENTIFIABLE PERSONAL DATA -- keep this bucket private)"
  # shellcheck disable=SC2086
  gcloud storage rsync --recursive $DRY "photov2" "$DEST/photographs"
else
  echo
  echo "  photographs NOT uploaded. Pass --include-photos to include them,"
  echo "  and only to a private bucket: they show the subject's face."
fi

echo
echo "done. Contents:"
gcloud storage ls "$DEST/" || true
echo
echo "This bucket should remain private. To check:"
echo "  gcloud storage buckets describe ${BUCKET} --format='value(iamConfiguration)'"
