#!/bin/bash
# Copies just the blobs/manifests needed for specific models from another
# Ollama instance's model store on the SAME machine into this one's --
# much faster than re-downloading, and avoids duplicating the full store.
# Relies on Ollama's content-addressable blob layout (blobs named
# sha256-<hex>, shared across models/tags when identical) and on the
# source directory being readable by the account running this script.
#
# Run this AS the destination account -- only reads from SRC_MODELS and
# writes to DST_MODELS, no sudo needed either side.
#
# Usage: ./copy_known_good_models.sh <src_models_dir> <dst_models_dir> <manifest-path>...
#   manifest-path is relative to .../models/manifests/, e.g.
#   registry.ollama.ai/library/qwen3.8/27b-mxfp8

set -euo pipefail

SRC="$1"; shift
DST="$1"; shift

mkdir -p "${DST}/blobs" "${DST}/manifests"

for rel in "$@"; do
    src_manifest="${SRC}/manifests/${rel}"
    dst_manifest="${DST}/manifests/${rel}"
    if [ ! -f "${src_manifest}" ]; then
        echo "MISSING manifest: ${src_manifest}" >&2
        exit 1
    fi
    echo "=== ${rel} ==="
    mkdir -p "$(dirname "${dst_manifest}")"
    cp -n "${src_manifest}" "${dst_manifest}" 2>/dev/null || true

    digests="$(python3 -c "
import json
d = json.load(open('${src_manifest}'))
digs = [d['config']['digest']] + [l['digest'] for l in d.get('layers', [])]
print('\n'.join(digs))
")"
    while IFS= read -r digest; do
        blob_name="$(echo "${digest}" | tr ':' '-')"
        expected_hash="${digest#sha256:}"
        src_blob="${SRC}/blobs/${blob_name}"
        dst_blob="${DST}/blobs/${blob_name}"

        # Verify, don't just trust existence -- a blob name is only ever
        # trustworthy if it actually hashes to what it claims. Catches a
        # truncated file left behind by an earlier interrupted run instead
        # of silently treating it as complete.
        if [ -f "${dst_blob}" ]; then
            actual_hash="$(shasum -a 256 "${dst_blob}" | awk '{print $1}')"
            if [ "${actual_hash}" = "${expected_hash}" ]; then
                echo "  already have ${blob_name} (verified)"
                continue
            fi
            echo "  ${blob_name} exists but hash mismatch (expected ${expected_hash}, got ${actual_hash}) -- re-copying"
        fi

        size=$(du -h "${src_blob}" | cut -f1)
        echo "  copying ${blob_name} (${size})"
        # Copy to a temp name first, verify, then atomic-rename into place --
        # a crash/interruption mid-copy leaves an orphaned .tmp file, never
        # a corrupt file sitting under the real (trusted-by-name) blob name.
        tmp_blob="${dst_blob}.tmp.$$"
        cp "${src_blob}" "${tmp_blob}"
        actual_hash="$(shasum -a 256 "${tmp_blob}" | awk '{print $1}')"
        if [ "${actual_hash}" != "${expected_hash}" ]; then
            echo "  COPY VERIFICATION FAILED for ${blob_name}: expected ${expected_hash}, got ${actual_hash}" >&2
            rm -f "${tmp_blob}"
            exit 1
        fi
        mv "${tmp_blob}" "${dst_blob}"
    done <<< "${digests}"
done

echo "=== done. dst usage: ==="
du -sh "${DST}"
