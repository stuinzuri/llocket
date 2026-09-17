#!/bin/bash
# Idempotent NFS auto-mount for the Llocket queue share. Meant to run at
# boot via a LaunchDaemon (see com.llocket.nas-mount.plist) -- retries
# briefly in case the network isn't up yet when launchd fires this at boot.
#
# No admin/sudo needed to run this: NFS mounts under a path the calling
# user owns don't require root on macOS, and a NAS export that doesn't
# enforce a reserved (privileged) client port doesn't need `resvport`
# either (that option itself requires root to bind).
#
# Usage: ./mount_llocket_nas.sh <nas_host> <nas_export> <mount_point>
#   nas_host     NFS server hostname, e.g. nas.local
#   nas_export   export path on that server, e.g. /data/llocket
#   mount_point  local directory to mount it at, e.g. ~/llocket/nas

set -uo pipefail

NAS_HOST="${1:?usage: mount_llocket_nas.sh <nas_host> <nas_export> <mount_point>}"
NAS_EXPORT="${2:?usage: mount_llocket_nas.sh <nas_host> <nas_export> <mount_point>}"
MOUNT_POINT="${3:?usage: mount_llocket_nas.sh <nas_host> <nas_export> <mount_point>}"
MAX_ATTEMPTS=10
RETRY_DELAY=10

mkdir -p "${MOUNT_POINT}"

if mount | grep -q " ${MOUNT_POINT} "; then
    echo "$(date -u +%FT%TZ) already mounted"
    exit 0
fi

attempt=1
while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    echo "$(date -u +%FT%TZ) mount attempt ${attempt}/${MAX_ATTEMPTS}"
    if mount -t nfs -o vers=3 "${NAS_HOST}:${NAS_EXPORT}" "${MOUNT_POINT}" 2>&1; then
        echo "$(date -u +%FT%TZ) mounted ${NAS_HOST}:${NAS_EXPORT} at ${MOUNT_POINT}"
        exit 0
    fi
    attempt=$((attempt + 1))
    sleep "${RETRY_DELAY}"
done

echo "$(date -u +%FT%TZ) FAILED to mount after ${MAX_ATTEMPTS} attempts" >&2
exit 1
