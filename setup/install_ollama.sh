#!/bin/bash
# Installs a fully independent Ollama instance under the CURRENT user's home
# directory -- no Homebrew, no admin/sudo, no shared state with any other
# Ollama instance on this Mac (own binary, own model store, own per-user
# LaunchAgent -- the same pattern works for running multiple isolated
# Ollama instances under different accounts on the same machine).
#
# Run this AS the account that should own the service (e.g. over SSH as
# that user) -- it only ever touches that user's own home directory and
# ~/Library/LaunchAgents, never anything requiring sudo.
#
# Usage: ./install_ollama.sh [version] [port] [base_dir]
#   version   Ollama release tag, e.g. v0.34.1 (default: v0.34.1)
#   port      port to bind OLLAMA_HOST to, 0.0.0.0:<port> (default: 11437)
#   base_dir  install root (default: $HOME/llocket/ollama)

set -euo pipefail

VERSION="${1:-v0.34.1}"
PORT="${2:-11437}"
BASE_DIR="${3:-$HOME/llocket/ollama}"
LABEL="com.llocket.ollama"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

echo "=== installing Ollama ${VERSION} for $(whoami) at ${BASE_DIR}, port ${PORT} ==="

mkdir -p "${BASE_DIR}/bin" "${BASE_DIR}/models"
cd "${BASE_DIR}"

if [ ! -f "ollama-darwin.tgz.${VERSION}" ]; then
    echo "--- downloading release assets ---"
    curl -fL -o "ollama-darwin.tgz.${VERSION}" \
        "https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-darwin.tgz"
    curl -fL -o "sha256sum.txt.${VERSION}" \
        "https://github.com/ollama/ollama/releases/download/${VERSION}/sha256sum.txt"
fi

echo "--- verifying checksum ---"
EXPECTED="$(grep 'ollama-darwin\.tgz$' "sha256sum.txt.${VERSION}" | awk '{print $1}')"
ACTUAL="$(shasum -a 256 "ollama-darwin.tgz.${VERSION}" | awk '{print $1}')"
if [ "${EXPECTED}" != "${ACTUAL}" ]; then
    echo "CHECKSUM MISMATCH for ${VERSION}: expected ${EXPECTED}, got ${ACTUAL}" >&2
    exit 1
fi
echo "checksum ok: ${ACTUAL}"

echo "--- extracting into bin/ ---"
tar -xzf "ollama-darwin.tgz.${VERSION}" -C bin
chmod +x bin/ollama

mkdir -p "$(dirname "${PLIST}")"
cat > "${PLIST}" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>EnvironmentVariables</key>
	<dict>
		<key>OLLAMA_HOST</key>
		<string>0.0.0.0:${PORT}</string>
		<key>OLLAMA_MODELS</key>
		<string>${BASE_DIR}/models</string>
	</dict>
	<key>KeepAlive</key>
	<true/>
	<key>Label</key>
	<string>${LABEL}</string>
	<key>LimitLoadToSessionType</key>
	<array>
		<string>Aqua</string>
		<string>Background</string>
		<string>LoginWindow</string>
		<string>StandardIO</string>
		<string>System</string>
	</array>
	<key>ProgramArguments</key>
	<array>
		<string>${BASE_DIR}/bin/ollama</string>
		<string>serve</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>StandardErrorPath</key>
	<string>${BASE_DIR}/serve.log</string>
	<key>StandardOutPath</key>
	<string>${BASE_DIR}/serve.log</string>
	<key>WorkingDirectory</key>
	<string>${BASE_DIR}</string>
</dict>
</plist>
PLIST_EOF

echo "--- (re)loading LaunchAgent ---"
launchctl bootout "user/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true

if launchctl bootstrap "gui/${UID_NUM}" "${PLIST}" 2>/tmp/bootstrap_gui.err; then
    DOMAIN="gui/${UID_NUM}"
    echo "loaded into gui/${UID_NUM}"
else
    echo "gui/${UID_NUM} failed ($(cat /tmp/bootstrap_gui.err)), trying user/${UID_NUM} (no GUI session needed) ---"
    launchctl bootstrap "user/${UID_NUM}" "${PLIST}"
    DOMAIN="user/${UID_NUM}"
    echo "loaded into user/${UID_NUM}"
fi

sleep 2
echo "--- verifying ---"
launchctl print "${DOMAIN}/${LABEL}" | head -20
curl -s "http://localhost:${PORT}/api/version" && echo
echo "=== done: ${LABEL} on port ${PORT}, domain ${DOMAIN} ==="
