#!/bin/bash
# Builds LiveTranslate.app (bootstrap variant) into packaging/dist/.
#
#   ./packaging/build_app.sh                 # ad-hoc signed (friend has to allow it once)
#   ./packaging/build_app.sh --sign "Developer ID Application: … (TEAMID)" [--notarize PROFILE]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/packaging/dist"
STAGE="$DIST/Live Translate"
APP="$STAGE/LiveTranslate.app"
IDENTITY="-"
NOTARY_PROFILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --sign) IDENTITY="$2"; shift 2 ;;
    --notarize) NOTARY_PROFILE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "==> cleaning"
rm -rf "$DIST"
mkdir -p "$STAGE" "$APP/Contents/MacOS" "$APP/Contents/Resources/bin" "$APP/Contents/Resources/payload"

echo "==> Info.plist"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Live Translate</string>
  <key>CFBundleDisplayName</key><string>Live Translate</string>
  <key>CFBundleIdentifier</key><string>de.enayati.livetranslate</string>
  <key>CFBundleExecutable</key><string>LiveTranslate</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Live Translate hört über das Mikrofon zu, um gesprochene Sprache lokal auf diesem Mac zu übersetzen.</string>
</dict>
</plist>
PLIST

echo "==> compiling launcher"
xcrun swiftc -O -target arm64-apple-macos13.0 \
  -o "$APP/Contents/MacOS/LiveTranslate" \
  "$ROOT/packaging/Launcher/main.swift"

echo "==> app icon"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
cat > "$ICONSET/../mkicon.swift" <<'ICON'
import AppKit
let size = 1024.0
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
let rect = NSRect(x: 0, y: 0, width: size, height: size)
let path = NSBezierPath(roundedRect: rect.insetBy(dx: size * 0.09, dy: size * 0.09),
                        xRadius: size * 0.2, yRadius: size * 0.2)
NSGradient(colors: [NSColor(srgbRed: 0.24, green: 0.47, blue: 0.95, alpha: 1),
                    NSColor(srgbRed: 0.42, green: 0.25, blue: 0.85, alpha: 1)])!
    .draw(in: path, angle: -90)
if let sym = NSImage(systemSymbolName: "waveform", accessibilityDescription: nil)?
    .withSymbolConfiguration(NSImage.SymbolConfiguration(pointSize: size * 0.42, weight: .medium)) {
    let s = sym.size
    let r = NSRect(x: (size - s.width) / 2, y: (size - s.height) / 2, width: s.width, height: s.height)
    NSColor.white.set()
    sym.draw(in: r, from: .zero, operation: .sourceOver, fraction: 1)
    r.fill(using: .sourceAtop)
}
img.unlockFocus()
let tiff = img.tiffRepresentation!
let png = NSBitmapImageRep(data: tiff)!.representation(using: .png, properties: [:])!
try! png.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
ICON
if xcrun swiftc -O -target arm64-apple-macos13.0 -o "$ICONSET/../mkicon" "$ICONSET/../mkicon.swift" 2>/dev/null \
   && "$ICONSET/../mkicon" "$ICONSET/../icon.png" 2>/dev/null; then
  for s in 16 32 64 128 256 512 1024; do
    sips -z $s $s "$ICONSET/../icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null 2>&1 || true
  done
  mv "$ICONSET/icon_32x32.png" "$ICONSET/icon_16x16@2x.png" 2>/dev/null || true
  mv "$ICONSET/icon_64x64.png" "$ICONSET/icon_32x32@2x.png" 2>/dev/null || true
  mv "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png" 2>/dev/null || true
  mv "$ICONSET/icon_1024x1024.png" "$ICONSET/icon_512x512@2x.png" 2>/dev/null || true
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null \
    || echo "    (icon skipped)"
else
  echo "    (icon skipped)"
fi

echo "==> bundling uv"
cp "$(command -v uv)" "$APP/Contents/Resources/bin/uv"
chmod +x "$APP/Contents/Resources/bin/uv"

echo "==> bundling payload"
cp "$ROOT/packaging/bootstrap.sh" "$APP/Contents/Resources/bootstrap.sh"
chmod +x "$APP/Contents/Resources/bootstrap.sh"
rsync -a \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude 'models' \
  --exclude 'transcripts' --exclude 'packaging' --exclude 'data/flores200_dataset' \
  --exclude 'data/bench' --exclude '*.wav' --exclude '*.aiff' \
  "$ROOT/" "$APP/Contents/Resources/payload/"

echo "==> signing ($IDENTITY)"
# inside-out: nested executables first, bundle last
codesign --force --timestamp --options runtime --sign "$IDENTITY" \
  "$APP/Contents/Resources/bin/uv" 2>/dev/null \
  || codesign --force --sign "$IDENTITY" "$APP/Contents/Resources/bin/uv"
codesign --force --timestamp --options runtime --sign "$IDENTITY" "$APP" 2>/dev/null \
  || codesign --force --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict "$APP" && echo "    signature ok"

echo "==> Anleitung"
cat > "$STAGE/Bitte zuerst lesen.txt" <<'TXT'
Live Translate — lokale Live-Übersetzung
========================================

Voraussetzungen
  · MacBook/Mac mit Apple-Silicon-Chip (M1 oder neuer), mindestens 16 GB RAM
  · rund 12 GB freier Speicherplatz
  · Internet für die einmalige Einrichtung

So geht's
  1. "LiveTranslate" in den Ordner "Programme" ziehen.
  2. Doppelklick.
     Falls macOS meckert ("Apple konnte nicht prüfen …"):
     Systemeinstellungen > Datenschutz & Sicherheit > ganz unten
     auf "Trotzdem öffnen" klicken, dann nochmal doppelklicken.
  3. Beim ERSTEN Start richtet sich die App selbst ein und lädt die
     Sprachmodelle herunter. Das dauert je nach Leitung 30-60 Minuten.
     Das Fenster darf dabei offen bleiben, der Mac darf nicht schlafen.
  4. Danach öffnet sich der Browser mit der Oberfläche.
     Mikrofon-Zugriff erlauben, links oben die Sprache wählen,
     auf "Start" klicken - fertig.

Ab dem zweiten Start
  Doppelklick, ein paar Sekunden warten, Browser öffnet sich. Kein Download mehr.

Wichtig
  · Das kleine Fenster "Live Translate" muss offen bleiben, solange du
    übersetzt - es ist der Server. Schließen = beenden.
  · Alles läuft lokal auf deinem Mac. Es wird nichts ins Internet geschickt.

Deinstallieren
  App in den Papierkorb, dazu den Ordner
  ~/Library/Application Support/LiveTranslate  löschen (das sind die ~11 GB).
TXT

echo "==> zipping"
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$DIST/LiveTranslate.zip"

if [ -n "$NOTARY_PROFILE" ]; then
  echo "==> notarizing"
  xcrun notarytool submit "$DIST/LiveTranslate.zip" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$DIST/LiveTranslate.zip"
  ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$DIST/LiveTranslate.zip"
fi

echo
echo "fertig:  $DIST/LiveTranslate.zip  ($(du -h "$DIST/LiveTranslate.zip" | cut -f1))"
