#!/system/bin/sh
set -eu

PKG=io.legado.app.cosytest.debug
ROOT=/data/user/0/$PKG
MODEL=$ROOT/files/cosyvoice3-mnn/model
WORK=$ROOT/cache/cosyvoice3-mnn-work/run-1784522304244
BENCH=$ROOT/cache/hift-core-benchmark
BIN=$ROOT/code_cache/cosy_hift_bench

rm -rf "$BENCH"
mkdir -p "$BENCH/output" "$BENCH/wavs"
LINE="$WORK/hift-input $BENCH/output 220"
printf '%s\n%s\n%s\n' "$LINE" "$LINE" "$LINE" > "$BENCH/manifest.txt"

for precision in high normal low; do
  for threads in 2 4 6 8; do
    name="${precision}-${threads}"
    rm -f "$BENCH/ready" "$BENCH/start" "$BENCH/report-$name.jsonl"
    touch "$BENCH/start"
    echo "=== $name ==="
    "$BIN" \
      "$MODEL/hift-f0.fp32.mnn" \
      "$MODEL/hift-core.fp32.mnn" \
      "$BENCH/manifest.txt" \
      "$threads" \
      "$BENCH/report-$name.jsonl" \
      "$BENCH/ready" \
      "$BENCH/start" \
      cpu \
      "$precision" \
      > "$BENCH/log-$name.txt" 2>&1 || true
    cat "$BENCH/report-$name.jsonl" 2>/dev/null || true
    if [ -f "$BENCH/output/hift-android.wav" ]; then
      cp "$BENCH/output/hift-android.wav" "$BENCH/wavs/$name.wav"
    fi
  done
done
