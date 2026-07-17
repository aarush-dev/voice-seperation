#!/bin/bash
FF=/c/Users/Aarush/voice-seperation/tools/ffmpeg-8.1.2-essentials_build/bin/ffmpeg.exe
OUT=/c/Users/Aarush/voice-seperation/data/wsj0_wav
n=0
for f in $(find /c/Users/Aarush/voice-seperation/data/csr_1 -name "*.wv1"); do
  # .../wsj0/<split>/<spk>/<utt>.wv1  -> keep the trailing wsj0/split/spk/utt
  rel=${f#*/wsj0/}
  split=$(echo "$rel" | cut -d/ -f1)
  spk=$(echo "$rel" | cut -d/ -f2)
  utt=$(basename "$f" .wv1)
  mkdir -p "$OUT/wsj0/$split/$spk"
  "$FF" -hide_banner -loglevel error -y -i "$f" "$OUT/wsj0/$split/$spk/$utt.wav" </dev/null
  n=$((n+1))
done
echo "decoded $n files"
find "$OUT" -name "*.wav" | wc -l
