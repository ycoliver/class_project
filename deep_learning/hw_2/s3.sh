#!/usr/bin/env bash
LIBRISPEECH_DIR="/llmchat/daixunlian/class_project/deep_learning/hw_2/datasets/LibriSpeech"
OUT_DIR="/llmchat/daixunlian/class_project/deep_learning/hw_2/models/s3"
# 修正：指向实际的 .onnx 文件
ONNX_PATH="/llmchat/daixunlian/class_project/deep_learning/hw_2/models/CosuVoice_300M/speech_tokenizer_v1.onnx"

mkdir -p "$OUT_DIR"

find "$LIBRISPEECH_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" \) | sort | while read -r f; do
    rel="${f#"$LIBRISPEECH_DIR"/}"
    id="${rel%.*}"; id="${id//\//-}"
    echo "$id $f"
done > "$OUT_DIR/wav.scp"

python3 "/llmchat/daixunlian/class_project/deep_learning/hw_2/CosyVoice/tools/extract_speech_token.py" \
    --dir "$OUT_DIR" \
    --onnx_path "$ONNX_PATH" \
    --num_workers 16


# python3 "/llmchat/daixunlian/class_project/deep_learning/hw_2/CosyVoice/tools/extract_speech_token_torch.py" \
#     --dir "$OUT_DIR" 