# 环境
./cosy

# 运行
## 数据准备
python data_processor.py

## speak token提取
bash s3.sh

## text特征提取
cd CosyVioce
python utt2text_and_feature.py

## 检查pt文件是否完整
python check_pt_file.py

## 补充作业要求的注意力机制
example_code.py

## 开始训练
python example_code.py

## 相关路径
MODEL_CHECKPOINT = "./model_checkpoint.pt"
UTT2_S3_PATH_TEST = "./models/s3/utt2speech_token_test.pt"
UTT2_TEXT_EMB_PATH_TEST = "./models/output_text_test.pt"
UTT2_WHISPER_PATH_TEST = "./models/output_whisper_test.pt"
COSYVOICE_MODEL_DIR = "./models/CosuVoice_300M"