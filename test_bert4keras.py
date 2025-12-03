# 终极完美版：修复分类报告类别匹配 + 全流程无错
import os
import sys
import types

# 环境变量锁定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# 核心依赖处理：模拟 keras.engine 解决导入错误
import tensorflow as tf
import tf_keras


# 1. 模拟 Keras 2.x 所需的 keras.engine.base_layer.Node
class MockNode:
    def __init__(self, *args, **kwargs):
        pass


# 2. 构建模拟模块层级
engine_module = types.ModuleType('keras.engine')
base_layer_module = types.ModuleType('keras.engine.base_layer')
base_layer_module.Node = MockNode
engine_module.base_layer = base_layer_module


# 3. 包装 tf_keras 注入 engine 属性
class KerasWrapper(tf_keras.__class__):
    def __init__(self):
        self.__dict__.update(tf_keras.__dict__)
        self.engine = engine_module


# 4. 替换系统模块映射
keras_wrapper = KerasWrapper()
sys.modules['keras'] = keras_wrapper
sys.modules['keras.backend'] = tf_keras.backend
sys.modules['keras.layers'] = tf_keras.layers
sys.modules['keras.models'] = tf_keras.models
sys.modules['keras.optimizers'] = tf_keras.optimizers
sys.modules['keras.losses'] = tf_keras.losses
sys.modules['keras.callbacks'] = tf_keras.callbacks
sys.modules['keras.engine'] = engine_module
sys.modules['keras.engine.base_layer'] = base_layer_module

# 其他依赖导入
import re
import jieba
import requests
import numpy as np
import pandas as pd
import zipfile
from io import BytesIO
from sklearn.metrics import accuracy_score, classification_report
from bert4keras.backend import set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator

# 初始化配置
np.random.seed(42)
tf.random.set_seed(42)
set_gelu("tanh")
print("=== 环境配置校验 ===")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"tf_keras 版本: {tf_keras.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"可用 GPU 数量: {len(gpus)}")
print("=" * 50)


# ===================== 1. 全局配置 =====================
class Config:
    MODEL_CACHE_DIR = "./model_cache/"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    OFFICIAL_BERT_ZIP_URL = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"
    ZIP_INNER_DIR = "chinese_L-12_H-768_A-12/"
    CONFIG_PATH = os.path.join(MODEL_CACHE_DIR, ZIP_INNER_DIR, "bert_config.json")
    CHECKPOINT_PATH = os.path.join(MODEL_CACHE_DIR, ZIP_INNER_DIR, "bert_model.ckpt")
    DICT_PATH = os.path.join(MODEL_CACHE_DIR, ZIP_INNER_DIR, "vocab.txt")
    TASK_TYPE = "multi_class"
    SAVE_DIR = "./bert_trained_model/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    BATCH_SIZE = 1
    EPOCHS = 1
    LEARNING_RATE = 1e-5
    MAX_LEN = 32  # 固定序列长度


# ===================== 2. BERT 模型下载与解压 =====================
def download_and_extract_bert(config):
    required_files = [
        config.CONFIG_PATH,
        config.CHECKPOINT_PATH + ".index",
        config.CHECKPOINT_PATH + ".data-00000-of-00001",
        config.DICT_PATH
    ]
    if all(os.path.exists(f) for f in required_files):
        print("✅ 已检测到 BERT 模型文件，直接复用")
        return config

    print("📥 开始下载官方中文 BERT 模型（约 400MB）...")
    try:
        response = requests.get(config.OFFICIAL_BERT_ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        zip_buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                zip_buffer.write(chunk)
                downloaded_size += len(chunk)
                progress = (downloaded_size / total_size) * 100 if total_size > 0 else 100
                print(f"下载进度: {progress:.1f}%", end="\r")

        print("\n✅ 下载完成，开始解压模型文件...")
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(config.MODEL_CACHE_DIR)
        print(f"✅ 解压完成，模型文件保存至: {config.MODEL_CACHE_DIR}")
    except Exception as e:
        print(f"\n❌ 模型下载/解压失败: {str(e)}")
        print("💡 解决方案：手动下载模型包并解压至 ./model_cache/")
        print(f"手动下载链接: {config.OFFICIAL_BERT_ZIP_URL}")
        sys.exit(1)
    return config


# ===================== 3. 词典加载 =====================
def load_bert_token_dict(dict_path):
    token_dict = {}
    with open(dict_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                token, idx = line.split('\t', 1)
                token_dict[token] = int(idx)
            else:
                token_dict[line] = len(token_dict)
    return token_dict


# ===================== 4. 数据加载与预处理 =====================
def load_sample_data():
    print("⚠️  未检测到自定义数据集，使用示例情感数据")
    data = {
        "comment": ["电影超棒", "手机太差", "天气很好", "餐厅难吃", "剧特效好", "服务极差", "性价比一般", "体验不错"],
        "multi_label": [0, 1, 2, 1, 0, 1, 2, 0]
    }
    train_df = pd.DataFrame(data).iloc[:5]
    val_df = pd.DataFrame(data).iloc[5:6]
    test_df = pd.DataFrame(data).iloc[6:]
    label_col = "multi_label"
    text_col = "comment"
    num_labels = len(train_df[label_col].unique())
    label_map = {0: "正面", 1: "负面", 2: "中性"}
    print(f"📊 任务信息：{Config.TASK_TYPE}，类别数：{num_labels}，标签映射：{label_map}")
    return train_df, val_df, test_df, num_labels, label_col, text_col, label_map


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\u4e00-\u9fa5]", "", str(text))
    return text.strip()


# ===================== 5. 数据生成器 =====================
class CustomDataGenerator(DataGenerator):
    def __init__(self, data, tokenizer, text_col, label_col, max_len, batch_size=32, shuffle=True):
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len
        super().__init__(data, batch_size, shuffle)
        self.data_len = len(data)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            text = item[self.text_col]
            if not text:
                token_ids = [0] * self.max_len
                segment_ids = [0] * self.max_len
            else:
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
                if len(token_ids) < self.max_len:
                    token_ids += [0] * (self.max_len - len(token_ids))
                    segment_ids += [0] * (self.max_len - len(segment_ids))
                elif len(token_ids) > self.max_len:
                    token_ids = token_ids[:self.max_len]
                    segment_ids = segment_ids[:self.max_len]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([item[self.label_col]])

            if len(batch_token_ids) == self.batch_size or is_end:
                padded_token_ids = sequence_padding(batch_token_ids, length=self.max_len)
                padded_segment_ids = sequence_padding(batch_segment_ids, length=self.max_len)
                yield (
                    np.array(padded_token_ids, dtype=np.int32),
                    np.array(padded_segment_ids, dtype=np.int32),
                    np.array(batch_labels, dtype=np.int32)
                )
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def to_tf_dataset(self):
        def generator_fn():
            for token_ids, segment_ids, labels in self:
                yield (
                    {
                        "Input-Token": token_ids,  # 形状：(1, 32)
                        "Input-Segment": segment_ids  # 形状：(1, 32)
                    },
                    labels  # 形状：(1, 1)
                )

        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_signature=(
                {
                    "Input-Token": tf.TensorSpec(shape=(self.batch_size, self.max_len), dtype=tf.int32),
                    "Input-Segment": tf.TensorSpec(shape=(self.batch_size, self.max_len), dtype=tf.int32)
                },
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.int32)
            )
        )
        return dataset.prefetch(tf.data.AUTOTUNE)


# ===================== 6. 模型构建 =====================
def build_bert_classifier(config, num_labels):
    try:
        bert_base = build_transformer_model(
            config_path=config.CONFIG_PATH,
            checkpoint_path=config.CHECKPOINT_PATH,
            model="bert",
            return_keras_model=True,
            verbose=0
        )
    except Exception as e:
        print(f"❌ BERT 基础模型构建失败: {str(e)}")
        sys.exit(1)

    # 提取 CLS token 特征（句子级特征）
    cls_output = tf_keras.layers.Lambda(lambda x: x[:, 0, :])(bert_base.output)

    output = tf_keras.layers.Dropout(rate=0.1, seed=42)(cls_output)
    output = tf_keras.layers.Dense(
        units=num_labels,
        activation="softmax",
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02, seed=42)
    )(output)

    model = tf_keras.models.Model(inputs=bert_base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return model


# ===================== 7. 模型评估（核心修复：类别匹配）=====================
def evaluate_model_performance(model, test_generator, label_map):
    print("\n=== 模型评估开始 ===")
    y_true = []
    y_pred = []
    for token_ids, segment_ids, labels in test_generator:
        pred = model.predict(
            {
                "Input-Token": token_ids,
                "Input-Segment": segment_ids
            },
            verbose=0
        )
        y_pred.extend(pred.argmax(axis=1))
        y_true.extend(labels.flatten())

    if len(y_true) == 0:
        print("⚠️  测试集无有效样本，跳过评估")
        return {"accuracy": 0.0}

    accuracy = accuracy_score(y_true, y_pred)
    print(f"准确率: {accuracy:.4f}")

    # 核心修复：筛选测试集实际存在的类别，确保 target_names 数量匹配
    actual_labels = sorted(list(set(y_true)))  # 实际存在的类别（去重+排序）
    actual_target_names = [label_map[label] for label in actual_labels]  # 对应标签名称

    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred,
        labels=actual_labels,  # 指定实际存在的类别
        target_names=actual_target_names,  # 匹配实际类别的标签名称
        zero_division=0
    ))
    print("=" * 50)
    return {"accuracy": accuracy}


# ===================== 8. 预测函数 =====================
def predict_single_text(text, model, tokenizer, config, label_map, threshold=0.5):
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "无效文本（无中文内容）", 0.0
    token_ids, segment_ids = tokenizer.encode(cleaned_text, maxlen=config.MAX_LEN)
    if len(token_ids) < config.MAX_LEN:
        token_ids += [0] * (config.MAX_LEN - len(token_ids))
        segment_ids += [0] * (config.MAX_LEN - len(segment_ids))
    elif len(token_ids) > config.MAX_LEN:
        token_ids = token_ids[:config.MAX_LEN]
        segment_ids = segment_ids[:config.MAX_LEN]
    pred = model.predict(
        {
            "Input-Token": np.array([token_ids]),
            "Input-Segment": np.array([segment_ids])
        },
        verbose=0
    )[0]
    pred_label_id = np.argmax(pred)
    confidence = pred[pred_label_id]
    return (label_map[pred_label_id], confidence) if confidence >= threshold else ("不确定（置信度不足）", confidence)


# ===================== 9. 主函数（全流程无错）=====================
def main():
    config = Config()
    config = download_and_extract_bert(config)

    # 加载数据
    train_df, val_df, test_df, num_labels, label_col, text_col, label_map = load_sample_data()

    # 加载词典
    print(f"\n🔤 加载词典文件: {config.DICT_PATH}")
    try:
        token_dict = load_bert_token_dict(config.DICT_PATH)
        tokenizer = Tokenizer(token_dict=token_dict, do_lower_case=True)
        print("✅ 词典加载成功")
    except Exception as e:
        print(f"❌ 词典加载失败: {str(e)}")
        sys.exit(1)

    # 文本清洗
    train_df["cleaned_text"] = train_df[text_col].apply(clean_text)
    val_df["cleaned_text"] = val_df[text_col].apply(clean_text)
    test_df["cleaned_text"] = test_df[text_col].apply(clean_text)
    text_col = "cleaned_text"

    # 创建生成器并转换为 Dataset
    train_generator = CustomDataGenerator(
        data=train_df.to_dict("records"),
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    train_dataset = train_generator.to_tf_dataset()

    val_generator = CustomDataGenerator(
        data=val_df.to_dict("records"),
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    val_dataset = val_generator.to_tf_dataset()

    test_generator = CustomDataGenerator(
        data=test_df.to_dict("records"),
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # 构建模型
    model = build_bert_classifier(config, num_labels)
    print(f"\n✅ 完整模型构建成功，总参数: {model.count_params():,}")
    print("\n🚀 开始模型训练（CPU 环境，1 轮训练）...")

    # 训练模型
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[
            tf_keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=0,
                mode="max",
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # 评估与保存
    evaluate_model_performance(model, test_generator, label_map)
    weight_save_path = os.path.join(config.SAVE_DIR, "bert_classifier_best.weights")
    model.save_weights(weight_save_path)
    print(f"✅ 模型权重已保存至: {weight_save_path}")

    # 预测示例
    print("\n=== 预测示例 ===")
    test_texts = ["产品质量太差了，完全不值这个价", "体验很好，超出预期", "一般般，没什么特别的", "垃圾产品，千万别买"]
    for text in test_texts:
        pred_label, pred_confidence = predict_single_text(text, model, tokenizer, config, label_map)
        print(f"文本: {text}")
        print(f"预测结果: {pred_label}，置信度: {pred_confidence:.3f}")
        print("-" * 30)


if __name__ == "__main__":
    main()