import pandas as pd
import jieba
import re
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# 配置BERT模型（自动下载预训练的中文BERT）
config_path = 'https://cdn.keras.io/models/bert-base-chinese/config.json'
checkpoint_path = 'https://cdn.keras.io/models/bert-base-chinese/bert_model.ckpt'
dict_path = 'https://cdn.keras.io/models/bert-base-chinese/vocab.txt'

print("正在加载BERT模型...")
# 初始化tokenizer和模型
tokenizer = Tokenizer(dict_path, do_lower_case=False)
model = build_transformer_model(config_path, checkpoint_path)
print("✅ BERT模型加载成功！")

# 数据准备
test_data = pd.DataFrame({
    "comment": [
        "这部电影剧情超棒，演技在线，推荐大家看！",
        "新买的手机续航太差，售后还敷衍，太坑了！",
        "今天天气不错，适合出门散步，心情很好～",
        "这个餐厅的菜又贵又难吃，避雷！"
    ],
    "scene_label": ["影视评价", "产品吐槽", "日常分享", "消费体验"],
    "emotion_label": ["正面", "负面", "正面", "负面"],
    "multi_labels": ["影视,推荐", "产品,差评", "日常,开心", "消费,避雷"]
})

# 预处理函数
stop_words = {"的", "了", "是", "我", "你", "他", "们", "在", "有", "就", "都"}


def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    words = jieba.lcut(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


test_data["cleaned_comment"] = test_data["comment"].apply(clean_text)


# BERT特征提取函数
def get_bert_features(texts):
    features = []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=128)
        # 获取BERT输出（CLS token的向量）
        vec = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
        features.append(vec[0])  # 取第一个token（CLS）的特征
    return np.array(features)


print("正在提取BERT特征...")
# 提取所有文本的BERT特征
bert_features = get_bert_features(test_data["cleaned_comment"].tolist())
print(f"✅ BERT特征提取完成，特征形状: {bert_features.shape}")

# 标签编码和模型训练
print("正在训练分类模型...")
scene_encoder = LabelEncoder().fit(test_data["scene_label"])
emotion_encoder = LabelEncoder().fit(test_data["emotion_label"])
mlb = MultiLabelBinarizer().fit(test_data["multi_labels"].str.split(","))

# 训练分类器
scene_clf = LogisticRegression(max_iter=1000).fit(bert_features, scene_encoder.transform(test_data["scene_label"]))
emotion_clf = LogisticRegression(max_iter=1000).fit(bert_features,
                                                    emotion_encoder.transform(test_data["emotion_label"]))
multi_clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(bert_features, mlb.transform(
    test_data["multi_labels"].str.split(",")))

print("✅ 模型训练完成！")


# 预测函数
def predict(comment):
    cleaned = clean_text(comment)
    feat = get_bert_features([cleaned])
    scene_pred = scene_encoder.inverse_transform(scene_clf.predict(feat))[0]
    emotion_pred = emotion_encoder.inverse_transform(emotion_clf.predict(feat))[0]
    multi_pred = mlb.inverse_transform(multi_clf.predict(feat))[0]

    return {
        "输入文本": comment,
        "场景分类": scene_pred,
        "情感倾向": emotion_pred,
        "多标签": multi_pred
    }


# 测试预测
print("\n=== 测试预测结果 ===")
test_comments = [
    "这部科幻电影特效太棒了，剧情也很精彩！",
    "这个品牌的手机质量太差，用了一周就坏了！"
]

for comment in test_comments:
    result = predict(comment)
    print(f"\n{result}")

print("\n🎉 所有任务完成！")