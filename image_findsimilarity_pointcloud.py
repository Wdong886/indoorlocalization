import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 1. 加载预训练的ResNet-50模型，并去掉最后的全连接层来获得特征向量
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类层
model.eval()  # 设置为评估模式

# 2. 定义图像预处理函数
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
    return input_batch

# 3. 提取图像特征向量
def extract_features(image_path, model, device):
    input_batch = preprocess_image(image_path).to(device)
    with torch.no_grad():
        features = model(input_batch)
    return features.squeeze()  # 移除多余的批次维度

# 4. 计算余弦相似度
def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1, tensor2, dim=0).item()

# 5. 寻找与查询图像最相似的图像
def find_most_similar_image(query_image_path, folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 提取查询图像的特征
    query_features = extract_features(query_image_path, model, device)

    max_similarity = -1
    most_similar_image_path = None

    # 遍历文件夹中的每张图像
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # 确保是图像文件
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 提取当前图像的特征
            image_features = extract_features(image_path, model, device)

            # 计算与查询图像的相似度
            similarity = cosine_similarity(query_features, image_features)

            # 更新最相似的图像
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image_path = image_path

    return most_similar_image_path

# 6. 使用示例
query_image_path = 'data/Query_image/test.jpg'  # 查询图像的路径
folder_path = 'data/image'          # 文件夹路径
most_similar_image_path = find_most_similar_image(query_image_path, folder_path)

print(f"最相似的图像路径: {most_similar_image_path}")
