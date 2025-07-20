from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision

# إعداد الجهاز (CPU أو GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# تعريف الفئات المطلوبة
id2label = {
    0: 'Cataract\nإعتام عدسة العين',
    1: 'Diabetic Retinopathy\nاعتلال الشبكية السكري',
    2: 'Glaucoma\nالمياه الزرقاء',
    3: 'Normal\nطبيعي'
}



# تعريف فئة Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)
        
        # تجميد طبقات معينة من الشبكة
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
            
        # إضافة طبقات جديدة
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()

    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x

# تحميل النموذج المدرب (على الـ CPU إذا لم يكن هناك CUDA)
model = Net().to(device)
model.load_state_dict(torch.load('eye_disease_model.pth', map_location=torch.device('cpu')))
model.eval()  # تعيين النموذج للوضع التقييمي

# تعريف التحويلات التي تم تطبيقها على الصور أثناء التدريب
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

# دالة لمُعالجة الصورة وتحويلها إلى Tensor
def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # إذا كانت الصورة تحتوي على 4 قنوات (مثل PNG)، حولها إلى 3 قنوات (RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # إضافة بعد الدُفعة والتحويل للجهاز
    return image

# دالة للتنبؤ بكلاس الصورة
def predict_image(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

# إنشاء تطبيق Flask
app = Flask(__name__)

# صفحة البداية
@app.route('/')
def index():
    return render_template('index.html')

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # معالجة الصورة والحصول على التنبؤ
        image_bytes = file.read()
        image_tensor = process_image(image_bytes)
        predicted_class = predict_image(image_tensor)
        
        # الحصول على التسمية من id2label
        predicted_label = id2label[predicted_class]
        return jsonify({'prediction': predicted_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
