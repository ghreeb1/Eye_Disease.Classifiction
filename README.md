
# 👁️ Eye Disease Classification System

![Demo Screenshot](https://raw.githubusercontent.com/ghreeb1/Eye_Disease.Classifiction/master/templates/1.png)

A deep learning application for detecting ocular diseases from retinal images using PyTorch and Flask.

---

## 🚀 Features

- 🧠 **Multi-disease detection**: Classifies multiple common eye conditions.
- 🌐 **Web interface**: User-friendly image upload and result display.
- ⚙️ **Pre-trained model**: Ready-to-use model file `eye_disease_model.pth`.
- 📓 **Jupyter notebook**: Full training pipeline in `eye_disease.ipynb`.

---

## 📂 Project Structure

```
Eye_Disease.Classification/
├── __pycache__/
├── templates/
│   └── upload image/         # HTML templates and static files
├── app.py                    # Flask backend
├── eye_disease.ipynb         # Jupyter notebook for model training
├── eye_disease_model.pth     # Pretrained PyTorch model
└── ...
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ghreeb1/Eye_Disease.Classification.git
cd Eye_Disease.Classification/templates/upload\ image/

# Install dependencies (Python 3.7+)
pip install torch flask pillow numpy
```

---

## 🖥️ Usage

Start the Flask server:

```bash
python app.py
```

Then visit: [http://localhost:5000](http://localhost:5000)

Upload an eye image to get a prediction.

---

## 🧠 Model Specifications

| Detail              | Specification                  |
|---------------------|--------------------------------|
| Architecture        | ResNet-50                      |
| Training Dataset    | 10,000 retinal images          |
| Classes             | 5 common eye diseases          |
| Validation Accuracy | 92.4%                          |
| Inference Time      | < 500 ms per image             |

---

## 📚 Supported Diseases

- Diabetic Retinopathy  
- Glaucoma  
- Cataracts  
- AMD (Age-related Macular Degeneration)  
- Healthy Eyes  

---

## 🤝 Contributing

1. Fork the project  
2. Create your feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

**Developer:**  
<h2 align="center">Mohamed Khaled</h2>

<p align="center">
  <a href="mailto:qq11gharipqq11@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/in/mohamed-khaled-3a9021263" target="_blank">
    <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/ghreeb1/Eye_Disease.Classification" target="_blank">
    <img src="https://img.shields.io/badge/-Project%20Link-24292F?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</p>
