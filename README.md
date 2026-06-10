🔢 Captcha Reader (Digits)

A Python-based digit CAPTCHA reader that uses **DdddOCR**, **OpenCV**, and **NumPy** to preprocess and recognize numeric CAPTCHA images with high accuracy. This project is designed for **web scraping** and **automation** workflows where digit-based CAPTCHA solving is required.

## ✨ Features

- 🔍 Detects and recognizes **digit-only CAPTCHAs**
- 🖼️ Image preprocessing using **OpenCV**
- ⚡ OCR powered by **DdddOCR**
- 📊 NumPy-based image manipulation
- 🤖 Suitable for web scraping and automation tasks
- 🚀 Lightweight and easy to integrate into existing Python projects

## 🛠️ Technologies Used

- Python 3.x
- DdddOCR
- OpenCV (cv2)
- NumPy

## 📂 Project Structure

```
Captcha_Reader-digits/
│── captcha_images/        # Input CAPTCHA images
│── output/                # Processed images (optional)
│── main.py                # Main execution file
│── requirements.txt       # Required dependencies
│── README.md
```

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/NILESHMESHRAM29/Capcha_Reader-digits-.git
cd Capcha_Reader-digits-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ddddocr opencv-python numpy
```

## ▶️ Usage

Run the project:

```bash
python main.py
```

The script will preprocess the CAPTCHA image and extract the digit sequence using DdddOCR.

## 📸 Workflow

1. Load CAPTCHA image
2. Preprocess image using OpenCV
3. Apply image enhancement techniques
4. Perform OCR using DdddOCR
5. Return recognized digits

## 🎯 Use Cases

- Web scraping automation
- CAPTCHA research
- OCR experimentation
- Educational purposes
- Automation testing

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests for improvements.

**Author:** Nilesh Meshram
