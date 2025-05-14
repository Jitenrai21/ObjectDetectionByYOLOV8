# ColorObjectDetectorApp 🎨📸

A Python-based color detection system for identifying colored objects via webcam.

---

## 👋 Welcome

Welcome to **ColorObjectDetectorApp**!  
This Python project lets you detect objects of specific colors (**pink**, **light blue**, **yellow**, **green**, **white**, **orange**) using your webcam. Built with **OpenCV**, it’s a simple and fun way to:

- Detect colored objects in real-time  
- Switch between colors with a key press  
- Visualize detection with bounding boxes and masks  

Point your webcam at a colored object and start detecting! 🚀

---

## 🌟 What It Does

### 🎨 Detect Colors

Identifies objects in the following colors:

- 💗 **Pink** – Press `p`  
- 💙 **Light Blue** – Press `b`  
- 💛 **Yellow** – Press `y`  
- 💚 **Green** – Press `g`  
- 🤍 **White** – Press `w`  
- 🧡 **Orange** – Press `o`  

### 📺 Show Feedback

- Bounding box around the detected object with a color label (e.g., `"Yellow Object"`)  
- Current color selection displayed on-screen (e.g., `"Color: Yellow"`)  
- Instructions for selecting colors  
- Optional mask window for visualizing HSV detection  

---

## 🔧 Customizable

All color HSV ranges are defined in `configs.py`.  
Easily modify or add colors to detect.

---

## 📂 Project Layout

```

ColorObjectDetectorApp/
├── env/                    # 🌐 Virtual environment
├── tracking/               # 🛠️ Detection scripts
│   ├── detector.py         # 🎨 Object detection logic
│   └── utils.py            # 🖼️ Visualization utilities
├── tests/                  # 🧪 Unit tests
│   ├── config.py           # ⚙️ Test configuration
│   ├── test_detector.py    # 🧪 Detection tests
├── configs.py              # ⚙️ Main configuration
├── main.py                 # 🚀 Main application
├── requirements.txt        # 📋 Dependencies
├── ProjectStructure.txt    # 📄 Project structure documentation

````

---

## 🛠️ Getting Started

### 📋 Prerequisites

- Python 3.8+  
- Webcam  

### 🧱 Required Packages

- `opencv-python`  
- `numpy`

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ColorObjectDetectorApp.git
cd ColorObjectDetectorApp
````

### 2. Set Up Virtual Environment

```bash
python -m venv env
```

**Activate Virtual Environment**

* **Windows:**

```bash
.\env\Scripts\activate
```

* **macOS/Linux:**

```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

---

## 🎮 How to Use

* Point your webcam at a colored object (e.g., pink cloth, blue pen)
* Press the corresponding key to select a color
* Press `q` to quit the application

### 🎨 Supported Colors

| Key | Color         | Description                |
| --- | ------------- | -------------------------- |
| `r` | 💗 Pink       | Detects pink objects       |
| `b` | 💙 Light Blue | Detects light blue objects |
| `y` | 💛 Yellow     | Detects yellow objects     |
| `g` | 💚 Green      | Detects green objects      |
| `w` | 🤍 White      | Detects white objects      |
| `o` | 🧡 Orange     | Detects orange objects     |

### 📊 On-Screen Feedback

* **Bounding Box:** Label with object name (e.g., `"Pink Object"`)
* **Color Label:** Shows current color selection
* **Instructions:** Color selection keys listed
* **Mask Window:** (Optional) Displays HSV detection mask if enabled in config

---

## 🧩 Customize It

* **Add New Colors:**
  Modify `COLOR_RANGES` in `config.py` and update `color_keys` in `main.py`

* **Adjust Detection Area:**
  Update `MIN_CONTOUR_AREA` in `config.py` to detect smaller/larger objects

* **Toggle Mask Display:**
  Set `SHOW_MASK = False` in `config.py` to hide the mask

* **Fine-Tune HSV Ranges:**
  Tweak HSV values in `config.py` based on lighting/environment

---

## 🐞 Troubleshooting

* **Webcam Not Working?**
  Ensure it's connected and functioning.
  Try changing `CAMERA_INDEX` in `config.py` (e.g., from `0` to `1`)

* **Dependencies Issue?**
  Verify you're inside the virtual environment when running `pip install`

* **Colors Not Detected?**

  * Use bright, solid-colored objects
  * Adjust HSV ranges in `config.py`
  * Improve lighting conditions

* **Text Overlapping?**
  Modify text position or font size in `main.py` (e.g., change `(10, 60)` to `(10, 70)`)

---

## 🤝 Contribute

Want to improve this app? Here’s how:

```bash
# Fork the repo
# Create a new branch
git checkout -b feature/your-feature-name

# Commit your changes
git commit -m "Add new feature"

# Push to GitHub
git push origin feature/your-feature-name

# Open a Pull Request 🚀
```

---

## 🙏 Acknowledgments

* **OpenCV** – For real-time computer vision
* **You** – For checking out this project! ❤️

---

Built by **Jiten Rai**
**Happy detecting! 🎉**

```
