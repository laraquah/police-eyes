# 👮 Police Eyes

**Police Eyes** is a facial recognition web app built with Streamlit and powered by OpenVINO. The goal? Detect potential matches between a known reference photo and a live webcam capture — and raise the alarm when needed.

### Purpose

This app is designed to identify individuals by comparing a captured photo against an uploaded reference image. If the system finds a strong match, it displays:

> **"Criminal Identified!"**

This simulation showcases how real-time face recognition technology could be applied in security kiosks or surveillance systems.

---

### How It Works

1. **Upload a Reference Image**  
   Upload a photo of the individual you want to compare against (e.g., a suspect’s image).

2. **Start Camera and Capture Live Image**  
   Activate the webcam via the app and take a live photo of the person standing in front of the kiosk.

3. **Face Detection & Matching**  
   The app detects faces in both images and extracts facial embeddings for comparison using OpenVINO's face re-identification model.

4. **Result**  
   If the similarity crosses a set threshold, the app shows **"Criminal Identified!"** on the screen. Otherwise, no match is flagged.

---

### Tech Stack

- **Streamlit**: Frontend interface
- **OpenVINO**: Real-time inference engine
- **CV2 / NumPy / SciPy**: Image processing and calculations

---

### Notes !!

- Ensure the reference photo has a clearly visible face.
- The webcam feed will only work if a reference photo is successfully uploaded.
- This app is for educational and demonstration purposes. It does **not** access any external databases or enforce real-world legal detection.

---

### Want to Try It?

Run it locally or deploy it using Streamlit Cloud, and simulate real-time face recognition instantly!

---

### Thank You! ⭐

Thanks for visiting **Police Eyes**!  :))

![Thank You for Visiting](https://media.giphy.com/media/vTlZw1SH0CNnW/giphy.gif)
