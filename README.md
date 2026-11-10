# Face Recognition System using OpenCV, YuNet, SFace & Streamlit

A real-time face recognition application built using Python, OpenCV, YuNet, SFace, NumPy, and Streamlit. The system detects faces, extracts facial embeddings, identifies registered individuals, and classifies unrecognized faces as unknown.

It also includes an admin authentication system that restricts face registration and deletion to authorized administrators.

## 🚀 Features

### Face Detection and Recognition
- Face detection using OpenCV YuNet.
- Face recognition using OpenCV SFace.
- Extracts 128-dimensional facial feature embeddings.
- Cosine similarity-based face matching.
- Known and Unknown face classification.
- Multiple-face detection and recognition.
- Adjustable recognition threshold.

### Face Registration
- Admin-only face registration.
- Upload multiple images of the same person.
- Detects and validates uploaded faces.
- Generates embeddings from valid images.
- Averages embeddings to create a person's face representation.
- Stores registered embeddings in a local JSON database.

### Admin Authentication
- Admin login and logout.
- Password-protected admin access.
- Only administrators can register new faces.
- Only administrators can delete registered faces.
- Normal users can access face recognition without admin login.

### User Interface
- Interactive Streamlit dashboard.
- Upload images for recognition.
- Capture images using a browser camera.
- Display original and recognized images.
- Show recognition similarity scores.
- View registered people.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing and face detection |
| YuNet | Face detection model |
| SFace | Face recognition model |
| NumPy | Numerical computations and embeddings |
| Streamlit | Web application interface |
| JSON | Local face embedding database |

## 📁 Project Structure

```text
FaceRecognition/
│
├── app.py
├── webcam.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── src/
│   ├── __init__.py
│   ├── recognizer.py
│   └── database.py
│
├── dataset/
│   ├── registered_faces/
│   └── face_database.json
│
└── screenshots/
```

**Note:** The face database, registered face images, model files, and admin secrets should remain private and should not be committed to GitHub.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Recognition-System.git
```

```bash
cd Face-Recognition-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the pre-trained models

Download the following ONNX models from the official OpenCV Zoo repositories.

**YuNet – Face Detection**

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Save the model as:

```text
models/face_detection_yunet_2023mar.onnx
```

**SFace – Face Recognition**

https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

Save the model as:

```text
models/face_recognition_sface_2021dec.onnx
```

### 5. Configure admin credentials

Create a `.streamlit` folder in the project root.

Inside it, create `secrets.toml`:

```toml
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "YourStrongPassword123!"
```

Replace the example password with a strong, private password.

Do not upload `secrets.toml` to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🖥️ Application Usage

### 1. Admin Login

- Open the application.
- Enter the admin username and password in the sidebar.
- Log in to access administrative functionality.

### 2. Register a Person

- Open the Register Person tab.
- Enter the person's name.
- Upload multiple images containing that person's face.
- The system detects and extracts facial embeddings.
- Valid embeddings are averaged and saved in the database.

### 3. Recognize Faces

- Open the Face Recognition tab.
- Upload an image or capture a photo using the browser camera.
- The system detects faces and extracts embeddings.
- Each detected face is compared against registered identities.
- The result displays the predicted identity or Unknown.

### 4. Manage Registered Faces

- Open the Manage Database tab.
- View registered identities.
- Admin can select and delete a registered person.
- Normal users can view registered names but cannot delete them.

### 5. Live Webcam Recognition

Run the separate webcam application:

```bash
python webcam.py
```

Press `Q` to exit the webcam window.

## 🧠 How the System Works

The application follows this pipeline:

```text
Input Image
     |
     v
Face Detection (YuNet)
     |
     v
Face Alignment (SFace)
     |
     v
Feature Extraction
     |
     v
128-D Face Embedding
     |
     v
Cosine Similarity Matching
     |
     v
Threshold Comparison
     |
     v
Known Person / Unknown
```

### Face Detection – YuNet

YuNet identifies faces and their facial landmarks in an input image.

### Face Alignment – SFace

Detected faces are aligned using facial landmarks to improve recognition consistency.

### Feature Extraction – SFace

SFace generates a 128-dimensional feature embedding representing the detected face.

### Similarity Matching

The extracted embedding is compared with stored embeddings using cosine similarity.

Cosine similarity:

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

A similarity score above the configured threshold is treated as a potential match. Otherwise, the face is classified as Unknown.

The default threshold of 0.363 is only a starting reference and should be calibrated using representative test images.

## 🔐 Admin Access Control

| Functionality | Normal User | Admin |
|---|---|---|
| Face Recognition | Allowed | Allowed |
| View Registered People | Allowed | Allowed |
| Register New Person | Restricted | Allowed |
| Delete Registered Person | Restricted | Allowed |
| Admin Login | Allowed | Allowed |
| Admin Logout | Not applicable | Allowed |

Admin credentials are configured through Streamlit secrets rather than being hardcoded into the application.

**Security note:** This is a local portfolio prototype. A public multi-user deployment should use robust server-side authentication, authorization, password hashing, and secure database access.

## 📊 Recognition Evaluation

The system should be evaluated using separate test images that were not used during registration.

Suggested evaluation metrics:

- Face detection rate.
- Genuine match rate.
- False acceptance rate (FAR).
- False rejection rate (FRR).
- Recognition accuracy.
- Unknown-person rejection performance.

A similarity score is not a probability or confidence percentage. Performance depends on image quality, lighting, facial pose, registration images, and threshold selection.

## 🔮 Future Improvements

- Database integration using SQLite or PostgreSQL.
- Secure admin authentication with hashed passwords.
- Role-based access control.
- Face recognition history and reports.
- Improved threshold calibration.
- Recognition performance evaluation dashboard.
- Additional deployment and security improvements.

## 🔒 Privacy and Responsible Use

Face embeddings and face images are sensitive biometric information.

- Register people only with their knowledge and consent.
- Keep registered face images and embeddings private.
- Never commit biometric data or admin credentials to a public repository.
- Restrict access to the stored face database.
- Provide a way to remove registered biometric data.
- Do not use this prototype as a production security or surveillance system.

## 👨‍💻 Author

Prawin Kumar



LinkedIn: https://www.linkedin.com/in/prawin-kumar91/

## 📄 License

This project is intended for educational and portfolio purposes. Add an appropriate open-source license if you plan to distribute the source code.
