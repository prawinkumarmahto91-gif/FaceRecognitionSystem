
import streamlit as st
import cv2
import numpy as np
import hmac

from src.recognizer import (
    load_models,
    extract_embeddings,
    recognize_image
)

from src.database import (
    load_database,
    register_embedding,
    delete_person
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)


# =====================================================
# ADMIN AUTHENTICATION
# =====================================================

def admin_login():

    st.sidebar.title("Admin Panel")

    if st.session_state.get("is_admin", False):

        st.sidebar.success("Logged in as Admin")

        if st.sidebar.button("Admin Logout", key="admin_logout"):

            st.session_state.is_admin = False
            st.rerun()

        return True

    st.sidebar.subheader("Admin Login")

    username = st.sidebar.text_input(
        "Admin Username",
        key="admin_username"
    )

    password = st.sidebar.text_input(
        "Admin Password",
        type="password",
        key="admin_password"
    )

    if st.sidebar.button("Admin Login", key="admin_login"):

        try:
            correct_username = st.secrets["ADMIN_USERNAME"]
            correct_password = st.secrets["ADMIN_PASSWORD"]

        except Exception:
            st.sidebar.error(
                "Admin credentials are not configured. "
                "Check .streamlit/secrets.toml"
            )
            return False

        username_ok = hmac.compare_digest(
            username,
            correct_username
        )

        password_ok = hmac.compare_digest(
            password,
            correct_password
        )

        if username_ok and password_ok:

            st.session_state.is_admin = True
            st.rerun()

        else:

            st.sidebar.error("Invalid admin credentials.")

    return False


# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def initialize_models():

    detector, recognizer = load_models()

    return detector, recognizer


try:

    detector, recognizer = initialize_models()

except Exception as e:

    st.error(f"Model loading failed: {e}")

    st.info(
        "Check that both YuNet and SFace ONNX model files "
        "are present inside the models folder."
    )

    st.stop()


# =====================================================
# LOAD DATABASE
# =====================================================

try:

    database = load_database()

except Exception as e:

    st.error(f"Database loading failed: {e}")
    st.stop()


# =====================================================
# ADMIN STATUS
# =====================================================

is_admin = admin_login()


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("👤 Face Recognition System")

st.write(
    "A face recognition application built using "
    "OpenCV YuNet, SFace, NumPy and Streamlit."
)

st.divider()


# =====================================================
# DASHBOARD
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Registered People",
        len(database)
    )

with col2:
    st.metric(
        "Detection Model",
        "YuNet"
    )

with col3:
    st.metric(
        "Recognition Model",
        "SFace"
    )

st.divider()


# =====================================================
# APPLICATION TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Face Recognition",
    "➕ Register Person",
    "🗂️ Manage Database"
])


# =====================================================
# TAB 1: FACE RECOGNITION
# =====================================================

with tab1:

    st.header("Face Recognition")

    st.write(
        "Upload an image or capture a photo using your camera "
        "to recognize registered faces."
    )

    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.363,
        step=0.01,
        help=(
            "Starting threshold only. Calibrate it using "
            "your own genuine and impostor test images."
        )
    )

    recognition_option = st.radio(
        "Choose Input Method",
        [
            "Upload Image",
            "Camera Capture"
        ],
        horizontal=True
    )

    uploaded_image = None

    if recognition_option == "Upload Image":

        uploaded_image = st.file_uploader(
            "Upload Face Image",
            type=["jpg", "jpeg", "png"],
            key="recognition_upload"
        )

    elif recognition_option == "Camera Capture":

        uploaded_image = st.camera_input(
            "Take a Photo",
            key="recognition_camera"
        )

    if uploaded_image is not None:

        file_bytes = np.frombuffer(
            uploaded_image.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Unable to read the uploaded image.")

        elif not database:

            st.warning(
                "No registered faces found. "
                "Ask an admin to register people first."
            )

        else:

            try:

                output, results = recognize_image(
                    image,
                    database,
                    detector,
                    recognizer,
                    threshold=threshold
                )

                st.subheader("Recognition Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        cv2.cvtColor(
                            output,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Recognized Image",
                        use_container_width=True
                    )

                st.subheader("Recognition Details")

                if results:

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

                    known_count = sum(
                        1 for item in results
                        if item["Identity"] != "Unknown"
                    )

                    unknown_count = len(results) - known_count

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Faces Detected",
                        len(results)
                    )

                    col2.metric(
                        "Known Faces",
                        known_count
                    )

                    col3.metric(
                        "Unknown Faces",
                        unknown_count
                    )

                else:

                    st.warning("No face detected in this image.")

            except Exception as e:

                st.error(f"Recognition error: {e}")


# =====================================================
# TAB 2: REGISTER PERSON
# =====================================================

with tab2:

    st.header("Register New Person")

    if is_admin:

        st.success("Admin access granted.")

        st.write(
            "Upload multiple images of the same person "
            "to register their face."
        )

        person_name = st.text_input(
            "Enter Person Name",
            placeholder="Example: Prawin",
            key="register_person_name"
        )

        uploaded_files = st.file_uploader(
            "Upload Face Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="register_face_images"
        )

        if uploaded_files:

            st.write(
                f"Total uploaded images: {len(uploaded_files)}"
            )

        if st.button(
            "Register Person",
            type="primary",
            key="register_person_button"
        ):

            if not person_name.strip():

                st.error("Please enter the person's name.")

            elif not uploaded_files:

                st.error("Please upload at least one face image.")

            elif any(
                name.lower() == person_name.strip().lower()
                for name in database
            ):

                st.error(
                    "This person is already registered. "
                    "Please use a different name."
                )

            else:

                embeddings = []

                progress = st.progress(0)

                status = st.empty()

                for index, uploaded_file in enumerate(uploaded_files):

                    try:

                        file_bytes = np.frombuffer(
                            uploaded_file.getvalue(),
                            dtype=np.uint8
                        )

                        image = cv2.imdecode(
                            file_bytes,
                            cv2.IMREAD_COLOR
                        )

                        if image is None:

                            status.warning(
                                f"Cannot read {uploaded_file.name}"
                            )

                        else:

                            image_embeddings, detected_faces = (
                                extract_embeddings(
                                    image,
                                    detector,
                                    recognizer
                                )
                            )

                            if len(image_embeddings) == 1:

                                embeddings.append(
                                    image_embeddings[0]
                                )

                                status.success(
                                    f"Face detected: {uploaded_file.name}"
                                )

                            elif len(image_embeddings) == 0:

                                status.warning(
                                    f"No face detected: {uploaded_file.name}"
                                )

                            else:

                                status.warning(
                                    f"Multiple faces found in "
                                    f"{uploaded_file.name}. "
                                    "Use an image with only one person."
                                )

                    except Exception as e:

                        status.error(
                            f"Error processing {uploaded_file.name}: {e}"
                        )

                    progress.progress(
                        (index + 1) / len(uploaded_files)
                    )

                if embeddings:

                    try:

                        database = register_embedding(
                            person_name.strip(),
                            embeddings,
                            database
                        )

                        st.success(
                            f"Successfully registered "
                            f"{person_name.strip()}!"
                        )

                        st.info(
                            f"Valid images used: {len(embeddings)}"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Registration failed: {e}"
                        )

                else:

                    st.error(
                        "Registration failed. "
                        "No valid face images were found."
                    )

    else:

        st.warning(
            "🔒 Admin access required. "
            "Please log in from the sidebar to register new faces."
        )


# =====================================================
# TAB 3: MANAGE DATABASE
# =====================================================

with tab3:

    st.header("Manage Registered Faces")

    if database:

        st.write(
            f"Total registered people: {len(database)}"
        )

        for index, person in enumerate(database.keys(), start=1):

            st.write(f"{index}. 👤 {person}")

        st.divider()

        if is_admin:

            st.subheader("Admin Delete Controls")

            selected_person = st.selectbox(
                "Select Person to Delete",
                list(database.keys()),
                key="delete_person_selection"
            )

            confirm_delete = st.checkbox(
                f"I confirm that I want to delete {selected_person}",
                key="confirm_delete"
            )

            if st.button(
                "Delete Registered Face",
                type="primary",
                disabled=not confirm_delete,
                key="delete_person_button"
            ):

                try:

                    database = delete_person(
                        selected_person,
                        database
                    )

                    st.success(
                        f"{selected_person} deleted successfully!"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Deletion failed: {e}"
                    )

        else:

            st.info(
                "Only an authenticated admin can delete registered faces."
            )

    else:

        st.info("No registered people found in the database.")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Face Recognition System | OpenCV YuNet + SFace + Streamlit"
)

st.caption(
    "Use only with the knowledge and consent of people "
    "whose face data is registered."
)
