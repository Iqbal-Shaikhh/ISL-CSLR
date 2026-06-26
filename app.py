import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from pyctcdecode import build_ctcdecoder
import tempfile 

# ==========================================
# 1. STANDARD MEDIAPIPE IMPORTS (WINDOWS)
# ==========================================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. ARCHITECTURE DEFINITION
# ==========================================
class SimpleConformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=15, groups=d_model, padding=7)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2),
            nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_conv = self.conv_norm(x).transpose(1, 2)
        x_conv = self.pointwise(F.silu(self.depthwise(x_conv))).transpose(1, 2)
        x = x + x_conv
        x = x + self.ff(x)
        return x

class EfficientConSignformer(nn.Module):
    def __init__(self, input_dim=1086, d_model=256, num_classes=181): 
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([SimpleConformerBlock(d_model) for _ in range(3)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return F.log_softmax(self.classifier(x), dim=-1)

# ==========================================
# 3. INITIALIZATION CACHE
# ==========================================
@st.cache_resource
def load_assets():
    df = pd.read_csv("dataset_mapping_omni.csv") 
    all_words = sorted(list(set(" ".join(df['translation'].str.lower()).split())))
    word_to_idx = {word: i+1 for i, word in enumerate(all_words)}
    word_to_idx["<blank>"] = 0
    VOCAB_SIZE = len(word_to_idx)
    
    vocab_list = [""] * VOCAB_SIZE
    for word, idx in word_to_idx.items():
        if word == "<blank>": vocab_list[idx] = ""
        else: vocab_list[idx] = word + " "
    decoder = build_ctcdecoder(labels=vocab_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientConSignformer(num_classes=VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("consignformer_omni.pth", map_location=device))
    model.eval()
    
    return model, decoder, device

model, decoder, device = load_assets()

pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_model = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_model = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==========================================
# 4. STREAMLIT UI & PROCESSING LOOP
# ==========================================
st.set_page_config(page_title="ISL Translator", layout="wide")
st.title("🗣️ Continuous Indian Sign Language Translator")
st.markdown("Developed with an Efficient Conformer architecture and Connectionist Temporal Classification (CTC).")

# NEW: Added "Upload Image" to the toggle
input_source = st.radio("Select Input Source:", ("Live Webcam", "Upload Video", "Upload Image"), horizontal=True)

st.markdown("---")

cap = None
img_frame = None
process_started = False

# Logic for selecting the source
if input_source == "Live Webcam":
    run = st.checkbox("Start Live Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        process_started = True

elif input_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a Sign Language Video (.mp4, .mov, .avi)", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        if st.button("Process Uploaded Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            process_started = True

elif input_source == "Upload Image":
    uploaded_image = st.file_uploader("Upload a Sign Language Letter (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        if st.button("Process Uploaded Image"):
            # Read bytes to numpy array, then decode into OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img_frame = cv2.imdecode(file_bytes, 1)
            process_started = True

# ==========================================
# 5A. VIDEO / WEBCAM PROCESSING LOOP
# ==========================================
if process_started and input_source in ["Live Webcam", "Upload Video"] and cap is not None:
    FRAME_WINDOW = st.image([])
    translation_text = st.empty()
    sequence_buffer = deque(maxlen=40)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if input_source == "Live Webcam":
            frame = cv2.flip(frame, 1) 
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        pose_res = pose_model.process(image)
        hands_res = hands_model.process(image)
        face_res = face_model.process(image)
        
        image.flags.writeable = True
        
        pose_arr = np.array([[res.x, res.y] for res in pose_res.pose_landmarks.landmark]).flatten() if pose_res.pose_landmarks else np.zeros(33*2)
        face_arr = np.array([[res.x, res.y] for res in face_res.multi_face_landmarks[0].landmark]).flatten() if face_res.multi_face_landmarks else np.zeros(468*2)
        lh_arr = np.zeros(21*2)
        rh_arr = np.zeros(21*2)
        
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for idx, handedness in enumerate(hands_res.multi_handedness):
                label = handedness.classification[0].label 
                landmarks = np.array([[res.x, res.y] for res in hands_res.multi_hand_landmarks[idx].landmark]).flatten()
                if label == 'Left': lh_arr = landmarks
                elif label == 'Right': rh_arr = landmarks
        
        frame_features = np.concatenate([pose_arr, face_arr, lh_arr, rh_arr])
        sequence_buffer.append(frame_features)
        
        if len(sequence_buffer) == 40:
            x = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                predicted_text = decoder.decode(probs, beam_width=50).strip()
            
            translation_text.markdown(f"### 🤖 Translation: **{predicted_text}**")
            
        if hands_res.multi_hand_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        FRAME_WINDOW.image(image)
        
    cap.release()
    if input_source == "Upload Video":
        st.success("✅ Video processing complete!")

# ==========================================
# 5B. STATIC IMAGE PROCESSING LOGIC
# ==========================================
if process_started and input_source == "Upload Image" and img_frame is not None:
    FRAME_WINDOW = st.image([])
    translation_text = st.empty()
    
    image = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    pose_res = pose_model.process(image)
    hands_res = hands_model.process(image)
    face_res = face_model.process(image)
    
    image.flags.writeable = True
    
    pose_arr = np.array([[res.x, res.y] for res in pose_res.pose_landmarks.landmark]).flatten() if pose_res.pose_landmarks else np.zeros(33*2)
    face_arr = np.array([[res.x, res.y] for res in face_res.multi_face_landmarks[0].landmark]).flatten() if face_res.multi_face_landmarks else np.zeros(468*2)
    lh_arr = np.zeros(21*2)
    rh_arr = np.zeros(21*2)
    
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for idx, handedness in enumerate(hands_res.multi_handedness):
            label = handedness.classification[0].label 
            landmarks = np.array([[res.x, res.y] for res in hands_res.multi_hand_landmarks[idx].landmark]).flatten()
            if label == 'Left': lh_arr = landmarks
            elif label == 'Right': rh_arr = landmarks
    
    frame_features = np.concatenate([pose_arr, face_arr, lh_arr, rh_arr])
    
    # 🧠 SYNTHETIC VIDEO SIMULATION
    # The Conformer expects temporal dimension. We duplicate the static features 40 times.
    simulated_sequence = [frame_features] * 40
    x = torch.tensor(np.array(simulated_sequence), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        predicted_text = decoder.decode(probs, beam_width=50).strip()
    
    translation_text.markdown(f"### 🤖 Prediction: **{predicted_text}**")
    
    if hands_res.multi_hand_landmarks:
        for hand_landmarks in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    FRAME_WINDOW.image(image)
    st.success("✅ Image processing complete!")