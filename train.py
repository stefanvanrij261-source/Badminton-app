import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ============================================
# CONFIGURATIE
# ============================================
DATASET_DIR = "video's"  # ✅ GEFIXT - wijst nu naar jouw video's folder
MODEL_SAVE_PATH = "models/badminton_model.h5"
FRAME_COUNT = 30  # Hoeveel frames per video
IMG_SIZE = 224  # Resolutie
BATCH_SIZE = 16
EPOCHS = 50

# ============================================
# STAP 1: LAAD ALLE VIDEOS EN LABELS
# ============================================
def load_video_data(dataset_dir, frame_count=FRAME_COUNT, img_size=IMG_SIZE):
    """
    Laad alle videos uit mappen
    Mapstructuur:
    video's/
    ├─ smash/
    │  ├─ video1.mp4
    │  ├─ video2.mp4
    ├─ clear/
    ├─ drop/
    └─ ...
    """
    videos = []
    labels = []
    label_to_idx = {}
    idx = 0
    
    dataset_path = Path(dataset_dir)
    
    # Verzamel alle slag-types (mappen)
    shot_types = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"✅ Gevonden slag-types: {shot_types}")
    
    for shot_type in shot_types:
        label_to_idx[shot_type] = idx
        shot_dir = dataset_path / shot_type
        
        video_files = list(shot_dir.glob("*.mp4")) + list(shot_dir.glob("*.avi"))
        print(f"  📹 {shot_type}: {len(video_files)} videos")
        
        for video_path in video_files:
            try:
                # LEES VIDEO
                frames = extract_frames(str(video_path), frame_count, img_size)
                if frames is not None:
                    videos.append(frames)
                    labels.append(idx)
            except Exception as e:
                print(f"  ⚠️  Fout bij {video_path}: {e}")
        
        idx += 1
    
    print(f"\n✅ Totaal videos geladen: {len(videos)}")
    return np.array(videos), np.array(labels), label_to_idx

# ============================================
# STAP 2: EXTRAHEER FRAMES UIT VIDEO
# ============================================
def extract_frames(video_path, frame_count=FRAME_COUNT, img_size=IMG_SIZE):
    """
    Lees video en pak evenveel frames eruit
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < frame_count:
        print(f"  ⚠️  Video {video_path} heeft te weinig frames ({total_frames} < {frame_count})")
        cap.release()
        return None
    
    # Pak gelijk verdeelde frames
    frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize en normalize
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame / 255.0  # Normalize to 0-1
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == frame_count:
        return np.array(frames)
    return None

# ============================================
# STAP 3: BOUW 3D CNN MODEL
# ============================================
def build_3d_cnn_model(num_classes, frame_count=FRAME_COUNT, img_size=IMG_SIZE):
    """
    Bouw 3D Convolutional Neural Network
    Dit ziet BEWEGING in videos (niet alleen foto's)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(frame_count, img_size, img_size, 3)),
        
        # 3D Conv Block 1
        layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same"),
        layers.MaxPooling3D((1, 2, 2)),
        layers.BatchNormalization(),
        
        # 3D Conv Block 2
        layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same"),
        layers.MaxPooling3D((1, 2, 2)),
        layers.BatchNormalization(),
        
        # 3D Conv Block 3
        layers.Conv3D(128, (3, 3, 3), activation="relu", padding="same"),
        layers.MaxPooling3D((1, 2, 2)),
        layers.BatchNormalization(),
        
        # Global Average Pooling
        layers.GlobalAveragePooling3D(),
        
        # Dense layers
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        
        # Output layer (softmax voor classificatie)
        layers.Dense(num_classes, activation="softmax")
    ])
    
    return model

# ============================================
# STAP 4: TRAIN MODEL
# ============================================
def train_model(X, y, label_to_idx, model_save_path=MODEL_SAVE_PATH):
    """
    Train het model
    """
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Training data: {len(X_train)}")
    print(f"📊 Test data: {len(X_test)}")
    
    # Bouw model
    num_classes = len(label_to_idx)
    model = build_3d_cnn_model(num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n🧠 Model architecture:")
    model.summary()
    
    # Train
    print("\n🚀 Start training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )
    
    # Evalueer op test set
    print("\n📋 Evaluatie op test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Sla model op
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\n💾 Model opgeslagen: {model_save_path}")
    
    # Sla labels op
    labels_path = os.path.join(os.path.dirname(model_save_path), "labels.json")
    import json
    with open(labels_path, 'w') as f:
        json.dump(label_to_idx, f)
    print(f"💾 Labels opgeslagen: {labels_path}")
    
    return model, history

# ============================================
# MAIN: RUN ALLES
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("🏸 BADMINTON SHOT RECOGNITION TRAINING")
    print("=" * 50)
    
    # Laad data
    print("\n📂 Laden video data...")
    X, y, label_to_idx = load_video_data(DATASET_DIR)
    
    if len(X) == 0:
        print("❌ FOUT: Geen videos gevonden!")
        print(f"Zorg dat je mappen hebt hier: {DATASET_DIR}")
        print("Voorbeeld structuur:")
        print("  video's/smash/video1.mp4")
        print("  video's/clear/video1.mp4")
        exit(1)
    
    # Train model
    print("\n" + "=" * 50)
    model, history = train_model(X, y, label_to_idx)
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLEET!")
    print("=" * 50)