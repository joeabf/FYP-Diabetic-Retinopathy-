import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from skimage.morphology import skeletonize

# ==========================================
# 1. BIOMARKER EXTRACTION FUNCTION
# ==========================================
def calculate_all_biomarkers(raw_oct, raw_octa):
    _, thresh_octa = cv2.threshold(raw_octa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = thresh_octa.shape[0] * thresh_octa.shape[1]
    vessel_pixels = np.count_nonzero(thresh_octa)
    vd = vessel_pixels / total_pixels

    bool_vessels = thresh_octa > 0
    skeleton = skeletonize(bool_vessels)
    sld = np.count_nonzero(skeleton) / total_pixels

    contours, _ = cv2.findContours(thresh_octa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ti = 0
    if len(contours) > 0:
        arc_lengths = sum(cv2.arcLength(c, False) for c in contours)
        areas = sum(cv2.contourArea(c) for c in contours)
        ti = (arc_lengths / (areas + 1e-5)) * 0.1

    inverted_octa = cv2.bitwise_not(thresh_octa)
    faz_contours, _ = cv2.findContours(inverted_octa, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    faz = 0
    if len(faz_contours) > 0:
        faz_contour = max(faz_contours, key=cv2.contourArea)
        faz = cv2.contourArea(faz_contour) / total_pixels

    cst = np.mean(raw_oct[raw_oct > 50]) if np.any(raw_oct > 50) else 0

    return vd, sld, ti, faz, cst

# ==========================================
# 2. DATA LOADING & ADVANCED PREPROCESSING
# ==========================================
# Update these paths to match your environment (e.g., Google Drive mount in Colab)
excel_path = '/content/drive/MyDrive/FYP_Model_Training/Text labels.xlsx'
df = pd.read_excel(excel_path)
df['ID'] = df['ID'].astype(str)

dataset_path = '/content/Final_Dataset/Final_Dataset'
img_size = (224, 224)

DEFAULT_THRESHOLD = 0.5  # Baseline decision threshold; compared against Youden's optimal

oct_images, octa_images, clinical_data, labels = [], [], [], []

print("⏳ 1/4: Extracting features, applying CLAHE + Blur, and loading dataset...")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for category in ['healthy', 'dr']:
    label = 0 if category == 'healthy' else 1
    oct_dir = os.path.join(dataset_path, category, 'OCT')
    octa_dir = os.path.join(dataset_path, category, 'OCTA')

    patient_files = [f for f in os.listdir(oct_dir) if f.endswith(('.bmp', '.png'))]

    for file in patient_files:
        patient_id = file.split('.')[0].split('_')[-1]
        patient_row = df[df['ID'] == patient_id]

        if not patient_row.empty:
            raw_oct = cv2.imread(os.path.join(oct_dir, file), cv2.IMREAD_GRAYSCALE)
            raw_octa = cv2.imread(os.path.join(octa_dir, file), cv2.IMREAD_GRAYSCALE)
            vd, sld, ti, faz, cst = calculate_all_biomarkers(raw_oct, raw_octa)

            img_oct = cv2.imread(os.path.join(oct_dir, file))
            lab_oct = cv2.cvtColor(img_oct, cv2.COLOR_BGR2LAB)
            l_oct, a_oct, b_oct = cv2.split(lab_oct)
            l_oct = clahe.apply(l_oct)
            img_oct = cv2.cvtColor(cv2.merge((l_oct, a_oct, b_oct)), cv2.COLOR_LAB2BGR)
            img_oct = cv2.GaussianBlur(img_oct, (3, 3), 0)
            img_oct = cv2.resize(img_oct, img_size) / 255.0

            img_octa = cv2.imread(os.path.join(octa_dir, file))
            lab_octa = cv2.cvtColor(img_octa, cv2.COLOR_BGR2LAB)
            l_octa, a_octa, b_octa = cv2.split(lab_octa)
            l_octa = clahe.apply(l_octa)
            img_octa = cv2.cvtColor(cv2.merge((l_octa, a_octa, b_octa)), cv2.COLOR_LAB2BGR)
            img_octa = cv2.GaussianBlur(img_octa, (3, 3), 0)
            img_octa = cv2.resize(img_octa, img_size) / 255.0

            oct_images.append(img_oct)
            octa_images.append(img_octa)
            clinical_data.append([vd, sld, ti, faz, cst])
            labels.append(label)

X_oct = np.array(oct_images)
X_octa = np.array(octa_images)
X_clinic = np.array(clinical_data)
Y = np.array(labels)
print(f"✅ Balanced dataset ready! Total images processed: {len(Y)}")

# ==========================================
# 3. SPLIT DATA
# ==========================================
# Split strategy: 15% blind test set first, then 20% of the remainder is carved
# out as an explicit validation set used for (a) early-stopping during training
# and (b) calibrating Youden's optimal threshold.  The threshold is therefore
# derived from data the model saw only as validation — never as training input —
# and the final test set remains completely untouched until evaluation.
# With ~126 samples this yields roughly 19 test, 21 val, and 86 train samples.
print("⏳ 2/4: Splitting data into train/val/test sets...")
X_oct_trainval, X_oct_test, X_octa_trainval, X_octa_test, X_clinic_trainval, X_clinic_test, Y_trainval, Y_test = train_test_split(
    X_oct, X_octa, X_clinic, Y, test_size=0.15, random_state=42, stratify=Y
)

X_oct_train, X_oct_val, X_octa_train, X_octa_val, X_clinic_train, X_clinic_val, Y_train, Y_val = train_test_split(
    X_oct_trainval, X_octa_trainval, X_clinic_trainval, Y_trainval,
    test_size=0.2, random_state=42, stratify=Y_trainval
)

print(f"  Train: {len(Y_train)} | Val: {len(Y_val)} | Test: {len(Y_test)}")

# Class weights: computed from the training labels so the model penalises
# misclassifying the minority class more heavily, improving DR recall.
class_weights_array = compute_class_weight(
    class_weight='balanced', classes=np.unique(Y_train), y=Y_train
)
class_weights = dict(enumerate(class_weights_array))
print(f"  Class weights: {class_weights}")

# ==========================================
# 4. BUILD HYBRID ARCHITECTURE (FROZEN BACKBONE)
# ==========================================
print("⏳ 3/4: Building Hybrid Neural Network (Frozen Backbone)...")
oct_input = Input(shape=(224, 224, 3), name="oct_image")
octa_input = Input(shape=(224, 224, 3), name="octa_image")
clinical_input = Input(shape=(5,), name="clinical_metrics")

# Shared backbone: the same EfficientNetB0 weights process both OCT and OCTA inputs.
# Weight sharing reduces the number of trainable parameters, which is beneficial
# given the small dataset, and allows the network to learn modality-agnostic features.
vision_backbone = EfficientNetB0(weights='imagenet', include_top=False)
vision_backbone.trainable = False  # Backbone stays fully frozen for the entire training process

x_oct = GlobalAveragePooling2D()(vision_backbone(oct_input, training=False))
x_octa = GlobalAveragePooling2D()(vision_backbone(octa_input, training=False))

y_clinic = Dense(32, activation='relu')(clinical_input)
y_clinic = Dense(16, activation='relu')(y_clinic)

combined = Concatenate()([x_oct, x_octa, y_clinic])
z = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
# Dropout 0.6: aggressive regularization is intentional given the very small dataset
# (182 samples). Combined with L2 (0.01), it strongly discourages overfitting.
z = Dropout(0.6)(z)
z = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(z)
output = Dense(1, activation='sigmoid', name="final_diagnosis")(z)

hybrid_model = Model(inputs=[oct_input, octa_input, clinical_input], outputs=output)
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 5. TRAINING (FROZEN BACKBONE)
# ==========================================
print("🚀 4/4: Training Model (Frozen Backbone)...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = hybrid_model.fit(
    x=[X_oct_train, X_octa_train, X_clinic_train],
    y=Y_train,
    epochs=30,
    batch_size=8,
    validation_data=([X_oct_val, X_octa_val, X_clinic_val], Y_val),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# Plot training curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True, linestyle='--')

ax[1].plot(history.history['loss'], label='Training Loss', color='blue')
ax[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# ==========================================
# 6. EVALUATE ON THE BLIND TEST SET
# ==========================================
print("\n📊 Generating Evaluation Metrics on Test Set...")

predictions = hybrid_model.predict([X_oct_test, X_octa_test, X_clinic_test])
predictions_binary = (predictions > DEFAULT_THRESHOLD).astype(int)
cm = confusion_matrix(Y_test, predictions_binary)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Healthy', 'Predicted DR'],
            yticklabels=['Actual Healthy', 'Actual DR'])
plt.title(f'Diagnostic Confusion Matrix — Default Threshold ({DEFAULT_THRESHOLD}) (Test Set)')
plt.show()

print(f"\n📑 Classification Report (Default {DEFAULT_THRESHOLD} Threshold, Test Set):")
print(classification_report(Y_test, predictions_binary, target_names=['Healthy', 'DR']))

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Test Set)', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# 7. APPLY OPTIMAL THRESHOLD (YOUDEN'S INDEX)
# ==========================================
# The optimal decision threshold is calibrated on the **validation** set so
# that the blind test-set evaluation remains unbiased.
print("\n🔍 Calculating Optimal Diagnostic Threshold on Validation Set...")

val_predictions = hybrid_model.predict([X_oct_val, X_octa_val, X_clinic_val])
fpr_val, tpr_val, thresholds_val = roc_curve(Y_val, val_predictions)
youden_index_val = tpr_val - fpr_val
optimal_idx = np.argmax(youden_index_val)
optimal_threshold = thresholds_val[optimal_idx]

print(f"🎯 Default AI Threshold: 0.500 (50%)")
print(f"🎯 Optimal Threshold (from validation set): {optimal_threshold:.3f} ({optimal_threshold * 100:.1f}%)")

# Apply the validation-derived threshold to the *blind* test set
predictions_optimal = (predictions >= optimal_threshold).astype(int)

cm_optimal = confusion_matrix(Y_test, predictions_optimal)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted Healthy', 'Predicted DR'],
            yticklabels=['Actual Healthy', 'Actual DR'])
plt.title(f'Diagnostic Confusion Matrix — Optimal Threshold ({optimal_threshold:.2f}, val-calibrated) (Test Set)')
plt.show()

print(f"\n📑 Classification Report (Optimal Threshold {optimal_threshold:.2f}, val-calibrated, Test Set):")
print(classification_report(Y_test, predictions_optimal, target_names=['Healthy', 'DR']))
