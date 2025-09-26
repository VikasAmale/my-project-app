import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import zipfile
import io
import os
import math
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# --- Placeholder for Disease Information ---
# NOTE: This is for demonstration purposes only and is NOT medical advice.
# In a real-world application, this information should be sourced from verified medical experts.
DISEASE_INFO = {
    "Melanoma": {
        "description": "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. While it is less common than other skin cancers, it is more dangerous because of its ability to spread to other organs more rapidly if not treated at an early stage.",
        "precautions": [
            "Limit exposure to ultraviolet (UV) radiation by seeking shade, especially during peak sun hours.",
            "Wear broad-spectrum sunscreen with a high SPF, along with protective clothing, hats, and sunglasses.",
            "Avoid tanning beds and sunlamps.",
            "Perform regular self-examinations of your skin to notice any new or changing moles or spots.",
            "Schedule annual professional skin checks with a dermatologist."
        ]
    },
    "Benign keratosis-like lesions": {
        "description": "These are non-cancerous (benign) skin growths that may appear as waxy, brown, black, or tan spots. They are very common, especially in older adults, and are not contagious.",
        "precautions": [
            "These lesions are generally harmless and do not require treatment.",
            "Monitor the growths for any significant changes in size, shape, or color.",
            "Consult a dermatologist if a lesion becomes irritated, itchy, or bleeds, or for cosmetic removal.",
            "Protecting your skin from the sun can help prevent the formation of new lesions."
        ]
    },
    # Add other potential class names from your dataset here
    "Default": {
        "description": "No specific information is available for this predicted category in the demo.",
        "precautions": ["For any health concerns, it is essential to consult with a qualified medical professional for an accurate diagnosis and advice."]
    }
}

# This function is corrected to be memory-safe by copying data in batches.
def load_features_memmap(saved_files, batch_size=1024):
    """
    Combine multiple chunked feature npy files into a single memmap in batch mode.
    This version is memory-safe and corrects the indexing bug.
    Returns memmap path, total_rows, feat_dim.
    """
    shapes = [np.load(f, mmap_mode='r').shape for f in saved_files]
    total_rows = sum(s[0] for s in shapes)
    feat_dim = shapes[0][1]

    # Use a temporary directory from the constants
    mempath = os.path.join(FEAT_DIR, f"features_{os.path.basename(saved_files[0])}.mmap")
    fp = np.memmap(mempath, dtype='float32', mode='w+', shape=(total_rows, feat_dim))

    current_pos = 0
    for f in saved_files:
        arr = np.load(f, mmap_mode='r')
        n_rows = arr.shape[0]
        # Process each chunk file in smaller batches to avoid large memory reads
        chunk_steps = math.ceil(n_rows / batch_size)
        for i in range(chunk_steps):
            start = i * batch_size
            end = min(n_rows, start + batch_size)
            batch_data = arr[start:end]
            batch_len = len(batch_data)
            fp[current_pos : current_pos + batch_len, :] = batch_data
            current_pos += batch_len # Correctly update position after each batch write
            
    fp.flush() # Ensure data is written to disk
    return mempath, total_rows, feat_dim


# This function is corrected to accept the image feature shape to prevent errors.
def combine_features_memmap(img_mempath, img_shape, tab_matrix, out_name, batch_size=1024):
    """
    Combines image features from a memmap with tabular features into a new, final memmap.
    This is done in memory-safe batches to prevent loading everything into RAM.
    """
    # Use the passed-in shape to correctly open the memmap file in 2D
    img_feats = np.memmap(img_mempath, dtype='float32', mode='r', shape=img_shape)
    
    # Now we can safely get the dimensions from the shape tuple
    rows, img_dim = img_shape
    tab_dim = tab_matrix.shape[1]
    
    # If no tabular data, just return the image features memmap
    if tab_dim == 0:
        return img_mempath, rows, img_dim

    # Create the new, combined memmap file
    combined_path = os.path.join(FEAT_DIR, f"combined_{out_name}.mmap")
    combined_dim = img_dim + tab_dim
    combined_fp = np.memmap(combined_path, dtype='float32', mode='w+', shape=(rows, combined_dim))
    
    # Copy data in batches to avoid high memory usage
    steps = math.ceil(rows / batch_size)
    for i in range(steps):
        start = i * batch_size
        end = min(rows, start + batch_size)
        
        # Write tabular batch
        combined_fp[start:end, :tab_dim] = tab_matrix[start:end]
        # Write image feature batch
        combined_fp[start:end, tab_dim:] = img_feats[start:end]
        
    combined_fp.flush() # Ensure data is written to disk
    return combined_path, rows, combined_dim


st.set_page_config(page_title="Hybrid AI Trainer (EffNetV2 + CatBoost)", layout="wide")
st.title("🚀 Hybrid AI Trainer (EfficientNetV2B0 + CatBoost)")

# -------------------------
# Sidebar params
# -------------------------
with st.sidebar:
    st.header("Training params")
    AUTO_START = st.checkbox("Auto-start training", value=False)
    IMG_SIZE = st.number_input("Image size (square)", min_value=64, max_value=512, value=224, step=32)
    TEST_SIZE = st.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    BATCH = st.number_input("Feature batch size", min_value=8, max_value=256, value=32, step=8)
    FEAT_DIR = st.text_input("Feature temp dir (optional)", value="")
    st.caption("EfficientNetV2B0 usually fine at 224x224.")
    st.header("Display")
    EXTRA_INFO = st.checkbox("Show data preview", value=True)

# -------------------------
# Data load
# -------------------------
st.header("1) Load dataset")
csv_file = st.file_uploader("Upload CSV (metadata) - required", type=["csv"])

# --- NEW: Handle large datasets by accepting a local path ---
st.subheader("Image Data Source")
st.info("For massive datasets (e.g., 100 GB), upload them to the server first and provide the local folder path. Browser uploads are not suitable for files larger than a few hundred MBs.")
data_source = st.radio(
    "How are you providing the images?",
    ('Use a local folder path on the server (for large datasets)', 'Upload a small ZIP file (<200MB)')
)

images_dir = None
tmpdir = tempfile.mkdtemp()

if FEAT_DIR:
    os.makedirs(FEAT_DIR, exist_ok=True)
else:
    FEAT_DIR = tmpdir

if data_source == 'Upload a small ZIP file (<200MB)':
    zip_file = st.file_uploader("Upload images ZIP", type=["zip"])
    if zip_file is not None:
        with st.spinner("Extracting ZIP..."):
            z = zipfile.ZipFile(io.BytesIO(zip_file.read()))
            z.extractall(tmpdir)
            extracted = list(Path(tmpdir).iterdir())
            # Handle cases where ZIP contains a single sub-folder
            if len(extracted) == 1 and extracted[0].is_dir():
                images_dir = str(extracted[0])
            else:
                images_dir = tmpdir
        st.success(f"ZIP extracted to temporary directory.")
else:
    images_dir_input = st.text_input("Enter the absolute path to the image folder on the server")
    if images_dir_input and Path(images_dir_input).is_dir():
        images_dir = images_dir_input
        st.success(f"Using image directory: {images_dir}")
    elif images_dir_input:
        st.error("The provided path is not a valid directory. Please check it.")

if csv_file is None:
    st.warning("Please upload your dataset CSV and provide the image source to continue.")
    st.stop()

df = pd.read_csv(csv_file)
st.success(f"CSV loaded. Rows: {len(df)} | Cols: {list(df.columns)}")

# -------------------------
# Columns expected
# -------------------------
image_col = "img_id"
target_col = "diagnostic"

if image_col not in df.columns or target_col not in df.columns:
    st.error(f"Expected columns not found. Need '{image_col}' for images and '{target_col}' for target.")
    st.stop()

# -------------------------
# Build image paths (stream-safe)
# -------------------------
def build_image_paths(df, image_col, images_dir):
    paths = []
    for v in df[image_col].astype(str).values:
        fname = Path(v).name
        if images_dir:
            candidate = Path(images_dir) / fname
            if candidate.exists():
                paths.append(str(candidate))
            else:
                paths.append("")  # missing
        else:
            if Path(v).exists():
                paths.append(str(Path(v)))
            else:
                paths.append("")
    return np.array(paths)

if images_dir is None:
    st.warning("Image source is not configured. Please upload a ZIP or provide a valid local path.")
    st.stop()

df["full_image_path"] = build_image_paths(df, image_col, images_dir)
df_proc = df[df["full_image_path"] != ""].reset_index(drop=True)

if df_proc.empty:
    st.error("No valid image paths found. Ensure the image folder path is correct or the ZIP contains the right images.")
    st.stop()

st.success(f"Working on {len(df_proc)} samples with valid images.")

# (The rest of the script remains the same until the prediction/evaluation sections)

# -------------------------
# Tabular preprocessing (simple)
# -------------------------
tab_cols = [c for c in df_proc.columns if c not in [image_col, target_col, "full_image_path"]]
tab_df = df_proc[tab_cols].copy()
X_tab = pd.DataFrame(index=tab_df.index)

for col in tab_df.columns:
    if tab_df[col].dtype == "object" or tab_df[col].dtype.name == "category":
        X_tab[col] = tab_df[col].astype("category").cat.codes.fillna(-1)
    elif np.issubdtype(tab_df[col].dtype, np.number):
        col_vals = tab_df[col].astype(float).fillna(tab_df[col].mean())
        scaler = StandardScaler()
        X_tab[col] = scaler.fit_transform(col_vals.values.reshape(-1, 1)).flatten()
    else:
        X_tab[col] = tab_df[col].astype("category").cat.codes.fillna(-1)

# -------------------------
# Target encode
# -------------------------
le = LabelEncoder()
y = le.fit_transform(df_proc[target_col].astype(str).values)
class_names = le.classes_.tolist()

if EXTRA_INFO:
    st.subheader("Data preview (first 5 rows)")
    st.dataframe(df_proc[[image_col, target_col, "full_image_path"] + tab_cols].head())

# -------------------------
# Train/test split
# -------------------------
st.write("Splitting data...")
X_img_paths = df_proc["full_image_path"].values
X_tab_values = X_tab.values if X_tab.shape[1] > 0 else np.zeros((len(df_proc), 0))

(X_img_paths_train, X_img_paths_test,
 X_tab_train, X_tab_test,
 y_train, y_test) = train_test_split(
    X_img_paths, X_tab_values, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

st.write(f"Training samples: {len(y_train)} — Test samples: {len(y_test)}")

# -------------------------
# Feature extractor (EfficientNetV2B0) cached
# -------------------------
@st.cache_resource
def get_feat_model(img_size):
    base = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base.input, outputs=x)

feat_model = get_feat_model(IMG_SIZE)

# -------------------------
# Streaming feature extraction: write features to disk in chunks
# -------------------------
def _load_and_preprocess(path, img_size):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    return preprocess_input(img)

def extract_and_save_features(paths, img_size, batch_size, out_prefix):
    saved_files = []
    n = len(paths)
    steps = math.ceil(n / batch_size)
    prog_bar = st.progress(0, text=f"Extracting features for {os.path.basename(out_prefix)}...")
    for i in range(steps):
        start = i * batch_size
        end = min(n, start + batch_size)
        batch_paths = paths[start:end]
        imgs = []
        for p in batch_paths:
            try:
                img = _load_and_preprocess(p, img_size)
                imgs.append(img.numpy())
            except Exception as e:
                imgs.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
        imgs_arr = np.stack(imgs, axis=0)
        feats = feat_model.predict(imgs_arr, verbose=0)
        fname = f"{out_prefix}_chunk_{i}.npy"
        np.save(fname, feats)
        saved_files.append(fname)
        prog_bar.progress((i + 1) / steps)
    prog_bar.empty()
    return saved_files


# -------------------------
# Training trigger
# -------------------------
start_training = st.button("Start training now") if not AUTO_START else True

if start_training:
    st.info("Extracting features for TRAIN set...")
    out_train_files = extract_and_save_features(X_img_paths_train, IMG_SIZE, BATCH, os.path.join(FEAT_DIR, "train_feats"))
    st.success(f"Saved {len(out_train_files)} train feature chunks.")
    
    st.info("Extracting features for TEST set...")
    out_test_files = extract_and_save_features(X_img_paths_test, IMG_SIZE, BATCH, os.path.join(FEAT_DIR, "test_feats"))
    st.success(f"Saved {len(out_test_files)} test feature chunks.")

    st.info("Building image feature memmaps...")
    train_img_mempath, train_img_rows, train_img_dim = load_features_memmap(out_train_files)
    test_img_mempath, test_img_rows, test_img_dim = load_features_memmap(out_test_files)
    st.success("Image features memmapped successfully.")

    st.info("Combining tabular and image features...")
    train_final_path, train_rows, final_dim = combine_features_memmap(
        train_img_mempath, (train_img_rows, train_img_dim), X_tab_train, "train"
    )
    test_final_path, test_rows, _ = combine_features_memmap(
        test_img_mempath, (test_img_rows, test_img_dim), X_tab_test, "test"
    )
    st.success(f"Final feature matrices created. Train rows: {train_rows}, Test rows: {test_rows}, Dimensions: {final_dim}")

    X_train = np.memmap(train_final_path, dtype='float32', mode='r', shape=(train_rows, final_dim))
    X_test = np.memmap(test_final_path, dtype='float32', mode='r', shape=(test_rows, final_dim))

    with st.spinner("Training CatBoost... (this may take time)"):
        cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, eval_metric="Accuracy",
                                 random_seed=42, early_stopping_rounds=30, verbose=50, task_type="CPU")
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)
        cat.fit(train_pool, eval_set=test_pool, use_best_model=True)

    st.success("Training finished.")

    y_pred = cat.predict(X_test).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    st.metric("Test accuracy", f"{acc:.4f}")

    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)),
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
    
    # ---------- MODIFIED UI: INSPECT PREDICTIONS WITH DETAILS ----------
    with st.expander("🔍 Inspect Test Set Predictions"):
        incorrect_indices = np.where(y_pred != y_test)[0]
        st.write(f"Found **{len(incorrect_indices)}** incorrect predictions out of {len(y_test)} test samples.")
        show_only_incorrect = st.checkbox("Show only incorrect predictions")

        indices_to_show = incorrect_indices if show_only_incorrect else np.arange(len(y_test))

        if len(indices_to_show) == 0:
            st.success("🎉 No incorrect predictions to show!")
        else:
            selected_idx_in_list = st.slider("Select a test sample to view:", 0, len(indices_to_show) - 1, 0)
            actual_idx = indices_to_show[selected_idx_in_list]
            img_path = X_img_paths_test[actual_idx]
            true_label = class_names[y_test[actual_idx]]
            predicted_label = class_names[y_pred[actual_idx]]
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_path, caption=f"Test Image #{actual_idx}", use_column_width=True)
            with col2:
                if true_label == predicted_label:
                    st.success(f"✔️ **Correct**")
                else:
                    st.error(f"❌ **Incorrect**")
                st.write(f"**Predicted Label:** `{predicted_label}`")
                st.write(f"**True Label:** `{true_label}`")
                
                # --- NEW: Display disease info ---
                st.markdown("---")
                predicted_info = DISEASE_INFO.get(predicted_label, DISEASE_INFO["Default"])
                st.subheader(f"About: {predicted_label}")
                st.info(predicted_info["description"])
                st.subheader("Suggested Precautions (Demonstration Only)")
                st.warning("🚨 This is not medical advice. Always consult a certified medical professional.")
                for p in predicted_info["precautions"]:
                    st.write(f"- {p}")

    model_path = Path(FEAT_DIR) / "catboost_model.cbm"
    cat.save_model(str(model_path))
    with open(model_path, "rb") as f:
        st.download_button("Download CatBoost model (.cbm)", data=f, file_name="trained_catboost.cbm")

    st.session_state["cat_model_path"] = str(model_path)
    st.session_state["class_names"] = class_names
    st.session_state["tab_cols"] = X_tab.columns.tolist()
    st.session_state["img_size"] = IMG_SIZE

# -------------------------
# Prediction UI
# -------------------------
st.header("2) Predict a single image (use trained model above)")

model_loaded = False
cat_model = None
if "cat_model_path" in st.session_state:
    try:
        cat_model = CatBoostClassifier()
        cat_model.load_model(st.session_state["cat_model_path"])
        model_loaded = True
    except Exception:
        model_loaded = False

uploaded_img = st.file_uploader("Upload image to predict (jpg/png)", type=["jpg", "jpeg", "png"])
select_row = st.selectbox("Or pick a row from CSV to predict using its image path", options=["-- none --"] + df_proc.index.astype(str).tolist())

def preprocess_bytes_and_feat(image_bytes, img_size):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    arr = preprocess_input(img.numpy())
    arr = np.expand_dims(arr, axis=0)
    feat = feat_model.predict(arr, verbose=0)
    return feat

if st.button("Load model from disk (if not loaded)"):
    model_path = Path(FEAT_DIR).joinpath("catboost_model.cbm")
    if model_path.exists():
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(model_path))
        model_loaded = True
        st.success("Model loaded.")
    else:
        st.error(f"No saved model found in feature dir: {FEAT_DIR}")

if model_loaded:
    st.write("Model ready for prediction.")
    if uploaded_img is not None or select_row != "-- none --":
        image_bytes = feat = None
        if uploaded_img is not None:
            image_bytes = uploaded_img.read()
        else:
            idx = int(select_row)
            path = df_proc.loc[idx, "full_image_path"]
            try:
                with open(path, "rb") as f: image_bytes = f.read()
            except Exception as e: st.error(f"Error reading image path {path}: {e}")

        if image_bytes:
            try: feat = preprocess_bytes_and_feat(image_bytes, st.session_state.get("img_size", IMG_SIZE))
            except Exception as e: st.error(f"Error processing image: {e}")
        
        if feat is not None:
            n_tab_cols = len(st.session_state.get("tab_cols", []))
            if n_tab_cols > 0:
                if select_row != "-- none --":
                    tab_row = X_tab.iloc[int(select_row)].values.reshape(1, -1)
                else:
                    tab_row = np.zeros((1, n_tab_cols), dtype='float32')
                final_vec = np.hstack((tab_row, feat))
            else:
                final_vec = feat

            probs = cat_model.predict_proba(final_vec)[0]
            pred_idx = int(cat_model.predict(final_vec)[0])
            class_names_loaded = st.session_state.get("class_names", [str(i) for i in range(len(probs))])
            predicted_label = class_names_loaded[pred_idx]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_bytes, caption="Input Image", width=300)
            with col2:
                prob_df = pd.DataFrame({"class": class_names_loaded, "probability": probs}).sort_values("probability", ascending=False).reset_index(drop=True)
                st.subheader("Prediction")
                st.write(f"Predicted label: **{predicted_label}**")
                st.dataframe(prob_df.head(5))

            # --- NEW: Display disease info for single prediction ---
            st.markdown("---")
            predicted_info = DISEASE_INFO.get(predicted_label, DISEASE_INFO["Default"])
            st.subheader(f"About: {predicted_label}")
            st.info(predicted_info["description"])
            st.subheader("Suggested Precautions (Demonstration Only)")
            st.warning("🚨 This is not medical advice. Always consult a certified medical professional.")
            for p in predicted_info["precautions"]:
                st.write(f"- {p}")

else:
    st.info("Train a model first (or load a saved model) to enable single-image prediction.")

st.caption("Feature files and temporary files are in: " + FEAT_DIR)