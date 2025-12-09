import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# -------- Page config --------
st.set_page_config(page_title="Sentiment Analysis Demo", page_icon="🎬")

# -------- Device (CPU / GPU) --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    """Load tokenizer + model one time and cache them."""
    model_id = "Haroun26/streamlit-demo"   # <- Hugging Face repo id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# -------- UI --------
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Type a movie review and the model will predict "
         "**Positive** or **Negative** sentiment.")

text = st.text_area(
    "✏️ Enter a review:",
    "This movie was absolutely fantastic, I loved it!",
    height=150,
)

if st.button("Analyze sentiment") and text.strip():
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    pred = probs.argmax(dim=-1).item()
    confidence = probs[0, pred].item()

    # Display result
    if pred == 1:
        st.success(f"✅ Positive sentiment ({confidence:.2f})")
    else:
        st.error(f"❌ Negative sentiment ({confidence:.2f})")
