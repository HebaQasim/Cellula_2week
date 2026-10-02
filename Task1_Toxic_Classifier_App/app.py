"""Task 2 app: classify a typed comment or an image (via its caption) and store every result in a CSV file."""
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Toxic Content Classifier", page_icon="🛡️")

st.title("🛡️ Toxic Content Classifier")
st.caption(
    "Write a comment or upload an image. An image is first described in words by BLIP, "
    "then the text is checked by the Task 1 BiLSTM."
)

# ---------- Loading: each spinner appears only while something is really loading ----------
with st.spinner("Starting the app... the first start can take about a minute.", show_time=True):
    from classifier import classify_text, load_classifier
    from database import load_records, save_record
    from imagecaption import generate_caption, load_captioner

with st.spinner("Loading the text classifier...", show_time=True):
    load_classifier()

with st.spinner("Loading the image captioning model (BLIP)...", show_time=True):
    load_captioner()

with st.sidebar:
    st.header("About")
    st.write("**Image captioning:** BLIP-1")
    st.write("**Text classification:** BiLSTM from Task 1")
    st.write("**Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate")
    st.write("**Database:** every input is saved in `database.csv`")


def show_result(output):
    """Show the final result and the probability of each label."""
    result = output["result"]
    if result == "non-toxic":
        st.success("Non-toxic", icon="✅")
    elif result == "no words to classify":
        st.warning("No words to classify", icon="⚠️")
    else:
        st.error(f"Toxic: {result}", icon="🚫")

    if output["probabilities"]:
        st.write("**Probability of each label** (🔴 = predicted)")
        for label, probability in output["probabilities"].items():
            marker = "🔴" if label in output["labels"] else "⚪"
            st.progress(probability, text=f"{marker} {label}: {probability:.0%}")


text_tab, image_tab, database_tab = st.tabs(["📝 Text", "🖼️ Image", "🗂️ Database"])

with text_tab:
    user_text = st.text_area(
        "Write a comment", placeholder="e.g. Thank you for fixing the article!", height=120
    )
    if st.button("Classify text", type="primary"):
        if user_text.strip() == "":
            st.warning("Please write some text first.")
        else:
            with st.spinner("Classifying the text..."):
                output = classify_text(user_text)
                save_record("text", user_text, output["result"])
            show_result(output)

with image_tab:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        if st.button("Classify image", type="primary"):
            with st.spinner("Describing the image...", show_time=True):
                caption = generate_caption(image)
            st.info(f"Caption: {caption}", icon="📝")
            with st.spinner("Classifying the caption..."):
                output = classify_text(caption)
                save_record("image", caption, output["result"])
            show_result(output)

with database_tab:
    records = load_records()
    st.metric("Stored inputs", len(records))
    if len(records) == 0:
        st.info("Nothing saved yet. Classify a text or an image first.")
    else:
        st.write("Newest first:")
        st.dataframe(records.iloc[::-1], hide_index=True)
