"""Image captioning for Task 2: describes an image in one short English sentence (BLIP-1)."""
from functools import lru_cache
from pathlib import Path

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 30  # the caption can be at most 30 tokens long


@lru_cache(maxsize=1)
def load_captioner():
    """Load BLIP's processor and model the first time; later calls reuse them."""
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    return processor, model


def generate_caption(image):
    """Take an image opened with PIL and return a short caption, e.g. 'a dog running on the beach'."""
    processor, model = load_captioner()

    image = image.convert("RGB")                                            # BLIP needs 3 colour channels
    inputs = processor(images=image, return_tensors="pt")                   # resize + turn pixels into numbers
    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)    # write the caption word by word
    caption = processor.decode(output_ids[0], skip_special_tokens=True)     # numbers -> words
    return caption.strip()


if __name__ == "__main__":
    # Runs only with "python imagecaption.py": put any photo named test_image.jpg next to this file
    test_image = Image.open(Path(__file__).parent / "test_image.jpg")
    print("Caption:", generate_caption(test_image))
