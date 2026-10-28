from transformers import pipeline
from PIL import Image
import easyocr

# Try to load your friend's text model.
# If it isn't ready yet, use a temporary placeholder so nothing crashes.
try:
    from text_model import score_text
except ImportError:
    print("WARNING: text_model.py not found, text score will always be 0")
    def score_text(text):
        return 0.0

# Load the image model (explicit pictures)
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# Load the OCR reader (reads words inside pictures)
reader = easyocr.Reader(["en"], gpu=False)


# Check 1: is the picture itself explicit? Returns 0 to 1
def get_image_score(path):
    image = Image.open(path).convert("RGB")
    results = classifier(image)

    nsfw_score = 0.0
    for item in results:
        if item["label"] == "nsfw":
            nsfw_score = item["score"]
    return nsfw_score


# Check 2a: read the words inside a picture
def extract_text(path):
    pieces = reader.readtext(path, detail=0)
    return " ".join(pieces)


# Run both checks and combine the results
def analyze_image(path):
    nsfw_score = get_image_score(path)

    text = extract_text(path)
    text_score = 0.0
    if text.strip() != "":
        text_score = score_text(text)

    flagged = False
    if nsfw_score > 0.5:
        flagged = True
    if text_score > 0.5:
        flagged = True

    return {
        "nsfw_score": nsfw_score,
        "text_found": text,
        "text_score": text_score,
        "flagged": flagged
    }


# Old function kept so nothing else breaks
def check_image(path):
    return analyze_image(path)["flagged"]


# Lets you test from the terminal
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    result = analyze_image(path)
    print("nsfw score:", result["nsfw_score"])
    print("text found:", result["text_found"])
    print("text score:", result["text_score"])
    print("flagged:", result["flagged"])
