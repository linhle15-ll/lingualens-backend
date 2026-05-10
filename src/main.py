# main.py
import sys
from models import Encoder, Decoder, Vocabulary
import io, json, pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from transformers import MarianMTModel, MarianTokenizer

from inference import generate_caption, beam_search_decode

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load everything ONCE at startup ──────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.json") as f:
    cfg = json.load(f)

# Force pickle to find Vocabulary in the right place
sys.modules['__main__'].Vocabulary = Vocabulary  # tells pickle where to look

with open("vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)

encoder = Encoder(cfg["ENCODED_SIZE"], cfg["EMBED_DIM"], cfg["INIT_DIM"], cfg["DROP_OUT"]).to(device)
encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
encoder.device = device
encoder.eval()

decoder = Decoder(cfg["EMBED_DIM"], cfg["DECODER_DIM"], cfg["VOCAB_SIZE"],
                  cfg["ENCODER_DIM"], cfg["ATTENTION_DIM"], cfg["DROP_OUT"]).to(device)
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
decoder.eval()

yolo = YOLO("yolov8n.pt")

hu_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hu")
hu_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Helper functions ──────────────────────────────────────────────────
def translate_to_hu(text: str) -> str:
    tokens = hu_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    out = hu_model.generate(**tokens, max_new_tokens=128)
    return hu_tokenizer.decode(out[0], skip_special_tokens=True)

# ── Endpoint ──────────────────────────────────────────────────────────
@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    # Read and decode the uploaded image
    img_bytes = await image.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Stage 1: Object detection
    det_result = yolo(img_pil, verbose=False)[0]
    seen = {}
    objects = []
    for box in det_result.boxes:
        label_en = det_result.names[int(box.cls[0])]
        conf = round(box.conf[0].item(), 3)
        b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        if label_en not in seen:
            seen[label_en] = translate_to_hu(label_en)
        objects.append({
            "label_en": label_en,
            "label_hu": seen[label_en],
            "confidence": conf,
            "box": b
        })

    # Stage 2: Caption generation
    img_tensor = transform(img_pil)
    caption_en, _ = generate_caption(
        encoder, decoder, img_tensor, vocabulary,
        max_len=cfg["SENTENCE_MAX_LEN"], beam_width=5, block_ngram_size=3
    )

    # Stage 3: Translate caption
    caption_hu = translate_to_hu(caption_en)

    return {
        "objects": objects,
        "caption_en": caption_en,
        "caption_hu": caption_hu
    }

@app.get("/health")
def health():
    return {"status": "ok"}