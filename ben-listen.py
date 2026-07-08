import argparse
import time

import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T


CLASS_NAMES = ["ben", "yes", "no", "hohoho", "eughhh", "none"]
TARGET_SAMPLE_RATE = 16000
CLIP_DURATION_SECONDS = 2.5
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_DURATION_SECONDS)
N_MELS = 64
MODEL_PATH = "ben_yees.pt"

device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def to_mono(waveform):
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def pad_or_trim(waveform, target_num_samples):
    current_num_samples = waveform.shape[1]
    if current_num_samples == target_num_samples:
        return waveform
    if current_num_samples > target_num_samples:
        return waveform[:, :target_num_samples]

    padded = torch.zeros((waveform.shape[0], target_num_samples), dtype=waveform.dtype)
    padded[:, :current_num_samples] = waveform
    return padded


class TinyBenCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path, device_name):
    model = TinyBenCNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device_name)
    model.load_state_dict(state_dict)
    model.to(device_name)
    model.eval()
    return model


def preprocess_waveform(waveform, sample_rate):
    waveform = to_mono(waveform)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = F.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = pad_or_trim(waveform, TARGET_NUM_SAMPLES)

    mel_transform = T.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_mels=N_MELS)
    db_transform = T.AmplitudeToDB()
    mel_spec_db = db_transform(mel_transform(waveform))
    return mel_spec_db.unsqueeze(0)


def record_clip(duration_seconds, sample_rate):
    print()
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    start_time = time.time()
    update_interval_seconds = 0.1

    while True:
        elapsed_seconds = time.time() - start_time
        remaining_seconds = max(0.0, duration_seconds - elapsed_seconds)
        print(f"Listening... {remaining_seconds:.1f}s ", end="\r", flush=True)

        if remaining_seconds <= 0.0:
            break

        time.sleep(update_interval_seconds)

    sd.wait()
    print("Listening... done     ")
    return torch.from_numpy(audio.T)


def predict_waveform(model, waveform, sample_rate, device_name):
    x = preprocess_waveform(waveform, sample_rate).to(device_name)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(probs.argmax().item())
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())
    return pred_name, confidence, probs


def main():
    parser = argparse.ArgumentParser(description="Listen on microphone and classify every 2.5 s clip.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the saved model weights")
    args = parser.parse_args()

    model = load_model(args.model, device)

    print(f"Loaded model from {args.model}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            waveform = record_clip(CLIP_DURATION_SECONDS, TARGET_SAMPLE_RATE)
            pred_name, confidence, _ = predict_waveform(model, waveform, TARGET_SAMPLE_RATE, device)
            print(f"Heard: {pred_name} ({confidence:.3f})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped listening.")


if __name__ == "__main__":
    main()
