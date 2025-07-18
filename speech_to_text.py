import torch
import torchaudio
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

# Load pretrained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load and prepare audio
def load_audio(file_path):
    print("Loading audio:", file_path)

    # Load audio using pydub to support both mp3 and wav
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)           # Convert to mono
    audio = audio.set_frame_rate(16000)     # Resample to 16kHz

    # Export to a BytesIO object in WAV format
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    # Load as tensor using torchaudio
    waveform, sample_rate = torchaudio.load(buffer)
    return waveform.squeeze(), sample_rate

# Transcribe using Wav2Vec2
def transcribe_wav2vec2(file_path):
    waveform, sample_rate = load_audio(file_path)

    # Prepare input
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Example usage
if __name__ == "__main__":
    file_path = "harvard.wav"  # Replace with "your_audio.mp3" to test mp3
    transcription = transcribe_wav2vec2(file_path)
    print("Transcription:", transcription)
