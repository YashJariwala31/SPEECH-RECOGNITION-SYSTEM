import sys
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_wav2vec2(audio_path):
    print("Loading model...")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        print("Resampling audio to 16kHz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    print("Transcribing...")
    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python speech_to_text.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcription = transcribe_wav2vec2(audio_file)
    print("\n🔊 Transcription:\n" + transcription)