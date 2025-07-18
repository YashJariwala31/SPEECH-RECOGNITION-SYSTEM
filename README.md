# 🧠 Wav2Vec2 Speech-to-Text Transcriber

This project converts speech audio files into text using Facebook AI's Wav2Vec2 model via HuggingFace. It supports both `.wav` and `.mp3` audio formats and runs entirely in the terminal without any GUI.

---

## ✅ Features

- Supports `.wav` and `.mp3` files
- Uses Facebook’s pretrained `wav2vec2-base-960h` model
- CLI-based, no GUI required
- Converts `.mp3` to `.wav` internally using `pydub`

---

## 🗂️ File Structure

```
speech_to_text_project/
├── harvard.wav               # Sample audio file
├── speech_to_text.py         # Main transcription script
└── README.md                 # This file
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/speech-to-text-wav2vec2.git
cd speech-to-text-wav2vec2
```

### 2. (Optional) Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For macOS/Linux
```

### 3. Install Dependencies

```bash
pip install transformers torchaudio pydub
```

---

## 🔧 MP3 Support – Install FFmpeg

### For Windows:

1. Download FFmpeg from: https://ffmpeg.org/download.html  
2. Extract the ZIP and copy the path to the `bin` folder (e.g., `C:\ffmpeg\bin`)
3. Add that path to your system `PATH`:
   - Search "Environment Variables" in Windows Search
   - Click **Environment Variables**
   - Under "System Variables" → select `Path` → click **Edit** → **New**
   - Paste the path to the `bin` folder
   - Click OK and restart your terminal

---

## 🚀 Usage

```bash
python speech_to_text.py path/to/audio.wav
```

Or with an MP3:

```bash
python speech_to_text.py path/to/audio.mp3
```

---

## 🔊 Example Output

```bash
🔊 Transcription:
THE STALE SMELL OF OLD BEER LINGERS.
```

---

## 📌 Notes

- The model works best with audio sampled at **16 kHz**
- `.mp3` files are automatically converted to `.wav` for processing
- Tested on Python 3.9+

---

## 🧠 Model Info

- [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- Provided by Hugging Face Transformers

---

## 📄 License

This project is licensed under the MIT License.