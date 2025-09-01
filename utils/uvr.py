import librosa
import soundfile as sf
from vocal import get_model, separate_vocal

device = "cuda" # or cpu
audio, sr = librosa.load("/home/lhy/audio/ase/LR-ASD/demo/002/pyavi/audio.wav", sr=44100, mono=False)
model = get_model(device) # download model from HF
audio_data = separate_vocal(model, audio, device, silent=False)
sf.write("vocal1.wav", format="WAV", data=audio_data.T, samplerate=sr)