import torch
import torchaudio

model_file = "/home/lhy/audio/ase/LR-ASD/pretrained_models/ECAPA2/ecapa2.pt"


ecapa2 = torch.jit.load(model_file, map_location='cuda')
ecapa2.half()
audio, sr = torchaudio.load('/home/lhy/audio/ase/LR-ASD/demo/102/pycrop/00003.wav') # sample rate of 16 kHz expected
audio = audio.to(device="cuda")
with torch.jit.optimized_execution(False):
  embedding = ecapa2(audio)
  
  
print(embedding.shape)


# import torch
# import torchaudio

# ecapa2 = torch.jit.load(model_file, map_location='cuda')
# ecapa2.half() # optional, but results in faster inference
# audio, sr = torchaudio.load('sample.wav') # sample rate of 16 kHz expected

# embedding = ecapa2(audio)
