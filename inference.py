from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

import warnings
warnings.filterwarnings("ignore")

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("/home/j3s/HieuPT/wav2vec/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("/home/j3s/HieuPT/wav2vec/wav2vec2-base-vietnamese-250h")

# define function to read in sound file
def map_to_array(batch):
    speech, samplerate = sf.read(batch["file"])
    batch["speech"] = speech
    return batch, samplerate

# load dummy dataset and read soundfiles
ds, samplerate = map_to_array({
    "file": '/home/j3s/HieuPT/wav2vec/wav2vec2-base-vietnamese-250h/audio-test/t1_0001-00010.wav'
})

# tokenize
input_values = processor(ds["speech"], return_tensors="pt", padding="longest", sampling_rate=samplerate).input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)