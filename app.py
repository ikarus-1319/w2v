from flask import Flask, request, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import logging
import os
from pydub import AudioSegment
from pydub.utils import mediainfo

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

processor = Wav2Vec2Processor.from_pretrained("/home/j3s/HieuPT/wav2vec/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("/home/j3s/HieuPT/wav2vec/wav2vec2-base-vietnamese-250h")

def map_to_array(batch):
    speech, samplerate = sf.read(batch["file"])
    batch["speech"] = speech
    return batch, samplerate

def check_sample_rate(file):
    info = mediainfo(file)
    sample_rate = int(info['sample_rate'])
    if sample_rate != 16000:
        print(f"Detected sample rate: {sample_rate}. Changing to 16000...")
        converted_file = os.path.splitext(file)[0] + '_converted.wav'
        audio = AudioSegment.from_file(file)
        audio = audio.set_frame_rate(16000)
        audio.export(converted_file, format='wav')
        return converted_file
    return file

@app.route('/wav-to-txt', methods=['POST'])
def wav_to_txt():
    file = request.form.get('file')

    file_format = os.path.splitext(file)[1][1:].lower()  # Get the file extension in lowercase
    if file_format != 'wav':
        converted_file = os.path.splitext(file)[0] + '.wav'  # Change file extension to '.wav'
        audio = AudioSegment.from_file(file, format=file_format)
        audio.export(converted_file, format='wav')
        file = converted_file

    file = check_sample_rate(file)

    ds, samplerate = map_to_array({"file": file})
    input_values = processor(ds["speech"], return_tensors="pt", padding="longest", sampling_rate=samplerate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    save_directory = "/home/j3s/HieuPT/wav2vec/audio-api"  
    file_name = os.path.basename(file)
    transcription_file_name = os.path.splitext(file_name)[0] + ".txt"
    save_path_wav = os.path.join(save_directory, file_name)
    sf.write(save_path_wav, ds["speech"], samplerate)
    save_path_txt = os.path.join(save_directory, transcription_file_name)
    with open(save_path_txt, 'w') as f:
        f.write(transcription)

    response = {'transcription': transcription}
    return jsonify(response)

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.DEBUG)
    app.run(host='192.168.122.101', port=5000, debug=True)
    