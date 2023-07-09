from datasets import Dataset, Audio
from argparse import ArgumentParser
from moviepy.editor import *
import os
import glob
import xmltodict
import json

def mp4_to_mp3(mp4: str, mp3: str) -> None:
    mp4_without_frames = AudioFileClip(mp4)
    mp4_without_frames.write_audiofile(mp3)
    mp4_without_frames.close()

def process_elan() -> dict:
    i = 1
    audio_trans = dict()
    cwd = os.getcwd()
    while os.path.exists(str(i)):
        print("Processing ELAN file No.", str(i))
        # Annotation
        eafpath = os.path.join(cwd, "{}/{}.eaf".format(str(i), str(i)))
        with open(eafpath, "r") as f:
            xml = f.read()
        dct = xmltodict.parse(xml)
        trans = dct["ANNOTATION_DOCUMENT"]["TIER"]["ANNOTATION"]["ALIGNABLE_ANNOTATION"]["ANNOTATION_VALUE"]

        # Audio
        if len(glob.glob(os.path.join(cwd, "{}/*.mp3".format(str(i))))) == 1:
            print("mp3 already found. Skipping...")
            mp3path = glob.glob(os.path.join(cwd, "{}/*.mp3".format(str(i))))[0]
        else:
            mp4path = glob.glob(os.path.join(cwd, "{}/*.mp4".format(str(i))))[0]
            mp3path = mp4path[:-3] + "mp3"
            mp4_to_mp3(mp4path, mp3path)

        audio_trans[str(i)] = {"audio_path": mp3path, "transcription": trans}
        i += 1
    return audio_trans

def process_audio(audio_trans: dict) -> Dataset:
    audio_list = []
    trans_list = []
    for v in audio_trans.values():
        audio_list.append(v["audio_path"])
        trans_list.append(v["transcription"])

    print("Audio list:", audio_list)
    audio_dataset = Dataset.from_dict({"audio": audio_list}).cast_column("audio",
                                                                         Audio(sampling_rate=16000))
    print(audio_dataset[0])
    audio_dataset = audio_dataset.add_column("sentence", trans_list)
    return audio_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str,
                        default="KichwaAudio.json",
                        help="Dataset name of Kichwa audio with transcript.")
    args = parser.parse_args()

    audio_trans = process_elan()
    print("ELAN files processed")
    audio_dataset = process_audio(audio_trans)
    print("Audio files processed")
    audio_dataset.to_json(args.output)
    print("Dataset created")
    
