from datasets import load_dataset, Dataset, Audio, concatenate_datasets
import pandas as pd
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import time
from argparse import ArgumentParser

# Local
import create_dataset

def load():
    audio_trans = create_dataset.process_elan()
    audio_dataset = create_dataset.process_audio(audio_trans)
    train_test = audio_dataset.train_test_split(test=0.2)
    train = train_test["train"]
    valid = train_test["test"]
    return train, valid

def extract_chars(batch: dict) -> dict:
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab],
            "all_text": [all_text]}

def create_vocab(dataset: Dataset) -> Dataset:
    vocab = dataset.map(
        extract_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names)
    return vocab

def remove_long_data(dataset: Dataset, max_seconds=15) -> Dataset:
    df = dataset.to_pandas()
    df["length"] = df["input_values"].apply(len)
    maxlen = max_seconds * 16000
    df = df[df["length"] < maxlen]
    df = df.drop("length", 1)
    dataset = Dataset.from_pandas(df)
    # Don't wait for gc
    del df
    return dataset

def preprocess(dataset: Dataset, num_proc=24) -> Dataset:
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )
    return dataset

def prepare_dataset(batch: dict) -> dict:
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["ipa"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

if __name__ == "__main__":
    parser = ArgumentParser(description="Kichwa ASR.")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--output", type=str, default="kichwaasr")
    parser.add_argument("--vocab", type=str, default="vocab.txt")
    args = parser.parse_args()
    
    # Load dataset
    start = time.time()
    train, valid = load()
    end = time.time()
    print("Time for loading data:", end - start)

    # Shuffle
    train = train.shuffle(seed=42)
    valid = valid.shuffle(seed=42)
    print("Dataset shuffled")

    # Create vocab
    print("Creating the vocabulary file and the tokenizer...")
    vocab_train = create_vocab(train)
    vocab_valid = create_vocab(valid)
    vocab_list = list(
        set(vocab_train["vocab"][0]) | set(vocab_valid["vocab"][0])
        )
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # Add special characters
    vocab_dict_ipa["[UNK]"] = len(vocab_dict)
    vocab_dict_ipa["[PAD]"] = len(vocab_dict)

    # Create the vocab file
    with open(args.vocab, "w") as f:
        json.dump(vocab_dict, f)
    print("Vocabulary created")

    # Tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(args.vocab,
                                     unk_token="[UNK]",
                                     pad_token="[PAD]",
                                     word_delimiter_token=" ")

    print("Defining the feature extractor...")
    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)

    print("Defining the processor...")
    # Define the processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    print("Preprocessing the data...")
    # Preprocess the dataset
    train = preprocess(train)
    valid = preprocess(valid)

    print("Removing long audio files...")
    # Remove long data
    train = remove_long_data(train, max_seconds=15)
    valid = remove_long_data(valid, max_seconds=15)
    
    # print("Training/validation data stats:")
    # print(len(train))
    # print(len(valid))
    # with open("data_stats.txt", "w") as f:
    #     f.write("Train: {}\nValid: {}".format(len(train), len(valid)))

    # data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )
    model.freeze_feature_extractor()

    # Output
    training_args = TrainingArguments(
        output_dir=args.output,
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.epoch,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=20,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor.feature_extractor,
        )
    
    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()
