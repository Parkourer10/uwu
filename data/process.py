"""
this script downloads openwebtext book corpus and daily dialog and tokenizes it. 
it makes the openwebtext 1/3 btw

"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, concatenate_datasets

num_proc = 16 #num of workers
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def get_datasets():
    return [
        "openwebtext",
        "bookcorpus",
        "daily_dialog",
        ("Salesforce/wikitext", "wikitext-103-raw-v1") 
    ]

def load_and_prepare_dataset(dataset_name, config=None, fraction=1.0):
    if isinstance(dataset_name, tuple):
        dataset_name, config = dataset_name

    print(f"Loading dataset: {dataset_name} ({config if config else 'default config'})")
    dataset = load_dataset(dataset_name, config, trust_remote_code=True)

    if fraction < 1.0:
        print(f"Reducing {dataset_name} to {fraction * 100:.1f}% of its size.")
        dataset["train"] = dataset["train"].shard(num_shards=int(1 / fraction), index=0)

    if "train" in dataset:
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
    elif "validation" in dataset and "train" in dataset:
        split_dataset = dataset.rename_column("validation", "val")
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a 'train' split!")

    return split_dataset

def process(example):
    text_column = 'text' if 'text' in example else list(example.keys())[0]
    ids = enc.encode_ordinary(example[text_column])  
    ids.append(enc.eot_token)  
    return {'ids': ids, 'len': len(ids)}

def validate_tokenization(dataset, eos_token): #this is so bad ;/
    missing_eos_count = 0
    for example in dataset:
        if eos_token not in example['ids']:
            missing_eos_count += 1
    if missing_eos_count > 0:
        print(f"Warning: {missing_eos_count} examples are missing EOS tokens.")

def write_to_file(datasets, filename, total_tokens):
    dtype = np.uint16  # GPT-2 tokens fit in uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))
    idx = 0

    for dataset_name, dset in datasets.items():
        print(f"Writing {dataset_name} data to {filename}...")
        total_batches = 1024

        for batch_idx in tqdm(range(total_batches), desc=f'Writing {dataset_name}'):
            # Get a shard of the dataset
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

    arr.flush()
    print(f"Saved  dataset to {filename}.")

if __name__ == '__main__':
    # Load all datasets and split into train/validation
    all_datasets = get_datasets()
    combined_train = []
    combined_val = []

    for dataset in all_datasets:
        try:
            fraction = 1.0  # Default to using the full dataset

            # Reduce OpenWebText to one-third
            if dataset == "openwebtext":
                fraction = 1 / 3

            if isinstance(dataset, tuple):
                dataset_name, config = dataset
                split_dataset = load_and_prepare_dataset(dataset_name, config, fraction)
            else:
                split_dataset = load_and_prepare_dataset(dataset, None, fraction)

            tokenized = split_dataset.map(
                process,
                remove_columns=['text'] if 'text' in split_dataset['train'].column_names else None,
                desc=f"Tokenizing {dataset}",
                num_proc=num_proc,
            )

            # Validate tokenization for EOS tokens
            validate_tokenization(tokenized['train'], enc.eot_token)
            validate_tokenization(tokenized['val'], enc.eot_token)

            combined_train.append(tokenized['train'])
            combined_val.append(tokenized['val'])
        except Exception as e:
            print(f"Failed to process dataset {dataset}: {e}")

    # Concatenate all train/validation splits
    combined_train_dataset = concatenate_datasets(combined_train)
    combined_val_dataset = concatenate_datasets(combined_val)

    # Calculate total token counts
    train_tokens = sum(combined_train_dataset['len'])
    val_tokens = sum(combined_val_dataset['len'])
    print(f"Total tokens in train dataset: {train_tokens}")
    print(f"Total tokens in validation dataset: {val_tokens}")

    # Write datasets to binary files
    write_to_file({'train': combined_train_dataset}, 'train.bin', train_tokens)
    write_to_file({'val': combined_val_dataset}, 'val.bin', val_tokens)
