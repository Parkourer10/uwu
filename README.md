fork of nanoGPT to train a 60m model on 1/3 of openwebtext , bookcorpus , wikitext and dailydialog
idk why am i doing this



## to train:
```bash
  python train.py
```

## to process data:
```bash
  python data/process.py
```

## to download the alr tokenized dataset:
```bash
  python data/download.py
```

## to continue training:
```bash
  python train.py config/continue.py
```

sorry.