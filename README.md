# 50.039 Theory and Practice of Deep Learning Project

#### Group Members
Lu Jiankun (10029590), Peng Shanshan (1002974),
Tan Jin Yan (1002722), Wei Letong (1002965)

## Usage 

#### 1. Preprocessing

```bash
put glove.6B.300d.txt (downloaded from https://nlp.stanford.edu/projects/glove/) in the same directory as train.py
python build_vocab.py --caption_path path_to_COCO_caption_file
python resize.py --image_dir path_to_COCO_training_set_images
```

#### 2. Train the model

```bash
python train.py    
```

#### 3. Run the GUI to test the model 

```bash
python demonstrator.py
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/sh/jsdcgn33vhnjayo/AACItONej3cy1cHkOin8Bnyha?dl=0). You put the model files in `./models/`.
