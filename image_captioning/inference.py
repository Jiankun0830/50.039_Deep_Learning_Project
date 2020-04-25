import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import json


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size).eval().to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab, args.num_layers).to(device)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
        
    # Train the models
    total_step = len(data_loader)
    output = []
    ids = []
    for i, (images, captions, lengths, img_ids) in enumerate(data_loader):
        img_id = img_ids[0]
        if img_id in ids: continue
        # Set mini-batch dataset
        images = images.to(device)
#         import pdb; pdb.set_trace()
        
        # Forward, backward and optimize
        feature = encoder(images)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption[1:-1])
        ids.append(img_id)
        output.append({"image_id": img_id, "caption":sentence})
        if i % 1000 == 0:
            print('Step [{}/{}]'.format(i, total_step))
#             with open('inference_result_{}.json'.format(i), 'w') as fout:
#                 json.dump(output , fout)
#         print(output[-1])
    
    with open('inference_result.json', 'w') as fout:
        json.dump(output , fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/val2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.ckpt', help='path for trained decoder')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)