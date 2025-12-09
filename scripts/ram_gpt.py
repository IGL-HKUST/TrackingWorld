import argparse
import glob
import torch
import os
from PIL import Image
import json
from dotenv import load_dotenv

import torchvision.transforms as TS

from ram.models import ram
from ram import inference_ram as inference

from openai import OpenAI
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def get_chatgpt_response(prompt):
    try:
        response = client.chat.completions.create(
            # messages=[
            #     {"role": "system", "content": "You are a data preprocessing assistant that will filter out key words from a list of words. \
            #      You will be given a list of words and you will need to filter out words that belong to living things that are able to move \
            #      by itself. Make sure that you go over every single word and explain if they can move and is living, explaining your thought process. \
            #      Return your response as only a json, with the reasoning for each word under the key 'reasoning' and the words that are living and can move under the key 'dynamic'."},
            #     {"role": "user", "content": f"{prompt}"}
            # ],
            messages=[
                {"role": "system", "content": "You are a data preprocessing assistant that will filter out key words from a list of words. \
                 You will be given a list of words and you will need to filter out words that are nouns are is able to move by itself. \
                 Make sure that you go over every single word and explain if the word is a noun, and whether it is able to move. Explain your thought process. \
                 Return your response as only a json, with the reasoning for each word under the key 'reasoning' and the words that are nouns and can move by itself under the key 'dynamic'."},
                {"role": "user", "content": f"{prompt}"}
            ],
            model="gpt-4o",
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(str(e))
        return str(e)


parser = argparse.ArgumentParser(
    description='Tagging images with RAM+GPT model')
parser.add_argument('--workdir',
                    metavar='DIR',
                    help='path to inputs',
                    default="sintel")
parser.add_argument('--output-dir',
                    metavar='DIR',
                    help='path to save outputs',
                    default="ram")
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='./preprocess/pretrained/ram_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')

if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])

    #######load model
    model = ram(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    dataset_path = args.workdir
    videos = sorted(os.listdir(dataset_path))

    #use glob to read all img files in the folder that ends with .png or .jpg
    for video in tqdm(videos):

        img_dir = os.path.join(dataset_path, video, 'rgb')

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))

        all_tags = set()

        for img_file in img_files:
            
            image_pil = Image.open(img_file).convert("RGB")
    
            raw_image = image_pil.resize(
                            (384, 384))
            raw_image  = transform(raw_image).unsqueeze(0).to(device)

            res = inference(raw_image , model)

            all_tags.update(set(res[0].split(' | ')))
        print(all_tags)
        response = get_chatgpt_response(f"{list(all_tags)}")
        # print(response)
        response['input'] = list(all_tags)
        response['dynamic'] = list(set(response['dynamic']))               # remove duplicates

        os.makedirs(os.path.join(dataset_path, video, args.output_dir), exist_ok=True)
        json.dump(response, open(os.path.join(dataset_path, video, args.output_dir, 'tags.json'), 'w'), indent=4)


