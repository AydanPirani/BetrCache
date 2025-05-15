from bs4 import BeautifulSoup
import os
import random

def get_dataset(folder = "flickr30k-images", captions_file = "flickr30k_captions.txt", num_return = 100):
    images = os.listdir(folder)

    dataset = {}
    with open(captions_file, "r") as file:
        text = file.read()

    soup = BeautifulSoup(text, 'html.parser')

    random_indexes = random.sample(range(len(images)), num_return)

    for i in random_indexes:
        img = images[i]
        img_tag = soup.find('a', href=img)
        if img_tag is None:
            continue

        # Find the <ul> following the image link
        ul_tag = img_tag.find_next('ul')
        if ul_tag is None:
            continue

        # Extract the captions from <li> tags
        captions = [li.get_text(strip=True) for li in ul_tag.find_all('li')]

        dataset[img] = captions

        # if captions:
        #     for i, caption in enumerate(captions, 1):
        #         print(f"{i}. {caption}")
        # else:
        #     print("Image or captions not found.")
    return dataset