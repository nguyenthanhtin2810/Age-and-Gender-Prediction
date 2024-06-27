from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd

def NumStr_to_NpArr(number_string):
    numbers = np.array([int(num) for num in number_string.split(" ")]).astype(np.uint8)
    size = int(np.sqrt(len(numbers)))
    numpy_array = numbers.reshape((size, -1)) # 48x48
    return numpy_array

class AgeGenderDataset(Dataset):
    def __init__(self, train=True, transform=None, gray=True):
        self.transform = transform
        self.gray = gray
        self.name_genders = ['male', 'female']
        data = pd.read_csv("age_gender.csv")
        data.drop(["ethnicity", "img_name"], axis=1, inplace=True)
        train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)
        if train:
            self.data = train_data
        else:
            self.data = test_data


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        numstr = self.data["pixels"].iloc[index]
        nparr = NumStr_to_NpArr(numstr)
        if self.gray:
            image = Image.fromarray(nparr)
        else:
            image = Image.fromarray(nparr).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age_label = self.data["age"].iloc[index]
        gender_label = self.data["gender"].iloc[index]
        return image, age_label, gender_label

if __name__ == '__main__':
    dataset = AgeGenderDataset(train=True)
    image, age_label, gender_label = dataset.__getitem__(2462)
    image.show()
    print(age_label, gender_label, image.size)
