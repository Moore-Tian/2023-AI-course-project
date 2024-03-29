import torch
import os
from PIL import Image
import numpy as np
from torch_CNN import CNN_dropout

# load network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
i = 2
model_path = f"./{i}CNN_model.pth"
model = CNN_dropout()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.to(device)
model.eval()

# image params
root_dir = "test_data/"

# label mapping
total_cnt = 0
correct_cnt = 0
with torch.no_grad():
    for label in range(12):
        for i in range(240):
            image_path = os.path.join(root_dir, str(label + 1), f"{i + 1}.bmp")
            image = Image.open(image_path)
            image_size = image.size
            image = np.where(np.array(image) > 0, 0, 1)
            image = torch.tensor(image).reshape(1, 1, image_size[0], image_size[1])
            image = image.to(device)
            image = image.to(torch.float)
            predict = model(image)
            total_cnt += 1
            if predict.argmax(dim=1)[0].item() == label:
                correct_cnt += 1
    print(f"correct = {correct_cnt}, total_cnt = {total_cnt}, accuracy = {correct_cnt / total_cnt}")
