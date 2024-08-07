from argparse import ArgumentParser
from model import AgeGenderCNN, AgeGenderResNet50
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def get_args():
    parser = ArgumentParser(description="Testing")
    parser.add_argument("--image-path", "-p", type=str, default=None)
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_custom_model.pt")
    parser.add_argument("--typemodel", "-tm", type=bool, default=0, help="0: Custom model, 1: finetune resnet50")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.typemodel:
        model = AgeGenderResNet50()
    else:
        model = AgeGenderCNN()
        
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        accuracy = checkpoint["cls_accuracy"]
        r2 = checkpoint["reg_r2"]
        print(f"Regression - R-squared: {r2}")
        print(f"Classification - Accuracy: {accuracy}")
    else:
        print("No checkpoint found!")
        exit(0)

    model.eval()
    ori_image = Image.open(args.imagepath)
    if args.typemodel:
        image = ori_image.convert('L').convert('RGB')
        transform = Compose([
            ToTensor(),
            Resize((224, 224), antialias=True)
        ])
    else:
        image = ori_image.convert('L')
        transform = Compose([
            ToTensor(),
            Resize((48, 48), antialias=True)
        ])
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        reg_output, cls_output = model(image)
    predicted_age = round(reg_output.item())
    predicted_gender = "Male" if cls_output.argmax().item() == 0 else "Female"

    width, height = 600, 700
    ori_image = ori_image.resize((width, height))
    draw = ImageDraw.Draw(ori_image)

    width, height = ori_image.size
    x, y = 0, 0

    text = f"Age: {predicted_age}. Gender: {predicted_gender}"
    text_color = (255, 255, 255)

    font_size = 25
    font = ImageFont.truetype("C:\Windows\Fonts/arial.ttf", font_size)

    x, y, text_width, text_height = draw.textbbox((x, y), text, font=font)
    background_for_text = (x, y, x + text_width, y + text_height)
    background_color = (0, 100, 0)
    draw.rectangle(background_for_text, fill=background_color)

    draw.text((x, y), text=text, font=font, fill=text_color)

    # image_filename = args.imagepath.split("\\")[2]
    # ori_image.save(f"test_image/predicted_{image_filename}")

    ori_image.show()
