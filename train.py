import torch
from dataset import AgeGenderDataset
from model import AgeGenderCNN, AgeGenderResNet50
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, r2_score
from torchvision.transforms import Compose, ToTensor, RandomAffine
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def get_args():
    parser = ArgumentParser(description="Training")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--typemodel", "-tm", type=bool, default=0, help="0: Custom model, 1: finetune resnet50")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.typemodel:
        train_transform = Compose([
            Resize((224, 224)),
            RandomAffine(
                degrees=(-5, 5),
                translate=(0.05, 0.05),
                scale=(0.8, 1.1),
                shear=0.8
            ),
            ToTensor(),
        ])
        test_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
        ])
        gray = False
        typemodel = 'resnet50'
    else:
        train_transform = Compose([
            RandomAffine(
                degrees=(-5, 5),
                translate=(0.05, 0.05),
                scale=(0.8, 1.1),
                shear=0.8
            ),
            ToTensor(),
        ])
        test_transform = ToTensor()
        gray = True
        typemodel = 'custom'

    train_dataset = AgeGenderDataset(train=True, transform=train_transform, gray=gray)

    test_dataset = AgeGenderDataset(train=False, transform=test_transform, gray=gray)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)
    if args.typemodel:
        model = AgeGenderResNet50().to(device)
    else:
        model = AgeGenderCNN().to(device)

    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_score = 0

    iters = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progess_bar = tqdm(train_dataloader, colour="green")
        for iter, (image_batch, age_batch, gender_batch) in enumerate(progess_bar):
            image_batch = image_batch.to(device)
            age_batch = age_batch.view(-1, 1).float().to(device)
            gender_batch = gender_batch.to(device)
            # forword
            reg_outputs, cls_outputs = model(image_batch)
            reg_loss_value = reg_criterion(reg_outputs, age_batch)
            cls_loss_value = cls_criterion(cls_outputs, gender_batch)
            progess_bar.set_description(f"Epoch [{epoch+1}/{args.epochs}]. Iteration [{iter+1}/{iters}]. Regression-MSELoss [{reg_loss_value:.3f}], Classification-CrossEntropyLoss [{cls_loss_value:.3f}]")
            writer.add_scalar("Train/Regression-MSELoss. ", reg_loss_value, epoch*iters + iter)
            writer.add_scalar("Train/Classification-CrossEntropyLoss. ", cls_loss_value, epoch*iters + iter)
            total_loss = reg_loss_value + cls_loss_value
            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        all_age_predictions = []
        all_gender_predictions = []
        all_age_actual = []
        all_gender_actual = []
        for iter, (image_batch, age_batch, gender_batch) in enumerate(test_dataloader):
            all_age_actual.extend(age_batch)
            all_gender_actual.extend(gender_batch)

            image_batch = image_batch.to(device)
            age_batch = age_batch.to(device)
            gender_batch = gender_batch.to(device)

            with torch.no_grad():
                age_predictions, gender_predictions = model(image_batch)
                all_age_predictions.extend(age_predictions.cpu())
                indices = torch.argmax(gender_predictions, dim=1)
                all_gender_predictions.extend(indices.cpu())

        all_age_actual = [age_actual.item() for age_actual in all_age_actual]
        all_age_predictions = [prediction.item() for prediction in all_age_predictions]
        r2 = r2_score(all_age_actual, all_age_predictions)

        all_gender_actual = [gender_actual.item() for gender_actual in all_gender_actual]
        all_gender_predictions = [prediction.item() for prediction in all_gender_predictions]
        accuracy = accuracy_score(all_gender_actual, all_gender_predictions)

        print(f"\nEpoch {epoch + 1}. Regression - R-squared: {r2}, Classification - Accuracy: {accuracy}\n")
        writer.add_scalar("Val/Regression - R-squared. ", r2, epoch)
        writer.add_scalar("Val/Classification - Accuracy. ", accuracy, epoch)

        total_score = (accuracy + r2)/2
        checkpoint = {
            "epoch": epoch + 1,
            "best_score": best_score,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{args.trained_models}/last_{typemodel}_model.pt")
        if total_score > best_score:
            checkpoint = {
                "best_score": best_score,
                "reg_r2": r2,
                "cls_accuracy": accuracy,
                "model": model.state_dict(),
            }
            torch.save(checkpoint, f"{args.trained_models}/best_{typemodel}_model.pt")
            best_score = total_score



