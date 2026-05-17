import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# TRANSFORMACIJE ZA PODATKE
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DATASETS IN DATALOADERS
train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transform)
valid_dataset = datasets.ImageFolder(root='dataset/val', transform=validation_transform)
test_dataset = datasets.ImageFolder(root='dataset/test', transform=validation_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Razredi: {class_names}")
print(f"Število razredov: {num_classes}")

# NAPRAVA (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Naprava: {device}")

# MODEL - MobileNetV2 s fin-tuningom
model = models.mobilenet_v2(pretrained=True)

# Zamenjaj zadnjo plast za nas problem (5 razredov)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

model.to(device)

# OPTIMIZER
optimizer = optim.Adam([
    {'params': model.features[-7:].parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.001}
])

# LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

# SPREMENLJIVKE ZA SLEDENJE
train_losses = []
valid_losses = []
valid_accuracies = []
best_accuracy = 0
patience = 10
patience_counter = 0

# TRENIRANJE IN VALIDACIJA
def train_epoch():
    """Ena epoca treniranja"""
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    return avg_loss

def validate():
    """Validacija modela"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    
    valid_losses.append(avg_loss)
    valid_accuracies.append(accuracy)
    
    return avg_loss, accuracy

def test():
    """Testiranje modela in primerjava s pravimi oznako"""
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"\nNatančnost na test nizu: {accuracy:.2f}%")
    
    # Prikazi confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Prikazi classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Nariši confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Prava oznaka')
    plt.xlabel('Predvidena oznaka')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy

# TRENIRANJE
num_epochs = 50
print("\n" + "="*50)
print("ZAČENJA SE TRENIRANJE")
print("="*50 + "\n")

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss, val_accuracy = validate()
    
    print(f"Epoca [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
    
    # Early stopping
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        patience_counter = 0
        # Shrani najbolji model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Nove najbolje! Shranjeno kot 'best_model.pth'")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping - noboljšanja v zadnjih {patience} epocah")
        break

# NALOŽI NAJBOLJI MODEL
model.load_state_dict(torch.load('best_model.pth'))

# TESTIRANJE
print("\n" + "="*50)
print("TESTIRANJE MODELA")
print("="*50)
test_accuracy = test()

# PRIKAZI GRAFE
plt.figure(figsize=(12, 4))

# Izguba
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoca')
plt.ylabel('Izguba')
plt.legend()
plt.title('Izguba med treningom')
plt.grid(True)

# Natančnost
plt.subplot(1, 2, 2)
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f'Best: {best_accuracy:.2f}%')
plt.xlabel('Epoca')
plt.ylabel('Natančnost (%)')
plt.legend()
plt.title('Natančnost na validacijskih podatkih')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("\n" + "="*50)
print("TRENIRANJE ZAKLJUČENO!")
print(f"Najboljša natančnost: {best_accuracy:.2f}%")
print(f"Test natančnost: {test_accuracy:.2f}%")
print("="*50)

