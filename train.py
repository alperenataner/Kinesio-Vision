import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split

# --- 1. DOSYA YOLU VE YAPILANDIRMA ---
# Senin belirttiğin yerel bilgisayarındaki dosya yolu
CSV_PATH = r"C:\Users\alper\OneDrive\Masaüstü\SOFTWARE\Kinesio-Vision\imagePoses.csv"

# Veri setinde tespit ettiğimiz 9 farklı egzersiz
CLASSES = ['pushup', 'pullup', 'chest dips', 'bent over row', 
           'shoulder press', 'squat', 'lunges', 'plank', 'glute bridge']
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

# --- 2. VERİ SETİ İŞLEME (DATASET) SINIFI ---
class KinesioDataset(Dataset):
    def __init__(self, csv_file):
        print("Veri seti yükleniyor ve metin tabanlı koordinatlar matrise çevriliyor...")
        self.data = pd.read_csv(csv_file)
        
        self.features = []
        self.labels = []
        
        for index, row in self.data.iterrows():
            label_str = row['excercise']
            if label_str not in class_to_idx:
                continue 
            
            label_idx = class_to_idx[label_str]
            raw_str = row['poseLandmarks']
            
            try:
                # String içindeki listeyi Python listesine çevir
                parsed_list = ast.literal_eval(raw_str)
                row_features = []
                
                # Her bir nokta verisinden x, y, z ve visibility değerlerini çek
                for item in parsed_list:
                    matches = re.findall(r'[x|y|z|visibility]:\s*([-+]?\d*\.\d+|\d+)', item)
                    row_features.extend([float(m) for m in matches])
                
                # 33 nokta x 4 eksen = 132 özellik olmalı
                if len(row_features) == 132:
                    self.features.append(row_features)
                    self.labels.append(label_idx)
            except Exception as e:
                continue # Hatalı formatlanmış satırları güvenli şekilde atla
        
        self.features = np.array(self.features, dtype=np.float32)
        # Çok sınıflı sınıflandırma için etiketler int64 (LongTensor) olmalı
        self.labels = np.array(self.labels, dtype=np.int64) 
        print(f"İşlem tamamlandı! Başarıyla yüklenen temiz örnek sayısı: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# --- 3. VERİ YÜKLEYİCİLERİ HAZIRLAMA ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan Donanım: {device}")

dataset = KinesioDataset(CSV_PATH)
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

train_loader = DataLoader(dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
test_loader = DataLoader(dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(test_idx))

# --- 4. KINESIO-NET YAPAY SİNİR AĞI MİMARİSİ ---
class KinesioNetMultiClass(nn.Module):
    def __init__(self):
        super(KinesioNetMultiClass, self).__init__()
        # Girdi: 132 özellik (33 nokta * 4 değer)
        self.fc1 = nn.Linear(132, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Çıktı: 9 Egzersiz Sınıfı
        self.out = nn.Linear(128, 9)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        # Çoklu sınıflandırmada Softmax işlemi CrossEntropyLoss içinde otomatik yapılır, 
        # bu yüzden doğrudan çıktı katmanını (logits) döndürüyoruz.
        return self.out(x)

model = KinesioNetMultiClass().to(device)

# --- 5. EĞİTİM MOTORU ---
# Çok sınıflı sınıflandırma (Multi-class Classification) için CrossEntropyLoss
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 150

print("\nModel Eğitimi Başlıyor...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Eğitim doğruluğunu (Accuracy) hesaplama
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    if (epoch+1) % 10 == 0:
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] | Kayıp (Loss): {epoch_loss/len(train_loader):.4f} | Doğruluk: %{accuracy:.2f}')

# Modeli Kaydetme
torch.save(model.state_dict(), 'kinesio_multiclass_model.pth')
print("\nEğitim tamamlandı! Model 'kinesio_multiclass_model.pth' olarak kaydedildi.")