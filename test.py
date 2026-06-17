import cv2
import torch
import torch.nn as nn
import mediapipe as mp

# --- 1. MODEL MİMARİSİ VE YÜKLEME ---
class KinesioNetMultiClass(nn.Module):
    def __init__(self):
        super(KinesioNetMultiClass, self).__init__()
        self.fc1 = nn.Linear(132, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(128, 9)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

CLASSES = ['pushup', 'pullup', 'chest dips', 'bent over row', 
           'shoulder press', 'squat', 'lunges', 'plank', 'glute bridge']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KinesioNetMultiClass().to(device)
model.load_state_dict(torch.load('kinesio_multiclass_model.pth', map_location=device))
model.eval()

# --- 2. MEDIAPIPE HAZIRLIĞI ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 3. VİDEO GİRİŞ VE ÇIKIŞ AYARLARI ---
GIRIS_VIDEOSU = "video10.mp4"  # Analiz edilecek videonun yolu
CIKIS_VIDEOSU = "kinesio_analiz_cikti10.mp4"

cap = cv2.VideoCapture(GIRIS_VIDEOSU)

# Orijinal videonun boyutlarını ve FPS değerini alma
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30 # Güvenlik amaçlı varsayılan FPS değeri

# Çıktı videosunu oluşturacak nesne
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(CIKIS_VIDEOSU, fourcc, fps, (frame_width, frame_height))

print(f"'{GIRIS_VIDEOSU}' analiz ediliyor...")

# --- 4. VİDEO İŞLEME DÖNGÜSÜ ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video analizi tamamlandı.")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        row_features = []
        for landmark in results.pose_landmarks.landmark:
            row_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        if len(row_features) == 132:
            input_tensor = torch.tensor([row_features], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                predicted_class = CLASSES[predicted.item()]
                
                # Tahmini ekrana yazdırma
                cv2.rectangle(image, (0, 0), (350, 70), (0, 0, 0), -1)
                cv2.putText(image, f"Hareket: {predicted_class.upper()}", (15, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # İşlenmiş kareyi orijinal boyutunda videoya kaydet
    out.write(image)
    
    # --- GÖRÜNTÜYÜ EKRANA SIĞDIRMA (RESIZE) ---
    hedef_genislik = 1080
    h, w, _ = image.shape
    if w > hedef_genislik:
        oran = hedef_genislik / w
        yeni_boyut = (hedef_genislik, int(h * oran))
        image_gosterim = cv2.resize(image, yeni_boyut)
    else:
        image_gosterim = image

    # İşlenmiş ve boyutlandırılmış kareyi ekranda göster
    cv2.imshow('Kinesio-Vision Video Analizi', image_gosterim)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"İşlem bitti. Analiz edilen video '{CIKIS_VIDEOSU}' olarak kaydedildi.")