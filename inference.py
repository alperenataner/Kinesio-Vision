import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import os

# --- MODEL MİMARİSİ ---
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

class VideoAnalyzer:
    def __init__(self, model_path='kinesio_multiclass_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KinesioNetMultiClass().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_video(self, input_path, output_path, progress_callback=None):
        cap = cv2.VideoCapture(input_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30
            
        # Çözünürlüğü optimize edelim (Maksimum boyut 854px - 480p kalitesi)
        # Yüksek çözünürlük yapay zekayı ve video kodlamasını çok yavaşlatır
        MAX_DIM = 854
        if frame_width > MAX_DIM or frame_height > MAX_DIM:
            scale = MAX_DIM / max(frame_width, frame_height)
            target_width = int(frame_width * scale)
            target_height = int(frame_height * scale)
        else:
            target_width = frame_width
            target_height = frame_height
            
        # Web uyumluluğu için VP8 codec ve .webm uzantısı kullanıyoruz
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        frame_count = 0
        predictions_list = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if target_width != frame_width or target_height != frame_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                results = self.pose.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    row_features = []
                    for landmark in results.pose_landmarks.landmark:
                        row_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    
                    if len(row_features) == 132:
                        input_tensor = torch.tensor([row_features], dtype=torch.float32).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            predicted_class = CLASSES[predicted.item()]
                            predictions_list.append(predicted_class)
                            
                            # Tahmini ekrana yazdırma
                            cv2.rectangle(image, (0, 0), (450, 70), (0, 0, 0), -1)
                            cv2.putText(image, f"Hareket: {predicted_class.upper()}", (15, 45), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                out.write(image)
                
                frame_count += 1
                if progress_callback and total_frames > 0:
                    # %100'ü geçmemesi için küçük bir güvenlik payı
                    percent = min(int((frame_count / total_frames) * 100), 100)
                    progress_callback(percent)
                    
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
        from collections import Counter
        if predictions_list:
            most_common_class = Counter(predictions_list).most_common(1)[0][0]
        else:
            most_common_class = "Tespit Edilemedi"
            
        return {
            "output_path": output_path,
            "primary_class": most_common_class.upper()
        }
