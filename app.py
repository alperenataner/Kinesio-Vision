import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from inference import VideoAnalyzer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # 500 MB limit

# Global dictionary to store processing jobs
jobs = {}

# Lazy initialization of the analyzer to avoid loading model on startup if not needed
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = VideoAnalyzer()
    return analyzer

def process_video_task(job_id, input_path, output_path):
    try:
        def update_progress(percent):
            jobs[job_id]['progress'] = percent
            
        analyzer_instance = get_analyzer()
        result = analyzer_instance.process_video(input_path, output_path, update_progress)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['primary_class'] = result.get('primary_class', 'Bilinmiyor')
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        print(f"Error processing video: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        
        # Save input file
        input_ext = os.path.splitext(filename)[1]
        input_filename = f"{job_id}{input_ext}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)
        
        # Output file path (WebM for best browser playback)
        output_filename = f"{job_id}.webm"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Initialize job
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'output_filename': output_filename
        }
        
        # Start processing thread
        thread = threading.Thread(target=process_video_task, args=(job_id, input_path, output_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id, 'status': 'processing'}), 202

@app.route('/status/<job_id>')
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    return jsonify(jobs[job_id])

@app.route('/video/<job_id>')
def serve_video(job_id):
    if job_id not in jobs:
        return "Job not found", 404
        
    job = jobs[job_id]
    if job['status'] != 'completed':
        return "Video not ready", 400
        
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], job['output_filename'])
    if not os.path.exists(output_path):
        return "Video file missing from server", 404
        
    return send_file(output_path, mimetype='video/webm')

if __name__ == '__main__':
    print("Flask sunucusu başlatılıyor... Tarayıcınızdan http://127.0.0.1:5000 adresine gidin.")
    app.run(debug=True, port=5000)
