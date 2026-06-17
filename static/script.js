document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadBtn = document.getElementById('upload-btn');
    
    const fileNameDisplay = document.getElementById('file-name-display');
    const selectedFileName = document.getElementById('selected-file-name');
    const uploadHint = document.getElementById('upload-hint');
    
    const uploadPanel = document.getElementById('upload-panel');
    const progressPanel = document.getElementById('progress-panel');
    const resultPanel = document.getElementById('result-panel');
    
    const progressBar = document.getElementById('progress-bar');
    const percentageText = document.getElementById('percentage');
    const progressText = document.getElementById('progress-text');
    
    const resultVideo = document.getElementById('result-video');
    const detectedClassBadge = document.getElementById('detected-class');
    const resetBtn = document.getElementById('reset-btn');

    let currentFile = null;

    // Handle Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });

    // Handle Browse Button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent triggering upload area click if nested
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    });

    function selectFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Lütfen geçerli bir video dosyası seçin.');
            return;
        }
        currentFile = file;
        
        // Update UI to show selected file
        selectedFileName.textContent = file.name;
        fileNameDisplay.classList.remove('hidden');
        uploadHint.classList.add('hidden');
        
        // Change "Dosya Seç" to "Başka Dosya Seç" and show Upload button
        browseBtn.textContent = 'Başka Dosya Seç';
        browseBtn.classList.replace('btn-primary', 'btn-secondary');
        uploadBtn.classList.remove('hidden');
    }

    uploadBtn.addEventListener('click', () => {
        if (currentFile) {
            startUpload(currentFile);
        }
    });

    // Handle File Upload
    function startUpload(file) {
        // UI Update
        uploadPanel.classList.add('hidden');
        progressPanel.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                pollStatus(data.job_id);
            }
        })
        .catch(err => {
            showError('Video yüklenirken bir hata oluştu.');
            console.error(err);
        });
    }

    // Polling Status
    function pollStatus(jobId) {
        const interval = setInterval(() => {
            fetch(`/status/${jobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    clearInterval(interval);
                    showError(data.error);
                } else if (data.status === 'processing') {
                    const pct = data.progress || 0;
                    progressBar.style.width = `${pct}%`;
                    percentageText.textContent = `${pct}%`;
                } else if (data.status === 'completed') {
                    clearInterval(interval);
                    progressBar.style.width = `100%`;
                    percentageText.textContent = `100%`;
                    
                    setTimeout(() => {
                        showResult(jobId, data.primary_class);
                    }, 500); // slight delay for smooth transition
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error);
                }
            })
            .catch(err => {
                console.error('Durum kontrolü hatası:', err);
            });
        }, 1000); // Poll every second
    }

    function showResult(jobId, primaryClass) {
        progressPanel.classList.add('hidden');
        resultPanel.classList.remove('hidden');
        
        // Update dashboard stats
        detectedClassBadge.textContent = primaryClass || "Tespit Edilemedi";
        
        // Add timestamp to prevent caching issues
        resultVideo.src = `/video/${jobId}?t=${new Date().getTime()}`;
        resultVideo.load();
        resultVideo.play();
    }

    function showError(msg) {
        progressText.textContent = `Hata: ${msg}`;
        progressText.classList.add('error-text');
        progressBar.style.background = 'var(--error)';
        
        setTimeout(() => {
            resetUI();
        }, 3000);
    }

    function resetUI() {
        currentFile = null;
        
        uploadPanel.classList.remove('hidden');
        progressPanel.classList.add('hidden');
        resultPanel.classList.add('hidden');
        
        progressBar.style.width = '0%';
        progressBar.style.background = '';
        percentageText.textContent = '0%';
        progressText.textContent = 'Bu işlem videonun uzunluğuna göre biraz sürebilir.';
        progressText.classList.remove('error-text');
        
        fileInput.value = '';
        resultVideo.src = '';
        
        // Reset file selection UI
        fileNameDisplay.classList.add('hidden');
        uploadHint.classList.remove('hidden');
        browseBtn.textContent = 'Dosya Seç';
        browseBtn.classList.replace('btn-secondary', 'btn-primary');
        uploadBtn.classList.add('hidden');
    }

    resetBtn.addEventListener('click', resetUI);
});
