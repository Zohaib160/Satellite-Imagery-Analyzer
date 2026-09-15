document.addEventListener('DOMContentLoaded', () => {
    // --- Tabs Logic ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- Drag and Drop File Input Logic ---
    function setupDropZone(dropZoneId, btnId, requireBoth = false, siblingZoneId = null) {
        const dropZone = document.getElementById(dropZoneId);
        const input = dropZone.querySelector('.drop-zone-input');
        const prompt = dropZone.querySelector('.drop-zone-prompt');
        const preview = dropZone.querySelector('.preview-img');
        const btn = document.getElementById(btnId);

        dropZone.addEventListener('click', () => input.click());

        input.addEventListener('change', () => {
            if (input.files.length) {
                updateThumbnail(dropZone, input.files[0]);
                checkEnableButton();
            }
        });

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('drop-zone--over');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropZone.addEventListener(type, () => {
                dropZone.classList.remove('drop-zone--over');
            });
        });

        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                updateThumbnail(dropZone, e.dataTransfer.files[0]);
                checkEnableButton();
            }
            dropZone.classList.remove('drop-zone--over');
        });

        function updateThumbnail(dropZoneElement, file) {
            prompt.style.display = 'none';
            preview.style.display = 'block';

            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                preview.src = reader.result;
            };
        }

        function checkEnableButton() {
            if (requireBoth && siblingZoneId) {
                const siblingInput = document.getElementById(siblingZoneId).querySelector('input');
                if (input.files.length > 0 && siblingInput.files.length > 0) {
                    btn.disabled = false;
                } else {
                    btn.disabled = true;
                }
            } else {
                btn.disabled = input.files.length === 0;
            }
        }
        
        return input;
    }

    const classifyInput = setupDropZone('classify-drop', 'btn-classify');
    const nirInput = setupDropZone('nir-drop', 'btn-ndvi', true, 'red-drop');
    const redInput = setupDropZone('red-drop', 'btn-ndvi', true, 'nir-drop');
    const beforeInput = setupDropZone('before-drop', 'btn-change', true, 'after-drop');
    const afterInput = setupDropZone('after-drop', 'btn-change', true, 'before-drop');


    // --- API Calls ---

    // Classify
    const btnClassify = document.getElementById('btn-classify');
    btnClassify.addEventListener('click', async () => {
        if (!classifyInput.files[0]) return;
        
        document.getElementById('classify-error').innerText = '';
        document.getElementById('classify-results').style.display = 'none';
        document.getElementById('classify-loading').style.display = 'flex';
        
        const formData = new FormData();
        formData.append('file', classifyInput.files[0]);

        try {
            const res = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Classification failed');
            
            document.getElementById('classify-pred').innerText = data.class;
            document.getElementById('classify-conf').innerText = `${(data.confidence * 100).toFixed(2)}%`;
            
            const probChart = document.getElementById('prob-chart');
            probChart.innerHTML = '';
            
            // Sort probs
            const sortedProbs = Object.entries(data.probabilities).sort((a,b) => b[1] - a[1]);
            
            sortedProbs.forEach(([className, prob]) => {
                const percentage = (prob * 100).toFixed(1);
                probChart.innerHTML += `
                    <div class="bar-row">
                        <div class="bar-label-container">
                            <span>${className}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: 0%"></div>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('classify-loading').style.display = 'none';
            document.getElementById('classify-results').style.display = 'block';
            
            // Animate bars
            setTimeout(() => {
                const bars = probChart.querySelectorAll('.bar-fill');
                sortedProbs.forEach(([, prob], i) => {
                    if (bars[i]) {
                        bars[i].style.width = `${prob * 100}%`;
                    }
                });
            }, 50);

        } catch (err) {
            document.getElementById('classify-loading').style.display = 'none';
            document.getElementById('classify-error').innerText = err.message;
        }
    });

    // NDVI
    async function processNDVI(isDemo = false) {
        document.getElementById('ndvi-error').innerText = '';
        document.getElementById('ndvi-results').style.display = 'none';
        document.getElementById('ndvi-loading').style.display = 'flex';
        
        let url = '/api/ndvi';
        const formData = new FormData();
        
        if (isDemo) {
            url += '?demo=true';
        } else {
            formData.append('nir', nirInput.files[0]);
            formData.append('red', redInput.files[0]);
        }

        try {
            const res = await fetch(url, {
                method: 'POST',
                body: isDemo ? null : formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'NDVI analysis failed');
            
            document.getElementById('ndvi-heatmap').src = `data:image/png;base64,${data.heatmap}`;
            
            const tbody = document.getElementById('ndvi-stats-body');
            tbody.innerHTML = '';
            Object.entries(data.statistics).forEach(([cat, pct]) => {
                tbody.innerHTML += `<tr><td>${cat}</td><td>${pct}%</td></tr>`;
            });
            
            document.getElementById('ndvi-loading').style.display = 'none';
            document.getElementById('ndvi-results').style.display = 'block';
            
        } catch (err) {
            document.getElementById('ndvi-loading').style.display = 'none';
            document.getElementById('ndvi-error').innerText = err.message;
        }
    }

    document.getElementById('btn-ndvi').addEventListener('click', () => processNDVI(false));
    document.getElementById('btn-ndvi-demo').addEventListener('click', () => processNDVI(true));

    // Change Detection
    async function processChange(isDemo = false) {
        document.getElementById('change-error').innerText = '';
        document.getElementById('change-results').style.display = 'none';
        document.getElementById('change-loading').style.display = 'flex';
        
        let url = '/api/change-detect';
        const formData = new FormData();
        
        if (isDemo) {
            url += '?demo=true';
        } else {
            formData.append('before', beforeInput.files[0]);
            formData.append('after', afterInput.files[0]);
        }

        try {
            const res = await fetch(url, {
                method: 'POST',
                body: isDemo ? null : formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Change detection failed');
            
            document.getElementById('change-map').src = `data:image/png;base64,${data.changemap}`;
            
            const tbody = document.getElementById('change-stats-body');
            tbody.innerHTML = '';
            Object.entries(data.statistics).forEach(([cat, pct]) => {
                tbody.innerHTML += `<tr><td>${cat}</td><td>${pct}%</td></tr>`;
            });
            
            document.getElementById('change-loading').style.display = 'none';
            document.getElementById('change-results').style.display = 'block';
            
        } catch (err) {
            document.getElementById('change-loading').style.display = 'none';
            document.getElementById('change-error').innerText = err.message;
        }
    }

    document.getElementById('btn-change').addEventListener('click', () => processChange(false));
    document.getElementById('btn-change-demo').addEventListener('click', () => processChange(true));
});
