// Enhanced AI Tour Guide Frontend with Dark Mode
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file');
    const uploadArea = document.getElementById('uploadArea');
    const uploadForm = document.getElementById('uploadForm');
    const processBtn = document.getElementById('processBtn');
    const themeToggle = document.getElementById('themeToggle');
    
    // Store uploaded image data URL
    let uploadedImageDataUrl = null;

    // Theme Management
    const initTheme = () => {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
    };

    const toggleTheme = () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        showToast(newTheme === 'dark' ? 'Темная тема включена' : 'Светлая тема включена', 'info');
    };

    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    initTheme();

    if (!fileInput || !uploadArea) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = 'var(--primary)';
            uploadArea.style.transform = 'scale(1.01)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = '';
            uploadArea.style.transform = '';
        }, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    }, false);

    // Handle file selection via input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showToast('Пожалуйста, выберите изображение (JPEG, PNG или WebP)', 'error');
            fileInput.value = '';
            return;
        }

        // Validate file size (10 MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            showToast('Файл слишком большой. Максимальный размер: 10 MB', 'error');
            fileInput.value = '';
            return;
        }

        // Show preview
        showImagePreview(file);
        showToast('Изображение загружено успешно', 'success');
    }

    function showImagePreview(file) {
        const uploadContent = document.getElementById('uploadContent');
        const previewContainer = document.getElementById('previewContainer');
        
        if (!uploadContent || !previewContainer) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            // Store the image data URL
            uploadedImageDataUrl = e.target.result;
            
            // Hide upload content, show preview
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'block';
            previewContainer.innerHTML = `
                <div style="position: relative; display: inline-block;">
                    <img src="${e.target.result}" alt="Preview" style="max-width: 100%; max-height: 300px; border-radius: 16px; box-shadow: 0 8px 32px var(--shadow-color); border: 1px solid var(--glass-border);">
                    <button type="button" onclick="resetUpload()" style="position: absolute; top: 12px; right: 12px; background: var(--error); color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; font-weight: 600;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">×</button>
                </div>
                <p style="margin-top: 16px; color: var(--text-secondary); font-weight: 500; font-size: 0.95rem;">
                    <svg style="width: 16px; height: 16px; display: inline-block; vertical-align: middle; margin-right: 6px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                    ${file.name} • ${formatFileSize(file.size)}
                </p>
            `;
        };
        reader.readAsDataURL(file);
    }

    // Make resetUpload globally accessible
    window.resetUpload = function() {
        const uploadContent = document.getElementById('uploadContent');
        const previewContainer = document.getElementById('previewContainer');
        
        fileInput.value = '';
        uploadedImageDataUrl = null;
        
        if (uploadContent) uploadContent.style.display = 'block';
        if (previewContainer) {
            previewContainer.style.display = 'none';
            previewContainer.innerHTML = '';
        }
        
        showToast('Загрузка отменена', 'info');
    };

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    function showToast(message, type = 'info') {
        // Remove existing toasts
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'toast';
        
        const icons = {
            success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
            error: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`,
            info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`
        };
        
        const colors = {
            success: { bg: '#10b981', border: '#059669' },
            error: { bg: '#ef4444', border: '#dc2626' },
            info: { bg: '#1E9FD8', border: '#1680B0' }
        };
        
        const color = colors[type] || colors.info;
        
        toast.innerHTML = `
            <div style="width: 20px; height: 20px; flex-shrink: 0;">${icons[type] || icons.info}</div>
            <span>${message}</span>
        `;
        
        toast.style.cssText = `
            position: fixed;
            top: 24px;
            right: 24px;
            background: ${color.bg};
            color: white;
            padding: 16px 20px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            font-weight: 600;
            font-size: 0.95rem;
            max-width: 400px;
            display: flex;
            align-items: center;
            gap: 12px;
            border: 2px solid ${color.border};
            animation: slideInRight 0.3s ease;
            backdrop-filter: blur(10px);
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // Add CSS animations for toasts
    if (!document.getElementById('toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
            @keyframes slideInRight {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOutRight {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Enhanced form submission with animated loading
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showToast('Пожалуйста, выберите изображение', 'error');
                return;
            }

            // Save uploaded image to sessionStorage before form submission
            if (uploadedImageDataUrl) {
                try {
                    sessionStorage.setItem('uploadedImage', uploadedImageDataUrl);
                } catch (err) {
                    console.warn('Could not save image to sessionStorage:', err);
                }
            }

            // Show animated loading state
            if (processBtn) {
                processBtn.disabled = true;
                processBtn.innerHTML = `
                    <div class="loading-spinner">
                        <div class="spinner-ring"></div>
                        <div class="spinner-ring"></div>
                        <div class="spinner-ring"></div>
                    </div>
                    <span>Обработка изображения...</span>
                `;
                processBtn.style.opacity = '0.8';
                
                // Add loading spinner styles
                if (!document.getElementById('loading-spinner-styles')) {
                    const spinnerStyle = document.createElement('style');
                    spinnerStyle.id = 'loading-spinner-styles';
                    spinnerStyle.textContent = `
                        .loading-spinner {
                            display: inline-flex;
                            position: relative;
                            width: 24px;
                            height: 24px;
                        }
                        .spinner-ring {
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            border: 3px solid transparent;
                            border-top-color: white;
                            border-radius: 50%;
                            animation: spinRing 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
                        }
                        .spinner-ring:nth-child(1) {
                            animation-delay: -0.45s;
                        }
                        .spinner-ring:nth-child(2) {
                            animation-delay: -0.3s;
                        }
                        .spinner-ring:nth-child(3) {
                            animation-delay: -0.15s;
                        }
                        @keyframes spinRing {
                            0% {
                                transform: rotate(0deg);
                            }
                            100% {
                                transform: rotate(360deg);
                            }
                        }
                    `;
                    document.head.appendChild(spinnerStyle);
                }
            }
        });
    }

    // Show uploaded image in results if available
    const resultContainer = document.querySelector('.result-section');
    if (resultContainer) {
        const resultImageContainer = document.getElementById('resultImageContainer');
        if (resultImageContainer) {
            // Try to get image from sessionStorage
            const savedImage = sessionStorage.getItem('uploadedImage');
            if (savedImage) {
                resultImageContainer.innerHTML = `<img src="${savedImage}" alt="Загруженное изображение">`;
                // Clear from sessionStorage after displaying
                sessionStorage.removeItem('uploadedImage');
            } else if (uploadedImageDataUrl) {
                resultImageContainer.innerHTML = `<img src="${uploadedImageDataUrl}" alt="Загруженное изображение">`;
            }
        }
        
        // Smooth scroll to results
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 300);
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + U to trigger file upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            fileInput.click();
        }
        
        // Escape to reset upload
        if (e.key === 'Escape' && uploadedImageDataUrl) {
            window.resetUpload();
        }
        
        // Ctrl/Cmd + D to toggle theme
        if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            toggleTheme();
        }
    });

    // Show welcome message on first load
    if (!document.querySelector('.result-section') && !document.querySelector('.alert')) {
        setTimeout(() => {
            showToast('Загрузите фото достопримечательности для распознавания', 'info');
        }, 800);
    }

    // Add paste support for images
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                e.preventDefault();
                const blob = items[i].getAsFile();
                if (blob) {
                    // Create a new File object with proper name
                    const file = new File([blob], `pasted-image-${Date.now()}.png`, { type: blob.type });
                    
                    // Create a DataTransfer object to set files
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                    
                    handleFileSelect(file);
                    showToast('Изображение вставлено из буфера обмена', 'success');
                }
                break;
            }
        }
    });

    // Add visual feedback for toggle switch
    const toggleSwitch = document.querySelector('.toggle-switch');
    if (toggleSwitch) {
        toggleSwitch.addEventListener('click', () => {
            const checkbox = toggleSwitch.querySelector('input[type="checkbox"]');
            if (checkbox) {
                setTimeout(() => {
                    const isChecked = checkbox.checked;
                    showToast(
                        isChecked ? 'Интернет-поиск включен' : 'Интернет-поиск отключен',
                        'info'
                    );
                }, 50);
            }
        });
    }
});

// ── Слайдер галереи ────────────────────────────────────────────────────
(function initGallerySlider() {
    const slider = document.getElementById('gallerySlider');
    if (!slider) return;

    const track = document.getElementById('galleryTrack');
    const prevBtn = document.getElementById('sliderPrev');
    const nextBtn = document.getElementById('sliderNext');
    const dotsContainer = document.getElementById('sliderDots');

    // Собираем только видимые слайды
    const slides = Array.from(track.querySelectorAll('.gallery-slide'))
        .filter(s => s.style.display !== 'none');

    if (slides.length <= 1) {
        // Один слайд — кнопки и точки не нужны
        if (prevBtn) prevBtn.style.display = 'none';
        if (nextBtn) nextBtn.style.display = 'none';
        return;
    }

    let current = 0;

    // Создаём точки
    slides.forEach((_, i) => {
        const dot = document.createElement('button');
        dot.className = 'slider-dot' + (i === 0 ? ' active' : '');
        dot.setAttribute('aria-label', `Слайд ${i + 1}`);
        dot.addEventListener('click', () => goTo(i));
        dotsContainer.appendChild(dot);
    });

    function goTo(index) {
        current = (index + slides.length) % slides.length;
        track.style.transform = `translateX(-${current * 100}%)`;
        dotsContainer.querySelectorAll('.slider-dot').forEach((d, i) => {
            d.classList.toggle('active', i === current);
        });
    }

    prevBtn.addEventListener('click', () => goTo(current - 1));
    nextBtn.addEventListener('click', () => goTo(current + 1));

    // Свайп на тач-устройствах
    let touchStartX = 0;
    slider.addEventListener('touchstart', e => {
        touchStartX = e.touches[0].clientX;
    }, { passive: true });
    slider.addEventListener('touchend', e => {
        const dx = e.changedTouches[0].clientX - touchStartX;
        if (Math.abs(dx) > 40) goTo(dx < 0 ? current + 1 : current - 1);
    }, { passive: true });

    // Клавиши ← →
    document.addEventListener('keydown', e => {
        if (!slider) return;
        if (e.key === 'ArrowLeft') goTo(current - 1);
        if (e.key === 'ArrowRight') goTo(current + 1);
    });
}());
