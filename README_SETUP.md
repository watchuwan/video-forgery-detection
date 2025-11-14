# 🎬 Video Forgery Detection System - Setup Complete ✅

docker compose -f docker-compose-full.yml up --build

## Dua Versi Tersedia

### 1. **GRADIO VERSION** (Web Interface)

- **URL**: http://localhost:7861
- **Status**: ✅ Sedang Running
- **File**: `main.py`
- **Fitur**:
  - Drag & drop upload
  - Real-time analysis
  - Public share link
  - Browser playback otomatis

### 2. **FLASK VERSION** (REST API + Web)

- **URL**: http://localhost:5000
- **File**: `app_flask.py`
- **Fitur**:
  - REST API endpoints
  - JSON responses
  - Modern HTML interface
  - Better untuk integration

---

## 🚀 Cara Menjalankan

### Gradio Only (Default - Sudah Running)

```bash
docker compose up -d
```

Akses: **http://localhost:7861**

### Flask Only

Edit `docker-compose.yml` line 17, ubah dari:

```yaml
CMD ["python", "main.py"]
```

ke:

```yaml
CMD ["python", "app_flask.py"]
```

Kemudian:

```bash
docker compose up --build -d
```

Akses: **http://localhost:5000**

### Keduanya Bersamaan

```bash
docker compose -f docker-compose-full.yml up -d
```

- Gradio: http://localhost:7861
- Flask: http://localhost:5000

---

## 📺 Coba Sekarang

**1. Buka browser ke**: http://localhost:7861
**2. Upload video dalam format**:

- MP4, AVI, MOV, FLV, WMV, MKV, WebM, M4V, 3GP
  **3. Klik "Analyze Video"
  **4. Lihat hasil deteksi forgery + confidence

---

## 🔗 API Endpoints (Flask)

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Get Classes

```bash
curl http://localhost:5000/api/classes
```

### Predict Video

```bash
curl -X POST -F "video=@video.avi" http://localhost:5000/api/predict
```

---

## 📊 File Structure

```
video-forgery-detection/
├── main.py                      # Gradio version
├── app_flask.py                 # Flask version (NEW)
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Default (Gradio only)
├── docker-compose-full.yml      # Both versions
├── library.txt                  # Python dependencies
├── DEPLOYMENT.md                # Detailed guide
├── model/
│   └── best_video_resnet_multi_class_model.pth
├── templates/
│   └── index.html              # Flask frontend (NEW)
└── uploads/                     # Uploaded videos
```

---

## ✨ Perbaikan dari AVI Error

✅ Input sekarang accept semua format (tidak hanya MP4)
✅ AVI otomatis convert ke MP4 (H.264 + yuv420p)
✅ Better error handling & messages
✅ Support 9+ video formats

---

## 🎯 Forgery Classes Detected

1. **Original** - Tidak ada forgery
2. **Insertion** - Frame dari video lain ditambahkan
3. **Deletion** - Frame dihapus
4. **Duplication** - Frame duplikat
5. **Horizontal Flipping** - Video flipped horizontal
6. **Vertical Flipping** - Video flipped vertikal
7. **Other Rotation** - Rotasi berbagai sudut
8. **Zooming** - Zoom in/out

---

## 🛑 Stop Container

```bash
docker compose down
```

## 📝 View Logs

```bash
docker logs video-forgery-detection-app
```

---

**Status**: ✅ **READY TO USE**
**Created**: November 14, 2025
**Docker**: ✅ Running
**Model**: ✅ Loaded
**Device**: CPU (GPU jika tersedia)
