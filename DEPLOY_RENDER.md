# 🚀 Deploying Emotion Monitor to Render.com

This repository is configured and ready for **1-click / Git deployment** to [Render](https://render.com).

---

## 📋 Prerequisites
1. A GitHub account.
2. A free account on [Render.com](https://render.com).
3. Push this project to your GitHub repository.

---

## 🛠️ Step-by-Step Deployment Instructions

### Method 1: Standard Python Web Service (Recommended)

1. Log in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top right corner and select **Web Service**.
3. Connect your GitHub repository.
4. Fill in the settings:
   * **Name**: `emotion-attention-monitor`
   * **Region**: Choose closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   * **Branch**: `main` (or `master`)
   * **Runtime**: `Python 3`
   * **Build Command**: 
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   * **Start Command**: 
     ```bash
     python server.py
     ```
   * **Plan**: `Free`
5. Click **Create Web Service**.
6. Render will automatically build and deploy your project!

---

### Method 2: Docker Deployment (Zero Dependency Issues)

If you prefer containerized deployment:
1. When creating the Web Service, choose **Docker** as the environment (Render will automatically detect the provided `Dockerfile`).
2. Set Port: `5000` or let Render set `$PORT`.
3. Click **Deploy**.

---

## ⚙️ Environment Variables (Optional)
Render automatically provisions `$PORT`. You can also configure:
* `PORT` = `10000` (Render default)
* `PYTHONUNBUFFERED` = `1`
