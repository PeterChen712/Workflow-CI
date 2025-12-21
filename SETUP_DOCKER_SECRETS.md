# Setup Docker Hub Secrets untuk GitHub Actions

## Langkah-langkah:

### 1. Buka GitHub Repository Settings
- Pergi ke: https://github.com/PeterChen712/Workflow-CI
- Klik **Settings** (tab paling kanan)

### 2. Buka Secrets and Variables
- Di sidebar kiri, cari **Secrets and variables** → **Actions**

### 3. Tambahkan Secret: DOCKER_USERNAME
- Klik tombol **New repository secret**
- Name: `DOCKER_USERNAME`
- Value: `peterchen712` (username Docker Hub Anda)
- Klik **Add secret**

### 4. Tambahkan Secret: DOCKER_PASSWORD
- Klik tombol **New repository secret** lagi
- Name: `DOCKER_PASSWORD`
- Value: (password Docker Hub Anda, ATAU lebih aman: gunakan Access Token)
- Klik **Add secret**

### 5. (Recommended) Gunakan Docker Access Token
Lebih aman dari password biasa:
1. Login ke https://hub.docker.com/
2. Klik profile icon → **Account Settings**
3. Tab **Security** → **New Access Token**
4. Description: `GitHub Actions Workflow-CI`
5. Access permissions: **Read, Write, Delete**
6. Generate token, copy token tersebut
7. Paste token ke GitHub secret `DOCKER_PASSWORD`

### 6. Verifikasi Secrets
Setelah ditambahkan, Anda akan melihat 2 secrets:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

### 7. Push Changes dan Trigger Workflow
```bash
cd "d:\Rudy\file rudy\UNHAS\Stupen\Class\9 - Belajar sistem\Workflow-CI"
git add .
git commit -m "Enable Docker build and create Dockerfile"
git push
```

### 8. Monitor GitHub Actions
- Buka: https://github.com/PeterChen712/Workflow-CI/actions
- Workflow akan otomatis running
- Tunggu sampai semua jobs (train, upload-artifacts, build-docker) berwarna **hijau** ✅

## Troubleshooting

**Jika build-docker gagal:**
- Cek apakah secrets sudah ditambahkan dengan nama yang benar
- Pastikan Docker Hub credentials valid
- Lihat logs di GitHub Actions untuk error detail

**Jika upload-artifacts gagal:**
- Normal jika tidak ada changes, job akan skip push

**Jika train gagal:**
- Cek apakah dependencies terinstall (pandas, numpy, scikit-learn, mlflow)
- Lihat logs untuk error detail
