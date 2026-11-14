@echo off
chcp 65001 >nul
call "D:\miniforge3\Scripts\activate.bat" dl
cd /d D:\dl-journey

echo ===== BASE  EPOCHS=5 HIDDEN=256 LR=0.001 =====
set EPOCHS=5
set HIDDEN=256
set LR=0.001
python playground\mlp_torch.py

echo ===== A  EPOCHS=1 HIDDEN=256 LR=0.001 =====
set EPOCHS=1
python playground\mlp_torch.py

echo ===== B  EPOCHS=5 HIDDEN=32 LR=0.001 =====
set EPOCHS=5
set HIDDEN=32
python playground\mlp_torch.py

echo ===== C  EPOCHS=5 HIDDEN=256 LR=0.1 =====
set HIDDEN=256
set LR=0.1
python playground\mlp_torch.py

echo ===== DONE =====
