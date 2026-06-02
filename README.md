# Solving-the-MSTSP-with-HGA

根據 IEEE 期刊 [Solving the Multi-Solution Traveling Salesman Problem with Hierarchical Genetic Algorithm](https://ieeexplore.ieee.org/document/10981731) 實現的 **GNN（圖神經網路）+ HGA（分層遺傳演算法）** 混合優化架構，用於快速且精確地求解多起點/多路徑旅行商問題。

本專案採用 **Web Workers 多執行緒並行島嶼模型** 進行網頁端演化，並支援兩種 GNN 推理方式：
1. **網頁端純 JS 神經網路引擎**：載入 GNN 權重 JSON 檔，在網頁端 CPU 進行動態推理。
2. **Python 高速預計算矩陣**：利用 Python CLI 工具在本地 GPU 進行高速推理，直接輸出預計算的機率矩陣，免去網頁端計算開銷，專門解決 **6,000+ 超商節點**的規模極限！

---

## 1. 網頁端執行與部署 (Web Server)

為了支持 Web Workers 的多線程運行並避免瀏覽器的 CORS 跨來源限制，本地執行時必須使用 Web 伺服器託管：

```shell
# 在專案根目錄下啟動本地 HTTP 伺服器
python3 -m http.server
```

啟動後，請在瀏覽器中打開：`http://localhost:8000`

---

## 2. Python CLI 圖神經網路工具 (`gnn_cli.py`)

本專案配備了一個功能強大的 Python CLI 工具 [gnn/gnn_cli.py](file:///Volumes/fju/仿生計算/Solving-the-MSTSP-with-HGA/gnn/gnn_cli.py)，支援 **GPU 硬件加速 (MPS / CUDA)**。它擁有兩種工作模式：**模型訓練模式** 與 **高速空間推理模式**。

### A. 環境準備
GNN 的 Python 虛擬環境與依賴已預先配置於 `gnn/myenv/` 中，您可以直接調用該環境下的 Python 直譯器：

```shell
# 升級/安裝依賴（如有需要）
gnn/myenv/bin/pip install -r gnn/requirements.txt
```

---

### B. 模式一：模型訓練與導出權重 (--train)
您可以隨時使用自定義的參數（如隱藏層維度、訓練世代、合成圖大小等）來訓練一個全新的 GNN 模型，並輸出可直接匯入網頁端的 JSON 權重檔案：

```shell
# 1. 訓練 128 維 GNN 模型並導出 JSON 權重（預設）
gnn/myenv/bin/python gnn/gnn_cli.py --train --epochs 40 --d_model 128 --out gnn/mstsp_gnn_weights.json

# 2. 訓練模型的同時，也導出標準的 standalone ONNX 模型 (.onnx)
gnn/myenv/bin/python gnn/gnn_cli.py --train --epochs 40 --d_model 128 --out gnn/mstsp_gnn_weights.json --export_onnx
```

**參數說明：**
*   `--train`：啟動模型訓練模式。
*   `--d_model`：神經網路隱藏層通道數（預設：128，推薦）。
*   `--epochs`：訓練世代（預設：40）。
*   `--train_samples`：用於訓練的合成圖數量（預設：300）。
*   `--nodes`：合成圖中的城市點數（預設：150）。
*   `--out`：導出的 JSON 權重檔案名稱（預設：`mstsp_gnn_weights.json`）。
*   `--export_onnx`：是否同步導出已嵌入參數的靜態 ONNX 模型。

---

### C. 模式二：超商大圖 GPU 高速預計算推理 (--cities_file)
針對點數極大的實用數據集（如 6,000+ 個點的 7-11 全台地圖），直接在網頁端用 CPU 進行矩陣推理會導致瀏覽器卡頓半分鐘。

此時，您可以將地圖座標檔送入 Python CLI，在本地 GPU 上進行並行計算，並導出**預計算好的邊機率矩陣 JSON 檔案**。將該矩陣檔案拖曳進網頁端即可**瞬間（0 毫秒）**載入 glowing 霓虹紫色路網！

```shell
# 1. 對 OK超商 (688 個點) 進行 GPU 加速推理
gnn/myenv/bin/python gnn/gnn_cli.py --cities_file src/tests/OK超商.js --weights gnn/mstsp_gnn_weights.json --out gnn/ok_probs.json

# 2. 對 全家超商 (3,449 個點) 進行 GPU 加速推理
gnn/myenv/bin/python gnn/gnn_cli.py --cities_file src/tests/全家.js --weights gnn/mstsp_gnn_weights.json --out gnn/family_probs.json

# 3. 對 7-11超商 (6,080 個點) 進行 GPU 加速推理
gnn/myenv/bin/python gnn/gnn_cli.py --cities_file src/tests/7-11超商.js --weights gnn/mstsp_gnn_weights.json --out gnn/seven_probs.json
```

**參數說明：**
*   `--cities_file`：要進行預測的超商地圖 JS/JSON 檔案路徑。
*   `--weights`：已訓練好的模型權重 JSON 路徑（預設：`gnn/mstsp_gnn_weights.json`）。
*   `--out`：導出的預計算邊機率矩陣檔案名稱（預設：`mstsp_gnn_probs.json`）。

---

## 3. 網頁端加載說明

在瀏覽器網頁端（`http://localhost:8000`）的 **GNN 混合優化** 模塊中：
1. 將下拉式選單切換至 **Custom (自訂 GNN)**。
2. 點擊 **選擇檔案** 按鈕。
3. 您可以選擇兩種格式的 JSON 匯入：
   * **匯入權重檔 (`mstsp_gnn_weights.json`)**：網頁會動態適應維度，並以極快速度（0.05ms）在瀏覽器 CPU 中執行神經網路前向計算（適用於小於 500 點的地圖）。
   * **匯入預計算矩陣 (`seven_probs.json`)**：網頁會自動辨識並直接載入機率矩陣，**計算開銷為 0ms**，是解決 **6,080 點 7-11 地圖**的終極優化方案！