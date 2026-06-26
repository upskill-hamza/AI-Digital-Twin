


# 🌍 AI-Powered Digital Twin of India's Climate

## 🚀 Quick Start
1. Clone the repo: `git clone <repo-url>`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) / `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Download the trained model from [Google Drive](https://drive.google.com/drive/folders/13ukf5Dx44Pto7rts10u78mV-rleI9iSF) and place in `models/`
6. Run the dashboard: `streamlit run dashboard/app.py`

## 🛠️ Reproducing from Scratch
- Raw IMD data → `data/raw/` (download links in section below)
- Run `scripts/create_sequences.py` to generate normalized training data
- Run `scripts/train_model.py` to train the ConvLSTM model
- Model is saved to `models/`


## 📥 Required Large Files

Download from [Google Drive](YOUR_SHARED_LINK) and place as follows:

- `convlstm_best.keras` → `models/`
- `climate_combined.nc` → `data/` (optional)
- `train_data.npz`, `val_data.npz`, `test_data.npz` → `data/` (optional, can be regenerated)

## 📊 Dashboard Features
- Preset scenarios: Heatwave, Drought, Heavy Rain
- AI forecast maps (rainfall & temperature) for Tamil Nadu & Kerala
- "What‑if" difference maps (scenario vs baseline)
- Observed IMD data toggle for comparison
- Full India map with state borders

## 🧩 Code Structure
- `dashboard/app.py`: Streamlit UI
- `src/digital_twin.py`: Recursive simulation loop and scenario modifier
- `src/data_loader.py`: Reads IMD binary grid files
- `src/preprocessing.py`: Normalization, regridding, sequence creation

## 📎 Data Sources
- IMD gridded rainfall: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
- IMD temperature: https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
- India GeoJSON: https://github.com/geohacker/india

## 🤝 Contributing
- Create a branch for new features
- Open a Pull Request with description
- Tag @yourname for review

