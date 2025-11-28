# Knowledge Aware State-Space Capsule Network (KCapMamba)  
### For Multivariate Time Series Classification  

This study proposes KACapMamba, a Knowledge-Aware State-Space Capsule Network, which integrates three attentive Mamba blocks with a routing layer to achieve hierarchical temporal modeling.  


## Table of Contents  
- [Environment Requirements](#environment-requirements)  
- [Data Preparation](#data-preparation)  
- [Training Configuration](#training-configuration)  
- [Run Training](#run-training)  
- [Project Structure](#project-structure)  
- [Experimental Analysis Tools](#experimental-analysis-tools)  
- [Citation](#citation)  


## Environment Requirements  
Ensure the following dependencies are installed for seamless execution:  
```bash
Python >= 3.6  
TensorFlow == 1.13.1 (or compatible 1.x version)  
NumPy >= 1.19.5  
pandas >= 1.1.5  
scikit-learn >= 0.24.2  
liac-arff >= 2.5.0  
```  

Install dependencies via pip:  
```bash
pip3 install numpy pandas scikit-learn liac-arff tensorflow-gpu==1.13.1
```  


## Data Preparation  
1. **Download Dataset**:  
   Obtain the UEA Time Series Classification datasets from the official repository:  
   [UEA Time Series Classification Website](https://www.timeseriesclassification.com/)  

2. **Organize Data**:  
   Extract the downloaded dataset(s) and place them in the `data/` directory with the following structure:  
   ```
   data/
   ├── AtrialFibrillation/       # Example dataset
   │   ├── AtrialFibrillation_TRAIN.arff
   │   └── AtrialFibrillation_TEST.arff
   └── [Other UEA datasets...]   # e.g., Coffee, ECG200, etc.
   ```  


## Training Configuration  
All hyperparameters and path settings are managed in `config/config.py`:  

```python
import os

# Data paths & settings
load_root = "../data/"               # Root directory for datasets
dataname = "AtrialFibrillation"      # Target dataset name
fill_strategy = 'column_mean'        # Missing value filling: column_mean/sample_mean
logs = "../data/log/"                # Directory for training logs

# Hardware & training parameters
devices = '0'                        # GPU device ID (e.g., '0' for single GPU, '0,1' for multi-GPU)
epochs = 1                           # Training epochs
options = 1                          # Optimizer selection (1=Adam, configurable)
learning_rate = 0.001                # Initial learning rate
decay_steps = 100                    # Learning rate decay interval
drop_rate = 0.8                      # Dropout rate for regularization

# Batch settings
batch_num = 1                        # batch_size = total_train_samples // batch_num
test_num = 1                         # test_batch_size = total_test_samples // test_num
```  

### Key Config Notes:  
- `fill_strategy`: Choose `column_mean` (fill NaNs with column-wise mean) or `sample_mean` (fill with sample-wise mean).  
- `batch_num`: Adjust based on GPU memory (larger values = smaller batch size).  


## Run Training  
Execute the training script from the root directory:  

```bash
cd execute
python3 main.py
```  

### Training Outputs:  
- Training logs (accuracy, loss, training time) are saved to `data/log/log.txt`.  
- Model checkpoints and performance metrics are automatically logged during training.  


## Project Structure  
```
├── config/                     # Configuration module
│   ├── config.py               # Hyperparameters & path settings
│   └── load.py                 # UEA dataset loading utilities
├── execute/                    # Execution scripts
│   └── main.py                 # Main training pipeline
├── modelib/                    # Model architecture
│   ├── capsule.py              # KCapMamba core implementation (SSM + capsule layers)
│   └── capsule_loss.py         # Custom loss functions (margin loss) & optimizers
├── processing/                 # Data preprocessing
│   └── process.py              # Normalization, one-hot encoding, missing value handling
├── AVG_rank/                   # Statistical analysis tools
│   └── avg_rank_calculator.py  # Compute average ranks across datasets
├── Statisticaltool/            # Significance testing
│   └── p_value_analysis.py     # P-value calculation for performance comparison
└── data/                       # Dataset & logs
    ├── [UEA Datasets...]
    └── log/                    # Training logs storage
```  


## Experimental Analysis Tools  
- **AVG_rank/**: Scripts to calculate average ranks of model performance across multiple datasets, enabling fair comparison with baselines.  
- **Statisticaltool/**: Tools for statistical significance testing (e.g., paired t-tests, p-value calculation) to validate model superiority.  


## Citation  
If you use this code or model in your research, please cite our work:  

```bibtex
@article{kcapmamba2025,
  title={Knowledge Aware State-Space Capsule Network for Multivariate Time Series Classification},
  year={2025},
}
```  


## Contact  
For questions, issues, or collaboration inquiries, please open an issue in the repository or contact [your-email@example.com].  

---  
*Last updated: 2025*
