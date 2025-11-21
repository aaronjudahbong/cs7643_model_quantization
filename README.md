# cs7643_model_quantization
CS7643 Model Quantization Final Project
# cs7643_model_quantization
CS7643 Model Quantization Final Project

### Steps
#### 1. Datasets
The datasets are not included in the repository to be compliant with license restrictions.
To obtain the datasets:

##### 1a. CityScapes Dataset
1. Register at https://www.cityscapes-dataset.com/.
2. Download:
   - `leftImg8bit_trainvaltest.zip`
   - `gtFine_trainvaltest.zip`
3. Place them in `data/`.
4. Unzip the files.

The data directory should look like the following:
```
data/
├── gtFine_trainId_checksum.txt
├── gtFine_trainvaltest/
│   └── gtFine/
│       ├── train/
│       │   ├── aachen/
│       │   ├── bochum/
│       │   └── ...
│       ├── val/
│       └── test/
└── leftImg8Bit_trainvaltest/
    └── leftImg8Bit/
        ├── train/
        │   ├── aachen/
        │   ├── bochum/
        │   └── ...
        ├── val/
        └── test/
```

##### 1b. Remapped Ground Truth Labels
Run the following script to generate remapped ground truth labels and corresponding colorized images:
```bash
python src/scripts/run_dataset_remapping.py
```

This should add two more folders into the ```data/``` directory:
```data/
├── gtFine_trainId_checksum.txt
├── gtFine_trainvaltest/
├── leftImg8Bit_trainvaltest/
├── gtFine_trainId/
└── gtFine_colorized/
```

##### 1c. Verify Checksum
Finally, to verify that the remapped ground truth labels are correct, run the following script to verify the checksum.
```bash
python src/scripts/verify_checksum.py
```