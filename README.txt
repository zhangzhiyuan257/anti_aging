1. anti_aging -> code -> main.py
This is used to calculate the performance of 41 feature descriptors on the AAP400 dataset

2. anti_aging -> code -> calculating_percentage.py
This is used to draw the length distribution map of polypeptides (Figure 2C)

3. anti_aging -> code -> augmentation_ESM.py
This is used for GAN-based data augmentation (Figure 6B)

4. anti_aging -> code -> ESM_ml.py
The augmented data were subsequently used to construct predictive model，including SVM, LR, XGBoost, MLP, and RF (Figure 6C、D)

5. anti_aging -> code -> main_augment_ml.py
 This is used for performance comparison of predictive models based on amino acid conservative substitution data augmentation(Figure 7B)

6. 1）anti_aging -> code -> main_ESM_dl.py; 2）anti_aging -> code -> models_2.py
This is used for construction of deep learning models based on data augmentation (Figure 8A-F)

7. 1）anti_aging -> code -> predict.py; 2）anti_aging -> code -> predict_4fold.py
This is used for performance and comparison of prediction models on AAP400 and independent test datasets (Figure 9A-B)
7. anti_aging -> code -> radar.py
This is used for drawing radar charts in this study.
  