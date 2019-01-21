# object-density-estimator
This is a proof of concept project meant to test the viability of a multi-column cnn for generating sacle aware object density heatmaps for object localization. In this case, we take only two columns, one with small kernel size and slightly more filters for detecting instances of the object that are further into the background and one with a larger kernel size and less filters for detecting large instances of the object.

Theoretically, concatenating both with a convolutional filter should take scale into account and provide a more accurate result that either one individually.

Files:

| File Name     | Explanation |
| ------------- | ----------- |
| img_preprocessing.py | Converts image and labels into the training data with ground truth heatmaps |
| save_data.ipynb | Takes images and applies preprocessing to them before saving them off |
| loss.py | Contains custom loss function |
| large_column_architecture.py | Defines the cnn model architecture of the large column |
| large_column_trainer.ipynb | Loads the large column architecture and trains a ranodmized network |
| small_column_architecture.py | Defineds the cnn model architecture of the small column |
| small_column_trainer.ipynb | Loads the small column architecture and trains a randomized network |
| multi-column-trainer.ipynb | Loads small and large column models and combines them into one model to be trained |
| tester.ipynb | Evaluates the quality of a trained model |

### Small Column:
![Small Column Diagram](https://i.imgur.com/haRX9yX.png)

### Large Column:
![Large Column Diagram](https://i.imgur.com/mvbIcxP.png)

### Multi Column:
![Mutli Column Diagram](https://i.imgur.com/XAXGXA3.png)
