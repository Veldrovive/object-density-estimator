# object-density-estimator
This is a proof of concept project meant to test the viability of a multi-column cnn for generating sacle aware object density heatmaps for object localization. In this case, we take only two columns, one with small kernel size and slightly more filters for detecting instances of the object that are further into the background and one with a larger kernel size and less filters for detecting large instances of the object.

Theoretically, concatenating both with a convolutional filter should take scale into account and provide a more accurate result that either one individually.

### Small Column:
![Small Column Diagram](https://i.imgur.com/haRX9yX.png)

### Large Column:
![Large Column Diagram](https://i.imgur.com/mvbIcxP.png)

### Multi Column:
![Mutli Column Diagram](https://i.imgur.com/XAXGXA3.png)
