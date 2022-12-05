# ComputerVisionMFGM

A classifier model using Convolutional Neural Networks.
It's trained on the [iMaterialist Challenge at FGVC 2017](https://www.kaggle.com/c/imaterialist-challenge-FGVC2017) dataset and mainly classifies if an image is a dress, shoe, pants, or outerwear.

In order to run, it requires:
- `Python 3.10`
- `numpy`
- `pandas`
- `keras`/`tensorflow`
- `scipy`

The `mfgm_plot.py` functions require `matplotlib`, `scikit-learn` and `seaborn`.

## Execution

A model can be trained by either running the `Run.py` script or the `Run.ipynb` notebook (recommended).

## Results

The model achieves up to ~70% accuracy on the task.

![Accuracy plot](graphs/apparel_vs_all_accuracy.png)

![Loss plot](graphs/apparel_vs_all_loss.png)
