# Training a Graph Attention Network (GAT) on the PROTEINS Dataset

## Overview
This project demonstrates how to train a Graph Attention Network (GAT) on the PROTEINS dataset using PyTorch. The GAT architecture is enhanced with modern techniques like skip connections, layer normalization, and global attention pooling. The pipeline includes dataset preparation, model implementation, training, and evaluation.

---

## Dataset
We use the PROTEINS dataset, a graph-based dataset from the `graphs-datasets` library. Each graph represents a protein structure, where:
- **Nodes** correspond to amino acids.
- **Edges** represent interactions between amino acids.
- **Labels** categorize the proteins into classes.

**Steps:**
1. Load the dataset with `datasets.load_dataset`.
2. Extract labels and compute class weights to handle imbalanced data.
3. Split the data into training and testing subsets.

---

## Model Architecture
The GAT model consists of:
1. **Input Normalization:** Batch normalization for input features.
2. **Graph Attention Layers:** Multi-head attention layers to learn node representations.
3. **Global Attention Pooling:** Aggregates node-level features for graph-level classification.
4. **Classifier:** A fully connected network for predicting class labels.

**Key Features:**
- **Skip Connections:** Improves gradient flow and reduces vanishing gradients.
- **Layer Normalization:** Ensures stable learning dynamics.
- **Dropout:** Mitigates overfitting.

---

## Training Pipeline
1. **Hyperparameters:**
   - Hidden dimensions: 256
   - Attention heads: 16
   - Dropout: 0.2
   - Learning rate: 0.0001
   - Epochs: 200

2. **Loss Function:**
   - Weighted cross-entropy to address class imbalance.

3. **Optimizer:**
   - Adam optimizer with optional gradient clipping for stability.

4. **Evaluation:**
   - Training accuracy and loss are logged.
   - Testing is performed every 5 epochs to monitor generalization.
   - Best model weights are saved based on test accuracy.

---

## Results
The best test accuracy achieved during training is logged, along with detailed metrics like precision, recall, and F1-score. The classification report provides insights into model performance across different classes.

---

## Usage
To run the project:
1. Clone the repository and install dependencies.
2. Load the PROTEINS dataset using the `datasets` library.
3. Run the training script:

```bash
python train_gat.py
```

4. Save and evaluate the best model:

```bash
python evaluate_gat.py
```

---

## Future Work
- Experiment with graph augmentation techniques (e.g., edge dropout).
- Introduce a validation split for hyperparameter tuning.
- Optimize batch processing for faster training.

---

## Acknowledgments
- **Dataset:** Provided by `graphs-datasets`.
- **Frameworks:** PyTorch and Scikit-learn for model implementation and evaluation.

---

Connect with me on [LinkedIn](https://linkedin.com) to discuss this project or explore collaboration opportunities!

