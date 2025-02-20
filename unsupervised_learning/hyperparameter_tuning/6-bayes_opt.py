#!/usr/bin/env python3
"""Bayesian Optimization with GPyOpt for an
Autoencoder Hyperparameter Tuning Module"""
import numpy as np
import GPyOpt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


def load_data():
    """Load and prepare the MNIST2500 dataset."""
    X = np.loadtxt(
        "/root/atlas-machine_learning/unsupervised_learning/data/"
        "mnist2500_X.txt"
    ) / 255.0
    Y = np.loadtxt(
        "/root/atlas-machine_learning/unsupervised_learning/data/"
        "mnist2500_labels.txt"
    )

    # keep only digits 0-4 as normal data
    normal_data = X[np.isin(Y, [0, 1, 2, 3, 4])]

    # 80% training, 20% validation
    split = int(0.8 * len(normal_data))
    return normal_data[:split], normal_data[split:]


def build_autoencoder(input_dim, neurons, latent_dim, dropout_rate):
    """Build an Autoencoder model."""
    inputs = Input(shape=(input_dim,))

    encoded = Dense(neurons, activation="relu")(inputs)
    encoded = Dropout(dropout_rate)(encoded)
    encoded = Dense(latent_dim, activation="relu")(encoded)

    decoded = Dense(neurons, activation="relu")(encoded)
    decoded = Dropout(dropout_rate)(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    return Model(inputs, decoded)


def train_autoencoder(params):
    """Trains an Autoencoder with given hyperparameters."""
    (learning_rate, latent_dim, neurons,
     dropout_rate, batch_size, epochs) = params[0]

    autoencoder = build_autoencoder(
        X_train.shape[1], int(neurons), int(latent_dim), dropout_rate
    )
    autoencoder.compile(optimizer=Adam(learning_rate), loss="mse")

    checkpoint_path = (
        f"autoencoder_lr{learning_rate:.4f}_lat{int(latent_dim)}"
        f"_neurons{int(neurons)}_drop{dropout_rate:.2f}"
        f"_batch{int(batch_size)}_epochs{int(epochs)}.h5"
    )

    # train with early stopping and checkpoint saving
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True
            )
        ]
    )

    # return the best val loss
    return min(history.history["val_loss"])


# defining hyperparameter space
bounds = [
    {"name": "learning_rate", "type": "continuous", "domain": (0.0001, 0.01)},
    {"name": "latent_dim", "type": "discrete", "domain": (4, 8, 16, 32)},
    {"name": "neurons", "type": "discrete", "domain": (32, 64, 128, 256)},
    {"name": "dropout_rate", "type": "continuous", "domain": (0.1, 0.5)},
    {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)},
    {"name": "epochs", "type": "discrete", "domain": (20, 30, 40, 50, 60)}
]

# load datasets
X_train, X_val = load_data()

# run bayesian pptimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=train_autoencoder, domain=bounds, acquisition_type="LCB", maximize=False
)
optimizer.run_optimization(
    max_iter=30,
    report_file="bayes_opt.txt",
    evaluations_file="bayes_opt_evals.txt"
)

# plot convergence plot automatically
optimizer.plot_convergence("convergence.png")

# hyperparameter evolution over optimization steps
plt.figure(figsize=(10, 6))
iterations = np.arange(len(optimizer.X))

# plot hyperparameters
plt.scatter(iterations, optimizer.X[:, 0], label="Learning Rate", color="blue")
plt.scatter(iterations, optimizer.X[:, 1], label="Latent Dim", color="orange")
plt.scatter(iterations, optimizer.X[:, 2], label="Neurons", color="green")
plt.scatter(iterations, optimizer.X[:, 3], label="Dropout Rate", color="red")
plt.scatter(iterations, optimizer.X[:, 4], label="Batch Size", color="purple")

plt.xlabel("Iteration")
plt.ylabel("Hyperparameter Values")
plt.title("Hyperparameter Evolution Over Optimization Steps")
plt.legend()
plt.grid(True)
plt.savefig("hyperparam_evolution.png")
plt.show()

# saving the best hyperparameters
best_hyperparams = optimizer.X[np.argmin(optimizer.Y)]
with open("best_hyperparams.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    f.write(f"Learning Rate: {best_hyperparams[0]:.4f}\n")
    f.write(f"Latent Dimension: {int(best_hyperparams[1])}\n")
    f.write(f"Neurons: {int(best_hyperparams[2])}\n")
    f.write(f"Dropout Rate: {best_hyperparams[3]:.2f}\n")
    f.write(f"Batch Size: {int(best_hyperparams[4])}\n")
    f.write(f"Epochs: {int(best_hyperparams[5])}\n")
