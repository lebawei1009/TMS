import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from NPI import ANN_RNN, ANN_MLP, ANN_CNN, ANN_VAR, train_NN, model_FC, model_EC, multi2one, device

# ---------------------- Configuration ----------------------
MODEL_TYPE = 'RNN'  # Options: 'RNN', 'MLP', 'CNN', 'VAR'
steps = 4
base_path = "D:/op/ANN/TMS_fMRI_2.0.0_AAL424/TMS_fMRI_2.0.0_TIS/resting"

for file in os.listdir(base_path):
    if not file.endswith(".npy"):
        continue

    file_path = os.path.join(base_path, file)
    data = np.load(file_path, allow_pickle=True)
    if data.ndim != 2:
        print(f"[Skipped] Invalid data shape: {file} -> shape={data.shape}")
        continue

    node_num = data.shape[1]
    input_X, target_Y = multi2one(data, steps)

    # ---------------------- Model Construction ----------------------
    if MODEL_TYPE == 'RNN':
        model = ANN_RNN(
            input_dim=node_num,
            hidden_dim=int(2.5 * node_num),
            latent_dim=int(2.5 * node_num),
            output_dim=node_num,
            data_length=steps
        )
    elif MODEL_TYPE == 'MLP':
        model = ANN_MLP(
            input_dim=steps * node_num,
            hidden_dim=2 * node_num,
            latent_dim=int(0.8 * node_num),
            output_dim=node_num
        )
    elif MODEL_TYPE == 'CNN':
        model = ANN_CNN(
            in_channels=node_num,
            hidden_channels=node_num,
            out_channels=int(0.8 * node_num),
            data_length=steps
        )
    elif MODEL_TYPE == 'VAR':
        model = ANN_VAR(
            input_dim=steps * node_num,
            output_dim=node_num
        )
    else:
        raise ValueError("Unsupported model type")

    # ---------------------- Model Training ----------------------
    trained_model, train_loss, test_loss = train_NN(model, input_X, target_Y)

    # ---------------------- Visualization Output Path ----------------------
    subject_id = os.path.splitext(file)[0]
    subject_dir = os.path.join(base_path, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # ---------------------- Loss Curve ----------------------
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label="train loss")
    plt.plot(test_loss, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{MODEL_TYPE} model training curve - {subject_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, f"{subject_id}_loss.png"))
    plt.close()

    # ---------------------- FC Plot ----------------------
    fc_matrix = model_FC(trained_model, node_num, steps)
    plt.figure(figsize=(8, 6))
    sns.heatmap(fc_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Functional Connectivity (FC) - {subject_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, f"{subject_id}_fc.png"))
    plt.close()

    # ---------------------- EC Plot ----------------------
    ec_matrix = model_EC(trained_model, input_X, target_Y, pert_strength=0.15)
    plt.figure(figsize=(8, 6))
    sns.heatmap(ec_matrix, cmap="bwr", vmin=-0.05, vmax=0.05)
    plt.title(f"Effective Connectivity (EC) - {subject_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, f"{subject_id}_ec.png"))
    plt.close()

    print(f"[✓] Visualization completed: {subject_id}")
