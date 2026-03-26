import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from models.tangent_model import TangentOperatorModel

DEVICE = 'cuda'
CHECKPOINT_PATH = "checkpoints/euclidean_ps7_hw8/best_model.pt"
DATA_PATH = "data_mixed_large/test.npz"
BATCH_SIZE = 32

dataset = TangentDataset(
    length=1024,
    family="euclidean",
    source="pregenerated",
    bank_path=DATA_PATH,
    patch_size=7,
    half_width=8,
    num_negatives=4,
    negative_min_offset=4,
    negative_max_offset=16,
    patch_mode="random_warp_symmetric",
    jitter_fraction=0.25,
    warp_sampling_prob=0.7,
    warp_sampling_strength=0.18,
    seed=123,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=tangent_collate_fn,
)

batch = next(iter(loader))

model = TangentOperatorModel(
    patch_size=7,
    operator_hidden_dims=[256, 256],
    signature_hidden_dims=[128, 64],
    signature_out_dim=64,
    signature_center_radius=0,
    head_dropout=0.0,
    normalize_projector=True,
    init_scale=0.05,
    center_operator=True,
    operator_bandwidth=None,
    learn_scale=False,
    centered_input_for_operator=True,
).to(DEVICE)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


# ====== MOVE BATCH ======
def move_batch(batch):
    batch.anchor = batch.anchor.to(DEVICE)
    batch.positive = batch.positive.to(DEVICE)
    batch.negatives = batch.negatives.to(DEVICE)
    batch.transform_matrix = batch.transform_matrix.to(DEVICE)
    batch.gt_first_anchor = batch.gt_first_anchor.to(DEVICE)
    batch.gt_second_anchor = batch.gt_second_anchor.to(DEVICE)
    return batch

batch = move_batch(batch)


# ====== FORWARD ======
with torch.no_grad():
    anchor_out = model(batch.anchor)
    positive_out = model(batch.positive)

    B, M, K, _ = batch.negatives.shape
    flat_neg = batch.negatives.view(B * M, K, 2)
    neg_out = model(flat_neg)


# ====== APPLY W TWICE ======
def apply_W(W, X):
    return torch.einsum('bij,bjd->bid', W, X)

anchor_field = anchor_out['field_first']
anchor_W2 = apply_W(anchor_out['operator'], anchor_field)


# ====== COSINE CHECK ======
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

center = K // 2

cos1 = cosine(anchor_field[:, center, :], batch.gt_first_anchor).mean().item()
cos2 = cosine(anchor_W2[:, center, :], batch.gt_second_anchor).mean().item()

print("\n=== COSINE CHECK ===")
print("First derivative cosine:", cos1)
print("Second derivative cosine:", cos2)


# ====== VISUALIZE ======
for i in range(3):
    pts = batch.anchor[i].cpu().numpy()
    f1 = anchor_field[i].cpu().numpy()
    f2 = anchor_W2[i].cpu().numpy()

    plt.figure(figsize=(12,4))

    # curve
    plt.subplot(1,3,1)
    plt.plot(pts[:,0], pts[:,1], '-o')
    plt.title("Curve")

    # W(X)X
    plt.subplot(1,3,2)
    plt.quiver(pts[:,0], pts[:,1], f1[:,0], f1[:,1])
    plt.title("W(X)X")

    # W^2(X)X
    plt.subplot(1,3,3)
    plt.quiver(pts[:,0], pts[:,1], f2[:,0], f2[:,1])
    plt.title("W^2(X)X")

    plt.tight_layout()
    plt.savefig(f"viz_{i}.png")
