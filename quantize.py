import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sk_model = joblib.load("model.joblib")
weights = sk_model.coef_
bias = sk_model.intercept_

unquant_params = {
    'weights': weights,
    'bias': bias
}
joblib.dump(unquant_params, "unquant_params.joblib")

def quantize_signed(x):
    max_val = np.max(np.abs(x))
    scale = 127 / (max_val + 1e-8)
    qx = np.round(x * scale).astype(np.int8)
    return qx, scale

def dequantize_signed(qx, scale):
    return qx.astype(np.float32) / scale

quant_w, scale_w = quantize_signed(weights)
quant_b, scale_b = quantize_signed(np.array([bias]))

quant_params = {
    'quant_weights': quant_w,
    'scale_weights': scale_w,
    'quant_bias': quant_b,
    'scale_bias': scale_b
}
joblib.dump(quant_params, "quant_params.joblib")

dequant_w = dequantize_signed(quant_w, scale_w)
dequant_b = dequantize_signed(quant_b, scale_b)[0]

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

model = LinearModel(X.shape[1])
with torch.no_grad():
    model.linear.weight.copy_(torch.tensor(dequant_w).unsqueeze(0))
    model.linear.bias.copy_(torch.tensor([dequant_b]))

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).squeeze().numpy()

r2 = r2_score(y_test, preds)
print("Quantized PyTorch Model R² Score:", r2)

torch.save(model.state_dict(), "quantized_model.joblib")

