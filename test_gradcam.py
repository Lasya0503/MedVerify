import tensorflow as tf
import numpy as np
import os
from keras.models import load_model # <-- Use this instead!

# 1. Load the COMPLETE model directly (no need to rebuild!)
model_path = os.path.join('app', 'models', 'Final_Balanced_Medicine_Model.keras')
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully in test_gradcam.py!")

# The MobileNetV2 base model is the first layer inside your Sequential model
base_model = model.layers[0]

# 2. Print last 10 layers of base model
print("\n📋 Last 10 base_model layers:")
for layer in base_model.layers[-10:]:
    print(f"  {layer.name:45s} | {type(layer).__name__}")

# 3. Test gradient flow
# We use model.inputs so the graph flows correctly from the very beginning to the very end
grad_model = tf.keras.models.Model(
    inputs=model.inputs, 
    outputs=[base_model.get_layer("out_relu").output, model.output]
)

dummy = tf.ones((1, 224, 224, 3), dtype=tf.float32)

with tf.GradientTape() as tape:
    # Watch the output of the conv layer to calculate gradients against it later
    conv_out, preds = grad_model(dummy, training=False)
    tape.watch(conv_out) 
    score = preds[:, 1] # Target class score

# Calculate gradients of the target class score with respect to the feature map
grads = tape.gradient(score, conv_out)

print(f"\n📊 Results:")
print(f"   conv_out shape : {conv_out.shape}")
print(f"   predictions    : {preds.numpy()}")
print(f"   grads          : {'None ❌' if grads is None else f'{tf.reduce_max(tf.abs(grads)).numpy():.8f}'}")