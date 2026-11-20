import numpy as np

model = np.load("trained_model.npz", allow_pickle = True)

mean = model["scaler_mean"]
scale = model["scaler_scale"]

W1 = model["W1"]
B1 = model["B1"]
W2 = model["W2"]
B2 = model["B2"]

classes = model["classes"]


def write_array(file, name, array):
    flat = array.flatten()

    file.write(f"const float {name}[{len(flat)}] = {{\n")

    for i, value in enumerate(flat):
        file.write(f"{value:.8f}f")

        if i != len(flat) - 1:
            file.write(", ")

            if (i + 1) % 8 == 0:
                file.write("\n")

    file.write("\n};\n\n")

with open("ModelWeights.h", "w") as file:

    file.write("#pragma once\n\n")

    file.write(f"#define MODEL_INPUT_SIZE {W1.shape[0]}\n")

    file.write(f"#define MODEL_HIDDEN_SIZE {W1.shape[1]}\n")

    file.write(f"#define MODEL_OUTPUT_SIZE {W2.shape[1]}\n\n")

    write_array(file, "MODEL_MEAN", mean)

    write_array(file, "MODEL_SCALE", scale)

    write_array(file, "MODEL_W1", W1)

    write_array(file, "MODEL_B1", B1)

    write_array(file, "MODEL_W2", W2)

    write_array(file, "MODEL_B2", B2)

print("Classes:", classes)
print("W1 shape:", W1.shape)
print("W2 shape:", W2.shape)
print("Created ModelWeights.h")