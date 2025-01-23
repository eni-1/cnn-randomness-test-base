import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                   BatchNormalization, Input, Concatenate, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import zlib

SIZE = 8
SEQ_LEN = 64
N_SAMPLES = 5000
BATCH = 32
EPOCHS = 100
LR = 0.0001
THRESH = 0.5
N_SEQ = 3
FRAMES_PER_SEQ = 200
PAUSE_FRAMES = 50
TOTAL_FRAMES = N_SEQ * (FRAMES_PER_SEQ + PAUSE_FRAMES)
FRAME_INTERVAL = 50


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.Wq = self.add_weight("Wq", (dim, dim), "glorot_uniform", True)
        self.Wk = self.add_weight("Wk", (dim, dim), "glorot_uniform", True)
        self.Wv = self.add_weight("Wv", (dim, dim), "glorot_uniform", True)
        super().build(input_shape)

    def call(self, x):
        q = tf.keras.backend.dot(x, self.Wq)
        k = tf.keras.backend.dot(x, self.Wk)
        v = tf.keras.backend.dot(x, self.Wv)
        scores = tf.keras.backend.dot(q, tf.transpose(k))
        scores = scores / tf.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        weights = tf.nn.softmax(scores)
        return tf.keras.backend.dot(weights, v)


def get_base_stats(mat):
    features = []
    rows, cols = mat.shape
    vals = np.bincount(mat.flatten(), minlength=2)
    total = rows * cols
    p1 = vals[1] / total
    p0 = vals[0] / total
    features.extend([
        abs(0.5 - p1),
        abs(p1 - p0),
        -p1 * np.log2(p1) - p0 * np.log2(p0) if p1 > 0 and p0 > 0 else 0,
        max(p1, p0)
    ])

    win_size = 4
    win_bias = []
    for i in range(rows - win_size + 1):
        for j in range(cols - win_size + 1):
            win = mat[i:i + win_size, j:j + win_size]
            win_bias.append(abs(0.5 - np.sum(win) / win.size))
    features.extend([np.mean(win_bias), np.max(win_bias), np.std(win_bias)])

    def get_runs(arr):
        runs = []
        curr = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                curr += 1
            else:
                runs.append(curr)
                curr = 1
        runs.append(curr)
        return runs
    runs_h = [get_runs(row) for row in mat]
    runs_v = [get_runs(col) for col in mat.T]
    all_runs = [r for runs in runs_h + runs_v for r in runs]
    features.extend([
        np.mean(all_runs),
        np.std(all_runs) if len(all_runs) > 1 else 0,
        max(all_runs),
        len([r for r in all_runs if r > 2]) / len(all_runs) if all_runs else 0
    ])

    return np.array(features)


def adv_stats(mat):
    features = []
    fft = np.abs(np.fft.fft2(mat))
    features.extend([np.mean(fft), np.std(fft)])

    def box_count(mat, size):
        if size == 0:
            return 0
        boxes = mat.reshape(mat.shape[0] // size, size,
                          mat.shape[1] // size, size)
        return np.sum(np.any(boxes, axis=(1, 3)))
    sizes = [1, 2, 4]
    counts = [box_count(mat, s) for s in sizes]
    eps = 1e-10
    try:
        frac_dim = np.polyfit(np.log(np.array(sizes) + eps),
                             np.log(np.array(counts) + eps), 1)[0]
        features.append(abs(frac_dim))
    except:
        features.append(0.0)
    str_mat = ''.join(map(str, mat.flatten()))
    features.append(len(zlib.compress(str_mat.encode())) / len(str_mat))
    return np.array(features)


def return_stats(mat):
    return np.concatenate([get_base_stats(mat), adv_stats(mat)])


def model(conv_shape, stats_shape):
    reg = l2(0.001)
    conv_in = Input(shape=conv_shape)
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(conv_in)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x1)
    x2 = BatchNormalization()(x2)
    x = GlobalAveragePooling2D()(x2)
    stats_in = Input(shape=stats_shape)
    sx = Dense(32, activation='relu', kernel_regularizer=reg)(stats_in)
    sx = BatchNormalization()(sx)

    def apply_attn(x):
        weights = Dense(x.shape[-1], activation='softmax', kernel_regularizer=reg)(x)
        return weights * x
    x = apply_attn(x)
    sx = apply_attn(sx)
    combined = Concatenate()([x, sx])
    x = Dense(64, activation='relu', kernel_regularizer=reg,
              kernel_initializer='he_normal')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=reg,
                kernel_initializer='glorot_normal')(x)

    return Model(inputs=[conv_in, stats_in], outputs=out)


def add_noise(mat, level=0.1):
    noise = np.random.random(mat.shape) < level
    mat = mat.copy()
    mat[noise] = 1 - mat[noise]
    return mat.astype(np.int32)


def gen_data(n_samples):
    rand_data = [np.random.randint(0, 2, (8, 8)) for _ in range(n_samples)]
    nonrand_data = []
    biases = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    for _ in range(n_samples // 3):
        bias = np.random.choice(biases)
        mat = np.random.choice([0, 1], (8, 8), p=[bias, 1 - bias])
        nonrand_data.append(add_noise(mat, 0.05))
    patterns = [
        np.tile([0, 1], (8, 4)),
        np.tile([[0] * 8, [1] * 8], (4, 1)),
        np.array([[int((i + j) % 2) for j in range(8)] for i in range(8)])
    ]
    nonrand_data.extend([add_noise(p) for p in patterns])
    remaining = n_samples - len(nonrand_data)
    for _ in range(remaining):
        mat = np.zeros((8, 8), dtype=np.int32)
        pat_type = np.random.choice(['block', 'diag', 'repeat', 'bias_block'])
        if pat_type == 'block':
            size = np.random.choice([2, 4])
            for i in range(0, 8, size):
                for j in range(0, 8, size):
                    mat[i:i + size, j:j + size] = np.random.choice([0, 1])
        elif pat_type == 'diag':
            for i in range(8):
                mat[i, i] = mat[i, 7 - i] = 1
        elif pat_type == 'repeat':
            base = np.random.randint(0, 2, (2, 2))
            mat = np.tile(base, (4, 4))
        else:
            for i in range(0, 8, 4):
                for j in range(0, 8, 4):
                    bias = np.random.choice([0.2, 0.8])
                    mat[i:i + 4, j:j + 4] = np.random.choice([0, 1], (4, 4), p=[bias, 1 - bias])
        nonrand_data.append(mat)

    return rand_data, nonrand_data


def plt_importance(model, X_test_conv, X_test_stats, feature_names):
    conv_input = tf.convert_to_tensor(X_test_conv[:100], dtype=tf.float32)
    stats_input = tf.convert_to_tensor(X_test_stats[:100], dtype=tf.float32)
    all_grads = []

    for i in range(10):
        with tf.GradientTape() as tape:
            stats = stats_input[i:i + 10]
            conv = conv_input[i:i + 10]
            tape.watch(stats)
            preds = model([conv, stats])
        grads = tape.gradient(preds, stats)
        all_grads.append(tf.abs(grads))

    importance = tf.reduce_mean(tf.concat(all_grads, axis=0), axis=0)
    scores = importance.numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    plt.figure(figsize=(12, 8))
    sorted_features = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)
    bars = plt.barh(range(len(names)), values, color='skyblue')
    plt.yticks(range(len(names)), names, fontsize=10)
    plt.title('Feature Importance', fontsize=14, pad=20)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def animate_frame(frame):
    total_frames = FRAMES_PER_SEQ + PAUSE_FRAMES
    seq_frame = frame % total_frames
    curr_seq = min(frame // total_frames, N_SEQ - 1)
    seq = sequences[curr_seq]
    matrix = matrices[curr_seq]
    pred_val, is_random = predictions[curr_seq]
    in_pause = seq_frame >= FRAMES_PER_SEQ
    seq_frame = min(seq_frame, FRAMES_PER_SEQ - 1)
    flips = min(SEQ_LEN, int(SEQ_LEN * (seq_frame + 1) / FRAMES_PER_SEQ))
    if in_pause:
        flips = SEQ_LEN

    for i in range(SEQ_LEN):
        if i < flips:
            coin_circles[i].set_data([i], [seq[i]])
            coin_circles[i].set_color('red' if seq[i] == 0 else 'green')
        else:
            coin_circles[i].set_data([], [])

    if flips == 0:
        display_mat = np.array([[]])
    else:
        row = (flips - 1) // SIZE
        col = (flips - 1) % SIZE
        display_mat = np.zeros((row + 1, SIZE))
        if row > 0:
            display_mat[:row, :] = matrix[:row, :]
        display_mat[row, :col + 1] = matrix[row, :col + 1]
    ax2.clear()
    ax2.set_title("Matrix Representation")
    ax2.set_xlim(0, SIZE)
    ax2.set_ylim(0, SIZE)

    if display_mat.size > 0:
        sns.heatmap(display_mat, cmap=['red', 'green'], cbar=False,
                   ax=ax2, linewidths=0.5, linecolor='black')
    if flips == SEQ_LEN:
        prob_text.set_text(f"Probability of Randomness: {pred_val:.4f}")
        pred_text.set_text("Likely Random" if is_random else "Likely Not Random")
        pred_text.set_color('blue' if is_random else 'red')
    else:
        prob_text.set_text(f"Analyzing sequence... ({int(flips / SEQ_LEN * 100)}%)")
        pred_text.set_text("Analyzing...")
        pred_text.set_color('black')

    return [prob_text, pred_text] + coin_circles


def setup_animation():
    global fig, ax1, ax2, ax3, coin_circles, prob_text, pred_text
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axs
    ax1.set_title("Sequence")
    ax1.set_xlim(0, SEQ_LEN)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['0', '1'])
    ax1.set_xticks([])
    coin_circles = [ax1.plot([], [], 'o', markersize=8, color='lightgray')[0]
                   for _ in range(SEQ_LEN)]
    ax2.set_title("Matrix Representation")
    sns.heatmap(np.zeros((SIZE, SIZE)), cmap=['red', 'green'],
                cbar=False, ax=ax2, linewidths=0.5, linecolor='black')
    ax3.set_title("Prediction")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    prob_text = ax3.text(0.5, 0.6, "Waiting for sequence...",
                        ha='center', fontsize=12)
    pred_text = ax3.text(0.5, 0.4, "Analyzing...",
                        ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def gen_sequences():
    global sequences, matrices, predictions
    sequences = []
    matrices = []
    predictions = []

    for i in range(N_SEQ):
        if i == 0:
            seq = np.random.randint(0, 2, size=SEQ_LEN).astype(np.int32)
        elif i == 1:
            base = np.tile([0, 1], SEQ_LEN // 2 + 1)[:SEQ_LEN].astype(np.int32)
            seq = base.copy()
            for j in range(2, SEQ_LEN, 4):
                if np.random.random() < 0.15:
                    seq[j] = seq[j - 1]
        else:
            seq = np.random.choice([0, 1], size=SEQ_LEN, p=[0.7, 0.3]).astype(np.int32)

        sequences.append(seq)
        matrix = seq.reshape(SIZE, SIZE)
        matrices.append(matrix)
        pred_val, is_random = predict(matrix)
        predictions.append((pred_val, is_random))
        print(f"Sequence {i + 1} prediction: {pred_val:.4f} "
              f"({'Random' if is_random else 'Not Random'})")


def predict(matrix):
    conv_input = np.expand_dims(np.expand_dims(matrix, 0), -1)
    stats_input = np.expand_dims(return_stats(matrix), 0)
    pred = model.predict([conv_input, stats_input], verbose=0)[0][0]
    return pred, pred > THRESH


def main():
    global model, sequences, matrices, predictions
    rand_data, nonrand_data = gen_data(N_SAMPLES)
    conv_data = np.array(rand_data + nonrand_data)[..., np.newaxis]
    stats_data = np.array([return_stats(m) for m in rand_data + nonrand_data])
    labels = np.concatenate([np.ones(N_SAMPLES), np.zeros(N_SAMPLES)])
    X_train_conv, X_test_conv, X_train_stats, X_test_stats, y_train, y_test = train_test_split(
        conv_data, stats_data, labels, test_size=0.2, stratify=labels, random_state=1
    )
    model = model((SIZE, SIZE, 1), (stats_data.shape[1],))
    opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            'val_accuracy', patience=50, min_delta=0.001, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            'val_loss', factor=0.5, patience=25, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras', 'val_accuracy', save_best_only=True)
    ]
    history = model.fit(
        [X_train_conv, X_train_stats],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.2,
        callbacks=callbacks
    )
    model.save('model.keras')
    feature_names = [
        'Bias', 'Mean Runs', 'Std Runs', 'Max Runs', 'Long Runs Ratio',
        'Pattern2x2', 'Pattern3x3', 'Pattern4x4', 'Entropy Mean', 'Entropy Std',
        'Mean FFT Mag', 'Std FFT Mag', 'Fractal Dim', 'Complexity'
    ]
    plt_importance(model, X_test_conv, X_test_stats, feature_names)
    gen_sequences()
    fig = setup_animation()
    try:
        ani = animation.FuncAnimation(
            fig, animate_frame, frames=TOTAL_FRAMES,
            interval=FRAME_INTERVAL, blit=True
        )
        ani.save("animation.gif", writer='pillow', fps=30)
    except Exception as e:
        print(f"Animation error: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
