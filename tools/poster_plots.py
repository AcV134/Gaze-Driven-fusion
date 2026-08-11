import matplotlib.pyplot as plt
import numpy as np

# Apply global styling tweaks
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

models = ['Gaze as Loss (Scratch)', 'Vanilla DISC (5 Epochs)']
colors = ['#1f77b4', '#2ca02c']

# --- DATA EXTRACTION FROM DOCUMENT ---

# Macro Metrics for Graphs 1 & 2
gaze_in_metrics = ['Gaze_BkmIoU', 'Gaze_InsmIoU', 'Gaze_mIoU']
data_gaze_in = {
    'Gaze as Loss (Scratch)':   [0.261056, 0.118597, 0.189826],
    'Vanilla DISC (5 Epochs)':  [0.262209, 0.115190, 0.188700]
}

gaze_out_metrics = ['Gaze_BkmIoU_out', 'Gaze_InsmIoU_out', 'Gaze_mIoU_out']
data_gaze_out = {
    'Gaze as Loss (Scratch)':   [0.248739, 0.105369, 0.177054],
    'Vanilla DISC (5 Epochs)':  [0.248126, 0.105237, 0.176681]
}

# Classes for Graphs 3 & 4
classes = [
    'bicycle', 'building', 'car', 'fence', 'motorcycle', 
    'other-ground', 'other-object', 'other-structure', 'other-vehicle', 
    'parking', 'person', 'pole', 'road', 'sidewalk', 
    'terrain', 'traffic-sign', 'truck', 'vegetation'
]

# Graph 3: In-Gaze Class-wise IoU (iou/2_in)
data_iou_in = {
    'Gaze as Loss (Scratch)': [
        0.003170, 0.422134, 0.381511, 0.095262, 0.043926, 0.066574, 
        0.077084, 0.220912, 0.123009, 0.195626, 0.048215, 0.070461, 
        0.543468, 0.280439, 0.140672, 0.044862, 0.275135, 0.384414
    ],
    'Vanilla DISC (5 Epochs)': [
        0.002502, 0.423742, 0.387356, 0.092270, 0.054215, 0.067769, 
        0.083045, 0.226759, 0.095084, 0.192666, 0.044340, 0.077265, 
        0.545113, 0.279958, 0.147738, 0.049965, 0.242936, 0.383869
    ]
}

# Graph 4: Out-Gaze Class-wise IoU (iou/3_out)
data_iou_out = {
    'Gaze as Loss (Scratch)': [
        0.001004, 0.370170, 0.274036, 0.081539, 0.029223, 0.060581, 
        0.109732, 0.142924, 0.070251, 0.152237, 0.074816, 0.127986, 
        0.583052, 0.333511, 0.113339, 0.057876, 0.203396, 0.401296
    ],
    'Vanilla DISC (5 Epochs)': [
        0.002264, 0.363293, 0.276686, 0.078146, 0.025383, 0.062311, 
        0.110078, 0.149024, 0.061588, 0.151774, 0.060543, 0.128349, 
        0.590004, 0.326895, 0.113369, 0.071674, 0.210571, 0.398317
    ]
}


# --- PLOTTING FUNCTIONS ---

def plot_macro_metrics(metrics, data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    x = np.arange(len(metrics))
    width = 0.28

    for i, model in enumerate(models):
        offset = x + (i - 0.5) * width
        rects = ax.bar(offset, data[model], width, label=model, color=colors[i], edgecolor='black', linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('IoU Score', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    
    max_val = max([max(v) for v in data.values()])
    ax.set_ylim(0, max_val * 1.25)
    
    ax.legend(fontsize=10, frameon=True, facecolor='white', loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_metrics(data, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    y = np.arange(len(classes))
    height = 0.35

    for i, model in enumerate(models):
        offset = y + (i - 0.5) * height
        ax.barh(offset, data[model], height, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('IoU Score', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Classes', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9.5)
    ax.invert_yaxis()
    
    max_val = max([max(v) for v in data.values()])
    ax.set_xlim(0, max_val * 1.15)
    
    ax.legend(fontsize=10, frameon=True, facecolor='white', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# --- GENERATE PLOTS ---
plot_macro_metrics(gaze_in_metrics, data_gaze_in, 'Graph 1: Comparison of In-Gaze Macro Metrics', 'graph1_scratch_vs_disc.png')
plot_macro_metrics(gaze_out_metrics, data_gaze_out, 'Graph 2: Comparison of Out-Gaze Macro Metrics', 'graph2_scratch_vs_disc.png')
plot_class_metrics(data_iou_in, 'Graph 3: Class-wise In-Gaze IoU (2_in)', 'graph3_scratch_vs_disc.png')
plot_class_metrics(data_iou_out, 'Graph 4: Class-wise Out-Gaze IoU (3_out)', 'graph4_scratch_vs_disc.png')