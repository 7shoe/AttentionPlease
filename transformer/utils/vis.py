import yaml
from matplotlib import pyplot as plt

def plot_two_runs():
    """
    Plots two training runs (train and val CEL)
    """

    with open('./logs/1748141440.yaml', 'r') as f:
        dd = yaml.safe_load(f)
    
    with open('./logs/1748143215.yaml', 'r') as f:
        ss = yaml.safe_load(f)

    # training & validation curves for dd
    plt.plot(dd['training_log']['train_times'],
             dd['training_log']['train_losses'],
             label="dd train", linewidth=1)
    plt.plot(dd['training_log']['val_times'],
             dd['training_log']['val_losses'],
             label="dd val",   linewidth=1)
    
    # training & validation curves for ss
    plt.plot(ss['training_log']['train_times'],
             ss['training_log']['train_losses'],
             label="ss train", linewidth=1)
    plt.plot(ss['training_log']['val_times'],
             ss['training_log']['val_losses'],
             label="ss val",   linewidth=1)
    
    plt.title("Training runs of Tranformers")
    plt.xlabel("Timestamp [Unix s]")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()