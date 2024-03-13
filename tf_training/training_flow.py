"""
Training Flow
"""

from baseline_training import run_baseline
from bert_training import run_bert
from cnn_training import run_cnn
from lstm_training import run_lstm
from model_selection import select_best_model
from prefect import flow, task
from swivel_training import run_swivel


@task
def train_baseline():
    """Train the baseline model"""
    run_baseline()


@task
def train_lstm():
    """Train the LSTM model"""
    run_lstm()


@task
def train_cnn():
    """Train the CNN model"""
    run_cnn()


@task
def train_bert():
    """Train the BERT model"""
    run_bert()


@task
def train_swivel():
    """Train the Swivel model"""
    run_swivel()


@task
def select_best_model_task():
    """Select the best model"""
    select_best_model()


@flow(name="Training Flow", log_prints=True)
def training_flow():
    """Training Flow"""
    train_baseline()
    train_bert()
    train_cnn()
    train_lstm()
    train_swivel()
    select_best_model_task()


if __name__ == "__main__":
    training_flow()
