from baseline_training import run_baseline
from bert_training import run_bert
from cnn_training import run_cnn
from lstm_training import run_lstm
from prefect import flow, task
from swivel_training import run_swivel


@task
def train_baseline():
    run_baseline()


@task
def train_lstm():
    run_lstm()


@task
def train_cnn():
    run_cnn()


# @task
# def train_bert():
#     run_bert()


@task
def train_swivel():
    run_swivel()


@flow(name="Training Flow", log_prints=True)
def training_flow():
    train_baseline()
    # train_bert()
    train_cnn()
    train_lstm()
    train_swivel()


if __name__ == "__main__":
    training_flow()
