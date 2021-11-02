# Tensorflow models LifeCycle managemenet [![Actions Status](https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/workflows/VerifyDockerCompose/badge.svg)](https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/actions)

This repository contains a docker-compose stack with MLflow, mysql, phpmyadmin and Minio. The networking is set up so running containers could communicate.

## Quickstart

The easiest way to understand the setup is by diving into it and interacting with it.

### 1. Clone the repository and create a virtual environment

```
git clone  https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow.git
cd Tensorflow-Lifecycle-management-with-MLFlow
```

This will clone the repo

```
pip install virtualenv
```
if you don't already have virtualenv installed
virtualenv venv to create your new environment (called 'venv' here)
```
source venv/bin/activate
```
to enter the virtual environment
```
pip install -r requirements.txt
```
to install the requirements in the current environment

### 2. Lunch the Docker Stack
make sure you have Docker installed
```
docker-compose build
```
this will build mlflow_server image
```
docker-compose up -d
```
after this, you have all container running so you can start training ML models

<img height="" src="https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/docker_stack.JPG" width="300"/>


#### Detail Summary

| **Container**     	| **Port**  	|
|---------------	|----------	|
| MLflow_server 	| 5000 	|
| Minio         	| 9000 	|
| Mysql         	| 80   	|
| phpmyadmin    	| 3306 	|

### 3. Start Training the ML models
```
python .\TensorFlow_training\baseline_training.py
```

```
 python .\TensorFlow_training\LSTM_training.py
```

```
 python .\TensorFlow_training\CNN_training.py
```

```
 python .\TensorFlow_training\swivel_training.py
```

```
 python .\TensorFlow_training\BERT_training.py
```


### 4. Visualize and compare models performances with MLflow UI to pick the best one
Access the MLflow Dashboard: http://localhost:5000

you can also access and query the database :  http://localhost:3306

the model artifacts are saved in minio, to access : http://localhost:9000

And finally to visualize tensorboard charts run the following command :
```
tensorboard --logdir .\TensorFlow_training\logs\fit
```
the open it in the browser http://localhost:6000


<table>
<tr>
<td style="width: 50%">
<h3>MLflow UI</h3>
<img src="https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/mlflow.JPG" alt="">
</td>
<td>
<h3>Mysql database in phpmyadmin</h3>
<img src="https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/mysql.JPG" alt="">
</td>
</tr>
<td style="width: 50%">
<h3>Models artifacts in Minio</h3>
<img src="https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/minio.JPG" alt="">
</td>
<td style="width: 50%">
<h3>tensorboard visualization</h3>
<img src="https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/tensorboard.JPG" alt="">
</td>
</table>


## Web app demo
demo![](https://github.com/amine-akrout/Tensorflow-Lifecycle-management-with-MLFlow/blob/main/demo/app_demo.png)