# Path: Makefile

start_prefect:
	prefect server start

training:
	cd tf_training && python training_flow.py

docker-tracking:
	docker compose  -f "docker-compose.yml" up -d --build mysql phpmyadmin s3 create_buckets mlflow

docker-tracking-down:
	docker compose  -f "docker-compose.yml" down

serve:
	cd tf_serving && docker compose -f "docker-compose.yml" up -d --build