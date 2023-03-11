# Docker compose to run Apache Airflow

* First define UID and GID for Airlow as environment variables

```sh
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
```

* Initialize postgres and redis, with airflow admin user:

```sh
docker-compose up airflow-init
```

* Then starts all the components

```sh
docker-compose up
```

* To use the airflow CLI, connect to the web server (or the worker or executor) container

```sh
docker exec airflow_airflow-webserver_1 airflow version
```

* For API use

```sh
curl -X GET --user "airflow:airflow" http://localhost:8080/api/v1/dags
```