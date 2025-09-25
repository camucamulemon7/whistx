# docker compose stop
# docker compose rm -f
docker compose build --no-cache
docker compose up --force-recreate 
# docker compose -f docker-compose.yaml -f docker-compose.override.yaml up --force-recreate --detach