# docker compose stop
# docker compose rm -f
docker compose build --no-cache
docker compose -f docker-compose.yaml up --force-recreate
# docker compose -f docker-compose.yaml up --force-recreate --detach