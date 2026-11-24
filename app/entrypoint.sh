#!/bin/sh
# Container start-up script.
# 1. If the models folder is empty (e.g. a brand new Kubernetes volume),
#    copy in the seed model that was trained when the image was built.
# 2. Start the Flask app with Gunicorn, a production web server.
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/model.joblib" ]; then
  echo "No model in $MODEL_DIR, copying seed model from the image"
  # copy to a temp name first, then rename, so several pods starting at once
  # never read a half-written file
  cp /app/seed_models/model.joblib "$MODEL_DIR/model.joblib.tmp.$$"
  cp /app/seed_models/metrics.json "$MODEL_DIR/metrics.json.tmp.$$"
  mv "$MODEL_DIR/model.joblib.tmp.$$" "$MODEL_DIR/model.joblib"
  mv "$MODEL_DIR/metrics.json.tmp.$$" "$MODEL_DIR/metrics.json"
fi

exec gunicorn \
  --bind "0.0.0.0:${PORT:-5000}" \
  --workers "${GUNICORN_WORKERS:-2}" \
  --threads "${GUNICORN_THREADS:-2}" \
  --timeout 60 \
  --access-logfile - \
  app:app
