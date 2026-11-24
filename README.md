# BTC 7-Day Forecast: Production MLOps Pipeline

Flask API + Docker + GitHub Actions + Airflow + Kubeflow + Kubernetes (Kind)

This README is also a **beginner's step-by-step guide**. Follow it top to bottom.
Every step says what to run, what you should see, and which screenshot to take
for the report (look for 📸).

---

## 0. The big picture (read this first)

```
            every day                                         you / users
   ┌──────────────────────────┐                                    │
   │  Airflow DAG             │                                    ▼
   │  1. fetch BTC prices     │      models/        ┌──────────────────────────┐
   │  2. build features       │ ───► model.joblib ─►│  Flask API (3+ pods)     │
   │  3. train + evaluate     │      metrics.json   │  /health /predict        │
   │  4. register if good     │                     │  /metrics /reload        │
   │  5. tell API to reload ──┼────────────────────►│  on Kubernetes           │
   └──────────────────────────┘                     └──────────────────────────┘
                                                              ▲
   git push ──► GitHub Actions: lint ► test ► build image ► push to Docker Hub
```

**Words you'll see a lot:**

| Word | Plain-English meaning |
|---|---|
| **API** | A program that answers web requests. Ours answers "what will BTC cost next week?" |
| **Flask** | A Python library for writing APIs. |
| **Gunicorn** | A "production" web server that runs the Flask app reliably and handles many requests at once. |
| **Docker image** | A packaged box containing your code + Python + libraries. Runs the same on every computer. |
| **Container** | A running copy of an image. |
| **Docker Compose** | Starts several containers together from one file (`docker-compose.yml`). |
| **CI/CD** | Automatic checks (CI) and automatic delivery (CD) every time you push code to GitHub. |
| **Airflow** | A scheduler that runs a list of tasks in order, every day, and shows you which ones passed. A "DAG" is that list of tasks. |
| **Kubeflow Pipelines** | Like Airflow, but each step runs as a container on Kubernetes and ML artifacts are tracked. |
| **Kubernetes (K8s)** | Runs many containers across machines, restarts them if they crash, and scales them up/down. |
| **Kind** | "Kubernetes IN Docker": a tiny Kubernetes cluster running on your laptop inside Docker. |
| **kubectl** | The command-line tool you use to talk to Kubernetes. |
| **Pod** | The smallest thing Kubernetes runs: one (or a few) containers. |
| **Manifest** | A YAML file describing what you want Kubernetes to run. `kubectl apply -f file.yaml` makes it happen. |

**The model:** a gradient-boosted trees model (scikit-learn `HistGradientBoostingRegressor`,
same family as XGBoost). It looks at the last 30 days of price behaviour (returns,
moving averages, volatility) and predicts the % change for each of the next 7 days.
It's checked against the most recent 7 real days and against a naive "price stays
the same" baseline.

---

## 1. Project structure

```
app/app.py                  Flask API (/health, /predict, /metrics, /reload)
app/Dockerfile              Multi-stage Dockerfile for the API
app/entrypoint.sh           Container start script (copies seed model, starts gunicorn)
src/etl_fetch.py            Download daily BTC-USD data (yfinance, CoinGecko backup)
src/features.py             Cleaning + feature engineering
src/train_forecast.py       Train, evaluate (MAE/RMSE/MAPE), register model
dags/btc_etl_pipeline.py    Airflow DAG
kubeflow/btc_kfp_pipeline.py   Kubeflow pipeline (KFP v2) -> compiles to .yaml
airflow/Dockerfile          Airflow image with our ML libraries
docker-compose.yml          API + Airflow + Postgres for local testing
.github/workflows/ci_cd.yml GitHub Actions: lint, test, build, push
k8s/                        Kubernetes manifests (+ kind-config.yaml)
loadtest/locustfile.py      Load test for the autoscaling demo
tests/                      pytest unit tests
```

---

## 2. Install the tools (do this once)

> **Windows users:** install **Git for Windows** and run every command in this guide
> in **Git Bash** (right-click in a folder → "Open Git Bash here"). Then the commands
> are identical to Mac/Linux.

| Tool | Get it from | Check it works |
|---|---|---|
| Git | https://git-scm.com/downloads | `git --version` |
| Python **3.10, 3.11 or 3.12** | https://www.python.org/downloads/ (tick "Add to PATH" on Windows) | `python --version` |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ | `docker run hello-world` |
| kubectl | https://kubernetes.io/docs/tasks/tools/ | `kubectl version --client` |
| Kind | https://kind.sigs.k8s.io/docs/user/quick-start/#installation | `kind version` |
| Helm (optional) | https://helm.sh/docs/intro/install/ | `helm version` |

Windows shortcut (PowerShell as admin): `winget install Kubernetes.kubectl Kubernetes.kind Helm.Helm`

**Docker Desktop memory:** Settings → Resources → set Memory to **at least 6 GB**
(8 GB is better). Airflow and Kubernetes together are heavy.

Accounts you need: **GitHub** and **Docker Hub** (both free).

---

## 3. Part A: Run the API on your laptop

```bash
cd btc-mlops

# 1) Create an isolated Python environment (keeps libraries for this project separate)
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements-dev.txt

# 2) Run the pipeline steps by hand
python src/etl_fetch.py            # downloads BTC prices -> data/raw/btc_usd.csv
python src/features.py             # builds features    -> data/processed/btc_features.csv
python src/train_forecast.py       # trains + saves     -> models/model.joblib, metrics.json

# 3) Run the unit tests
pytest -v                          # 📸 screenshot: all tests passed

# 4) Start the API with Gunicorn (Mac/Linux)
cd app && gunicorn --bind 0.0.0.0:5000 app:app
# Windows: Gunicorn doesn't run on Windows, use:  python app/app.py
# (Gunicorn runs inside Docker later, which is what counts for production)
```

In a **second terminal**, try the endpoints:

```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"forecast_days": 3}'
curl http://localhost:5000/metrics
# error handling demo (should return 400):
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"forecast_days": 99}'
```

📸 Screenshot the `/predict` and `/metrics` responses and the 400 error.

Stop the server with `Ctrl+C`.

---

## 4. Part B: Docker, Docker Compose, CI/CD

### B1. Build the image

```bash
docker build -f app/Dockerfile -t btc-forecast-api:local .
docker images btc-forecast-api                # 📸 shows image size
docker run --rm -p 5000:5000 btc-forecast-api:local
```
The build downloads data and trains a "seed" model inside the image, so a new
container can answer predictions immediately. 📸 Screenshot the end of the build output.

### B2. Docker Compose (API + Airflow + Postgres)

```bash
cp .env.example .env            # Linux only: edit .env and set AIRFLOW_UID to the output of `id -u`
docker compose up -d --build    # first time: 5-10 minutes
docker compose ps               # 📸 all services "running"/"healthy" (airflow-init shows "exited (0)", that's correct)
```
- API: http://localhost:5000/health
- Airflow UI: http://localhost:8080 (login **airflow / airflow**)

### B3/B4. GitHub Actions

1. Create a **new empty repository** on GitHub (e.g. `btc-mlops`).
2. Create a Docker Hub **access token**: Docker Hub → Account settings → Personal access tokens → Generate (Read & Write).
3. In the GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**, add:
   - `DOCKER_USERNAME` = your Docker Hub username
   - `DOCKER_PASSWORD` = the access token from step 2
4. Push the code:
   ```bash
   git init
   git add .
   git commit -m "BTC MLOps pipeline"
   git branch -M main
   git remote add origin https://github.com/<your-username>/btc-mlops.git
   git push -u origin main
   ```
5. Open the **Actions** tab. You'll see lint → test → build-and-push.
   📸 Screenshot the green pipeline, and your image on Docker Hub showing the `latest` and commit-SHA tags.

---

## 5. Part C: Airflow DAG and Kubeflow pipeline

### C1. Airflow
1. Open http://localhost:8080 → find **btc_etl_pipeline**.
2. Turn the toggle **on** (unpause), then click ▶ **Trigger DAG**.
3. Click the run → **Graph** view. Boxes turn dark green as tasks succeed.
   📸 Screenshot the Graph view with all tasks green.
4. Click `train_and_evaluate` → **Logs** to see MAE/RMSE/MAPE. 📸
5. Check the API picked up the new model: `curl http://localhost:5000/health`
   (the `model_version` changed).

DAG integrity test inside the container:
```bash
docker compose exec airflow-scheduler pytest /opt/airflow/tests/test_dag.py -v
```

### C2. Kubeflow
```bash
pip install -r kubeflow/requirements.txt
python kubeflow/btc_kfp_pipeline.py      # writes kubeflow/btc_kfp_pipeline.yaml
```
📸 Screenshot the command output and the start of the YAML file. That compiled YAML
is the deliverable. Installing full Kubeflow needs lots of RAM and isn't required.

---

## 6. Part D: Kubernetes on Kind

```bash
# 1) Create the cluster (port mappings let the Ingress and NodePort work)
kind create cluster --name mlops --config k8s/kind-config.yaml
kubectl get nodes                                     # STATUS = Ready

# 2) Install the NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait -n ingress-nginx --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller --timeout=180s

# 3) Install metrics-server (the HPA needs CPU numbers from it)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

**Point the deployment at your image.** Replace `YOUR_DOCKERHUB_USERNAME` in
`k8s/deployment.yaml` with your Docker Hub username (any text editor works).
*No Docker Hub yet?* Use the local image instead:
`kind load docker-image btc-forecast-api:local --name mlops` and set
`image: btc-forecast-api:local` in the file.

```bash
# 4) Deploy everything
kubectl apply -f k8s/configmap-secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

kubectl rollout status deployment/btc-forecast-api
kubectl get pods,svc,pvc,hpa,ingress -o wide          # 📸 the key screenshot
```

**Add the Ingress hostname** to your hosts file:
- Windows: open Notepad **as administrator**, open `C:\Windows\System32\drivers\etc\hosts`
- Mac/Linux: `sudo nano /etc/hosts`

Add this line: `127.0.0.1 btc-forecast.local`

```bash
curl http://localhost:30080/health                    # via NodePort
curl -X POST http://btc-forecast.local/predict        # via Ingress  📸
```

### D5. Autoscaling demo
Terminal 1:
```bash
kubectl get hpa btc-forecast-api-hpa --watch
```
Terminal 2:
```bash
locust -f loadtest/locustfile.py --host http://localhost:30080 \
  --headless -u 100 -r 10 -t 3m --csv loadtest/results
```
Within 1-2 minutes CPU goes above 60% and REPLICAS rises toward 8. After the test
stops, it scales back down. 📸 Screenshot the `--watch` output and `kubectl get pods`.
For a chart for the report, run Locust without `--headless`, open http://localhost:8089,
start the test, and screenshot the Charts tab.

Note: the HPA's minimum is 2, so when there's no traffic it may scale the 3 starting pods down to 2. That's expected.

---

## 7. Part E: Verification

```bash
# E1. Self-healing: delete a pod and watch a new one appear
kubectl get pods
kubectl delete pod <one-pod-name>
kubectl get pods -w                       # 📸 new pod ContainerCreating -> Running

# Model survives restarts because it's on the PVC:
kubectl exec deploy/btc-forecast-api -- ls -l /app/models

# E2. End-to-end
#  - Airflow DAG run green (section 5)            📸
#  - GitHub Actions run green (section 4)         📸
#  - Live prediction from Kubernetes:             📸
curl -X POST http://btc-forecast.local/predict -H "Content-Type: application/json" -d '{"forecast_days": 7}'
```

### Screenshot checklist for the PDF report
- [ ] `pytest -v` passing
- [ ] API responses: /health, /predict, /metrics, a 400 error
- [ ] `docker build` output + `docker images` size
- [ ] `docker compose ps`
- [ ] GitHub Actions green run + Docker Hub tags
- [ ] Airflow Graph view (all green) + training log with metrics
- [ ] Kubeflow compile output / YAML
- [ ] `kubectl get pods,svc,pvc,hpa,ingress`
- [ ] HPA scaling up under load + Locust chart
- [ ] Pod deletion and self-healing
- [ ] 7-day prediction from the live Kubernetes endpoint

---

## 8. Common problems

| Problem | Fix |
|---|---|
| `entrypoint.sh: no such file or directory` when the container starts | Windows line endings. Run `git add --renormalize . && git commit -m "fix eol"` and rebuild. `.gitattributes` prevents this for new clones. |
| yfinance error / empty data | Yahoo sometimes rate-limits. The code falls back to CoinGecko automatically. Just retry. |
| HPA shows `<unknown>/60%` | metrics-server isn't ready. Wait 1-2 minutes. Check that the `kubectl patch` command above ran. |
| Pods stuck `ImagePullBackOff` | Wrong image name in `deployment.yaml`, or the image isn't public on Docker Hub. |
| Pods stuck `Pending` | PVC not bound: `kubectl get pvc`. Kind's `standard` class binds when the first pod starts. |
| `btc-forecast.local` doesn't resolve | The hosts file line is missing, or the file was saved without admin rights. |
| Port 80 already in use when creating the Kind cluster | Another program (IIS, Skype, Apache) uses port 80. Stop it, or delete `hostPort: 80` from `kind-config.yaml` and use the NodePort only. |
| Airflow very slow or containers restarting | Give Docker Desktop more memory (Settings → Resources). |
| CI `black --check` fails | Run `black .` locally, commit, push again. |

## 9. Clean up
```bash
docker compose down -v
kind delete cluster --name mlops
```
