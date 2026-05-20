# Deploying ECH0-PRIME to Azure Kubernetes Service (AKS)

This directory contains the Kubernetes manifests needed to deploy ECH0-PRIME to your existing Microsoft Azure Kubernetes cluster that is used for BBB (Business Automation).

## Prerequisites

1. Your AKS cluster is running and `kubectl` is configured to connect to it.
2. You have a container registry (like Azure Container Registry - ACR) where you can push the Docker image.
3. You have logged into Azure CLI: `az login`

## Deployment Steps

### 1. Create the Namespace

First, create the namespace used by these manifests:

```bash
kubectl create namespace bbb-production
```

### 2. Build and Push the Docker Image

You need to build the Docker image and push it to your registry. If using Azure Container Registry (ACR):

```bash
# Set your ACR name
ACR_NAME="youracrname"
az acr login --name $ACR_NAME

# Build the image using the Dockerfile in the project root
docker build -t ${ACR_NAME}.azurecr.io/echo-prime:latest ../../

# Push the image
docker push ${ACR_NAME}.azurecr.io/echo-prime:latest
```

### 3. Update the Image Reference

Edit `kubernetes/azure/deployment.yaml` and replace `image: echo-prime:latest` with your actual ACR image URL: `${ACR_NAME}.azurecr.io/echo-prime:latest`.

### 4. Configure Secrets

Edit `kubernetes/azure/config.yaml` to include your actual secrets (like `ECH0_LICENSE_SECRET`, `OPENAI_API_KEY`, etc.). *Do not commit the real secrets to version control!*

### 5. Apply the Manifests

Apply the configurations, Redis, and ECH0-PRIME deployment:

```bash
kubectl apply -f kubernetes/azure/config.yaml
kubectl apply -f kubernetes/azure/redis.yaml
kubectl apply -f kubernetes/azure/deployment.yaml
```

### 6. Verify the Deployment

Check that the pods are running:

```bash
kubectl get pods -n bbb-production
```

Get the external IP address of the LoadBalancer:

```bash
kubectl get svc echo-prime-service -n bbb-production
```

Once the `EXTERNAL-IP` is assigned, you can access the dashboard at `http://<EXTERNAL-IP>` and Gradio at `http://<EXTERNAL-IP>:7860`.

## Connecting to the BBB Engine

ECH0-PRIME contains the BBB Autonomous Engine (`bbb_autonomous/bbb_core_engine.py`). It will run based on the entrypoint and environment variables specified in `config.yaml`.
