#!/bin/bash
# Deploy Fridge Detector Backend to Amazon ECR & ECS

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "INFO: Starting Production Deployment Script"

# 1. Detect AWS Environment Settings
echo "INFO: Detecting AWS configuration..."
AWS_REGION=$(aws configure get region 2>/dev/null || echo "eu-west-3")
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")

# Fallback values
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "WARNING: Could not automatically detect AWS Account ID. Please ensure your AWS CLI is configured."
    read -p "Enter your AWS Account ID: " AWS_ACCOUNT_ID
fi

echo "Configuration:"
echo "AWS Account ID : $AWS_ACCOUNT_ID"
echo "AWS Region     : $AWS_REGION"
echo "ECR Repo Name  : fridge-detector"
read -p "Are these settings correct? (y/n): " confirm
if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    read -p "Enter AWS Region: " AWS_REGION
    read -p "Enter AWS Account ID: " AWS_ACCOUNT_ID
fi

ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
ECR_REPO_URL="$ECR_REGISTRY/fridge-detector"

# 2. Build the Docker Image
echo "Step 1: Building Docker Image"
echo "INFO: Building image 'fridge-detector' (ignoring large weight files to keep size small)..."
docker build --platform linux/amd64 -t fridge-detector .

# 3. Authenticate with ECR
echo "Step 2: Authenticating with AWS ECR"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# 4. Check if ECR Repo exists, create if missing
echo "Step 3: Verifying ECR Repository"
if ! aws ecr describe-repositories --repository-names fridge-detector --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "INFO: ECR Repository 'fridge-detector' does not exist. Creating it now..."
    aws ecr create-repository --repository-name fridge-detector --region "$AWS_REGION"
else
    echo "INFO: ECR Repository 'fridge-detector' verified."
fi

# 5. Tag and Push Image
echo "Step 4: Tagging and Pushing Image to ECR"
docker tag fridge-detector:latest "$ECR_REPO_URL:latest"
docker push "$ECR_REPO_URL:latest"

# 6. Instruct on Service Redeployment
echo "Step 5: Deployment Complete"
echo "The new image has been pushed to ECR: $ECR_REPO_URL:latest"
echo ""
echo "To deploy the changes to your ECS Service, run the following command:"
echo "  aws ecs update-service --cluster default --service fridge-detector-8bfa --force-new-deployment --region $AWS_REGION"
echo ""
echo "Make sure your ECS service / task definition specifies the environment variables:"
echo "  APP_ENV=production"
echo "  BUCKET_NAME=whatieat-assets"
echo "This guarantees it pulls the weights from S3 on startup."
