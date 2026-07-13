#!/bin/bash
# Manage Production ECS Fargate Service

set -e

CLUSTER="default"
SERVICE="fridge-detector-8bfa"
REGION="eu-west-3"

ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Usage: ./manage_prod.sh [start|stop|status]"
    exit 1
fi

if [ "$ACTION" == "stop" ]; then
    echo "INFO: Stopping Fargate task (setting desired count to 0)..."
    aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 0 --region "$REGION" >/dev/null
    echo "SUCCESS: Service stopped. Fargate task is terminating. Billing stopped."

elif [ "$ACTION" == "start" ]; then
    echo "INFO: Starting Fargate task (setting desired count to 1)..."
    aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 1 --region "$REGION" >/dev/null
    echo "SUCCESS: Service starting up. Please wait ~1-2 minutes, then run './manage_prod.sh status' to get the public IP."

elif [ "$ACTION" == "status" ]; then
    echo "INFO: Fetching service status..."
    TASK_ARN=$(aws ecs list-tasks --cluster "$CLUSTER" --service-name "$SERVICE" --region "$REGION" --query "taskArns[0]" --output text 2>/dev/null)

    if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" == "None" ]; then
        echo "STATUS: Service is stopped (no tasks running)."
        exit 0
    fi

    echo "INFO: Task found: $TASK_ARN"
    TASK_STATUS=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$REGION" --query "tasks[0].lastStatus" --output text)
    echo "STATUS: Task is currently in state: $TASK_STATUS"

    ENI_ID=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$REGION" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)

    if [ -z "$ENI_ID" ] || [ "$ENI_ID" == "None" ]; then
        echo "INFO: Network interface not attached yet."
        exit 0
    fi

    PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" --region "$REGION" --query "NetworkInterfaces[0].Association.PublicIp" --output text 2>/dev/null)

    if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" == "None" ]; then
        echo "INFO: Public IP address not allocated yet."
    else
        echo "SUCCESS: Backend is running!"
        echo "Public IP: $PUBLIC_IP"
        echo ""
        echo "Start Expo on client using:"
        echo "  EXPO_PUBLIC_API_URL=http://$PUBLIC_IP:8000 EXPO_PUBLIC_API_KEY=lemmafrappeataporte npm run start"
    fi
fi
