# Walkthrough — Vision Backend Improvements & Dockerization

We have successfully implemented the quantity mapping rules, added API Key authentication, written and optimized the Dockerfile, and verified the serving API locally and inside the container.

---

## Architecture Flow Diagram (Excalidraw Reference)

Use this ASCII schema to design your architecture in Excalidraw:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               JURY PRESENTATION FLOW                            │
└─────────────────────────────────────────────────────────────────────────────────┘

                  ┌────────────────────────────────────────┐
                  │           Expo Go App (Phone)          │
                  └────────────────────────────────────────┘
                    /                                    \
                   / (1) Scan QR Code                     \ (3) POST /vision/scan
                  /  to download Metro Bundle              \    with X-API-Key Header
                 v                                          v
      ┌──────────────────────┐                     ┌──────────────────────────────┐
      │   Expo Tunnel / EAS  │                     │   AWS App Runner (Cloud)     │
      │   (Public Internet)  │                     │   (Public HTTPS Endpoint)    │
      └──────────────────────┘                     └──────────────────────────────┘
                 |                                                |
                 | (2) Transfers JS Bundle                        | (4) Routes Request
                 |                                                v
                 |                                 ┌──────────────────────────────┐
                 v                                 │    FastAPI Docker Container  │
       ┌────────────────────┐                      ├──────────────────────────────┤
       │  Developer Laptop  │                      │ 1. API Key Auth Validator    │
       │   (Runs Metro)     │                      │    (X-API-Key == API_KEY)    │
       └────────────────────┘                      │ 2. PyTorch Faster R-CNN      │
                                                   │    (Detects Bounding Boxes)  │
                                                   │ 3. SAM 2.1 ViT-T Predictor   │
                                                   │    (Generates Segment Masks) │
                                                   │ 4. Mask IoU NMS Filter       │
                                                   │    (Removes overlaps)        │
                                                   │ 5. Quantity Mapping Rules    │
                                                   │    (Count/Pack/Level)        │
                                                   └──────────────────────────────┘
                                                                  |
                                                                  │ (5) Returns JSON
                                                                  v
                                                   ┌──────────────────────────────┐
                                                   │  JSON Payload:               │
                                                   │  - Ingredients & Quantities  │
                                                   │  - Segment Contours/Polygons │
                                                   └──────────────────────────────┘
```

---

## Expo Go Tunneling for Jury Presentation (Step-by-Step)

Due to security limitations in modern Expo versions, the stock Expo Go app from the App/Play Store cannot load custom published production bundles over public links. 

To let the jury test the app on their own phones over the internet, follow this setup:

### **Why Tunneling is Necessary**
* Local Metro servers run on your local network IP (e.g., `192.168.x.x`). 
* In a presentation setting (like a university Wi-Fi network), **Client Isolation** is usually enabled, blocking devices from connecting to each other.
* Tunneling creates a secure, public HTTPS tunnel using `ngrok` (e.g. `exp://xxxx.ngrok-free.app`), allowing Expo Go to download your code from your laptop over the internet, regardless of Wi-Fi settings.

### **Presentation Setup Checklist**
1. **Deploy your Backend to AWS App Runner** (see ECR/App Runner steps below).
2. **Update App config**: In your Expo app source code, make sure your api requests point to your public AWS App Runner URL (e.g., `https://xxxxxx.us-east-1.awsapprunner.com`) and include the `X-API-Key` header:
   ```typescript
   headers: {
     'X-API-Key': 'your-secret-key-configured-on-aws',
     'Content-Type': 'multipart/form-data',
   }
   ```
3. **Run Metro with Tunnel**: On your laptop, launch Expo Metro with the tunnel flag:
   ```bash
   npx expo start --tunnel
   ```
4. **Present the QR code**: Let the jury scan the terminal's QR code using the stock Expo Go app on their phones.
5. **Scan and Test**: The jury's phone will download your JavaScript code from the tunnel and communicate directly with your public AWS backend to process fridge scans!

---

## Changes Made

### 1. API Key Authentication (Security)
* **File modified**: [serve_api.py](file:///Users/ayouba/Documents/laboratoire/fridge_detector/scripts/serve_api.py)
* **Logic added**:
  * Added `verify_api_key` dependency checking for `X-API-Key` in request headers.
  * Checks against `API_KEY` environment variable.
  * If `API_KEY` is not set (e.g. local development), it automatically bypasses authentication to simplify testing.
  * Enforced on `POST /vision/scan` endpoint.

### 2. Quantity Mapping & Rules implementation
* **File modified**: [serve_api.py](file:///Users/ayouba/Documents/laboratoire/fridge_detector/scripts/serve_api.py)
* **Logic added**:
  * Categorized all 31 classes into three groups: `COUNTABLE_CLASSES` (tomato, egg, apple, etc.), `PACKAGE_CLASSES` (milk, cheese, butter, etc.), and `BULK_CLASSES` (rice, bulgur, spinach, etc.).
  * For **Countables** and **Packages**: Returns the integer count of detections (e.g. `'7'` for tomatoes) with units `count` or `pack` respectively.
  * For **Bulk items**: Calculates the cumulative area of the detected masks (or boxes as fallback). If the area covers less than $5\%$ of the image, it returns `'low'`. Between $5\%$ and $15\%$ returns `'medium'`, and $\ge 15\%$ returns `'high'`. The unit is set to `level` and method is set to `area_proxy`.
  * Added additional fields in the `ingredients` response list: `unit`, `method`, and `quantity_confidence` (average confidence of the detections for that class).

### 3. Missing Dependency Fix
* **File modified**: [pyproject.toml](file:///Users/ayouba/Documents/laboratoire/fridge_detector/pyproject.toml)
* **Logic added**: Added `sam2>=1.1.0` to the project's dependencies to ensure it is installed correctly during the Docker build process.

### 4. Dockerization & CPU-only Optimization
* **File created**: [Dockerfile](file:///Users/ayouba/Documents/laboratoire/fridge_detector/Dockerfile)
* **Optimization**:
  * Configured `uv pip install --system` using the `--extra-index-url https://download.pytorch.org/whl/cpu` flag. This forces `uv` to pull CPU-only wheels for PyTorch and torchvision, preventing the installation of heavy CUDA libraries and keeping the Docker image size small.
* **Image Size**: Reduced from `4.13 GB` to **`2.45 GB`** by adding a [.dockerignore](file:///Users/ayouba/Documents/laboratoire/fridge_detector/.dockerignore) file that filters out unnecessary checkpoints (`latest.pt`, SAM 1 weights) and temporary files.

---

## Verification & Testing Results

### 1. Local FastAPI Health Check (Dockerized)
The container was started on port 8000. Calling `/health` returned:
```json
{
  "status": "ok",
  "device": "cpu",
  "checkpoint": "/app/checkpoints/best.pt",
  "sam": {
    "enabled": true,
    "status": "SAM 2 enabled (tiny on cpu)"
  }
}
```

### 2. API Security Check
* **Without API Key header (when API_KEY is set in environment)**: Returns `401 Unauthorized` with `{"detail":"Unauthorized: Invalid or missing X-API-Key header"}`.
* **With X-API-Key header**: Successfully processes request and returns status code `200 OK`.

### 3. Scan Output Example (Tomato)
Posting the `5tomato_jpg` image returned the correct quantity of `'7'` with unit `'count'`:
```json
{
  "ingredients": [
    {
      "id": "tomato",
      "name": "tomato",
      "quantity": "7",
      "unit": "count",
      "method": "instance_count",
      "quantity_confidence": 1.0
    }
  ],
  "confidence": 0.9999904632568359,
  "detections": [...]
}
```

---

## AWS App Runner Deployment Instructions

To deploy the optimized Docker container to **AWS App Runner**, follow these steps:

### Step 1: Create an ECR Repository and Push the Image
```bash
# 1. Authenticate Docker with your AWS Account
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com

# 2. Create the repository
aws ecr create-repository --repository-name fridge-detector --region <your-region>

# 3. Tag your local image
docker tag fridge-detector:latest <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/fridge-detector:latest

# 4. Push the image
docker push <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/fridge-detector:latest
```

### Step 2: Create AWS App Runner Service
1. Open the **AWS App Runner** Console.
2. Click **Create Service**.
3. Select **Container registry** -> **Amazon ECR**.
4. Choose your image: `<aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/fridge-detector:latest`.
5. Under **Configure service**:
   * **Port**: `8000` (FastAPI port).
   * **CPU & Memory**: Select **1 vCPU** and **4 GB RAM** (required to ensure fast CPU-only model execution and fit model weights).
   * **Environment Variables**: Add `API_KEY` with your secret key (e.g. `mysecretkey`).
6. Create the service. AWS App Runner will provision a public URL with HTTPS (e.g., `https://xxxxxx.us-east-1.awsapprunner.com`) which your colleague can immediately hook into the Expo App's configuration!
