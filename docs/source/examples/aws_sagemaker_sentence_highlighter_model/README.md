# Sentence Highlighting Model Deployment

This directory contains scripts for deploying the OpenSearch Sentence Highlighting model to AWS SageMaker with advanced auto-scaling capabilities. The deployment process automatically downloads the model from OpenSearch artifacts, packages it, and deploys it as a SageMaker endpoint with configurable resource-based scaling.

## Prerequisites

1. AWS Account with appropriate permissions:
   - **IAM Permissions**:
     - `iam:CreateRole`
     - `iam:GetRole`
     - `iam:AttachRolePolicy`
   - **SageMaker Permissions**:
     - `sagemaker:CreateModel`
     - `sagemaker:CreateEndpoint`
     - `sagemaker:CreateEndpointConfig`
   - **Auto-scaling Permissions**:
     - `application-autoscaling:RegisterScalableTarget`
     - `application-autoscaling:PutScalingPolicy`
     - `application-autoscaling:DescribeScalingPolicies`
   - **S3 Permissions**:
     - `s3:PutObject`
     - `s3:GetObject`
     - `s3:CreateBucket` (for default bucket creation)

2. AWS credentials configured locally:
   ```bash
   # Configure using AWS CLI
   aws configure
   ```

3. Python 3.10 or higher
4. Required Python packages (installed via requirements.txt)

## Files

**Project Path**: `docs/source/examples/aws_sagemaker_sentence_highlighter_model/`

- `deploy.py`: Main deployment script
- `inference.py`: Model inference code for SageMaker
- `requirements.txt`: Python package dependencies

## Quick Start

1. Navigate to the deployment directory:
   ```bash
   cd docs/source/examples/aws_sagemaker_sentence_highlighter_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the deployment script:
   ```bash
   # Basic deployment with default resource-based scaling
   python deploy.py
   
   # Advanced deployment with custom scaling parameters
   python deploy.py --resource-scaling-only --target-cpu-utilization 65 --max-instances 8
   ```

The script will:
1. Download the model from OpenSearch artifacts
2. Package the model with necessary dependencies
3. Create a SageMaker role if it doesn't exist
4. Upload the model to the default SageMaker S3 bucket
5. Deploy the model to a SageMaker endpoint
6. Configure auto-scaling policies based on CPU/GPU utilization
7. Test the deployed endpoint

## Auto-Scaling Configuration

The deployment script supports advanced auto-scaling based on resource utilization, perfect for handling variable workloads and batch inference scenarios.

### Scaling Options

```bash
# View all available options
python deploy.py --help
```

**Key Parameters:**
- `--instance-type`: SageMaker instance type (default: ml.g5.xlarge)
- `--initial-instances`: Starting number of instances (default: 1)
- `--min-instances`: Minimum instances for auto-scaling (default: 1)
- `--max-instances`: Maximum instances for auto-scaling (default: 10)
- `--target-cpu-utilization`: CPU % target for scaling (default: 70)
- `--target-gpu-utilization`: GPU % target for scaling (default: 60)
- `--resource-scaling-only`: Use only CPU/GPU scaling, disable API request-based scaling
- `--no-auto-scaling`: Disable auto-scaling entirely

### Scaling Examples

**Resource-Only Scaling (Recommended for Batch Inference):**
```bash
python deploy.py \
  --resource-scaling-only \
  --instance-type ml.g5.xlarge \
  --initial-instances 2 \
  --max-instances 6 \
  --target-cpu-utilization 65 \
  --target-gpu-utilization 55
```

**High-Performance Setup:**
```bash
python deploy.py \
  --instance-type ml.g5.2xlarge \
  --initial-instances 3 \
  --max-instances 20 \
  --target-cpu-utilization 50 \
  --target-gpu-utilization 40
```

**Cost-Optimized Setup:**
```bash
python deploy.py \
  --instance-type ml.g5.xlarge \
  --initial-instances 1 \
  --max-instances 4 \
  --target-cpu-utilization 80 \
  --target-gpu-utilization 70
```

**Single Instance (No Auto-scaling):**
```bash
python deploy.py \
  --instance-type ml.g5.xlarge \
  --initial-instances 1 \
  --no-auto-scaling
```

### Scaling Behavior

**Resource-Based Scaling:**
- **Primary**: CPU utilization monitoring with 4-8 minute response times
- **Secondary**: GPU utilization monitoring (for GPU instances)
- **Response Times**: Fast scale-out (4 mins), conservative scale-in (8 mins)

**Mixed Scaling (Default):**
- Resource-based scaling takes priority
- API request-based scaling as backup with slower response times
- Best for mixed workloads

**Resource-Only Mode (`--resource-scaling-only`):**
- Completely ignores API request counts
- Perfect for batch inference where 1 API call = many model inferences
- Scales purely based on actual CPU/GPU resource consumption

## API Usage

Once deployed, the model endpoint accepts POST requests with the following format:

### Input Format

```json
{
    "question": "What is the main topic discussed?",
    "context": "This is a long text document containing multiple sentences. The model will identify which sentences are most relevant to answering the question. It processes the entire context and returns the sentences that best help answer the provided question."
}
```

**Required fields:**
- `question` (string): The question you want to find relevant sentences for
- `context` (string): The text document containing multiple sentences to search through

### Output Format

```json
{
    "highlights": [
        {
            "start": 45,
            "end": 123,
            "text": "This sentence is relevant to the question.",
            "position": 2
        },
        {
            "start": 200,
            "end": 285,
            "text": "Another relevant sentence that helps answer the question.",
            "position": 5
        }
    ]
}
```

**Response fields:**
- `highlights` (array): List of highlighted sentences
  - `start` (integer): Character position where the sentence starts in the original context
  - `end` (integer): Character position where the sentence ends in the original context
  - `text` (string): The actual text of the highlighted sentence
  - `position` (integer): The sentence number in the original context (0-indexed)

### Example Usage

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Prepare the input
payload = {
    "question": "What are the benefits of machine learning?",
    "context": "Machine learning is a powerful technology. It can help automate many tasks. The benefits include improved efficiency and accuracy. However, it requires good data quality. Machine learning models can make predictions on new data."
}

# Make the prediction
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse the response
result = json.loads(response['Body'].read().decode())
print(result)
```

## Environment Variables

You can customize the deployment using these environment variables:

### Optional Variables:

**Instance Configuration:**
- `INSTANCE_TYPE`: SageMaker instance type (default: "ml.g5.xlarge")
  - **CPU instances**: "ml.m5.xlarge", "ml.m5.2xlarge", "ml.c5.xlarge"
  - **GPU instances**: "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"
  - Example: `export INSTANCE_TYPE="ml.g5.xlarge"`

**Scaling Configuration:**
- `INITIAL_INSTANCE_COUNT`: Starting number of instances (default: 1)
- `MIN_INSTANCE_COUNT`: Minimum instances for auto-scaling (default: 1)  
- `MAX_INSTANCE_COUNT`: Maximum instances for auto-scaling (default: 10)
- `TARGET_CPU_UTILIZATION`: CPU target percentage (default: 70.0)
- `TARGET_GPU_UTILIZATION`: GPU target percentage (default: 60.0)
- `TARGET_INVOCATIONS_PER_INSTANCE`: API requests target (default: 500)
- `AUTO_SCALING_ENABLED`: Enable auto-scaling (default: true)
- `RESOURCE_SCALING_ONLY`: Use only resource-based scaling (default: false)

**AWS Configuration:**
- `AWS_PROFILE`: AWS credentials profile (default: "default")
  - Example: `export AWS_PROFILE="my-profile"`
- `AWS_REGION`: AWS region (default: from AWS configuration)
  - Example: `export AWS_REGION="us-west-2"`

### Environment Variable Example:
```bash
# Configure for batch inference workload
export INSTANCE_TYPE="ml.g5.xlarge"
export INITIAL_INSTANCE_COUNT="2"
export MAX_INSTANCE_COUNT="6"
export TARGET_CPU_UTILIZATION="65.0"
export TARGET_GPU_UTILIZATION="55.0"
export RESOURCE_SCALING_ONLY="true"

# Deploy with environment settings
python deploy.py
```

## GPU Deployment

To deploy with GPU acceleration for faster inference:

### Deploy with GPU:
```bash
# Set GPU instance type
export INSTANCE_TYPE="ml.g5.xlarge"

# Deploy
python deploy.py
```

### GPU Verification:
The inference logs will show GPU usage:
```
Environment Information:
CUDA Available: True
CUDA Device Name: NVIDIA T4
Using GPU for inference
Model device detected: cuda:0 - Moving tensors to this device for inference
```

Example configuration:
```bash
# Optional: Set custom instance type and region
export AWS_REGION="us-west-2"
export INSTANCE_TYPE="ml.g5.xlarge"

# Then run the deployment script
python deploy.py
```

Note: The script uses the default SageMaker bucket for your account (`sagemaker-{region}-{account_id}`), which is automatically created if it doesn't exist.

## Monitoring

### Deployment Monitoring

The script provides detailed logging of the deployment process. Check the logs for:
- Download progress
- Packaging status
- S3 upload status
- Deployment progress
- Endpoint creation status
- Auto-scaling policy configuration
- Test results

### Auto-Scaling Monitoring

**AWS Console Monitoring:**
1. Navigate to **SageMaker > Endpoints** in AWS Console
2. Select your endpoint to view metrics:
   - CPU Utilization
   - GPU Utilization (for GPU instances)
   - Invocations per instance
   - Current instance count

**CloudWatch Metrics:**
```bash
# View endpoint metrics using AWS CLI
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name CPUUtilization \
  --dimensions Name=EndpointName,Value=your-endpoint-name \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 300 \
  --statistics Average
```

**Auto-Scaling Activity:**
```bash
# Check scaling activities
aws application-autoscaling describe-scaling-activities \
  --service-namespace sagemaker \
  --resource-id endpoint/your-endpoint-name/variant/AllTraffic
```

### Performance Optimization

**For Batch Inference Workloads:**
- Use `--resource-scaling-only` to avoid API request-based scaling
- Set conservative CPU/GPU targets (65-75%) to handle processing spikes
- Start with 2+ instances to distribute load
- Monitor CloudWatch for optimal target values

**For Real-time Inference:**
- Use mixed scaling (default) for balanced performance
- Set aggressive targets (50-60%) for fast response
- Higher instance counts for high availability

## Cleanup

To avoid unnecessary charges, delete the endpoint and auto-scaling policies when not needed:

### Quick Cleanup (SageMaker Console)
1. Go to **SageMaker > Endpoints** in AWS Console
2. Select your endpoint and click **Delete**
3. Auto-scaling policies are automatically removed with the endpoint

### Programmatic Cleanup
```python
import boto3

# Replace with your actual endpoint name
endpoint_name = "semantic-highlighter-20240101123456"

# Delete the SageMaker endpoint
sagemaker = boto3.client('sagemaker')
try:
    sagemaker.delete_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} deletion initiated")
except Exception as e:
    print(f"Error deleting endpoint: {e}")

# Auto-scaling policies are automatically cleaned up with the endpoint
```

### CLI Cleanup
```bash
# Delete endpoint using AWS CLI
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name

# Verify deletion
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name
```

**Note**: Deleting the endpoint automatically removes associated auto-scaling policies and targets.

## Troubleshooting

Common issues and solutions:

1. **AWS Credentials**: Ensure AWS credentials are properly configured
2. **Memory Issues**: If packaging fails, ensure sufficient disk space
3. **Network Issues**: Check network connection if download fails
4. **Permission Issues**: Verify AWS IAM permissions

## Support

For issues and questions:
- Check OpenSearch documentation
- Submit issues to the OpenSearch GitHub repository
- Contact the OpenSearch community 