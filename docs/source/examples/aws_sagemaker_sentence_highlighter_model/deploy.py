import os
import time
import logging
import boto3
import sagemaker
import shutil
import tarfile
import requests
import zipfile
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScalingConfig:
    """Configuration for SageMaker endpoint scaling"""
    instance_type: str = 'ml.g5.xlarge'
    initial_instance_count: int = 1
    min_instance_count: int = 1
    max_instance_count: int = 10
    target_invocations_per_instance: int = 10000
    target_cpu_utilization: float = 70.0
    target_gpu_utilization: float = 60.0
    auto_scaling_enabled: bool = True
    resource_scaling_only: bool = False
    
    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        return cls(
            instance_type=os.getenv('INSTANCE_TYPE', 'ml.g5.xlarge'),
            initial_instance_count=int(os.getenv('INITIAL_INSTANCE_COUNT', '1')),
            min_instance_count=int(os.getenv('MIN_INSTANCE_COUNT', '1')),
            max_instance_count=int(os.getenv('MAX_INSTANCE_COUNT', '10')),
            target_invocations_per_instance=int(os.getenv('TARGET_INVOCATIONS_PER_INSTANCE', '500')),
            target_cpu_utilization=float(os.getenv('TARGET_CPU_UTILIZATION', '70.0')),
            target_gpu_utilization=float(os.getenv('TARGET_GPU_UTILIZATION', '60.0')),
            auto_scaling_enabled=os.getenv('AUTO_SCALING_ENABLED', 'true').lower() == 'true',
            resource_scaling_only=os.getenv('RESOURCE_SCALING_ONLY', 'false').lower() == 'true'
        )
    
    def validate(self):
        """Validate the scaling configuration"""
        if self.min_instance_count > self.initial_instance_count:
            raise ValueError(f"Min instances ({self.min_instance_count}) cannot be greater than initial instances ({self.initial_instance_count})")
        
        if self.initial_instance_count > self.max_instance_count:
            raise ValueError(f"Initial instances ({self.initial_instance_count}) cannot be greater than max instances ({self.max_instance_count})")
        
        if self.min_instance_count < 1:
            raise ValueError(f"Min instances must be at least 1, got {self.min_instance_count}")
        
        if self.max_instance_count < 1:
            raise ValueError(f"Max instances must be at least 1, got {self.max_instance_count}")
        
        if self.target_invocations_per_instance < 1:
            raise ValueError(f"Target invocations per instance must be at least 1, got {self.target_invocations_per_instance}")
        
        if not (0 < self.target_cpu_utilization <= 100):
            raise ValueError(f"Target CPU utilization must be between 0 and 100, got {self.target_cpu_utilization}")
            
        if not (0 < self.target_gpu_utilization <= 100):
            raise ValueError(f"Target GPU utilization must be between 0 and 100, got {self.target_gpu_utilization}")
        
        # Validate instance type format
        if not self.instance_type.startswith('ml.'):
            raise ValueError(f"Instance type must start with 'ml.', got {self.instance_type}")
        
        logger.info("Scaling configuration validated successfully")
    
    def print_config(self):
        """Print the current scaling configuration"""
        logger.info("=" * 60)
        logger.info("SCALING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Instance Type: {self.instance_type}")
        logger.info(f"Initial Instance Count: {self.initial_instance_count}")
        logger.info(f"Auto-scaling Enabled: {self.auto_scaling_enabled}")
        logger.info(f"Resource Scaling Only: {self.resource_scaling_only}")
        
        if self.auto_scaling_enabled:
            logger.info(f"Min Instance Count: {self.min_instance_count}")
            logger.info(f"Max Instance Count: {self.max_instance_count}")
            logger.info(f"Target CPU Utilization: {self.target_cpu_utilization}%")
            logger.info(f"Target GPU Utilization: {self.target_gpu_utilization}%")
            
            if self.resource_scaling_only:
                logger.info("Target Invocations: DISABLED (Resource scaling only)")
            else:
                logger.info(f"Target Invocations Per Instance: {self.target_invocations_per_instance}")
        
        logger.info("=" * 60)

def parse_arguments():
    """Parse command line arguments"""
    # Get default config from environment
    default_config = ScalingConfig.from_environment()
    
    parser = argparse.ArgumentParser(
        description="Deploy SageMaker Sentence Highlighter with configurable scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with default settings (resource-based scaling enabled)
  python deploy.py
  
  # Deploy with resource-only scaling (NO invocation-based scaling)
  python deploy.py --resource-scaling-only --target-cpu-utilization 60 --target-gpu-utilization 50
  
  # Deploy with high performance setup - aggressive resource scaling
  python deploy.py --instance-type ml.g5.2xlarge --initial-instances 3 --max-instances 20 --target-cpu-utilization 50 --target-gpu-utilization 40
  
  # Deploy without auto-scaling (single instance)
  python deploy.py --instance-type ml.g5.xlarge --initial-instances 1 --no-auto-scaling
  
  # Deploy for batch inference workloads (your use case)
  python deploy.py --resource-scaling-only --instance-type ml.g5.xlarge --initial-instances 2 --max-instances 6 --target-cpu-utilization 65 --target-gpu-utilization 55
        """
    )
    
    parser.add_argument(
        '--instance-type',
        default=default_config.instance_type,
        help=f'SageMaker instance type (default: {default_config.instance_type})'
    )
    
    parser.add_argument(
        '--initial-instances',
        type=int,
        default=default_config.initial_instance_count,
        help=f'Initial number of instances (default: {default_config.initial_instance_count})'
    )
    
    parser.add_argument(
        '--min-instances',
        type=int,
        default=default_config.min_instance_count,
        help=f'Minimum number of instances for auto-scaling (default: {default_config.min_instance_count})'
    )
    
    parser.add_argument(
        '--max-instances',
        type=int,
        default=default_config.max_instance_count,
        help=f'Maximum number of instances for auto-scaling (default: {default_config.max_instance_count})'
    )
    
    parser.add_argument(
        '--target-invocations',
        type=int,
        default=default_config.target_invocations_per_instance,
        help=f'Target invocations per instance for scaling (default: {default_config.target_invocations_per_instance})'
    )
    
    parser.add_argument(
        '--target-cpu-utilization',
        type=float,
        default=default_config.target_cpu_utilization,
        help=f'Target CPU utilization percentage for scaling (default: {default_config.target_cpu_utilization})'
    )
    
    parser.add_argument(
        '--target-gpu-utilization',
        type=float,
        default=default_config.target_gpu_utilization,
        help=f'Target GPU utilization percentage for scaling (default: {default_config.target_gpu_utilization})'
    )
    
    parser.add_argument(
        '--resource-scaling-only',
        action='store_true',
        help='Use only CPU/GPU resource scaling, disable invocation-based scaling'
    )
    
    parser.add_argument(
        '--no-auto-scaling',
        action='store_true',
        help='Disable auto-scaling'
    )
    
    return parser.parse_args()

def setup_scaling_config(args):
    """Setup scaling configuration based on arguments"""
    # Start with defaults from environment
    default_config = ScalingConfig.from_environment()
    
    # Create configuration with arguments, using defaults as fallback
    config = ScalingConfig(
        instance_type=args.instance_type if args.instance_type else default_config.instance_type,
        initial_instance_count=args.initial_instances,
        min_instance_count=args.min_instances,
        max_instance_count=args.max_instances,
        target_invocations_per_instance=args.target_invocations,
        target_cpu_utilization=args.target_cpu_utilization,
        target_gpu_utilization=args.target_gpu_utilization,
        auto_scaling_enabled=not args.no_auto_scaling if args.no_auto_scaling else default_config.auto_scaling_enabled,
        resource_scaling_only=args.resource_scaling_only
    )
    
    # Validate the configuration
    config.validate()
    
    logger.info("Scaling configuration setup completed")
    return config

def get_endpoint_name():
    """Generate a unique endpoint name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"semantic-highlighter-{timestamp}"

def download_and_extract_model(url, extract_dir):
    """Download zip file and extract model"""
    try:
        # Create a temporary directory for the zip file
        temp_dir = "temp_download"
        os.makedirs(temp_dir, exist_ok=True)
        zip_path = os.path.join(temp_dir, "model.zip")
        
        # Download zip file
        logger.info(f"Downloading model zip from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model zip downloaded successfully")
        
        # Extract zip file
        logger.info(f"Extracting zip file to {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the .pt file in the extracted contents
        pt_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))
        
        if not pt_files:
            raise FileNotFoundError("No .pt file found in the extracted contents")
        
        if len(pt_files) > 1:
            raise ValueError(f"Expected exactly one .pt file, but found {len(pt_files)}: {pt_files}")
        
        model_path = pt_files[0]
        logger.info(f"Found model file at: {model_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error downloading and extracting model: {str(e)}")
        raise

def prepare_model_package():
    """Prepare model package for deployment"""
    try:
        logger.info("Preparing model package...")
        model_dir = "model"

        # Clean up existing model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # model from OpenSearch pre-trained model hub
        model_url = "https://artifacts.opensearch.org/models/ml-models/amazon/sentence-highlighting/opensearch-semantic-highlighter-v1/1.0.0/torch_script/sentence-highlighting_opensearch-semantic-highlighter-v1-1.0.0-torch_script.zip"

        # Download and extract model
        model_path = download_and_extract_model(model_url, model_dir)
        model_filename = os.path.basename(model_path)
        
        # Create code directory for inference script
        code_dir = os.path.join(model_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        
        # Copy inference script to code directory
        if not os.path.exists("inference.py"):
            raise FileNotFoundError("inference.py not found")
        shutil.copy("inference.py", code_dir)
        
        # Copy requirements.txt to code directory
        if not os.path.exists("requirements.txt"):
            raise FileNotFoundError("requirements.txt not found")
        shutil.copy("requirements.txt", code_dir)
        
        # Create model.tar.gz
        if os.path.exists("model.tar.gz"):
            os.remove("model.tar.gz")
            
        logger.info("Creating model.tar.gz...")
        with tarfile.open("model.tar.gz", "w:gz") as tar:
            tar.add(model_path, arcname=model_filename)
            tar.add(code_dir, arcname="code")
        
        logger.info("Model package prepared successfully")
        
    except Exception as e:
        logger.error(f"Error preparing model package: {str(e)}")
        raise

def create_sagemaker_role():
    """Create a new SageMaker execution role with necessary permissions."""
    try:
        logger.info("Creating new SageMaker role...")
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole'
        
        # Create the role
        try:
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument='''{
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "sagemaker.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }'''
            )
            logger.info(f"Created new role: {role_name}")
        except iam.exceptions.EntityAlreadyExistsException:
            logger.info(f"Role {role_name} already exists")
            return f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/{role_name}'
        
        # Attach necessary policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        ]
        
        for policy in policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
            logger.info(f"Attached policy {policy} to role {role_name}")
        
        # Get the role ARN
        role_arn = f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/{role_name}'
        logger.info(f"Created role with ARN: {role_arn}")
        return role_arn
        
    except Exception as e:
        logger.error(f"Failed to create SageMaker role: {str(e)}")
        raise

def test_endpoint(endpoint_name):
    """Test the deployed endpoint with sample data"""
    try:
        logger.info("Testing deployed endpoint...")
        
        # Create SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime')
        
        # Test data
        test_data = {
            "question": "What are the symptoms of heart failure?",
            "context": "Hypertensive heart disease is the No. 1 cause of death associated with high blood pressure. It refers to a group of disorders that includes heart failure, ischemic heart disease, and left ventricular hypertrophy (excessive thickening of the heart muscle). Heart failure does not mean the heart has stopped working. Rather, it means that the heart's pumping power is weaker than normal or the heart has become less elastic. With heart failure, blood moves through the heart's pumping chambers less effectively, and pressure in the heart increases, making it harder for your heart to deliver oxygen and nutrients to your body. To compensate for reduced pumping power, the heart's chambers respond by stretching to hold more blood. This keeps the blood moving, but over time, the heart muscle walls may weaken and become unable to pump as strongly. As a result, the kidneys often respond by causing the body to retain fluid (water) and sodium. The resulting fluid buildup in the arms, legs, ankles, feet, lungs, or other organs, and is called congestive heart failure. High blood pressure may also bring on heart failure by causing left ventricular hypertrophy, a thickening of the heart muscle that results in less effective muscle relaxation between heart beats. This makes it difficult for the heart to fill with enough blood to supply the body's organs, especially during exercise, leading your body to hold onto fluids and your heart rate to increase. Symptoms of heart failure include: Shortness of breath Swelling in the feet, ankles, or abdomen Difficulty sleeping flat in bed Bloating Irregular pulse Nausea Fatigue Greater need to urinate at night High blood pressure can also cause ischemic heart disease. This means that the heart muscle isn't getting enough blood. Ischemic heart disease is usually the result of atherosclerosis or hardening of the arteries (coronary artery disease), which impedes blood flow to the heart. Symptoms of ischemic heart disease may include: Chest pain which may radiate (travel) to the arms, back, neck, or jaw Chest pain with nausea, sweating, shortness of breath, and dizziness; these associated symptoms may also occur without chest pain Irregular pulse Fatigue and weakness Any of these symptoms of ischemic heart disease warrant immediate medical evaluation. Your doctor will look for certain signs of hypertensive heart disease, including: High blood pressure Enlarged heart and irregular heartbeat Fluid in the lungs or lower extremities Unusual heart sounds Your doctor may perform tests to determine if you have hypertensive heart disease, including an electrocardiogram, echocardiogram, cardiac stress test, chest X-ray, and coronary angiogram. In order to treat hypertensive heart disease, your doctor has to treat the high blood pressure that is causing it. He or she will treat it with a variety of drugs, including diuretics, beta-blockers, ACE inhibitors, calcium channel blockers, angiotensin receptor blockers, and vasodilators. In addition, your doctor may advise you to make changes to your lifestyle, including: Diet: If heart failure is present, you should lower your daily intake of sodium to 1,500 mg or 2 g or less per day, eat foods high in fiber and potassium, limit total daily calories to lose weight if necessary, and limit intake of foods that contain refined sugar, trans fats, and cholesterol. Monitoring your weight: This involves daily recording of weight, increasing your activity level (as recommended by your doctor), resting between activities more often, and planning your activities. Avoiding tobacco products and alcohol Regular medical checkups: During follow-up visits, your doctor will make sure you are staying healthy and that your heart disease is not getting worse."
        }
        
        logger.info("Sending test request to endpoint...")
        logger.info(f"Test question: {test_data['question']}")
        
        # Make prediction
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        logger.info("Test successful!")
        logger.info("Test results:")
        logger.info(json.dumps(result, indent=2))
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error("This indicates there may be an issue with the deployment")
        return False

def setup_auto_scaling(endpoint_name, config: ScalingConfig, variant_name='AllTraffic'):
    """Configure auto-scaling for the endpoint"""
    try:
        logger.info("Setting up auto-scaling for the endpoint...")
        
        # Create auto-scaling client
        autoscaling_client = boto3.client('application-autoscaling')
        
        # Register scalable target
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        
        logger.info(f"Registering scalable target: {resource_id}")
        logger.info(f"Min capacity: {config.min_instance_count}, Max capacity: {config.max_instance_count}")
        
        autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=config.min_instance_count,
            MaxCapacity=config.max_instance_count
        )
        
        # Create CPU utilization scaling policy (PRIMARY)
        cpu_policy_name = f"{endpoint_name}-cpu-scaling-policy"
        logger.info(f"Creating CPU utilization scaling policy: {cpu_policy_name}")
        logger.info(f"Target CPU utilization: {config.target_cpu_utilization}%")
        
        autoscaling_client.put_scaling_policy(
            PolicyName=cpu_policy_name,
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': config.target_cpu_utilization,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantCPUUtilization'
                },
                'ScaleOutCooldown': 240,  # Faster scale-out for resource pressure
                'ScaleInCooldown': 480    # Conservative scale-in
            }
        )
        
        logger.info("CPU utilization scaling policy configured")
        
        # Create GPU utilization scaling policy (SECONDARY) if instance supports GPU
        if 'g4' in config.instance_type or 'g5' in config.instance_type or 'p3' in config.instance_type or 'p4' in config.instance_type:
            gpu_policy_name = f"{endpoint_name}-gpu-scaling-policy"
            logger.info(f"Creating GPU utilization scaling policy: {gpu_policy_name}")
            logger.info(f"Target GPU utilization: {config.target_gpu_utilization}%")
            
            autoscaling_client.put_scaling_policy(
                PolicyName=gpu_policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': config.target_gpu_utilization,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantGPUUtilization'
                    },
                    'ScaleOutCooldown': 300,
                    'ScaleInCooldown': 600
                }
            )
            
            logger.info("GPU utilization scaling policy configured")
        else:
            logger.info("Instance type doesn't support GPU - skipping GPU scaling policy")
        
        # Only add invocation-based scaling if NOT in resource-only mode
        if not config.resource_scaling_only:
            policy_name = f"{endpoint_name}-invocation-scaling-policy"
            logger.info(f"Creating invocation-based scaling policy: {policy_name}")
            logger.info(f"Target invocations per instance: {config.target_invocations_per_instance}")
            
            policy_response = autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': float(config.target_invocations_per_instance),
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleOutCooldown': 600,  # Slower than resource-based scaling
                    'ScaleInCooldown': 900    # Much slower scale-in for invocations
                }
            )
            
            logger.info("Invocation-based scaling policy configured")
            logger.info(f"Policy ARN: {policy_response['PolicyARN']}")
        else:
            logger.info("Resource scaling only mode - skipping invocation-based scaling")
        
        logger.info("Auto-scaling configured successfully")
        logger.info("Resource-based scaling will take priority over invocation-based scaling")
        
    except Exception as e:
        logger.error(f"Failed to setup auto-scaling: {str(e)}")
        logger.warning("Continuing without auto-scaling...")

def deploy_model(config: ScalingConfig):
    try:
        # Print scaling configuration
        config.print_config()
        
        # Prepare model package first
        prepare_model_package()
        
        # Generate unique endpoint name
        endpoint_name = get_endpoint_name()
        logger.info(f"Using endpoint name: {endpoint_name}")
        
        # Initialize SageMaker session
        logger.info("Initializing SageMaker session...")
        session = sagemaker.Session()
        
        # Create and get execution role
        role = create_sagemaker_role()
        logger.info(f"Using execution role: {role}")
        
        # Get default bucket (creates it if it doesn't exist)
        bucket = session.default_bucket()
        logger.info(f"Using default SageMaker bucket: {bucket}")

        # Upload model to S3
        s3_prefix = 'semantic-highlighter'
        logger.info(f"Uploading model to S3: {bucket}/{s3_prefix}")
        
        try:
            s3_path = session.upload_data('model.tar.gz', bucket=bucket, key_prefix=s3_prefix)
            logger.info(f"Model uploaded successfully to {s3_path}")
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {str(e)}")
            raise
        
        # Create SageMaker model
        logger.info("Creating SageMaker model...")
        model = PyTorchModel(
            model_data=s3_path,
            role=role,
            framework_version='2.5',
            py_version='py311',
            entry_point='inference.py',
            source_dir='.',
            sagemaker_session=session
        )
        logger.info("SageMaker model created successfully")

        # Deploy endpoint
        logger.info("Starting endpoint deployment...")
        logger.info("This may take several minutes...")
        start_time = time.time()
        
        try:
            predictor = model.deploy(
                initial_instance_count=config.initial_instance_count,
                instance_type=config.instance_type,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            end_time = time.time()
            logger.info(f"Endpoint deployed successfully in {end_time - start_time:.2f} seconds")
            logger.info(f"Endpoint name: {endpoint_name}")
            
            # Test the deployed endpoint
            logger.info("=" * 60)
            logger.info("TESTING DEPLOYED ENDPOINT")
            logger.info("=" * 60)
            
            test_success = test_endpoint(endpoint_name)
            
            # Setup auto-scaling if enabled
            if config.auto_scaling_enabled:
                setup_auto_scaling(endpoint_name, config)
            else:
                logger.info("Auto-scaling is disabled")
            
            if test_success:
                logger.info("Deployment and testing completed successfully!")
                logger.info(f"Your endpoint '{endpoint_name}' is ready to use.")
                
                if config.auto_scaling_enabled:
                    logger.info("Auto-scaling has been configured to handle concurrent requests")
                    logger.info(f"The endpoint will scale between {config.min_instance_count} and {config.max_instance_count} instances")
                    logger.info(f"based on invocations per instance (target: {config.target_invocations_per_instance}) and CPU utilization")
            else:
                logger.warning("Deployment completed but testing failed. Check CloudWatch logs for details.")

            return predictor
        except Exception as e:
            logger.error(f"Deployment failed with error: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def main():
    """Main function to handle command line arguments and deploy the model"""
    args = parse_arguments()
    
    # Setup scaling configuration
    config = setup_scaling_config(args)
    
    # Deploy the model
    deploy_model(config)

if __name__ == "__main__":
    main()
