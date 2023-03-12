
from modules import scripts, script_callbacks, shared
import gradio as gr
# move all private sagemaker call into middlewares 
import sagemaker, boto3
from sagemaker.jumpstart.notebook_utils import list_jumpstart_models
from sagemaker import image_uris, model_uris, script_uris
from sagemaker import hyperparameters
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
from sagemaker.tuner import HyperparameterTuner
from sagemaker import get_execution_role

session = sagemaker.Session()
iam = boto3.client('iam')

class sageMakerScript(scripts.Script):
    def title(self):
        return "AWS SageMaker Training"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass
    def process(self):
        pass

def onModelLoaded(sd_model):
    # placeholder for SageMaker training
    pass

def onScriptUnloaded():
    # placeholder for SageMaker training
    pass

def finetune(sagemaker_role, train_model_id, hyperparameters, training_instance_type, training_dataset_s3_path, s3_output_location):

    # Currently, not all the stable diffusion models in jumpstart support finetuning. Thus, we manually select a model which supports finetuning.
    train_model_id, train_model_version, train_scope = (
        "model-txt2img-stabilityai-stable-diffusion-v2-1-base",
        "*",
        "training",
    )

    # Retrieve the docker image
    train_image_uri = image_uris.retrieve(
        region=None,
        framework=None,  # automatically inferred from model_id
        model_id=train_model_id,
        model_version=train_model_version,
        image_scope=train_scope,
        instance_type=training_instance_type,
    )

    # Retrieve the training script. This contains all the necessary files including data processing, model training etc.
    train_source_uri = script_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, script_scope=train_scope
    )
    # Retrieve the pre-trained model tarball to further fine-tune
    train_model_uri = model_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
    )

    # Retrieve the default hyper-parameters for fine-tuning the model
    hyperparameters = hyperparameters.retrieve_default(
        model_id=train_model_id, model_version=train_model_version
    )

    # [Optional] Override default hyperparameters with custom values
    hyperparameters["max_steps"] = "400"

    # Prepare training job
    training_job_name = name_from_base(f"sd-{train_model_id}-transfer-learning")

    sd_estimator = Estimator(
        role=sagemaker_role,
        image_uri=train_image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",  # Entry-point file in source_dir and present in train_source_uri.
        instance_count=1,
        instance_type=training_instance_type,
        max_run=360000,
        hyperparameters=hyperparameters,
        output_path=s3_output_location,
        base_job_name=training_job_name,
    )

    # Launch a SageMaker Training job by passing s3 path of the training data
    sd_estimator.fit({"training": training_dataset_s3_path}, logs=True)


def deploy(*args, **kwargs):
    pass

def onUiTabs():
    with gr.Blocks(analytics_enabled=False) as aws_sd_inf:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="Train your model with built-in models or create your own embedding")

                newEmbedding = gr.Textbox(label="Name")

                try:
                    sagemaker_role = sagemaker.get_execution_role()
                except ValueError:
                    sagemaker_role = iam.get_role(RoleName='SageMakerRole')['Role']['Arn']

                # choose built-in model
                filter_value = "task == txt2img"
                txt2img_models = list_jumpstart_models(filter=filter_value)

                dropdown_components = []
                train_model_id = gr.Dropdown(
                    label="Select a model",
                    choices=txt2img_models,
                    value=txt2img_models[0],
                    type="value",
                )

                # choose hyperparameters
                hyperparameters = gr.Textbox(label='Hyperparameters')

                # input training data bucket and prefix
                training_data_bucket = gr.Textbox(label='Training data bucket')
                training_data_prefix = gr.Textbox(label='Training data prefix')
                
                training_dataset_s3_path = f"s3://{training_data_bucket}/{training_data_prefix}"

                output_bucket = session.default_bucket()
                output_prefix = "sd-training"

                s3_output_location = f"s3://{output_bucket}/{output_prefix}/output"

                # select training instance
                training_instance_type = gr.Dropdown(
                    label="Select a training instance",
                    choices=["ml.g4dn.2xlarge", "ml.g4dn.4xlarge", "ml.g4dn.8xlarge", "ml.g4dn.12xlarge", "ml.g4dn.16xlarge"],
                    value="ml.g4dn.2xlarge",
                    type="value",
                )

                # submit training job
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(value="")

                    with gr.Column():
                        submitTrain = gr.Button(value="Submit training job", variant='primary')

                # deploy model
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(value="")

                    with gr.Column():
                        deployModel = gr.Button(value="Deploy trained model", variant='primary')

            with gr.Column():
                output = gr.Text(value="", show_label=False)

        submitTrain.click(
            fn=finetune,
            inputs=[
                # sagemaker_role,
                # train_model_id,
                # hyperparameters,
                # training_dataset_s3_path,
                # training_instance_type,
                # s3_output_location
            ],
            outputs=[
                *dropdown_components,
                output
            ]
        )
        # deployModel.click(
        #     fn=deploy,
        #     inputs=[
        #         sagemaker_role,
        #     ],
        #     outputs=[
        #         *dropdown_components,
        #         output
        #     ]
        # )

    return [(aws_sd_inf, "Create AWS training job", "aws_sd_inf")]

script_callbacks.on_script_unloaded(onModelLoaded)
script_callbacks.on_model_loaded(onModelLoaded)
script_callbacks.on_ui_tabs(onUiTabs)
