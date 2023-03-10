from projen.awscdk import AwsCdkPythonApp

project = AwsCdkPythonApp(
    author_email="yike5460@163.com",
    author_name="yike5460",
    cdk_version="2.1.0",
    module_name="sd_aws_extensions",
    name="sd-aws-extensions",
    version="0.1.0",
)

project.synth()