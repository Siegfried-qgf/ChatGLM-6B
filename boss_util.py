import logging
import os
from pathlib import Path
from botocore.config import Config
import boto3


boss_config = dict(
    AccessKey="XsoNq0QPQJToqHsG",
                   SecretKey="ylQHuuY8LIkIbjSHqsoW0tUD9aehQQse",
                   Bucket="llm-data",
                   Endpoint="http://shjd-inner-boss.bilibili.co",
                   region="shjd-inner"
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'begin progress ...')


model_list = [
    "config.json",
"generation_config.json",
"pytorch_model-00001-of-00033.bin",
"pytorch_model-00002-of-00033.bin",
"pytorch_model-00003-of-00033.bin",
"pytorch_model-00004-of-00033.bin",
"pytorch_model-00005-of-00033.bin",
"pytorch_model-00006-of-00033.bin",
"pytorch_model-00007-of-00033.bin",
"pytorch_model-00008-of-00033.bin",
"pytorch_model-00009-of-00033.bin",
"pytorch_model-00010-of-00033.bin",
"pytorch_model-00011-of-00033.bin",
"pytorch_model-00012-of-00033.bin",
"pytorch_model-00013-of-00033.bin",
"pytorch_model-00014-of-00033.bin",
"pytorch_model-00015-of-00033.bin",
"pytorch_model-00016-of-00033.bin",
"pytorch_model-00017-of-00033.bin",
"pytorch_model-00018-of-00033.bin",
"pytorch_model-00019-of-00033.bin",
"pytorch_model-00020-of-00033.bin",
"pytorch_model-00021-of-00033.bin",
"pytorch_model-00022-of-00033.bin",
"pytorch_model-00023-of-00033.bin",
"pytorch_model-00024-of-00033.bin",
"pytorch_model-00025-of-00033.bin",
"pytorch_model-00026-of-00033.bin",
"pytorch_model-00027-of-00033.bin",
"pytorch_model-00028-of-00033.bin",
"pytorch_model-00029-of-00033.bin",
"pytorch_model-00030-of-00033.bin",
"pytorch_model-00031-of-00033.bin",
"pytorch_model-00032-of-00033.bin",
"pytorch_model-00033-of-00033.bin",
"pytorch_model.bin.index.json",
"special_tokens_map.json",
"tokenizer.model",
"tokenizer_config.json"
]

model_alpaca_list = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "pytorch_model-00001-of-00003.bin",
    "pytorch_model-00002-of-00003.bin",
    "pytorch_model-00003-of-00003.bin",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "trainer_state.json",
    "training_args.bin"
]

model_glm_1m_cn = [
    "added_tokens.json",
    "cog-pretrain.model",
    "config.json",
    "configuration_glm.py",
    "modeling_glm.py",
    "pytorch_model.bin",
    "tokenization_glm.py",
    "tokenizer_config.json"
]


def data_downloader():
    file_list = [
        "dial_final.json",
    ]

    my_config = Config(s3={'addressing_style': 'path'})  # 必须: 必须添加这项配置

    s3_resource = boto3.resource('s3',
                                 aws_access_key_id=boss_config['AccessKey'],
                                 aws_secret_access_key=boss_config['SecretKey'],
                                 region_name=boss_config["region"],
                                 endpoint_url=boss_config["Endpoint"],  # 替换为申请 Bucket 时提供的 Endpoint
                                 config=my_config)

    os.makedirs("./data", exist_ok=True)
    for f in file_list:
        remote_path = f"waka/{f}"
        local_path = f"./data/{f}"
        logger.info(f)

        s3_resource.Object(boss_config["Bucket"], remote_path).download_file(local_path)


def data_transfer(file_names, local_dir, remote_dir, trans_type):
    """
    local_files: list of str
    remote_dir: str of remote path
    trans_type: "download" or "upload"

    """
    my_config = Config(s3={'addressing_style': 'path'})  # 必须: 必须添加这项配置

    s3_resource = boto3.resource('s3',
                                 aws_access_key_id=boss_config['AccessKey'],
                                 aws_secret_access_key=boss_config['SecretKey'],
                                 region_name=boss_config["region"],
                                 endpoint_url=boss_config["Endpoint"],  # 替换为申请 Bucket 时提供的 Endpoint
                                 config=my_config)

    if trans_type == "download":
        # os.makedirs(local_dir, exist_ok=True)
        for f_name in file_names:
            remote_path = f"{remote_dir}/{f_name}"
            local_path = f"{local_dir}/{f_name}"

            logger.info(f_name)

            s3_resource.Object(boss_config["Bucket"], remote_path).download_file(local_path)
    
    elif trans_type == "upload":
        for f_name in file_names:
            remote_path = f"{remote_dir}/{f_name}"
            local_path = f"{local_dir}/{f_name}"

            logger.info(f_name)

            s3_resource.Object(boss_config["Bucket"], remote_path).upload_file(local_path)
    
    else:
        raise ValueError(f"Wrong trans_type: {trans_type}. You have to choose between download and upload.")


def upload_data():
    file_names = ["train_0.json"]
    data_transfer(file_names=file_names, local_dir="/workspace/ChatGLM-6B/dataset/ultrachat", remote_dir="xiaofeng/ultrachat", trans_type="upload")


def upload_model():
    data_transfer(file_names=['model_multiturn_1024.pt'], local_dir="/workspace/javis/samples/glm/model_multiturn", remote_dir="model_multiturn", trans_type="upload")


def download_llama():
    os.makedirs('model_llama_7b', exist_ok=True)
    os.makedirs('model_finetune', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    data_transfer(file_names=model_list, local_dir="model_llama_7b", remote_dir="model_llama_7b", trans_type="download")
    data_transfer(file_names=model_alpaca_list, local_dir="model_finetune", remote_dir="model_alpaca", trans_type="download")
    data_transfer(file_names=['guanaco_data_mixed.json'], local_dir="dataset", remote_dir="data_llama_7b", trans_type="download")


def download_alpaca(data_file):
    data_file = Path(data_file).name
    os.makedirs('model_finetune_500k', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    data_transfer(file_names=model_alpaca_list, local_dir="model_finetune_500k", remote_dir="model_alpaca_500k", trans_type="download")
    data_transfer(file_names=[data_file], local_dir="dataset", remote_dir="data_llama_7b", trans_type="download")

if __name__ == "__main__":
    upload_data()







    
