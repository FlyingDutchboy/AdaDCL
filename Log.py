import logging
from datetime import datetime
import os

from Args import args
log_directory = "./log"
log_save_path = ''
def setup_logger():
    global log_save_path
    # 确保主日志目录存在
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 创建 args.data 对应的子目录
    log_subdirectory = os.path.join(log_directory, args.data)
    if not os.path.exists(log_subdirectory):
        os.makedirs(log_subdirectory)

    # 创建以“时间_任务名”命名的独立文件夹
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_save_path = os.path.join(log_subdirectory, f"{timestamp}_{args.name}")
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # 定义日志文件的路径
    log_filename = os.path.join(log_save_path, f"{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # 配置日志
    logging.basicConfig(
        filename=log_filename,  # 日志文件名
        level=logging.INFO,  # 日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
    )
def log_print(*args, **kwargs):
    message = " ".join(map(str, args))  # 将所有参数拼接为字符串
    logging.info(message)  # 写入日志文件
    print(message, **kwargs)  # 同时输出到控制台（可选）

