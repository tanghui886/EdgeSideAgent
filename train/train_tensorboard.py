# 导入必要的库
from unsloth import FastVisionModel
import torch
import os
from datetime import datetime
import threading
from transformers import TrainerCallback

# 设置TensorBoard日志目录
log_dir=f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir,exist_ok=True)

# 导入TensorBoard
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil
import time

# 全局变量用于硬件监控
monitoring_active=True

# 创建TensorBoard写入器
writer=SummaryWriter(log_dir=log_dir)


# 监控硬件信息的函数
def log_hardware_metrics(step):
    try:
        # CPU使用率（每个核心和总体）
        cpu_percent=psutil.cpu_percent(interval=1)
        cpu_percent_per_core=psutil.cpu_percent(interval=1,percpu=True)

        writer.add_scalar('Hardware/CPU_Usage_Total',cpu_percent,step)
        # for i, core_usage in enumerate(cpu_percent_per_core):
        #    writer.add_scalar(f'Hardware/CPU_Core_{i}_Usage', core_usage, step)

        # CPU频率
        try:
            cpu_freq=psutil.cpu_freq()
            if cpu_freq:
                writer.add_scalar('Hardware/CPU_Frequency_MHz',cpu_freq.current,step)
                writer.add_scalar('Hardware/CPU_Frequency_Min_MHz',cpu_freq.min,step)
                writer.add_scalar('Hardware/CPU_Frequency_Max_MHz',cpu_freq.max,step)
        except:
            pass

        # 内存使用情况
        memory=psutil.virtual_memory()
        writer.add_scalar('Hardware/Memory_Usage_GB',memory.used / (1024 ** 3),step)
        writer.add_scalar('Hardware/Memory_Available_GB',memory.available / (1024 ** 3),step)
        writer.add_scalar('Hardware/Memory_Total_GB',memory.total / (1024 ** 3),step)
        writer.add_scalar('Hardware/Memory_Percent',memory.percent,step)

        # 交换内存
        swap=psutil.swap_memory()
        writer.add_scalar('Hardware/Swap_Usage_GB',swap.used / (1024 ** 3),step)
        writer.add_scalar('Hardware/Swap_Total_GB',swap.total / (1024 ** 3),step)
        writer.add_scalar('Hardware/Swap_Percent',swap.percent,step)

        # 磁盘IO
        disk_io=psutil.disk_io_counters()
        if disk_io:
            writer.add_scalar('Hardware/Disk_Read_Bytes_MB',disk_io.read_bytes / (1024 ** 2),step)
            writer.add_scalar('Hardware/Disk_Write_Bytes_MB',disk_io.write_bytes / (1024 ** 2),step)
            writer.add_scalar('Hardware/Disk_Read_Count',disk_io.read_count,step)
            writer.add_scalar('Hardware/Disk_Write_Count',disk_io.write_count,step)

        # 网络IO
        net_io=psutil.net_io_counters()
        writer.add_scalar('Hardware/Network_Received_Bytes_MB',net_io.bytes_recv / (1024 ** 2),step)
        writer.add_scalar('Hardware/Network_Sent_Bytes_MB',net_io.bytes_sent / (1024 ** 2),step)

        # GPU使用情况
        try:
            gpus=GPUtil.getGPUs()
            for i,gpu in enumerate(gpus):
                writer.add_scalar(f'Hardware/GPU_{i}_Usage_Percent',gpu.load * 100,step)
                writer.add_scalar(f'Hardware/GPU_{i}_Memory_Used_MB',gpu.memoryUsed,step)
                writer.add_scalar(f'Hardware/GPU_{i}_Memory_Total_MB',gpu.memoryTotal,step)
                writer.add_scalar(f'Hardware/GPU_{i}_Memory_Percent',gpu.memoryUtil * 100,step)
                writer.add_scalar(f'Hardware/GPU_{i}_Temperature_C',gpu.temperature,step)

                # GPU功率（如果支持）
                if hasattr(gpu,'powerDraw'):
                    writer.add_scalar(f'Hardware/GPU_{i}_Power_W',gpu.powerDraw,step)

        except Exception as e:
            print(f"无法获取GPU信息: {e}")

        # 进程级别的内存使用
        process=psutil.Process()
        process_memory=process.memory_info()
        writer.add_scalar('Process/RSS_MB',process_memory.rss / (1024 ** 2),step)
        writer.add_scalar('Process/VMS_MB',process_memory.vms / (1024 ** 2),step)

        # 进程CPU使用率
        writer.add_scalar('Process/CPU_Percent',process.cpu_percent(),step)

        print(f"硬件监控完成 - 步骤: {step}")

    except Exception as e:
        print(f"硬件监控出错: {e}")


# 硬件监控线程函数
def hardware_monitoring_thread():
    """独立的硬件监控线程"""
    step=0
    while monitoring_active:
        try:
            log_hardware_metrics(step)
            step+=1
            time.sleep(5)  # 每5秒记录一次硬件信息
        except Exception as e:
            print(f"硬件监控线程出错: {e}")
            time.sleep(1)


# 启动硬件监控线程
monitor_thread=threading.Thread(target=hardware_monitoring_thread,daemon=True)
monitor_thread.start()

# 加载模型和分词器
model,tokenizer=FastVisionModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2.5-VL-7B-Instruct",  # 指明要去加载的模型在本地的路径
    load_in_4bit=False,  # 使用 4bit 量化减少内存的使用。 默认是 False，则会使用 16bit 训练
    use_gradient_checkpointing="unsloth"  # 设置为 True 或者 设置为 "unsloth" 支持更长的上下文
)

model=FastVisionModel.get_peft_model(
    model,

    # 首先整体上设置多模态大模型中哪些层去进行LoRA微调
    finetune_vision_layers=False,  # 是否微调视觉相关层
    finetune_language_layers=True,  # 是否微调语言模型相关的层
    finetune_attention_modules=True,  # 是否微调注意力相关层
    finetune_mlp_modules=True,  # 是否微调 MLP 层，影响验证集准确率

    # LoRA 相关核心参数
    r=16,  # rank值，越大调整的参数就会越多，当然也会越准；但容易过拟合
    lora_alpha=16,  # 推荐至少 alpha==r，影响训练过程中收敛的速度和效果
    lora_dropout=0,  # dropout ratio 用于放置过拟合的手段
    bias="none",  # 截距项，是否添加偏置项

    # 配置其它的一些超参数
    # random_state = 3407, # 随机种子，如果不去设置，那么本后的本质就是随机种子这个数值本身会是个随机数，确保使用具备可重复性
    use_rslora=False,  # 如果设置成True，它就会去使用 rank stabilized LoRA
    loftq_config=None,  # 关于 LoftQ 配置，用于量化
    # target_modules = "all-linear", # 可选项，可以指导需要应用LoRA的具体模块是哪些
)

from datasets import load_dataset,Dataset
from sklearn.model_selection import train_test_split
# 假设你已加载 train_dataset
dataset = load_dataset("./data/", split="train")
test_dataset = load_dataset("./data/", split="test")

# 转为 list 或 Dataset 对象便于划分
if isinstance(dataset, list):
    # 如果已经是 list
    train_val_split = Dataset.from_list(dataset).train_test_split(test_size=0.1, seed=42) #0.1 数据集的10%作为验证集，剩余的90%作为训练集 42 随机种子 确保每次运行代码时数据集的划分方式相同，使结果可重现
else:
    # 如果是 Hugging Face Dataset
    train_val_split = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = train_val_split['train'] # 训练集
val_dataset = train_val_split['test']  # 验证集

instruction="你是一名专业的放射科医生。请准确描述你在图片中看到的内容。"
def convert_to_conversation(sample):
    conversation=[
        {
            "role":"user",
            "content":[{"type":"text","text":instruction},{"type":"image","image":sample['image']}]
        },
        {
            "role":"assistant",
            "content":[{"type":"text","text":sample['caption']}]
        }
    ]
    return {"messages":conversation}

converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
converted_val_dataset = [convert_to_conversation(sample) for sample in val_dataset]
converted_test_dataset = [convert_to_conversation(sample) for sample in test_dataset]  # 最后评估用
# converted_dataset=[convert_to_conversation(sample) for sample in dataset]

# 导入必要模块
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer,SFTConfig

class ValidationCallback(TrainerCallback):
    def __init__(self, trainer, val_dataset, eval_steps=5, metric="loss"):
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.eval_steps = eval_steps
        self.best_metric = float('inf') if metric == "loss" else 0.0
        self.best_step = 0
        self.metric = metric  # 'loss' or 'accuracy' etc.

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\n>>> 步骤 {state.global_step}: 开始验证集评估...")

            # 评估验证集
            eval_results = self.trainer.evaluate(eval_dataset=self.val_dataset)
            val_loss = eval_results["eval_loss"]

            # 写入 TensorBoard
            writer.add_scalar("Validation/Loss", val_loss, state.global_step)

            print(f"验证损失: {val_loss:.6f}")

            # 判断是否为最佳模型
            is_best = False
            if self.metric == "loss":
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self.best_step = state.global_step
                    is_best = True
                    print(f"🎉 新的最佳模型！步骤: {state.global_step}, 验证损失: {val_loss:.6f}")

            # 如果是最佳模型，保存
            if is_best:
                save_path = f"./checkpoints/best_model_step_{state.global_step}"
                self.trainer.save_model(save_path)
                print(f"✅ 最佳模型已保存至: {save_path}")

            # 可选：记录到 TensorBoard
            writer.add_scalar("Validation/Best_Loss", self.best_metric, state.global_step)

FastVisionModel.for_training(model)  # 切换训练模式！
# 配置训练器
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model,tokenizer),  # 必须要设置的
    train_dataset=converted_train_dataset,
    eval_dataset=converted_val_dataset,
    args=SFTConfig(
        # 基础训练的相关参数
        per_device_train_batch_size=2,  # 对应每个显卡的批次包含的样本数
        gradient_accumulation_steps=4,  # 梯度累积步数，目的就是节省显存空间，用时间换空间了
        max_steps=50,  # 最大训练次数，当我们给定 max_steps后，它会覆盖 num_train_epochs
        # num_train_epochs = 3,           # 完整训练一次，把训练样本交给模型学习一遍
        save_steps=5,
        save_total_limit=10,
        # save_every_n_epochs_equivalent = 1,  # 相当于每“1轮”保存一次（即使你用的是 max_steps）
        # save_steps = max(1, max_steps // 10),  # 至少保存1次，最多10次
        output_dir = "./checkpoints",
        # 优化器、学习率相关参数设置
        learning_rate=2e-4,  # 学习率 越小越稳，但是可能学不到东西；越大 学习越快，但容易“跳过最优解”
        warmup_steps=5,  # 预热步数/预热迭代次数
        lr_scheduler_type="linear",  # 线性学习率调整器
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),

        # Adam 优化器本身结合动量和自适应学习率，能够适应稀疏梯度，在许多任务上表现都很出色。
        # AdamW 8-bit 是 AdamW 优化器的一种低精度变体，目的降低内存消耗和计算成本。
        # 低精度计算：标准的是 32bit 浮点数运算，每个数值占用32位，8bit是不是相当于仅是原来的1/4，这种低精度训练方法特别适合内存受限的环境去训练大模型，
        # 例如在边缘设备上进行训练，或者需要训练特别大的大模型的时候。
        optim="adamw_8bit",  # 使用 8bit AdamW 优化器来调整参数
        weight_decay=0.01,  # 权重衰减，其实本质上是和正则项有关系的 L1/L2

        # 添加日志记录回调
        logging_dir=log_dir,
        logging_steps=1,
        report_to=["tensorboard"],
    )
)
validation_callback = ValidationCallback(trainer, converted_val_dataset, eval_steps=5)
trainer.add_callback(validation_callback)

import torch._dynamo.config

# 提高缓存大小限制（默认可能是 64）
torch._dynamo.config.cache_size_limit=128

# 关闭严格模式（允许更多重新编译）
torch._dynamo.config.suppress_errors=True

try:
    trainer_stats=trainer.train()

finally:
    print("🏆 最终测试集评估（仅此一次！）")
    test_results=trainer.evaluate(eval_dataset=converted_test_dataset)
    for k,v in test_results.items():
        print(f"{k}: {v:.6f}")
        writer.add_scalar(f"Final_Test/{k}",v,0)
    # 模型保存到本地，LoRA
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

    # 保存合并的模型
    # model.save_pretrained_merged("merged_model",tokenizer,save_method="merged_16bit")

    # 停止监控线程
    monitoring_active=False
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=2.0)

    # 关闭TensorBoard写入器
    writer.close()
    print(f"\n✅ 训练完成！最佳模型步骤: {getattr(validation_callback,'best_step','N/A')}")
    print(f"训练完成，TensorBoard日志保存在: {log_dir}")
    print("要启动TensorBoard，请运行: tensorboard --logdir=runs/")