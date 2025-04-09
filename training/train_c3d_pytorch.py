import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data.dataclass.NoduleCube import normal_cube_to_tensor

plt.rcParams['font.family'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import logging
import time
from datetime import datetime
from models.pytorch_c3d_tiny import C3dTiny

# 模型参数
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
CUBE_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置日志
def setup_logger(log_dir="./pytorch_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建logger
    logger = logging.getLogger('c3d_training')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class Luna16DataSet(Dataset):
    def __init__(self, files, labels, tranform =None):
        self.files = files
        self.labels = labels
        self.transform = tranform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_file = self.files[idx]
        item_label = self.labels[idx]
        item_label = torch.tensor(item_label, dtype=torch.long)
        item_data = np.load(npy_file)
        torch_item_data = normal_cube_to_tensor(item_data)
        if self.transform is not None:
            torch_item_data = self.transform(torch_item_data)
        return torch_item_data,item_label

def load_train_val_data(postive_dir, negative_dir):
    """
        从文件夹加载训练集和验证集
    :param postive_dir:
    :param negative_dir:
    :return:
    """
    postive_files = glob.glob(os.path.join(postive_dir, "*.npy"))
    negative_files = glob.glob(os.path.join(negative_dir, "*.npy"))
    min_samples = min(len(postive_files) ,len(negative_files))

    pos_files = random.sample(postive_files ,min_samples)
    neg_files = random.sample(negative_files, 2*min_samples)
    all_files = pos_files + neg_files
    labels = np.concatenate([np.ones(len(pos_files)), np.zeros(len(neg_files))])
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)

    files_train,files_val,label_train,label_val = train_test_split(all_files, labels,test_size=0.2, random_state=42)
    return files_train,files_val,label_train,label_val

def train_model(model,train_loader, val_loader, optimizer, criterion, scheduler, logger, writer, epoches, save_dir):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    train_accs = []
    os.makedirs(save_dir, exist_ok= True)
    for epoch in range(epoches):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for i, (inputs, labels ) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 检查损失值是否为NaN
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"警告：损失值包含NaN或Inf，跳过此批次")
                continue
                
            # 后向传播loss
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            #
            train_loss +=loss.item()
            _,predicteds = outputs.max(1)
            total +=labels.size(0)
            correct +=predicteds.eq(labels).sum().item()
            # 打印批次
            if (i + 1) % 100 == 0:
                print(f"{epoch +1}/{epoches}, Batch [{i + 1}/ {len(train_loader)}], Loss: {loss.item():.4f}")
        # 本次 epoch 平均训练损失
        epoch_train_loss = train_loss / len(train_loader)
        # 本次epoch 平均准确率
        epoch_train_acc = 100.0 * correct/total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        # 计算平均训练损失和准确率

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs,val_labels in val_loader:
                val_inputs = val_inputs.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                #
                val_outputs = model(val_inputs)
                batch_val_loss = criterion(val_outputs, val_labels)
                val_loss += batch_val_loss.item()
                _,predicted  = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += predicted.eq(val_labels).sum().item()
        # 计算平均验证损失和准确率
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
            
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 打印epoch信息
        logger.info(f'Epoch [{epoch+1}/{epoches}], '
                   f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
                   f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, '
                   f'Time: {epoch_time:.2f}s')
        
        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'Epoch [{epoch+1}]: 保存最佳模型, 验证准确率: {epoch_val_acc:.2f}%')
        
        # 每5个epoch保存一次检查点
        # if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_val_acc
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    logger.info(f'训练完成，最终模型已保存')
    
    # 绘制损失和准确率曲线
    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir)
    
    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制并保存损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建目录
    log_dir = "./pytorch_logs"
    checkpoint_dir = "./pytorch_checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(log_dir)
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    
    # 加载数据
    pos_sample_dir = r"J:\luna16_processed\positive_npys"
    neg_sample_dir = r"J:\luna16_processed\negative_npys"
    files_train, files_val, labels_train, labels_val = load_train_val_data(pos_sample_dir, neg_sample_dir)
    
    logger.info(f"训练集: {len(files_train)}个样本")
    logger.info(f"验证集: {len(files_val)}个样本")
    
    # 创建数据集和数据加载器
    train_dataset = Luna16DataSet(files_train, labels_train)
    val_dataset = Luna16DataSet(files_val, labels_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 创建模型
    model = C3dTiny().to(DEVICE)
    logger.info(f"模型结构:\n{model}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-8)
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # 训练模型
    train_losses, val_losses, train_accs, val_accs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        logger=logger,
        writer=writer,
        epoches=EPOCHS,
        save_dir=checkpoint_dir
    )
    
    # 关闭TensorBoard writer
    writer.close()
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main()

