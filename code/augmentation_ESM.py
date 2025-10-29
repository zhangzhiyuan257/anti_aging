# pip install torch torchvision torchaudio  # PyTorch（建议 CUDA 版本）
# pip install fair-esm  # ESM 官方库
# pip install biopython  # 处理蛋白质序列（可选）
import torch
# print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用
import esm
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from feature_extraction import DDE, readFasta, CKSAAP, CTriad

class Generator(nn.Module):

    def __init__(self, shape1):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(shape1, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, shape1),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output

class Discriminator(nn.Module):

    def __init__(self,shape1):
        super(Discriminator, self).__init__()

        self.fc1=nn.Linear(shape1, 512)
        self.relu=nn.LeakyReLU(0.2)
        self.fc2=nn.Linear(512, 256)
        self.relu=nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, inputs):
        out=self.fc1(inputs)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)
        return out.view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)  # 需要计算梯度
    # interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    # 计算梯度
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),  # 自动匹配设备和形状
        create_graph=True,  # 保留计算图以用于二阶导数
        retain_graph=True,   # 保留计算图
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
data_test = []
with open("../data/posi_0.csv", "r") as f:
    for i in f.readlines():
        tem = i.split(",")
        tem[1] = tem[1].split("\n")[0]
        tem = tuple(tem)
        data_test.append(tem)
data_test = data_test[1:]
print(data_test)  # Check if the file opens correctly

# 加载模型和 tokenizer
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 切换到推理模式

# 检查是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
sequences = data_test;
batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
token_embeddings = results["representations"][33]  # (batch_size, seq_len+2, 1280)

# 获取每个氨基酸的特征（去掉 CLS 和 EOS 标记）
per_residue_embeddings = token_embeddings[:, 1:-1, :]  # (1, seq_len, 1280)
# print(per_residue_embeddings.shape)
# 获取整个序列的全局特征（平均池化）
sequence_embedding = per_residue_embeddings.mean(dim=1)  # (1, 1280)
print(sequence_embedding.shape)

data = sequence_embedding
np.savetxt('../result/result_GAN/data_positive.txt',data)

FIXED_GENERATOR = False
LAMBDA = .1
CRITIC_ITERS = 5
BATCH_SIZE = len(data)
print("BATCH_SIZE is:", BATCH_SIZE)
print(data.shape)
print(data.shape[0])
ITERS = 100000
use_cuda = False
netG = Generator(data.shape[1])
netD = Discriminator(data.shape[1])
netD.apply(weights_init)
netG.apply(weights_init)
if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
one = torch.tensor(1, dtype=torch.float)  ###torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()
###iteration process
# 初始化存储列表
d_losses = []
g_losses = []
wasserstein_distances = []
for iteration in range(ITERS):
    for p in netD.parameters():
        p.requires_grad = True
    # data = inf_train_gen('data_positive')
    real_data = torch.FloatTensor(data)
    if use_cuda:
        real_data = real_data.cuda()
    real_data_v = real_data
    with torch.no_grad():
        noise = torch.randn(BATCH_SIZE, data.shape[1],device=real_data.device if use_cuda else torch.device('cpu'))
    # noisev = autograd.Variable(noise, volatile=True)
        fake = netG(noise, real_data_v)
    # fake = autograd.Variable(netG(noisev, real_data_v).data)
        fake_output=fake.detach().cpu().numpy()
    for iter_d in range(CRITIC_ITERS):
        netD.zero_grad()
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)
        with torch.no_grad():
            noise = torch.randn(BATCH_SIZE, data.shape[1],device=real_data.device if use_cuda else torch.device('cpu'))
            fake = netG(noise, real_data_v)
        D_fake = netD(fake.detach())
        D_fake = D_fake.mean()
        D_fake.backward(one)
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

        # 记录判别器损失和W距离（每次CRITIC_ITERS都记录）
        d_losses.append(D_cost.item())
        wasserstein_distances.append(Wasserstein_D.item())
    ###save generated sample features every 200 iteration
    if iteration%200 == 0:
        fake_writer = open("../result/result_GAN/Iteration_"+str(iteration)+".txt","w")
        for rowIndex in range(len(fake_output)):
            for columnIndex in range(len(fake_output[0])):
                fake_writer.write(str(fake_output[rowIndex][columnIndex]) + ",")
            fake_writer.write("\n")
        fake_writer.flush()
        fake_writer.close()
    if not FIXED_GENERATOR:
        netG.zero_grad()
        for p in netD.parameters():
            p.requires_grad = False
        # real_data = torch.Tensor(data,device='cuda' if use_cuda else 'cpu')
        # real_data_v = autograd.Variable(real_data)
        noise = torch.randn(BATCH_SIZE, data.shape[1])
        if use_cuda:
            noise = noise.cuda()
        # noisev = autograd.Variable(noise)
        fake = netG(noise, real_data)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
        # 记录生成器损失（每次大迭代记录一次）
        g_losses.append(G_cost.item())
np.savetxt('../result/result_GAN/g_losses1.txt',np.array(g_losses))
np.savetxt('../result/result_GAN/d_losses1.txt',np.array(d_losses))
np.savetxt('../result/result_GAN/wasserstein_distances1.txt',np.array(wasserstein_distances))
# 训练完成后绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(d_losses, label="Discriminator Loss (D_loss)")
plt.plot(g_losses, label="Generator Loss (G_loss)")
plt.plot(wasserstein_distances, label="Wasserstein Distance")
plt.xlabel("Iterations")
plt.ylabel("Loss/Value")
plt.title("WGAN-GP Training Loss")
plt.legend()
plt.grid()
plt.savefig("../result/training_loss_curve1.png")  # 保存图像
plt.show()
####test model result, LOOCV to select optimal pseudo samples
with open("../result/result_GAN/data_positive.txt") as f:
    MatrixFeatures = [list(x.split(" ")) for x in f]
realFeatures = [line[:] for line in MatrixFeatures[:]]
realDataset = np.array(realFeatures, dtype='float32')
# Adding equal numbers of binary labels
label=[]
for rowIndex in range(len(realDataset)):
    label.append(1)
for rowIndex in range(len(realDataset)):
    label.append(0)
labelArray=np.asarray(label)
opt_diff_accuracy_05=0.5
opt_Epoch=0
opt_accuracy=0
allresult=[]
epochs = []
accuracies = []
for indexEpoch in range(250):
    epoch = indexEpoch * 200
    with open("../result/result_GAN/Iteration_"+str(epoch)+".txt") as f:
          MatrixFeatures = [list(x.split(",")) for x in f]
    fakeFeatures = [line[:-1] for line in MatrixFeatures[:]]
    fakedataset = np.array(fakeFeatures, dtype='float32')
    realFakeFeatures=np.vstack((realDataset, fakedataset))
    prediction_list=[]
    real_list=[]
    ####LOOCV
    loo = LeaveOneOut()
    loo.get_n_splits(realFakeFeatures)
    for train_index, test_index in loo.split(realFakeFeatures):
        X_train, X_test = realFakeFeatures[train_index], realFakeFeatures[test_index]
        y_train, y_test = labelArray[train_index], labelArray[test_index]
        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        predicted_y = knn.predict(X_test)
        prediction_list.append(predicted_y)
        real_list.append(y_test)
    accuracy=accuracy_score(real_list, prediction_list)
    epochs.append(epoch)
    accuracies.append(accuracy)
    allresult.append(str(indexEpoch)+"%"+str(accuracy))
    diff_accuracy_05=abs(accuracy-0.5)
    if diff_accuracy_05 < opt_diff_accuracy_05:
        opt_diff_accuracy_05=diff_accuracy_05
        opt_Epoch=epoch
        opt_accuracy=accuracy
print(str(opt_Epoch)+"%"+str(opt_accuracy))

# 绘制Accuracy曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, 'b-', label='Accuracy (1-NN LOOCV)')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (0.5)')
plt.scatter(opt_Epoch, opt_accuracy, color='g', s=100, label=f'Best Epoch: {opt_Epoch}\nAccuracy: {opt_accuracy:.3f}')
plt.xlabel('Training Iteration (Epoch)')
plt.ylabel('Accuracy')
plt.title('GAN Evaluation: Discrimination Accuracy vs Training Epoch')
plt.legend()
plt.grid()
plt.savefig('../result/result_GAN/accuracy_curve.png', dpi=300, bbox_inches='tight')
plt.show()

np.savetxt('../result/result_GAN/metrics.txt',np.array(allresult))
# 保存单条序列特征
# np.save("protein1_embeddings.npy", per_residue_embeddings.cpu().numpy())

# # 保存全局特征
# np.save("protein1_global_embedding.npy", sequence_embedding.cpu().numpy())
# # 提取多层的特征（例如第 6, 12, 33 层）
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[6, 12, 33])
#
# layer6_embeddings = results["representations"][6]  # 第 6 层特征
# layer12_embeddings = results["representations"][12]  # 第 12 层特征
#
# from scipy.spatial.distance import cosine

# emb1 = sequence_embedding1.cpu().numpy().flatten()
# emb2 = sequence_embedding2.cpu().numpy().flatten()
#
# similarity = 1 - cosine(emb1, emb2)  # 余弦相似度