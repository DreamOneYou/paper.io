


# Brain Tumor Segmentation with Missing Modalities via Latent Multi-source Correlation Representation



#          通过模态之间潜在的相关性，进行模态缺省的脑肿瘤分割

![](D:\typora\files\photo\22_1.png)



摘要：多模态MR图像可为脑肿瘤分割的精确性提供补充信息。但是，在临床实践中通常存在模态缺失。由于多模态之间存在很强的相关性，因此提出了一种新颖的相关性表示块来专门发现潜在的多模态相关性。由于获得了相关表示，即使在模态缺失的情况下，分割也会变得更加鲁棒。模型参数估计模块首先映射每个编码器产生的各自参数，然后，在这些参数下，相关表达式模块将所有单个表示转换为潜在的多模态相关表示。最后，通过注意机制将跨模态的相关表示融合到一个共享表示中，以强调分割的最重要特征。我们在BraTS 2018数据集上评估了我们的模型，其性能优于当前的最新方法，并且在缺少一种或多种模态时会产生可靠的结果。

关键词：脑肿瘤分割：多模态缺失：模态融合：潜在相关性：深度学习

## 1 引入

​		脑肿瘤是世界上最具攻击性和致命性的癌症之一，脑肿瘤的早期诊断在脑肿瘤的临床评估和治疗计划中起着重要作用。 磁共振成像（MRI）通常在放射学中用于诊断脑肿瘤，因为它依赖于可变的采集参数，例如T1加权（T1），对比增强的T1加权（T1c），T2- 加权（T2）和流体衰减反转恢复（FLAIR）图像。 不同的序列可以提供补充信息来分析神经胶质瘤的不同子区域。 T2和FLAIR适合检测肿瘤周围水肿的肿瘤，而T1和T1c适合检测没有肿瘤周围水肿的肿瘤核心[16]。 因此，应用多模态图像可以减少信息不确定性并提高临床诊断和分割精度。

​		由于医学专家分割脑肿瘤是昂贵且费时的，近来，已经有许多关于脑肿瘤自动分割的研究[5,6,9,14,15]，就需要完整的模态信息。但是，在临床实践中，成像方式通常不完整或缺失。当前，提出了许多方法来处理医学图像分割中模态缺失，这些方法可以大致分为三类：（1）在模态的所有可能子集上训练模型，这既复杂又费时。 （2）合成丢失的模态，然后使用完整的成像模态进行分割，而这需要额外的网络进行合成，并且合成的质量会直接影响分割性能。 （3）在潜在空间中融合可用模态以学习共享特征表示，然后将其投影到分割空间。这种方法比前两种方法更有效，因为它不需要学习多种可能的多模态子集，并且不会受到合成模态质量的影响。近来，有许多基于利用潜在特征表示来进行模态缺失的分割方法。当前最先进的网络架构来自Havaei，提出的HeMIS [7]分别学习每种模态的特征表示，然后跨各个模态计算第一和第二时刻，以估计最终的分割。但是，计算单个表示的均值和方差不能学习共享的潜在表示。 Lau等。 [11]引入了一个统一表示网络，该网络通过使用均值函数进行分割将可变数量的输入模态映射到一个统一表示中，而对潜在表示进行平均可以对潜在表示进行平均，从而可能会丢失一些重要信息。 Chen等。 [4]使用特征分解将输入模态分解为内容代码和外观代码，然后通过门控策略将内容代码融合到共享表示中进行分割。尽管该方法更加复杂且耗时，但因为每种模式都需要两个编码器，并且他们提出的融合方法仅从空间方向对内容代码进行加权，而无需考虑信道方向。Shen等。 [13]使用对抗性损失来形成域适应模型，以将特征图谱从缺失模态转换为完整模态的特征图，这只能应对一种缺失的模态情况。

​		模态缺失的分割挑战是学习一个共享的潜在表示，它可以获取图像模态的任何子集并产生鲁棒的分割。 为了有效地学习单个表示的潜在表示，在本文中，我们提出了一种新颖的脑肿瘤分割网络来应对模态缺失情况。 **我们的方法的主要贡献有三个方面：1）引入相关表示块以发现潜在的多模态相关表示。 2）提出了一种基于注意力机制并获得相关表示的融合策略，以学习不同方式下沿通道和空间方向的权重图。 3）提出了第一个多模态分割网络，该网络能够描述潜在的多模态相关表示，并且可以帮助对丢失的数据进行分割。**

## 2 方法

​		我们的网络受到U-Net架构的启发[12]。 为了在模态缺失的情况下保持健壮，我们将其调整为基于多编码器的框架。 它首先将3D可用模态作为每个编码器中的输入。 独立的编码器不仅可以学习特定于模态的特征表示，而且可以避免模态之间的错误匹配。 为了考虑到多模态之间的强相关性，我们提出了一个名为CR的模块，以发现模态之间的相关性。 然后，通过称为“融合”的注意力机制融合跨模态的相关表示，以强调针对分割的最具区分性的表示。 最后，对融合的潜在表示进行解码以形成最终的分割结果。 网络架构方案如图1所示。

![](D:\typora\files\优化算法\1.png)

### 2.1 对多源潜在相关性建模

​		受以下事实启发：多种MR方式之间存在很强的相关性，因为同一场景（同一位患者）被不同的方式所观察到[10]。从图2所示的MR图像的联合强度，我们可以观察到每两种模态之间强度分布的强烈相关性。为此，可以合理地假设模式之间的潜在表示形式也存在很强的相关性。并且我们引入了一个相关表示（CR）块（见图3）来发现潜在的相关性。 CR模块包括两个模块：模型参数估计模块（MPE模块）和线性相关表达式模块（LCE模块）。首先将输入模态{X i，...，X n}，其中n ＝ 4，输入到独立编码器f i（具有学习参数θ），以学习特定于模态的表示f i（X i |θi）。然后，MPE模块（一个具有两个完全连接的网络且具有LeakyReLU的网络）将特定于模态的表示fi（X i |θi）映射到一组独立参数Γi = {αi，βi，γi，δi }，Γi对每个模态都是唯一。最终，可以通过LCE模块（等式1）获得相关表示Fi（X i |θi）。由于我们有四个模态，因此我们可以从完整模态中学习四个相关性。对于测试，如果缺少一种模态，则可以从具有可用模态的学习相关表达式中大致恢复其特征表示。我们用最相似的模态作为缺失模态输入，以始终为训练模型提供四个输入。通过我们这种方法，我们不会丢失用于分割的缺失模态的信息。

![](D:\typora\files\优化算法\2.png)

![](D:\typora\files\优化算法\3.png)

### 2.2 融合策略

​		所提出的融合策略基于两个块：一个CR块（用于搜索四个模态的特征表示之间的潜在相关性）和一个融合块（如图4所示），**该融合块旨在去加权四个模态的特征表示，这个权值是基于他们对最终分割的贡献来确定**。在我们的结构中，我们独立使用四个编码器来获取对应于四个模态的四个特征表示。然后，**将CR块应用于编码器的最后一层**，以学习每个模态的潜在特征表示。在我们的相关假设下（图2），每个特征表示与其他三个模态线性相关。因此，CR块学习具有潜在相关性的四个新表示。**为了学习四种特征表示对分割的贡献，我们提出了一种基于注意力机制的融合块，**该融合块可以选择性地强调特征表示。**融合块由通道（模态）注意力模块和空间注意力模块组成**。第一个模块将四个特征表示作为输入来获取通道权重。而第二个模块着重于空间位置以获得空间权重。通过乘法将这两个权重与输入表示F组合在一起，以获得两个注意表示F c和F s，最后将它们加起来以获得融合表示F f。**模态的权重越大，对其细分的最终贡献就越大**。这样，**由于融合块，我们可以发现最相关的特征，而由于CR块，我们可以恢复缺失的特征**，从而在缺少模态的情况下使分割更加鲁棒。所提出的融合块可以直接适用于任何多峰融合问题，并且它鼓励网络沿空间方向和通道方向学习更多有意义的表示，**这优于简单的均值或最大融合方法。**

![](D:\typora\files\优化算法\4.png)

### 2.3 网路结构和学习过程

​		详细的网络架构框架如图5所示。当在图像中分割的不同区域时，可能需要使用不同的接收域，**由于接收域的局限性，标准U-Net无法获得足够的语义特征。受扩张卷积的启发，我们在编码器部分和解码器部分都使用了具有扩张卷积的残差块**（比率= 2、4）（res_dil块）来获得多个尺度的特征。**编码器包括卷积块，一个通过跳跃连接的res dil块**。所有卷积均为3×3×3。每个解码器级别均从上采样层开始，然后进行卷积以将特征数量减少2倍。然后将上采样特征与编码器相应级别的特征进行联结。联结后，我们使用res_dil块来增加感受野。另外，我们采用深监督策略[8]，通过集合来自不同级别的分割层以形成最终的网络输出。该网络由总损耗函数训练：L total = L dice + L 1，其中L 1是平均绝对损失。

![](D:\typora\files\优化算法\5.png)

### 3 数据和实施细节

​		**数据和预处理**。 实验中使用的数据集来自BraTS 2018数据集。 训练集包括285位患者，每位患者都有四种图像模式，包括T1，T1c，T2和FLAIR。 挑战结果，分为三个细分类别：完整肿瘤，肿瘤核心和增强肿瘤。 所提供的数据已由组织者进行了预处理：共同注册到相同的解剖模板，内插到相同的分辨率（1mm 3）并进行颅骨剥离。 真实标签已由专家手动标记。 我们使用标准程序进行了额外的预处理。 N4ITK [1]方法用于校正MRI数据的失真，强度归一化用于归一化每个患者的MR图像到正常化。 为了利用图像的空间上下文信息，我们使用3D图像并将其尺寸从155×240×240调整为128×128×128。

​		**实施细节**。 我们的网络在Keras中实现。 使用**Adam**优化器（初始学习率= 5e-4）对模型进行优化，总共50个epoch，每过10个epoch，学习率递减0.5。 我们将数据集随机分为80％训练和20％测试。 所有结果均通过在线评估平台获得。

![](D:\typora\files\优化算法\6.png)

## 4 实验结果

​		**定量分析**。我们方法的主要优点是使用相关表示，它可以发现模态之间的潜在相关表示，从而使模型在不存在模态的情况下变得健壮。为了证明我们模型的有效性，我们使用Dice得分作为度量标准，并比较其他两种方法。 （1）HeMIS [7]，当前最先进的分割方法，伴随着模态缺失。 （2）Org，我们模型的一个特殊案例，其中没有相关表示块。从表1可以看出，对于所有肿瘤区域，我们的方法在大多数情况下均能达到最佳效果。与HeMIS相比，我们的方法的Dice得分在缺少模态时逐渐下降，而在HeMIS中性能下降更为严重。与Org相比，相关表示块使模型在缺少模态的情况下更加健壮，这证明了所提出组件的有效性，也证明了我们的假设。我们还可以发现，**缺少FLAIR模式会导致所有区域的骰子得分急剧下降，因为FLAIR是显示整个肿瘤的主要模式。缺少T1和T2模式会导致所有区域的骰子得分略有下降。尽管缺少T1c方式会导致肿瘤核心和增强型肿瘤的骰子得分严重降低，因为T1c是显示肿瘤核心和增强肿瘤区域的主要方式**。

​		**质量分析**。 为了评估模型的鲁棒性，我们在BraTS 2018数据集上随机选择了三个示例，并在图6中显示了分割结果。 我们可以观察到，随着缺失模态数量的增加，由我们的健壮模型产生的分割结果只会略微下降，而不会突然急剧下降。 即使使用FLAIR和T1c模式，我们也可以实现不错的分割结果。

![](D:\typora\files\优化算法\7.png)

## 5 结论

​		我们提出了一种基于潜在多源相关表示和使用注意力机制的融合模块的新型多模态脑肿瘤分割网络，以使模型对丢失的数据具有鲁棒性。 我们证明了我们的方法可以在BraTS 2018数据集上使用完整模态和缺失模态产生竞争性结果。 比较结果还表明，FLAIR和T1c分别在分割完整的肿瘤和肿瘤核心方面具有重要作用。 所提出的方法可以推广到具有其他模态的其他分割任务（例如MR和CT图像）。将来，我们将在其他分割数据集上测试我们的方法，并与其他潜在表示学习方法进行比较[2,3]。 此外，我们将研究更复杂的模型来描述多源相关表示并将其适应丢失数据的问题。

## 参考文献

1. Avants, B.B., Tustison, N., Song, G.: Advanced normalization tools (ANTS).
   Insight J 2, 1–35 (2009)
2. Chartsias, A., Joyce, T., Giuffrida, M.V., Tsaftaris, S.A.: Multimodal MR synthesis
   via modality-invariant latent representation. IEEE Trans. Med. Imaging 37(3),
   803–814 (2017)
   Latent Multi-source Correlation Representation 541
3. Chartsias, A., et al.: Multimodal cardiac segmentation using disentangled repre-
   sentation learning. In: Pop, M., et al. (eds.) STACOM 2019. LNCS, vol. 12009, pp.
   128–137. Springer, Cham (2020). https://doi.org/10.1007/978-3-030-39074-7 14
4. Chen, C., Dou, Q., Jin, Y., Chen, H., Qin, J., Heng, P.-A.: Robust multimodal
   brain tumor segmentation via feature disentanglement and gated fusion. In: Shen,
   D., et al. (eds.) MICCAI 2019. LNCS, vol. 11766, pp. 447–456. Springer, Cham
   (2019). https://doi.org/10.1007/978-3-030-32248-9 50
5. Cui, S., Mao, L., Jiang, J., Liu, C., Xiong, S.: Automatic semantic segmentation of
   brain gliomas from MRI images using a deep cascaded neural network. J. Health-
   care Eng. 2018 (2018). Article ID 4940593
6. Havaei, M., et al.: Brain tumor segmentation with deep neural networks. Med.
   Image Anal. 35, 18–31 (2017)
7. Havaei, M., Guizard, N., Chapados, N., Bengio, Y.: HeMIS: hetero-modal image
   segmentation. In: Ourselin, S., Joskowicz, L., Sabuncu, M.R., Unal, G., Wells,
   W. (eds.) MICCAI 2016. LNCS, vol. 9901, pp. 469–477. Springer, Cham (2016).
   https://doi.org/10.1007/978-3-319-46723-8 54
8. Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K.H.: Brain
   tumor segmentation and radiomics survival prediction: contribution to the BRATS
   2017 challenge. In: Crimi, A., Bakas, S., Kuijf, H., Menze, B., Reyes, M. (eds.)
   BrainLes 2017. LNCS, vol. 10670, pp. 287–297. Springer, Cham (2018). https://
   doi.org/10.1007/978-3-319-75238-9 25
9. Kamnitsas, K., et al.: Efficient multi-scale 3D cnn with fully connected CRF for
   accurate brain lesion segmentation. Med. Image Anal. 36, 61–78 (2017)
10. Lapuyade-Lahorgue, J., Xue, J.H., Ruan, S.: Segmenting multi-source images using
    hidden Markov fields with copula-based multivariate statistical distributions. IEEE
    Trans. Image Process. 26(7), 3187–3195 (2017)
11. Lau, K., Adler, J., Sj¨ olund, J.: A unified representation network for segmentation
    with missing modalities. arXiv preprint arXiv:1908.06683 (2019)
12. Ronneberger, O., Fischer, P., Brox, T.: U-net: convolutional networks for biomed-
    ical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F.
    (eds.) MICCAI 2015. LNCS, vol. 9351, pp. 234–241. Springer, Cham (2015).
    https://doi.org/10.1007/978-3-319-24574-4 28
13. Shen, Y., Gao, M.: Brain tumor segmentation on MRI with missing modalities.
    In: Chung, A.C.S., Gee, J.C., Yushkevich, P.A., Bao, S. (eds.) IPMI 2019. LNCS,
    vol. 11492, pp. 417–428. Springer, Cham (2019). https://doi.org/10.1007/978-3-
    030-20351-1 32
14. Wang, G., Li, W., Ourselin, S., Vercauteren, T.: Automatic brain tumor segmenta-
    tion using cascaded anisotropic convolutional neural networks. In: Crimi, A., Bakas,
    S., Kuijf, H., Menze, B., Reyes, M. (eds.) BrainLes 2017. LNCS, vol. 10670, pp.
    178–190. Springer, Cham (2018). https://doi.org/10.1007/978-3-319-75238-9 16
15. Zhao, X., Wu, Y., Song, G., Li, Z., Zhang, Y., Fan, Y.: A deep learning model
    integrating fcnns and CRFs for brain tumor segmentation. Med. Image Anal. 43,
    98–111 (2018)
16. Zhou, T., Ruan, S., Canu, S.: A review: deep learning for medical image segmen-
    tation using multi-modality fusion. Array 3, 100004 (2019)
