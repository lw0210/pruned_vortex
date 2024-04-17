# original version from NeuralRecon
# https://github.com/zju3dv/NeuralRecon

# Modified by Noah Stier

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# 
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
#    1. Definitions.
# 
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
# 
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
# 
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
# 
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
# 
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
# 
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
# 
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
# 
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
# 
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
# 
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
# 
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
# 
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
# 
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
# 
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
# 
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
# 
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
# 
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
# 
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
# 
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
# 
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
# 
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
# 
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
# 
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
# 
#    END OF TERMS AND CONDITIONS
# 
#    APPENDIX: How to apply the Apache License to your work.
# 
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
# 
#    Copyright [yyyy] [name of copyright owner]
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torchvision


import torch.nn as nn
import math
from torch.nn import functional as F
# class SqueezeExcite(nn.Module):
#     def __init__(self,
#                  input_c: int,   # block input channel
#                  expand_c: int,  # block expand channel
#                  se_ratio: float = 0.25):
#         super(SqueezeExcite, self).__init__()
#         squeeze_c = int(input_c * se_ratio)
#         self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
#         self.act1 = nn.SiLU()  # alias Swish   换成nn.Hardswish(inplace=True)
#         # self.act1 = nn.Hardswish()
#
#         self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
#         # self.act2 = nn.Sigmoid() #F.hardsigmoid()
#         self.act2 = F.hardsigmoid
#
#     def forward(self, x):
#         scale = x.mean((2, 3), keepdim=True)
#         scale = self.conv_reduce(scale)
#         scale = self.act1(scale)
#         scale = self.conv_expand(scale)
#         scale = self.act2(scale)
#         return scale * x




# class eca_block(nn.Module):
#     # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
#     def __init__(self, in_channel, b=1, gama=2):
#         # 继承父类初始化
#         super(eca_block, self).__init__()
#
#         # 根据输入通道数自适应调整卷积核大小
#         kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
#         # 如果卷积核大小是奇数，就使用它
#         if kernel_size % 2:
#             kernel_size = kernel_size
#         # 如果卷积核大小是偶数，就把它变成奇数
#         else:
#             kernel_size = kernel_size + 1
#
#         # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
#         padding = kernel_size // 2
#
#         # 全局平均池化，输出的特征图的宽高=1
#         self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
#         # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
#         # 这个1维卷积需要好好了解一下机制，这是改进SENet的重要不同点
#         self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
#                               bias=False, padding=padding)
#         # sigmoid激活函数，权值归一化
#         self.sigmoid = nn.Sigmoid()
#
#     # 前向传播
#     def forward(self, inputs):
#         # 获得输入图像的shape
#         b, c, h, w = inputs.shape
#
#         # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
#         x = self.avg_pool(inputs)
#         # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
#         x = x.view([b, 1, c])  # 这是为了给一维卷积
#         # 1D卷积 [b,1,c]==>[b,1,c]
#         x = self.conv(x)
#         # 权值归一化
#         x = self.sigmoid(x)
#         # 维度调整 [b,1,c]==>[b,c,1,1]
#         x = x.view([b, c, 1, 1])
#
#         # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
#         outputs = x * inputs
#         return outputs







class MnasMulti(torch.nn.Module):
    def __init__(self, output_depths, pretrained=True):
        super().__init__()
        MNASNet = torchvision.models.mnasnet1_0(pretrained=pretrained)
        self.conv0 = torch.nn.Sequential(
            MNASNet.layers._modules["0"],
            MNASNet.layers._modules["1"],
            MNASNet.layers._modules["2"],
            MNASNet.layers._modules["3"],
            MNASNet.layers._modules["4"],
            MNASNet.layers._modules["5"],
            MNASNet.layers._modules["6"],
            MNASNet.layers._modules["7"],
            MNASNet.layers._modules["8"],
        )
        self.conv1 = MNASNet.layers._modules["9"]
        self.conv2 = MNASNet.layers._modules["10"]

        final_chs = 80

        self.inner1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_depths[1]),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(output_depths[1], final_chs, 1, bias=False),
        )
        self.inner2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_depths[2]),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(output_depths[2], final_chs, 1, bias=False),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[0], 1, bias=False),
        )
        self.out2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[1], 3, bias=False, padding=1),
        )
        self.out3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(final_chs),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(final_chs, output_depths[2], 3, bias=False, padding=1),
        )


        # self.eca = eca_block(in_channel=24)
        # self.eca1 = eca_block(in_channel=40)
        # self.eca2 = eca_block(in_channel=80)

        # self.se = SqueezeExcite(24, 24)




        torch.nn.init.kaiming_normal_(self.inner1[2].weight)
        torch.nn.init.kaiming_normal_(self.inner2[2].weight)
        torch.nn.init.kaiming_normal_(self.out1[2].weight)
        torch.nn.init.kaiming_normal_(self.out2[2].weight)
        torch.nn.init.kaiming_normal_(self.out3[2].weight)

    def forward(self, x):
        conv0 = self.conv0(x)

        # conv0 = self.eca(conv0)###############
        # conv0 = self.se(conv0)################


        conv1 = self.conv1(conv0)

        # conv1 = self.eca1(conv1)#################

        conv2 = self.conv2(conv1)

        # conv2 = self.eca2(conv2)#####################



        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["coarse"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=False
        ) + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["medium"] = out

        intra_feat = torch.nn.functional.interpolate(
            intra_feat, scale_factor=2, mode="bilinear", align_corners=False
        ) + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["fine"] = out

        return outputs
