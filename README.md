# A Dynamic Graph Neural Network Architecture for Multi-tasking on Image data
## INTRODUCTION

The Graph Convolution Networks (GCNs) are usually applied on graph data, such as social networks , citation networks and biochemical graphs. Using GCN for directly processes the image data has been started from 2018. There few GCN-based papers on image segmentation task. The advantages of graph representation of the image
include:
1) graph is a generalized data structure that grid (nodes lie on 2D lattices) and sequence can be viewed as a specialcase of graph;
2) Convolution Neural Networks (CNNs) apply sliding window on the image and introduce the shift-invariance and ordered information;
3) convolution filters only extract local features and neglect long-range self-similarity information, which is the vital information existing in image data.

Since the graph neural network (GNN) [1] was first proposed, the techniques for processing graphs have been researched a lot. In this section, we review the development of graph neural network, especially GCN and
its applications on visual tasks. In recent work, Ahmed Elmoogy et al., [2] used the graph neural network (GNN) for image pose estimation. They leverage the power of GNN to process the pretrained features as output of ResNet50 CNN architecture. For building connections (edges) between images in the form of binary adjacency matrix, they used 𝑘-nearest neighbors (KNN) algorithm [3] to search for the nearest 𝑘 neighbors of every node based on the 𝐿_2 distance.

However, for the first time according to my knowledge, Lu et al., [4] applied graph convolution network in image semantic segmentation to extract image features and the graph structure is constructed based on the 𝑘 nearest neighbour methods where the weight adjacent matrix is generated with the Gaussian kernel function. Moreover, Han et al. [5] proposed a Vision GNN (ViG), which splits the image into many blocks regarded as nodes and constructs a graph representation by connecting the nearest neighbors, then uses GNNs to process it for image recognition and object detection tasks. Motivated by the success of ViG model, in [6], authors proposed a ViGUNet to utilize the powerful functions of ViG for 2D medical image segmentation. Admittedly, this model needs a considerable computational resources ( nearly 0.7 billion parameters).

In this project, I am going to  propose a dynamic multi task graph neural network (MG-Unet) for simultaneous image segsegmentation and pose estimation which is the first model to construct a multi-task graph structures. MG-Unet has a U-shaped architecture with the encoder, the decoder, skip connections and fully connected layers. It was noticed that the down-sampling part of the U-net is shared between the segmentation task and the regression task, and such a scheme may help the network learn better by feature sharing, reducing over-fitting, and also reducing the computational budget for inference. The proposed MG-Unet considers both local features extracted by CNN and long distance connections encoded by GCN. Additionally, for transforming image data into graph data, instead of using 𝑘-nearest neighbors algorithm, we used inner product to estimate the similarity between two nodes and construct the edges of the graph. The experimental results demonstrate that our proposed architecture outperforms related existing works like [7] and [8].

I hope you enjoye.
## MATERIALS AND METHODS
In my project, I used 500 image frames from the fetoscopic camera as input to the network, and as the output of regression task, an estimation of the placenta orientation with respect to the camera is inferred. Details regarding how to formulate relative orientation of the placental surface is given on [7].

MG-Unet is a U-shape model with symmetrical architecture, whose architecture can be seen in Figure 1. It consists of three paths of graph encoders and three paths of graph decoders (performing down-sampling from image domain to the graph domain and up-sampling from the graph back to the image domain, respectively) and three fully connected (FC) layers, with the softmax operation of the final layer removed for the regression task. So, in this way, the MGUnet has two output branches; one with the up-sampling branch of the MG-Unet, and one with three FC layers. Mostly similar to the structure of U-Net, the feature extractor employs CNNs of two 3 × 3 convolutions, each followed by a batch normalization layer and parametric rectified linear unit (PReLU) layer.We used a convolutional layer with stride 2 for downsampling operation and a bilinear for upsampling operation with the scale factor 2. In order to transform and exchange information among all the nodes, two layers of graph convolution are used.

## Graph Convolution Operation
The purpose of GCN is to update the node represention through multiple layers by aggregating the information of the neighbors of every node through message passing to have final hidden features for every node. MG-Unet first builds an image’s graph structure by dividing it into 𝑁 patches, converting them into feature vectors as the same size of the number of channels, and then recognizing them as a set of nodes
$𝑉 = \{ 𝑣_1, 𝑣_2,... , 𝑣_𝑁 \}$.
![1](https://github.com/user-attachments/assets/9c1e8e17-bf41-4d4a-9a1e-70301d350eb6)

The intuition for construction the edges of the graph, is to learn a graph structure that reflects the similarities within the images, this similarity is mathematically defined as the inner-product
between node vector features.
![2](https://github.com/user-attachments/assets/407b46d3-e526-4f7c-81e2-dc4e59352eae)

In this way, a graph representation 𝐺(𝑉, 𝜀) is constructed where 𝜀 is set of all edges. Then for input feature 𝑋, the aggregation operation calculates the representation of a node by aggregating features of neighboring nodes. If 𝐴= 𝐴 + 𝐼 is the matrix of all nodes’connections which normally called an adjacency matrix, and 𝐼 is identity matrix to add self loops, ˆ
𝐷
is node degree matrix where
ˆ𝐷
𝑖 =
Í
𝑗 ˆ
𝐴
𝑖 𝑗 , we can use the normalized graph Laplacian
matrix ˆ
𝐷
−0.5 ˆ
𝐴
ˆ𝐷
0.5 to approximate the graph convolution
using Fourier transform properties. GCN follows the node
representation update as:
𝑋′ = 𝜎( ˆ
𝐷
−0.5 ˆ
𝐴
ˆ𝐷
0.5 𝑋 𝜃)
where 𝜎 is the non-linear activation function and 𝜃 is the
learning parameters or the weights.

