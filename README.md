# Welcome to DeepKlapred: A deep learning framework for identifying lysine lactylation sites via multi-view feature fusion
Lysine lactylation (Kla) is a post-translational modification (PTM) that holds significant importance in the regulation of various biological processes. In this study, we propose a novel framework that integrates sequence embedding with sequence descriptors to enhance the representation of protein sequence features. Our framework employs a BiGRU-Transformer architecture to capture both local and global dependencies within the sequence, while incorporating six sequence descriptors to extract biochemical properties and evolutionary patterns. Additionally, we apply a cross-attention fusion mechanism to combine sequence embeddings with descriptor-based features, enabling the model to capture complex interactions between different feature representations. Our model demonstrated excellent performance in predicting Kla sites, achieving an accuracy of 0.998 on the training set and 0.969 on the independent set. Additionally, through attention analysis and motif discovery, our model provided valuable insights into key sequence patterns and regions that are crucial for Kla modification. This work not only deepens the understanding of Kla’s functional roles but also holds the potential to positively impact future research in protein modification prediction and functional annotation.

This lysine lactylation sites prediction tool developed by a team from the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study](https://github.com/GGCL7/DeepKlapred/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/GGCL7/DeepKlapred/tree/main/Data)
# Source code
We provide the source code and you can find them [code](https://github.com/GGCL7/DeepKlapred/tree/main/code)
