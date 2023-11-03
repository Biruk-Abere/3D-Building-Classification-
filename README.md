# Multi Modal Diffusion Models For 3D Building Classification

# Introduction 

3D building classification is the computational process of categorizing three-dimensional representations of architectural structures. This involves analyzing volumetric data to percieve distinct architectural features,spatial configurations,and other instrinisc characterstics of buildings.

We've developed a methodology that intergrates a generative capabilities with classification strengths. At the core of our approach is a hierarchical latent space that captures features of 3D building structures. This representation is then subjected to a diffusion process ,which perturbs the data iteratively. Post-diffusion, a deep neural network is employed to denoise the data, extracting essential features that are crucial for classification.

Following the denoising, a classification model is introduced which predicts the building category based on the denoised latent representation. Our system is designed to handle multi-modal data, allowing it to process and integrate information from various data types, enhancing its classification accuracy.

# Problem Definition 

**Deep Generative Models (Diffusion Models)**:

These models are at the forefront of machine learning research, particularly for their ability to generate new, previously unseen data samples that closely resemble the distribution of the training data. One of the prominent types of generative models is the "Diffusion Model." As the name suggests, these models employ a diffusion process, which can be visualized as a gradual spreading or dispersing mechanism. The core idea is to start with random noise, which is essentially unstructured data, and then iteratively refine this noise through the diffusion process until it morphs into a structured data sample that aligns with the training data's characteristics.

**Recognizing 3D Buildings:**

While the generation of new data samples is fascinating, the primary task at hand is the recognition and classification of three-dimensional representations of buildings. This involves analyzing the intricate details and features of 3D models to categorize them accurately. The challenge is not just about creating new 3D samples but more about understanding and classifying the existing ones based on their inherent attributes.

**3D Representation Types:**

To achieve accurate classification, it's essential to understand the various ways buildings can be represented in 3D:

- **Multiple Images:** This involves capturing different 2D views or angles of the building. By analyzing multiple perspectives, one can get a more comprehensive understanding of the building's 3D structure.

- **Point Clouds:** These are collections of data points in a 3D coordinate system. Each point represents a tiny portion of the external surfaces of the building, and together, they form a 3D representation of the structure. Point clouds are often derived from technologies like LIDAR.

- **Voxels:** Think of voxels as the 3D counterpart of 2D pixels. In a 3D space, voxels represent values on a regular grid. They can be used to create volumetric representations, where each voxel might denote the presence (or absence) of a part of the building.

- **SDF (Signed Distance Function):** This is a more mathematical representation. An SDF represents a shape in 3D space using a continuous function. For any given point in space, the function's value indicates the shortest distance between that point and the shape's surface. The "signed" aspect means that the distance is positive outside the shape and negative inside, providing a clear distinction between interior and exterior regions. This representation is particularly useful for complex geometries and can be employed in various computer graphics and computational geometry tasks.

# Approach Using Diffussion Model 

**Methodology:**

The core of this approach is inspired by the groundbreaking paper titled “Your Diffusion Model is Secretly a Zero-Shot Classifier”. This paper presents a novel perspective on diffusion models, suggesting that while they are primarily designed for generative tasks, they possess an inherent capability for classification, especially in scenarios where labeled data is scarce, termed as "Zero-Shot" classification. In essence, the diffusion process, which typically transforms random noise into structured data samples, can also be harnessed to determine the likelihood of a data sample belonging to a particular class.

**Building Net Dataset:**

This dataset serves as the foundation for the entire project. It comprises various 3D representations of buildings, and the models will be trained and tested on this data. The overarching objective is not just to fit the model to the data but to enhance its classification performance, ensuring it can accurately categorize different building types based on their 3D point-cloud representations.    

**Goal:**

The primary aim is to push the boundaries of what's achievable in terms of classification scores on the BuildingNet dataset. Specifically, the focus is on point-cloud representations of buildings. Point clouds, with their intricate details and depth, present unique challenges, and achieving high classification accuracy on them would signify a significant advancement in the field.

       

# Diffusion Model For Classification 

**Understanding Diffusion Models:**

Diffusion models are a subset of generative models that operate based on the principle of simulating a diffusion process. This process begins with a known data point and introduces random noise to it iteratively, diffusing it until it converges to a simple Gaussian noise. The reverse of this process, where the Gaussian noise is transformed back into a structured data sample, embodies the generative capability of the model. In essence, diffusion models mimic the natural process of diffusion to generate new data samples from random noise.

**Classification Via Denoising:**

A pivotal insight in the realm of diffusion models is their potential for classification through the denoising process. Traditionally, denoising aims to recover the original structure of data by reversing the noise addition. However, for classification, this denoising process can be conditioned on specific class labels. This means that during the denoising phase, the model can be instructed to generate a sample that aligns with a particular class. By observing how well the model denoises the data into a specific class, one can infer the likelihood of the data belonging to that class.


**Zero-Shot Classification:**

One of the standout capabilities of diffusion models is their aptitude for zero-shot classification. This involves classifying data into categories that the model hasn't encountered during its training phase. The methodology is as follows:

* Train the diffusion model as a generative model on a dataset without explicitly using class labels.
* During the classification phase, utilize the denoising process conditioned on class labels (including those not seen during training) to determine which class the model is most likely to generate.

**Scoring Mechanism:**

The scoring mechanism is integral to the classification process in diffusion models. Here's how it works:
* For each potential class, the reverse diffusion process (denoising) is executed, conditioned on that specific class.
* The model then measures the likelihood or score of the denoised sample aligning with each class.
* The data point is subsequently assigned to the class that yields the highest likelihood or score, effectively classifying it.

        

# Combining LION with Diffusion Classifier

**Integration:**

The fusion of LION, a model adept at generating high-quality 3D shapes, with the diffusion classifier, a model designed for robust classification using noise-added data, presents a novel architecture that capitalizes on the strengths of both methodologies.

**LION's Role:** LION, with its hierarchical latent space, captures both global and local features of 3D building structures. Its generative capabilities allow for the creation of detailed 3D shapes, which can be used as input for the diffusion classifier.

**Diffusion Classifier's Role:** Upon receiving the 3D structures generated by LION, the diffusion classifier applies its noise-adding process, followed by the denoising mechanism conditioned on class labels. This process determines the likelihood of the data belonging to a specific class, effectively classifying the 3D structure.

**Multi-modal Learning:**

LION's inherent adaptability makes it an ideal candidate for multi-modal learning, where the model is trained to process and integrate information from multiple types of data sources or formats.

**3D Data + Textual Descriptions:** LION can be conditioned on textual prompts, allowing it to generate 3D structures based on specific textual descriptions. For instance, given a textual description like "Victorian-style house with a large porch," LION could generate a corresponding 3D model. The diffusion classifier can then classify this model based on its features and the associated textual description.

**3D Data + 2D Images:** LION can also be integrated with 2D images or blueprints of buildings. These 2D images can provide supplementary context and details that might not be immediately evident in the 3D model alone. By processing both the 3D and 2D data, the combined model can achieve a more comprehensive understanding and classification of the building.

 **Latent Space Fusion:** After processing different modalities, their respective latent representations can be fused. Techniques like canonical correlation analysis (CCA) or simple concatenation, followed by a neural network, can be employed to merge information from these diverse sources.

**End-to-End Training:** To maximize the benefits of multi-modal data, an end-to-end training approach can be adopted. This ensures that the model learns to map from raw inputs (from all modalities) directly to classifications, allowing it to discern the best way to amalgamate information from the various sources.
