# Explainability of deep neural networks for MRI analysis of brain tumors

Artificial intelligence (AI), in particular deep neural networks, has achieved remarkable results for medical image analysis in several applications. Yet the lack of explainability of deep neural models is considered the principal restriction before applying these methods in clinical practice. In this study, we propose a NeuroXAI framework for explainable AI of deep learning networks to increase the trust of medical experts. NeuroXAI implements seven state-of-the-art explanation methods providing visualization maps to help make deep learning models transparent. NeuroXAI has been applied to two applications of the most widely investigated problems in brain imaging analysis, i.e., image classification and segmentation using magnetic resonance (MR) modality. Visual attention maps of multiple XAI methods have been generated and compared for both applications. Another experiment demonstrated that NeuroXAI can provide information flow visualization on internal layers of a segmentation CNN. Due to its open architecture, ease of implementation, and scalability to new XAI methods, NeuroXAI could be utilized to assist radiologists and medical professionals in the detection and diagnosis of brain tumors in the clinical routine of cancer patients.


# Installation
NeuroXAI has been only tested on Linux (Ubuntu 18.04).
To use NeuroXAI, it is strongly recommended that you install it in a virtual environment [How to do on Ubuntu 18.04](https://www.linode.com/docs/guides/create-a-python-virtualenv-on-ubuntu-18-04/)

Then, Run 
```
$ pip install -r requirements.txt
```

# Use Case Example
In order to demonstrate the capabilities of using NeuroXAI for addressing brain imaging tasks, it has been applied two applications of the most widely investigated problems in brain imaging analysis, i.e., image classification and segmentation using magnetic resonance (MR) modality. 

For the classification task, samples from the [BraTS 2019 dataset](https://www.med.upenn.edu/cbica/brats2019/data.html) was employed. Visual attention maps of the classification are presentated in the following figure:
![GUI](https://github.com/razeineldin/NeuroXAI/blob/main/results/sample_classification_results.png)
> Fig 1. NeuroXAI sample classification explanation maps.

For the segmentation task, samples from the [BraTS 2021 dataset](http://braintumorsegmentation.org/) was employed.  Visual attention maps of the segmentation are presentated in the following figure:
![GUI](https://github.com/razeineldin/NeuroXAI/blob/main/results/sample_segmentation_results.png)
> Fig 2. NeuroXAI sample segmentation explanation maps.

Another experiment demonstrated that NeuroXAI can provide information flow visualization on internal layers of a segmentation CNN as in the following figure:
![GUI](https://github.com/razeineldin/NeuroXAI/blob/main/results/sample_layer_flow.png)
> Fig 3. NeuroXAI network dissection explanation maps.


# License
This project is licensed under the Apache-2.0 License - see the [LICENSE.txt](LICENSE.txt) file for details

# Citation
The work has been published in the International Journal of Computer Assisted Radiology and Surgery, after the presentation in the Computer Assisted Radiology and Surgery Conference (CARS 2022). If you find this extension usefull, feel free to use it (or part of it) in your project and please cite the following paper:
    
    @article{zeineldin2022explainability,
    title={Explainability of deep neural networks for MRI analysis of brain tumors},
    author={Zeineldin, Ramy A and Karar, Mohamed E and Elshaer, Ziad and Wirtz, Christian R and Burgert, Oliver and Mathis-Ullrich, Franziska and others},
    journal={International Journal of Computer Assisted Radiology and Surgery},
    pages={1--11},
    year={2022},
    publisher={Springer}
  }
    
# Disclaimer
*NeuroXAI* is for research purposes and not intended for clinical use. Therefore, The user assumes full responsibility to comply with the appropriate regulations.


