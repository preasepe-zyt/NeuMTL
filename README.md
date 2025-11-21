# NeuMTL
<h1> Environment Setup<br> <h4>conda env create -f NeuMTL.yml<br> 
<h1> Create Dataset<br>
<h4> Dataset 1 (*data1*) comprises three commonly used drug–target interaction benchmarks: Davis, KIBA, and BindingDB.<br>Dataset 2 (*data2*) includes blood–brain barrier (BBB) permeability data and neurotoxicity data.<br> 
<h3> Create Dataset 1<br> 
<h4>conda activate NeuMTL<br> python create create_data.py<br> 
<h3> Create Dataset 2<br> 
<h4>conda activate NeuMTL<br> python create create_data2.py<br>
<h1> Model Training<br>
<h4>python training.py 0/1/2<br>

<h1> File introduction<br>
<h4>model.py</h4> Defines the overall NeuMTL model architecture. <h4>gated_attention.py</h4> Implements the gated attention mechanism for modality-aware feature weighting. <h4>shared_attention.py</h4> Implements the shared attention layer for representation learning across tasks. <h4>Task_Specific_Attention.py</h4> Implements task-specific attention modules to capture unique task-level feature contributions. <h4>gra.py</h4> Multi-task gradient regulation strategies. <h4>training.py</h4> Controls the full training pipeline. <h4>utils.py</h4> Contains utility functions including loss computation, evaluation metrics, logging, and checkpoint saving. <h4>create_data.py</h4> Data preprocessing for drug–target datastes. <h4>create_data2.py</h4> Data preprocessing for BBB permeability and neurotoxicity datastes. <h4>framework.jpg</h4> Illustration of the NeuMTL multimodal multi-task framework. 
<img src="framework.jpg" alt=""/>
