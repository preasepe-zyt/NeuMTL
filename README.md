# NeuMTL
<h1>Environment Setup</h1>
<h4>
conda env create -f NeuMTL.yml <br>
conda activate NeuMTL
</h4>

<h1>Dataset Preparation</h1>
<p>
Dataset 1 (<i>data1</i>) consists of three widely used drug–target affinity benchmarks: Davis, KIBA, and BindingDB.  
Dataset 2 (<i>data2</i>) contains blood–brain barrier (BBB) permeability data and neurotoxicity information.
</p>

<h3>Generate Dataset 1</h3>
<h4>
python create_data.py
</h4>

<h3>Generate Dataset 2</h3>
<h4>
python create_data2.py
</h4>

<h4>conda env create -f NeuMTL.yml<br>
<img src="framework.jpg" alt=""/>
