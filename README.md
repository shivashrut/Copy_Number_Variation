# CNVEM: Copy Number Variation detection using Modified EM

## Installation
The following software must be installed on your machine:

Linux LST 14 or 16 <br/>
Python : tested with version 3.7<br/>
Python >= 3.7<br/>

### Python dependencies
* numpy == 1.21.2
* seaborn == 0.11.2
* pysam == 0.18.0
* matplotlib == 3.5.1

You can install the above package using the following command：

pip install numpy sklearn pysam matplotlib


## Running
CNVEM requires two input files, a bam file after sort and a reference folder, the folder contains the reference sequence for each chromosome of the bam file.

Note: At present, the method only supports single-chromosome samples. The whole genomes can be detected by batch file.

### Runnig command
python3 CNVEM.py [reference] [bamfile] [binSize] [output] [chrom]

[reference]: the reference folder path

[bamfile]: a bam file after sort

[binSize]: the window size ('1000'by default)

[output]: output file with CN

[chrom]: Chromosome number. (Eg. '17')

### Run the default example
python CNVEM.py .fastq .bam 1000 .output 21
