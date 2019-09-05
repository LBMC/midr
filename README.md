# IDR

This tools is designed to compute the Irreproducible Discovery Rate (IDR)
from NarrowPeaks files for two or more replicates.
It's an implementation of the method described in the following paper using
Gaussian copula.

> LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility
> of high-throughput experiments. The annals of applied statistics, 2011,
> vol. 5, no 3, p. 1752-1779.

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Prerequisites

To run **idr** on you computer you need to have python (>= 3) installed.

```sh
python3 --version
```

### Installing

To easily install **idr** on you computer using `pip` run the following command:

```
pip3 install idr
```

Otherwise you can clone this repository:

```
git clone url
cd idr/src/
python3 setup.py install
```

Given a list of peak calls in NarrowPeaks format and the corresponding peak
call for the merged replicate. This tools compute and append a IDR column to
NarrowPeaks files.

### Dependencies

The **idr** package depend on the following python3 library:

- [scipy>=1.3](https://scipy.org) [DOI:10.1109/MCSE.2007.58](https://doi.org/10.1109/MCSE.2007.58) [DOI:10.1109/MCSE.2011.36](https://doi.org/10.1109/MCSE.2011.36)

> Travis E. Oliphant. Python for Scientific Computing, Computing in Science &
> Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58

> K. Jarrod Millman and Michael Aivazis. Python for Scientists and Engineers,
> Computing in Science & Engineering, 13, 9-12 (2011),
> DOI:10.1109/MCSE.2011.36


- [numpy>=1.16](https://numpy.org/) [DOI:10.1109/MCSE.2011.37](https://doi.org/10.1109/MCSE.2010.118)

> Travis E, Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).

> Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array:
> A Structure for Efficient Numerical Computation, Computing in Science &
> Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

- [matplotlib>=3.1](https://github.com/matplotlib/matplotlib/tree/v3.1.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3264781.svg)](https://doi.org/10.5281/zenodo.3264781)

>  J. D. Hunter, "Matplotlib: A 2D Graphics Environment",
> Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.

- [pynverse>=0.1](https://pypi.org/project/pynverse/)

## Usage

**idr** takes as input file in the [NarrowPeaks format](https://genome.ucsc.edu/FAQ/FAQformat.html#format12),
and output NarrowPeaks files with an additional *idr* column.

Computing *IDR* between three replicates

```
$ idr -m merged_peak_calling.NarrowPeaks \
     -f replicate1_.NarrowPeaks replicate2.NarrowPeaks replicate3.NarrowPeaks \
     -o results
```

where `replicate1_.NarrowPeaks` is the output of the peak caller on the 
alignment file corresponding to the first replicate and 
`merged_peak_calling.NarrowPeaks` is the output of the peak caller on the merge
of the replicates alignment files.
`results` is the directory where we want to output our results.

Displaying help:

```
$ idr -h
usage: idr [-h] --merged FILE --files FILES [FILES ...] [--output DIR]        
           [--score SCORE_COLUMN] [--threshold THRESHOLD] [--debug]           
           [--verbose]                                                        
                                                                              
Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files        
                                                                              
Implementation of the IDR methods for two or more replicates.                 
                                                                              
LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility  
of high-throughput experiments. The annals of applied statistics, 2011,       
vol. 5, no 3, p. 1752-1779.                                                   
                                                                              
Given a list of peak calls in NarrowPeaks format and the corresponding peak   
call for the merged replicate. This tools compute and append a IDR column to  
NarrowPeaks files.                                                            
                                                                              
optional arguments:                                                           
  -h, --help            show this help message and exit                       
                                                                              
IDR settings:                                                                 
  --merged FILE, -m FILE                                                      
                        file of the merged NarrowPeaks                        
  --files FILES [FILES ...], -f FILES [FILES ...]                             
                        list of NarrowPeaks files                             
  --output DIR, -o DIR  output directory (default: results)                   
  --score SCORE_COLUMN, -s SCORE_COLUMN                                       
                        NarrowPeaks score column to compute the IDR on, one of
                        'score', 'signalValue', 'pValue' or 'qValue' (default:
                        signalValue)                                          
  --threshold THRESHOLD, -t THRESHOLD                                         
                        Threshold value for the precision of the estimator    
                        (default: 0.01)                                       
  --debug, -d           enable debugging (default: False)                     
  --verbose             log to console (default: False)                       
```


## Authors

* **Laurent Modolo** - *Initial work*

## License

This project is licensed under the CeCiLL License- see the [LICENSE](LICENSE) file for details