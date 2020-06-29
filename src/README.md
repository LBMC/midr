# IDR

This tool is designed to compute the Irreproducible Discovery Rate (IDR)
from NarrowPeaks files for two or more replicates.
It’s an implementation of the method described in the following paper using
Gaussian copula.

> LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility
> of high-throughput experiments. The annals of applied statistics, 2011,
> vol. 5, no 3, p. 1752-1779. doi:10.1214/11-AOAS466

The default method for the IDR computation is the one developped in the 
following paper using Archimedean copula.

> Measuring Reproducibility of High-Throughput Deep-Sequencing Experiments 
> Based on Self-adaptive Mixture Copula. Part of the Lecture Notes in Computer
> Science book series (LNCS, volume 7818) PAKDD 2013: Advances in Knowledge 
> Discovery and Data Mining p 301-313, isbn: 978-3-642-37453-1

All the estimators for multivariate Archimedean copula are implemented from the 
two following papers:

> Likelihood inference for Archimedean copulas in high dimensions under known
> margins, Journal of Multivariate Analysis, 2012, p133-150, 
> doi:10.1016/j.jmva.2012.02.019

> Archimedean Copulas in High Dimensions: Estimators and Numerical Challenges
> Motivated by Financial Applications | Journal de la Société Française de
> Statistique, Vol. 154 No. 1 (2013), ISSN: 2102-6238

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Prerequisites

To run **midr** on your computer you need to have python (>= 3) installed.

```sh
python3 --version
```

### Installing

To easily install **midr** on your computer using `pip` run the following command:

```
pip3 install midr
```

Otherwise you can clone this repository:

```
git clone git@gitbio.ens-lyon.fr:/LBMC/sbdm/midr.git
cd midr/src/
python3 setup.py install
```

Given a list of peak calls in NarrowPeaks format and the corresponding peak
call for the merged replicate. This tool computes and appends a IDR column to
NarrowPeaks files.

### Dependencies

The **idr** package depends on the following python3 library:

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

- [pandas>=0.25.0](https://pandas.pydata.org)
> McKinney. Data Structures for Statistical Computing in Python, Proceedings
> of the 9th Python in Science Conference, 51-56 (2010) [(publisher link)](http://conference.scipy.org/proceedings/scipy2010/mckinney.html)

- [pynverse>=0.1](https://pypi.org/project/pynverse/)

- [mpmath>=1.1.0](http://mpmath.org/)
- [cython>=0.28.0](https://cython.org/)

## Usage

**idr** Takes as input file in the [NarrowPeaks format](https://genome.ucsc.edu/FAQ/FAQformat.html#format12),
and output NarrowPeaks files with an additional *idr* column.

Computing *IDR* between three replicates

```
$ midr -m merged_peak_calling.NarrowPeaks \
     -f replicate1_.NarrowPeaks replicate2.NarrowPeaks replicate3.NarrowPeaks \
     -o results
```

Where `replicate1_.NarrowPeaks` is the output of the peak caller on the 
alignment file corresponding to the first replicate and 
`merged_peak_calling.NarrowPeaks` is the output of the peak caller on the merge
of the replicates alignment files.
`Results` are the directory where we want to output our results.

Displaying help:

```
$ midr -h
usage: midr [-h] [--merged FILE] [--files FILES [FILES ...]] [--output DIR] [--score SCORE_COLUMN] [--threshold THRESHOLD] [--merge_function MERGE_FUNCTION] [--size SIZE_MERGE] [--nodrop] [--method METHOD]
            [--cpu CPU] [--debug] [--verbose] [--matrix FILE]

Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

Implementation of the IDR methods for two or more replicates.

LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility
of high-throughput experiments. The annals of applied statistics, 2011,
vol. 5, no 3, p. 1752-1779.

Given a list of peak calls in NarrowPeaks format and the corresponding peak
call for the merged replicate. This tool computes and appends a IDR column to
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
                        NarrowPeaks score column to compute the IDR on, one of 'score', 'signalValue', 'pValue' or 'qValue' (default: signalValue)
  --threshold THRESHOLD, -t THRESHOLD
                        Threshold value for the precision of the estimators (default: 0.0001)
  --merge_function MERGE_FUNCTION, -mf MERGE_FUNCTION
                        function to determine the score to keep for overlapping peak within a replica ('sum', 'max', 'mean', 'median', 'min') (default: max)
  --size SIZE_MERGE, -ws SIZE_MERGE
                        distance (bp) to add before and after each peak before merging finding match between --merged file and --files files (default: 100)
  --nodrop, -nd         don't drop peak unmatched in any bed. The score of the absent peak is set to 0.0 (default: True)
  --method METHOD, -mt METHOD
                        copula model to use('archimedean' or 'gaussian' (default: archimedean)
  --cpu CPU, -cpu CPU   number of thread to use for merging the beds files (default: 1)
  --debug, -d           enable debugging (default: False)
  --verbose, -v         log to console (default: False)
  --matrix FILE         matrix file of the peaks score in raw (tsv format), replace the --merge and --files options if used
```


## Authors

* **Laurent Modolo** - *Initial work*

## License

This project is licensed under the CeCiLL License- see the [LICENSE](LICENSE) file for details.
