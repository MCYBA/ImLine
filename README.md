# ImLine

the image marking and angular face map creating software.

#### Supported Environments

|                         |                                         |
|-------------------------|-----------------------------------------|
| **Operating systems**   | Linux                                   |
| **Python versions**     | Python 3.x (64-bit)                     |
| **Distros**             | Ubuntu                                |
| **Package managers**    | APT, pip                                |
| **Languages**           | English                                 |
|                         |                                         |

### Installation

Clone the GitHub repository and run

```Shell
sudo ./install.sh
```

in the repository directory.

for development mode: `sudo ./install-dev.sh`


<sup><i>If there is a failure try `sudo -H ./install-dev.sh`</i></sup>

### Usage

```
usage: imline [-h] [--ripe-dataset RIPE_DATASET]
              [--environment {production,development,testing}] [-v]
              [--version]
              {multiply-dataset,mark-object,mark-key-points,create-dataset,create-maps}
              ...

positional arguments:
  {multiply-dataset,mark-object,mark-key-points,create-dataset,create-maps}
                        officiate the sub-jobs
    multiply-dataset    Mark found objects of the given images to the ImLine.
    mark-object         Mark found objects of the given images to the ImLine.
    mark-key-points     Mark key-points of the given images to the ImLine.
    create-dataset      Create a dataset that become images that fetched from
                        Google with given `query` parameter
    create-maps         Create maps via key-points of the given images to the
                        ImLine.

optional arguments:
  -h, --help            show this help message and exit

Others:
  --ripe-dataset RIPE_DATASET
                        Images folder that keeps already marked images and
                        their key-points"s JSON data.
  --environment {production,development,testing}
                        The running environment. It specify the configuration
                        files and logs. To use: either `production`,
                        `development` or `testing`. Default is production
  -v, --verbose         Print various debugging logs to console for debug
                        problems
  --version             Display the version number of ImLine.
```

<br>


**Supported Distributions:** Linux Mint. This release is fully supported. Any other Debian based distributions are partially supported.

### Contribute

If you want to contribute to T_System then please read [this guide]().
