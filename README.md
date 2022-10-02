# SpacePuma

```SpacePuma``` is an interactive extension to matplotlib designed for use within a jupyter environment. It was originally developed for the processing of astrochemical data but has demonstrated uses across a range of other data-handling fields. The extension combines a series of tools that allow for user input directly through a ```matplotlib```-powered GUI and indirectly through python. Data products, including figures, calculations, and data manipulations, can be exported to a python dictionary or saved through integration with ```pickle```. 

Customization of ```SpacePuma``` is possible and highly encouraged, with explanations of the necessary modifications covered in the tutorials. All the code is available publicly on GitHub. Tutorial notebooks are included within the Tutorials directory. For detailed instructions on how to modify or add to the tools, check our documentation page [here]().

## Installation

To install ```SpacePuma```, grab the latest version from PyPi using pip:

```
$ pip install spacepuma
```

This will install a couple of dependencies, including matplotlib, scipy, ipympl, and seaborn, each of which should be installed automatically if you don't already have them. To verify that everything installed properly, the tutorials should work without issue.
Now, all the necessary packages should be installed!

## Tutorials

Once you have installed the package, you can make use of some of the tutorials we have designed. To access the tutorials, clone the repository here and enter the tutorials folder and launch the tutorial notebook:

```bash
cd Tutorials
jupyter lab 
```

These tutorials should orient new users to the tools and provide a significant explanation of the powers of the package as a whole. Please report any issues through GitHub where they will be addressed in a timely manner.

Happy coding!

*N.B. These tools were originally developed for the Öberg Astrochemistry Group. The name```SpacePuma``` was chosen following the space feline naming convention of the lab. It stands for: **Space Puma: Astrochemistry Code for Experiments Pertaining to the Understanding of More Astrochemisty***



