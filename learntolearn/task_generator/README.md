<div align="center">
    <h1>🌀 SAP-NERF</h1>
    <p><i>Data generation for meta-learning of object-NeRFs using the Sapien simulator</i></p>
</div>

## Introduction  
This repository uses Sapien models (available [here](https://sapien.ucsd.edu/downloads)) and ShapeNet models (available [here](https://shapenet.org/)) to generate data that can be used to meta-learn a category-specific object NeRF. 
The output result is in the form that can be consumed by [PRENOM](https://github.com/snt-arg/RO-MAP-NG). 

**The version of Sapien used is 2.2**. Other requirements include numpy, pillow, and openCV.

## Usage  
The models need to be downloaded to the `sapiens_data` directory, with folders named by the `<category_name>`, for e.g., `sapiens_data/laptop`.  
To generate data for a category, modify `constants.py` and use the script as follows:  
```bash
python generate_meta_dataset.py <category_name>
```
The results will be in the `output` folder, in the respective category.  
**Note:** For using ShapeNet data, you can use the helper script `process_shapenet_category.py ` to convert the models to a format usable by the Sapien simulator. Ofcourse, this requires you registering for the ShapeNet dataset in the first place. The process to generate reconstruction data would be the same as before. 


## Acknowledgements
Code was inspired and/or adapted from the following repositories: 
* [articulated-object-nerf](https://github.com/zubair-irshad/articulated-object-nerf)
