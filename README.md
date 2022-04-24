# The Girl with a Voxel Earring

[@maajor](https://github.com/maajor)'s submission to [Taichi Voxel Challenge](https://github.com/taichi-dev/voxel-challenge/issues/1)

![](./girl_with_pearl_ring.jpg)

To run:  
```sh
pip3 install -r requirements.txt
python3 main.py
```

A lot of inspiration from [IQ's sdf article](https://iquilezles.org/articles/distfunctions/)  



# How this work  

1. Make your work in [MagicaCSG](https://ephtracy.github.io/index.html?page=magicacsg])  
Please put all your primitives in [-64,0,-64]~[64,128,64]  
Then save to a mcsg file.

2. Generate code with  
`python3 mcsg_to_py.py -i your-file.mcsg -o main.py`

3. Run your code with  
`python3 main.py`