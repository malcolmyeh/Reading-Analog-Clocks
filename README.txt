1. Install requirements
    - Jupyter Notebook
    - tensorflow (tensorflow-gpu if required gpu drivers are installed)
    - OpenCV
    - matplotlib
    - pandas
    - numpy
    - IPython
    - argparse

2. Generate dataset

    usage: generate_clocks.py [-h] [--skew | --no-skew] N
    generate_clocks.py: error: the following arguments are required: N

    example: generate_clocks.py 100000 --no-skew

    This will generate clock images under the folder /images as well as a csv file label.csv 
    containing the corresponding hour and minute labels. 

    Note: skew applies randomized perspective transform. cnn.ipynb expects at least 95010 images 
    (80000 train, 15000 validation, 10 display), but can be changed on inside cell Import Dataset

4. Detect time using OpenCV

    usage: usage: parse_clock.py [-h] [--demo | --no-demo] path_to_image
    parse_clock.py: error: the following arguments are required: path_to_image

    example: parse_clock.py clock.jpg --demo

    The demo option will show intermediary steps (skew correction using bounding rectangle and circle,
    Hough line detection).

    Note: This currently does not provide consistent results and there is an issue where if an invalid
    bounding circle/rectangle is wrongly detected, the resulting warped image is not usable. Limitations
    of this program are discussed in the final report. 

5. Train and predict time using CNN
    
    jupyter notebook

    Run cells sequentially. Import dataset with current parameters takes ~5 minutes. Training model
    (on RTX 3060) takes ~ 8minutes. Alternatively, load a pretrained model in the Load Model cell. 

