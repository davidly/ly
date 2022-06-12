from contextlib import nullcontext
from PIL import Image
import sys
import os
from os.path import exists
import re
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from consts import ModelSize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

print("--> Starting DALL-E app. This might take up to two minutes.")

from dalle_model import DalleModel
dalleModel = None

# convert all non-alpha-numeric characters to a hyphen so filenames are easy to work with

def makeFilename(filename):
    return re.sub(r'\W+', '-', filename)

def generate_images_api( folder_name, text_prompt, num_images ):
    print( f"generating images for input: " + text_prompt)

    # for debugging, create a tiny blank image
    #img = np.zeros([10,10,3],dtype=np.uint8)
    #img[:] = 255
    #generated_imgs = []
    #generated_imgs.append( Image.fromarray(img))

    # call the model to generate the image(s)

    generated_imgs = dalleModel.generate_images(text_prompt, num_images)

    # save the images to files under the folder

    filename = makeFilename( text_prompt )
    i = 0
    generated_images = []

    if generated_imgs != None:
        for img in generated_imgs:
            output_path = folder_name + "/" + filename

            # if > 1 image is being generated, append a number to the filename
            if 1 != num_images:
                output_path += str(i)

            img.save(output_path + ".png", format="PNG")
            i += 1

    print(f"Created {i} images from text prompt [{text_prompt}]")
    return 

def show_usage():
    print( f"usage: ly argument")
    print( f"    argument can be either a string for a single image or a filename where each unique line generates an image")
    quit()

def rm_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def generate_images_for_lyrics( input_file ):

    # read the input file
    input_lines = None
    with open( input_file ) as file:
        input_lines = file.readlines()

    # strip trailing white space
    input_lines = [line.rstrip() for line in input_lines]

    # remove empty lines
    input_lines = list(filter(None, input_lines))

    # remove duplicate lines
    input_lines = rm_duplicate_preserve_order( input_lines)

    # make an output folder for the images with the same name as the input file minus the extension + "_images"

    folder = os.path.splitext(input_file)[0]
    folder += "_images"
    if not os.path.exists( folder ):
         os.mkdir( folder )

    # Invoke everything in parallel, though the ML code blocks on the first caller into the model
    # until nearly all processing is complete -- so the first image generated is mostly in serial
    # then everything else is parallelized. 
    # I tried for 2 hours to get the GPU to be used, but XLA on WSL is apparently problematic.
    # So as a fallback this uses all cores in parallel.

    pool = Pool()
    for line in input_lines:
        pool.apply_async( generate_images_api, (folder,line,1,))

    pool.close()
    pool.join()
    return

# Here's the actual app -- generate images for lines in a file or for the input argument
 
print( f"loading model...")
dalle_version = ModelSize.MINI
#dalle_version = ModelSize.MEGA
dalle_version = ModelSize.MEGA_FULL
dalleModel = DalleModel(dalle_version)
print( f"model loaded")

if len(sys.argv) != 2 :
    show_usage()

input = sys.argv[1]    

if exists( input ):
    generate_images_for_lyrics( input )
else:
    generate_images_api( "./", input, 1 )    

print(f"app is complete")

