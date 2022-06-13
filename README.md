# ly
Generates a lyric video using DALL-E

Steps to take:

    1) Clone https://github.com/saharmor/dalle-playground on a Linux machine. I used WSL on Windows.
    2) Follow the instructions for that repo to get your machine in a state where app.py works (though it's not used here)
    3) Go to the backend folder and copy ly.pt to that location.
    4) Get lyrics for the song you want. The repo has lucy.txt as an example. Put in the backend folder
      a) Use the ID app https://github.com/davidly/id to extract lyrics from flac and MP3 files
      b) Or just google it
    5) python ly.py lucy.txt
    6) This will take an hour or more on a CPU depending on the number of lyrics. I spent two hours attempting to get a GPU
       working with JAX on WSL and failed. Others have discovered this is a combination that doesn't seem to work.
    7) PNG files will appear in the input filename _images folder, in this case lucy_images.
    8) Use https://github.com/davidly/cv to create an MP4 slideshow: cv /i:lucy.txt /w:256 /h:256 /o:lucy.mp4 /d:2000 /c
    
Sample output file: https://twitter.com/davidly/status/1536133798806491136
     
